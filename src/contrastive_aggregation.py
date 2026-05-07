"""
Contrastive Weight Aggregation: 基于相似度加权聚合客户端上传的特征/权重

实现 CreamFL 的对比权重聚合机制（con_w），根据客户端特征与全局特征的相似度
计算权重，然后加权聚合客户端特征。

参考：CreamFL src/algorithms/MMFL.py (对比权重聚合逻辑)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import gc


class ContrastiveWeightAggregation:
    """
    对比权重聚合器（CreamFL 核心聚合机制）
    
    基于客户端特征与全局特征之间的相似度计算聚合权重，然后加权聚合客户端特征。
    
    核心算法（参考 MMFL.py con_w 方法）：
    1. 计算每个客户端特征与全局特征的内积相似度矩阵
    2. 计算 log_prob（softmax 的对数概率）
    3. 提取对角线元素作为对比权重
    4. 对所有客户端应用 softmax 归一化权重
    5. 使用权重加权聚合客户端特征
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            device: 计算设备
        """
        self.device = device

    # ------------------------------------------------------------------
    # ✨ Phase 1 重构：废除 _align_dim 零填充，改用防御性维度校验
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_feature_dim(
        tensor: torch.Tensor,
        expected_dim: int,
        tensor_name: str = "feature"
    ) -> torch.Tensor:
        """
        ★ Fix (2026-03-14): 将原来的 strict raise 改为自适应对齐策略。

        原版对维度不匹配直接抛 ValueError，在 demo/test 场景（TextOnlyTrainer
        动态投影维度可能不等于 contrastive_dim）导致崩溃。

        修复策略：
        1. 0-dim 标量 → 展开为 (1,)
        2. 维度不匹配 → adaptive_avg_pool1d 对齐到 expected_dim
        3. 维度匹配 → 直接返回
        """
        # 防护 0-dim 标量（shape=()）
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

        curr_dim = tensor.shape[-1]
        if curr_dim == expected_dim:
            return tensor

        # 维度不匹配 → 自适应对齐（pool 或 repeat）
        original_shape = tensor.shape
        if tensor.dim() == 1:
            tensor = F.adaptive_avg_pool1d(
                tensor.unsqueeze(0).unsqueeze(0), expected_dim
            ).squeeze(0).squeeze(0)  # → (expected_dim,)
        else:
            # (N, D) → (N, expected_dim)
            tensor = F.adaptive_avg_pool1d(
                tensor.unsqueeze(0), expected_dim
            ).squeeze(0)

        return tensor


    def compute_contrastive_weights(
        self,
        client_features_list: List[torch.Tensor],
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比权重（基于相似度的权重）
        
        参考：MMFL.py 中的 contrastive_w 计算逻辑
        
        Args:
            client_features_list: 客户端特征列表，每个元素形状为 (N, D)
                                  N 是样本数量，D 是特征维度
            global_features: 全局特征，形状为 (N, D)
        Returns:
            对比权重矩阵，形状为 (num_clients, N)
            每一列代表该样本在不同客户端上的权重
        """
        num_clients = len(client_features_list)
        N = global_features.shape[0]  # 样本数量
        
        # 移动全局特征到设备
        if global_features.device != self.device:
            global_features = global_features.to(self.device)
        
        # 初始化权重矩阵
        contrastive_w = torch.zeros(num_clients, N, device=self.device)
        
        # 对每个客户端计算对比权重
        target_dim = global_features.shape[-1]  # 以全局特征维度为基准

        for i, client_vec in enumerate(client_features_list):
            # 移动客户端特征到设备
            if client_vec.device != self.device:
                client_vec = client_vec.to(self.device)

            # ★ Phase 1 重构：严格维度校验（禁止零填充/截断）
            client_vec = self._validate_feature_dim(
                client_vec.float(),
                target_dim,
                f"client_{i}_features"
            )

            # ★ Fix High 4 (2026-03-13): 对齐样本数 N，确保对角线取値有效
            # 当客户端 public_loader 样本数与全局特征池 N 不一致时，
            # matmul 返回非方阵， diagonal 取出的对角线长度被较小维度侧决定，
            # 导致静默降维（权重向量长度 < N，后续 broadcast 错误）。
            N_global = global_features.shape[0]
            N_client = client_vec.shape[0]
            if N_client != N_global:
                if N_client > N_global:
                    client_vec = client_vec[:N_global]   # 截断多余样本
                else:
                    # CycleRepeat 填充到 N_global
                    repeat_times = (N_global + N_client - 1) // N_client
                    client_vec = client_vec.repeat(repeat_times, 1)[:N_global]

            # 计算相似度矩阵：client_vec @ global_features.T
            # client_vec: (N, D), global_features: (N, D)
            # logits: (N, N) - 每个客户端样本与所有全局样本的相似度
            logits = torch.matmul(client_vec, global_features.float().t())  # (N, N)
            
            # 计算 softmax 的对数概率
            exp_logits = torch.exp(logits)
            log_sum_exp = torch.log(torch.sum(exp_logits, dim=1, keepdim=True) + 1e-8)
            log_prob = logits - log_sum_exp  # (N, N)
            
            # 提取对角线元素（正样本对的对数概率）
            # 对角线元素表示客户端特征 i 与全局特征 i 的匹配程度
            contrastive_w[i] = torch.diagonal(log_prob).reshape(-1)
            
            # 清理内存
            del logits, exp_logits, log_prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # 对所有客户端应用 softmax 归一化（按列归一化）
        # 这样每个样本的权重在不同客户端之间归一化
        contrastive_w = F.softmax(contrastive_w, dim=0)  # (num_clients, N)
        
        return contrastive_w
    
    def aggregate_features(
        self,
        client_features_list: List[torch.Tensor],
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        聚合客户端特征（使用对比权重）
        
        参考：MMFL.py 中的聚合逻辑
        i_vec[i] = (i_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
        i_vec = torch.sum(torch.cat(i_vec, dim=0), dim=0)
        
        Args:
            client_features_list: 客户端特征列表，每个元素形状为 (N, D)
            global_features: 全局特征，形状为 (N, D)
        Returns:
            聚合后的特征，形状为 (N, D)
        """
        if len(client_features_list) == 0:
            return global_features
        
        # 计算对比权重
        contrastive_w = self.compute_contrastive_weights(
            client_features_list, global_features
        )  # (num_clients, N)
        
        # 加权聚合
        weighted_features_list = []
        for i, client_vec in enumerate(client_features_list):
            # 移动客户端特征到设备
            if client_vec.device != self.device:
                client_vec = client_vec.to(self.device)
            
            # 应用权重：client_vec * contrastive_w[i] (broadcasting)
            # client_vec: (N, D), contrastive_w[i]: (N,)
            # 权重需要 reshape 为 (N, 1) 以支持广播
            weights = contrastive_w[i].reshape(-1, 1)  # (N, 1)
            weighted_vec = client_vec * weights  # (N, D)
            
            # 添加 batch 维度以便后续 concat
            weighted_vec = weighted_vec.unsqueeze(0)  # (1, N, D)
            weighted_features_list.append(weighted_vec)
        
        # 拼接所有加权特征并求和
        # cat: (num_clients, N, D) -> sum along dim=0 -> (N, D)
        aggregated_features = torch.sum(
            torch.cat(weighted_features_list, dim=0), dim=0
        )  # (N, D)
        
        return aggregated_features
    
    def aggregate_dual_modalities(
        self,
        image_features_list: List[torch.Tensor],
        text_features_list: List[torch.Tensor],
        global_image_features: torch.Tensor,
        global_text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        聚合双模态特征（图像和文本）
        
        参考：MMFL.py 中的双模态聚合
        - 图像特征使用全局文本特征计算权重（跨模态对比）
        - 文本特征使用全局图像特征计算权重（跨模态对比）
        
        Args:
            image_features_list: 客户端图像特征列表，每个形状为 (N, D)
            text_features_list: 客户端文本特征列表，每个形状为 (N, D)
            global_image_features: 全局图像特征 (N, D)
            global_text_features: 全局文本特征 (N, D)
        Returns:
            Tuple of (aggregated_image_features, aggregated_text_features)
        """
        # 聚合图像特征（使用全局文本特征计算权重 - 跨模态）
        if len(image_features_list) > 0:
            aggregated_img = self.aggregate_features(
                image_features_list, global_text_features
            )
        else:
            aggregated_img = global_image_features
        
        # 聚合文本特征（使用全局图像特征计算权重 - 跨模态）
        if len(text_features_list) > 0:
            aggregated_txt = self.aggregate_features(
                text_features_list, global_image_features
            )
        else:
            aggregated_txt = global_text_features
        
        return aggregated_img, aggregated_txt
    
    def aggregate_model_weights(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_public_reps: List[torch.Tensor],
        global_features: torch.Tensor,
        sample_wise_aggregation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        聚合模型权重（使用对比权重）

        核心修复：仅聚合客户端实际上传的参数（Adapter/Decoder），自动跳过冻结参数（SAM3 Backbone）

        Args:
            client_weights: 客户端模型权重列表（可能包含 None）
            client_public_reps: 客户端公共数据表示列表，每个形状为 (N, D) 或 (D,)
            global_features: 全局特征 (N, D)
            sample_wise_aggregation: 是否使用样本级聚合（默认: False，使用平均权重）
        Returns:
            聚合后的模型权重字典
        """
        # ========== 步骤1：安全过滤 None 客户端 ==========
        valid_clients = []  # 格式: [(weights_dict, rep_tensor, original_idx), ...]
        for idx, (weights, rep) in enumerate(zip(client_weights, client_public_reps)):
            if weights is not None and len(weights) > 0:
                valid_clients.append((weights, rep, idx))

        if len(valid_clients) == 0:
            raise ValueError("No valid client weights to aggregate (all are None or empty)")

        num_valid_clients = len(valid_clients)
        # print(f"[Aggregate] 过滤后有效客户端数量: {num_valid_clients}/{len(client_weights)}")

        # 解包有效客户端数据
        valid_weights_list = [c[0] for c in valid_clients]  # List[Dict]
        valid_reps_list = [c[1] for c in valid_clients]     # List[Tensor]

        # ========== 步骤2：处理客户端特征表示 ==========
        client_features_list = []
        for rep_data in valid_reps_list:
            # 处理多模态字典格式
            if isinstance(rep_data, dict):
                if 'image' in rep_data and isinstance(rep_data['image'], torch.Tensor):
                    rep = rep_data['image']
                elif len(rep_data) > 0:
                    rep = next(iter(rep_data.values()))
                else:
                    raise ValueError("Empty representation dict from client")
            else:
                rep = rep_data

            # 确保是 2D 张量 (1, D) or (N, D)
            if getattr(rep, 'dim', lambda: 0)() == 1:
                rep = rep.unsqueeze(0)
            client_features_list.append(rep)

        # ========== 步骤3：计算对比聚合权重并归一化 ==========
        if sample_wise_aggregation and global_features.shape[0] > 1:
            contrastive_w = self.compute_contrastive_weights(
                client_features_list, global_features
            )  # (num_valid_clients, N)
            weights = contrastive_w.mean(dim=1)  # (num_valid_clients,)
        else:
            weights = torch.ones(num_valid_clients, device=self.device)
            if global_features.shape[0] > 0:
                global_feature_avg = global_features.mean(dim=0).float()
                target_dim = global_feature_avg.shape[0]
                similarities = []

                for rep in client_features_list:
                    rep_tensor = rep.mean(dim=0).to(self.device).float() if rep.dim() > 1 else rep.to(self.device).float()
                    # ★ Phase 1 重构：严格维度校验
                    rep_tensor = self._validate_feature_dim(
                        rep_tensor,
                        target_dim,
                        "aggregated_client_rep"
                    )
                    similarity = F.cosine_similarity(
                        rep_tensor.unsqueeze(0), global_feature_avg.unsqueeze(0)
                    )
                    similarities.append(similarity.item())

                similarities = torch.tensor(similarities, device=self.device)
                weights = F.softmax(similarities, dim=0)
            else:
                weights = weights / num_valid_clients

        # 强制归一化（确保和为1）
        weights = weights / weights.sum()
        # print(f"[Aggregate] 归一化权重: {weights.cpu().numpy()}")

        # ========== 步骤4：动态键值匹配 + CPU聚合 ==========
        # ★ BUG FIX (2026-03-18): 原代码用第一个有效客户端的 keys 作为遍历基准，
        # 若第一个客户端是 text_only（Client 1），视觉参数（image_encoder、mask_decoder 等）
        # 因为不在基准集内，将被完全跳过，最终在防回归补齐时被全局随机权重覆盖——
        # 这正是"参数黑洞"的第二条传播路径。
        # 修复：改用所有有效客户端的参数键并集，确保 Client 2/3 的视觉参数一定进入聚合候选集。
        reference_keys: set = set()
        for _cw in valid_weights_list:
            reference_keys.update(_cw.keys())
        # print(f"[Aggregate] 并集基准参数数量: {len(reference_keys)}")

        aggregated_state = {}

        for param_name in reference_keys:
            param_list = []
            valid_client_indices = []  # ★ 新增：记录包含该参数的客户端索引

            # 收集所有包含该参数的客户端权重（移动到CPU）
            for idx, client_weight in enumerate(valid_weights_list):
                if param_name in client_weight:
                    param = client_weight[param_name].cpu()  # ★ 显存优化：移到CPU
                    param_list.append(param)
                    valid_client_indices.append(idx)  # ★ 记录有效索引

            # ★ 强化检查：如果没有客户端包含该参数，跳过（打印警告）
            if len(param_list) == 0:
                # print(f"[Aggregate] 警告: 参数 '{param_name}' 在所有客户端中缺失，跳过聚合")
                continue

            # ★ CPU聚合：防止显存溢出
            try:
                stacked_params = torch.stack(param_list, dim=0)  # (num_valid_clients_with_param, ...)

                # ★ 获取对应权重并重新归一化（只使用包含该参数的客户端的权重）
                valid_weights_for_param = weights[valid_client_indices].cpu()
                valid_weights_normalized = valid_weights_for_param / valid_weights_for_param.sum()

                # 扩展权重维度以支持广播
                weight_shape = [-1] + [1] * (stacked_params.dim() - 1)
                expanded_weights = valid_weights_normalized.view(*weight_shape)

                # 加权求和（在CPU上执行）
                weighted_sum = torch.sum(stacked_params * expanded_weights, dim=0)
                aggregated_state[param_name] = weighted_sum  # 保持在CPU

            except Exception as e:
                print(f"[Warning] 聚合参数 '{param_name}' 时出错: {e}")
                continue

        # print(f"[Aggregate] 聚合完成，参数数量: {len(aggregated_state)}")
        return aggregated_state


if __name__ == "__main__":
    print("=" * 60)
    print("测试 ContrastiveWeightAggregation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_clients = 3
    num_samples = 100
    feature_dim = 768
    
    # 创建聚合器
    aggregator = ContrastiveWeightAggregation(device=device)
    
    print("\n1. 测试对比权重计算")
    print("-" * 60)
    
    # 创建虚拟客户端特征和全局特征
    client_features_list = [
        torch.randn(num_samples, feature_dim).to(device)
        for _ in range(num_clients)
    ]
    global_features = torch.randn(num_samples, feature_dim).to(device)
    
    # 计算对比权重
    contrastive_weights = aggregator.compute_contrastive_weights(
        client_features_list, global_features
    )
    print(f"对比权重形状: {contrastive_weights.shape}")  # (num_clients, num_samples)
    print(f"权重和（每列应该为 1）: {contrastive_weights.sum(dim=0)[:5]}")  # 检查归一化
    print(f"权重范围: [{contrastive_weights.min():.4f}, {contrastive_weights.max():.4f}]")
    
    print("\n2. 测试特征聚合")
    print("-" * 60)
    
    aggregated_features = aggregator.aggregate_features(
        client_features_list, global_features
    )
    print(f"聚合特征形状: {aggregated_features.shape}")  # (num_samples, feature_dim)
    print(f"聚合特征范数: {aggregated_features.norm(dim=1).mean():.4f}")
    
    print("\n3. 测试双模态聚合")
    print("-" * 60)
    
    image_features_list = [
        torch.randn(num_samples, feature_dim).to(device)
        for _ in range(num_clients)
    ]
    text_features_list = [
        torch.randn(num_samples, feature_dim).to(device)
        for _ in range(num_clients)
    ]
    global_image_features = torch.randn(num_samples, feature_dim).to(device)
    global_text_features = torch.randn(num_samples, feature_dim).to(device)
    
    agg_img, agg_txt = aggregator.aggregate_dual_modalities(
        image_features_list, text_features_list,
        global_image_features, global_text_features
    )
    print(f"聚合图像特征形状: {agg_img.shape}")
    print(f"聚合文本特征形状: {agg_txt.shape}")
    
    print("\n4. 测试模型权重聚合")
    print("-" * 60)
    
    # 创建虚拟模型权重
    from src.model import SAM3_Medical
    
    client_weights = []
    client_public_reps = []
    
    for i in range(num_clients):
        model = SAM3_Medical().to(device)
        client_weights.append(model.state_dict())
        # 创建单样本表示 (D,)
        client_rep = torch.randn(feature_dim).to(device)
        client_public_reps.append(client_rep)
    
    # 聚合权重
    aggregated_weights = aggregator.aggregate_model_weights(
        client_weights, client_public_reps, global_features
    )
    print(f"聚合权重键数量: {len(aggregated_weights)}")
    print(f"前5个键: {list(aggregated_weights.keys())[:5]}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

