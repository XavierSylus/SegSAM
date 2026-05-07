"""
CreamFL Contrastive Losses: Privacy-preserving knowledge distillation
实现跨模态和模态内对比学习损失（CreamFL 核心机制）

参考：
- CreamFL src/algorithms/MMFL.py (对比学习聚合机制)
- CreamFL src/algorithms/ClientTrainer.py (MOON-style 对比学习)
- CreamFL src/criterions/probemb.py (MCSoftContrastiveLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    CreamFL 对比学习损失（核心机制）
    
    实现三种对比学习机制：
    1. Intra-modal 对比学习：本地特征 vs 全局特征（正样本）+ 本地特征 vs 旧模型特征（负样本）
    2. Inter-modal 对比学习：本地图像特征 vs 全局文本特征
    3. 对比学习聚合权重：计算本地特征与全局特征的对角线相似度权重
    
    参考：CreamFL ClientTrainer.py 和 MMFL.py
    """
    
    def __init__(
        self,
        tau: float = 0.5,
        temperature: float = 0.5,
        use_moon: bool = True
    ):
        """
        Args:
            tau: 温度参数（用于 inter-modal，默认: 0.5）
            temperature: 温度参数（用于 intra-modal MOON loss，默认: 0.5）
            use_moon: 是否使用 MOON-style 对比学习（包含负样本，默认: True）
        """
        super(ContrastiveLoss, self).__init__()
        self.tau = tau  # Inter-modal 温度
        self.temperature = temperature  # Intra-modal MOON 温度
        self.use_moon = use_moon
    
    def compute_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        计算两个特征之间的相似度（内积）
        
        参考：MMFL.py 中的 logits = torch.matmul(vec, global_feature.T)
        
        Args:
            x1: 第一个特征 (B, D) 或 (B, N, D)
            x2: 第二个特征 (D,) 或 (B, D) 或 (N, D)
        Returns:
            相似度矩阵或向量
        """
        # 如果 x1 是 3D，进行全局平均池化
        if x1.dim() == 3:
            x1 = x1.mean(dim=1)  # (B, N, D) -> (B, D)
        
        # 如果 x2 是 3D，进行全局平均池化
        if x2.dim() == 3:
            x2 = x2.mean(dim=1)  # (B, N, D) -> (B, D)
        
        # 如果 x2 是 1D，扩展为 2D
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)  # (D,) -> (1, D)

        # ★ Fix (2026-03-14): 维度自动对齐
        # local_features 来自 extract_features (contrastive_dim=1024)，
        # global_rep 由服务器以 embed_dim=768 初始化，两者维度可能不同。
        # 以较小维度为基准，将较大维度的一方 adaptive_avg_pool1d 对齐，确保 matmul 合法。
        d1, d2 = x1.shape[-1], x2.shape[-1]
        if d1 != d2:
            target_dim = min(d1, d2)
            if d1 > target_dim:
                x1 = F.adaptive_avg_pool1d(x1.unsqueeze(0), target_dim).squeeze(0)
            if d2 > target_dim:
                x2 = F.adaptive_avg_pool1d(x2.unsqueeze(0), target_dim).squeeze(0)

        # ★ Fix (2026-04-06): L2 归一化，将内积约束为余弦相似度 ∈ [-1, 1]
        # 修复原始内积过大导致 logsigmoid 饱和、梯度消失的问题
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)

        # 计算余弦相似度（内积）
        if x2.shape[0] == 1:
            similarity = torch.matmul(x1, x2.t())  # (B, 1)
        else:
            similarity = torch.matmul(x1, x2.t())  # (B, B) 或 (B, N)
        
        return similarity
    
    def intra_modal_loss(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        old_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        模态内对比学习损失（MOON-style）
        
        参考：ClientTrainer.py 中的 intra-modal contrasting
        
        Args:
            local_features: 本地特征 (B, D) 或 (B, N, D)
            global_features: 全局特征（正样本）(B, D) 或 (D,)
            old_features: 旧模型特征（负样本）(B, D) 或 (B, N, D)，可选
        Returns:
            模态内对比损失
        """
        # 归一化特征（如果需要）
        if local_features.dim() == 3:
            local_features = local_features.mean(dim=1)  # (B, N, D) -> (B, D)
        
        # 计算正样本相似度（本地特征 vs 全局特征）
        pos_sim = self.compute_similarity(local_features, global_features)  # (B, 1) 或 (B,)
        if pos_sim.dim() == 2:
            pos_sim = pos_sim.squeeze(1)  # (B, 1) -> (B,)
        
        if self.use_moon and old_features is not None:
            # MOON-style：包含负样本（本地特征 vs 旧模型特征）
            if old_features.dim() == 3:
                old_features = old_features.mean(dim=1)
            
            # 计算负样本相似度
            neg_sim = self.compute_similarity(local_features, old_features)
            if neg_sim.dim() == 2:
                neg_sim = neg_sim.squeeze(1)
            
            # 构造 logits: [pos_sim, neg_sim]
            logits = torch.stack([pos_sim, neg_sim], dim=1)  # (B, 2)
            
            # 应用温度缩放
            logits = logits / self.temperature
            
            # 标签：正样本为 0（第一个位置）
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
            
            # 交叉熵损失
            loss = F.cross_entropy(logits, labels)
        else:
            # 简化版本：仅使用正样本（最大化相似度）
            # 使用负对数似然：-log(sigmoid(sim / tau))
            logits = pos_sim / self.temperature
            loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def inter_modal_loss(
        self,
        local_features: torch.Tensor,
        global_cross_features: torch.Tensor,
        positive_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        跨模态对比学习损失
        
        参考：ClientTrainer.py 中的 inter-modal contrasting
        logits = torch.div(torch.matmul(im_feature, global_txt_feature.T), tau)
        
        Args:
            local_features: 本地特征（例如：图像特征）(B, D) 或 (B, N, D)
            global_cross_features: 全局跨模态特征（例如：文本特征）(N, D)
            positive_indices: 正样本索引 (B,)，可选（默认：对角线）
        Returns:
            跨模态对比损失
        """
        # 归一化特征
        if local_features.dim() == 3:
            local_features = local_features.mean(dim=1)  # (B, N, D) -> (B, D)
        
        # ★ Fix (2026-03-14): 改用 compute_similarity 替代裸写的 torch.matmul。
        # compute_similarity 内置 F.adaptive_avg_pool1d 维度对齐逻辑，
        # 可自动处理 local_features (contrastive_dim=1024) 与
        # global_cross_features (text_dim=768) 之间的维度不匹配。
        # logits: (B, N) - 每个本地特征与所有全局特征的相似度
        similarity_matrix = self.compute_similarity(local_features, global_cross_features)  # (B, N)
        logits = similarity_matrix / self.tau  # 温度缩放
        
        # 确定正样本索引
        if positive_indices is None:
            # 如果 global_cross_features 的 batch size 与 local_features 相同，使用对角线
            if global_cross_features.shape[0] == local_features.shape[0]:
                positive_indices = torch.arange(
                    local_features.shape[0],
                    device=logits.device
                )
            else:
                # 否则使用第一个索引（假设对应关系）
                positive_indices = torch.zeros(
                    local_features.shape[0],
                    dtype=torch.long,
                    device=logits.device
                )
        
        # 交叉熵损失（需要 ≥2 个类别才有非退化解）
        if logits.shape[-1] >= 2:
            loss = F.cross_entropy(logits, positive_indices)
        else:
            # ★ Fix (2026-04-06): batch_size=1 退化回退
            # softmax 只有 1 个元素时 cross_entropy 恒为 0（数学恒等式）
            # 改用 -logsigmoid(sim/temperature) 直接优化相似度
            loss = -F.logsigmoid(logits.squeeze() / self.temperature).mean()
        
        return loss
    
    def compute_contrastive_weights(
        self,
        local_features_list: list,
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比学习聚合权重（用于 CreamFL 聚合机制）
        
        参考：MMFL.py 中的 con_w 方法
        logits = torch.matmul(vec, global_feature.T)
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        contrastive_w = torch.diagonal(log_prob)
        
        Args:
            local_features_list: 本地特征列表，每个元素形状为 (N, D)
            global_features: 全局特征 (N, D)
        Returns:
            对比权重 (num_clients, N)
        """
        num_clients = len(local_features_list)
        N = global_features.shape[0]
        
        # 初始化权重矩阵
        contrastive_w = torch.zeros(num_clients, N, device=global_features.device)
        
        for i, local_vec in enumerate(local_features_list):
            # 计算相似度矩阵
            logits = torch.matmul(local_vec, global_features.t())  # (N, N)
            
            # 计算 log 概率（softmax 的 log）
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True) + 1e-8)
            
            # 提取对角线元素（正样本对的对数概率）
            contrastive_w[i] = torch.diagonal(log_prob).reshape(-1)
        
        # 对客户端维度应用 softmax（归一化权重）
        contrastive_w = F.softmax(contrastive_w, dim=0)  # (num_clients, N)
        
        return contrastive_w
    
    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        global_cross_features: Optional[torch.Tensor] = None,
        old_features: Optional[torch.Tensor] = None,
        positive_indices: Optional[torch.Tensor] = None,
        use_inter: bool = True,
        use_intra: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        计算对比学习损失
        
        Args:
            local_features: 本地特征 (B, D) 或 (B, N, D)
            global_features: 全局特征（正样本，用于 intra-modal）(B, D) 或 (D,)
            global_cross_features: 全局跨模态特征（用于 inter-modal）(N, D)，可选
            old_features: 旧模型特征（负样本）(B, D) 或 (B, N, D)，可选
            positive_indices: 正样本索引（用于 inter-modal）(B,)，可选
            use_inter: 是否计算 inter-modal 损失（默认: True）
            use_intra: 是否计算 intra-modal 损失（默认: True）
        Returns:
            损失字典，包含：
                - 'intra_loss': 模态内损失
                - 'inter_loss': 跨模态损失
                - 'total_loss': 总损失
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Intra-modal 损失
        if use_intra:
            intra_loss = self.intra_modal_loss(
                local_features, global_features, old_features
            )
            loss_dict['intra_loss'] = intra_loss
            total_loss += intra_loss
        
        # Inter-modal 损失
        if use_inter and global_cross_features is not None:
            inter_loss = self.inter_modal_loss(
                local_features, global_cross_features, positive_indices
            )
            loss_dict['inter_loss'] = inter_loss
            total_loss += inter_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class CreamContrastiveLoss(nn.Module):
    """
    兼容旧版本的 CreamContrastiveLoss（向后兼容）
    
    内部使用新的 ContrastiveLoss
    """
    
    def __init__(self, tau: float = 0.07):
        """
        Args:
            tau: Temperature parameter for contrastive learning (default: 0.07)
        """
        super(CreamContrastiveLoss, self).__init__()
        # ★ Fix High 5 (2026-03-13): intra-modal 温度（MOON style）应为 0.5
        # 原来 temperature=tau=0.07 导致 intra-modal loss 数值极大（梯度爆炸风险）。
        # 参考 CreamFL 原论文 Table 3：inter-modal tau=0.07, intra-modal temperature=0.5
        self.contrastive_loss = ContrastiveLoss(tau=tau, temperature=0.5, use_moon=False)
        self.tau = tau
    
    def forward(
        self,
        local_features: torch.Tensor,
        global_text_rep: torch.Tensor,
        global_image_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算跨模态和模态内对比损失（向后兼容接口）
        
        Args:
            local_features: 本地图像特征 (B, N, D) 或 (B, D)
            global_text_rep: 全局文本表示 (B, D) 或 (D,)
            global_image_rep: 全局图像表示 (B, D) 或 (D,)
        Returns:
            Tuple of (L_inter, L_intra) losses
        """
        # 使用新的 ContrastiveLoss 计算
        loss_dict = self.contrastive_loss(
            local_features=local_features,
            global_features=global_image_rep,
            global_cross_features=global_text_rep.unsqueeze(0) if global_text_rep.dim() == 1 else global_text_rep,
            use_inter=True,
            use_intra=True
        )
        
        L_inter = loss_dict.get('inter_loss', torch.tensor(0.0, device=local_features.device))
        L_intra = loss_dict.get('intra_loss', torch.tensor(0.0, device=local_features.device))
        
        return L_inter, L_intra
    
    def compute_total_loss(
        self,
        local_features: torch.Tensor,
        global_text_rep: torch.Tensor,
        global_image_rep: torch.Tensor,
        lambda_inter: float = 1.0,
        lambda_intra: float = 1.0
    ) -> torch.Tensor:
        """
        Compute weighted sum of inter-modal and intra-modal losses.
        
        Args:
            local_features: Local image features
            global_text_rep: Global text representation
            global_image_rep: Global image representation
            lambda_inter: Weight for inter-modal loss (default: 1.0)
            lambda_intra: Weight for intra-modal loss (default: 1.0)
        Returns:
            Total contrastive loss
        """
        L_inter, L_intra = self.forward(local_features, global_text_rep, global_image_rep)
        total_loss = lambda_inter * L_inter + lambda_intra * L_intra
        return total_loss


if __name__ == "__main__":
    print("=" * 60)
    print("测试 CreamFL 对比学习损失")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 6
    feature_dim = 768
    num_tokens = 64  # 简化测试：64 个 tokens
    
    # 创建虚拟输入
    local_features = torch.randn(batch_size, num_tokens, feature_dim).to(device)
    global_text_rep = torch.randn(feature_dim).to(device)  # 单个全局文本表示
    global_image_rep = torch.randn(feature_dim).to(device)  # 单个全局图像表示
    old_features = torch.randn(batch_size, num_tokens, feature_dim).to(device)  # 旧模型特征
    
    print("\n1. 测试新的 ContrastiveLoss（完整功能）")
    print("-" * 60)
    
    contrastive_loss = ContrastiveLoss(tau=0.5, temperature=0.5, use_moon=True)
    
    # 测试 Intra-modal 损失（带 MOON）
    loss_dict_intra = contrastive_loss(
        local_features=local_features,
        global_features=global_image_rep,
        old_features=old_features,
        use_inter=False,
        use_intra=True
    )
    print(f"Intra-modal 损失 (带 MOON): {loss_dict_intra['intra_loss'].item():.4f}")
    
    # 测试 Inter-modal 损失
    global_text_rep_batch = torch.randn(batch_size, feature_dim).to(device)
    loss_dict_inter = contrastive_loss(
        local_features=local_features,
        global_features=global_image_rep,
        global_cross_features=global_text_rep_batch,
        use_inter=True,
        use_intra=False
    )
    print(f"Inter-modal 损失: {loss_dict_inter['inter_loss'].item():.4f}")
    
    # 测试完整损失（Inter + Intra）
    loss_dict_full = contrastive_loss(
        local_features=local_features,
        global_features=global_image_rep,
        global_cross_features=global_text_rep_batch,
        old_features=old_features,
        use_inter=True,
        use_intra=True
    )
    print(f"总损失: {loss_dict_full['total_loss'].item():.4f}")
    print(f"  - Intra-modal: {loss_dict_full['intra_loss'].item():.4f}")
    print(f"  - Inter-modal: {loss_dict_full['inter_loss'].item():.4f}")
    
    print("\n2. 测试对比学习聚合权重（CreamFL 聚合机制）")
    print("-" * 60)
    
    num_clients = 3
    num_samples = 100
    local_features_list = [
        torch.randn(num_samples, feature_dim).to(device) 
        for _ in range(num_clients)
    ]
    global_features_pool = torch.randn(num_samples, feature_dim).to(device)
    
    contrastive_weights = contrastive_loss.compute_contrastive_weights(
        local_features_list, global_features_pool
    )
    print(f"对比权重形状: {contrastive_weights.shape}")  # (num_clients, num_samples)
    print(f"权重和（每列应该为 1）: {contrastive_weights.sum(dim=0)[:5]}")  # 检查归一化
    
    print("\n3. 测试向后兼容的 CreamContrastiveLoss")
    print("-" * 60)
    
    cream_loss = CreamContrastiveLoss(tau=0.07)
    
    # 计算损失
    L_inter, L_intra = cream_loss(local_features, global_text_rep, global_image_rep)
    print(f"L_inter: {L_inter.item():.4f}")
    print(f"L_intra: {L_intra.item():.4f}")
    
    # 计算总损失
    total_loss = cream_loss.compute_total_loss(
        local_features, global_text_rep, global_image_rep
    )
    print(f"总对比学习损失: {total_loss.item():.4f}")
    
    print("\n4. 测试批次级别的全局表示")
    print("-" * 60)
    
    global_text_rep_batch = torch.randn(batch_size, feature_dim).to(device)
    global_image_rep_batch = torch.randn(batch_size, feature_dim).to(device)
    
    L_inter_batch, L_intra_batch = cream_loss(
        local_features, global_text_rep_batch, global_image_rep_batch
    )
    print(f"L_inter (batch): {L_inter_batch.item():.4f}")
    print(f"L_intra (batch): {L_intra_batch.item():.4f}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


# =============================================================================
# ★ 独立工具函数：log_dice_loss（代数重排 + 拉普拉斯平滑）
# =============================================================================

def log_dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Log-Dice Loss（代数重排 + 拉普拉斯平滑），绝对杜绝除零梯度爆炸。

    数学推导：
    原始 Dice Loss = 1 - (2|P∩T| + ε) / (|P| + |T| + ε)

    当 T 全为 0（纯背景切片）时，分子 ≈ 0，分母 ≈ ε，
    loss ≈ 1 - 0/ε → 梯度 ∝ 1/ε²，ε 越小梯度越大，引起爆炸。

    代数重排为对数减法形式：

        Log-Dice = log(|P| + |T| + smooth) - log(2|P∩T| + smooth)

    全背景时（P ≈ 0, T = 0）：
        log(0 + 0 + 1.0) - log(0 + 1.0) = log(1) - log(1) = 0
    → 梯度为 0，完全安全。

    参数：
        probs:  已经过 sigmoid 的概率图，形状任意，必须已展平
                OR 形状 (B, C, H, W)（函数内部保留 B/C 维度，展平 H×W）
        target: 对应真实标签（0/1 float），形状与 probs 一致
        smooth: 拉普拉斯平滑因子（强烈建议 1.0，远大于传统 1e-5 的原因：
                当 |P|+|T|=0 时，log(0+1e-5) ≈ -11.5 而非 0，
                会产生不期望的负 loss。smooth=1.0 保证全背景时 loss=0。）
    返回：
        标量 loss 值（batch 内均值）

    笔误说明（已修复）：
        旧版曾误写 `P.shape[11]` 索引越界，导致 IndexError。
        正确写法为保留 Batch 第 0 维（shape[0]）和 Channel 第 1 维（shape[1]），
        使用 view(B, num_ch, -1) 仅展平空间维度 H×W，或直接 view(B, -1) 展平全部。
    """
    # ① 安全校验
    assert probs.shape == target.shape, (
        f"[log_dice_loss] probs/target 形状不一致: {probs.shape} vs {target.shape}"
    )
    if probs.dim() < 2:
        raise ValueError(
            f"[log_dice_loss] 输入维度过低 ({probs.dim()}D)，至少需要 2D 张量。"
        )

    B = probs.shape[0]  # Batch 维度
    # ★ 关键修复：保留 Batch 维，展平其余所有维度（C×H×W 或 H×W）
    # 不能硬编码为 P.shape[11]（IndexError），必须使用 view(B, -1)
    P = probs.contiguous().view(B, -1)   # (B, C*H*W)
    T = target.contiguous().view(B, -1)  # (B, C*H*W)

    # ② 计算交集和总和
    intersection = (P * T).sum(dim=1)        # (B,)
    sum_pt       = P.sum(dim=1) + T.sum(dim=1)  # (B,)

    # ③ 对数减法（等价于对数 Dice，但无除零风险）
    # 全背景时：sum_pt=0, intersection=0
    #   -> log(0+smooth) - log(0+smooth) = 0  🟢
    # 完美分割时：2*intersection ≈ sum_pt
    #   -> log(X+smooth) - log(X+smooth) ≈ 0  🟢
    # 完全错误时：intersection=0, sum_pt 较大
    #   -> log(large+smooth) - log(smooth) > 0  → 惩罚增大  🟢
    loss_per_batch = (
        torch.log(sum_pt + smooth) -
        torch.log(2.0 * intersection + smooth)
    )  # (B,)，值域 [0, +∞)

    return loss_per_batch.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.
    用于对抗 BraTS 98.6% 背景占比引发的"零 Dice 坍塌"问题。

    公式: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    关键设计：
    - 内部使用 F.binary_cross_entropy_with_logits 接收 logits，保证数值稳定
    - alpha 平衡前/背景权重；gamma 将梯度聚焦于难分类的小病灶像素
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: 前景像素权重（建议值 0.25，背景权重为 1-alpha=0.75）
            gamma: 聚焦因子，越大越关注难分类样本（建议值 2.0）
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   模型输出 Logits，形状 (B, C, H, W)，未经过 sigmoid
            target: 真实标签，形状 (B, C, H, W)，值 0 或 1（浮点型）
        Returns:
            标量损失值
        """
        # ① 逐像素 BCE（不做 reduction），with_logits 保证数值稳定性
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        # ② p_t = exp(-bce)，等价于 sigmoid(pred) when target==1 else 1-sigmoid(pred)
        p_t = torch.exp(-bce)
        # ③ 难分类像素 p_t 低 -> (1-p_t)^gamma 大 -> 自动增大梯度权重
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss —— Dice Loss 的泛化版本，专为不平衡医学图像分割设计。

    公式（Salehi 2017 标准）:
        Tversky(a, b) = (TP + smooth) / (TP + a*FP + b*FN + smooth)
        Loss = 1 - Tversky

    参数选择策略（BraTS 建议）：
        alpha=0.3 : 轻罚假阳性（FP），允许模型"放心预测"前景
        beta=0.7  : 重罚假阴性（FN），逼迫模型不敢漏诊病灶
        -> 相当于在 Dice 基础上对漏诊施加 2.3x 额外惩罚

    参考：
        Salehi S S M, et al. Tversky Loss Function for Image Segmentation
        Using 3D Fully Convolutional Deep Networks. MICCAI Workshop 2017.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5
    ):
        """
        Args:
            alpha:  假阳性 (FP) 惩罚权重（建议值 0.3）
            beta:   假阴性 (FN) 惩罚权重（建议值 0.7）
            smooth: 平滑项，防止除零（默认 1e-5）
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 Tversky Loss，逐通道独立计算后求均值。

        Args:
            pred:   模型输出 Logits，形状 (B, C, H, W)，未经过 sigmoid
            target: 真实标签，形状 (B, C, H, W)，值 0 或 1（浮点型）
        Returns:
            标量损失值
        """
        # ① 将 Logits 转换为前景概率
        pred_probs = torch.sigmoid(pred)  # (B, C, H, W)

        num_channels = pred.shape[1]
        tversky_scores = []

        for c in range(num_channels):
            pred_c = pred_probs[:, c, :, :].reshape(-1)   # (B*H*W,)
            target_c = target[:, c, :, :].reshape(-1)     # (B*H*W,)

            # ② 全零目标通道（该类在此 batch 中完全不存在）-> 跳过，贡献 0 loss
            if target_c.sum() < self.smooth:
                tversky_scores.append(torch.tensor(1.0, device=pred.device))
                continue

            tp = (pred_c * target_c).sum()               # 真阳性
            fp = (pred_c * (1.0 - target_c)).sum()       # 假阳性
            fn = ((1.0 - pred_c) * target_c).sum()       # 假阴性

            tversky_c = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            # ③ 防止数值越界
            tversky_c = torch.clamp(tversky_c, 0.0, 1.0)
            tversky_scores.append(tversky_c)

        # ④ 逐通道 Tversky 均值 -> Loss = 1 - Tversky
        mean_tversky = torch.stack(tversky_scores).mean()
        return 1.0 - mean_tversky


class RobustMedicalLoss(nn.Module):
    """
    健壮医学图像分割损失函数 v5（2026-03-18 全向量化重构）
    专为极端类别不平衡设计（BraTS：98.6% 背景）。

    双路并行：
      路径 A（Tversky，α=0.3/β=0.7）：sigmoid→TP/FP/FN→无条件全局聚合，重罚假阳性。
                                       ★ 严禁任何 active_mask 截断！
      路径 B（Log-Dice）：active_mask（≥100px）门控 + 代数重排对数减法 + 零 for-loop 向量化。
    total = w_tversky · loss_A + w_log_dice · loss_B
    """

    _SMOOTH: float = 1.0

    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        w_tversky: float = 1.0,
        w_log_dice: float = 1.0,
        min_positive_pixels: int = 100,
    ) -> None:
        super().__init__()
        if not (0.0 < tversky_alpha < 1.0 and 0.0 < tversky_beta < 1.0):
            raise ValueError(
                f"[RobustMedicalLoss] α/β 必须在 (0,1) 内，当前: α={tversky_alpha}, β={tversky_beta}"
            )
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.w_tversky = w_tversky
        self.w_log_dice = w_log_dice
        self.min_positive_pixels = min_positive_pixels
        self.smooth: float = self._SMOOTH

    # -----------------------------------------------------------------------
    # 辅助方法 2：mask 格式自动对齐 → One-Hot (B, C, H, W)
    # -----------------------------------------------------------------------
    def _align_mask_to_pred(
        self,
        pred: torch.Tensor,    # (B, C, H, W)
        mask: torch.Tensor,    # 支持 (B,C,H,W) / (B,1,H,W) / (B,H,W)
    ) -> torch.Tensor:         # (B, C, H, W) float
        """
        将 mask 自动对齐为与 pred 相同的 One-Hot 格式。

        支持格式：
            (a) (B, C, H, W)  — 已是 One-Hot，直接返回
            (b) (B, 1, H, W)  — 类别索引，执行 One-Hot 编码
            (c) (B, H, W)     — 类别索引（无通道维），先 unsqueeze 再编码
        """
        num_classes: int = pred.shape[1]

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)    # (B, H, W) → (B, 1, H, W)

        mask_channels: int = mask.shape[1]

        if mask_channels == num_classes:
            return mask.float()

        if mask_channels == 1:
            mask_long: torch.Tensor = mask.squeeze(1).long()         # (B, H, W)
            one_hot: torch.Tensor = F.one_hot(
                mask_long, num_classes=num_classes
            )                                                         # (B, H, W, C)
            return one_hot.permute(0, 3, 1, 2).contiguous().float()  # (B, C, H, W)

        raise RuntimeError(
            f"\n{'='*70}\n"
            f"[RobustMedicalLoss] 无法自动对齐 mask 通道数\n"
            f"{'='*70}\n"
            f"  pred.shape = {pred.shape}  (期望 mask 通道数 = {num_classes})\n"
            f"  mask.shape = {mask.shape}  (实际 mask 通道数 = {mask_channels})\n"
            f"{'='*70}\n"
            f"请确保 mask 为以下格式之一：\n"
            f"  (a) One-Hot   (B, {num_classes}, H, W)\n"
            f"  (b) 类别索引  (B, 1, H, W)，值域 [0, {num_classes-1}]\n"
        )

    # -----------------------------------------------------------------------
    # 路径 A：Tversky Loss（★ 严禁 active_mask 截断）
    # -----------------------------------------------------------------------
    def _tversky_loss(
        self,
        P: torch.Tensor,    # (B, C, N) sigmoid 概率，N = H*W
        T: torch.Tensor,    # (B, C, N) One-Hot 标签 float
    ) -> torch.Tensor:      # 标量
        """
        无条件 Tversky Loss，覆盖全部通道和全部样本。

        公式（Salehi 2017 标准）：
            TP  = Σ(P · T)         ∈ ℝ^(B,C)
            FP  = Σ(P · (1−T))    ∈ ℝ^(B,C)
            FN  = Σ((1−P) · T)    ∈ ℝ^(B,C)
            tversky = (TP + ε) / (TP + α·FP + β·FN + ε)
            loss    = 1 − mean(tversky)

        ★ 禁令：
            - 不对 TP/FP/FN 乘任何 active_mask
            - 不对任何通道 skip 或 continue
            - 分母保证 > 0（ε=smooth=1.0 > 0 恒成立）
        """
        smooth: float = self.smooth   # 1.0

        TP: torch.Tensor = (P * T).sum(dim=2)              # (B, C)
        FP: torch.Tensor = (P * (1.0 - T)).sum(dim=2)     # (B, C)
        FN: torch.Tensor = ((1.0 - P) * T).sum(dim=2)     # (B, C)

        numerator: torch.Tensor   = TP + smooth
        denominator: torch.Tensor = (
            TP
            + self.tversky_alpha * FP   # α·FP（轻罚假阳性，α=0.3）
            + self.tversky_beta  * FN   # β·FN（重罚漏诊/假阴性，β=0.7）
            + smooth
        )

        # smooth=1.0 时分母永远 > 0；clamp 作双保险
        tversky_score: torch.Tensor = numerator / denominator.clamp(min=smooth)
        tversky_score = tversky_score.clamp(0.0, 1.0)  # 防浮点累积误差

        # 对全部 B 和 C 取均值（无条件，绝不 mask 截断）
        return 1.0 - tversky_score.mean()

    # -----------------------------------------------------------------------
    # 路径 B：向量化 Log-Dice（零 Python for-loop）
    # -----------------------------------------------------------------------
    def _log_dice_loss(
        self,
        probs: torch.Tensor,   # (B, C, H, W) sigmoid 概率
        target: torch.Tensor,  # (B, C, H, W) One-Hot float
    ) -> torch.Tensor:
        s = self.smooth
        B, C, H, W = probs.shape

        # active_mask：Batch 内每通道前景像素总数 ≥ 阈值，(C,) bool
        pos_counts = target.sum(dim=(0, 2, 3))                          # (C,)
        active_mask = pos_counts >= float(self.min_positive_pixels)     # (C,) bool

        if not active_mask.any():
            return probs.sum() * 0.0  # 全背景：安全归零，保持计算图联通

        # 展平空间维，仅取激活通道 → (B, C_active, N)
        P_flat = probs.view(B, C, -1)[:, active_mask, :]   # (B, C_active, N)
        T_flat = target.view(B, C, -1)[:, active_mask, :]  # (B, C_active, N)

        # 跨 B 和 N 求和 → (C_active,)
        intersection = (P_flat * T_flat).sum(dim=(0, 2))
        sum_pt = P_flat.sum(dim=(0, 2)) + T_flat.sum(dim=(0, 2))

        # 代数重排对数减法：全背景时 sum=inter=0 → log(s)-log(s)=0，梯度安全
        log_dice_per_ch = torch.log(sum_pt + s) - torch.log(2.0 * intersection + s)
        return log_dice_per_ch.mean()

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,    # (B, C, H, W) Logits，未经 sigmoid
        target: torch.Tensor,  # (B,C,H,W)/(B,1,H,W)/(B,H,W)
    ) -> torch.Tensor:
        assert pred.dim() == 4, f"[RobustMedicalLoss] pred 必须 4D，实际: {pred.shape}"

        target = self._align_mask_to_pred(pred, target)

        assert pred.shape == target.shape, (
            f"[RobustMedicalLoss] 对齐后形状不匹配: pred={pred.shape}, target={target.shape}"
        )

        B, C = pred.shape[0], pred.shape[1]
        probs = torch.sigmoid(pred)       # (B, C, H, W)
        P = probs.view(B, C, -1)          # (B, C, H*W)
        T = target.view(B, C, -1).float() # (B, C, H*W)

        loss_A = self._tversky_loss(P, T)
        loss_B = self._log_dice_loss(probs, target)

        return self.w_tversky * loss_A + self.w_log_dice * loss_B


