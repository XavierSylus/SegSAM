"""
Server Manager: 服务器端管理器

控制联邦学习的通信轮次（Round），分发全局模型权重和特征表示，
收集客户端更新，执行聚合和知识蒸馏。

参考：CreamFL src/algorithms/MMFL.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple, Any
from torch.utils.data import DataLoader
import random
import gc

from src.server import CreamAggregator
from src.contrastive_aggregation import ContrastiveWeightAggregation
from src.model import SAM3_Medical, DEVICE


class ServerManager:
    """
    服务器管理器
    
    负责联邦学习的整体流程控制：
    1. 管理通信轮次（Round）
    2. 分发全局模型权重和特征表示
    3. 收集客户端更新
    4. 执行特征聚合和知识蒸馏
    5. 更新全局模型
    """
    
    def __init__(
        self,
        global_model: SAM3_Medical,
        device: str = DEVICE,
        aggregation_method: str = "contrastive_weighted",
        server_lr: float = 1e-4,
        kd_weight: float = 1.0,
        client_num_per_round: Optional[int] = None,
        enable_distillation: bool = True,
        use_fp16: bool = False,
        grad_clip: float = 0.0
    ):
        """
        Args:
            global_model: 全局模型实例
            device: 计算设备
            aggregation_method: 聚合方法（'contrastive_weighted', 'similarity_weighted', 'fedavg'）
            server_lr: 服务器学习率（用于知识蒸馏）
            kd_weight: 知识蒸馏损失权重
            client_num_per_round: 每轮参与的客户端数量（None 表示全部参与）
            enable_distillation: 是否启用知识蒸馏
            use_fp16: 是否使用混合精度训练
            grad_clip: 梯度裁剪阈值
        """
        self.device = device
        self.aggregation_method = aggregation_method
        self.client_num_per_round = client_num_per_round
        self.enable_distillation = enable_distillation
        self.current_round = 0
        
        # 初始化聚合器
        self.aggregator = CreamAggregator(
            global_model=global_model,
            device=device,
            aggregation_method=aggregation_method
        )
        
        # 初始化对比权重聚合器（如果使用）
        if aggregation_method == "contrastive_weighted":
            self.contrastive_aggregator = ContrastiveWeightAggregation(device=device)
        else:
            self.contrastive_aggregator = None
        
        # 设置知识蒸馏（如果启用）
        if enable_distillation:
            optimizer = optim.Adam(global_model.parameters(), lr=server_lr)
            self.aggregator.setup_distillation(
                optimizer=optimizer,
                kd_weight=kd_weight,
                use_fp16=use_fp16,
                grad_clip=grad_clip
            )
        
        # 全局特征表示（用于对比学习和聚合）
        self.global_img_feature = None  # (N, D)
        self.global_txt_feature = None  # (N, D)
        self.distill_index = None  # List of indices
    
    def generate_global_representations(
        self,
        public_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        生成全局特征表示（使用当前全局模型）
        
        参考：MMFL.py 中生成 global_img_feature 和 global_txt_feature
        
        Args:
            public_loader: 公共数据集 DataLoader（用于提取特征）
            eval_loader: 评估数据集 DataLoader（可选，如果与 public_loader 不同）
        Returns:
            Tuple of (global_img_feature, global_txt_feature, distill_index)
            - global_img_feature: 全局图像特征 (N, D)
            - global_txt_feature: 全局文本特征 (N, D)
            - distill_index: 样本索引列表
        """
        model = self.aggregator.get_global_model()
        model.eval()
        
        img_features = []
        txt_features = []
        distill_index = []
        
        loader = eval_loader if eval_loader is not None else public_loader
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                # 处理批次数据
                images = batch_data.get('images', None)
                captions = batch_data.get('captions', None)
                caption_lens = batch_data.get('caption_lens', None)
                index = batch_data.get('index', None)
                
                if images is not None:
                    images = images.to(self.device)
                if captions is not None:
                    captions = captions.to(self.device)
                if caption_lens is not None:
                    caption_lens = caption_lens.to(self.device)
                
                # 前向传播获取特征
                output = self._forward_model(model, images, captions, caption_lens)
                
                # 提取特征
                out_img = output.get('image_features', output.get('features', None))
                out_txt = output.get('text_features', output.get('caption_features', None))
                
                if out_img is not None:
                    img_features.append(out_img.cpu().detach())
                if out_txt is not None:
                    txt_features.append(out_txt.cpu().detach())
                
                # 收集索引
                if index is not None:
                    if torch.is_tensor(index):
                        index = index.tolist()
                    distill_index.extend(index)
                else:
                    # 如果没有提供索引，使用批次索引
                    batch_size = images.shape[0] if images is not None else captions.shape[0]
                    start_idx = batch_idx * loader.batch_size
                    distill_index.extend(range(start_idx, start_idx + batch_size))
        
        # 合并特征
        if img_features:
            global_img_feature = torch.cat(img_features, dim=0)  # (N, D)
        else:
            global_img_feature = None
        
        if txt_features:
            global_txt_feature = torch.cat(txt_features, dim=0)  # (N, D)
        else:
            global_txt_feature = None
        
        # 存储全局特征
        self.global_img_feature = global_img_feature
        self.global_txt_feature = global_txt_feature
        self.distill_index = distill_index
        
        return global_img_feature, global_txt_feature, distill_index
    
    def _forward_model(
        self,
        model: nn.Module,
        images: Optional[torch.Tensor],
        captions: Optional[torch.Tensor],
        caption_lens: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """执行模型前向传播"""
        if captions is not None:
            if hasattr(model, 'forward'):
                output = model(images, captions, caption_lens)
            else:
                output = model({'images': images, 'captions': captions, 'caption_lens': caption_lens})
        else:
            if hasattr(model, 'forward'):
                output = model(images)
            else:
                output = model({'images': images})
        
        if not isinstance(output, dict):
            output = {'features': output}
        
        return output
    
    def select_clients(
        self,
        all_clients: List[Any],
        num_clients: Optional[int] = None
    ) -> List[Any]:
        """
        选择参与当前轮训练的客户端
        
        参考：MMFL.py 中的随机采样客户端
        
        Args:
            all_clients: 所有客户端列表
            num_clients: 选择的客户端数量（None 表示全部）
        Returns:
            选中的客户端列表
        """
        if num_clients is None:
            num_clients = self.client_num_per_round
        
        if num_clients is None or num_clients >= len(all_clients):
            return all_clients
        else:
            return random.sample(all_clients, num_clients)
    
    def distribute_global_model(
        self,
        clients: List[Any]
    ):
        """
        分发全局模型权重到客户端
        
        Args:
            clients: 客户端列表（每个客户端应该有 load_model_state 方法）
        """
        global_state = self.aggregator.get_global_model().state_dict()
        
        for client in clients:
            if hasattr(client, 'trainer') and hasattr(client.trainer, 'load_model_state'):
                client.trainer.load_model_state(global_state)
            elif hasattr(client, 'load_model_state'):
                client.load_model_state(global_state)
    
    def collect_client_updates(
        self,
        clients: List[Any],
        public_loader: DataLoader
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
        """
        收集客户端更新的权重和特征表示
        
        参考：MMFL.py 中的收集客户端特征
        
        Args:
            clients: 客户端列表
            public_loader: 公共数据集 DataLoader（用于生成客户端特征）
        Returns:
            Tuple of (client_weights, client_img_features, client_txt_features)
            - client_weights: 客户端模型权重列表
            - client_img_features: 客户端图像特征列表，每个形状为 (N, D)
            - client_txt_features: 客户端文本特征列表，每个形状为 (N, D)
        """
        client_weights = []
        client_img_features = []
        client_txt_features = []
        
        for client in clients:
            # 获取客户端更新的权重
            if hasattr(client, 'trainer') and hasattr(client.trainer, 'get_model_state'):
                weights = client.trainer.get_model_state()
            elif hasattr(client, 'get_model_state'):
                weights = client.get_model_state()
            else:
                # 如果客户端没有 get_model_state 方法，尝试获取模型状态
                if hasattr(client, 'model'):
                    weights = client.model.state_dict()
                else:
                    continue
            
            client_weights.append(weights)
            
            # 生成客户端特征表示（使用公共数据集）
            # 这里假设客户端有 generate_logits 或类似方法
            if hasattr(client, 'trainer') and hasattr(client.trainer, 'generate_logits'):
                trainer = client.trainer
                _vec, _ = trainer.generate_logits(public_loader)
                
                if _vec.get('img') is not None:
                    client_img_features.append(_vec['img'])
                if _vec.get('txt') is not None:
                    client_txt_features.append(_vec['txt'])
            elif hasattr(client, 'generate_features'):
                # 自定义特征生成方法
                img_feat, txt_feat = client.generate_features(public_loader)
                if img_feat is not None:
                    client_img_features.append(img_feat)
                if txt_feat is not None:
                    client_txt_features.append(txt_feat)
        
        return client_weights, client_img_features, client_txt_features
    
    def aggregate_features(
        self,
        client_img_features: List[torch.Tensor],
        client_txt_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        聚合客户端特征（使用对比权重）
        
        参考：MMFL.py 中的 aggregation 函数
        
        Args:
            client_img_features: 客户端图像特征列表
            client_txt_features: 客户端文本特征列表
        Returns:
            Tuple of (aggregated_img_features, aggregated_txt_features)
        """
        if self.contrastive_aggregator is not None and self.global_img_feature is not None:
            # 使用对比权重聚合
            if client_img_features and self.global_txt_feature is not None:
                aggregated_img = self.contrastive_aggregator.aggregate_features(
                    client_img_features, self.global_txt_feature
                )
            else:
                aggregated_img = None
            
            if client_txt_features and self.global_img_feature is not None:
                aggregated_txt = self.contrastive_aggregator.aggregate_features(
                    client_txt_features, self.global_img_feature
                )
            else:
                aggregated_txt = None
            
            return aggregated_img, aggregated_txt
        else:
            # 简单平均聚合（回退方法）
            if client_img_features:
                aggregated_img = torch.stack(client_img_features, dim=0).mean(dim=0)
            else:
                aggregated_img = None
            
            if client_txt_features:
                aggregated_txt = torch.stack(client_txt_features, dim=0).mean(dim=0)
            else:
                aggregated_txt = None
            
            return aggregated_img, aggregated_txt
    
    def run_round(
        self,
        clients: List[Any],
        public_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_clients_per_round: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行一轮联邦学习
        
        参考：MMFL.py 中的 train 方法
        
        流程：
        1. 生成全局特征表示
        2. 选择参与训练的客户端
        3. 分发全局模型权重和特征表示
        4. 客户端本地训练
        5. 收集客户端更新
        6. 聚合特征和权重
        7. 知识蒸馏更新全局模型
        
        Args:
            clients: 所有客户端列表
            public_loader: 公共数据集 DataLoader
            eval_loader: 评估数据集 DataLoader（可选）
            num_clients_per_round: 每轮参与的客户端数量
        Returns:
            轮次信息字典，包含聚合和蒸馏的统计信息
        """
        round_info = {
            'round': self.current_round,
            'num_clients': 0,
            'aggregation_done': False,
            'distillation_done': False
        }
        
        # 1. 生成全局特征表示
        global_img_feat, global_txt_feat, distill_index = self.generate_global_representations(
            public_loader, eval_loader
        )
        
        # 2. 选择参与训练的客户端
        selected_clients = self.select_clients(clients, num_clients_per_round)
        round_info['num_clients'] = len(selected_clients)
        
        if len(selected_clients) == 0:
            return round_info
        
        # 3. 分发全局模型权重
        self.distribute_global_model(selected_clients)
        
        # 4. 客户端本地训练
        global_reps = self.aggregator.get_global_reps()
        client_public_reps_from_training = []
        
        for client in selected_clients:
            if hasattr(client, 'trainer') and hasattr(client.trainer, 'run'):
                # 调用客户端的 run 方法进行本地训练
                updated_state, local_public_reps, training_stats = client.trainer.run(
                    global_reps,
                    lambda_cream=0.02  # ★ Fix: 0.1 → 0.02，与全局配置保持一致
                )
                # 保存客户端训练返回的公共数据表示
                client_public_reps_from_training.append(local_public_reps)
            elif hasattr(client, 'train'):
                # 自定义训练方法
                result = client.train(global_reps)
                if isinstance(result, tuple) and len(result) >= 2:
                    client_public_reps_from_training.append(result[1])  # 假设第二个返回值是公共表示
                else:
                    # 使用占位符
                    embed_dim = self.aggregator.get_global_model().embed_dim
                    client_public_reps_from_training.append(torch.zeros(embed_dim))
            else:
                # 如果没有训练方法，使用占位符
                embed_dim = self.aggregator.get_global_model().embed_dim
                client_public_reps_from_training.append(torch.zeros(embed_dim))
        
        # 5. 收集客户端更新
        client_weights, client_img_features, client_txt_features = self.collect_client_updates(
            selected_clients, eval_loader if eval_loader is not None else public_loader
        )
        
        # 6. 聚合模型权重和特征
        # 6.1 聚合模型权重
        # 使用从客户端训练中获取的公共数据表示
        if len(client_public_reps_from_training) == len(client_weights):
            client_public_reps = client_public_reps_from_training
        else:
            # 如果数量不匹配，使用特征的平均值作为替代
            client_public_reps = []
            for i in range(len(client_weights)):
                img_feat = client_img_features[i] if i < len(client_img_features) and client_img_features else None
                txt_feat = client_txt_features[i] if i < len(client_txt_features) and client_txt_features else None
                
                if img_feat is not None:
                    # 使用图像特征的平均值
                    rep = img_feat.mean(dim=0) if img_feat.dim() > 1 else img_feat
                elif txt_feat is not None:
                    # 使用文本特征的平均值
                    rep = txt_feat.mean(dim=0) if txt_feat.dim() > 1 else txt_feat
                else:
                    # 使用占位符
                    embed_dim = self.aggregator.get_global_model().embed_dim
                    rep = torch.zeros(embed_dim)
                client_public_reps.append(rep)
        
        # 聚合模型权重
        aggregated_state = self.aggregator.aggregate_weights(
            client_weights, client_public_reps
        )
        round_info['aggregation_done'] = True
        
        # 6.2 聚合特征
        if client_img_features or client_txt_features:
            aggregated_img, aggregated_txt = self.aggregate_features(
                client_img_features, client_txt_features
            )
        else:
            aggregated_img = aggregated_txt = None
        
        # 7. 知识蒸馏（如果启用）
        if self.enable_distillation and aggregated_img is not None:
            history = self.aggregator.distill_global_model(
                public_loader=public_loader,
                aggregated_img_features=aggregated_img,
                aggregated_txt_features=aggregated_txt,
                distill_index=distill_index,
                modality_types=["image", "text"] if aggregated_txt is not None else ["image"],
                num_epochs=1
            )
            round_info['distillation_done'] = True
            round_info['distillation_history'] = history
        
        # 更新轮次计数
        self.current_round += 1
        
        # 清理内存
        del client_weights, client_img_features, client_txt_features
        gc.collect()
        
        return round_info
    
    def run(
        self,
        clients: List[Any],
        public_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_rounds: int = 50,
        num_clients_per_round: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        运行多轮联邦学习
        
        Args:
            clients: 所有客户端列表
            public_loader: 公共数据集 DataLoader
            eval_loader: 评估数据集 DataLoader（可选）
            num_rounds: 总轮次数
            num_clients_per_round: 每轮参与的客户端数量
        Returns:
            所有轮次的信息列表
        """
        all_rounds_info = []
        
        for round_num in range(num_rounds):
            print(f"\n{'='*60}")
            print(f"Round {round_num + 1}/{num_rounds}")
            print(f"{'='*60}")
            
            round_info = self.run_round(
                clients, public_loader, eval_loader, num_clients_per_round
            )
            round_info['round'] = round_num
            
            all_rounds_info.append(round_info)
            
            print(f"Round {round_num + 1} completed:")
            print(f"  - Clients participated: {round_info['num_clients']}")
            print(f"  - Aggregation done: {round_info['aggregation_done']}")
            print(f"  - Distillation done: {round_info['distillation_done']}")
        
        return all_rounds_info
    
    def get_global_model(self) -> SAM3_Medical:
        """获取当前全局模型"""
        return self.aggregator.get_global_model()
    
    def get_global_reps(self) -> Dict[str, torch.Tensor]:
        """获取当前全局特征表示"""
        return self.aggregator.get_global_reps()


if __name__ == "__main__":
    print("=" * 60)
    print("测试 ServerManager")
    print("=" * 60)
    
    from src.model import SAM3_Medical
    from src.client import ClientTrainer
    from torch.utils.data import DataLoader, TensorDataset
    
    device = DEVICE
    
    # 创建全局模型
    global_model = SAM3_Medical().to(device)
    
    # 创建服务器管理器
    server_manager = ServerManager(
        global_model=global_model,
        device=device,
        aggregation_method="contrastive_weighted",
        enable_distillation=True
    )
    
    print("\n服务器管理器创建成功！")
    print(f"当前轮次: {server_manager.current_round}")
    print(f"聚合方法: {server_manager.aggregation_method}")
    print(f"是否启用蒸馏: {server_manager.enable_distillation}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

