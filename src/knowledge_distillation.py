"""
Knowledge Distillation: 服务器端知识蒸馏

通过将聚合后的客户端特征作为目标，使用公共数据集对全局模型进行蒸馏训练。
这是 CreamFL 的核心机制之一。

参考：CreamFL src/algorithms/MMFL.py (知识蒸馏逻辑)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import operator

# Import AMP autocast using the new API
if torch.cuda.is_available():
    from torch.amp import autocast
else:
    # For CPU, create a no-op context manager
    class _NoOpAutocast:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    autocast = lambda device_type='cpu', **kwargs: _NoOpAutocast()


class KnowledgeDistillation:
    """
    知识蒸馏器（CreamFL 服务器端蒸馏机制）
    
    使用聚合后的客户端特征作为软标签，在公共数据集上训练全局模型。
    
    核心流程（参考 MMFL.py distill 方法）：
    1. 接收聚合后的客户端特征（img_vec, txt_vec）
    2. 在公共数据集上迭代
    3. 计算全局模型输出与聚合特征之间的 MSE 损失
    4. 反向传播更新全局模型
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        kd_weight: float = 1.0,
        use_fp16: bool = False,
        grad_clip: float = 0.0
    ):
        """
        Args:
            model: 全局模型（将被蒸馏的模型）
            optimizer: 优化器
            device: 计算设备
            kd_weight: 知识蒸馏损失权重（默认: 1.0）
            use_fp16: 是否使用混合精度训练（默认: False）
            grad_clip: 梯度裁剪阈值（默认: 0.0，不裁剪）
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.kd_weight = kd_weight
        self.use_fp16 = use_fp16
        self.grad_clip = grad_clip
        
        # MSE 损失函数（用于特征蒸馏）
        self.mse_loss = nn.MSELoss()
        
        # 混合精度 scaler（如果需要）
        if use_fp16 and device == "cuda" and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
            except ImportError:
                self.scaler = None
                self.use_fp16 = False
        else:
            self.scaler = None
            self.use_fp16 = False
    
    def compute_feature_similarity(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算特征相似度损失（MSE）
        
        参考：MMFL.py 中的 code_sim 函数
        output = output.sum(axis=1) if len(output.shape) == 3 else output
        target = target.type_as(output)
        return client_loss_cri(output, target.type_as(output))
        
        Args:
            output: 模型输出特征，形状为 (B, D) 或 (B, N, D)
            target: 目标特征（聚合的客户端特征），形状为 (B, D)
        Returns:
            MSE 损失值
        """
        # 如果 output 是 3D，在空间维度上求和
        if len(output.shape) == 3:
            output = output.sum(dim=1)  # (B, N, D) -> (B, D)
        
        # 确保 target 与 output 类型和形状匹配
        target = target.type_as(output)
        
        # 计算 MSE 损失
        loss = self.mse_loss(output, target)
        
        return loss
    
    def distill_step(
        self,
        batch_data: Dict[str, torch.Tensor],
        aggregated_features: Dict[str, torch.Tensor],
        distill_dict: Dict[int, int],
        modality_types: List[str] = ["image", "text"]
    ) -> Dict[str, torch.Tensor]:
        """
        执行一步知识蒸馏
        
        参考：MMFL.py 中的蒸馏循环
        
        Args:
            batch_data: 批次数据字典，包含：
                       - 'images': 图像 (B, C, H, W)
                       - 'captions': 文本（可选）
                       - 'caption_lens': 文本长度（可选）
                       - 'index': 样本索引 (B,)
            aggregated_features: 聚合后的客户端特征字典
                               - 'img_vec': 聚合图像特征 (N, D)
                               - 'txt_vec': 聚合文本特征 (N, D)
            distill_dict: 索引映射字典，将公共数据集索引映射到聚合特征索引
            modality_types: 要蒸馏的模态类型列表（默认: ["image", "text"]）
        Returns:
            损失字典，包含：
                - 'total_loss': 总损失
                - 'img_loss': 图像损失（如果适用）
                - 'txt_loss': 文本损失（如果适用）
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 移动数据到设备
        images = batch_data.get('images', None)
        if images is not None:
            images = images.to(self.device)
        
        captions = batch_data.get('captions', None)
        caption_lens = batch_data.get('caption_lens', None)
        if captions is not None:
            captions = captions.to(self.device)
        if caption_lens is not None:
            caption_lens = caption_lens.to(self.device)
        
        index = batch_data.get('index', None)
        if index is None:
            raise ValueError("batch_data must contain 'index' field")
        
        # 将 index 转换为列表（如果是 tensor）
        if torch.is_tensor(index):
            index = index.tolist()
        
        # 获取样本索引在聚合特征中的位置
        d_idx = operator.itemgetter(*index)(distill_dict)  # 获取对应的索引
        
        # 初始化损失
        total_loss = 0.0
        loss_dict = {}
        
        # 前向传播
        if self.use_fp16:
            with autocast(device_type='cuda', enabled=True):
                output = self._forward_model(images, captions, caption_lens)
        else:
            output = self._forward_model(images, captions, caption_lens)
        
        # 计算各个模态的蒸馏损失
        if "image" in modality_types and 'img_vec' in aggregated_features:
            out_img = output.get('image_features', output.get('features', None))
            if out_img is not None:
                target_img = aggregated_features['img_vec'][d_idx, :].type_as(out_img)
                img_loss = self.compute_feature_similarity(out_img, target_img)
                total_loss += self.kd_weight * img_loss
                loss_dict['img_loss'] = img_loss.item()
        
        if "text" in modality_types and 'txt_vec' in aggregated_features:
            out_txt = output.get('text_features', output.get('caption_features', None))
            if out_txt is not None:
                target_txt = aggregated_features['txt_vec'][d_idx, :].type_as(out_txt)
                txt_loss = self.compute_feature_similarity(out_txt, target_txt)
                total_loss += self.kd_weight * txt_loss
                loss_dict['txt_loss'] = txt_loss.item()
        
        # 如果没有指定模态但 output 是单一特征
        if len(loss_dict) == 0 and 'features' in output:
            # 默认使用 'features' 字段
            features = output['features']
            if 'img_vec' in aggregated_features:
                target = aggregated_features['img_vec'][d_idx, :].type_as(features)
                total_loss = self.kd_weight * self.compute_feature_similarity(features, target)
                loss_dict['loss'] = total_loss.item()
        
        if total_loss == 0.0:
            raise ValueError("No valid modality found for distillation")
        
        loss_dict['total_loss'] = total_loss.item()
        
        # 反向传播
        if self.use_fp16 and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
        
        return loss_dict
    
    def _forward_model(
        self,
        images: Optional[torch.Tensor],
        captions: Optional[torch.Tensor],
        caption_lens: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        执行模型前向传播
        
        Args:
            images: 图像张量
            captions: 文本张量（可选）
            caption_lens: 文本长度（可选）
        Returns:
            输出字典，包含特征
        """
        # 根据模型接口调用前向传播
        if captions is not None:
            # 多模态输入
            if hasattr(self.model, 'forward'):
                output = self.model(images, captions, caption_lens)
            else:
                output = self.model({'images': images, 'captions': captions, 'caption_lens': caption_lens})
        else:
            # 单模态输入（图像）
            if hasattr(self.model, 'forward'):
                output = self.model(images)
            else:
                output = self.model({'images': images})
        
        # 如果输出不是字典，转换为字典
        if not isinstance(output, dict):
            output = {'features': output}
        
        return output
    
    def distill(
        self,
        public_loader: DataLoader,
        aggregated_features: Dict[str, torch.Tensor],
        distill_index: List[int],
        modality_types: List[str] = ["image", "text"],
        num_epochs: int = 1
    ) -> Dict[str, List[float]]:
        """
        在公共数据集上执行知识蒸馏
        
        参考：MMFL.py 中的 distill 方法
        
        Args:
            public_loader: 公共数据集 DataLoader
            aggregated_features: 聚合后的客户端特征字典
                               - 'img_vec': (N, D)
                               - 'txt_vec': (N, D)
            distill_index: 公共数据集的索引列表（用于映射到聚合特征）
            modality_types: 要蒸馏的模态类型列表
            num_epochs: 蒸馏轮数（默认: 1）
        Returns:
            训练历史字典，包含各模态的损失列表
        """
        # 创建索引映射字典
        distill_dict = {b: a for a, b in enumerate(distill_index)}
        
        # 初始化训练历史
        history = {
            'total_loss': [],
            'img_loss': [],
            'txt_loss': []
        }
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_total_loss = []
            epoch_img_loss = []
            epoch_txt_loss = []
            
            for batch_idx, batch_data in enumerate(public_loader):
                # 执行蒸馏步骤
                loss_dict = self.distill_step(
                    batch_data, aggregated_features, distill_dict, modality_types
                )
                
                # 记录损失
                epoch_total_loss.append(loss_dict['total_loss'])
                if 'img_loss' in loss_dict:
                    epoch_img_loss.append(loss_dict['img_loss'])
                if 'txt_loss' in loss_dict:
                    epoch_txt_loss.append(loss_dict['txt_loss'])
            
            # 计算平均损失
            history['total_loss'].append(sum(epoch_total_loss) / len(epoch_total_loss))
            if epoch_img_loss:
                history['img_loss'].append(sum(epoch_img_loss) / len(epoch_img_loss))
            if epoch_txt_loss:
                history['txt_loss'].append(sum(epoch_txt_loss) / len(epoch_txt_loss))
        
        return history


if __name__ == "__main__":
    print("=" * 60)
    print("测试 KnowledgeDistillation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    feature_dim = 768
    num_samples = 100
    
    # 创建虚拟模型
    from src.model import SAM3_Medical
    model = SAM3_Medical().to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 创建蒸馏器
    distiller = KnowledgeDistillation(
        model=model,
        optimizer=optimizer,
        device=device,
        kd_weight=1.0,
        use_fp16=False,
        grad_clip=1.0
    )
    
    print("\n1. 测试特征相似度计算")
    print("-" * 60)
    
    # 3D 特征
    output_3d = torch.randn(batch_size, 64, feature_dim).to(device)
    target = torch.randn(batch_size, feature_dim).to(device)
    loss_3d = distiller.compute_feature_similarity(output_3d, target)
    print(f"3D 特征损失: {loss_3d.item():.4f}")
    
    # 2D 特征
    output_2d = torch.randn(batch_size, feature_dim).to(device)
    loss_2d = distiller.compute_feature_similarity(output_2d, target)
    print(f"2D 特征损失: {loss_2d.item():.4f}")
    
    print("\n2. 测试蒸馏步骤")
    print("-" * 60)
    
    # 创建聚合特征
    aggregated_features = {
        'img_vec': torch.randn(num_samples, feature_dim).to(device),
        'txt_vec': torch.randn(num_samples, feature_dim).to(device)
    }
    
    # 创建批次数据
    batch_data = {
        'images': torch.randn(batch_size, 3, 1024, 1024).to(device),
        'index': [0, 1, 2, 3]  # 样本索引
    }
    
    # 创建索引映射
    distill_index = list(range(num_samples))
    distill_dict = {b: a for a, b in enumerate(distill_index)}
    
    # 执行蒸馏步骤（需要模型输出字典格式）
    # 这里我们模拟模型输出
    class MockModel(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.feature_dim = feature_dim
        
        def forward(self, images):
            B = images.shape[0]
            return {
                'features': torch.randn(B, self.feature_dim, device=images.device)
            }
    
    mock_model = MockModel(feature_dim).to(device)
    mock_optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
    mock_distiller = KnowledgeDistillation(
        model=mock_model,
        optimizer=mock_optimizer,
        device=device,
        kd_weight=1.0
    )
    
    # 修改 batch_data 以匹配接口
    batch_data['index'] = torch.tensor([0, 1, 2, 3])
    
    try:
        loss_dict = mock_distiller.distill_step(
            batch_data, aggregated_features, distill_dict, modality_types=["image"]
        )
        print(f"蒸馏步骤完成")
        print(f"  总损失: {loss_dict['total_loss']:.4f}")
        if 'img_loss' in loss_dict:
            print(f"  图像损失: {loss_dict['img_loss']:.4f}")
    except Exception as e:
        print(f"蒸馏步骤测试需要完整的模型接口: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

