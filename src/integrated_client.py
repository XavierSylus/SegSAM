"""
Integrated Client Trainer for Multimodal Federated Learning

整合了：
1. SAM3MedicalIntegrated模型
2. CreamFL多模态对比学习
3. FedFMS联邦训练策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import copy

# Import AMP
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler, autocast
else:
    GradScaler = None
    class _NoOpAutocast:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    autocast = lambda enabled=False, **kwargs: _NoOpAutocast()

from src.integrated_model import SAM3MedicalIntegrated, DEVICE, LR
from src.server import CreamAggregator


class IntegratedClientTrainer:
    """
    整合的客户端训练器
    
    整合了：
    - SAM3MedicalIntegrated模型
    - CreamFL多模态对比学习损失
    - FedFMS联邦训练策略
    """
    
    def __init__(
        self,
        model: SAM3MedicalIntegrated,
        private_loader: DataLoader,
        public_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = DEVICE,
        use_amp: bool = True,
        lambda_seg: float = 1.0,
        lambda_cream: float = 0.02,  # ★ Fix: 0.1 → 0.02，与全局配置保持一致
        grad_clip: float = 1.0          # ★ 梯度裁剪阈值，0.0=禁用
    ):
        """
        Args:
            model: SAM3MedicalIntegrated模型实例
            private_loader: 私有分割数据DataLoader
            public_loader: 公共对比学习数据DataLoader
            optimizer: 优化器
            device: 设备
            use_amp: 是否使用混合精度
            lambda_seg: 分割损失权重
            lambda_cream: CreamFL对比损失权重
        """
        self.model = model.to(device)
        self.private_loader = private_loader
        self.public_loader = public_loader
        self.device = device
        self.use_amp = use_amp
        self.lambda_seg = lambda_seg
        self.lambda_cream = lambda_cream
        self.grad_clip = grad_clip
        
        # 初始化优化器
        if optimizer is None:
            # 使用 filter 确保只传入可训练参数，增强通用性
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = torch.optim.Adam(trainable_params, lr=LR)
        else:
            self.optimizer = optimizer
        
        # 混合精度scaler
        if use_amp and device == "cuda" and torch.cuda.is_available():
            if GradScaler is not None:
                self.scaler = GradScaler()
            else:
                self.scaler = None
                self.use_amp = False
        else:
            self.scaler = None
            self.use_amp = False
        
        # 分割损失
        self.seg_criterion = nn.BCEWithLogitsLoss()
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """计算Dice损失"""
        pred_probs = torch.sigmoid(pred)
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def _segmentation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算分割损失（BCE + Dice）"""
        # [Fix] Binarize target: Treat all tumor subclasses (1, 2, 4) as foreground (1.0)
        # This prevents Dice > 1.0 and weight explosion in BCE
        target_binary = (target > 0).float()
        
        bce_loss = self.seg_criterion(pred, target_binary)
        dice_loss = self._dice_loss(pred, target_binary)
        return bce_loss + dice_loss
    
    def train_epoch(
        self,
        global_reps: Dict[str, torch.Tensor],
        num_local_epochs: int = 1
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, dict]:
        """
        训练一个epoch
        
        Args:
            global_reps: 全局表示 {'global_text_rep': (D,), 'global_image_rep': (D,)}
            num_local_epochs: 本地训练轮数
        
        Returns:
            (updated_state_dict, local_public_rep, metrics_dict)
        """
        self.model.train()
        
        global_image_rep = global_reps['global_image_rep'].to(self.device)
        # ★ 硬性要求：global_text_rep 必须 .detach()，阻断反向传播追踪到服务器计算图
        _raw_text_rep = global_reps.get('global_text_rep', None)
        global_text_rep_detached = (
            _raw_text_rep.detach().to(self.device) if _raw_text_rep is not None else None
        )
        
        local_public_reps_list = []
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cream_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_local_epochs):
            private_iter = iter(self.private_loader)
            public_iter = iter(self.public_loader)
            
            try:
                while True:
                    # 私有数据：分割训练
                    try:
                        private_batch = next(private_iter)
                        if isinstance(private_batch, (list, tuple)):
                            private_img, private_mask = private_batch[0], private_batch[1]
                        else:
                            private_img = private_batch
                            private_mask = None
                        private_img = private_img.to(self.device)
                        if private_mask is not None:
                            private_mask = private_mask.to(self.device)
                    except StopIteration:
                        break
                    
                    # 公共数据：对比学习
                    try:
                        public_batch = next(public_iter)
                        if isinstance(public_batch, (list, tuple)):
                            public_img = public_batch[0]
                        else:
                            public_img = public_batch
                        public_img = public_img.to(self.device)
                    except StopIteration:
                        public_iter = iter(self.public_loader)
                        public_batch = next(public_iter)
                        if isinstance(public_batch, (list, tuple)):
                            public_img = public_batch[0]
                        else:
                            public_img = public_batch
                        public_img = public_img.to(self.device)
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    
                    with autocast(enabled=self.use_amp):
                        # 私有数据：分割
                        output_private = self.model(
                            private_img,
                            return_features=False,
                            global_text_rep=global_text_rep_detached,
                        )
                        seg_logits = output_private['logits']
                        
                        if private_mask is not None:
                            seg_loss = self._segmentation_loss(seg_logits, private_mask)
                        else:
                            seg_loss = torch.tensor(0.0, device=self.device)
                        
                        # 公共数据：对比学习
                        public_features = self.model.extract_features(public_img)
                        
                        # 计算对比损失（使用全局表示）
                        # 将全局表示扩展为batch维度
                        global_rep_expanded = global_image_rep.unsqueeze(0).expand(
                            public_features.size(0), -1
                        )
                        
                        # 计算对比损失
                        if self.model.contrastive_loss_fn is not None:
                            # 需要将特征reshape为 (B, K, D) 格式
                            if len(public_features.shape) == 3:
                                # (B, N, D) -> 取平均或池化 -> (B, D)
                                public_features_pooled = public_features.mean(dim=1)
                            else:
                                public_features_pooled = public_features
                            
                            # 扩展维度以匹配CreamFL的输入格式
                            public_features_expanded = public_features_pooled.unsqueeze(1)  # (B, 1, D)
                            global_rep_expanded_3d = global_rep_expanded.unsqueeze(1)  # (B, 1, D)
                            
                            cream_loss = self.model.compute_contrastive_loss(
                                public_features_expanded,
                                global_rep_expanded_3d
                            )
                        else:
                            cream_loss = torch.tensor(0.0, device=self.device)
                        
                        # 总损失
                        total_batch_loss = self.lambda_seg * seg_loss + self.lambda_cream * cream_loss
                    
                    # 反向传播 + 梯度裁剪（防 Logits 爆炸）
                    if self.use_amp and self.scaler is not None:
                        self.scaler.scale(total_batch_loss).backward()  # ① scaled backward
                        if self.grad_clip > 0.0:
                            self.scaler.unscale_(self.optimizer)         # ② 解除缩放，恢复真实梯度量级
                            torch.nn.utils.clip_grad_norm_(
                                [p for group in self.optimizer.param_groups
                                 for p in group['params'] if p.grad is not None],
                                max_norm=self.grad_clip
                            )                                            # ③ 裁剪（必须在 unscale 之后）
                        self.scaler.step(self.optimizer)                 # ④ 参数更新
                        self.scaler.update()                             # ⑤ 更新缩放因子
                    else:
                        total_batch_loss.backward()                      # ① backward
                        if self.grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                [p for group in self.optimizer.param_groups
                                 for p in group['params'] if p.grad is not None],
                                max_norm=self.grad_clip
                            )                                            # ② 裁剪
                        self.optimizer.step()                            # ③ 参数更新
                    
                    # 收集公共数据表示（用于聚合）
                    with torch.no_grad():
                        if len(public_features.shape) == 3:
                            public_rep = public_features.mean(dim=1).mean(dim=0)  # (D,)
                        else:
                            public_rep = public_features.mean(dim=0)  # (D,)
                        local_public_reps_list.append(public_rep.cpu())
                    
                    # 更新统计
                    total_loss += total_batch_loss.item()
                    total_seg_loss += seg_loss.item()
                    total_cream_loss += cream_loss.item()
                    num_batches += 1
            
            except Exception as e:
                print(f"Error in training loop: {e}")
                break
        
        # 计算平均公共表示
        if len(local_public_reps_list) > 0:
            local_public_reps_stacked = torch.stack(local_public_reps_list, dim=0)
            local_public_rep = local_public_reps_stacked.mean(dim=0)  # (D,)
        else:
            local_public_rep = torch.zeros_like(global_image_rep.cpu())
        
        # 返回更新的模型状态和指标
        metrics = {
            'total_loss': total_loss / max(num_batches, 1),
            'seg_loss': total_seg_loss / max(num_batches, 1),
            'cream_loss': total_cream_loss / max(num_batches, 1),
            'num_batches': num_batches
        }
        
        # 获取过滤后的状态字典（仅包含可训练参数和相关 Buffer）
        trainable_param_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
        full_state = self.model.state_dict()
        filtered_state = {}
        for k, v in full_state.items():
            if k in trainable_param_names or 'adapter' in k.lower() or 'decoder' in k.lower():
                filtered_state[k] = v.clone().cpu()
        
        return filtered_state, local_public_rep, metrics
    
    def get_model(self) -> SAM3MedicalIntegrated:
        """获取当前模型"""
        return self.model
    
    def update_model(self, state_dict: Dict[str, torch.Tensor]):
        """更新模型参数。使用 strict=False 以兼容缺失的冻结编码器权重。"""
        # 加载模型，允许缺失冻结层参数
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        # 验证完整性：确保所有可训练参数都已被正确加载
        trainable_param_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
        missing_trainable = [k for k in missing_keys if k in trainable_param_names]
        
        if missing_trainable:
            raise RuntimeError(f"致命错误: 关键可训练参数在加载时缺失! 请检查权重匹配: {missing_trainable}")
            
        if unexpected_keys:
            print(f"警告: 发现未知的参数键: {unexpected_keys}")
