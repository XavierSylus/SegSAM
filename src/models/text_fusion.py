"""
文本-图像空间特征融合模块
实现门控机制（Gated Fusion）用于融合文本特征和图像空间特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    门控融合模块（Gated Fusion）
    使用门控机制自适应地融合文本和图像空间特征
    
    输入：
        - image_features: (B, C, H, W) 图像特征
        - text_features: (B, D) 文本特征
    
    输出：
        - fused_features: (B, C, H, W) 融合后的特征
    """
    
    def __init__(
        self,
        image_channels: int,
        text_dim: int,
        hidden_dim: int = 256
    ):
        """
        Args:
            image_channels: 图像特征通道数 C
            text_dim: 文本特征维度 D
            hidden_dim: 门控网络隐藏层维度（默认: 256）
        """
        super(GatedFusion, self).__init__()
        self.image_channels = image_channels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # 文本特征投影器：将 text_features (B, D) 投影到 (B, C)
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, image_channels),
        )
        
        # 门控网络：根据文本和图像特征计算门控权重
        # 输入：拼接的文本投影和图像全局特征 (B, C + C) = (B, 2C)
        # 输出：门控值 (B, C)
        self.gate_net = nn.Sequential(
            nn.Linear(image_channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, image_channels),
            nn.Sigmoid()  # 输出 [0, 1] 之间的门控值
        )
        
        # 特征变换层：对文本和图像特征分别变换
        self.text_transform = nn.Sequential(
            nn.Linear(image_channels, image_channels),
            nn.LayerNorm(image_channels)
        )
        
        self.image_transform = nn.Sequential(
            nn.Conv2d(image_channels, image_channels, kernel_size=1),
            nn.BatchNorm2d(image_channels)
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        融合图像和文本特征
        
        Args:
            image_features: 图像特征 (B, C, H, W)
            text_features: 文本特征 (B, D)
        
        Returns:
            融合后的特征 (B, C, H, W)
        """
        B, C, H, W = image_features.shape
        
        # 步骤 1: 将文本特征投影到与图像通道相同的维度
        # text_features: (B, D) -> (B, C)
        text_projected = self.text_projector(text_features)  # (B, C)
        
        # 步骤 2: 计算图像特征的全局表示（Global Average Pooling）
        # image_features: (B, C, H, W) -> (B, C)
        image_global = F.adaptive_avg_pool2d(image_features, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
        
        # 步骤 3: 拼接文本投影和图像全局特征，计算门控值
        # concat: (B, 2C)
        concat_features = torch.cat([text_projected, image_global], dim=1)  # (B, 2C)
        gate = self.gate_net(concat_features)  # (B, C)
        
        # 步骤 4: 扩展门控值和文本特征到空间维度
        # gate: (B, C) -> (B, C, 1, 1) -> (B, C, H, W)
        gate_spatial = gate.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)  # (B, C, H, W)
        
        # text_projected: (B, C) -> (B, C, 1, 1) -> (B, C, H, W)
        text_spatial = text_projected.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)  # (B, C, H, W)
        
        # 步骤 5: 变换特征
        text_transformed = self.text_transform(text_spatial.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, C, H, W)
        image_transformed = self.image_transform(image_features)  # (B, C, H, W)
        
        # 步骤 6: 使用门控机制融合特征
        # 输出 = gate * text + (1 - gate) * image
        fused_features = gate_spatial * text_transformed + (1 - gate_spatial) * image_transformed
        
        return fused_features


if __name__ == '__main__':
    """
    测试 GatedFusion 模块的输入输出形状
    """
    print("=" * 60)
    print("测试 GatedFusion 模块")
    print("=" * 60)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 定义测试参数
    batch_size = 4
    image_channels = 256
    height = 32
    width = 32
    text_dim = 512
    
    print(f"\n测试参数:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Channels (C): {image_channels}")
    print(f"  Image Size (H, W): ({height}, {width})")
    print(f"  Text Dimension (D): {text_dim}")
    
    # 创建虚拟输入
    image_features = torch.randn(batch_size, image_channels, height, width).to(device)
    text_features = torch.randn(batch_size, text_dim).to(device)
    
    print(f"\n输入形状:")
    print(f"  image_features: {image_features.shape}  # (B, C, H, W)")
    print(f"  text_features:  {text_features.shape}      # (B, D)")
    
    # 创建 GatedFusion 模块
    fusion_module = GatedFusion(
        image_channels=image_channels,
        text_dim=text_dim,
        hidden_dim=256
    ).to(device)
    
    print(f"\n模型参数:")
    total_params = sum(p.numel() for p in fusion_module.parameters())
    trainable_params = sum(p.numel() for p in fusion_module.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 前向传播
    print(f"\n执行前向传播...")
    with torch.no_grad():
        fused_features = fusion_module(image_features, text_features)
    
    print(f"\n输出形状:")
    print(f"  fused_features: {fused_features.shape}  # (B, C, H, W)")
    
    # 验证输出形状
    expected_shape = (batch_size, image_channels, height, width)
    assert fused_features.shape == expected_shape, \
        f"输出形状不匹配！期望 {expected_shape}，得到 {fused_features.shape}"
    
    print(f"\n✅ 形状验证通过！")
    
    # 测试梯度反向传播
    print(f"\n测试梯度反向传播...")
    fusion_module.train()
    image_features.requires_grad = True
    text_features.requires_grad = True
    
    fused_features = fusion_module(image_features, text_features)
    loss = fused_features.mean()
    loss.backward()
    
    print(f"  损失值: {loss.item():.6f}")
    print(f"  image_features.grad 形状: {image_features.grad.shape}")
    print(f"  text_features.grad 形状: {text_features.grad.shape}")
    
    # 检查梯度是否存在
    has_grad_image = image_features.grad is not None and not torch.isnan(image_features.grad).any()
    has_grad_text = text_features.grad is not None and not torch.isnan(text_features.grad).any()
    
    if has_grad_image and has_grad_text:
        print(f"\n✅ 梯度反向传播验证通过！")
    else:
        print(f"\n❌ 梯度反向传播失败！")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
