"""
SAM3_Medical: Adapted SAM3 model for medical image segmentation
Only Adapter and MaskDecoder parameters are trainable; ImageEncoder is frozen.
"""

# SAM3_Medical: 适应医学图像分割的SAM3模型
# 只有Adapter和MaskDecoder参数可训练；ImageEncoder参数冻结。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
from pathlib import Path
import numpy as np

# 处理导入路径：支持直接运行和作为模块导入
# 注意: text_fusion.py 已更新，旧版 TextFeatureProjector 不再可用
# 如果需要使用文本融合功能，请改用 integrated_model.py 中的新实现
# try:
#     from src.models.text_fusion import TextFeatureProjector, GatedFusion
# except ImportError:
#     # 如果直接运行此文件，添加项目根目录到路径
#     project_root = Path(__file__).parent.parent
#     if str(project_root) not in sys.path:
#         sys.path.insert(0, str(project_root))
#     from src.models.text_fusion import TextFeatureProjector, GatedFusion


# Global Constants
BATCH_SIZE = 1  # 降低以避免显存不足
LR = 2e-4
IMG_SIZE = 1024
ROUNDS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Adapter(nn.Module):
    """
    Simple residual adapter block for feature adaptation.
    Architecture: Linear(dim, skip) -> ReLU -> Linear(skip, dim)
    """
    
    def __init__(self, dim: int, skip: int = 64):
        super(Adapter, self).__init__()
        self.dim = dim
        self.skip = skip
        
        self.down_proj = nn.Linear(dim, skip)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(skip, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (B, N, dim) or (B, dim)
        Returns:
            Adapted features with residual connection
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual


class MockViTEncoder(nn.Module):
    """
    Mock ViT-based image encoder simulating SAM3's image encoder.
    Since SAM3 code might not be public, we create a ViT-like structure.
    """
    
    def __init__(self, img_size: int = 1024, patch_size: int = 16, embed_dim: int = 768, num_heads: Optional[int] = None):
        super(MockViTEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate or validate num_heads
        if num_heads is None:
            # Default behavior: assume head_dim = 64
            self.num_heads = embed_dim // 64
        else:
            self.num_heads = num_heads
            
        if embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({self.num_heads})")
        
        # Patch embedding
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Transformer blocks (simplified)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=self.num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(12)
        ])
        
        # Layer norm
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, 3, H, W)
        Returns:
            Encoded features of shape (B, N, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln(x)
        return x


class MaskDecoder(nn.Module):
    """
    Decoder for generating segmentation masks from encoded features.

    ★ 2026-03-01 Critical Fix: 添加自适应bias初始化，防止Logits崩塌
    """

    def __init__(
        self,
        embed_dim: int = 768,
        decoder_dim: int = 256,
        num_classes: int = 1,
        foreground_ratio: float = 0.01  # ← 新增：前景像素占比（用于bias初始化）
    ):
        super(MaskDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim

        # Feature projection
        self.proj = nn.Linear(embed_dim, decoder_dim)

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(decoder_dim, decoder_dim // 2, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(decoder_dim // 2, decoder_dim // 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(decoder_dim // 4, decoder_dim // 8, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(decoder_dim // 8, num_classes, kernel_size=2, stride=2)

        # Batch norms
        self.bn1 = nn.BatchNorm2d(decoder_dim // 2)
        self.bn2 = nn.BatchNorm2d(decoder_dim // 4)
        self.bn3 = nn.BatchNorm2d(decoder_dim // 8)

        # Activations
        self.relu = nn.ReLU()

        # ★★★ Critical Fix (2026-03-01): 自适应bias初始化 ★★★
        # 问题：默认bias=0导致模型初始预测全为背景（logits<0）
        # 解决：根据前景/背景比例初始化bias，使初始预测接近数据分布
        # 公式：bias = log(p / (1-p))，其中p是前景像素占比
        # 例如：p=0.01 → bias≈-4.6，p=0.1 → bias≈-2.2
        if self.upsample4.bias is not None:
            # 计算初始bias（logit space）
            # 裁剪foreground_ratio防止log(0)或log(1)
            p = max(0.001, min(0.999, foreground_ratio))
            initial_bias = np.log(p / (1.0 - p))

            # 设置bias
            nn.init.constant_(self.upsample4.bias, initial_bias)

            print(f"[MaskDecoder Init] 自适应bias初始化: foreground_ratio={foreground_ratio:.4f} → bias={initial_bias:.4f}")
        
    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        """
        Args:
            x: Encoded features of shape (B, N, embed_dim)
            spatial_shape: Target spatial shape (H, W) for reshaping
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        B, N, D = x.shape
        H, W = spatial_shape
        
        # Project to decoder dimension
        x = self.proj(x)  # (B, N, decoder_dim)
        
        # Reshape to spatial format
        x = x.transpose(1, 2).reshape(B, self.decoder_dim, H, W)
        
        # Upsampling path
        x = self.relu(self.bn1(self.upsample1(x)))
        x = self.relu(self.bn2(self.upsample2(x)))
        x = self.relu(self.bn3(self.upsample3(x)))
        x = self.upsample4(x)  # (B, num_classes, H*16, W*16)
        
        return x


class SAM3_Medical(nn.Module):
    """
    SAM3_Medical: Adapted SAM3 model for medical image segmentation.
    Only Adapter and MaskDecoder parameters are trainable.
    ImageEncoder parameters are frozen.
    """
    
    def __init__(
        self,
        img_size: int = IMG_SIZE,
        embed_dim: int = 768,
        decoder_dim: int = 256,
        num_classes: int = 1,
        adapter_skip: int = 64,
        use_text_fusion: bool = False,
        text_dim: Optional[int] = None,
        fusion_type: str = "gated",
        num_heads: Optional[int] = None,
        shared_encoder: Optional[nn.Module] = None  # 共享编码器参数
    ):
        super(SAM3_Medical, self).__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes
        self.use_text_fusion = use_text_fusion
        
        # Calculate default num_heads if not provided
        if num_heads is None:
            self.num_heads = embed_dim // 64
        else:
            self.num_heads = num_heads
            
        if embed_dim % self.num_heads != 0:
             raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({self.num_heads})")
        
        # Image encoder (frozen) - 支持共享编码器
        if shared_encoder is not None:
            # 使用外部传入的共享编码器（内存安全策略）
            self.image_encoder = shared_encoder
            self._uses_shared_encoder = True
        else:
            # 创建新的编码器实例
            self.image_encoder = MockViTEncoder(
                img_size=img_size, 
                embed_dim=embed_dim,
                num_heads=self.num_heads
            )
            self._uses_shared_encoder = False
        
        # 梯度截断：确保编码器所有参数的 requires_grad=False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Adapter modules (trainable) - inject into transformer blocks
        self.adapters = nn.ModuleList([
            Adapter(dim=embed_dim, skip=adapter_skip)
            for _ in range(len(self.image_encoder.transformer_blocks))
        ])
        
        # Text fusion modules (trainable, if enabled)
        # 注意: 文本融合功能已迁移到 integrated_model.py
        # 如果需要使用文本融合，请使用 SAM3MedicalIntegrated 类
        if use_text_fusion:
            raise NotImplementedError(
                "SAM3_Medical 类的文本融合功能已废弃。"
                "请使用 src.integrated_model.SAM3MedicalIntegrated 类，"
                "它支持新的 GatedFusion 模块。"
            )
            # if text_dim is None:
            #     raise ValueError("use_text_fusion=True 时需要指定 text_dim")
            # 
            # # 文本特征投影器
            # self.text_projector = TextFeatureProjector(
            #     text_dim=text_dim,
            #     embed_dim=embed_dim,
            #     num_layers=2
            # )
            # 
            # # 门控融合模块
            # self.text_fusion = GatedFusion(
            #     embed_dim=embed_dim,
            #     fusion_type=fusion_type
            # )
        else:
            self.text_projector = None
            self.text_fusion = None
        
        # Mask decoder (trainable)
        self.mask_decoder = MaskDecoder(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            num_classes=num_classes
        )

        # ★ 痛点2修复（2026-03-17）：医学分割专属降维头
        # SAM3 原始 Mask Decoder 输出 num_classes 通道（默认1），存在模态歧义。
        # 此处强制加 1x1 Conv2d 将任意 num_classes 通道降到 1 通道，
        # 完全解耦 iou_scores 通道选择逻辑，确保二分类输出纯净。
        # 即使 num_classes=1，此层也保持网络结构一致性（仅 1→1 线性映射）。
        self.medical_seg_head = nn.Conv2d(
            in_channels=num_classes,
            out_channels=1,
            kernel_size=1,
            bias=True
        )
        # 初始化：恒等映射（weight=1, bias=0），避免引入初始偏移
        nn.init.constant_(self.medical_seg_head.weight, 1.0)
        nn.init.constant_(self.medical_seg_head.bias, 0.0)
        print(f"[SAM3_Medical Init] medical_seg_head: {num_classes}ch → 1ch (1x1 Conv, identity init)")

        # Calculate spatial dimensions after patch embedding
        patch_size = 16
        self.spatial_h = img_size // patch_size
        self.spatial_w = img_size // patch_size
    
    def train(self, mode: bool = True):
        """
        重写 train() 方法以永久锁定编码器为评估模式。
        
        这是内存安全共享编码器策略的关键部分：
        - 即使模型整体进入训练模式，编码器也保持 eval 模式
        - 确保 BatchNorm、Dropout 等层的行为一致
        
        Args:
            mode: 如果为 True，设置模型为训练模式；否则为评估模式
        Returns:
            self
        """
        super().train(mode)
        # 永久锁定 Image Encoder 为评估模式
        if hasattr(self, 'image_encoder'):
            self.image_encoder.eval()
        return self
        
    def _inject_adapters(self, features: torch.Tensor) -> torch.Tensor:
        """
        Mock injection of adapters into transformer features.
        In real implementation, this would be done within transformer blocks.
        
        Args:
            features: Features from transformer blocks (B, N, embed_dim)
        Returns:
            Adapted features
        """
        # Apply adapters sequentially (simulating injection)
        for adapter in self.adapters:
            # Apply adapter to each token
            features = adapter(features)
        return features
    
    def forward(
        self,
        images: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """
        ★ 痛点2修复（2026-03-17）：medical_seg_head 降维，抛弃 iou_scores 通道选择

        Forward pass for segmentation.

        Args:
            images: Input images of shape (B, 3, H, W)
            text_features: Optional text features of shape (B, text_dim) or (B, D)
        Returns:
            logits: 单通道分割 Logits (B, 1, 256, 256)，未经 sigmoid
            None:   iou_predictions 已废弃（固定返回 None），
                    调用方不应再使用 iou_scores 做通道选择。
        """
        # Encode images (frozen encoder)
        with torch.no_grad():
            image_features = self.image_encoder(images)  # (B, N, embed_dim)

        # Apply adapters to image features (trainable)
        image_features = self._inject_adapters(image_features)

        # Text fusion (if enabled and text features provided)
        if self.use_text_fusion and text_features is not None:
            # Project text features to embedding space
            text_features_proj = self.text_projector(text_features)  # (B, embed_dim)

            # Fuse text and image features
            features = self.text_fusion(image_features, text_features_proj)
        else:
            features = image_features

        # Decode to masks (trainable)
        raw_logits = self.mask_decoder(features, spatial_shape=(self.spatial_h, self.spatial_w))
        # raw_logits: (B, num_classes, spatial_h*16, spatial_w*16)

        # ★ 痛点2核心修复：1x1 Conv 强制降维到单通道，彻底消除多通道歧义
        #   若 num_classes=1：医学分割头做恒等线性映射，等价于直接输出
        #   若 num_classes>1：强制聚合多通道为单个二分类输出（背景 vs 肿瘤）
        raw_logits = self.medical_seg_head(raw_logits)  # (B, 1, H', W')

        # ✅ 上采样到标准 256x256 输出尺寸
        logits = F.interpolate(raw_logits, size=(256, 256), mode="bilinear", align_corners=False)
        # logits: (B, 1, 256, 256)，严格对齐单通道二分类

        # ★ iou_predictions 已废弃：不再计算伪IoU，直接返回 None
        # 调用方（client.py / metrics.py）通过 isinstance(raw_output, tuple)
        # 解包时，需要处理 iou_scores=None 的情况（已在 validate() 中兼容）
        return logits, None
    
    def extract_features(
        self,
        images: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract bottleneck embeddings (before decoder) for CreamFL distillation.
        This is used for contrastive learning on public data.
        
        Args:
            images: Input images of shape (B, 3, H, W)
            text_features: Optional text features of shape (B, text_dim) or (B, D)
        Returns:
            Bottleneck embeddings of shape (B, N, embed_dim)
        """
        # Encode images (frozen encoder) - 只对 encoder 使用 no_grad
        with torch.no_grad():
            image_features = self.image_encoder(images)  # (B, N, embed_dim)
        
        # 确保 features 需要梯度（用于 adapter 训练）
        # detach() 断开与 encoder 的计算图，requires_grad_(True) 允许 adapter 的梯度传播
        image_features = image_features.detach().requires_grad_(True)
        
        # Apply adapters to image features (trainable) - these features are used for distillation
        image_features = self._inject_adapters(image_features)
        
        # Text fusion (if enabled and text features provided)
        if self.use_text_fusion and text_features is not None:
            # Project text features to embedding space
            text_features_proj = self.text_projector(text_features)  # (B, embed_dim)
            
            # Fuse text and image features
            features = self.text_fusion(image_features, text_features_proj)
        else:
            features = image_features
        
        # Return bottleneck embeddings (before decoder)
        return features
    
    def get_trainable_params(self) -> list:
        """
        Get list of trainable parameters (Adapter, MaskDecoder, medical_seg_head, and Text Fusion if enabled).
        
        Returns:
            List of trainable parameters
        """
        trainable_params = []
        trainable_params.extend(list(self.adapters.parameters()))
        trainable_params.extend(list(self.mask_decoder.parameters()))
        # ★ 痛点2修复：medical_seg_head 参与训练
        trainable_params.extend(list(self.medical_seg_head.parameters()))
        
        # Add text fusion parameters if enabled
        if self.use_text_fusion:
            trainable_params.extend(list(self.text_projector.parameters()))
            trainable_params.extend(list(self.text_fusion.parameters()))
        
        return trainable_params


if __name__ == "__main__":
    # Test the model
    model = SAM3_Medical(img_size=IMG_SIZE).to(DEVICE)
    
    # Create dummy input
    dummy_images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Test forward pass
    masks, iou_preds = model(dummy_images)
    print(f"Forward masks shape: {masks.shape}")       # Should be (B, 1, 256, 256)
    print(f"Forward iou_preds shape: {iou_preds.shape}")  # Should be (B, 1)
    
    # Test feature extraction
    features = model.extract_features(dummy_images)
    print(f"Feature extraction output shape: {features.shape}")  # Should be (B, N, embed_dim)
    
    # Check trainable parameters
    trainable_params = model.get_trainable_params()
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total_trainable:,} / {total_params:,}")
    
    # Verify encoder is frozen
    encoder_params = sum(p.numel() for p in model.image_encoder.parameters())
    print(f"Frozen encoder parameters: {encoder_params:,}")
    print(f"All encoder params require_grad=False: {all(not p.requires_grad for p in model.image_encoder.parameters())}")

