"""
SAM3 组件加载器
用于加载 SAM3 的 ImageEncoder, PromptEncoder, MaskDecoder 组件
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn

# 添加 sam3 路径
sam3_path = Path(__file__).parent.parent / "core_projects" / "sam3-main"
sys.path.insert(0, str(sam3_path))

try:
    from sam3.model_builder import (
        build_sam3_image_model,
        _create_vit_backbone,
        _create_vision_backbone,
    )
    from sam3.sam.prompt_encoder import PromptEncoder
    from sam3.sam.mask_decoder import MaskDecoder
    from sam3.model.vitdet import ViT
    from sam3.sam.transformer import TwoWayTransformer
    HAS_SAM3 = True
except ImportError as e:
    print(f"警告: 无法导入 SAM3 模块: {e}")
    print("请确保已正确安装 SAM3 依赖")
    HAS_SAM3 = False
    build_sam3_image_model = None
    _create_vit_backbone = None
    _create_vision_backbone = None
    PromptEncoder = None
    MaskDecoder = None
    ViT = None
    TwoWayTransformer = None


class SAM3ImageEncoder:
    """
    SAM3 Image Encoder 包装器
    从 SAM3 模型中提取或创建 Image Encoder (ViT Backbone)
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        compile_mode: Optional[str] = None,
    ):
        """
        初始化 Image Encoder
        
        Args:
            model: 已加载的 SAM3 模型（如果提供，从中提取 encoder）
            checkpoint_path: 检查点路径（如果提供，加载权重）
            device: 设备
            compile_mode: 编译模式（"default" 或 None）
        """
        self.device = device
        self.compile_mode = compile_mode
        
        if model is not None:
            # 从现有模型中提取 encoder
            self.encoder = self._extract_from_model(model)
        else:
            # 创建新的 encoder
            self.encoder = self._create_encoder()
        
        # 加载权重（如果提供）
        if checkpoint_path is not None:
            self.load_weights(checkpoint_path)
        
        # 移动到设备
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
    
    def _extract_from_model(self, model: nn.Module) -> nn.Module:
        """从 SAM3 模型中提取 Image Encoder"""
        # SAM3 模型的 backbone 结构
        # backbone -> visual -> trunk (ViT)
        if hasattr(model, 'backbone'):
            backbone = model.backbone
            if hasattr(backbone, 'visual'):
                visual = backbone.visual
                if hasattr(visual, 'trunk'):
                    return visual.trunk
                return visual
            return backbone
        
        raise ValueError("无法从模型中提取 Image Encoder")
    
    def _create_encoder(self) -> nn.Module:
        """创建新的 Image Encoder"""
        if _create_vit_backbone is None:
            raise ImportError("无法导入 _create_vit_backbone，请确保 SAM3 已正确安装")
        return _create_vit_backbone(compile_mode=self.compile_mode)
    
    def load_weights(self, checkpoint_path: str):
        """从检查点加载权重"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # 提取 encoder 相关的权重
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        
        # 查找 encoder 权重（可能以 "backbone.visual.trunk" 或类似前缀开头）
        encoder_state = {}
        for key, value in checkpoint.items():
            if "backbone" in key or "visual" in key or "trunk" in key:
                # 移除前缀
                new_key = key.replace("backbone.visual.trunk.", "")
                new_key = new_key.replace("backbone.visual.", "")
                new_key = new_key.replace("backbone.", "")
                encoder_state[new_key] = value
        
        if encoder_state:
            missing_keys, unexpected_keys = self.encoder.load_state_dict(
                encoder_state, strict=False
            )
            if missing_keys:
                print(f"警告: 缺少以下权重: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"警告: 意外的权重: {unexpected_keys[:5]}...")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.encoder(images)
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


class SAM3PromptEncoder:
    """
    SAM3 Prompt Encoder 包装器
    用于编码点、框和掩码提示
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,
        device: str = "cuda",
    ):
        """
        初始化 Prompt Encoder
        
        Args:
            embed_dim: 嵌入维度
            image_embedding_size: 图像嵌入的空间尺寸 (H, W)
            input_image_size: 输入图像尺寸 (H, W)
            mask_in_chans: 掩码输入通道数
            device: 设备
        """
        if not HAS_SAM3 or PromptEncoder is None:
            raise ImportError("需要安装 SAM3 才能使用 PromptEncoder")
        
        self.device = device
        self.encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=input_image_size,
            mask_in_chans=mask_in_chans,
        )
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
    
    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码提示
        
        Args:
            points: (坐标, 标签) 元组
            boxes: 边界框
            masks: 掩码
        
        Returns:
            (sparse_embeddings, dense_embeddings) 元组
        """
        return self.encoder(points, boxes, masks)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SAM3MaskDecoder:
    """
    SAM3 Mask Decoder 包装器
    用于从图像嵌入和提示嵌入生成掩码
    """
    
    def __init__(
        self,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        device: str = "cuda",
    ):
        """
        初始化 Mask Decoder
        
        Args:
            transformer_dim: Transformer 维度
            num_multimask_outputs: 多掩码输出数量
            device: 设备
        """
        if not HAS_SAM3 or MaskDecoder is None or TwoWayTransformer is None:
            raise ImportError("需要安装 SAM3 才能使用 MaskDecoder")
        
        self.device = device
        
        # 创建 TwoWayTransformer
        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )
        
        # 创建 MaskDecoder
        self.decoder = MaskDecoder(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
        )
        
        self.decoder = self.decoder.to(device)
        self.decoder.eval()
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成掩码
        
        Args:
            image_embeddings: 图像嵌入 (B, C, H, W)
            image_pe: 图像位置编码 (1, C, H, W)
            sparse_prompt_embeddings: 稀疏提示嵌入 (B, N, C)
            dense_prompt_embeddings: 密集提示嵌入 (B, C, H, W)
            multimask_output: 是否输出多个掩码
        
        Returns:
            (masks, iou_predictions) 元组
        """
        return self.decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def load_sam3_components(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    image_embedding_size: Tuple[int, int] = (64, 64),
    input_image_size: Tuple[int, int] = (1024, 1024),
    load_from_model: bool = False,
    model_path: Optional[str] = None,
) -> Tuple[SAM3ImageEncoder, SAM3PromptEncoder, SAM3MaskDecoder]:
    """
    便捷函数：加载所有 SAM3 组件
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        image_embedding_size: 图像嵌入尺寸
        input_image_size: 输入图像尺寸
        load_from_model: 是否从完整模型加载
        model_path: 模型路径（如果 load_from_model=True）
    
    Returns:
        (image_encoder, prompt_encoder, mask_decoder) 元组
    """
    # 加载完整模型（如果需要）
    model = None
    if load_from_model and model_path:
        model = build_sam3_image_model(
            checkpoint_path=model_path,
            device=device,
            eval_mode=True,
            load_from_HF=False,  # ★ 禁止 HuggingFace 下载，使用本地权重
        )
    
    # 创建 Image Encoder
    image_encoder = SAM3ImageEncoder(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    
    # 创建 Prompt Encoder
    prompt_encoder = SAM3PromptEncoder(
        embed_dim=256,
        image_embedding_size=image_embedding_size,
        input_image_size=input_image_size,
        device=device,
    )
    
    # 创建 Mask Decoder
    mask_decoder = SAM3MaskDecoder(
        transformer_dim=256,
        device=device,
    )
    
    return image_encoder, prompt_encoder, mask_decoder


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("SAM3 组件加载器测试")
    print("=" * 60)
    
    if not HAS_SAM3:
        print("\n错误: SAM3 模块未正确安装")
        print("请确保已正确安装 SAM3 依赖")
        exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 测试创建组件
    print("\n1. 创建 Prompt Encoder...")
    try:
        prompt_encoder = SAM3PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            device=device,
        )
        print("   ✓ Prompt Encoder 创建成功")
    except Exception as e:
        print(f"   ✗ Prompt Encoder 创建失败: {e}")
    
    print("\n2. 创建 Mask Decoder...")
    try:
        mask_decoder = SAM3MaskDecoder(
            transformer_dim=256,
            device=device,
        )
        print("   ✓ Mask Decoder 创建成功")
    except Exception as e:
        print(f"   ✗ Mask Decoder 创建失败: {e}")
    
    print("\n3. 创建 Image Encoder...")
    try:
        image_encoder = SAM3ImageEncoder(device=device)
        print("   ✓ Image Encoder 创建成功")
    except Exception as e:
        print(f"   ✗ Image Encoder 创建失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    print("\n使用示例:")
    print("""
from src.sam3_components_loader import load_sam3_components

# 加载所有组件
image_encoder, prompt_encoder, mask_decoder = load_sam3_components(
    checkpoint_path="path/to/checkpoint.pth",
    device="cuda"
)

# 使用 Image Encoder
images = torch.randn(1, 3, 1024, 1024).to("cuda")
image_embeddings = image_encoder(images)

# 使用 Prompt Encoder
points = (torch.tensor([[[100, 100]]]), torch.tensor([[1]]))
sparse_emb, dense_emb = prompt_encoder(points=points)

# 使用 Mask Decoder
masks, iou = mask_decoder(
    image_embeddings=image_embeddings,
    image_pe=prompt_encoder.encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_emb,
    dense_prompt_embeddings=dense_emb,
)
    """)

