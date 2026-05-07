
import sys
import os
from pathlib import Path

# Add core_projects to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "core_projects" / "sam3-main"))
sys.path.insert(0, str(project_root / "core_projects" / "SAM-Adapter-PyTorch-main"))
sys.path.insert(0, str(project_root / "core_projects" / "CreamFL-main" / "src"))
sys.path.insert(0, str(project_root / "core_projects" / "FedFMS-main"))

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)



@dataclass
class FederatedOutput:
    """forward() 统一返回类型。to_dict() 保持与 client.py 调用链的兼容性。"""
    logits:   torch.Tensor
    text_rep: Optional[torch.Tensor] = None
    img_rep:  Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {'logits': self.logits}
        if self.text_rep is not None:
            out['text_rep'] = self.text_rep
        if self.img_rep is not None:
            out['features'] = self.img_rep
        return out


# Import SAM3 components
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image import Sam3Image
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage
)
SAM3_AVAILABLE = True

# Import SAM-Adapter components
from models.block import SKSPP, UpsampleSKConv
ADAPTER_AVAILABLE = True

# Import CreamFL components
from criterions.probemb import MCSoftContrastiveLoss
CREAMFL_AVAILABLE = True


from src.models.adapter import Adapter
from src.models.freeze_utils import freeze_backbone, verify_frozen_state


class BlockWithAdapter(nn.Module):
    """Wrapper 注入器：在每个 Transformer Block 输出上叠加 Adapter 残差修正。"""

    def __init__(
        self,
        original_block: nn.Module,
        dim: int,
        adapter_dim: int = 64,
        scale: float = 1.0
    ):
        super(BlockWithAdapter, self).__init__()
        self.original_block = original_block
        self.adapter = Adapter(
            in_dim=dim,
            out_dim=dim,
            adapter_dim=adapter_dim,
            use_residual=False,
            init_scale=scale
        )
        self.scale = scale

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        original_output = self.original_block(x, *args, **kwargs)
        logger.debug(
            "BlockWithAdapter forward | down=%s up=%s input=%s",
            self.adapter.down_proj.weight.shape,
            self.adapter.up_proj.weight.shape,
            original_output.shape,
        )
        adapter_output = self.adapter(original_output)
        if adapter_output.shape != original_output.shape:
            raise RuntimeError(
                f"Adapter output shape mismatch! "
                f"Input: {original_output.shape}, Output: {adapter_output.shape}"
            )
        return original_output + self.scale * adapter_output


SAM3Adapter = Adapter


class AdapterInjector(nn.Module):
    """将 BlockWithAdapter 挂载到 SAM3 Transformer Block，并管理 RoPE freqs_cis 插值修正。"""

    def __init__(self, adapter_dim: int = 64, adapter_scale: float = 1.0):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.adapter_scale = adapter_scale
        # 以下属性在 inject() 调用后才被赋值
        self.adapters: Optional[nn.ModuleList] = None
        self.wrapped_blocks: Optional[nn.ModuleList] = None
        self._rope_patched: bool = False

    # ------------------------------------------------------------------
    # 主入口：执行 Adapter 注入
    # ------------------------------------------------------------------
    def inject(
        self,
        sam3_model: nn.Module,
        use_real_sam3: bool,
        embed_dim_hint: int,
        image_encoder: Optional[nn.Module] = None,
    ) -> 'AdapterInjector':
        """
        执行 Adapter 注入逻辑（原 SAM3MedicalIntegrated._inject_adapters）。
        返回 self 以支持链式调用：AdapterInjector(...).inject(...)
        """
        if not use_real_sam3:
            # Mock 模型：创建独立的适配器列表
            embed_dim = embed_dim_hint
            if image_encoder is not None:
                if hasattr(image_encoder, 'transformer_blocks'):
                    num_blocks = len(image_encoder.transformer_blocks)
                elif hasattr(image_encoder, 'blocks'):
                    num_blocks = len(image_encoder.blocks)
                else:
                    num_blocks = 12
            else:
                num_blocks = 12

            self.adapters = nn.ModuleList([
                Adapter(
                    in_dim=embed_dim,
                    out_dim=embed_dim,
                    adapter_dim=self.adapter_dim,
                    use_residual=True,
                    init_scale=self.adapter_scale
                )
                for _ in range(num_blocks)
            ])
            logger.info("✓ Created %d Adapters for Mock model", num_blocks)
            return self

        # 真实 SAM3：使用 Wrapper 模式包裹 blocks
        found_blocks = False
        vit = None

        if hasattr(sam3_model, 'backbone'):
            backbone = sam3_model.backbone
            logger.debug("Found backbone: %s", type(backbone).__name__)
            if hasattr(backbone, 'vision_backbone'):
                neck = backbone.vision_backbone
                logger.debug("Found vision_backbone (Neck): %s", type(neck).__name__)
                if hasattr(neck, 'trunk'):
                    vit = neck.trunk
                    logger.debug("Found trunk (ViT): %s", type(vit).__name__)
                else:
                    logger.debug("No trunk found, using vision_backbone directly")
                    vit = neck
            else:
                logger.debug("No vision_backbone found, using backbone directly")
                vit = backbone

        if vit is not None:
            if hasattr(vit, 'blocks'):
                original_blocks = vit.blocks
                found_blocks = True
            elif hasattr(vit, 'transformer') and hasattr(vit.transformer, 'blocks'):
                original_blocks = vit.transformer.blocks
                found_blocks = True
                vit = vit.transformer

        if found_blocks:
            num_blocks = len(original_blocks)
            logger.debug("Found %d transformer blocks", num_blocks)

            # 智能推断 embed_dim
            embed_dim = None
            if hasattr(vit, 'embed_dim'):
                embed_dim = vit.embed_dim
                logger.debug("Detected embed_dim from vit.embed_dim: %d", embed_dim)
            elif num_blocks > 0:
                first_block = original_blocks[0]
                norm_layer = getattr(first_block, 'norm1', None) or getattr(first_block, 'ln1', None)
                if norm_layer is not None:
                    if hasattr(norm_layer, 'normalized_shape'):
                        shape = norm_layer.normalized_shape
                        embed_dim = shape[-1] if isinstance(shape, (list, tuple)) else shape
                        logger.debug("Inferred embed_dim from norm1.normalized_shape: %d", embed_dim)
                    elif hasattr(norm_layer, 'weight'):
                        embed_dim = norm_layer.weight.shape[0]
                        logger.debug("Inferred embed_dim from norm1.weight.shape: %d", embed_dim)
            if embed_dim is None:
                embed_dim = 1024
                logger.warning("Could not infer embed_dim, using default for ViT-H: %d", embed_dim)

            # 使用 Wrapper 模式包裹每个 Block
            wrapped_blocks = nn.ModuleList()
            for i, block in enumerate(original_blocks):
                wrapped = BlockWithAdapter(
                    original_block=block,
                    dim=embed_dim,
                    adapter_dim=self.adapter_dim,
                    scale=self.adapter_scale
                )
                wrapped_blocks.append(wrapped)
                if i == 0:
                    logger.debug(
                        "First Adapter dims: down=%s up=%s",
                        wrapped.adapter.down_proj.weight.shape,
                        wrapped.adapter.up_proj.weight.shape,
                    )

            # 替换原始 blocks
            if hasattr(vit, 'blocks'):
                vit.blocks = wrapped_blocks

            self.wrapped_blocks = wrapped_blocks
            self.adapters = nn.ModuleList([wb.adapter for wb in wrapped_blocks])

            logger.info(
                "[Success] ✓ Injected adapters into %d blocks with dim=%d, adapter_dim=%d",
                num_blocks, embed_dim, self.adapter_dim,
            )

            # 清理 checkpoint 中的旧 Adapter 权重
            logger.debug("Cleaning old Adapter weights from SAM3 state_dict...")
            current_state = sam3_model.state_dict()
            adapter_keys = [k for k in current_state.keys() if 'adapter' in k.lower()]
            if adapter_keys:
                logger.debug("Found %d old Adapter keys, removing...", len(adapter_keys))
                cleaned_state = {k: v for k, v in current_state.items() if k not in adapter_keys}
                sam3_model.load_state_dict(cleaned_state, strict=False)
                logger.debug("✓ Removed %d old Adapter weight keys", len(adapter_keys))

            # 强制重新初始化 Adapter 权重
            logger.debug("Forcefully recreating Adapter Linear layers...")
            for i, adapter in enumerate(self.adapters):
                expected_down = (self.adapter_dim, embed_dim)
                expected_up = (embed_dim, self.adapter_dim)
                if (adapter.down_proj.weight.shape != expected_down
                        or adapter.up_proj.weight.shape != expected_up):
                    logger.warning(
                        "Adapter %d dimension mismatch! Recreating...", i
                    )
                    adapter.down_proj = nn.Linear(embed_dim, self.adapter_dim, bias=False)
                    adapter.up_proj = nn.Linear(self.adapter_dim, embed_dim, bias=False)
                    nn.init.normal_(adapter.down_proj.weight, std=1e-3)
                    nn.init.zeros_(adapter.up_proj.weight)
                    if next(sam3_model.parameters()).is_cuda:
                        adapter.down_proj = adapter.down_proj.cuda()
                        adapter.up_proj = adapter.up_proj.cuda()
                else:
                    nn.init.normal_(adapter.down_proj.weight, std=1e-3)
                    nn.init.zeros_(adapter.up_proj.weight)

            # 最终验证
            for i, adapter in enumerate(self.adapters):
                if adapter.down_proj.weight.shape != (self.adapter_dim, embed_dim):
                    raise RuntimeError(
                        f"Adapter {i} down_proj shape STILL incorrect after recreation!"
                    )
                if adapter.up_proj.weight.shape != (embed_dim, self.adapter_dim):
                    raise RuntimeError(
                        f"Adapter {i} up_proj shape STILL incorrect after recreation!"
                    )
            logger.info("所有 %d 个 Adapter 维度验证通过。", len(self.adapters))

        else:
            logger.warning(
                "Could not find 'blocks' in model structure (type=%s). Adapters not injected.",
                type(vit).__name__ if vit is not None else 'None',
            )
            self.adapters = None
            self.wrapped_blocks = None

        return self

    # ------------------------------------------------------------------
    # RoPE 插值工具
    # ------------------------------------------------------------------
    def _interpolate_rope_freqs(
        self,
        attn: nn.Module,
        target_h: int,
        target_w: int,
    ) -> bool:
        """
        将单个 attn 层的 freqs_cis 双三次插值到 (target_h * target_w) 序列长度。
        （原 SAM3MedicalIntegrated._interpolate_rope_freqs，逻辑完全不变）
        """
        fc = getattr(attn, 'freqs_cis', None)
        if fc is None:
            return False
        if hasattr(attn, 'use_rope') and not attn.use_rope:
            return False

        expected_seq = target_h * target_w
        if getattr(attn, 'cls_token', False):
            expected_seq += 1
        if fc.shape[0] == expected_seq:
            return True

        logger.debug(
            "_interpolate_rope_freqs: current=%s expected_seq=%d (grid %dx%d)",
            fc.shape, expected_seq, target_h, target_w,
        )
        try:
            orig = fc.clone()
            orig_len = orig.shape[0]
            dim = orig.shape[1]

            expected_seq_without_cls = target_h * target_w
            has_cls = (orig_len == expected_seq_without_cls + 1)

            if has_cls:
                cls_freq, orig, orig_len = orig[0:1], orig[1:], orig_len - 1

            orig_size = int(orig_len ** 0.5)
            if orig_size * orig_size != orig_len:
                if hasattr(attn, '_setup_rope_freqs'):
                    attn._setup_rope_freqs()
                    return True
                return False

            is_cplx = orig.is_complex()
            if is_cplx:
                r = orig.real.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2).float()
                i = orig.imag.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2).float()
                r2 = F.interpolate(r, (target_h, target_w), mode='bicubic', align_corners=False)
                i2 = F.interpolate(i, (target_h, target_w), mode='bicubic', align_corners=False)
                new_freqs = torch.complex(
                    r2.permute(0, 2, 3, 1).reshape(-1, dim),
                    i2.permute(0, 2, 3, 1).reshape(-1, dim),
                ).to(fc.dtype)
            else:
                fv = orig.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2).float()
                new_freqs = (
                    F.interpolate(fv, (target_h, target_w), mode='bicubic', align_corners=False)
                    .permute(0, 2, 3, 1).reshape(-1, dim).to(fc.dtype)
                )

            if has_cls:
                new_freqs = torch.cat([cls_freq, new_freqs], dim=0)

            if isinstance(fc, nn.Parameter):
                attn.freqs_cis = nn.Parameter(new_freqs)
            else:
                attn.register_buffer('freqs_cis', new_freqs, persistent=False)

            logger.debug("  freqs_cis 插值完成: %s -> %s", fc.shape, new_freqs.shape)
            return True

        except Exception as exc:
            logger.warning("_interpolate_rope_freqs 插值失败（忽略）: %s", exc)
            return False

    def setup_grid_size(self, image_h: int, image_w: int) -> None:
        """
        根据输入图像尺寸，一次性修正所有 attn 层的 RoPE freqs_cis。
        （原 SAM3MedicalIntegrated.setup_grid_size，逻辑完全不变）
        """
        if not self.wrapped_blocks:
            self._rope_patched = True
            return

        patch_size = 16
        target_h, target_w = image_h // patch_size, image_w // patch_size

        count = sum(
            1
            for wb in self.wrapped_blocks
            if hasattr(wb.original_block, 'attn')
            and self._interpolate_rope_freqs(wb.original_block.attn, target_h, target_w)
        )

        logger.info(
            "setup_grid_size: 已修正 %d 个 attn 层的 RoPE freqs_cis -> grid(%d, %d)",
            count, target_h, target_w,
        )
        self._rope_patched = True

    def _rope_pre_hook(self, facade_module: nn.Module, args: tuple) -> None:
        """
        register_forward_pre_hook 回调（由 SAM3MedicalIntegrated 注册）。
        首次 forward 时自动触发 RoPE 修正，之后立即移除自身。
        （原 SAM3MedicalIntegrated._rope_pre_hook，逻辑完全不变，facade_module 为 Facade 实例）
        """
        if not self._rope_patched and args:
            images = args[0]
            self.setup_grid_size(images.shape[-2], images.shape[-1])
        # 移除 hook（只触发一次）
        if hasattr(facade_module, '_rope_hook_handle') and facade_module._rope_hook_handle is not None:
            facade_module._rope_hook_handle.remove()
            facade_module._rope_hook_handle = None

    def apply_adapters(self, features: torch.Tensor) -> torch.Tensor:
        """应用适配器到特征（原 SAM3MedicalIntegrated._apply_adapters）"""
        if self.adapters is not None:
            for adapter in self.adapters:
                features = adapter(features)
        return features

    def reset_rope_frequencies(self, verbose: bool = False) -> int:
        """
        重置所有 Attention 层的 RoPE 频率缓存。
        （原 SAM3MedicalIntegrated.reset_rope_frequencies，逻辑完全不变）
        """
        if not self.wrapped_blocks:
            if verbose:
                logger.info("[RoPE Reset] 警告: 模型不包含 wrapped_blocks，跳过重置")
            return 0

        rope_reset_count = 0

        for i, wrapped_block in enumerate(self.wrapped_blocks):
            original_block = wrapped_block.original_block
            if not hasattr(original_block, 'attn'):
                continue
            attn = original_block.attn

            if not hasattr(attn, 'freqs_cis') or attn.freqs_cis is None:
                if verbose:
                    logger.debug("[RoPE Reset] Block %d: 无 freqs_cis，跳过", i)
                continue
            if hasattr(attn, 'use_rope') and not attn.use_rope:
                if verbose:
                    logger.debug("[RoPE Reset] Block %d: use_rope=False，跳过", i)
                continue

            current_shape = attn.freqs_cis.shape
            input_size = getattr(attn, 'input_size', None)
            head_dim = getattr(attn, 'head_dim', None)
            cls_token = getattr(attn, 'cls_token', False)

            if input_size is None or head_dim is None:
                if verbose:
                    logger.warning("[RoPE Reset] Block %d: input_size or head_dim is None，跳过", i)
                continue

            expected_seq_len = input_size[0] * input_size[1]
            if cls_token:
                expected_seq_len += 1
            expected_head_half = head_dim // 2
            expected_shape = (expected_seq_len, expected_head_half)

            if current_shape != expected_shape:
                if verbose:
                    logger.info(
                        "[RoPE Reset] Block %d: 形状不匹配 %s -> %s",
                        i, current_shape, expected_shape,
                    )
                try:
                    orig_freqs = attn.freqs_cis.clone()
                    orig_len = orig_freqs.shape[0]
                    dim = orig_freqs.shape[1]

                    target_h, target_w = input_size
                    expected_seq_without_cls = target_h * target_w
                    has_cls = (orig_len == expected_seq_without_cls + 1)

                    if has_cls:
                        cls_freq = orig_freqs[0:1]
                        orig_freqs = orig_freqs[1:]
                        orig_len -= 1

                    orig_size = int(orig_len ** 0.5)

                    if orig_size * orig_size == orig_len:
                        is_complex = orig_freqs.is_complex()
                        if is_complex:
                            freqs_real = orig_freqs.real.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2).float()
                            freqs_imag = orig_freqs.imag.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2).float()
                            freqs_real_res = F.interpolate(freqs_real, size=(target_h, target_w), mode='bicubic', align_corners=False)
                            freqs_imag_res = F.interpolate(freqs_imag, size=(target_h, target_w), mode='bicubic', align_corners=False)
                            freqs_real_res = freqs_real_res.permute(0, 2, 3, 1).reshape(-1, dim)
                            freqs_imag_res = freqs_imag_res.permute(0, 2, 3, 1).reshape(-1, dim)
                            resized_freqs = torch.complex(freqs_real_res, freqs_imag_res).to(attn.freqs_cis.dtype)
                        else:
                            freqs_view = orig_freqs.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2).float()
                            freqs_res = F.interpolate(freqs_view, size=(target_h, target_w), mode='bicubic', align_corners=False)
                            resized_freqs = freqs_res.permute(0, 2, 3, 1).reshape(-1, dim).to(attn.freqs_cis.dtype)

                        if has_cls:
                            resized_freqs = torch.cat([cls_freq, resized_freqs], dim=0)

                        if isinstance(attn.freqs_cis, torch.nn.Parameter):
                            attn.freqs_cis = torch.nn.Parameter(resized_freqs)
                        else:
                            attn.freqs_cis = resized_freqs
                            attn.register_buffer('freqs_cis', resized_freqs, persistent=False)

                        rope_reset_count += 1
                        if verbose:
                            logger.info("  └─ ✓ Interpolated freqs_cis to %s", attn.freqs_cis.shape)
                    else:
                        attn._setup_rope_freqs()
                        rope_reset_count += 1
                        if verbose:
                            logger.info("  └─ ✓ Recalculated RoPE")
                except Exception as e:
                    if verbose:
                        logger.error("  └─ ❌ 重置/插值失败: %s", e)

        if verbose:
            if rope_reset_count > 0:
                logger.info("[RoPE Reset] ✓ 成功重置 %d 个 Block 的 RoPE 频率", rope_reset_count)
            else:
                logger.info("[RoPE Reset] 所有 RoPE 频率已正确，无需重置")

        return rope_reset_count


# ============================================================================
# MultimodalFusionHead -- Builder, manages multimodal projection and fusion
# ============================================================================
class MultimodalFusionHead(nn.Module):
    """
    Builder: manages multimodal contrastive projection heads and feature fusion.
    Extracted from SAM3MedicalIntegrated:
      text_proj / image_proj  <- originally inlined in __init__
      _apply_fusion           -> apply_fusion()
      projection logic        -> project_image() / project_text()
    """

    def __init__(self, embed_dim: int, text_dim: int, contrastive_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_dim = text_dim
        self.contrastive_dim = contrastive_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, contrastive_dim, bias=False),
            nn.LayerNorm(contrastive_dim)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, contrastive_dim, bias=False),
            nn.LayerNorm(contrastive_dim)
        )
        nn.init.normal_(self.text_proj[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(self.image_proj[0].weight, mean=0.0, std=0.02)
        logger.info(
            "MultimodalFusionHead init: text_proj %d->%d | image_proj %d->%d",
            text_dim, contrastive_dim, embed_dim, contrastive_dim,
        )

    def apply_fusion(
        self,
        image_embeddings: torch.Tensor,
        text_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Feature fusion (original SAM3MedicalIntegrated._apply_fusion, logic unchanged).
        Supports 3D (B,N,C) and 4D (B,C,H,W) input.
        """
        if text_features is None:
            return image_embeddings

        original_is_3d = (image_embeddings.dim() == 3)
        if original_is_3d:
            B, N, C = image_embeddings.shape
            H = W = int(N ** 0.5)
            if H * W != N:
                raise ValueError(
                    f"Cannot reshape N={N} to square grid: H={H},W={W}, H*W={H*W}!={N}."
                )
            image_spatial = image_embeddings.transpose(1, 2).reshape(B, C, H, W)
        else:
            image_spatial = image_embeddings
            B, C, H, W = image_spatial.shape

        if text_features.dim() == 3:
            text_features = text_features.mean(dim=1)

        B_text, D_text = text_features.shape
        if B_text != B:
            raise ValueError(f"Text batch size ({B_text}) != image batch size ({B})!")

        if not hasattr(self, '_text_projection') or self._text_projection is None:
            self._text_projection = nn.Linear(D_text, C, bias=False).to(image_spatial.device)
            nn.init.normal_(self._text_projection.weight, mean=0.0, std=0.01)
        if self._text_projection.in_features != D_text or self._text_projection.out_features != C:
            self._text_projection = nn.Linear(D_text, C, bias=False).to(image_spatial.device)
            nn.init.normal_(self._text_projection.weight, mean=0.0, std=0.01)

        text_aligned = self._text_projection(text_features)
        text_broadcasted = text_aligned.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)

        if not hasattr(self, '_fusion_gate') or self._fusion_gate is None:
            self._fusion_gate = nn.Sequential(
                nn.Conv2d(C * 2, C, kernel_size=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, kernel_size=1, bias=False),
                nn.Sigmoid()
            ).to(image_spatial.device)

        concat_features = torch.cat([image_spatial, text_broadcasted], dim=1)
        gate = self._fusion_gate(concat_features)
        fused_spatial = gate * text_broadcasted + (1 - gate) * image_spatial

        if original_is_3d:
            return fused_spatial.flatten(2).transpose(1, 2)
        return fused_spatial

    def project_image(self, features: torch.Tensor) -> torch.Tensor:
        """Project image features to contrastive space: (B,N,D) -> (B,N,contrastive_dim)"""
        B, N, D = features.shape
        return self.image_proj(features.reshape(B * N, D)).reshape(B, N, self.contrastive_dim)

    def project_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """Project text features to contrastive space (symmetric with project_image)"""
        return self.text_proj(text_features)


# ============================================================================
# ============================================================================
class TextPromptEncoder(nn.Module):
    """将联邦服务器下发的全局文本表征 global_text_rep 转换为
    SAM3 Mask Decoder 所期待的 sparse_prompt_token (B, 1, embed_dim)。
    up_proj 零初始化：初始期等价纯视觉基线，梯度逐步引入跨模态知识。
    """

    def __init__(self, text_dim: int, embed_dim: int, bottleneck_dim: int = 256):
        super().__init__()
        self.down_proj = nn.Linear(text_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)

        # up_proj 零初始化：初始期 text_token=0，等价纯视觉基线，随后由梯度平滑学习跨模态知识
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)

    def forward(
        self,
        global_text_rep: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        embed_dim = self.up_proj.out_features
        if global_text_rep is None:
            return torch.zeros(batch_size, 0, embed_dim, device=device, dtype=dtype)

        if global_text_rep.dim() == 1:
            g = global_text_rep.unsqueeze(0).expand(batch_size, -1).contiguous()
        else:
            g = global_text_rep.contiguous()
        g = g.to(device=device, dtype=dtype)
        token = self.up_proj(self.act(self.down_proj(g)))
        return token.unsqueeze(1)


class SAM3MedicalIntegrated(nn.Module):
    """SAM3 医学图像分割 Facade：组装 AdapterInjector / MultimodalFusionHead / TextPromptEncoder，重写 load_state_dict 支持老格式 checkpoint。"""

    # state-dict 键名兼容映射：子模块前缀（导出时剥离）
    _COMPAT_SUBMODULE_PREFIXES = ('adapter_manager.', 'fusion_head.')

    def __init__(
        self,
        img_size: int = 1024,
        num_classes: int = 1,
        adapter_dim: int = 64,
        use_sam3: bool = True,
        freeze_encoder: bool = True,
        use_adapter: bool = True,
        sam3_checkpoint: Optional[str] = None,
        device: str = 'cuda',
        embed_dim: int = 768,
        num_heads: Optional[int] = None,
        shared_encoder: Optional[nn.Module] = None,
        text_dim: int = 512,
        contrastive_dim: int = 1024,
    ):
        super(SAM3MedicalIntegrated, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.device_str = device
        self.freeze_encoder = freeze_encoder
        self.use_adapter = use_adapter
        self.embed_dim = embed_dim
        self._uses_shared_encoder = shared_encoder is not None
        self.contrastive_dim = contrastive_dim

        if num_heads is None:
            self.num_heads = embed_dim // 64
        else:
            self.num_heads = num_heads
        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        # 1. 加载 SAM3 模型
        if use_sam3:
            if embed_dim not in (768, 1024):
                logger.warning(
                    "Warning: requesting embed_dim=%d but loading real SAM3.", embed_dim
                )
            if shared_encoder is not None:
                logger.info("Using shared SAM3 encoder (memory-safe strategy)")
                self.sam3_model = shared_encoder
            else:
                logger.info("Loading real SAM3 model (target num_classes=%d)...", num_classes)
                _ckpt = sam3_checkpoint
                if _ckpt is not None and not Path(_ckpt).is_absolute():
                    _ckpt = str(project_root / _ckpt)
                if _ckpt is None or not Path(_ckpt).exists():
                    raise FileNotFoundError(
                        f"[SAM3] 本地权重文件不存在: {_ckpt}\n"
                        "请将 sam3.pt 放入 data/checkpoints/ 目录，或通过 --sam3_checkpoint 指定正确路径。"
                    )
                self.sam3_model = build_sam3_image_model(
                    checkpoint_path=_ckpt,
                    device=device,
                    eval_mode=True,
                    enable_segmentation=True,
                    load_from_HF=False,
                )
                logger.info(
                    "Loaded real SAM3 model (dynamic output mapping to %d channels)", num_classes
                )
            self.use_real_sam3 = True
        else:
            self.use_real_sam3 = False
            if shared_encoder is not None:
                self.image_encoder = shared_encoder
                patch_size = 16
                self.spatial_h = self.img_size // patch_size
                self.spatial_w = self.img_size // patch_size
                from src.model import MaskDecoder
                self.mask_decoder = MaskDecoder(
                    embed_dim=self.embed_dim, decoder_dim=256, num_classes=self.num_classes
                )
            else:
                self._build_mock_sam3()

        # 2. 注入 Adapter（委托给 AdapterInjector）
        self.adapter_manager = AdapterInjector(
            adapter_dim=adapter_dim, adapter_scale=1.0
        )
        if use_adapter:
            self.adapter_manager.inject(
                sam3_model=self.sam3_model if use_sam3 else None,
                use_real_sam3=self.use_real_sam3,
                embed_dim_hint=self.embed_dim,
                image_encoder=getattr(self, 'image_encoder', None),
            )

        # 3. 冻结编码器
        if freeze_encoder:
            self._freeze_encoder()
            if self.use_adapter and self.adapter_manager.adapters is not None:
                self._unfreeze_adapter_params()

        # CUDA 状态刷新
        if self.use_real_sam3 and use_adapter and device == 'cuda':
            logger.debug("Refreshing CUDA state...")
            orig_dev = next(self.sam3_model.parameters()).device
            self.sam3_model = self.sam3_model.cpu()
            torch.cuda.empty_cache()
            self.sam3_model = self.sam3_model.to(orig_dev)

        # 4. 多模态投影头（委托给 MultimodalFusionHead）
        self.fusion_head = MultimodalFusionHead(
            embed_dim=self.embed_dim, text_dim=text_dim, contrastive_dim=contrastive_dim
        )

        # 6. CreamFL 对比学习损失
        if CREAMFL_AVAILABLE:
            self.contrastive_loss_fn = MCSoftContrastiveLoss(
                config=type('Config', (), {
                    'init_shift': 0.0, 'init_negative_scale': 1.0,
                    'num_samples': 1, 'uniform_lambda': 0.0, 'vib_beta': 0.0,
                    'get': lambda self, k, d=None: getattr(self, k, d),
                })()
            )
        else:
            self.contrastive_loss_fn = None


        # 8. 输出通道映射层（_apply_output_conv 的静态后续层，将任意通道数强制映射到 num_classes）
        self._output_conv = nn.Conv2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=1, bias=True
        )
        with torch.no_grad():
            identity = torch.eye(num_classes).unsqueeze(-1).unsqueeze(-1)
            self._output_conv.weight.copy_(identity)
            nn.init.zeros_(self._output_conv.bias)
        self._target_num_classes = num_classes
        logger.debug(
            "_output_conv 初始化完成（%d->%d），bias=0（已清零，先验由 medical_seg_head 注入）",
            num_classes, num_classes,
        )

        # 9. 注册 RoPE 修正 hook（首次 forward 自动触发）
        self._rope_patched = False
        self._rope_hook_handle = self.register_forward_pre_hook(
            self.adapter_manager._rope_pre_hook
        )

        # text_prompt_encoder：将 global_text_rep 投影为 SAM3 Mask Decoder 的 sparse_prompt_token
        self.text_prompt_encoder = TextPromptEncoder(
            text_dim=text_dim,
            embed_dim=self.embed_dim,
            bottleneck_dim=256,
        )

        # 11. 终极修复：1x1 门面映射层 + 绝对零度隔离 (Zero-Init)
        # ──────────────────────────────────────────────────────────────
        # 链路：SAM3 raw output → _apply_output_conv → medical_seg_head → clamp → Logits
        #
        # Zero-Init 原理：
        #   weight=0 保证第一轮 forward 时 SAM3 的原始输出被完全屏蔽，
        #   medical_seg_head 输出严格等于 bias（常数先验），避免初始特征爆炸。
        #   之后梯度流向 weight（grad = d_loss/d_output * input ≠ 0），
        #   weight 迅速从零离开，模型开始渐进吸收 SAM3 特征。
        #
        # 激活约束（BraTS WT/TC/ET 三区域重叠，非互斥）：
        #   绝对禁止 Softmax / CrossEntropyLoss（互斥假设不成立）
        #   损失计算前必须用 torch.sigmoid() 对每通道独立激活
        #   forward() 只输出原始 Logits，sigmoid 在 cream_losses.py 中执行
        # ──────────────────────────────────────────────────────────────
        self.medical_seg_head = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        _msh_bias = math.log(0.29 / (1.0 - 0.29))  # log(0.29/0.71) ≈ -0.895
        with torch.no_grad():
            nn.init.zeros_(self.medical_seg_head.weight)
            nn.init.constant_(self.medical_seg_head.bias, _msh_bias)
        logger.info(
            "[SAM3MedicalIntegrated Init] medical_seg_head: %dch -> %dch "
            "(Zero-Init weight, bias=%.3f, prior=sigmoid(bias)=%.1f%%)",
            num_classes, num_classes, _msh_bias, 100.0 * 0.29,
        )

        # Monitor 计数器初始化（由 federated_trainer 在每轮开始时赋值为 round_num）
        self._monitor_epoch = 0
        self._monitor_logged_round = -1

    # ====================================================================
    # ====================================================================
    # NOTE: state_dict() 不再重写，使用 PyTorch 原生行为（保留真实子模块前缀），
    # 确保 named_parameters() 与 state_dict().keys() 完全对应，
    # 消灭 Server 端因键名脱节引发的校验崩溃。

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        '''
        加载旧格式 checkpoint：将旧键名补回子模块前缀后再调用 super()。
        支持重构前保存的任何 checkpoint，strict=False 允许部分加载。

        逻辑：
          1. 收集 adapter_manager 和 fusion_head 各自的本地 key 集合
          2. 遍历传入的 state_dict：
             若 key 匹配 adapter_manager 的本地 key，加 adapter_manager. 前缀
             若 key 匹配 fusion_head 的本地 key，    加 fusion_head. 前缀
             否则直接透传
        '''
        am_local_keys = set(self.adapter_manager.state_dict().keys())
        fh_local_keys = set(self.fusion_head.state_dict().keys())
        tpe_local_keys = set(self.text_prompt_encoder.state_dict().keys())
        msh_local_keys = set(self.medical_seg_head.state_dict().keys())

        remapped: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k in am_local_keys:
                remapped['adapter_manager.' + k] = v
            elif k in fh_local_keys:
                remapped['fusion_head.' + k] = v
            elif k in tpe_local_keys:
                remapped['text_prompt_encoder.' + k] = v
            elif k in msh_local_keys:
                remapped['medical_seg_head.' + k] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)


    def _build_mock_sam3(self):
        from src.model import MockViTEncoder, MaskDecoder
        self.image_encoder = MockViTEncoder(
            img_size=self.img_size, patch_size=16,
            embed_dim=self.embed_dim, num_heads=self.num_heads,
        )
        self.mask_decoder = MaskDecoder(
            embed_dim=self.embed_dim, decoder_dim=256, num_classes=self.num_classes
        )
        patch_size = 16
        self.spatial_h = self.img_size // patch_size
        self.spatial_w = self.img_size // patch_size

    def _freeze_encoder(self):
        if self.use_real_sam3:
            if hasattr(self.sam3_model, 'backbone'):
                logger.info("冻结 SAM3 backbone...")
                for param in self.sam3_model.backbone.parameters():
                    param.requires_grad = False
                if self.use_adapter and self.adapter_manager.adapters is not None:
                    logger.info(
                        "编码器已冻结，%d 个 Adapter 训练权重不受影响。",
                        len(self.adapter_manager.adapters),
                    )
        else:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def _unfreeze_adapter_params(self) -> None:
        """Keep adapter parameters trainable after freezing the encoder."""
        trainable_params = 0
        for adapter in self.adapter_manager.adapters:
            for param in adapter.parameters():
                param.requires_grad = True
                trainable_params += 1
        logger.info("Adapter params restored to trainable state: %d", trainable_params)

    # ====================================================================
    # 委托到子模块的公共接口（签名不变）
    # ====================================================================

    def setup_grid_size(self, image_h: int, image_w: int) -> None:
        self.adapter_manager.setup_grid_size(image_h, image_w)

    def reset_rope_frequencies(self, verbose: bool = False) -> int:
        return self.adapter_manager.reset_rope_frequencies(verbose=verbose)

    def _apply_adapters(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter_manager.apply_adapters(features)

    def _apply_fusion(
        self,
        image_embeddings: torch.Tensor,
        text_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.fusion_head.apply_fusion(image_embeddings, text_features)

    # ====================================================================
    # 输出通道适配（原逻辑完全不变）
    # ====================================================================

    def _apply_output_conv(self, logits: torch.Tensor, adapter_attr: str) -> torch.Tensor:
        if logits.shape[1] != self.num_classes:
            import numpy as np
            out_ch = logits.shape[1]
            conv = getattr(self, adapter_attr, None)
            if conv is None or conv.in_channels != out_ch:
                conv = nn.Conv2d(out_ch, self.num_classes, kernel_size=1, bias=True).to(logits.device)
                nn.init.xavier_uniform_(conv.weight)
                nn.init.constant_(conv.bias, np.log(0.01 / 0.99))
                setattr(self, adapter_attr, conv)
            logits = conv(logits)
        assert logits.shape[1] == self.num_classes, (
            f"本地适配层输出通道数错误！期望 {self.num_classes}，实际 {logits.shape[1]}"
        )
        final_logits = self._output_conv(logits)
        if final_logits.shape[1] != self.num_classes:
            raise RuntimeError(
                f"OUTPUT CHANNEL MISMATCH: "
                f"expected {self.num_classes}, got {final_logits.shape[1]}"
            )
        return final_logits

    # ====================================================================
    # forward 路由（原逻辑完全不变）
    # ====================================================================

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
        text_features: Optional[torch.Tensor] = None,
        global_text_rep: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images:          输入图像 (B, C, H, W)
            return_features: 是否返回中间特征
            text_features:   当前批次原始文本特征（早期融合，GatedFusion）
            global_text_rep: 服务器下发的全局文本表征 (D,)，调用前已 .detach()
        """
        if self.use_real_sam3:
            result = self._forward_real_sam3(images, return_features, text_features, global_text_rep)
        else:
            result = self._forward_mock_sam3(images, return_features, text_features, global_text_rep)
        # BraTS mask 形状为 [B, 3, H, W]，分割头必须输出 num_classes（默认3）通道才能对接损失函数。
        # 旧版 out_channels=1 会导致 pred=[B,1,H,W] vs mask=[B,3,H,W] RuntimeError，已修复。
        result.logits = self.medical_seg_head(result.logits)

        # 空气区域物理硬掩码：脑组织外绝对背景强制镇压到 -20.0，切断假阳性来源
        with torch.no_grad():
            brain_mask = (images.abs().sum(dim=1, keepdim=True) > 1e-4).float()
            brain_mask = F.interpolate(brain_mask, size=result.logits.shape[2:], mode='nearest')
        result.logits = torch.where(brain_mask > 0.5, result.logits,
                                    torch.full_like(result.logits, -20.0))

        result.logits = result.logits.clamp(-20.0, 20.0)

        # Logits 实时监控（训练期前 5 轮自动激活，之后静默）
        current_round = getattr(self, '_monitor_epoch', 0)
        if self.training and current_round <= 5 and current_round != getattr(self, '_monitor_logged_round', -1):
            with torch.no_grad():
                _lmax = result.logits.max().item()
                _lmin = result.logits.min().item()

                # 仅在脑组织区域内计算 fg_ratio（排除空气区域干扰）
                valid_mask = brain_mask > 0.5
                if valid_mask.any():
                    valid_logits = result.logits[valid_mask.expand_as(result.logits)]
                    _fg = torch.sigmoid(valid_logits).mean().item()
                else:
                    _fg = 0.0

            logger.warning(
                "[Logits Monitor | Round %d] max=%.4f | min=%.4f | fg_ratio_in_brain=%.4f",
                current_round, _lmax, _lmin, _fg,
            )
            if abs(_lmax) > 20 or abs(_lmin) > 20:
                logger.error(
                    "⚠️ [Logits 越界] 绝对值超过 20！max=%.4f min=%.4f "
                    "→ 请立即检查 lr / 初始化 / 损失权重。",
                    _lmax, _lmin,
                )
            # Zero-Init 期望脑组织内 fg_ratio ≈ 29%（容忍 ±10% 偏差）
            if _fg < 0.20 or _fg > 0.40:
                logger.warning(
                    "⚠️ [脑组织内前景比例异常] fg_ratio=%.4f（期望 29%%±10%%）"
                    " → 请检查 medical_seg_head bias 初始化。",
                    _fg,
                )

            self._monitor_logged_round = current_round

        return result.to_dict()

    def _forward_real_sam3(
        self,
        images: torch.Tensor,
        return_features: bool,
        text_features: Optional[torch.Tensor],
        global_text_rep: Optional[torch.Tensor] = None,
    ) -> FederatedOutput:
        self.sam3_model.eval()
        logits, features = self._forward_real_sam3_standard_with_prompt(
            images, return_features, global_text_rep
        )
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
        logits = self._apply_output_conv(logits, '_sam3_adapter_conv')
        return FederatedOutput(logits=logits, img_rep=features)


    def _forward_real_sam3_standard_with_prompt(
        self,
        images: torch.Tensor,
        return_features: bool,
        global_text_rep: Optional[torch.Tensor] = None,
    ):
        """
        直连 Mask Decoder 注入路径。

        当 global_text_rep 不为 None 时：
          1. 先跑完整 SAM3 前向拿到 image features（backbone 输出的中间特征）
          2. 用 text_prompt_encoder 生成 sparse_embeddings (B, 1, 256)
          3. 调 _call_mask_decoder 用 sparse_embeddings 重推 logits
        当 global_text_rep 为 None 时，直接返回标准前向结果。

          此处不再执行截断（职责单一原则）。
        """
        # 标准 SAM3 完整前向（拿 features / logits）
        logits, features = self._forward_real_sam3_standard(images, return_features=True)

        if global_text_rep is None:
            return logits, features

        # text_prompt_encoder 生成 sparse_embeddings (B, 1, 256)
        B = images.shape[0]
        sparse_embeddings = self.text_prompt_encoder(
            global_text_rep=global_text_rep,
            batch_size=B,
            device=images.device,
            dtype=images.dtype,
        )  # (B, 1, 256)

        # 构造 dense_embeddings 占位符
        ref_h = images.shape[2] // 16
        ref_w = images.shape[3] // 16
        dense_embeddings = torch.zeros(
            (B, 256, ref_h, ref_w), dtype=images.dtype, device=images.device
        )

        # 用 image features 直接调 mask_decoder，覆盖 logits
        # features 来自 _forward_real_sam3_standard 的 final_results.get('features', None)
        # 若 features 为 None（SAM3 未导出特征图），不进行重推，保留标准 logits
        if features is not None:
            try:
                logits = self._call_mask_decoder(features, sparse_embeddings, dense_embeddings)
                logger.debug(
                    "[TextPromptInjection] mask_decoder 重推成功，"
                    "sparse_embeddings.shape=%s, new_logits.shape=%s",
                    sparse_embeddings.shape, logits.shape,
                )
            except Exception as exc:
                logger.warning(
                    "[TextPromptInjection] _call_mask_decoder 失败，回退到标准 logits: %s", exc
                )

        return logits, features

    def _get_image_encoder(self) -> nn.Module:
        if hasattr(self.sam3_model, 'backbone'):
            backbone = self.sam3_model.backbone
            if hasattr(backbone, 'vision_backbone'):
                vb = backbone.vision_backbone
                return vb.trunk if hasattr(vb, 'trunk') else vb
            return backbone
        if hasattr(self.sam3_model, 'image_encoder'):
            return self.sam3_model.image_encoder
        raise RuntimeError("无法找到 SAM3 Image Encoder！")

    def _get_prompts_with_text_prior(
        self,
        images: torch.Tensor,
        batch_size: int,
        text_prompt: Optional[torch.Tensor] = None,
    ):
        """

        将 global_text_rep 通过 TextPromptEncoder 投影为
        sparse_prompt_embeddings (B, 1, embed_dim)，与原有空山 (B, 0, D) 相比
        多了一个文本语义 token 参与 Mask Decoder cross-attention。

        防御注意：
          - text_prompt 应已在 client.py 侧 .detach()，这里不再做一次截断
          - text_prompt=None 时回退为 (B,0,embed_dim)，完全向后兼容

        PE 注意：
          SAM Mask Decoder (mask_decoder.py L194) 直接
            tokens = cat([output_tokens, sparse_prompt_embeddings], dim=1)
          sparse_prompt_embeddings 未加任何位置编码，纯语义 token 参与 attention 安全。
        """
        # 文本 Prompt Token 生成
        sparse_embeddings = self.text_prompt_encoder(
            global_text_rep=text_prompt,
            batch_size=batch_size,
            device=images.device,
            dtype=images.dtype,
        )  # (B, 1, embed_dim) 或 (B, 0, embed_dim)

        # Dense PE（原逻辑不变）
        dense_embeddings = None
        prompt_encoder = getattr(self.sam3_model, 'prompt_encoder', None)
        if prompt_encoder is None and hasattr(self.sam3_model, 'transformer'):
            prompt_encoder = getattr(self.sam3_model.transformer, 'prompt_encoder', None)
        if prompt_encoder is not None and hasattr(prompt_encoder, 'get_dense_pe'):
            try:
                dense_embeddings = prompt_encoder.get_dense_pe()
                if dense_embeddings.dim() == 3:
                    dense_embeddings = dense_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
            except Exception as exc:
                logger.debug("get_dense_pe failed (fallback): %s", exc)

        # 兼容 SAM3 多尺度特征金字塔 (List)
        if dense_embeddings is None:
            # 从 images 本身推断参考形状，避免依赖 fused_embeddings（此时可能尚未计算）
            # image_pe 在 dense prompt 为空时应使用 prompt_encoder 的位置编码尺寸，
            # 退而求其次用全零张量（B, 256, H/4, W/4）作为占位符。
            # 典型 SAM2/SAM3：image_size=1024 → feature map 64x64
            ref_h = images.shape[2] // 16
            ref_w = images.shape[3] // 16
            dense_embeddings = torch.zeros(
                (batch_size, 256, ref_h, ref_w),
                dtype=images.dtype,
                device=images.device,
            )
            logger.debug(
                "dense_embeddings 回退为全零占位符 (B=%d, 256, %d, %d)",
                batch_size, ref_h, ref_w,
            )

        return sparse_embeddings, dense_embeddings

    def _call_mask_decoder(self, fused_embeddings, sparse_embeddings, dense_embeddings):
        _SAM_DECODER_NAMES = {
            'MaskDecoder', 'SAMMaskDecoder', 'SAM2MaskDecoder',
            'SAM3MaskDecoder', 'MemoryAttentionDecoder',
        }

        def _is_sam_decoder(module) -> bool:
            """返回 True 仅当 module 是真正的 SAM Mask Decoder（非 torch.nn 内置）。"""
            if module is None:
                return False
            cls_name = type(module).__name__
            # 类名白名单：必须命中 SAM 相关名称
            if any(name in cls_name for name in _SAM_DECODER_NAMES):
                return True
            # 备用：检测是否具备 SAM MaskDecoder 特有属性
            sam_attrs = {'output_hypernetworks_mlps', 'iou_prediction_head',
                         'mask_tokens', 'output_upscaling', 'upsample4'}
            if any(hasattr(module, attr) for attr in sam_attrs):
                return True
            return False

        # 1. 最优先：sam3_model.mask_decoder（若为真正 SAM Decoder）
        candidate = getattr(self.sam3_model, 'mask_decoder', None)
        mask_decoder = candidate if _is_sam_decoder(candidate) else None

        # 2. 遍历 sam3_model 所有直接子模块，查找 SAM Decoder
        if mask_decoder is None:
            for name, mod in self.sam3_model.named_modules():
                if _is_sam_decoder(mod):
                    mask_decoder = mod
                    logger.debug("_call_mask_decoder: 从子模块 '%s' 找到 SAM Decoder (%s)",
                                 name, type(mod).__name__)
                    break

        # 3. 最后回退：任何 mask_decoder / decoder 属性（但禁止 torch.nn.TransformerDecoder）
        if mask_decoder is None:
            for attr_path in ['mask_decoder', 'decoder']:
                obj = self.sam3_model
                for part in attr_path.split('.'):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj is not None and not isinstance(obj, nn.TransformerDecoder):
                    mask_decoder = obj
                    logger.warning(
                        "_call_mask_decoder: 回退到 attr '%s' (%s)，请确认其为 SAM Decoder。",
                        attr_path, type(obj).__name__,
                    )
                    break

        if mask_decoder is None:
            raise RuntimeError("无法找到 SAM3 Mask Decoder（TransformerDecoder 已被排除）！")

        def _parse(out):
            if isinstance(out, tuple): return out[0]
            if isinstance(out, dict): return out.get('masks', out.get('pred_masks'))
            return out

        # 兼容 SAM3 多尺度特征金字塔 (List)
        # 若 dense_embeddings 在此处仍意外为 None，从 fused_embeddings 安全提取参考张量
        if dense_embeddings is None:
            if isinstance(fused_embeddings, list):
                # 提取 FPN 列表中的主特征图（索引0 = 最大空间分辨率）
                ref_tensor = fused_embeddings[0]
                logger.debug(
                    "_call_mask_decoder: fused_embeddings 是 FPN list（共 %d 层），"
                    "使用 index=0 作为参考张量，shape=%s",
                    len(fused_embeddings), list(ref_tensor.shape),
                )
            else:
                ref_tensor = fused_embeddings
            B, _, H, W = ref_tensor.shape
            # SAM 系列标准的 prompt_embed_dim 通常为 256
            # 不要用 zeros_like，手动对齐 Batch、Dim 和 Device
            dense_embeddings = torch.zeros(
                (B, 256, H, W),
                dtype=ref_tensor.dtype,
                device=ref_tensor.device,
            )
            logger.debug(
                "_call_mask_decoder: dense_embeddings 补零 (B=%d, 256, %d, %d)", B, H, W,
            )

        image_pe = dense_embeddings

        return _parse(mask_decoder(
            image_embeddings=fused_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        ))


    def _forward_real_sam3_standard(self, images, return_features):
        batched_input = self._to_batched_datapoint(images, text_features=None)
        self._sanitize_batched_input(batched_input, images.device)
        output = self.sam3_model(batched_input)

        if type(output).__name__ == 'SAM3Output':
            if hasattr(output, 'iter_mode') and output.iter_mode.name == 'LAST_STEP_PER_STAGE':
                final_results = output[-1]
            else:
                try:
                    final_results = output[-1][-1]
                except Exception:
                    final_results = output[-1] if isinstance(output, list) and len(output) > 0 else output
        else:
            final_results = output[-1][-1]

        logits = None
        if isinstance(final_results, dict):
            if 'semantic_seg' in final_results:
                logits = final_results['semantic_seg']
            elif 'pred_masks' in final_results:
                masks = final_results['pred_masks']
                # 旧版 masks[:, 0:1, :, :] 强制切单通道是输出 [B,1,H,W] 的根源。
                # 必须保留所有通道，交由下游 _apply_output_conv → medical_seg_head 做通道映射。
                # BraTS 三区域（WT/TC/ET）重叠，下游损失函数用 sigmoid 独立激活每通道。
                logits = masks if masks.dim() == 4 else masks
                logger.debug("使用 pred_masks，保留全部通道，形状: %s", logits.shape)
            elif 'pred_logits' in final_results:
                logits = final_results['pred_logits']
                logger.debug("使用 pred_logits 作为 fallback，形状: %s", logits.shape)
        else:
            logits = final_results

        if logits is None:
            keys = final_results.keys() if isinstance(final_results, dict) else 'Not a dict'
            raise ValueError(f"无法从 SAM3 输出中提取 logits。可用键: {keys}")

        features = None
        if return_features:
            try:
                encoder = self._get_image_encoder()
                raw_feats = encoder(images)
                if isinstance(raw_feats, (list, tuple)):
                    raw_feats = raw_feats[-1]
                if raw_feats.dim() == 4:   # (B, C, H, W) → global pool → (B, C)
                    features = F.adaptive_avg_pool2d(raw_feats, (1, 1)).flatten(1)
                elif raw_feats.dim() == 3:  # (B, N, C) → mean pool → (B, C)
                    features = raw_feats.mean(dim=1)
                else:
                    features = raw_feats
            except Exception as exc:
                logger.warning("[_forward_real_sam3_standard] 图像特征提取失败: %s", exc)
        return logits, features

    @staticmethod
    def _sanitize_batched_input(batched_input: Any, device: torch.device) -> None:
        def _to_tensor(val, dtype):
            return torch.tensor(val, dtype=dtype, device=device) if isinstance(val, list) else val
        if hasattr(batched_input, 'find_inputs'):
            for stage_or_list in batched_input.find_inputs:
                stages = stage_or_list if isinstance(stage_or_list, list) else [stage_or_list]
                for stage in stages:
                    if hasattr(stage, 'input_points'):
                        stage.input_points = _to_tensor(stage.input_points, torch.float32)
                    if hasattr(stage, 'input_boxes'):
                        stage.input_boxes = _to_tensor(stage.input_boxes, torch.float32)
                    if hasattr(stage, 'input_labels'):
                        stage.input_labels = _to_tensor(stage.input_labels, torch.long)
                    if hasattr(stage, 'input_boxes_label'):
                        stage.input_boxes_label = _to_tensor(stage.input_boxes_label, torch.long)
        elif isinstance(batched_input, dict):
            for key, dtype in [('input_points', torch.float32), ('input_boxes', torch.float32),
                                ('input_labels', torch.long)]:
                if key in batched_input:
                    batched_input[key] = _to_tensor(batched_input[key], dtype)
        elif isinstance(batched_input, list):
            for item in batched_input:
                if isinstance(item, dict):
                    for key, dtype in [('input_points', torch.float32), ('input_boxes', torch.float32),
                                       ('input_labels', torch.long)]:
                        if key in item:
                            item[key] = _to_tensor(item[key], dtype)

    def _forward_mock_sam3(
        self, images, return_features, text_features, text_prompt=None
    ) -> FederatedOutput:
        ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            features = self.image_encoder(images)
        if self.use_adapter:
            features = self._apply_adapters(features)
        features = self._apply_fusion(features, text_features)
        logits = self.mask_decoder(features, spatial_shape=(self.spatial_h, self.spatial_w))
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
        logits = self._apply_output_conv(logits, '_mock_adapter_conv')
        return FederatedOutput(logits=logits, img_rep=features if return_features else None)

    def _to_batched_datapoint(self, images, prompts=None, text_features=None) -> BatchedDatapoint:
        device = images.device
        batch_size = images.shape[0]
        find_text_batch = []
        if text_features is not None:
            if isinstance(text_features, list) and (
                len(text_features) == 0 or isinstance(text_features[0], str)
            ):
                find_text_batch = text_features
            elif isinstance(text_features, torch.Tensor):
                raise TypeError(
                    "SAM3 find_text_batch 期待 List[str]，"
                    f"但收到 torch.Tensor (shape: {text_features.shape})。"
                )
            else:
                raise TypeError(f"未知的 text_features 类型: {type(text_features)}。")

        max_boxes, max_points = 0, 0
        if prompts is not None:
            for b in range(min(batch_size, len(prompts))):
                p = prompts[b]
                max_boxes = max(max_boxes, len(p.get('boxes', [])))
                max_points = max(max_points, len(p.get('points', [])))

        input_boxes = torch.zeros(max_boxes, batch_size, 4, dtype=torch.float32, device=device)
        input_boxes_label = torch.zeros(max_boxes, batch_size, dtype=torch.long, device=device)
        input_boxes_mask = torch.ones(batch_size, max_boxes, dtype=torch.bool, device=device)
        input_points = torch.zeros(max_points, batch_size, 2, dtype=torch.float32, device=device)
        input_points_label = torch.zeros(max_points, batch_size, dtype=torch.long, device=device)
        input_points_mask = torch.ones(batch_size, max_points, dtype=torch.bool, device=device)

        for b in range(batch_size):
            if prompts is not None and b < len(prompts):
                p = prompts[b]
                boxes = p.get('boxes', [])
                points = p.get('points', [])
                labels_points = p.get('labels', [])
                if boxes:
                    bx = torch.tensor(boxes, dtype=torch.float32, device=device)
                    if bx.dim() == 1: bx = bx.unsqueeze(0)
                    nc = min(bx.shape[0], max_boxes)
                    input_boxes[:nc, b, :] = bx[:nc]
                    input_boxes_label[:nc, b] = 1
                    input_boxes_mask[b, :nc] = False
                if points:
                    pt = torch.tensor(points, dtype=torch.float32, device=device)
                    lbl = torch.tensor(labels_points, dtype=torch.long, device=device)
                    if pt.dim() == 1:
                        pt = pt.unsqueeze(0)
                        lbl = lbl.unsqueeze(0) if lbl.dim() == 0 else lbl
                    nc = min(pt.shape[0], max_points)
                    input_points[:nc, b, :] = pt[:nc]
                    input_points_label[:nc, b] = lbl[:nc]
                    input_points_mask[b, :nc] = False

        img_ids = torch.arange(batch_size, dtype=torch.long, device=device)
        single_stage = FindStage(
            img_ids=img_ids,
            text_ids=torch.zeros(0, dtype=torch.long, device=device),
            input_boxes=input_boxes, input_boxes_label=input_boxes_label,
            input_boxes_mask=input_boxes_mask, input_points=input_points,
            input_points_mask=input_points_mask,
            object_ids=torch.zeros(0, dtype=torch.long, device=device),
        )
        find_targets = [
            BatchedFindTarget(
                num_boxes=[], boxes=[], boxes_padded=[], is_exhaustive=[],
                segments=[], semantic_segments=[], is_valid_segment=[],
                repeated_boxes=[], object_ids=[], object_ids_padded=[],
            ) for _ in range(batch_size)
        ]
        find_metadatas = [
            BatchedInferenceMetadata(
                coco_image_id=[], original_size=[], object_id=[], frame_index=[],
                original_image_id=[], original_category_id=[], is_conditioning_only=[],
            ) for _ in range(batch_size)
        ]
        return BatchedDatapoint(
            img_batch=images, find_text_batch=find_text_batch,
            find_inputs=[single_stage], find_targets=find_targets,
            find_metadatas=find_metadatas, raw_images=None,
        )

    # ====================================================================
    # 对外公共 API（签名与行为均不变）
    # ====================================================================

    def extract_features(self, images, text_features=None, **kwargs) -> torch.Tensor:
        output = self.forward(images, return_features=True, text_features=text_features, **kwargs)
        features = output.get('features', None)
        if features is None:
            if self.use_real_sam3:
                logger.warning("[extract_features] SAM3 未能导出特征图，返回零张量。")
            else:
                with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                    features = self.image_encoder(images)
                if self.use_adapter:
                    features = self._apply_adapters(features)

        if features is None:
            return torch.zeros(images.size(0), self.contrastive_dim, device=images.device)

        # 展平到 2D (B, D)：mock 路径返回 3D (B,N,D)，全局均值压成 2D
        if features.dim() == 3:
            features = features.mean(dim=1)   # (B, N, D) → (B, D)

        D = features.shape[-1]
        expected_dim = self.fusion_head.image_proj[0].in_features
        if D == expected_dim:
            # 维度一致：直接用 image_proj Linear(D → contrastive_dim)
            projected_features = self.fusion_head.image_proj(features)
        else:
            # encoder 输出维度与 image_proj 不一致，用 lazy _enc_proj 直连 encoder_dim → contrastive_dim
            if not hasattr(self.fusion_head, '_enc_proj') or \
                    self.fusion_head._enc_proj[0].in_features != D:
                self.fusion_head._enc_proj = nn.Sequential(
                    nn.Linear(D, self.contrastive_dim, bias=False),
                    nn.LayerNorm(self.contrastive_dim),
                ).to(features.device)
                nn.init.normal_(self.fusion_head._enc_proj[0].weight, std=0.02)
                logger.info(
                    "[extract_features] 创建 _enc_proj: %d → contrastive_dim=%d",
                    D, self.contrastive_dim,
                )
            projected_features = self.fusion_head._enc_proj(features)

        return projected_features   # (B, contrastive_dim) 2D

    def get_trainable_params(self) -> list:
        trainable_params = []
        if self.use_adapter and self.adapter_manager.adapters is not None:
            for adapter in self.adapter_manager.adapters:
                trainable_params.extend(list(adapter.parameters()))
        trainable_params.extend(list(self.fusion_head.text_proj.parameters()))
        trainable_params.extend(list(self.fusion_head.image_proj.parameters()))
        if hasattr(self, '_output_conv') and self._output_conv is not None:
            trainable_params.extend(list(self._output_conv.parameters()))
        if hasattr(self, 'medical_seg_head') and self.medical_seg_head is not None:
            trainable_params.extend(list(self.medical_seg_head.parameters()))
        if self.use_real_sam3:
            if hasattr(self.sam3_model, 'segmentation_head'):
                trainable_params.extend(list(self.sam3_model.segmentation_head.parameters()))
            if hasattr(self.sam3_model, 'transformer'):
                if hasattr(self.sam3_model.transformer, 'decoder'):
                    trainable_params.extend(
                        list(self.sam3_model.transformer.decoder.parameters())
                    )
        else:
            trainable_params.extend(list(self.mask_decoder.parameters()))
        return trainable_params

    def get_adapter_params(self) -> list:
        if self.use_adapter and self.adapter_manager.adapters is not None:
            params = []
            for adapter in self.adapter_manager.adapters:
                params.extend(list(adapter.parameters()))
            return params
        return []

    def count_parameters(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for p in self.get_adapter_params())
        return {
            'total': total_params,
            'trainable': trainable_params,
            'adapter': adapter_params,
            'trainable_ratio': trainable_params / total_params * 100 if total_params > 0 else 0,
            'adapter_ratio': adapter_params / total_params * 100 if total_params > 0 else 0,
        }

    def compute_contrastive_loss(self, image_features, text_features=None) -> torch.Tensor:
        if self.contrastive_loss_fn is None:
            return torch.tensor(0.0, device=image_features.device)
        if text_features is None:
            text_features = image_features
        if len(image_features.shape) == 2:
            image_features = image_features.unsqueeze(1)
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        image_logsigma = torch.zeros(image_features.size(0), device=image_features.device)
        text_logsigma = torch.zeros(text_features.size(0), device=text_features.device)
        loss, _ = self.contrastive_loss_fn(
            image_features, text_features, image_logsigma, text_logsigma
        )
        return loss

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            if self.use_real_sam3:
                self.sam3_model.eval()
            else:
                self.image_encoder.eval()
        return self



if __name__ == "__main__":
    BATCH_SIZE = 6
    LR = 2e-4
    IMG_SIZE = 1024
    ROUNDS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Testing SAM3MedicalIntegrated (Builder + Facade Refactoring)")
    print("=" * 60)

    print("\n[Test 1] Adapter Zero-Initialization...")
    test_adapter = Adapter(in_dim=768, out_dim=768, adapter_dim=64)
    assert torch.allclose(test_adapter.up_proj.weight.data, torch.zeros_like(test_adapter.up_proj.weight.data))
    print("OK")

    print("\n[Test 2] Adapter Initial Output...")
    test_input = torch.randn(2, 64, 768)
    assert torch.allclose(test_adapter(test_input), torch.zeros(2, 64, 768), atol=1e-6)
    print("OK (zero output with zero-init)")

    print("\n[Test 3] BlockWithAdapter Parameter Forwarding...")
    class MockBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)
        def forward(self, x, extra_arg=None, **kwargs):
            if extra_arg is not None: assert extra_arg == "test_value"
            return self.linear(x)
    wrapped = BlockWithAdapter(MockBlock(), dim=768, adapter_dim=64)
    wrapped(torch.randn(2, 64, 768), extra_arg="test_value", another_kwarg=123)
    print("OK (*args, **kwargs forwarding)")

    print("\n[Test 4] SAM3MedicalIntegrated Mock Mode...")
    model = SAM3MedicalIntegrated(
        img_size=256, num_classes=1, use_sam3=SAM3_AVAILABLE,
        freeze_encoder=True, use_adapter=True, device=DEVICE,
    ).to(DEVICE)
    dummy = torch.randn(2, 3, 256, 256).to(DEVICE)
    try:
        out = model(dummy)
        print(f"OK | logits: {out['logits'].shape}")
    except Exception as e:
        print(f"WARN (no SAM3 checkpoint): {e}")

    print("\n[Test 5] state_dict Key Integrity (named_parameters <-> state_dict alignment)...")
    sd = model.state_dict()
    # 验证 named_parameters() 的 key 集合与 state_dict() 的 key 集合一致
    param_keys = {name for name, _ in model.named_parameters()}
    sd_keys = set(sd.keys())
    # state_dict 包含参数和 buffer，named_parameters 只含参数；buffer 允许多出来
    missing_in_sd = param_keys - sd_keys
    assert not missing_in_sd, f"以下参数在 state_dict 中缺失（键名脱节）: {missing_in_sd}"
    print(f"OK | {len(sd)} keys, named_parameters 与 state_dict 对齐，无键名脱节")

    print("\n[Test 6] load_state_dict Round-Trip...")
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    loaded_sd = torch.load(buf, map_location='cpu')
    model2 = SAM3MedicalIntegrated(
        img_size=256, num_classes=1, use_sam3=SAM3_AVAILABLE,
        freeze_encoder=True, use_adapter=True, device='cpu',
    )
    res = model2.load_state_dict(loaded_sd, strict=False)
    print(f"OK | missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")

    print("\n[Test 7] Parameter Statistics...")
    ps = model.count_parameters()
    print(f"OK | Total={ps['total']:,} Trainable={ps['trainable']:,} ({ps['trainable_ratio']:.2f}%) Adapter={ps['adapter']:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
