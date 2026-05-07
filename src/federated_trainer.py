"""
SegSAM 图像分割训练器

此模块将训练循环逻辑封装到 FederatedTrainer 类中，提高代码的可维护性和可复用性。
严格保持与原脚本 scripts/train_brats_federated.py 的逻辑一致性。
"""

# ★ Fix Medium 4: 服务器无 GUI 环境下强制使用非交互式 Matplotlib 后端
import matplotlib
matplotlib.use('Agg')

import os
import math
import time
import random
import logging
import traceback
import platform
import hashlib
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, Dict, Tuple, Optional, Any
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to python path to allow running as script directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_BRATS_NUM_CLASSES   = 3      # BraTS 肿瘤子区域（WT/TC/ET）
_BERT_TEXT_DIM       = 768    # BERT-base 输出维度
_CONTRASTIVE_DIM     = 1024   # 对比学习嵌入空间维度
_EARLY_STOP_PATIENCE = 20     # Early Stopping 耐心轮数
_VAL_PLOT_INTERVAL   = 5      # 验证集评估 & 绘图间隔（轮）
_SEG_HEAD_LR_INIT    = 1e-3   # 分割头 Zero-Init 唤醒学习率（必须 ≥ 1e-3 才能打破零初始化死锁）
_ADAPTER_LR_MAX      = 1e-4   # Adapter/LoRA 参数学习率上限

from src.config_manager import FederatedConfig
from src.integrated_model import SAM3MedicalIntegrated as SAM3_Medical
# ★ Fix Critical 2: 引入三个具体 Trainer 子类，移除已废弃的 ClientTrainer
from src.client import TextOnlyTrainer, ImageOnlyTrainer, MultimodalTrainer
from src.server import CreamAggregator
from src.logger import create_logger
from data.heterogeneous_dataset_loader import create_heterogeneous_data_loaders


# ★ Fix Critical 2: 工厂函数 - 根据 modality 字符串实例化正确的 Trainer 子类
def create_client_trainer(modality: str, **kwargs):
    """
    根据模态类型实例化对应的客户端训练器子类。

    Args:
        modality: 模态类型字符串，取值为 'text_only' | 'image_only' | 'multimodal'
        **kwargs: 传递给 Trainer 构造函数的参数
                  (private_loader, public_loader, device, use_amp,
                   local_epochs, dataset_name, embed_dim, grad_clip)
    Returns:
        对应的 Trainer 子类实例
    Raises:
        ValueError: 如果 modality 不在已知类型中
    """
    modality_map = {
        'text_only': TextOnlyTrainer,
        'image_only': ImageOnlyTrainer,
        'multimodal': MultimodalTrainer,
    }
    trainer_cls = modality_map.get(modality)
    if trainer_cls is None:
        raise ValueError(
            f"未知的客户端模态类型: '{modality}'，"
            f"有效值为 {list(modality_map.keys())}"
        )
    return trainer_cls(**kwargs)


def compute_gradient_conflict_from_vectors(
    client_grad_vectors: List[Optional[torch.Tensor]],
    client_modalities: List[str],
) -> Optional[float]:
    """Compute the adapter-gradient conflict angle from real client gradient vectors."""
    try:
        idx_img = client_modalities.index('image_only')
        idx_multi = client_modalities.index('multimodal')
    except ValueError:
        return None

    img_vec = client_grad_vectors[idx_img]
    multi_vec = client_grad_vectors[idx_multi]
    if img_vec is None or multi_vec is None:
        print("[WARNING][GradConflict] 缺少真实 adapter 梯度向量，跳过冲突角计算")
        return None

    img_vec = img_vec.detach().float().flatten().cpu()
    multi_vec = multi_vec.detach().float().flatten().cpu()
    if img_vec.numel() == 0 or multi_vec.numel() == 0:
        print("[WARNING][GradConflict] adapter 梯度向量为空，跳过冲突角计算")
        return None
    if img_vec.numel() != multi_vec.numel():
        print(
            "[WARNING][GradConflict] adapter 梯度向量维度不一致，"
            f"image_only={img_vec.numel()} multimodal={multi_vec.numel()}，跳过冲突角计算"
        )
        return None

    img_norm = torch.linalg.norm(img_vec)
    multi_norm = torch.linalg.norm(multi_vec)
    if img_norm.item() <= 1e-12 or multi_norm.item() <= 1e-12:
        print("[WARNING][GradConflict] adapter 梯度范数为 0，跳过冲突角计算")
        return None

    cos_sim = torch.dot(img_vec, multi_vec) / (img_norm * multi_norm)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0).item()
    angle_deg = math.degrees(math.acos(cos_sim))
    print(
        f"[GradConflict] true adapter gradient conflict angle: {angle_deg:.2f}° "
        f"(cos_sim={cos_sim:.4f}, adapter_elems={img_vec.numel()})"
    )
    return angle_deg


class FederatedTrainer:
    """
    联邦学习训练器
    
    封装了联邦学习训练的完整流程，包括：
    - 环境和模型初始化
    - 客户端配置和训练
    - 服务器聚合
    - 验证集评估
    - 检查点管理
    - 训练可视化
    """
    
    def __init__(self, config: FederatedConfig):
        """
        初始化联邦训练器
        
        Args:
            config: FederatedConfig 配置对象
        """
        self.config = config
        
        # 核心组件（延迟初始化）
        self.device = None
        self.global_model = None
        self.server = None
        self.client_configs = None
        self.client_trainers = None
        self.client_states = None
        self.val_loader = None
        self.logger = None
        
        # 训练状态
        self.training_history = {
            'rounds': [],
            'avg_losses': [],
            'avg_seg_losses': [],
            'avg_cream_losses': [],
            'client_losses': [],
            'global_text_rep_norms': [],
            'global_image_rep_norms': [],
            'val_metrics': [],
            'run_metadata': {},
            # ★ 论文数据收集（新增）
            'lr_history': [],          # 每轮主干 LR（float）
            'gpu_mem_mb': [],          # 每轮 GPU 峰值显存（MB），CPU 环境为 0
            'round_time_sec': [],      # 每轮完整训练耗时（秒）
            'grad_conflict_deg': [],   # adapter 梯度冲突角（度），SegSAM image-only 无多模态时记录 None
        }
        self.last_val_metrics = {}
        self.best_val_dice = 0.0
        
        # 检查点目录：绑定到各组的 log_dir 下，避免不同实验组互相覆盖
        if not self.config.log_dir:
            self.config.log_dir = str(Path(self.config.data_root) / "logs")
        self.checkpoint_dir = Path(self.config.log_dir) / "checkpoints"

    @staticmethod
    def _normalize_client_id(client_id: str) -> str:
        return str(client_id).replace("_", "").lower()

    def _get_enabled_config_clients(self) -> Dict[str, Dict[str, Any]]:
        enabled_clients: Dict[str, Dict[str, Any]] = {}
        if not getattr(self.config, "clients", None):
            return enabled_clients

        for client_cfg in self.config.clients:
            if not client_cfg.get("enabled", True):
                continue
            client_id = str(client_cfg.get("client_id", "")).strip()
            if not client_id:
                raise ValueError("config.clients contains empty client_id")
            norm_id = self._normalize_client_id(client_id)
            if norm_id in enabled_clients:
                raise ValueError(f"Duplicate client_id in config.clients: {client_id}")
            enabled_clients[norm_id] = client_cfg
        return enabled_clients

    def _validate_group_protocol(self, enabled_clients: Dict[str, Dict[str, Any]]) -> None:
        exp_name = str(getattr(self.config, "experiment_name", "") or "").lower()
        modalities = sorted(str(c.get("modality", "")).strip() for c in enabled_clients.values())
        lambda_cream = float(getattr(self.config, "lambda_cream", 0.0))
        agg = str(getattr(self.config, "aggregation_method", "")).lower()
        use_decoupled = bool(getattr(self.config, "use_decoupled_agg", False))

        if "segsam" in exp_name or "image_only" in exp_name:
            if modalities != ["image_only"]:
                raise ValueError(f"SegSAM image-only protocol mismatch: modalities={modalities}")
            if use_decoupled:
                raise ValueError("SegSAM image-only protocol mismatch: use_decoupled_agg must be false")
            if abs(lambda_cream) > 1e-12:
                raise ValueError(f"SegSAM image-only protocol mismatch: lambda_cream must be 0, got {lambda_cream}")

        if "groupb" in exp_name or "group_b" in exp_name:
            if modalities != ["image_only", "multimodal"]:
                raise ValueError(f"Group B protocol mismatch: modalities={modalities}")
            if use_decoupled:
                raise ValueError("Group B protocol mismatch: use_decoupled_agg must be false")
            if lambda_cream <= 0:
                raise ValueError(f"Group B protocol mismatch: lambda_cream must be > 0, got {lambda_cream}")
            if agg != "fedavg":
                raise ValueError(f"Group B protocol mismatch: aggregation_method must be fedavg, got {agg}")

        if "groupc" in exp_name or "group_c" in exp_name:
            if modalities != ["image_only", "multimodal"]:
                raise ValueError(f"Group C protocol mismatch: modalities={modalities}")
            if not use_decoupled:
                raise ValueError("Group C protocol mismatch: use_decoupled_agg must be true")
            if lambda_cream <= 0:
                raise ValueError(f"Group C protocol mismatch: lambda_cream must be > 0, got {lambda_cream}")

    def _validate_client_protocol_after_filter(self) -> None:
        if not bool(getattr(self.config, "strict_protocol_check", True)):
            return
        if not self.client_configs:
            raise ValueError("Protocol check failed: filtered client_configs is empty")

        expected_clients = self._get_enabled_config_clients()
        if not expected_clients:
            return

        actual_norm_to_id: Dict[str, str] = {}
        for client_id in self.client_configs.keys():
            norm_id = self._normalize_client_id(client_id)
            if norm_id in actual_norm_to_id:
                raise ValueError(f"Protocol check failed: duplicate actual client id {client_id}")
            actual_norm_to_id[norm_id] = client_id

        expected_ids = set(expected_clients.keys())
        actual_ids = set(actual_norm_to_id.keys())
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        if missing:
            raise ValueError(f"Protocol check failed: missing clients {missing}")
        if unexpected:
            raise ValueError(f"Protocol check failed: unexpected clients {unexpected}")

        modality_mismatches = []
        for norm_id, expected_cfg in expected_clients.items():
            actual_id = actual_norm_to_id[norm_id]
            expected_modality = str(expected_cfg.get("modality", "")).strip()
            actual_modality = str(self.client_configs[actual_id].get("modality", "")).strip()
            if expected_modality != actual_modality:
                modality_mismatches.append(
                    f"{actual_id}: expected={expected_modality}, actual={actual_modality}"
                )
        if modality_mismatches:
            raise ValueError("Protocol check failed: " + "; ".join(modality_mismatches))

        self._validate_group_protocol(expected_clients)

    def _should_allow_text_param_upload(self) -> bool:
        """B 组需要让 image_only 客户端上传 text 参数以复现实验污染。"""
        lambda_cream = float(getattr(self.config, "lambda_cream", 0.0))
        use_decoupled = bool(getattr(self.config, "use_decoupled_agg", False))
        return lambda_cream > 0.0 and not use_decoupled

    def _should_enable_text_assist_in_seg(self) -> bool:
        """文本辅助分割与是否解耦聚合解耦，避免 C 组把监督一并关掉。"""
        return bool(getattr(self.config, "enable_text_assist_in_seg", True))

    @staticmethod
    def _stable_json_dumps(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _sha256_json(cls, payload: Any) -> str:
        return hashlib.sha256(cls._stable_json_dumps(payload).encode("utf-8")).hexdigest()

    def _build_run_identity(self) -> Dict[str, Any]:
        config_snapshot = self.config.to_dict()
        client_entries: List[Dict[str, Any]] = []
        if self.client_configs:
            for client_id in sorted(self.client_configs.keys(), key=self._normalize_client_id):
                client_cfg = self.client_configs[client_id]
                client_entries.append({
                    "client_id": str(client_id),
                    "modality": str(client_cfg.get("modality", "")).strip(),
                    "enabled": bool(client_cfg.get("enabled", True)),
                    "data_source": client_cfg.get("data_source"),
                })

        protocol_payload = {
            "experiment_name": getattr(self.config, "experiment_name", None),
            "seed": int(getattr(self.config, "seed", 3407)),
            "rounds": int(getattr(self.config, "rounds", 0)),
            "batch_size": int(getattr(self.config, "batch_size", 0)),
            "local_epochs": int(getattr(self.config, "local_epochs", 0)),
            "lambda_cream": float(getattr(self.config, "lambda_cream", 0.0)),
            "aggregation_method": str(getattr(self.config, "aggregation_method", "")).lower(),
            "use_decoupled_agg": bool(getattr(self.config, "use_decoupled_agg", False)),
            "strict_protocol_check": bool(getattr(self.config, "strict_protocol_check", True)),
            "data_root": str(getattr(self.config, "data_root", "")),
            "clients": client_entries,
        }

        return {
            "config_hash": self._sha256_json(config_snapshot),
            "protocol_hash": self._sha256_json(protocol_payload),
            "protocol_payload": protocol_payload,
        }

    def _set_random_seed(self) -> None:
        seed = int(getattr(self.config, "seed", 3407))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Prefer deterministic behavior for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Seed] Using random seed: {seed}")

    def _collect_run_metadata(self) -> Dict[str, Any]:
        run_identity = self._build_run_identity()
        metadata: Dict[str, Any] = {
            "seed": int(getattr(self.config, "seed", 3407)),
            "experiment_name": getattr(self.config, "experiment_name", None),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": self.device,
            "hostname": platform.node(),
            "run_started_at": datetime.now().isoformat(),
            "config_hash": run_identity["config_hash"],
            "protocol_hash": run_identity["protocol_hash"],
            "protocol_payload": run_identity["protocol_payload"],
        }
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_root),
                text=True,
            ).strip()
        except Exception:
            commit = "unknown"
        metadata["git_commit"] = commit
        return metadata

    def setup_environment(self):
        """初始化环境和全局模型"""
        print("\n[1/4] 初始化全局模型...")

        # 确定实际使用的设备
        self.device = self.config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[警告] CUDA 不可用，切换到 CPU")
            self.device = "cpu"
        self._set_random_seed()

        # 初始化全局模型（num_classes 从 config 读取，SegSAM image-only=1，多模态全集=3）
        self.global_model = SAM3_Medical(
            img_size=self.config.img_size,
            num_classes=self.config.num_classes,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            use_sam3=not self.config.use_mock,
            sam3_checkpoint=self.config.sam3_checkpoint if not self.config.use_mock else None,
            text_dim=_BERT_TEXT_DIM,
            contrastive_dim=_CONTRASTIVE_DIM
        ).to(self.device)

        print(f"  [OK] 全局模型已初始化")

        # ★★★ Critical Fix: 显式重置 RoPE 频率缓存 ★★★
        # 问题：SAM3 预训练权重中的 freqs_cis 是 1024 尺寸，但我们用 256
        # 解决：在训练开始前强制重置，避免每个 forward 都动态重计算
        print(f"  [RoPE] 正在为 img_size={self.config.img_size} 重置 RoPE 频率缓存...")
        try:
            if hasattr(self.global_model, 'reset_rope_frequencies'):
                # ✅ 修复：reset_rope_frequencies() 不接受 img_size 参数
                # RoPE频率会根据模型初始化时的img_size自动计算
                rope_reset_count = self.global_model.reset_rope_frequencies(verbose=True)
                print(f"  [OK] RoPE 频率已重置（方法 1: reset_rope_frequencies, {rope_reset_count} blocks）")
            elif hasattr(self.global_model, 'wrapped_blocks'):
                # 手动遍历所有 blocks 重置
                rope_reset_count = 0
                for i, wrapped_block in enumerate(self.global_model.wrapped_blocks):
                    original_block = wrapped_block.block if hasattr(wrapped_block, 'block') else wrapped_block
                    attn = original_block.attn if hasattr(original_block, 'attn') else None

                    if attn and hasattr(attn, '_setup_rope_freqs') and getattr(attn, 'use_rope', False):
                        attn._setup_rope_freqs()
                        rope_reset_count += 1

                if rope_reset_count > 0:
                    print(f"  [OK] RoPE 频率已重置（方法 2: 手动重置 {rope_reset_count} 个 blocks）")
                else:
                    print(f"  [SKIP] 未找到需要重置的 RoPE blocks")
            else:
                print(f"  [SKIP] 模型不支持 RoPE 重置，跳过")
        except Exception as e:
            print(f"  [警告] RoPE 重置失败: {e}")
            print(f"  将在 forward 时动态重计算（性能会降低）")

        # 初始化服务器
        print("[2/4] 初始化服务器...")
        self.server = CreamAggregator(
            self.global_model,
            device=self.device,
            aggregation_method=self.config.aggregation_method,
            global_rep_alpha=self.config.global_rep_alpha,
            strict_aggregation_guard=getattr(self.config, "strict_aggregation_guard", False),
            proxy_k_batches=getattr(self.config, "proxy_k_batches", 1),
        )
        print(f"  [OK] 服务器已初始化")
    
    def setup_clients(self):
        """设置客户端配置和训练器（串行训练模式）"""
        print("[3/5] 设置客户端配置（串行训练模式）...")
        
        try:
            # 导入串行训练辅助函数
            from scripts.setup_serial_clients import setup_serial_clients
            from scripts.serial_training_utils import get_trainable_state_dict, load_trainable_state_dict
            
            # 保存这些函数供后续使用
            self.get_trainable_state_dict = get_trainable_state_dict
            self.load_trainable_state_dict = load_trainable_state_dict
            
            # 获取客户端配置（不创建模型）
            self.client_configs = setup_serial_clients(
                data_root=self.config.data_root,
                batch_size=self.config.batch_size,
                img_size=self.config.img_size,
                max_samples=self.config.max_samples,
                embed_dim=self.config.embed_dim,
                configured_clients=self.config.clients,
            )

            # config.clients 白名单过滤：由 yaml 决定当前实验组加载哪些客户端
            # 注意：config 里 client_id 可能无下划线（client2），
            # 而 setup_serial_clients 返回的 key 带下划线（client_2），需兼容两种格式
            if hasattr(self.config, 'clients') and self.config.clients:
                allowed_ids = {
                    c['client_id'] for c in self.config.clients
                    if c.get('enabled', True)
                }
                self.client_configs = {
                    k: v for k, v in self.client_configs.items()
                    if any(k == aid or k.replace('_', '') == aid.replace('_', '') for aid in allowed_ids)
                }
                print(f"[Config-Driven Filter] 保留客户端: {list(self.client_configs.keys())}")

            self._validate_client_protocol_after_filter()


            print(f"    - {len(self.client_configs)} 个客户端配置已创建")
            for client_id, cfg in self.client_configs.items():
                print(f"      * {client_id}: {cfg['modality']}")
        except Exception as e:
            print(f"\n错误: 客户端配置设置失败")
            print(f"详细信息: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 初始化客户端 Trainer 对象（无模型）
        print("[4/5] 初始化客户端训练器（无状态）...")
        self.client_trainers = {}
        allow_text_param_upload = self._should_allow_text_param_upload()
        enable_text_assist_in_seg = self._should_enable_text_assist_in_seg()
        for client_id, cfg in self.client_configs.items():
            # ★ Fix Critical 2: 使用工厂函数，根据模态实例化对应子类
            self.client_trainers[client_id] = create_client_trainer(
                modality=cfg['modality'],
                private_loader=cfg['private_loader'],
                public_loader=cfg['public_loader'],
                device=self.device,
                use_amp=self.config.use_amp,
                local_epochs=self.config.local_epochs,
                dataset_name='BraTS',
                grad_clip=getattr(self.config, 'grad_clip', 1.0),
                accumulation_steps=getattr(self.config, 'accumulation_steps', 1),
                enable_text_assist_in_seg=enable_text_assist_in_seg,
                allow_text_param_upload=allow_text_param_upload,
                baseline_method=getattr(self.config, 'baseline_method', 'none'),
                fedprox_mu=getattr(self.config, 'fedprox_mu', 0.0),
            )
        
        # ★ Fix Medium 5: 初始化本地表征使用 contrastive_dim 而非 embed_dim
        contrastive_dim = getattr(self.config, 'contrastive_dim', _CONTRASTIVE_DIM)
        print("[5/5] 初始化客户端状态缓存（CPU）...")
        self.client_states = {}
        for client_id in self.client_configs.keys():
            # 从全局模型提取初始可训练参数，并显式移到 CPU
            initial_trainable_state = self.get_trainable_state_dict(self.global_model)
            initial_trainable_state_cpu = {k: v.cpu() for k, v in initial_trainable_state.items()}

            self.client_states[client_id] = {
                'weights': initial_trainable_state_cpu,
                'opt_state': None,
                'local_reps': torch.zeros(contrastive_dim)  # ★ 使用 contrastive_dim
            }
        
        print(f"    - 客户端状态缓存已初始化 ({len(self.client_states)} 个客户端)")
        if len(self.client_states) > 0:
            first_client_id = list(self.client_states.keys())[0]
            cache_size_mb = sum(p.numel() for p in self.client_states[first_client_id]['weights'].values()) * 4 / 1024 / 1024
            print(f"    - 每个客户端缓存大小: ~{cache_size_mb:.1f} MB (仅可训练参数)")
    
    def setup_validation(self):
        """准备验证集数据加载器"""
        print("\n准备验证集数据加载器...")

        # val_client_ids 优先使用 client_configs.keys()：
        # 该集合的 client_id（如 client_2）已由 setup_serial_clients 验证，
        # 与磁盘目录命名（带下划线）严格一致。
        # config.clients 中的 client_id（如 client2，无下划线）与目录不匹配，
        # 不能作为路径构造的来源。
        val_client_ids = []
        if self.client_configs:
            # 过滤掉 text_only（无图像，无法做 Dice 评估）
            val_client_ids = [
                cid for cid, cfg in self.client_configs.items()
                if cfg.get('modality') != 'text_only'
            ]
        if not val_client_ids:
            print("  ⚠ 未能从 client_configs 中找到图像客户端，跳过验证集加载")

        val_loaders = []
        for client_id in val_client_ids:
            try:
                _modality = self.client_configs[client_id].get('modality', 'image_only')
                val_loaders_dict = create_heterogeneous_data_loaders(
                    data_root=self.config.data_root,
                    split="val",
                    client_configs=[{
                        'client_id': client_id,
                        'modality': _modality,
                    }],
                    batch_size=self.config.batch_size,
                    image_size=self.config.img_size,
                    shuffle=False,
                    max_samples=self.config.max_samples,
                    include_text_features=False,
                    is_validation=True,
                    load_public=False,
                )
                if client_id in val_loaders_dict:
                    val_private, _ = val_loaders_dict[client_id]
                    if val_private is not None:
                        val_loaders.append(val_private)
            except Exception as e:
                print(f"  警告: 无法加载验证集 ({client_id}): {e}")
        
        # 合并所有验证集
        if val_loaders:
            val_datasets = [loader.dataset for loader in val_loaders]
            val_concat_dataset = ConcatDataset(val_datasets)
            self.val_loader = DataLoader(val_concat_dataset, batch_size=self.config.batch_size, shuffle=False)
            print(f"  [OK] 验证集准备完成，共 {len(val_concat_dataset)} 个样本")
        else:
            self.val_loader = None
            print("  ⚠ 未找到验证集数据，将跳过验证集评估")
    
    def setup_logging(self):
        """初始化日志记录器"""
        if self.config.log_type != 'none':
            print("\n初始化日志记录器...")
            config_dict = self.config.to_dict()
            self.logger = create_logger(
                log_type=self.config.log_type,
                experiment_name=self.config.experiment_name,
                project_name=self.config.wandb_project,
                log_dir=self.config.log_dir or str(Path(self.config.data_root) / "logs"),
                wandb_entity=self.config.wandb_entity,
                config=config_dict
            )
            print(f"[OK] 日志记录器已初始化（类型: {self.config.log_type}）")
    
    def train(self):
        """主训练循环"""
        # 环境设置
        self.setup_environment()
        
        # 客户端设置
        self.setup_clients()
        
        # 验证集设置
        self.setup_validation()
        
        # 日志设置
        self.setup_logging()
        self.training_history['run_metadata'] = self._collect_run_metadata()
        print(
            f"[RunMeta] git={self.training_history['run_metadata'].get('git_commit', 'unknown')}, "
            f"seed={self.training_history['run_metadata'].get('seed')}, device={self.device}, "
            f"protocol={self.training_history['run_metadata'].get('protocol_hash', 'unknown')[:12]}"
        )
        
        print("\n" + "=" * 60)
        print("开始 SegSAM 训练...")
        print("=" * 60)
        
        # 检查点恢复
        start_round = 1
        if self.config.resume_from or self.config.resume_from_checkpoint:
            start_round = self._resume_from_checkpoint()
        
        # Early Stopping 状态
        self.best_val_dice = 0.0
        _saved_best_dice = 0.0
        _patience_counter = 0
        _best_model_path = self.checkpoint_dir / "best_model.pth"

        # 主训练循环
        for round_num in range(start_round, self.config.rounds + 1):
            self._train_single_round(round_num)

            # 验证
            if self.val_loader is not None and (
                round_num % getattr(self.config, 'val_interval', 1) == 0 or round_num == 1
            ):
                self._evaluate_validation(round_num)

                # Early Stopping 逻辑
                if self.best_val_dice > _saved_best_dice:
                    _saved_best_dice = self.best_val_dice
                    _patience_counter = 0
                    # 保存最佳模型权重
                    try:
                        _best_model_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            self.global_model.state_dict(),
                            _best_model_path
                        )
                        print(f"  [Best] 新最佳 Val Dice={_saved_best_dice:.4f}，已保存 best_model.pth")
                    except Exception as _e:
                        print(f"  [WARN] best_model 保存失败: {_e}")
                else:
                    _patience_counter += 1
                    print(f"  [EarlyStopping] patience={_patience_counter}/{_EARLY_STOP_PATIENCE}，当前最佳 Dice={_saved_best_dice:.4f}")
                    if _patience_counter >= _EARLY_STOP_PATIENCE:
                        print(f"\n🚨 [Early Stopping] 触发！连续 {_EARLY_STOP_PATIENCE} 次验证无提升，停止训练。")
                        print(f"   最佳 Val Dice = {_saved_best_dice:.4f}（已保存至 {_best_model_path}）")
                        break

            # 检查点保存
            if self.config.checkpoint_interval > 0 and round_num % self.config.checkpoint_interval == 0:
                print(f"\n  [检查点] 保存检查点（第 {round_num} 轮）...")
                try:
                    self.save_checkpoint(round_num)
                except Exception as e:
                    print(f"  [FAIL] 检查点保存失败: {e}")
                    traceback.print_exc()

            # 绘制训练曲线
            if round_num == 1 or round_num == self.config.rounds or round_num % _VAL_PLOT_INTERVAL == 0:
                try:
                    self.plot_training_curves(round_num)
                except Exception as e:
                    print(f"  [FAIL] 绘图失败: {e}")

        # 训练完成
        self._finalize_training()

        return 0



    def _train_single_round(self, round_num: int):
        """执行单轮联邦训练"""
        _round_start_time = time.time()
        # 重置 GPU 显存峰值计数器（仅在 CUDA 可用时有效）
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"\n{'=' * 60}")
        print(f"Round {round_num}/{self.config.rounds}")
        print(f"{'=' * 60}")
        
        round_client_updates = {}
        round_client_reps = {}
        round_client_stats = {}
        round_client_grad_vectors = {}
        fedprox_mode = str(getattr(self.config, 'baseline_method', 'none')).lower() == 'fedprox'
        round_global_reference_state = None
        if fedprox_mode:
            round_global_reference_state = self.get_trainable_state_dict(self.global_model)
            print(
                f"[Baseline] FedProx active: mu={getattr(self.config, 'fedprox_mu', 0.0):.4f}, "
                f"shared_round_global_keys={len(round_global_reference_state)}"
            )
        
        # === 串行训练：逐个客户端训练 ===
        for client_idx, (client_id, cfg) in enumerate(self.client_configs.items(), 1):
            print(f"\n[Client {client_idx}/{len(self.client_configs)}] Training {client_id} ({cfg['modality']})...")
            
            # Step 1: 加载客户端状态到全局模型（GPU）
            print(f"  [1/5] Loading {client_id} state to GPU...")
            state_to_load = self.client_states[client_id]['weights']
            if fedprox_mode and round_global_reference_state is not None:
                state_to_load = round_global_reference_state
                print("      [FedProx] Using round-global snapshot as client init")
            self.load_trainable_state_dict(
                self.global_model,
                state_to_load,
                device=self.device
            )
            self.global_model.to(self.device)
            
            # Step 2: 动态裁剪优化器（根据模态构建参数列表）
            # ★ 精准路由（2026-03-14重构）：不同模态的客户端只优化自己负责的参数，
            # 彻底消除 backward 后无关参数出现 None 梯度的警告与 AMP 崩溃。
            print(f"  [2/5] Creating AdamW optimizer (modality-aware)...")
            client_modality = cfg['modality']
            if client_modality == 'text_only':
                # 文本客户端：仅优化文本投影层，绝不包含图像/分割参数
                opt_params = list(self.global_model.fusion_head.text_proj.parameters())
                print(f"      [text_only] optimizer params = fusion_head.text_proj only")
            elif client_modality == 'image_only':
                # 图像客户端：从全量可训练参数中剔除 fusion_head.text_proj 的参数
                text_proj_param_ids = {
                    id(p) for p in self.global_model.fusion_head.text_proj.parameters()
                }
                opt_params = [
                    p for p in self.global_model.get_trainable_params()
                    if id(p) not in text_proj_param_ids
                ]
                print(f"      [image_only] optimizer params = get_trainable_params() - text_proj")
            else:
                # multimodal：传入完整的可训练参数（原默认行为）
                opt_params = list(self.global_model.get_trainable_params())
                print(f"      [multimodal] optimizer params = get_trainable_params() (full)")
            # ★ 痛点3-a 修复（2026-03-17）：Adapter/PEFT 参数强制 lr=1e-4
            # 诊断：Logits 飙到 82.45 是因为 Adapter 参数随视觉主干 lr 一起爆炸更新。
            # 修复：将 opt_params 拆分为三组：
            #   - medical_seg_head：Zero-Init 冷启动需要 1e-3 暴力唤醒
            #   - adapter_* / lora_* 等 PEFT 参数：强制 lr=1e-4 防过拟合
            #   - 其余可训练参数（mask_decoder 主干等）：使用 config.lr
            _SEG_HEAD_LR    = _SEG_HEAD_LR_INIT
            _ADAPTER_LR     = _ADAPTER_LR_MAX
            _SEG_HEAD_KW    = ('medical_seg_head',)
            _PEFT_KEYWORDS  = ('adapter', 'lora', 'text_adapter')
            # 用 named_parameters 获取模型名称，以便按关键词分组
            _all_named_params = dict(self.global_model.named_parameters())
            _seg_head_param_ids = set()
            _peft_param_ids     = set()
            for _name, _p in self.global_model.named_parameters():
                if not _p.requires_grad:
                    continue
                if any(kw in _name.lower() for kw in _SEG_HEAD_KW):
                    _seg_head_param_ids.add(id(_p))
                elif any(kw in _name.lower() for kw in _PEFT_KEYWORDS):
                    _peft_param_ids.add(id(_p))
            # 筛选当前模态 opt_params 中三类参数
            _seg_head_params = [p for p in opt_params if id(p) in _seg_head_param_ids]
            _peft_params     = [p for p in opt_params if id(p) in _peft_param_ids]
            _main_params     = [p for p in opt_params if id(p) not in _seg_head_param_ids and id(p) not in _peft_param_ids]
            _param_groups = []
            if _seg_head_params:
                _param_groups.append({
                    'params': _seg_head_params,
                    'lr': _SEG_HEAD_LR,
                    'initial_lr': _SEG_HEAD_LR,
                    'weight_decay': 1e-2,
                })
                print(f"      [LR] Seg Head params:     {len(_seg_head_params)} params → lr={_SEG_HEAD_LR:.0e} (Zero-Init 唤醒, cosine 调度)")
            if _peft_params:
                _param_groups.append({
                    'params': _peft_params,
                    'lr': _ADAPTER_LR,
                    'initial_lr': _ADAPTER_LR,
                })
                print(f"      [LR] PEFT/Adapter params: {len(_peft_params)} params → lr={_ADAPTER_LR:.0e} (cosine 调度)")
            if _main_params:
                _param_groups.append({
                    'params': _main_params,
                    'lr': self.config.lr,
                    'initial_lr': self.config.lr,
                })
                print(f"      [LR] Main params:         {len(_main_params)} params → lr={self.config.lr:.2e} (可调度)")
            if not _param_groups:
                _param_groups = [{'params': opt_params, 'lr': self.config.lr, 'initial_lr': self.config.lr}]
            optimizer = torch.optim.AdamW(_param_groups, weight_decay=1e-2)

            # LR 调度器：计算 cosine 衰减比例，各参数组按 initial_lr × cosine_factor 更新
            # seg_head 1e-3 → ~1e-5，main params 5e-5 → 1e-6，后期 LR 比值保持在 10:1
            _lr_base   = self.config.lr
            _lr_min    = getattr(self.config, 'lr_min', 1e-6)
            _warmup    = getattr(self.config, 'lr_warmup_rounds', 0)
            _scheduler = getattr(self.config, 'lr_scheduler', 'none')

            if _warmup > 0 and round_num <= _warmup:
                _cosine_factor = round_num / _warmup
            elif _scheduler == 'cosine':
                _progress = (round_num - _warmup) / max(self.config.rounds - _warmup, 1)
                _cosine_factor = _lr_min / _lr_base + 0.5 * (1 - _lr_min / _lr_base) * (1 + math.cos(math.pi * _progress))
            elif _scheduler == 'linear':
                _progress = round_num / max(self.config.rounds, 1)
                _cosine_factor = 1.0 - _progress * (1 - _lr_min / _lr_base)
            else:
                _cosine_factor = 1.0

            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * _cosine_factor


            
            #恢复优化器状态（如果存在）
            if (not fedprox_mode) and self.client_states[client_id]['opt_state'] is not None:
                try:
                    optimizer.load_state_dict(self.client_states[client_id]['opt_state'])
                    print(f"      ✓ Optimizer state restored")
                except Exception as e:
                    print(f"      ⚠ Failed to restore optimizer state: {e}")
            
            # Step 3: 训练
            print(f"  [3/5] Training...")
            trainer = self.client_trainers[client_id]

            try:
                # 准备全局表示：用真实前向传播的 Proxy 替换随机 EMA 噪声
                public_loader = cfg.get('public_loader', None)
                disable_global_rep_update = bool(
                    getattr(self.config, 'disable_global_rep_update', False)
                )
                if self.config.lambda_cream > 0 and disable_global_rep_update:
                    frozen_global_reps = self.server.get_global_reps()
                    global_img_rep = frozen_global_reps.get('global_image_rep')
                    global_text_rep = frozen_global_reps.get('global_text_rep')
                    print("      [Ablation] Global representation refresh disabled; using frozen server reps")
                elif public_loader is not None and self.config.lambda_cream > 0:
                    global_img_rep, global_text_rep = self.server.generate_and_dispatch_global_proxies(
                        self.global_model, public_loader, self.device
                    )
                else:
                    global_img_rep, global_text_rep = None, None
                global_reps = {
                    'global_image_rep': global_img_rep,
                    'global_text_rep': global_text_rep,
                }

                # ★ 核心修复：所有客户端统一使用 trainer.run() 训练
                # client.py Phase 2 已通过多态分发解决模态逻辑：
                # - TextOnlyTrainer: 计算文本对比学习损失，不做分割
                # - ImageOnlyTrainer: 分割 + Cream Loss（无文本特征）
                # - MultimodalTrainer: 分割 + Cream Loss（含文本特征）
                #
                # ★ Fix Critical 2: 删除不存在的 client_modality 参数
                # trainer.run() 签名为 (model, optimizer, global_reps, lambda_cream)
                # ★ Logits 监控：将当前联邦轮次注入模型，供 forward() 内监控代码使用
                self.global_model._monitor_epoch = round_num
                if cfg['modality'] == 'multimodal':
                    print(f"      [LossRule] total = seg + ({self.config.lambda_cream:.4f} * cream)")
                elif cfg['modality'] == 'image_only':
                    print("      [LossRule] total = seg (cream monitor only)")
                elif cfg['modality'] == 'text_only':
                    print("      [LossRule] total = cream (+prox if enabled), lambda_cream ignored by design")
                if fedprox_mode:
                    print(f"      [Baseline] FedProx enabled (mu={getattr(self.config, 'fedprox_mu', 0.0):.4f})")

                updated_weights, img_rep, txt_rep, stats = trainer.run(
                    model=self.global_model,
                    optimizer=optimizer,
                    global_reps=global_reps,
                    lambda_cream=self.config.lambda_cream,
                    global_reference_state=round_global_reference_state
                )
                grad_vector = getattr(trainer, '_last_adapter_grad_vector', None)
                round_client_grad_vectors[client_id] = (
                    grad_vector.detach().cpu() if grad_vector is not None else None
                )

                # 合并图像和文本表征为单一 local_reps（用于聚合）
                # 优先使用图像表征，如果不存在则使用文本表征
                local_reps = img_rep if img_rep is not None else txt_rep

                # ★ Fix High 3 (2026-03-13): 保留客户端训练结果
                # 原来把 updated_weights 覆盖为全局模型状态，导致客户端梯度更新被丢弃，
                # 每轮聚合的都是全局模型自身参数，联邦学习实质上空跑。
                # 修复：updated_weights 已由 client.get_model_state() 过滤
                # （只含 requires_grad=True 参数，排除 RoPE buffer），直接使用即可。
                # 此处只做防御性的 RoPE key 二次过滤，确保不含任何 buffer。
                if updated_weights is not None:
                    _rope_filter = {'freqs_cis', 'freqs_cos', 'freqs_sin', 'relative_coords'}
                    updated_weights = {
                        k: v for k, v in updated_weights.items()
                        if not any(rk in k for rk in _rope_filter)
                    }

                print(
                    f"      Loss: {stats.get('avg_loss', 0):.4f}, "
                    f"Seg: {stats.get('avg_seg_loss', 0):.4f}, "
                    f"Cream: {stats.get('avg_cream_loss', 0):.4f}, "
                    f"NonSegComponent: {stats.get('avg_non_seg_component', 0):.4f}"
                )
                print(f"      Batches: {stats.get('num_batches', 0)}, LR: {self.config.lr * _cosine_factor:.2e}")

            except Exception as e:
                logging.error(f"Training failed with error:\n{traceback.format_exc()}")
                # 重新抛出，让错误可见
                raise
            
            # Step 4: 保存客户端状态回 CPU（内存安全）
            print(f"  [4/5] Saving state to CPU cache...")

            # ★ 修复 (2026-03-14): 统一保存所有模态返回的有效权重
            # 原逻辑把 text_only 的权重强制置 None，导致 text_proj 训练更新永远丢失，
            # 全局文本对齐能力永远在原地踏步 —— 这是逻辑闭环断裂的致命漏洞。
            # 修复：无论什么模态，只要 updated_weights 非空，就写入 CPU 缓存。
            # text_only 客户端返回的 text_only_state 里只含 text_proj 相关键，
            # 上传给 Server 后聚合代码会按键名做局部覆盖，其余参数保持全局值不变。
            if updated_weights is not None:
                self.client_states[client_id]['weights'] = {k: v.cpu() for k, v in updated_weights.items()}
                if cfg['modality'] == 'text_only':
                    print(f"      [text_only] Saved text-related weights (e.g., fusion_head.text_proj)")
            else:
                # 异常情况防御：所有模态都不应该返回 None 权重
                print(f"      [Warning] {client_id} ({cfg['modality']}) returned None weights, falling back to global model")
                self.client_states[client_id]['weights'] = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

            if fedprox_mode:
                self.client_states[client_id]['opt_state'] = None
            else:
                self.client_states[client_id]['opt_state'] = optimizer.state_dict()

            # 保存本地表征（用于聚合）
            if local_reps is not None:
                self.client_states[client_id]['local_reps'] = local_reps.cpu() if local_reps.device.type != 'cpu' else local_reps
            else:
                # 理论上不应该发生，因为每个客户端至少应该有一个表征
                raise ValueError(f"客户端 {client_id} 没有返回任何表征（img_rep 和 txt_rep 均为 None）")

            # Step 5: 收集更新
            print(f"  [5/5] Collecting updates...")

            # ★ 修复 (2026-03-14): 所有客户端统一上传各自的权重字典
            # text_only 上传只含 text_proj 的局部字典 → Server FedAvg 按键名局部覆盖
            # image_only 上传不含 text_proj 的字典 → Server 补全 text_proj 用全局值
            # multimodal 上传完整字典
            # Server 端 aggregate_weights 已有按键名补全机制，可安全处理局部字典
            round_client_updates[client_id] = self.client_states[client_id]['weights']
            print(f"      [OK] Collected weights and representations from {client_id}")

            round_client_reps[client_id] = self.client_states[client_id]['local_reps']
            round_client_stats[client_id] = stats
            
            # 清理 GPU 内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"  [OK] {client_id} completed")
        
        # === 服务器聚合 ===
        # ★ 修复 (2026-03-14): 所有客户端均上传各自的权重字典
        #   text_only   → 只含 text_proj 的局部字典
        #   image_only  → 不含 text_proj 的局部字典
        #   multimodal  → 完整字典
        # aggregate_weights 会用全局模型 state_dict 补全缺失键，再做 FedAvg

        # 统计有效的权重上传数量（所有模态都应非 None）
        valid_weight_clients = {cid: w for cid, w in round_client_updates.items() if w is not None}
        num_valid_weights = len(valid_weight_clients)
        num_total_clients = len(round_client_updates)

        print(f"\n[Aggregation] Collected {num_total_clients} client updates:")
        print(f"  - With model weights: {num_valid_weights} clients (will participate in FedAvg)")
        if num_total_clients - num_valid_weights > 0:
            print(f"  - ⚠ {num_total_clients - num_valid_weights} client(s) returned None weights (unexpected)")

        # 使用 sorted() 确保严格的确定性
        client_ids_sorted = sorted(round_client_updates.keys())

        # ★ 修复A：解耦聚合模态路由——必须从 client_configs（已过滤的实验子集）提取
        # 原Bug：从 config.clients（配置全集）提取，SegSAM image-only 过滤后 ID 错位导致 client_modalities=None
        # 修复：直接从 self.client_configs 建立映射，严格对应 client_ids_sorted
        # 同时移除 use_decoupled_agg 条件守卫：单模态实验同样需要解耦路由保证视觉参数进入聚合
        client_modality_map = {cid: cfg['modality'] for cid, cfg in self.client_configs.items()}
        client_modalities = [client_modality_map.get(cid, 'image_only') for cid in client_ids_sorted]
        print(f"  [Decoupled Agg] 客户端模态（来源: client_configs）: {client_modalities}")

        # 探针 modality 独立保留，不受路由开关影响
        grad_conflict_deg = compute_gradient_conflict_from_vectors(
            [round_client_grad_vectors[cid] for cid in client_ids_sorted],
            client_modalities,
        )
        self.server._last_grad_conflict_deg = grad_conflict_deg
        probe_modalities = client_modalities

        # use_decoupled_agg=False → 关闭路由白名单，text_proj 参与全量聚合（Group B 文本污染实验）
        if not self.config.use_decoupled_agg:
            client_modalities = None
            self.server._last_grad_conflict_deg = compute_gradient_conflict_from_vectors(
                [round_client_grad_vectors[cid] for cid in client_ids_sorted],
                probe_modalities
            )

        aggregated_state = self.server.aggregate_weights(
            [round_client_updates[cid] for cid in client_ids_sorted],
            [round_client_reps[cid] for cid in client_ids_sorted],
            client_modalities=client_modalities  # ★ 新增参数
        )


        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_state, strict=False)
        
        # 获取更新后的全局表示
        updated_global_reps = self.server.get_global_reps()
        
        # 计算平均训练损失
        client_losses = [s.get('avg_loss', 0.0) for s in round_client_stats.values()]
        client_seg_losses = [s.get('avg_seg_loss', 0.0) for s in round_client_stats.values()]
        client_cream_losses = [s.get('avg_cream_loss', 0.0) for s in round_client_stats.values()]
        
        avg_train_loss = sum(client_losses) / len(client_losses) if len(client_losses) > 0 else 0.0
        avg_seg_loss = sum(client_seg_losses) / len(client_seg_losses) if len(client_seg_losses) > 0 else 0.0
        avg_cream_loss = sum(client_cream_losses) / len(client_cream_losses) if len(client_cream_losses) > 0 else 0.0
        
        # 日志记录
        print(f"  ✓ Round {round_num} Summary:")
        print(f"    Avg Loss: {avg_train_loss:.4f} (Seg: {avg_seg_loss:.4f}, Cream: {avg_cream_loss:.4f})")
        
        # 记录到日志系统
        if self.logger is not None:
            log_metrics = {
                'Train_Loss': avg_train_loss,
                'Seg_Loss': avg_seg_loss,
                'Cream_Loss': avg_cream_loss,
            }
            # 如果存在最后一次验证指标，也一并记录
            if self.last_val_metrics:
                val_dice = self.last_val_metrics.get('dice', 0.0)
                val_iou = self.last_val_metrics.get('iou', 0.0)
                val_hd95 = self.last_val_metrics.get('hd95', float('inf'))
                log_metrics['Val_Dice'] = val_dice
                log_metrics['Val_IoU'] = val_iou
                if val_hd95 != float('inf') and not np.isnan(val_hd95):
                    log_metrics['Val_HD95'] = val_hd95
            self.logger.log(log_metrics, step=round_num)
        
        # 记录训练历史
        self.training_history['rounds'].append(round_num)
        self.training_history['avg_losses'].append(avg_train_loss)
        self.training_history['avg_seg_losses'].append(avg_seg_loss)
        self.training_history['avg_cream_losses'].append(avg_cream_loss)
        self.training_history['client_losses'].append({
            client_id: stat.get('avg_loss', 0.0) for client_id, stat in round_client_stats.items()
        })
        self.training_history['global_text_rep_norms'].append(
            updated_global_reps['global_text_rep'].norm().item()
        )
        self.training_history['global_image_rep_norms'].append(
            updated_global_reps['global_image_rep'].norm().item()
        )
        # ★ 论文数据收集：LR / 显存 / 耗时 / 梯度冲突
        self.training_history['lr_history'].append(float(self.config.lr * _cosine_factor))
        _peak_mem_mb = 0.0
        if torch.cuda.is_available():
            _peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        self.training_history['gpu_mem_mb'].append(round(_peak_mem_mb, 2))
        _elapsed = time.time() - _round_start_time
        self.training_history['round_time_sec'].append(round(_elapsed, 2))
        # grad_conflict_deg 由真实 adapter 梯度向量计算后写入 server 缓存
        # 此处仅读取缓存的最新值（SegSAM image-only 无多模态时为 None）
        _conflict = getattr(self.server, '_last_grad_conflict_deg', None)
        self.training_history['grad_conflict_deg'].append(_conflict)
        print(f"  [PaperData] Round {round_num}: LR={self.config.lr * _cosine_factor:.2e}, "
              f"GPU峰值={_peak_mem_mb:.0f}MB, 耗时={_elapsed:.1f}s, "
              f"GradConflict={f'{_conflict:.1f}°' if _conflict is not None else 'N/A'}")
        
        # 详细总结（每10轮或第1轮）
        if round_num % 10 == 0 or round_num == 1:
            print(f"\n  === Round {round_num} Detailed Summary ===")
            print(f"    Training Loss: {avg_train_loss:.4f}")
            print(f"    Seg Loss: {avg_seg_loss:.4f}")
            print(f"    Cream Loss: {avg_cream_loss:.4f}")
            print(f"    Global Text Rep Norm: {updated_global_reps['global_text_rep'].norm().item():.4f}")
            print(f"    Global Image Rep Norm: {updated_global_reps['global_image_rep'].norm().item():.4f}")
            print(f"    Participating Clients: {len(round_client_updates)}")
            # 显示每个客户端的损失
            for client_id, stats in round_client_stats.items():
                print(f"      * {client_id}: Loss={stats.get('avg_loss', 0):.4f}, Seg={stats.get('avg_seg_loss', 0):.4f}, Cream={stats.get('avg_cream_loss', 0):.4f}")
    
    def _evaluate_validation(self, round_num: int):
        """验证集评估"""
        print(f"\n[Validation] Evaluating on validation set...")
        try:
            # 使用全局模型进行验证
            verbose_diagnosis = (round_num == 1 or round_num % 50 == 0)
            
            # 使用任意一个 trainer 的 validate 方法
            any_trainer = list(self.client_trainers.values())[0]
            val_metrics = any_trainer.validate(
                model=self.global_model,
                test_loader=self.val_loader,
                compute_hd95=True,
                verbose=verbose_diagnosis
            )
            
            print(f"  Dice: {val_metrics.get('dice', 0):.4f}, IoU: {val_metrics.get('iou', 0):.4f}")
            if 'hd95' in val_metrics and val_metrics['hd95'] != float('inf'):
                print(f"  HD95: {val_metrics['hd95']:.2f} mm")
            
            # 记录验证指标到日志系统
            if self.logger is not None:
                val_dice = val_metrics.get('dice', 0.0)
                val_iou = val_metrics.get('iou', 0.0)
                val_hd95 = val_metrics.get('hd95', float('inf'))
                log_metrics = {
                    'Val_Dice': val_dice,
                    'Val_IoU': val_iou,
                }
                if val_hd95 != float('inf') and not np.isnan(val_hd95):
                    log_metrics['Val_HD95'] = val_hd95
                self.logger.log(log_metrics, step=round_num)
            
            # 保存最后一次验证指标
            self.last_val_metrics = val_metrics.copy()

            # ★ 修复C：Early Stopping 追踪逻辑——必须用 MAX 判断，不能简单覆盖
            # 将 best_val_dice 的 MAX 判断移至此处，作为唯一可信的更新点
            _current_dice = val_metrics.get('dice', 0.0)
            if _current_dice > self.best_val_dice:
                self.best_val_dice = _current_dice
                print(f"  [EarlyStopping] ★ 新最佳 Val Dice={self.best_val_dice:.4f}（已更新 self.best_val_dice）")
            else:
                print(f"  [EarlyStopping] 当前 Dice={_current_dice:.4f} ≤ 历史最佳 Dice={self.best_val_dice:.4f}，不更新")

            self.training_history['val_metrics'].append({
                'round': round_num,
                'dice': val_metrics.get('dice', 0.0),
                'iou': val_metrics.get('iou', 0.0),
                'hd95': val_metrics.get('hd95', float('inf')),
                'val_loss': val_metrics.get('val_loss', 0.0)
            })

            # 显示验证集指标
            print(f"\n    Validation Metrics:")
            print(f"      Dice: {val_metrics.get('dice', 0.0):.4f}")
            print(f"      IoU: {val_metrics.get('iou', 0.0):.4f}")
            if 'hd95' in val_metrics and val_metrics['hd95'] != float('inf'):
                print(f"      HD95: {val_metrics['hd95']:.2f} mm")
            else:
                print(f"      HD95: N/A")
                
        except Exception as e:
            print(f"  ⚠ Validation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _resume_from_checkpoint(self) -> int:
        """从检查点恢复训练"""
        checkpoint_path = Path(self.config.resume_from or self.config.resume_from_checkpoint)
        
        if checkpoint_path.exists():
            try:
                print(f"\n从检查点恢复: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 加载服务器状态
                if 'server_state' in checkpoint:
                    self.server.load_state_dict(checkpoint['server_state'], strict=True)
                    print("  [OK] 服务器状态已恢复")
                
                # 获取恢复的轮数
                resume_round = checkpoint.get('round', 0)
                print(f"  [OK] 将从第 {resume_round + 1} 轮继续训练")
                
                # 加载训练历史
                self.training_history = checkpoint.get('training_history', self.training_history)
                print("  [OK] 训练历史已恢复")
                
                # 加载客户端状态缓存
                if 'client_states' in checkpoint and checkpoint['client_states'] is not None:
                    self.client_states = checkpoint['client_states']
                    print("  [OK] 客户端状态缓存已恢复")
                
                return resume_round + 1
                
            except Exception as e:
                print(f"  [FAIL] 检查点恢复失败: {e}")
                print("  将从头开始训练...")
                import traceback
                traceback.print_exc()
                return 1
        else:
            print(f"  ⚠ 检查点文件不存在: {checkpoint_path}")
            print("  将从头开始训练...")
            return 1
    
    def _finalize_training(self):
        """训练完成后的处理"""
        print("\n" + "=" * 60)
        print("SegSAM 训练完成！")
        print("=" * 60)
        
        # 最终模型统计
        print("\n最终全局模型统计:")
        final_model = self.server.get_global_model()
        total_params = sum(p.numel() for p in final_model.parameters())
        trainable_params = sum(p.numel() for p in final_model.get_trainable_params())
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数量: {trainable_params:,}")
        print(f"  - 冻结参数量: {total_params - trainable_params:,}")
        
        # 保存最终模型
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.checkpoint_dir / "final_model.pth"
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'global_text_rep': self.server.global_text_rep.cpu(),
            'global_image_rep': self.server.global_image_rep.cpu(),
        }, save_path)
        print(f"\n最终模型已保存到: {save_path}")
        
        # 保存最终检查点
        print("\n保存最终检查点...")
        try:
            final_checkpoint_path = self.save_checkpoint(self.config.rounds)
            print(f"最终检查点已保存到: {final_checkpoint_path}")
        except Exception as e:
            print(f"  [FAIL] 最终检查点保存失败: {e}")
        
        # 保存训练历史记录
        history_path = self.checkpoint_dir / "training_history.json"
        self.training_history['final_stats'] = {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'frozen_params': int(total_params - trainable_params),
            'total_rounds': self.config.rounds,
            'final_avg_loss': float(self.training_history['avg_losses'][-1]) if self.training_history['avg_losses'] else 0.0
        }
        self.training_history['training_time'] = datetime.now().isoformat()
        self.training_history.setdefault('run_metadata', {})
        self.training_history['run_metadata']['run_finished_at'] = datetime.now().isoformat()
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        run_meta_path = self.checkpoint_dir / "run_metadata.json"
        with open(run_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history.get('run_metadata', {}), f, indent=2, ensure_ascii=False)
        print(f"Run metadata saved to: {run_meta_path}")
        print(f"训练历史已保存到: {history_path}")
        
        # 最终验证集评估
        if self.val_loader is not None:
            print("\n" + "=" * 60)
            print("最终验证集评估")
            print("=" * 60)
            try:
                any_trainer = list(self.client_trainers.values())[0]
                final_val_metrics = any_trainer.validate(
                    model=self.global_model,
                    test_loader=self.val_loader,
                    compute_hd95=True,
                    verbose=True
                )
                
                print(f"\n最终评估指标:")
                print(f"  Dice 系数: {final_val_metrics.get('dice', 0.0):.4f}")
                print(f"  IoU: {final_val_metrics.get('iou', 0.0):.4f}")
                if 'hd95' in final_val_metrics and final_val_metrics['hd95'] != float('inf'):
                    print(f"  HD95: {final_val_metrics['hd95']:.2f} mm")
                else:
                    print(f"  HD95: N/A")
                
                # 保存分割掩码
                if self.config.save_masks:
                    print(f"\n保存分割掩码...")
                    mask_save_dir = self.checkpoint_dir / "segmentation_masks"
                    try:
                        self.save_segmentation_masks(
                            self.global_model,
                            self.val_loader,
                            mask_save_dir,
                            self.config.max_masks
                        )
                        print(f"  [OK] 分割掩码已保存到: {mask_save_dir}")
                    except Exception as e:
                        print(f"  [FAIL] 保存分割掩码失败: {e}")
                
                self.training_history['final_val_metrics'] = final_val_metrics
                
            except Exception as e:
                print(f"最终验证集评估失败: {e}")
                import traceback
                traceback.print_exc()
            print("=" * 60)
        
        # 记录最终总结到日志系统
        if self.logger is not None:
            summary = {}
            if self.training_history['avg_losses']:
                summary['final_train_loss'] = self.training_history['avg_losses'][-1]
                summary['initial_train_loss'] = self.training_history['avg_losses'][0]
            if self.training_history.get('avg_cream_losses'):
                summary['final_cream_loss'] = self.training_history['avg_cream_losses'][-1]
            if self.training_history['val_metrics']:
                best_dice = max(m['dice'] for m in self.training_history['val_metrics'])
                best_iou = max(m['iou'] for m in self.training_history['val_metrics'])
                summary['best_val_dice'] = best_dice
                summary['best_val_iou'] = best_iou
            if 'final_val_metrics' in self.training_history:
                final_metrics = self.training_history['final_val_metrics']
                summary['final_val_dice'] = final_metrics.get('dice', 0.0)
                if 'hd95' in final_metrics and final_metrics['hd95'] != float('inf'):
                    summary['final_val_hd95'] = final_metrics['hd95']
            self.logger.log_summary(summary)
            self.logger.close()
        
        # 打印训练总结
        print("\n" + "=" * 60)
        print("训练总结")
        print("=" * 60)
        if self.training_history['avg_losses']:
            print(f"初始损失: {self.training_history['avg_losses'][0]:.4f}")
            print(f"最终损失: {self.training_history['avg_losses'][-1]:.4f}")
        
        if self.training_history['val_metrics']:
            best_dice = max(m['dice'] for m in self.training_history['val_metrics'])
            best_dice_round = next(m['round'] for m in self.training_history['val_metrics'] if m['dice'] == best_dice)
            print(f"\n训练过程中的最佳验证指标:")
            print(f"  Dice 系数: {best_dice:.4f} (第 {best_dice_round} 轮)")
        
        print(f"\n文件保存位置:")
        print(f"  - 模型文件: {save_path}")
        print(f"  - 训练历史: {history_path}")
        print("=" * 60)

    def save_checkpoint(self, round_num: int) -> Path:
        """保存训练检查点"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'round': round_num,
            'server_state': self.server.get_state_dict(),
            'training_history': self.training_history,
            'client_states': self.client_states,
            'config': self.config.to_dict(),
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        print(f"  [OK] 检查点已保存: {checkpoint_path}")
        
        # 清理旧检查点
        keep_max = self.config.keep_max_checkpoints or self.config.keep_checkpoint_max
        if keep_max > 0:
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("checkpoint_round_*.pth"),
                key=lambda x: int(x.stem.split('_')[-1]),
                reverse=True
            )
            for old_checkpoint in checkpoint_files[keep_max:]:
                try:
                    old_checkpoint.unlink()
                    print(f"  [OK] 已删除旧检查点: {old_checkpoint.name}")
                except Exception as e:
                    print(f"  ⚠ 删除旧检查点失败 ({old_checkpoint.name}): {e}")
        
        return checkpoint_path
    
    def plot_training_curves(self, current_round: int):
        """生成并保存训练曲线图"""
        plot_dir = self.checkpoint_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[绘图] 正在生成训练曲线（第 {current_round} 轮）...")
        
        try:
            rounds_x = self.training_history['rounds']
            
            if not rounds_x:
                print("  ⚠ 没有训练数据可绘制")
                return
            
            # 1. 损失曲线
            plt.figure(figsize=(12, 6))
            plt.plot(rounds_x, self.training_history['avg_losses'], label='Train Loss', marker='o', linewidth=2)
            
            if self.training_history.get('avg_seg_losses'):
                plt.plot(rounds_x, self.training_history['avg_seg_losses'], label='Seg Loss', linestyle=':', marker='s', linewidth=1.5)
            
            if self.training_history.get('avg_cream_losses'):
                plt.plot(rounds_x, self.training_history['avg_cream_losses'], label='Cream Loss', linestyle='--', marker='^', linewidth=1.5)
            
            # 添加验证损失
            if self.training_history.get('val_metrics'):
                val_rounds = [m['round'] for m in self.training_history['val_metrics']]
                val_losses = [m.get('val_loss', 0.0) for m in self.training_history['val_metrics']]
                if val_losses and any(l > 0 for l in val_losses):
                    plt.plot(val_rounds, val_losses, label='Val Loss', marker='x', linestyle='-.', linewidth=2, color='red')
            
            plt.title(f'Training Loss Curves (Round {current_round})', fontsize=14, fontweight='bold')
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / 'loss_curve.png', dpi=150)
            plt.close()
            print(f"  [OK] 损失曲线已保存: {plot_dir / 'loss_curve.png'}")
            
            # 2. 验证指标曲线（Dice & IoU）
            if self.training_history['val_metrics']:
                val_rounds = [m['round'] for m in self.training_history['val_metrics']]
                val_dice = [m['dice'] for m in self.training_history['val_metrics']]
                val_iou = [m['iou'] for m in self.training_history['val_metrics']]
                
                plt.figure(figsize=(12, 6))
                plt.plot(val_rounds, val_dice, label='Dice Score', marker='s', linewidth=2, color='blue')
                plt.plot(val_rounds, val_iou, label='IoU Score', marker='^', linewidth=2, color='green')
                plt.title(f'Validation Metrics: Dice & IoU (Round {current_round})', fontsize=14, fontweight='bold')
                plt.xlabel('Round', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.ylim([0, 1.0])
                plt.legend(loc='best', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / 'metrics_dice_iou.png', dpi=150)
                plt.close()
                print(f"  [OK] Dice/IoU 曲线已保存: {plot_dir / 'metrics_dice_iou.png'}")
                
                # 3. HD95 曲线
                val_hd95 = [m['hd95'] if m['hd95'] != float('inf') else np.nan for m in self.training_history['val_metrics']]
                valid_hd95_indices = [i for i, v in enumerate(val_hd95) if not np.isnan(v)]
                
                if valid_hd95_indices:
                    plt.figure(figsize=(12, 6))
                    plt.plot(
                        [val_rounds[i] for i in valid_hd95_indices],
                        [val_hd95[i] for i in valid_hd95_indices],
                        label='HD95', color='red', marker='x', linewidth=2
                    )
                    plt.title(f'Validation Metric: HD95 (Round {current_round})', fontsize=14, fontweight='bold')
                    plt.xlabel('Round', fontsize=12)
                    plt.ylabel('HD95 (mm)', fontsize=12)
                    plt.legend(loc='best', fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(plot_dir / 'metrics_hd95.png', dpi=150)
                    plt.close()
                    print(f"  [OK] HD95 曲线已保存: {plot_dir / 'metrics_hd95.png'}")
                else:
                    print(f"  ⚠ 暂无有效的 HD95 数据，跳过 HD95 曲线")
                    
        except Exception as e:
            print(f"  [FAIL] 绘图失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_segmentation_masks(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        save_dir: Path,
        max_samples: int = 50
    ) -> Path:
        """保存分割掩码到文件"""
        save_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        
        saved_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if saved_count >= max_samples:
                    break
                    
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, masks = batch[0], batch[1]
                else:
                    images = batch
                    masks = None
                
                images = images.to(self.device)
                if masks is not None:
                    masks = masks.to(self.device)
                else:
                    continue
                
                # 前向传播
                pred = model(images)
                
                # 处理模型返回字典的情况
                if isinstance(pred, dict):
                    pred = pred.get('logits', pred.get('out', list(pred.values())[0]))
                
                # 处理每个样本
                batch_size = images.shape[0]
                for i in range(batch_size):
                    if saved_count >= max_samples:
                        break
                    
                    # 获取预测掩码（与验证口径一致：WT channel=1）
                    wt_channel_idx = 1
                    if pred.dim() == 4:
                        if pred.shape[1] > wt_channel_idx:
                            pred_probs = torch.sigmoid(pred[i, wt_channel_idx]).cpu().numpy()
                            pred_mask = (pred_probs > 0.5).astype(np.uint8)
                        elif pred.shape[1] == 1:
                            pred_probs = torch.sigmoid(pred[i, 0]).cpu().numpy()
                            pred_mask = (pred_probs > 0.5).astype(np.uint8)
                        else:
                            continue
                    else:
                        pred_probs = torch.sigmoid(pred[i]).cpu().numpy()
                        pred_mask = (pred_probs > 0.5).astype(np.uint8)

                    # 获取真实掩码（与验证口径一致：WT channel=1）
                    if masks.dim() == 4:
                        if masks.shape[1] > wt_channel_idx:
                            true_mask = (masks[i, wt_channel_idx].cpu().numpy() > 0.5).astype(np.uint8)
                        elif masks.shape[1] == 1:
                            true_mask = (masks[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
                        else:
                            continue
                    else:
                        true_mask = (masks[i].cpu().numpy() > 0.5).astype(np.uint8)
                    
                    # 转换为0-255的uint8格式
                    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                    true_mask_uint8 = (true_mask * 255).astype(np.uint8)
                    
                    # 保存预测掩码
                    pred_img = Image.fromarray(pred_mask_uint8, mode='L')
                    pred_path = save_dir / f"pred_mask_{saved_count:04d}.png"
                    pred_img.save(pred_path)
                    
                    # 保存真实掩码
                    true_img = Image.fromarray(true_mask_uint8, mode='L')
                    true_path = save_dir / f"true_mask_{saved_count:04d}.png"
                    true_img.save(true_path)
                    
                    saved_count += 1
        
        return save_dir

