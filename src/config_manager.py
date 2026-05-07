"""
SegSAM 图像分割配置管理器

此模块提供集中化配置管理，用于统一管理 SegSAM 图像分割训练参数。
通过使用数据类(dataclass)，确保配置的类型安全和可维护性。
"""

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Set
import torch


@dataclass
class FederatedConfig:
    """
    SegSAM 图像分割训练配置类
    
    此配置类包含了SegSAM 图像分割训练过程中的所有超参数和设置。
    所有参数都有合理的默认值，可以通过命令行参数或配置文件覆盖。
    
    参数分组：
    - 数据相关: data_root, max_samples
    - 训练超参数: rounds, batch_size, lr, local_epochs, lambda_cream
    - 模型架构: img_size, embed_dim, num_heads
    - 系统设置: device, use_amp, use_mock
    - 路径配置: sam3_checkpoint, checkpoint_dir, output_dir
    - 聚合方法: aggregation_method
    """
    
    # ==================== 数据相关 ====================
    data_root: str = "data/federated_split"
    """数据根目录路径，包含 SegSAM 训练数据"""
    
    max_samples: Optional[int] = None
    """每个客户端的最大样本数，用于快速测试（None表示使用全部数据）"""
    
    num_clients: Optional[int] = None
    """训练客户端数量（None 表示使用配置中的所有客户端）"""
    
    # ==================== 训练超参数 ====================
    rounds: int = 50
    """SegSAM 训练轮数"""
    
    batch_size: int = 4
    """训练批次大小"""
    
    lr: float = 1e-4
    """学习率（Learning Rate）"""

    lr_scheduler: str = "none"
    """学习率调度器类型：cosine | linear | step | none"""

    lr_warmup_rounds: int = 0
    """学习率 Warmup 轮数（前 N 轮从 lr_min 线性增长到 lr）"""

    lr_min: float = 1e-6
    """Cosine 衰减的最小学习率下界"""

    accumulation_steps: int = 1
    """梯度累加步数（1 = 无累加）；有效 batch = batch_size × accumulation_steps"""

    local_epochs: int = 1
    """客户端本地训练的轮数"""
    
    lambda_cream: float = 0.05  # Step2: 0.02 → 0.05
    """CREAM对比损失的权重系数（★ Fix: 0.1 → 0.02，让 seg_loss 主导 total_loss）"""
    
    # ==================== 模型架构参数 ====================
    img_size: int = 256
    """输入图像的尺寸（正方形，单边长度）"""
    
    embed_dim: int = 768
    """嵌入向量的维度"""
    
    num_heads: Optional[int] = None
    """Transformer注意力头的数量（None表示自动计算）"""

    num_classes: int = 1
    """分割输出通道数（WT 二分类=1；BraTS 3 区域=3）"""

    text_dim: int = 768
    """文本编码器输出维度（BERT-base=768），传给 MultimodalFusionHead.text_proj"""

    grad_clip: float = 1.0
    seed: int = 3407
    """梯度裁剪阈値（0.0 表示不裁剪）"""
    
    # ==================== 系统设置 ====================
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    """训练设备（'cuda' 或 'cpu'，自动检测）"""
    
    use_amp: bool = True
    """是否使用自动混合精度训练（Automatic Mixed Precision）"""
    
    use_mock: bool = False
    """是否使用Mock SAM3模型（用于测试或当真实权重不可用时）"""
    
    
    # ==================== 路径配置 ====================
    sam3_checkpoint: str = "data/checkpoints/sam3.pt"
    """SAM3预训练权重文件路径"""
    
    checkpoint_dir: str = "checkpoints/federated"
    """训练检查点保存目录"""
    
    output_dir: str = "outputs/federated"
    """训练输出（日志、图表等）保存目录"""
    
    # ==================== 聚合方法 ====================
    aggregation_method: str = "contrastive_weighted"
    """
    联邦聚合方法，可选值：
    - 'fedavg': 标准联邦平均
    - 'contrastive_weighted': CREAM对比加权聚合
    """
    
    global_rep_alpha: float = 0.9
    """全局表示的更新权重（EMA系数）"""

    disable_global_rep_update: bool = False

    strict_aggregation_guard: bool = False
    """是否在聚合发现关键参数缺失时立即失败"""

    proxy_k_batches: int = 1
    """生成全局 proxy 时使用的公共批次数量（>=1）"""

    baseline_method: str = "none"
    """外部 baseline 方法：none | fedprox"""

    fedprox_mu: float = 0.0
    """FedProx 近端约束系数；仅在 baseline_method=fedprox 时生效"""

    # ==================== 联邦学习客户端配置 ====================
    use_decoupled_agg: bool = False
    """是否使用解耦功能聚合（Decoupled Aggregation）"""

    clients: Optional[list] = None
    """客户端配置列表，每个元素包含 client_id, modality, data_source 等字段"""

    strict_protocol_check: bool = True
    """是否启用实验协议硬校验（Fail-Fast）"""

    # ==================== 检查点管理 ====================
    checkpoint_interval: int = 10
    """检查点保存间隔（每隔多少轮保存一次）"""
    
    keep_max_checkpoints: int = 5
    """最多保留的检查点数量（0表示保留所有）"""
    
    keep_checkpoint_max: int = 5  # 兼容性别名
    """最多保留的检查点数量（keep_max_checkpoints的别名，保持向后兼容）"""
    
    resume_from: Optional[str] = None
    """从指定检查点恢复训练（检查点文件路径）"""
    
    resume_from_checkpoint: Optional[str] = None  # 兼容性别名
    """从指定检查点恢复训练（resume_from的别名，保持向后兼容）"""
    
    # ==================== 验证与评估 ====================
    val_interval: int = 5
    """验证间隔（每隔多少轮进行一次验证）"""
    
    save_masks: bool = False
    """是否保存分割掩码图像"""
    
    max_masks: int = 50
    """最多保存的掩码图像数量"""
    
    # ==================== 日志配置 ====================
    log_type: str = "tensorboard"
    """
    日志记录类型，可选值：
    - 'tensorboard': 使用TensorBoard
    - 'wandb': 使用Weights & Biases
    - 'both': 同时使用两者
    - 'none': 不记录日志
    """
    
    log_dir: Optional[str] = None
    """日志保存目录（None表示使用data_root/logs）"""
    
    experiment_name: Optional[str] = None
    """实验名称（用于日志记录）"""
    
    wandb_project: str = "SegSAM"
    """WandB项目名称"""
    
    wandb_entity: Optional[str] = None
    """WandB实体名称（team或用户名）"""
    
    def __post_init__(self):
        """
        数据类初始化后的验证和处理
        
        执行以下检查：
        1. 验证参数的有效性
        2. 确保设备可用
        3. 转换路径为 Path 对象
        """
        # 验证参数范围
        if self.rounds <= 0:
            raise ValueError(f"rounds 必须大于 0，当前值: {self.rounds}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前值: {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr 必须大于 0，当前值: {self.lr}")
        if self.local_epochs <= 0:
            raise ValueError(f"local_epochs 必须大于 0，当前值: {self.local_epochs}")
        if self.lambda_cream < 0:
            raise ValueError(f"lambda_cream 不能为负数，当前值: {self.lambda_cream}")
        if self.img_size <= 0:
            raise ValueError(f"img_size 必须大于 0，当前值: {self.img_size}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim 必须大于 0，当前值: {self.embed_dim}")
        if self.proxy_k_batches <= 0:
            raise ValueError(f"proxy_k_batches 必须大于 0，当前值: {self.proxy_k_batches}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        allowed_baselines = {"none", "fedprox"}
        if self.baseline_method not in allowed_baselines:
            raise ValueError(
                f"baseline_method 无效: {self.baseline_method}，允许值: {sorted(allowed_baselines)}"
            )
        if self.fedprox_mu < 0:
            raise ValueError(f"fedprox_mu 不能为负数，当前值: {self.fedprox_mu}")
        if self.baseline_method == "fedprox" and self.fedprox_mu <= 0:
            raise ValueError(
                f"baseline_method=fedprox 时 fedprox_mu 必须大于 0，当前值: {self.fedprox_mu}"
            )
        if self.baseline_method != "fedprox" and abs(self.fedprox_mu) > 1e-12:
            raise ValueError(
                f"baseline_method={self.baseline_method} 时 fedprox_mu 必须为 0，当前值: {self.fedprox_mu}"
            )

        # 实验协议校验：客户端配置必须完整且可审计
        if self.clients is not None:
            if not isinstance(self.clients, list) or len(self.clients) == 0:
                raise ValueError("clients 必须是非空列表")

            allowed_modalities = {"text_only", "image_only", "multimodal"}
            seen_client_ids: Set[str] = set()
            enabled_count = 0

            for idx, client_cfg in enumerate(self.clients):
                if not isinstance(client_cfg, dict):
                    raise ValueError(f"clients[{idx}] 必须是字典，当前类型: {type(client_cfg).__name__}")

                client_id = str(client_cfg.get("client_id", "")).strip()
                modality = str(client_cfg.get("modality", "")).strip()
                enabled = bool(client_cfg.get("enabled", True))

                if not client_id:
                    raise ValueError(f"clients[{idx}] 缺少 client_id")
                if modality not in allowed_modalities:
                    raise ValueError(
                        f"clients[{idx}] 的 modality 无效: {modality}，"
                        f"允许值: {sorted(allowed_modalities)}"
                    )

                normalized_id = client_id.replace("_", "").lower()
                if normalized_id in seen_client_ids:
                    raise ValueError(f"检测到重复 client_id（忽略下划线后）: {client_id}")
                seen_client_ids.add(normalized_id)

                if enabled:
                    enabled_count += 1

            if enabled_count == 0:
                raise ValueError("clients 中至少需要一个 enabled=true 的客户端")
        
        # 确保 CUDA 可用时才使用
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[警告] CUDA 不可用，自动切换到 CPU")
            self.device = "cpu"
        
        # 如果使用 Mock 模式，不需要检查权重路径
        if not self.use_mock and not Path(self.sam3_checkpoint).exists():
            print(f"[警告] SAM3权重文件不存在: {self.sam3_checkpoint}")
            print(f"[提示] 请确保权重文件存在，或使用 --use_mock 标志")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        
        Returns:
            包含所有配置参数的字典
        """
        return asdict(self)
    
    def save(self, path: str) -> None:
        """
        保存配置到 JSON 文件
        
        Args:
            path: 保存路径
        """
        import json
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[OK] 配置已保存到: {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'FederatedConfig':
        """
        从 JSON 文件加载配置

        Args:
            path: 配置文件路径

        Returns:
            FederatedConfig 实例
        """
        import json
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {load_path}")

        with open(load_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        print(f"[OK] 配置已加载: {load_path}")
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> 'FederatedConfig':
        """
        从 YAML 文件加载配置

        Args:
            path: YAML配置文件路径

        Returns:
            FederatedConfig 实例
        """
        import yaml
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {load_path}")

        with open(load_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 展平嵌套的配置字典
        flattened = {}

        # 处理训练配置
        if 'training' in config_dict:
            training = config_dict['training']
            flattened['batch_size'] = training.get('batch_size', 4)
            flattened['lr'] = training.get('learning_rate', 1e-4)
            flattened['rounds'] = training.get('rounds', 50)
            flattened['local_epochs'] = training.get('local_epochs', 1)
            flattened['lr_scheduler'] = training.get('lr_scheduler', 'none')
            flattened['lr_warmup_rounds'] = training.get('lr_warmup_rounds', 0)
            flattened['lr_min'] = training.get('lr_min', 1e-6)
            flattened['accumulation_steps'] = training.get('accumulation_steps', 1)
            flattened['grad_clip'] = training.get('grad_clip', 1.0)

        # 处理CREAM配置
        if 'cream' in config_dict:
            cream = config_dict['cream']
            flattened['lambda_cream'] = cream.get('lambda_cream', 0.02)  # ★ Fix: 0.1 → 0.02

        # 处理模型配置
        if 'model' in config_dict:
            model = config_dict['model']
            flattened['img_size'] = model.get('img_size', 256)
            flattened['embed_dim'] = model.get('embed_dim', 768)
            flattened['num_heads'] = model.get('num_heads')
            flattened['num_classes'] = model.get('num_classes', 1)
            flattened['text_dim']    = model.get('text_dim', 768)

        # 处理服务器配置
        if 'server' in config_dict:
            server = config_dict['server']
            flattened['aggregation_method'] = server.get('aggregation_method', 'contrastive_weighted')
            flattened['global_rep_alpha'] = server.get('global_rep_alpha', 0.9)
            flattened['disable_global_rep_update'] = server.get('disable_global_rep_update', False)
            flattened['strict_aggregation_guard'] = server.get('strict_aggregation_guard', False)
            flattened['proxy_k_batches'] = server.get('proxy_k_batches', 1)

        # 处理 baseline 配置
        if 'baseline' in config_dict:
            baseline = config_dict['baseline']
            flattened['baseline_method'] = str(baseline.get('method', 'none')).lower()
            flattened['fedprox_mu'] = baseline.get('mu', 0.0)

        # 处理联邦学习配置
        if 'federated' in config_dict:
            federated = config_dict['federated']
            flattened['use_decoupled_agg'] = federated.get('use_decoupled_agg', False)
            flattened['clients'] = federated.get('clients', None)

        # 处理选项配置
        if 'options' in config_dict:
            options = config_dict['options']
            flattened['use_amp'] = options.get('use_amp', True)
            flattened['use_mock'] = options.get('use_dummy', False)  # use_dummy -> use_mock
            flattened['strict_protocol_check'] = options.get('strict_protocol_check', True)

        # 处理日志配置
        if 'logging' in config_dict:
            logging = config_dict['logging']
            flattened['log_type'] = logging.get('log_type', 'tensorboard')
            flattened['log_dir'] = logging.get('log_dir')
            flattened['experiment_name'] = logging.get('experiment_name')
            flattened['wandb_project'] = logging.get('wandb_project', 'SegSAM')
            flattened['wandb_entity'] = logging.get('wandb_entity')

        # 处理检查点配置
        if 'checkpoint' in config_dict:
            checkpoint = config_dict['checkpoint']
            flattened['checkpoint_interval'] = checkpoint.get('checkpoint_interval', 10)
            flattened['keep_max_checkpoints'] = checkpoint.get('keep_checkpoint_max', 5)
            flattened['resume_from'] = checkpoint.get('resume_from_checkpoint')

        # 处理验证配置
        if 'validation' in config_dict:
            validation = config_dict['validation']
            flattened['val_interval'] = validation.get('val_interval', 5)
            flattened['save_masks'] = validation.get('save_masks', False)
            flattened['max_masks'] = validation.get('max_masks', 50)

        # 处理顶层配置
        if 'data_root' in config_dict:
            flattened['data_root'] = config_dict['data_root']
        if 'device' in config_dict:
            flattened['device'] = config_dict['device']
        if 'seed' in config_dict:
            flattened['seed'] = config_dict['seed']
        if 'system' in config_dict and 'seed' in config_dict['system']:
            flattened['seed'] = config_dict['system']['seed']

        print(f"[OK] YAML配置已加载: {load_path}")
        return cls(**flattened)
    
    def __repr__(self) -> str:
        """
        字符串表示，用于打印配置摘要
        
        Returns:
            配置摘要字符串
        """
        return (
            f"FederatedConfig(\n"
            f"  Data: {self.data_root} (max_samples={self.max_samples})\n"
            f"  Training: rounds={self.rounds}, batch_size={self.batch_size}, "
            f"lr={self.lr}, local_epochs={self.local_epochs}, seed={self.seed}\n"
            f"  Model: img_size={self.img_size}, embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}\n"
            f"  Device: {self.device}, AMP={self.use_amp}, Mock={self.use_mock}\n"
            f"  Aggregation: {self.aggregation_method}, lambda_cream={self.lambda_cream}, "
            f"disable_global_rep_update={self.disable_global_rep_update}\n"
            f"  Baseline: {self.baseline_method}, fedprox_mu={self.fedprox_mu}\n"
            f")"
        )


def load_config_from_args(args: Optional[argparse.Namespace] = None) -> FederatedConfig:
    """
    从命令行参数加载配置
    
    此函数提供了向后兼容的接口，允许从命令行参数创建配置对象。
    如果没有提供参数，将使用默认值。
    
    Args:
        args: 命令行参数命名空间（如果为 None，则解析 sys.argv）
        
    Returns:
        FederatedConfig 实例
        
    示例:
        # 方式1: 直接解析命令行参数
        >>> config = load_config_from_args()
        
        # 方式2: 从现有的 args 对象创建
        >>> parser = create_argument_parser()
        >>> args = parser.parse_args()
        >>> config = load_config_from_args(args)
    """
    if args is None:
        parser = create_argument_parser()
        args = parser.parse_args()
    
    # 从 args 构建配置字典
    config_dict = {}
    
    # 映射所有参数
    arg_mapping = {
        'data_root': 'data_root',
        'rounds': 'rounds',
        'batch_size': 'batch_size',
        'lr': 'lr',
        'lambda_cream': 'lambda_cream',
        'local_epochs': 'local_epochs',
        'img_size': 'img_size',
        'embed_dim': 'embed_dim',
        'num_heads': 'num_heads',
        'max_samples': 'max_samples',
        'num_clients': 'num_clients',
        'use_mock': 'use_mock',
        'device': 'device',
        'sam3_checkpoint': 'sam3_checkpoint',
        'checkpoint_dir': 'checkpoint_dir',
        'output_dir': 'output_dir',
        'seed': 'seed',
        'aggregation_method': 'aggregation_method',
        'keep_max_checkpoints': 'keep_max_checkpoints',
        'resume_from': 'resume_from',
        'val_interval': 'val_interval',
        'save_masks': 'save_masks',
        'max_masks': 'max_masks',
    }
    
    for arg_name, config_name in arg_mapping.items():
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:  # 只设置非 None 的值
                config_dict[config_name] = value
    
    return FederatedConfig(**config_dict)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    
    此函数定义了所有支持的命令行参数，保持与原始训练脚本的兼容性。
    
    Returns:
        配置好的 ArgumentParser 实例
        
    示例:
        >>> parser = create_argument_parser()
        >>> args = parser.parse_args()
        >>> config = load_config_from_args(args)
    """
    parser = argparse.ArgumentParser(
        description="SegSAM 图像分割训练配置",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ==================== 数据相关 ====================
    data_group = parser.add_argument_group('数据配置')
    data_group.add_argument(
        '--data_root',
        type=str,
        default='data/federated_split',
        help='数据根目录路径'
    )
    data_group.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='每个客户端最大样本数（用于快速测试）'
    )
    data_group.add_argument(
        '--clients',
        type=int,
        default=None,
        dest='num_clients',
        help='训练客户端数量（默认使用所有可用客户端）'
    )
    
    # ==================== 训练超参数 ====================
    training_group = parser.add_argument_group('训练超参数')
    training_group.add_argument(
        '--rounds',
        type=int,
        default=50,
        help='SegSAM 训练轮数'
    )
    training_group.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='训练批次大小'
    )
    training_group.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='学习率'
    )
    training_group.add_argument(
        '--local_epochs',
        type=int,
        default=1,
        help='客户端本地训练轮数'
    )
    training_group.add_argument(
        '--lambda_cream',
        type=float,
        default=0.1,
        help='CREAM对比损失权重'
    )
    
    # ==================== 模型架构 ====================
    model_group = parser.add_argument_group('模型架构')
    model_group.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='输入图像尺寸'
    )
    model_group.add_argument(
        '--embed_dim',
        type=int,
        default=768,
        help='嵌入向量维度'
    )
    model_group.add_argument(
        '--num_heads',
        type=int,
        default=None,
        help='Transformer注意力头数（None表示自动计算）'
    )
    
    # ==================== 系统设置 ====================
    system_group = parser.add_argument_group('系统设置')
    system_group.add_argument(
        '--device',
        type=str,
        default=None,
        help="训练设备 (e.g., 'cuda', 'cpu')"
    )
    system_group.add_argument(
        '--use_mock',
        action='store_true',
        help='使用Mock SAM3模型（用于测试）'
    )
    system_group.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='Random seed'
    )
    
    # ==================== 路径配置 ====================
    path_group = parser.add_argument_group('路径配置')
    path_group.add_argument(
        '--sam3_checkpoint',
        type=str,
        default='data/checkpoints/sam3.pt',
        help='SAM3预训练权重路径'
    )
    path_group.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints/federated',
        help='训练检查点保存目录'
    )
    path_group.add_argument(
        '--output_dir',
        type=str,
        default='outputs/federated',
        help='训练输出保存目录'
    )
    
    # ==================== 聚合方法 ====================
    agg_group = parser.add_argument_group('聚合配置')
    agg_group.add_argument(
        '--aggregation_method',
        type=str,
        default='contrastive_weighted',
        choices=['fedavg', 'contrastive_weighted'],
        help='联邦聚合方法'
    )
    
    # ==================== 检查点管理 ====================
    checkpoint_group = parser.add_argument_group('检查点管理')
    checkpoint_group.add_argument(
        '--keep_max_checkpoints',
        type=int,
        default=5,
        help='最多保留的检查点数量（0表示保留所有）'
    )
    checkpoint_group.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='从指定检查点恢复训练（检查点文件路径）'
    )
    
    # ==================== 验证与评估 ====================
    eval_group = parser.add_argument_group('验证与评估')
    eval_group.add_argument(
        '--val_interval',
        type=int,
        default=5,
        help='验证间隔（每隔多少轮进行一次验证）'
    )
    eval_group.add_argument(
        '--save_masks',
        action='store_true',
        help='是否保存分割掩码图像'
    )
    eval_group.add_argument(
        '--max_masks',
        type=int,
        default=50,
        help='最多保存的掩码图像数量'
    )
    
    return parser


def load_config(
    config_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
    **kwargs
) -> FederatedConfig:
    """
    统一的配置加载接口
    
    此函数提供了多种方式加载配置：
    1. 从配置文件加载（如果提供 config_path）
    2. 从命令行参数加载（如果提供 args 或解析 sys.argv）
    3. 从关键字参数加载
    4. 使用默认值
    
    参数优先级（从高到低）：
    kwargs > config_path > args > defaults
    
    Args:
        config_path: JSON配置文件路径（可选）
        args: 命令行参数命名空间（可选）
        **kwargs: 直接指定的配置参数（优先级最高）
        
    Returns:
        FederatedConfig 实例
        
    示例:
        # 方式1: 使用默认值
        >>> config = load_config()
        
        # 方式2: 从配置文件加载
        >>> config = load_config(config_path='config.json')
        
        # 方式3: 从命令行参数加载
        >>> config = load_config()  # 自动解析 sys.argv
        
        # 方式4: 混合使用（覆盖特定参数）
        >>> config = load_config(config_path='config.json', rounds=100, lr=1e-5)
    """
    # 1. 基础配置：从文件或命令行
    if config_path is not None:
        # 从配置文件加载
        base_config = FederatedConfig.load(config_path)
        config_dict = base_config.to_dict()
    elif args is not None:
        # 从提供的命令行参数加载
        base_config = load_config_from_args(args)
        config_dict = base_config.to_dict()
    else:
        # 尝试解析命令行参数
        import sys
        if len(sys.argv) > 1:
            # 有命令行参数，解析它们
            base_config = load_config_from_args()
            config_dict = base_config.to_dict()
        else:
            # 没有任何参数，使用默认值
            config_dict = {}
    
    # 2. 应用关键字参数覆盖（优先级最高）
    config_dict.update(kwargs)
    
    # 3. 创建最终配置对象
    return FederatedConfig(**config_dict)


if __name__ == "__main__":
    """
    模块测试和示例用法
    
    运行此模块可以：
    1. 测试配置加载功能
    2. 查看配置的默认值
    3. 生成示例配置文件
    """
    print("=" * 70)
    print("SegSAM 配置管理器 - 测试模式")
    print("=" * 70)
    
    # 测试1: 加载默认配置
    print("\n[测试1] 加载默认配置:")
    print("-" * 70)
    config = load_config()
    print(config)
    
    # 测试2: 从命令行参数加载
    print("\n[测试2] 支持的命令行参数:")
    print("-" * 70)
    parser = create_argument_parser()
    parser.print_help()
    
    # 测试3: 保存配置到文件
    print("\n[测试3] 保存配置到文件:")
    print("-" * 70)
    config.save("example_config.json")
    
    # 测试4: 从文件加载配置
    print("\n[测试4] 从文件加载配置:")
    print("-" * 70)
    loaded_config = FederatedConfig.load("example_config.json")
    print(loaded_config)
    
    # 测试5: 使用关键字参数覆盖
    print("\n[测试5] 使用关键字参数覆盖:")
    print("-" * 70)
    custom_config = load_config(rounds=100, lr=1e-5, batch_size=8)
    print(custom_config)
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)

