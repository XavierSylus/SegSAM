"""
配置管理系统
支持从 YAML 配置文件加载所有超参数
"""
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class Config:
    """
    配置类，用于加载和管理所有训练超参数
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        从配置字典初始化配置对象
        
        Args:
            config_dict: 配置字典
        """
        # 训练基础参数
        self.data_root = config_dict.get('data_root', 'data/federated_split')
        self.device = config_dict.get('device', 'cuda')
        
        # 训练超参数
        training = config_dict.get('training', {})
        self.batch_size = training.get('batch_size', 6)
        self.learning_rate = training.get('learning_rate', 2e-4)
        self.rounds = training.get('rounds', 50)
        self.local_epochs = training.get('local_epochs', 1)
        
        # Cream 相关参数
        cream = config_dict.get('cream', {})
        self.lambda_cream = cream.get('lambda_cream', 0.05)  # Step2: 0.02 → 0.05
        self.tau = cream.get('tau', 0.07)
        
        # 模型参数
        model = config_dict.get('model', {})
        self.img_size = model.get('img_size', 1024)
        self.embed_dim = model.get('embed_dim', 768)
        self.decoder_dim = model.get('decoder_dim', 256)
        self.num_classes = model.get('num_classes', 1)
        self.adapter_skip = model.get('adapter_skip', 64)
        
        # 服务器聚合参数
        server = config_dict.get('server', {})
        self.aggregation_method = server.get('aggregation_method', 'contrastive_weighted')
        self.global_rep_alpha = server.get('global_rep_alpha', 0.9)
        
        # 训练选项
        options = config_dict.get('options', {})
        self.use_amp = options.get('use_amp', True)
        self.grad_clip = options.get('grad_clip', 1.0)   # ★ 默认开启 norm=1.0

        self.use_dummy = options.get('use_dummy', False)
        
        # 日志记录选项
        logging = config_dict.get('logging', {})
        self.log_type = logging.get('log_type', 'tensorboard')  # wandb, tensorboard, both, none
        self.log_dir = logging.get('log_dir', None)  # 如果为 None 则使用默认路径
        self.experiment_name = logging.get('experiment_name', None)  # 如果为 None 则自动生成
        self.wandb_project = logging.get('wandb_project', 'SegSAM')
        self.wandb_entity = logging.get('wandb_entity', None)
        
        # 检查点选项
        checkpoint = config_dict.get('checkpoint', {})
        self.checkpoint_interval = checkpoint.get('checkpoint_interval', 10)  # 每隔 N 轮保存检查点
        self.resume_from_checkpoint = checkpoint.get('resume_from_checkpoint', None)  # 检查点路径，如果为 None 则从头开始
        self.keep_checkpoint_max = checkpoint.get('keep_checkpoint_max', 5)  # 最多保留的检查点数量（0 表示保留所有）
        
        # 保存配置字典的原始引用（用于扩展）
        self._config_dict = config_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置对象转换为字典
        
        Returns:
            配置字典
        """
        return {
            'data_root': self.data_root,
            'device': self.device,
            'training': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'rounds': self.rounds,
                'local_epochs': self.local_epochs,
            },
            'cream': {
                'lambda_cream': self.lambda_cream,
                'tau': self.tau,
            },
            'model': {
                'img_size': self.img_size,
                'embed_dim': self.embed_dim,
                'decoder_dim': self.decoder_dim,
                'num_classes': self.num_classes,
                'adapter_skip': self.adapter_skip,
            },
            'server': {
                'aggregation_method': self.aggregation_method,
                'global_rep_alpha': self.global_rep_alpha,
            },
            'options': {
                'use_amp': self.use_amp,
                'grad_clip': self.grad_clip,
                'use_dummy': self.use_dummy,
            },
            'logging': {
                'log_type': self.log_type,
                'log_dir': self.log_dir,
                'experiment_name': self.experiment_name,
                'wandb_project': self.wandb_project,
                'wandb_entity': self.wandb_entity,
            },
            'checkpoint': {
                'checkpoint_interval': self.checkpoint_interval,
                'resume_from_checkpoint': self.resume_from_checkpoint,
                'keep_checkpoint_max': self.keep_checkpoint_max,
            },
        }
    
    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        return f"Config(data_root={self.data_root}, rounds={self.rounds}, batch_size={self.batch_size}, lr={self.learning_rate}, lambda_cream={self.lambda_cream})"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """
        从 YAML 文件加载配置
        
        Args:
            config_path: YAML 配置文件路径
        
        Returns:
            Config 对象
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise ValueError(f"配置文件为空或格式错误: {config_path}")
        
        return cls(config_dict)
    
    @classmethod
    def from_args(cls, args: Optional[argparse.Namespace] = None) -> 'Config':
        """
        从命令行参数加载配置
        
        Args:
            args: argparse.Namespace 对象，如果为 None 则从 sys.argv 解析
        
        Returns:
            Config 对象
        """
        parser = argparse.ArgumentParser(description="SegSAM 图像分割训练配置")
        parser.add_argument(
            '--config',
            type=str,
            required=True,
            help='配置文件路径（例如: configs/segsam_image_only.yaml）'
        )
        
        if args is None:
            args = parser.parse_args()
        
        return cls.from_yaml(args.config)
    
    def merge_from_dict(self, override_dict: Dict[str, Any]):
        """
        用字典中的值覆盖当前配置（递归合并）
        
        Args:
            override_dict: 覆盖配置字典
        """
        def deep_merge(base: dict, override: dict) -> dict:
            result = copy.deepcopy(base)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = deep_merge(self._config_dict, override_dict)
        self.__init__(merged)
    
    def save_yaml(self, save_path: str):
        """
        保存配置到 YAML 文件
        
        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    便捷函数：加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None 则从命令行参数解析
    
    Returns:
        Config 对象
    """
    if config_path is None:
        return Config.from_args()
    else:
        return Config.from_yaml(config_path)


if __name__ == "__main__":
    # 测试配置加载
    import sys
    
    if len(sys.argv) > 1:
        config = Config.from_yaml(sys.argv[1])
        print("配置加载成功:")
        print(config)
        print("\n详细配置:")
        import json
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("用法: python -m src.config <config.yaml>")


