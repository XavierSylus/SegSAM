"""
联邦学习客户端封装 (Federated Learning Client Wrapper)
===================================================

【设计理念】
这个模块提供了一个高层次的 Client 封装,整合了:
1. 客户端配置 (ClientConfig)
2. 模态感知数据加载器 (ModalityAwareDataLoader)
3. 客户端训练器 (ClientTrainer)

【核心优势】
✅ 配置驱动初始化 - 从配置自动推断模态类型和数据路径
✅ 类型安全训练 - 自动根据模态类型处理数据
✅ 容错验证 - 在训练开始前验证数据与配置的一致性

【使用示例】
>>> from src.client_wrapper import FederatedClient
>>> from src.client_config import CLIENT_CONFIGS_EXAMPLE
>>>
>>> # 从配置创建客户端
>>> client = FederatedClient.from_config(CLIENT_CONFIGS_EXAMPLE[0])
>>>
>>> # 执行训练
>>> results = client.train(global_model, global_reps)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging

# 模块导入
try:
    from src.client_config import ClientConfig, get_modality_type
    from src.modality_aware_dataloader import (
        ModalityAwareDataLoader,
        create_modality_aware_loader
    )
    from src.client import ClientTrainer
    from src.model import SAM3_Medical, DEVICE
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.client_config import ClientConfig, get_modality_type
    from src.modality_aware_dataloader import (
        ModalityAwareDataLoader,
        create_modality_aware_loader
    )
    from src.client import ClientTrainer
    from src.model import SAM3_Medical, DEVICE


class FederatedClient:
    """
    联邦学习客户端封装

    整合配置、数据加载和训练逻辑的高级接口
    """

    def __init__(
        self,
        client_config: ClientConfig,
        private_loader: DataLoader,
        public_loader: Optional[DataLoader] = None,
        device: str = DEVICE,
        use_amp: bool = True,
        local_epochs: int = 1,
        dataset_name: str = "BraTS",
        embed_dim: int = 1024
    ):
        """
        Args:
            client_config: 客户端配置字典
            private_loader: 私有数据 DataLoader (原始 PyTorch DataLoader)
            public_loader: 公共数据 DataLoader (原始 PyTorch DataLoader,可选)
            device: 训练设备
            use_amp: 是否使用混合精度训练
            local_epochs: 本地训练轮数
            dataset_name: 数据集名称
            embed_dim: 嵌入维度
        """
        self.client_config = client_config
        self.client_id = client_config['client_id']
        self.modality_type = get_modality_type(client_config)
        self.device = device

        # 设置日志
        self.logger = logging.getLogger(f"FederatedClient.{self.client_id}")

        # 包装为模态感知数据加载器
        self.private_loader = create_modality_aware_loader(
            private_loader,
            client_config,
            strict_validation=True
        )

        if public_loader is not None:
            self.public_loader = create_modality_aware_loader(
                public_loader,
                client_config,
                strict_validation=True
            )
        else:
            self.public_loader = None

        # 创建客户端训练器 (不传入模型,在训练时传入)
        # 注意:这里我们传入原始 DataLoader,因为 ClientTrainer 内部会自己处理数据
        # 但我们需要确保数据格式正确,所以先用模态感知加载器验证一次
        self._validate_data_format()

        # 创建训练器 (使用原始 DataLoader,因为 ClientTrainer 内部有自己的数据处理逻辑)
        self.trainer = ClientTrainer(
            private_loader=private_loader,
            public_loader=public_loader,
            device=device,
            use_amp=use_amp,
            local_epochs=local_epochs,
            dataset_name=dataset_name,
            embed_dim=embed_dim
        )

        self.logger.info(
            f"初始化客户端: {self.client_id} | "
            f"模态类型: {self.modality_type} | "
            f"设备: {device}"
        )

    def _validate_data_format(self) -> None:
        """
        验证数据格式与配置的一致性

        在训练开始前,检查第一个批次的数据是否符合配置的 modalities

        Raises:
            ValueError: 如果数据格式与配置不符
        """
        try:
            # 验证私有数据
            private_iter = iter(self.private_loader)
            first_batch = next(private_iter)

            # 检查必需的键是否存在
            if self.modality_type == "multimodal":
                if first_batch['image'] is None or first_batch['text_feature'] is None:
                    raise ValueError(
                        f"客户端 {self.client_id}: 配置为多模态但数据缺少图像或文本特征"
                    )
            elif self.modality_type == "image_only":
                if first_batch['image'] is None:
                    raise ValueError(
                        f"客户端 {self.client_id}: 配置为纯图像但数据缺少图像"
                    )
            elif self.modality_type == "text_only":
                if first_batch['text_feature'] is None:
                    raise ValueError(
                        f"客户端 {self.client_id}: 配置为纯文本但数据缺少文本特征"
                    )

            self.logger.info(f"✅ 数据格式验证通过: {self.client_id}")

        except StopIteration:
            self.logger.warning(f"⚠️ 警告: 私有数据加载器为空: {self.client_id}")
        except Exception as e:
            self.logger.error(f"❌ 数据格式验证失败: {self.client_id}")
            raise

    def train(
        self,
        model: SAM3_Medical,
        optimizer: torch.optim.Optimizer,
        global_reps: Dict[str, torch.Tensor],
        lambda_cream: float = 0.05  # Step2: 0.02 → 0.05
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, dict]:
        """
        执行本地训练

        Args:
            model: SAM3_Medical 模型实例
            optimizer: 优化器
            global_reps: 全局表示字典
            lambda_cream: 对比学习损失权重

        Returns:
            Tuple of (updated_model_state_dict, local_public_reps, training_stats)
        """
        self.logger.info(
            f"开始本地训练: {self.client_id} | "
            f"模态: {self.modality_type} | "
            f"本地轮数: {self.trainer.local_epochs}"
        )

        # 调用底层训练器的 run 方法
        updated_state, local_reps, stats = self.trainer.run(
            model=model,
            optimizer=optimizer,
            global_reps=global_reps,
            lambda_cream=lambda_cream
        )

        self.logger.info(
            f"训练完成: {self.client_id} | "
            f"平均损失: {stats.get('avg_loss', 0):.4f} | "
            f"批次数: {stats.get('num_batches', 0)}"
        )

        return updated_state, local_reps, stats

    def validate(
        self,
        model: SAM3_Medical,
        test_loader: DataLoader,
        compute_hd95: bool = True,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        在测试集上验证模型

        Args:
            model: SAM3_Medical 模型实例
            test_loader: 测试数据加载器
            compute_hd95: 是否计算 HD95
            verbose: 是否打印详细信息

        Returns:
            评估指标字典
        """
        self.logger.info(f"开始验证: {self.client_id}")

        metrics = self.trainer.validate(
            model=model,
            test_loader=test_loader,
            compute_hd95=compute_hd95,
            verbose=verbose
        )

        self.logger.info(
            f"验证完成: {self.client_id} | "
            f"Dice: {metrics.get('dice', 0):.4f}"
        )

        return metrics

    @classmethod
    def from_config(
        cls,
        client_config: ClientConfig,
        private_dataset: torch.utils.data.Dataset,
        public_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 6,
        device: str = DEVICE,
        **trainer_kwargs
    ) -> 'FederatedClient':
        """
        从配置和数据集创建客户端 (工厂方法)

        Args:
            client_config: 客户端配置
            private_dataset: 私有数据集 (PyTorch Dataset)
            public_dataset: 公共数据集 (PyTorch Dataset,可选)
            batch_size: 批次大小
            device: 训练设备
            **trainer_kwargs: 传递给 ClientTrainer 的额外参数

        Returns:
            FederatedClient 实例

        示例:
            >>> from src.client_config import CLIENT_CONFIGS_EXAMPLE
            >>> from torch.utils.data import TensorDataset
            >>>
            >>> config = CLIENT_CONFIGS_EXAMPLE[0]
            >>> dummy_images = torch.randn(100, 3, 256, 256)
            >>> dummy_masks = torch.randn(100, 1, 256, 256)
            >>> private_dataset = TensorDataset(dummy_images, dummy_masks)
            >>>
            >>> client = FederatedClient.from_config(
            ...     config,
            ...     private_dataset,
            ...     batch_size=4
            ... )
        """
        # 创建 DataLoader
        private_loader = DataLoader(
            private_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        public_loader = None
        if public_dataset is not None:
            public_loader = DataLoader(
                public_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

        return cls(
            client_config=client_config,
            private_loader=private_loader,
            public_loader=public_loader,
            device=device,
            **trainer_kwargs
        )

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"FederatedClient("
            f"id={self.client_id}, "
            f"modality={self.modality_type}, "
            f"device={self.device}"
            f")"
        )


# ==================== 单元测试 ====================

if __name__ == "__main__":
    import sys
    from torch.utils.data import TensorDataset

    # 添加路径
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.client_config import CLIENT_CONFIGS_EXAMPLE

    # 设置日志
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 测试联邦学习客户端封装\n")
    print("=" * 80)

    # 测试 1: 创建纯图像客户端
    print("\n测试 1: 创建纯图像客户端 (image_only)")
    print("-" * 60)

    image_config = CLIENT_CONFIGS_EXAMPLE[1]  # client_2 (image_only)

    # 创建模拟数据集
    dummy_images = torch.randn(20, 3, 256, 256)
    dummy_masks = torch.randn(20, 1, 256, 256)
    image_dataset = TensorDataset(dummy_images, dummy_masks)

    # 使用工厂方法创建客户端
    image_client = FederatedClient.from_config(
        image_config,
        image_dataset,
        batch_size=4,
        use_amp=False,  # CPU 测试禁用 AMP
        local_epochs=1
    )

    print(f"✅ 成功创建客户端: {image_client}")

    # 测试 2: 创建多模态客户端
    print("\n测试 2: 创建多模态客户端 (multimodal)")
    print("-" * 60)

    multimodal_config = CLIENT_CONFIGS_EXAMPLE[2]  # client_3 (multimodal)

    # 创建多模态数据集
    class MultimodalDataset(torch.utils.data.Dataset):
        def __init__(self, size=20):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return (
                torch.randn(3, 256, 256),  # image
                torch.randn(512),          # text_feature
                torch.randn(1, 256, 256)   # mask
            )

    multimodal_dataset = MultimodalDataset()

    multimodal_client = FederatedClient.from_config(
        multimodal_config,
        multimodal_dataset,
        batch_size=4,
        use_amp=False,
        local_epochs=1
    )

    print(f"✅ 成功创建客户端: {multimodal_client}")

    # 测试 3: 执行训练 (使用模拟模型)
    print("\n测试 3: 执行本地训练")
    print("-" * 60)

    # 创建模拟模型和优化器
    from src.model import SAM3_Medical

    model = SAM3_Medical(img_size=256)
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=1e-4)

    # 创建模拟全局表示
    global_reps = {
        'global_text_rep': torch.randn(model.embed_dim),
        'global_image_rep': torch.randn(model.embed_dim)
    }

    # 执行训练
    print(f"开始训练客户端: {image_client.client_id}")
    updated_state, local_reps, stats = image_client.train(
        model=model,
        optimizer=optimizer,
        global_reps=global_reps,
        lambda_cream=0.05  # Step2: 0.02 → 0.05
    )

    print(f"  - 平均损失: {stats['avg_loss']:.4f}")
    print(f"  - 批次数: {stats['num_batches']}")
    print(f"  - 本地表示形状: {local_reps.shape}")
    print("✅ 训练测试通过")

    print("\n" + "=" * 80)
    print("🎉 联邦学习客户端封装测试完成")
    print("=" * 80)
