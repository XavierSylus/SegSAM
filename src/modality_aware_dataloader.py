"""
模态感知数据加载器 (Modality-Aware DataLoader)
==============================================

【设计原则】
1. 配置驱动 (Configuration-Driven): 数据解析逻辑完全由 ClientConfig 驱动
2. 容错验证 (Fail-Fast Validation): 如果数据结构与声明的 modalities 不符,立即抛出异常
3. 类型安全 (Type-Safe): 使用明确的返回值类型,避免 Magic Numbers

【核心改进】
✅ 摒弃 len(batch) 魔法数字推断
✅ 基于配置的 modalities 显式解析数据
✅ 清晰的错误消息,方便调试

【数据格式约定】
- 纯图像 (image_only): DataLoader 返回 (image, mask)
- 纯文本 (text_only): DataLoader 返回 (text_feature, label)
- 多模态 (multimodal): DataLoader 返回 (image, text_feature, mask)
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Union, Optional, Literal
from pathlib import Path

try:
    from src.client_config import ClientConfig, get_modality_type
except ModuleNotFoundError:
    # 如果作为脚本直接运行,使用相对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.client_config import ClientConfig, get_modality_type


class ModalityAwareDataParser:
    """
    模态感知数据解析器

    根据客户端配置的 modalities 字段,正确解析 DataLoader 批次数据
    """

    def __init__(self, client_config: ClientConfig):
        """
        Args:
            client_config: 客户端配置字典
        """
        self.client_config = client_config
        self.client_id = client_config['client_id']
        self.modalities = client_config['modalities']
        self.modality_type = get_modality_type(client_config)

    def parse_batch(
        self,
        batch: Union[Tuple, List, Dict],
        batch_idx: int = 0
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        解析批次数据,返回标准化的字典格式

        Args:
            batch: DataLoader 返回的批次数据
            batch_idx: 批次索引 (用于错误消息)

        Returns:
            标准化的数据字典,包含:
            - 'image': 图像张量 (B, C, H, W) 或 None
            - 'text_feature': 文本特征 (B, D) 或 None
            - 'mask': 分割掩码 (B, 1, H, W) 或 None
            - 'label': 标签 (B,) 或 (B, num_classes) 或 None

        Raises:
            ValueError: 如果数据结构与配置的 modalities 不符
        """
        # 如果已经是字典格式,直接返回 (假设数据集已正确格式化)
        if isinstance(batch, dict):
            return self._validate_dict_batch(batch, batch_idx)

        # 否则根据 modality_type 解析 tuple/list 格式
        if isinstance(batch, (tuple, list)):
            return self._parse_tuple_batch(batch, batch_idx)

        # 未知格式
        raise ValueError(
            f"客户端 {self.client_id} (Batch {batch_idx}): "
            f"未知的批次数据格式: {type(batch)}. "
            f"期望 tuple, list 或 dict."
        )

    def _parse_tuple_batch(
        self,
        batch: Union[Tuple, List],
        batch_idx: int
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        解析 tuple/list 格式的批次数据

        【数据格式约定】
        - image_only: (image, mask) - 2 个元素
        - text_only: (text_feature, label) - 2 个元素
        - multimodal: (image, text_feature, mask) - 3 个元素

        Args:
            batch: tuple 或 list
            batch_idx: 批次索引

        Returns:
            标准化的数据字典

        Raises:
            ValueError: 如果数据结构与 modalities 不符
        """
        batch_len = len(batch)

        # ==================== 多模态 (Multimodal) ====================
        if self.modality_type == "multimodal":
            if batch_len != 3:
                raise ValueError(
                    f"客户端 {self.client_id} (Batch {batch_idx}): "
                    f"配置为多模态 (multimodal) 但批次数据只有 {batch_len} 个元素. "
                    f"期望 3 个元素: (image, text_feature, mask). "
                    f"\n💡 提示: 请检查 Dataset.__getitem__() 的返回值是否正确."
                )

            image, text_feature, mask = batch
            return {
                'image': image,
                'text_feature': text_feature,
                'mask': mask,
                'label': None
            }

        # ==================== 纯图像 (Image-Only) ====================
        elif self.modality_type == "image_only":
            if batch_len != 2:
                raise ValueError(
                    f"客户端 {self.client_id} (Batch {batch_idx}): "
                    f"配置为纯图像 (image_only) 但批次数据有 {batch_len} 个元素. "
                    f"期望 2 个元素: (image, mask). "
                    f"\n💡 提示: 如果数据集包含文本特征,请更新配置的 modalities 为 ['image', 'text']."
                )

            image, mask = batch
            return {
                'image': image,
                'text_feature': None,
                'mask': mask,
                'label': None
            }

        # ==================== 纯文本 (Text-Only) ====================
        elif self.modality_type == "text_only":
            if batch_len != 2:
                raise ValueError(
                    f"客户端 {self.client_id} (Batch {batch_idx}): "
                    f"配置为纯文本 (text_only) 但批次数据有 {batch_len} 个元素. "
                    f"期望 2 个元素: (text_feature, label). "
                    f"\n💡 提示: 如果数据集包含图像,请更新配置的 modalities 为 ['image', 'text']."
                )

            text_feature, label = batch
            return {
                'image': None,
                'text_feature': text_feature,
                'mask': None,
                'label': label
            }

        else:
            raise RuntimeError(f"未知的 modality_type: {self.modality_type}")

    def _validate_dict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        验证并标准化字典格式的批次数据

        Args:
            batch: 字典格式的批次数据
            batch_idx: 批次索引

        Returns:
            标准化的数据字典

        Raises:
            ValueError: 如果缺少必需的键
        """
        # 标准化键名 (兼容不同的命名约定)
        standardized = {
            'image': batch.get('image', batch.get('images', batch.get('inp', None))),
            'text_feature': batch.get('text_feature', batch.get('text_features', batch.get('text', None))),
            'mask': batch.get('mask', batch.get('masks', batch.get('gt', batch.get('label', None)))),
            'label': batch.get('label', batch.get('labels', None))
        }

        # 验证必需的键是否存在
        if self.modality_type == "multimodal":
            if standardized['image'] is None:
                raise ValueError(
                    f"客户端 {self.client_id} (Batch {batch_idx}): "
                    f"多模态配置但批次数据缺少 'image' 键. "
                    f"可用键: {list(batch.keys())}"
                )
            if standardized['text_feature'] is None:
                raise ValueError(
                    f"客户端 {self.client_id} (Batch {batch_idx}): "
                    f"多模态配置但批次数据缺少 'text_feature' 键. "
                    f"可用键: {list(batch.keys())}"
                )

        elif self.modality_type == "image_only":
            if standardized['image'] is None:
                raise ValueError(
                    f"客户端 {self.client_id} (Batch {batch_idx}): "
                    f"纯图像配置但批次数据缺少 'image' 键. "
                    f"可用键: {list(batch.keys())}"
                )

        elif self.modality_type == "text_only":
            if standardized['text_feature'] is None:
                raise ValueError(
                    f"客户端 {self.client_id} (Batch {batch_idx}): "
                    f"纯文本配置但批次数据缺少 'text_feature' 键. "
                    f"可用键: {list(batch.keys())}"
                )

        return standardized


class ModalityAwareDataLoader:
    """
    模态感知数据加载器包装器

    封装 PyTorch DataLoader,提供自动的批次数据解析功能
    """

    def __init__(
        self,
        dataloader: DataLoader,
        client_config: ClientConfig,
        strict_validation: bool = True
    ):
        """
        Args:
            dataloader: PyTorch DataLoader 实例
            client_config: 客户端配置
            strict_validation: 是否启用严格验证 (默认: True)
                              如果为 True,数据不符合配置时立即抛出异常
                              如果为 False,仅打印警告并尝试继续
        """
        self.dataloader = dataloader
        self.client_config = client_config
        self.strict_validation = strict_validation
        self.parser = ModalityAwareDataParser(client_config)

    def __iter__(self):
        """迭代器接口"""
        self._batch_idx = 0
        self._iter = iter(self.dataloader)
        return self

    def __next__(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        返回下一个解析后的批次

        Returns:
            标准化的数据字典

        Raises:
            StopIteration: 当 DataLoader 耗尽时
            ValueError: 当数据验证失败时 (如果 strict_validation=True)
        """
        batch = next(self._iter)

        try:
            parsed_batch = self.parser.parse_batch(batch, self._batch_idx)
            self._batch_idx += 1
            return parsed_batch
        except ValueError as e:
            if self.strict_validation:
                raise
            else:
                # 非严格模式:打印警告并返回原始数据
                print(f"⚠️ 警告: {e}")
                print(f"   降级为原始数据格式,跳过解析")
                self._batch_idx += 1
                return {'raw_batch': batch}

    def __len__(self) -> int:
        """返回批次数量"""
        return len(self.dataloader)


# ==================== 工厂函数 ====================

def create_modality_aware_loader(
    dataloader: DataLoader,
    client_config: ClientConfig,
    strict_validation: bool = True
) -> ModalityAwareDataLoader:
    """
    创建模态感知数据加载器 (工厂函数)

    Args:
        dataloader: PyTorch DataLoader
        client_config: 客户端配置
        strict_validation: 是否启用严格验证

    Returns:
        ModalityAwareDataLoader 实例

    示例:
        >>> from src.client_config import CLIENT_CONFIGS_EXAMPLE
        >>> config = CLIENT_CONFIGS_EXAMPLE[0]  # client_1 (text_only)
        >>>
        >>> # 创建 PyTorch DataLoader (假设已有 dataset)
        >>> raw_loader = DataLoader(dataset, batch_size=4)
        >>>
        >>> # 包装为模态感知加载器
        >>> aware_loader = create_modality_aware_loader(raw_loader, config)
        >>>
        >>> # 使用
        >>> for batch in aware_loader:
        ...     print(batch['text_feature'])  # 自动解析
    """
    return ModalityAwareDataLoader(dataloader, client_config, strict_validation)


# ==================== 单元测试 ====================

if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    try:
        from src.client_config import CLIENT_CONFIGS_EXAMPLE
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.client_config import CLIENT_CONFIGS_EXAMPLE

    print("\n🧪 测试模态感知数据加载器\n")
    print("=" * 80)

    # ==================== 测试 1: 纯图像客户端 ====================
    print("\n测试 1: 纯图像客户端 (image_only)")
    print("-" * 60)

    image_config = CLIENT_CONFIGS_EXAMPLE[1]  # client_2 (image_only)
    print(f"客户端: {image_config['client_id']}")
    print(f"模态类型: {get_modality_type(image_config)}")

    # 创建模拟数据: (image, mask)
    dummy_images = torch.randn(10, 3, 256, 256)
    dummy_masks = torch.randn(10, 1, 256, 256)
    image_dataset = TensorDataset(dummy_images, dummy_masks)
    image_loader = DataLoader(image_dataset, batch_size=2)

    # 包装为模态感知加载器
    aware_image_loader = create_modality_aware_loader(image_loader, image_config)

    # 测试解析
    for batch_idx, batch in enumerate(aware_image_loader):
        print(f"  Batch {batch_idx}:")
        print(f"    - image shape: {batch['image'].shape}")
        print(f"    - mask shape: {batch['mask'].shape}")
        print(f"    - text_feature: {batch['text_feature']}")
        if batch_idx >= 1:  # 只打印前2个 batch
            break

    print("✅ 纯图像客户端测试通过")

    # ==================== 测试 2: 多模态客户端 ====================
    print("\n测试 2: 多模态客户端 (multimodal)")
    print("-" * 60)

    multimodal_config = CLIENT_CONFIGS_EXAMPLE[2]  # client_3 (multimodal)
    print(f"客户端: {multimodal_config['client_id']}")
    print(f"模态类型: {get_modality_type(multimodal_config)}")

    # 创建模拟数据: (image, text_feature, mask)
    dummy_images = torch.randn(10, 3, 256, 256)
    dummy_text_features = torch.randn(10, 512)
    dummy_masks = torch.randn(10, 1, 256, 256)

    # 使用自定义 Dataset 类来返回 3 个元素
    class MultimodalDataset(torch.utils.data.Dataset):
        def __init__(self, images, text_features, masks):
            self.images = images
            self.text_features = text_features
            self.masks = masks

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.text_features[idx], self.masks[idx]

    multimodal_dataset = MultimodalDataset(dummy_images, dummy_text_features, dummy_masks)
    multimodal_loader = DataLoader(multimodal_dataset, batch_size=2)

    # 包装为模态感知加载器
    aware_multimodal_loader = create_modality_aware_loader(multimodal_loader, multimodal_config)

    # 测试解析
    for batch_idx, batch in enumerate(aware_multimodal_loader):
        print(f"  Batch {batch_idx}:")
        print(f"    - image shape: {batch['image'].shape}")
        print(f"    - text_feature shape: {batch['text_feature'].shape}")
        print(f"    - mask shape: {batch['mask'].shape}")
        if batch_idx >= 1:
            break

    print("✅ 多模态客户端测试通过")

    # ==================== 测试 3: 错误检测 ====================
    print("\n测试 3: 数据格式错误检测")
    print("-" * 60)

    # 使用纯图像配置,但提供 3 个元素的数据 (应该报错)
    try:
        wrong_loader = create_modality_aware_loader(multimodal_loader, image_config)
        for batch in wrong_loader:
            pass  # 触发错误
        print("❌ 应该抛出异常但没有")
    except ValueError as e:
        print(f"✅ 正确捕获错误:")
        print(f"   {e}")

    # ==================== 测试 4: 字典格式数据 ====================
    print("\n测试 4: 字典格式批次数据")
    print("-" * 60)

    # 使用返回字典的 Dataset
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 256, 256),
                'text_feature': torch.randn(512),
                'mask': torch.randn(1, 256, 256)
            }

    dict_dataset = DictDataset()
    dict_loader = DataLoader(dict_dataset, batch_size=2)
    aware_dict_loader = create_modality_aware_loader(dict_loader, multimodal_config)

    for batch_idx, batch in enumerate(aware_dict_loader):
        print(f"  Batch {batch_idx}:")
        print(f"    - image shape: {batch['image'].shape}")
        print(f"    - text_feature shape: {batch['text_feature'].shape}")
        print(f"    - mask shape: {batch['mask'].shape}")
        if batch_idx >= 1:
            break

    print("✅ 字典格式数据测试通过")

    print("\n" + "=" * 80)
    print("🎉 模态感知数据加载器测试完成")
    print("=" * 80)
