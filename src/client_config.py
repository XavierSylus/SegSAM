"""
客户端配置模块 - 显式声明客户端模态类型与数据路径
==========================================================

【架构原则】
1. 显式声明 (Explicit Declaration): 每个客户端的模态类型必须在配置中明确指定
2. 单一数据源 (Single Source of Truth): 所有客户端数据都是 Private Data
3. 类型安全 (Type Safety): 使用 TypedDict 和枚举来保证配置正确性

【废弃的旧逻辑】
❌ 通过 has_public/has_private 文件夹推断客户端类型
❌ 通过 len(batch) 返回值数量推断模态类型
❌ 通过 Magic Numbers 猜测数据结构

【新的显式配置范式】
✅ client_configs: 显式声明每个客户端的模态和数据路径
✅ modalities: 明确列表 ['text'] | ['image'] | ['image', 'text']
✅ 配置驱动的数据解析逻辑
"""

from typing import List, Dict, Literal, TypedDict, Optional
from enum import Enum
from pathlib import Path


class Modality(str, Enum):
    """模态类型枚举 (Modality Types)"""
    TEXT = "text"
    IMAGE = "image"


class ClientConfig(TypedDict):
    """
    客户端配置类型定义

    Attributes:
        client_id: 客户端唯一标识符 (例如: 'client_1', 'client_A')
        modalities: 该客户端支持的模态列表 (例如: ['text'], ['image'], ['image', 'text'])
        data_path: 数据路径 (所有本地训练数据都视为 Private Data)
        description: 可选的描述信息 (用于调试和日志)

    示例:
        >>> config = {
        ...     'client_id': 'client_1',
        ...     'modalities': ['text'],
        ...     'data_path': 'data/federated_split/client_1/private/',
        ...     'description': '纯文本客户端 (Text-Only Client)'
        ... }
    """
    client_id: str
    modalities: List[Literal["text", "image"]]
    data_path: str
    description: Optional[str]


# ==================== 标准配置示例 ====================

# 示例 1: 三个异构客户端配置 (Heterogeneous Clients)
CLIENT_CONFIGS_EXAMPLE: List[ClientConfig] = [
    {
        'client_id': 'client_1',
        'modalities': ['text'],
        'data_path': 'data/federated_split/client_1/private/',
        'description': '纯文本客户端 - 仅训练文本编码器和 Adapter'
    },
    {
        'client_id': 'client_2',
        'modalities': ['image'],
        'data_path': 'data/federated_split/client_2/private/',
        'description': '纯图像客户端 - 仅训练图像编码器、Mask Decoder 和 Adapter'
    },
    {
        'client_id': 'client_3',
        'modalities': ['image', 'text'],
        'data_path': 'data/federated_split/client_3/private/',
        'description': '多模态客户端 - 训练全部组件'
    },
]

# 示例 2: 同构多模态客户端配置 (Homogeneous Multimodal Clients)
CLIENT_CONFIGS_HOMOGENEOUS: List[ClientConfig] = [
    {
        'client_id': 'hospital_A',
        'modalities': ['image', 'text'],
        'data_path': 'data/hospital_A/training_data/',
        'description': '医院 A - 多模态数据'
    },
    {
        'client_id': 'hospital_B',
        'modalities': ['image', 'text'],
        'data_path': 'data/hospital_B/training_data/',
        'description': '医院 B - 多模态数据'
    },
    {
        'client_id': 'hospital_C',
        'modalities': ['image', 'text'],
        'data_path': 'data/hospital_C/training_data/',
        'description': '医院 C - 多模态数据'
    },
]


# ==================== 配置验证工具 ====================

class ClientConfigValidator:
    """客户端配置验证器 - 确保配置合法性"""

    @staticmethod
    def validate_config(config: ClientConfig) -> None:
        """
        验证单个客户端配置

        Args:
            config: 客户端配置字典

        Raises:
            ValueError: 如果配置不合法
        """
        # 1. 检查必填字段
        if 'client_id' not in config or not config['client_id']:
            raise ValueError("client_id 是必填字段且不能为空")

        if 'modalities' not in config or not config['modalities']:
            raise ValueError(f"客户端 {config['client_id']}: modalities 是必填字段且不能为空列表")

        if 'data_path' not in config or not config['data_path']:
            raise ValueError(f"客户端 {config['client_id']}: data_path 是必填字段且不能为空")

        # 2. 检查 modalities 合法性
        valid_modalities = {Modality.TEXT.value, Modality.IMAGE.value}
        for modality in config['modalities']:
            if modality not in valid_modalities:
                raise ValueError(
                    f"客户端 {config['client_id']}: 非法的模态类型 '{modality}'. "
                    f"合法值: {valid_modalities}"
                )

        # 3. 检查 modalities 无重复
        if len(config['modalities']) != len(set(config['modalities'])):
            raise ValueError(
                f"客户端 {config['client_id']}: modalities 包含重复项: {config['modalities']}"
            )

        # 4. 检查 modalities 非空
        if not config['modalities']:
            raise ValueError(f"客户端 {config['client_id']}: modalities 不能为空列表")

    @staticmethod
    def validate_configs(configs: List[ClientConfig]) -> None:
        """
        验证客户端配置列表

        Args:
            configs: 客户端配置列表

        Raises:
            ValueError: 如果配置不合法
        """
        if not configs:
            raise ValueError("客户端配置列表不能为空")

        # 验证每个配置
        for config in configs:
            ClientConfigValidator.validate_config(config)

        # 检查 client_id 唯一性
        client_ids = [config['client_id'] for config in configs]
        if len(client_ids) != len(set(client_ids)):
            duplicates = [cid for cid in client_ids if client_ids.count(cid) > 1]
            raise ValueError(f"检测到重复的 client_id: {set(duplicates)}")


# ==================== 配置工具函数 ====================

def get_modality_type(config: ClientConfig) -> Literal["text_only", "image_only", "multimodal"]:
    """
    根据配置返回模态类型标签 (用于与服务端聚合逻辑兼容)

    Args:
        config: 客户端配置

    Returns:
        模态类型标签: 'text_only' | 'image_only' | 'multimodal'

    示例:
        >>> config = {'client_id': 'c1', 'modalities': ['text'], 'data_path': '...'}
        >>> get_modality_type(config)
        'text_only'
    """
    modalities_set = set(config['modalities'])

    if modalities_set == {Modality.TEXT.value}:
        return "text_only"
    elif modalities_set == {Modality.IMAGE.value}:
        return "image_only"
    elif modalities_set == {Modality.TEXT.value, Modality.IMAGE.value}:
        return "multimodal"
    else:
        raise ValueError(f"未知的模态组合: {config['modalities']}")


def filter_clients_by_modality(
    configs: List[ClientConfig],
    required_modality: Literal["text", "image"],
    include_multimodal: bool = True
) -> List[str]:
    """
    根据模态筛选客户端 ID 列表

    Args:
        configs: 客户端配置列表
        required_modality: 必需的模态类型 ('text' 或 'image')
        include_multimodal: 是否包含多模态客户端 (默认: True)

    Returns:
        符合条件的客户端 ID 列表

    示例:
        >>> # 筛选所有拥有 'image' 模态的客户端 (用于 mask_decoder 参数聚合)
        >>> image_clients = filter_clients_by_modality(configs, 'image')
        >>> # 输出: ['client_2', 'client_3'] (image_only + multimodal)

        >>> # 仅筛选纯图像客户端
        >>> pure_image_clients = filter_clients_by_modality(configs, 'image', include_multimodal=False)
        >>> # 输出: ['client_2']
    """
    filtered_ids = []

    for config in configs:
        modalities_set = set(config['modalities'])

        if required_modality in modalities_set:
            # 如果要求包含多模态,或者该客户端不是多模态,则加入
            is_multimodal = len(modalities_set) > 1
            if include_multimodal or not is_multimodal:
                filtered_ids.append(config['client_id'])

    return filtered_ids


def get_modality_types_list(configs: List[ClientConfig]) -> List[Literal["text_only", "image_only", "multimodal"]]:
    """
    返回客户端模态类型列表 (用于服务端聚合)

    Args:
        configs: 客户端配置列表

    Returns:
        模态类型标签列表,顺序与 configs 一致

    示例:
        >>> configs = CLIENT_CONFIGS_EXAMPLE
        >>> get_modality_types_list(configs)
        ['text_only', 'image_only', 'multimodal']
    """
    return [get_modality_type(config) for config in configs]


def print_config_summary(configs: List[ClientConfig]) -> None:
    """
    打印配置摘要 (调试用)

    Args:
        configs: 客户端配置列表
    """
    print("=" * 80)
    print("联邦学习客户端配置摘要 (Federated Learning Client Configuration Summary)")
    print("=" * 80)

    for i, config in enumerate(configs, 1):
        modality_type = get_modality_type(config)
        print(f"\n客户端 {i}: {config['client_id']}")
        print(f"  - 模态类型: {modality_type}")
        print(f"  - 支持模态: {', '.join(config['modalities'])}")
        print(f"  - 数据路径: {config['data_path']}")
        if 'description' in config and config['description']:
            print(f"  - 描述: {config['description']}")

    print("\n" + "=" * 80)
    print(f"总客户端数: {len(configs)}")

    # 统计模态分布
    modality_counts = {
        'text_only': 0,
        'image_only': 0,
        'multimodal': 0
    }
    for config in configs:
        modality_type = get_modality_type(config)
        modality_counts[modality_type] += 1

    print(f"模态分布:")
    print(f"  - 纯文本客户端: {modality_counts['text_only']}")
    print(f"  - 纯图像客户端: {modality_counts['image_only']}")
    print(f"  - 多模态客户端: {modality_counts['multimodal']}")
    print("=" * 80)


# ==================== 单元测试 ====================

if __name__ == "__main__":
    print("\n🧪 测试客户端配置模块\n")

    # 测试 1: 验证配置
    print("测试 1: 配置验证")
    print("-" * 60)
    try:
        ClientConfigValidator.validate_configs(CLIENT_CONFIGS_EXAMPLE)
        print("✅ 配置验证通过")
    except ValueError as e:
        print(f"❌ 配置验证失败: {e}")

    # 测试 2: 打印配置摘要
    print("\n测试 2: 打印配置摘要")
    print("-" * 60)
    print_config_summary(CLIENT_CONFIGS_EXAMPLE)

    # 测试 3: 模态类型转换
    print("\n测试 3: 模态类型转换")
    print("-" * 60)
    for config in CLIENT_CONFIGS_EXAMPLE:
        modality_type = get_modality_type(config)
        print(f"{config['client_id']}: {modality_type}")

    # 测试 4: 筛选客户端
    print("\n测试 4: 根据模态筛选客户端")
    print("-" * 60)

    # 筛选拥有 image 模态的客户端 (用于 mask_decoder 聚合)
    image_clients = filter_clients_by_modality(CLIENT_CONFIGS_EXAMPLE, 'image')
    print(f"拥有 'image' 模态的客户端 (含多模态): {image_clients}")

    # 筛选拥有 text 模态的客户端 (用于 text_encoder 聚合)
    text_clients = filter_clients_by_modality(CLIENT_CONFIGS_EXAMPLE, 'text')
    print(f"拥有 'text' 模态的客户端 (含多模态): {text_clients}")

    # 仅筛选纯图像客户端
    pure_image_clients = filter_clients_by_modality(CLIENT_CONFIGS_EXAMPLE, 'image', include_multimodal=False)
    print(f"纯图像客户端 (不含多模态): {pure_image_clients}")

    # 测试 5: 获取模态类型列表 (用于服务端)
    print("\n测试 5: 获取模态类型列表 (用于服务端聚合)")
    print("-" * 60)
    modality_types = get_modality_types_list(CLIENT_CONFIGS_EXAMPLE)
    print(f"模态类型列表: {modality_types}")

    # 测试 6: 错误配置检测
    print("\n测试 6: 错误配置检测")
    print("-" * 60)
    invalid_configs = [
        {
            'client_id': 'bad_client',
            'modalities': ['invalid_modality'],  # 非法模态
            'data_path': 'some/path'
        }
    ]
    try:
        ClientConfigValidator.validate_configs(invalid_configs)
        print("❌ 应该抛出异常但没有")
    except ValueError as e:
        print(f"✅ 正确捕获错误: {e}")

    print("\n" + "=" * 80)
    print("🎉 客户端配置模块测试完成")
    print("=" * 80)
