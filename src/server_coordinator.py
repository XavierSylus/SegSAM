"""
服务端模态筛选与解耦聚合示例 (Server-Side Modality Filtering)
============================================================

【目标】
演示如何在服务端使用新的显式配置系统进行模态感知的解耦聚合

【核心场景】
1. 筛选拥有特定模态的客户端 ID 列表
2. 根据模态类型进行解耦参数聚合
3. 适配现有的 CreamAggregator 聚合逻辑

【示例流程】
1. 服务端接收所有客户端的配置
2. 根据参数类型筛选参与聚合的客户端:
   - mask_decoder: 只聚合 image_only + multimodal 客户端
   - text_encoder: 只聚合 text_only + multimodal 客户端
   - adapter: 全员聚合
3. 调用 CreamAggregator.aggregate_weights() 进行聚合
"""

import torch
from typing import List, Dict, Optional
from pathlib import Path

# 模块导入
try:
    from src.client_config import (
        ClientConfig,
        filter_clients_by_modality,
        get_modality_types_list,
        print_config_summary
    )
    from src.server import CreamAggregator
    from src.model import SAM3_Medical, DEVICE
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.client_config import (
        ClientConfig,
        filter_clients_by_modality,
        get_modality_types_list,
        print_config_summary
    )
    from src.server import CreamAggregator
    from src.model import SAM3_Medical, DEVICE


class FederatedServerCoordinator:
    """
    联邦学习服务端协调器

    负责根据客户端配置进行模态感知的聚合
    """

    def __init__(
        self,
        client_configs: List[ClientConfig],
        global_model: SAM3_Medical,
        device: str = DEVICE,
        aggregation_method: str = "contrastive_weighted"
    ):
        """
        Args:
            client_configs: 所有参与联邦学习的客户端配置列表
            global_model: 全局模型
            device: 设备
            aggregation_method: 聚合方法
        """
        self.client_configs = client_configs
        self.global_model = global_model
        self.device = device

        # 创建 CreamAggregator
        self.aggregator = CreamAggregator(
            global_model=global_model,
            device=device,
            aggregation_method=aggregation_method
        )

        # 提取模态类型列表 (用于传递给 aggregator)
        self.modality_types = get_modality_types_list(client_configs)

        print(f"✅ 初始化服务端协调器:")
        print(f"  - 客户端数量: {len(client_configs)}")
        print(f"  - 聚合方法: {aggregation_method}")
        print(f"  - 模态类型列表: {self.modality_types}")

    def get_image_modality_clients(self) -> List[str]:
        """
        获取所有拥有 'image' 模态的客户端 ID 列表

        用途: 用于 mask_decoder 参数聚合

        Returns:
            客户端 ID 列表

        示例:
            >>> coordinator = FederatedServerCoordinator(client_configs, model)
            >>> image_clients = coordinator.get_image_modality_clients()
            >>> print(image_clients)
            ['client_2', 'client_3']  # image_only + multimodal
        """
        return filter_clients_by_modality(
            self.client_configs,
            required_modality='image',
            include_multimodal=True
        )

    def get_text_modality_clients(self) -> List[str]:
        """
        获取所有拥有 'text' 模态的客户端 ID 列表

        用途: 用于 text_encoder 参数聚合

        Returns:
            客户端 ID 列表

        示例:
            >>> text_clients = coordinator.get_text_modality_clients()
            >>> print(text_clients)
            ['client_1', 'client_3']  # text_only + multimodal
        """
        return filter_clients_by_modality(
            self.client_configs,
            required_modality='text',
            include_multimodal=True
        )

    def aggregate_client_updates(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_public_reps: List[torch.Tensor],
        global_features_for_contrastive: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        聚合客户端更新 (使用解耦聚合)

        Args:
            client_weights: 客户端模型权重列表 (按照 client_configs 顺序)
            client_public_reps: 客户端公共表示列表 (按照 client_configs 顺序)
            global_features_for_contrastive: 用于对比权重计算的全局特征

        Returns:
            聚合后的全局模型状态字典

        示例:
            >>> # 假设有 3 个客户端完成训练
            >>> client_weights = [client1_state, client2_state, client3_state]
            >>> client_reps = [client1_rep, client2_rep, client3_rep]
            >>>
            >>> # 服务端执行聚合
            >>> aggregated_state = coordinator.aggregate_client_updates(
            ...     client_weights,
            ...     client_reps
            ... )
            >>>
            >>> # 内部自动根据参数类型选择性聚合:
            >>> # - mask_decoder: 只聚合 client2 和 client3 (image_only + multimodal)
            >>> # - text_encoder: 只聚合 client1 和 client3 (text_only + multimodal)
            >>> # - adapter: 聚合全部 3 个客户端
        """
        print("\n" + "=" * 80)
        print("开始解耦聚合 (Decoupled Aggregation)")
        print("=" * 80)

        # 打印模态筛选信息
        print("\n模态筛选结果:")
        print(f"  - 拥有 'image' 模态的客户端: {self.get_image_modality_clients()}")
        print(f"  - 拥有 'text' 模态的客户端: {self.get_text_modality_clients()}")
        print(f"  - 模态类型列表: {self.modality_types}")

        # 调用 CreamAggregator 的 aggregate_weights 方法
        # 传入 client_modalities 参数以启用解耦聚合
        aggregated_state = self.aggregator.aggregate_weights(
            client_weights=client_weights,
            client_public_reps=client_public_reps,
            global_features_for_contrastive=global_features_for_contrastive,
            client_modalities=self.modality_types  # ✅ 传入模态类型列表
        )

        print("\n" + "=" * 80)
        print("解耦聚合完成")
        print("=" * 80)

        return aggregated_state

    def get_global_representations(self) -> Dict[str, torch.Tensor]:
        """
        获取当前全局表示 (用于下一轮客户端训练)

        Returns:
            包含 'global_text_rep' 和 'global_image_rep' 的字典
        """
        return self.aggregator.get_global_reps()

    def get_global_model(self) -> SAM3_Medical:
        """
        获取当前全局模型

        Returns:
            全局模型实例
        """
        return self.aggregator.get_global_model()


# ==================== 完整示例代码 ====================

def example_federated_training_round():
    """
    演示完整的联邦学习训练轮次

    包括:
    1. 初始化服务端和客户端
    2. 客户端本地训练
    3. 服务端聚合
    """
    print("\n" + "=" * 80)
    print("联邦学习训练轮次示例 (Federated Learning Round)")
    print("=" * 80)

    # ==================== 步骤 1: 准备客户端配置 ====================
    print("\n步骤 1: 准备客户端配置")
    print("-" * 60)

    from src.client_config import CLIENT_CONFIGS_EXAMPLE

    client_configs = CLIENT_CONFIGS_EXAMPLE
    print_config_summary(client_configs)

    # ==================== 步骤 2: 初始化服务端 ====================
    print("\n步骤 2: 初始化服务端协调器")
    print("-" * 60)

    global_model = SAM3_Medical(img_size=256)
    coordinator = FederatedServerCoordinator(
        client_configs=client_configs,
        global_model=global_model,
        aggregation_method="similarity_weighted"  # 使用相似度加权聚合 (更简单的测试)
    )

    # ==================== 步骤 3: 模拟客户端训练 ====================
    print("\n步骤 3: 模拟客户端训练")
    print("-" * 60)

    # 创建模拟的客户端更新 (实际应由客户端训练产生)
    num_clients = len(client_configs)
    client_weights = []
    client_public_reps = []

    for i, config in enumerate(client_configs):
        print(f"  - 客户端 {config['client_id']} 完成训练")

        # 模拟更新后的模型权重 (实际应由 ClientTrainer.run() 返回)
        client_model = SAM3_Medical(img_size=256)
        client_weights.append(client_model.state_dict())

        # 模拟公共表示 (实际应由 ClientTrainer.run() 返回)
        client_rep = torch.randn(global_model.embed_dim)
        client_public_reps.append(client_rep)

    # ==================== 步骤 4: 服务端聚合 ====================
    print("\n步骤 4: 服务端执行解耦聚合")
    print("-" * 60)

    aggregated_state = coordinator.aggregate_client_updates(
        client_weights=client_weights,
        client_public_reps=client_public_reps
    )

    print(f"\n✅ 聚合完成,全局模型已更新")
    print(f"   - 聚合参数数量: {len(aggregated_state)}")

    # ==================== 步骤 5: 获取全局表示 ====================
    print("\n步骤 5: 获取全局表示 (用于下一轮)")
    print("-" * 60)

    global_reps = coordinator.get_global_representations()
    print(f"  - global_text_rep shape: {global_reps['global_text_rep'].shape}")
    print(f"  - global_image_rep shape: {global_reps['global_image_rep'].shape}")

    print("\n" + "=" * 80)
    print("🎉 联邦学习训练轮次示例完成")
    print("=" * 80)


# ==================== 模态筛选实用函数 ====================

def filter_clients_for_parameter(
    param_name: str,
    client_configs: List[ClientConfig]
) -> List[str]:
    """
    根据参数名称筛选应该参与聚合的客户端 ID

    Args:
        param_name: 参数名称 (例如: 'mask_decoder.layers.0.weight')
        client_configs: 客户端配置列表

    Returns:
        应该参与该参数聚合的客户端 ID 列表

    示例:
        >>> # 筛选 mask_decoder 参数的参与客户端
        >>> mask_decoder_clients = filter_clients_for_parameter(
        ...     'mask_decoder.layers.0.weight',
        ...     client_configs
        ... )
        >>> print(mask_decoder_clients)
        ['client_2', 'client_3']  # image_only + multimodal
        >>>
        >>> # 筛选 text_encoder 参数的参与客户端
        >>> text_encoder_clients = filter_clients_for_parameter(
        ...     'text_encoder.layer.0.weight',
        ...     client_configs
        ... )
        >>> print(text_encoder_clients)
        ['client_1', 'client_3']  # text_only + multimodal
        >>>
        >>> # 筛选 adapter 参数的参与客户端
        >>> adapter_clients = filter_clients_for_parameter(
        ...     'adapter.conv1.weight',
        ...     client_configs
        ... )
        >>> print(adapter_clients)
        ['client_1', 'client_2', 'client_3']  # 全员参与
    """
    # 规则 1: mask_decoder 相关参数 -> 只聚合 image_only 和 multimodal
    if 'mask_decoder' in param_name:
        return filter_clients_by_modality(
            client_configs,
            required_modality='image',
            include_multimodal=True
        )

    # 规则 2: text_encoder 相关参数 -> 只聚合 text_only 和 multimodal
    if 'text_encoder' in param_name:
        return filter_clients_by_modality(
            client_configs,
            required_modality='text',
            include_multimodal=True
        )

    # 规则 3: adapter 或其他参数 -> 全员聚合
    return [config['client_id'] for config in client_configs]


# ==================== 主测试 ====================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("\n🧪 测试服务端模态筛选与解耦聚合\n")
    print("=" * 80)

    # 测试 1: 模态筛选
    print("\n测试 1: 模态筛选函数")
    print("-" * 60)

    from src.client_config import CLIENT_CONFIGS_EXAMPLE

    print("测试参数: 'mask_decoder.layers.0.weight'")
    mask_clients = filter_clients_for_parameter(
        'mask_decoder.layers.0.weight',
        CLIENT_CONFIGS_EXAMPLE
    )
    print(f"  参与聚合的客户端: {mask_clients}")

    print("\n测试参数: 'text_encoder.layer.0.weight'")
    text_clients = filter_clients_for_parameter(
        'text_encoder.layer.0.weight',
        CLIENT_CONFIGS_EXAMPLE
    )
    print(f"  参与聚合的客户端: {text_clients}")

    print("\n测试参数: 'adapter.conv1.weight'")
    adapter_clients = filter_clients_for_parameter(
        'adapter.conv1.weight',
        CLIENT_CONFIGS_EXAMPLE
    )
    print(f"  参与聚合的客户端: {adapter_clients}")

    print("\n✅ 模态筛选测试通过")

    # 测试 2: 完整联邦学习训练轮次
    print("\n" + "=" * 80)
    print("测试 2: 完整联邦学习训练轮次")
    print("=" * 80)

    example_federated_training_round()

    print("\n" + "=" * 80)
    print("🎉 服务端模态筛选与解耦聚合测试完成")
    print("=" * 80)
