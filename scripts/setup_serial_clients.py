"""
串行客户端设置脚本 (异构联邦学习版本)

此模块负责为串行训练模式设置客户端配置。
支持三种异构客户端类型:
- text_only: 仅文本数据 (如医学报告)
- image_only: 仅图像数据 (MRI扫描 + 分割标签)
- multimodal: 图像+文本配对数据

作者: SegSAM Team
日期: 2026-02-28
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
# ✅ 使用新的异构数据加载器
from data.heterogeneous_dataset_loader import create_heterogeneous_data_loaders


def setup_serial_clients(
    data_root: str,
    batch_size: int,
    img_size: int,
    max_samples: Optional[int] = None,
    embed_dim: int = 768,
    configured_clients: Optional[List[Dict]] = None,
) -> Dict[str, Dict]:
    """
    设置异构串行训练的客户端配置

    此函数为每个客户端准备数据加载器和元信息，但不创建模型。
    模型将在训练循环中按需创建和销毁。

    Args:
        data_root: 数据根目录 (如 "data/federated_split")
        batch_size: 批次大小
        img_size: 图像尺寸
        max_samples: 最大样本数（用于快速测试）
        embed_dim: 嵌入维度
        configured_clients: YAML 中启用的客户端列表；为 None 时加载默认三客户端实验。

    Returns:
        客户端配置字典，格式:
        {
            'client_id': {
                'modality': 'text_only' | 'image_only' | 'multimodal',
                'private_loader': DataLoader,
                'public_loader': DataLoader | None,
                'embed_dim': int
            }
        }
    """
    print("=" * 70)
    print("[异构联邦学习] 设置串行客户端")
    print("=" * 70)
    print(f"数据根目录: {data_root}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {img_size}")
    print(f"最大样本数: {max_samples if max_samples else '不限制'}")
    print("=" * 70)

    client_configs = {}

    default_client_types = [
        {
            'client_id': 'client_1',
            'modality': 'text_only'      # 纯文本客户端
        },
        {
            'client_id': 'client_2',
            'modality': 'image_only'     # 纯图像客户端
        },
        {
            'client_id': 'client_3',
            'modality': 'multimodal'     # 多模态客户端
        }
    ]
    if configured_clients is None:
        client_types = default_client_types
    else:
        client_types = []
        for client in configured_clients:
            if not client.get('enabled', True):
                continue
            client_id = str(client['client_id']).strip()
            compact_id = client_id.replace('_', '').lower()
            if compact_id.startswith('client') and compact_id[6:].isdigit():
                client_id = f"client_{compact_id[6:]}"
            client_types.append({
                'client_id': client_id,
                'modality': str(client['modality']).strip(),
            })
        if not client_types:
            raise ValueError("configured_clients 中没有启用的客户端")

    try:
        # ✅ 使用新的异构数据加载器（一次性为所有客户端创建）
        loaders_dict = create_heterogeneous_data_loaders(
            data_root=data_root,
            split="train",
            client_configs=client_types,
            batch_size=batch_size,
            image_size=img_size,
            num_workers=0,  # Windows 上设为 0 避免多进程问题
            shuffle=True,
            max_samples=max_samples
        )

        # 整理为标准格式
        for client_type in client_types:
            client_id = client_type['client_id']
            modality = client_type['modality']

            if client_id not in loaders_dict:
                print(f"  [警告] {client_id} 未在加载器字典中找到，跳过")
                continue

            private_loader, public_loader = loaders_dict[client_id]

            # 验证数据加载器
            if private_loader is None:
                print(f"  [错误] {client_id} 的私有数据加载器为空，跳过")
                continue

            # 保存配置
            client_configs[client_id] = {
                'modality': modality,
                'private_loader': private_loader,
                'public_loader': public_loader,
                'embed_dim': embed_dim
            }

            # 打印数据集信息
            print(f"\n[{client_id}] 配置成功")
            print(f"  - 模态类型: {modality}")
            print(f"  - Private数据: {len(private_loader.dataset)} 样本")
            if public_loader:
                print(f"  - Public数据: {len(public_loader.dataset)} 样本")
            else:
                print(f"  - Public数据: 无")

    except Exception as e:
        print(f"\n[错误] 数据加载器创建失败!")
        print(f"  原因: {e}")
        import traceback
        traceback.print_exc()
        raise

    if not client_configs:
        raise RuntimeError(
            "没有成功配置任何客户端！\n"
            "请检查:\n"
            "1. 数据目录是否存在: data/federated_split/train/client_X/\n"
            "2. 数据文件是否完整 (图像, 标签, 文本特征)\n"
            "3. heterogeneous_dataset_loader.py 是否正常工作"
        )

    print("\n" + "=" * 70)
    print(f"[完成] 成功配置 {len(client_configs)} 个异构客户端")
    print("=" * 70)
    print("\n配置摘要:")
    for client_id, cfg in client_configs.items():
        print(f"  {client_id}: {cfg['modality']}")
    print()

    return client_configs


def test_data_loading(client_configs: Dict[str, Dict]):
    """
    测试数据加载器是否正常工作

    Args:
        client_configs: setup_serial_clients() 返回的配置字典
    """
    print("=" * 70)
    print("测试数据加载")
    print("=" * 70)

    for client_id, cfg in client_configs.items():
        modality = cfg['modality']
        private_loader = cfg['private_loader']
        public_loader = cfg['public_loader']

        print(f"\n[{client_id}] 测试 ({modality})...")

        # 测试 private 数据
        try:
            for batch_idx, batch_data in enumerate(private_loader):
                print(f"  Private Batch {batch_idx + 1}:")

                if modality == "text_only":
                    text_features, = batch_data
                    print(f"    Text features: {text_features.shape}")
                    print(f"    Text stats: mean={text_features.mean().item():.6f}, "
                          f"std={text_features.std().item():.6f}")

                elif modality == "image_only":
                    images, masks = batch_data
                    print(f"    Images: {images.shape}")
                    print(f"    Masks: {masks.shape}")
                    print(f"    Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")

                elif modality == "multimodal":
                    images, masks, text_features = batch_data
                    print(f"    Images: {images.shape}")
                    print(f"    Masks: {masks.shape}")
                    print(f"    Text features: {text_features.shape}")

                if batch_idx >= 0:  # 只测试第一个 batch
                    break

        except Exception as e:
            print(f"  [错误] Private 数据迭代失败: {e}")
            import traceback
            traceback.print_exc()

        # 测试 public 数据
        if public_loader:
            try:
                for batch_idx, batch_data in enumerate(public_loader):
                    print(f"  Public Batch {batch_idx + 1}:")

                    if modality == "text_only":
                        text_features, = batch_data
                        print(f"    Text features: {text_features.shape}")

                    elif modality == "image_only":
                        images, = batch_data
                        print(f"    Images: {images.shape}")

                    elif modality == "multimodal":
                        images, text_features = batch_data
                        print(f"    Images: {images.shape}")
                        print(f"    Text features: {text_features.shape}")

                    if batch_idx >= 0:  # 只测试第一个 batch
                        break

            except Exception as e:
                print(f"  [错误] Public 数据迭代失败: {e}")

    print("\n" + "=" * 70)
    print("✓ 数据加载测试完成")
    print("=" * 70)


if __name__ == "__main__":
    """测试客户端设置"""
    print("\n" * 2)
    print("=" * 70)
    print("异构串行客户端设置 - 测试模式")
    print("=" * 70)
    print()

    # 使用测试配置
    try:
        configs = setup_serial_clients(
            data_root="data/federated_split",
            batch_size=2,
            img_size=256,
            max_samples=5  # 限制样本数以加快测试
        )

        print("\n配置的客户端:")
        for client_id, cfg in configs.items():
            print(f"  {client_id}: {cfg['modality']}")

        # 测试数据加载
        print("\n")
        test_data_loading(configs)

        print("\n✓ 所有测试完成！")
        print("\n下一步:")
        print("1. 修改 src/client.py 添加 text_only 训练逻辑")
        print("2. 修改 src/federated_trainer.py 集成异构客户端")
        print("3. 运行完整的联邦训练流程")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

