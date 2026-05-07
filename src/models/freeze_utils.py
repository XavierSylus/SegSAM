"""
模型冻结工具
用于冻结 SAM3 的 backbone 参数，只保留 Adapter 为可训练
"""

import torch
import torch.nn as nn
from typing import Optional, List, Set, Dict, Any
import warnings


def freeze_backbone(
    model: nn.Module,
    adapter_module_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    冻结 SAM3 的 backbone 参数，只保留 Adapter 为可训练
    
    Args:
        model: SAM3 模型实例
        adapter_module_names: Adapter 模块的名称列表（用于识别哪些模块是 Adapter）
                            如果为 None，将自动查找包含 "adapter" 的模块
        verbose: 是否打印详细信息
    
    Returns:
        统计信息字典:
        {
            'total_params': int,  # 总参数数量
            'frozen_params': int,  # 冻结参数数量
            'trainable_params': int,  # 可训练参数数量
            'frozen_modules': List[str],  # 冻结的模块名称
            'trainable_modules': List[str],  # 可训练的模块名称
        }
    
    Example:
        >>> model = SAM3_Medical()
        >>> stats = freeze_backbone(model)
        >>> print(f"可训练参数: {stats['trainable_params']:,}")
    """
    if adapter_module_names is None:
        adapter_module_names = []
    
    # 统计信息
    stats = {
        'total_params': 0,
        'frozen_params': 0,
        'trainable_params': 0,
        'frozen_modules': [],
        'trainable_modules': [],
    }
    
    # 第一步：冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False
        stats['total_params'] += param.numel()
        stats['frozen_params'] += param.numel()
    
    # 第二步：识别 Adapter 模块并设置为可训练
    adapter_modules = _find_adapter_modules(model, adapter_module_names)
    
    for module_name, module in adapter_modules.items():
        # 设置 Adapter 模块的参数为可训练
        for param_name, param in module.named_parameters():
            param.requires_grad = True
            # 更新统计信息
            stats['frozen_params'] -= param.numel()
            stats['trainable_params'] += param.numel()
        
        stats['trainable_modules'].append(module_name)
    
    # 第三步：识别其他可能需要训练的模块（如 MaskDecoder）
    # 查找可能包含 "decoder" 的模块
    decoder_modules = _find_decoder_modules(model)
    for module_name, module in decoder_modules.items():
        # 检查是否已经被识别为 Adapter
        if module_name not in adapter_modules:
            for param_name, param in module.named_parameters():
                if not param.requires_grad:  # 如果还没有被设置为可训练
                    param.requires_grad = True
                    stats['frozen_params'] -= param.numel()
                    stats['trainable_params'] += param.numel()

            if module_name not in stats['trainable_modules']:
                stats['trainable_modules'].append(module_name)

    # 第四步：★ 解冻多模态投影层（text_proj 和 image_proj）
    # 这些投影层对于对比学习至关重要，必须参与训练
    projection_modules = _find_projection_modules(model)
    for module_name, module in projection_modules.items():
        # 检查是否已经被识别为可训练模块
        if module_name not in adapter_modules and module_name not in decoder_modules:
            for param_name, param in module.named_parameters():
                if not param.requires_grad:  # 如果还没有被设置为可训练
                    param.requires_grad = True
                    stats['frozen_params'] -= param.numel()
                    stats['trainable_params'] += param.numel()

            if module_name not in stats['trainable_modules']:
                stats['trainable_modules'].append(module_name)
    
    # 收集冻结的模块名称（排除 Adapter 和 Decoder）
    for name, module in model.named_modules():
        if name and name not in stats['trainable_modules']:
            # 检查是否有参数
            has_params = any(p.numel() > 0 for p in module.parameters())
            if has_params and name not in stats['frozen_modules']:
                # 只添加顶层模块（避免重复）
                is_top_level = '.' not in name or name.split('.')[0] not in stats['frozen_modules']
                if is_top_level:
                    stats['frozen_modules'].append(name)
    
    # 打印统计信息
    if verbose:
        # 同时传递投影层信息
        projection_modules = _find_projection_modules(model)
        _print_freezing_stats(stats, adapter_modules, decoder_modules, projection_modules)
    
    return stats


def _find_adapter_modules(
    model: nn.Module,
    adapter_module_names: List[str]
) -> Dict[str, nn.Module]:
    """
    查找模型中的所有 Adapter 模块
    
    Args:
        model: 模型实例
        adapter_module_names: 用户指定的 Adapter 模块名称列表
    
    Returns:
        Adapter 模块字典 {module_name: module}
    """
    adapter_modules = {}
    
    # 方法1: 使用用户指定的名称
    for name in adapter_module_names:
        try:
            module = _get_module_by_name(model, name)
            if module is not None:
                adapter_modules[name] = module
        except AttributeError:
            warnings.warn(f"未找到指定的 Adapter 模块: {name}")
    
    # 方法2: 自动查找包含 "adapter" 的模块
    for name, module in model.named_modules():
        name_lower = name.lower()
        if 'adapter' in name_lower and name not in adapter_modules:
            # 检查是否是 Adapter 类的实例
            if isinstance(module, nn.Module) and hasattr(module, 'down_proj'):
                adapter_modules[name] = module
            elif isinstance(module, nn.ModuleList):
                # 如果是 ModuleList，添加其中的每个 Adapter
                for idx, sub_module in enumerate(module):
                    if hasattr(sub_module, 'down_proj') or hasattr(sub_module, 'adapter'):
                        adapter_modules[f"{name}[{idx}]"] = sub_module
    
    # 方法3: 查找类型为 Adapter 的模块
    for name, module in model.named_modules():
        if name and name not in adapter_modules:
            # 检查模块类型名称
            module_type = type(module).__name__
            if 'Adapter' in module_type:
                adapter_modules[name] = module
    
    return adapter_modules


def _find_decoder_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """
    查找模型中的 Decoder 模块（通常也需要训练）

    Args:
        model: 模型实例

    Returns:
        Decoder 模块字典 {module_name: module}
    """
    decoder_modules = {}

    for name, module in model.named_modules():
        name_lower = name.lower()
        if 'decoder' in name_lower and name:
            decoder_modules[name] = module

    return decoder_modules


def _find_projection_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """
    查找模型中的投影层模块（用于多模态对比学习）

    这些模块通常命名为：
    - text_proj / text_projector / text_projection
    - image_proj / image_projector / image_projection

    Args:
        model: 模型实例

    Returns:
        投影层模块字典 {module_name: module}
    """
    projection_modules = {}

    # 直接查找顶层投影层模块
    if hasattr(model, 'text_proj') and isinstance(model.text_proj, nn.Module):
        projection_modules['text_proj'] = model.text_proj

    if hasattr(model, 'image_proj') and isinstance(model.image_proj, nn.Module):
        projection_modules['image_proj'] = model.image_proj

    # 兼容性：查找其他可能的命名（text_projector, image_projector 等）
    for name, module in model.named_modules():
        if not name:
            continue
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ['text_proj', 'image_proj', 'text_projection', 'image_projection']):
            if name not in projection_modules:
                projection_modules[name] = module

    return projection_modules


def _get_module_by_name(model: nn.Module, module_name: str) -> Optional[nn.Module]:
    """
    根据名称获取模块
    
    Args:
        model: 模型实例
        module_name: 模块名称（支持点号分隔的路径）
    
    Returns:
        模块实例或 None
    """
    try:
        parts = module_name.split('.')
        module = model
        for part in parts:
            # 处理索引（如 "adapters[0]"）
            if '[' in part and ']' in part:
                base_name = part[:part.index('[')]
                idx = int(part[part.index('[')+1:part.index(']')])
                module = getattr(module, base_name)[idx]
            else:
                module = getattr(module, part)
        return module
    except (AttributeError, IndexError, ValueError):
        return None


def _print_freezing_stats(
    stats: Dict[str, Any],
    adapter_modules: Dict[str, nn.Module],
    decoder_modules: Dict[str, nn.Module],
    projection_modules: Dict[str, nn.Module] = None
) -> None:
    """打印冻结统计信息"""
    if projection_modules is None:
        projection_modules = {}

    print("=" * 60)
    print("模型参数冻结统计")
    print("=" * 60)

    print(f"\n总参数数量: {stats['total_params']:,}")
    print(f"冻结参数数量: {stats['frozen_params']:,} ({stats['frozen_params']/stats['total_params']*100:.2f}%)")
    print(f"可训练参数数量: {stats['trainable_params']:,} ({stats['trainable_params']/stats['total_params']*100:.2f}%)")

    if adapter_modules:
        print(f"\n找到 {len(adapter_modules)} 个 Adapter 模块:")
        for name in adapter_modules.keys():
            module = adapter_modules[name]
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  - {name}: {num_params:,} 个可训练参数")

    if decoder_modules:
        print(f"\n找到 {len(decoder_modules)} 个 Decoder 模块:")
        for name in decoder_modules.keys():
            module = decoder_modules[name]
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  - {name}: {num_params:,} 个可训练参数")

    # ★ 新增：打印投影层统计
    if projection_modules:
        print(f"\n找到 {len(projection_modules)} 个 Projection 模块（多模态对比学习）:")
        for name in projection_modules.keys():
            module = projection_modules[name]
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  - {name}: {num_params:,} 个可训练参数")

    if stats['frozen_modules']:
        print(f"\n冻结的主要模块 ({len(stats['frozen_modules'])} 个):")
        for name in stats['frozen_modules'][:10]:  # 只显示前10个
            print(f"  - {name}")
        if len(stats['frozen_modules']) > 10:
            print(f"  ... 还有 {len(stats['frozen_modules']) - 10} 个模块")

    print("=" * 60)


def verify_frozen_state(model: nn.Module, verbose: bool = True) -> bool:
    """
    验证模型的冻结状态
    
    Args:
        model: 模型实例
        verbose: 是否打印详细信息
    
    Returns:
        是否所有非 Adapter/Decoder 模块都已冻结
    """
    # 查找 Adapter 和 Decoder 模块
    adapter_modules = _find_adapter_modules(model, [])
    decoder_modules = _find_decoder_modules(model)
    trainable_module_names = set(adapter_modules.keys()) | set(decoder_modules.keys())
    
    # 检查所有参数
    all_frozen = True
    issues = []
    
    for name, param in model.named_parameters():
        # 检查参数是否属于 Adapter 或 Decoder
        is_trainable_module = any(
            name.startswith(module_name) or module_name in name
            for module_name in trainable_module_names
        )
        
        if param.requires_grad and not is_trainable_module:
            all_frozen = False
            issues.append(f"{name}: 应该冻结但 requires_grad=True")
        elif not param.requires_grad and is_trainable_module:
            issues.append(f"{name}: 应该可训练但 requires_grad=False")
    
    if verbose:
        if all_frozen and not issues:
            print("✓ 模型冻结状态验证通过")
        else:
            print("✗ 模型冻结状态验证失败:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... 还有 {len(issues) - 10} 个问题")
    
    return all_frozen and not issues


def get_trainable_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    获取所有可训练参数
    
    Args:
        model: 模型实例
    
    Returns:
        可训练参数列表
    """
    return [p for p in model.parameters() if p.requires_grad]


def get_frozen_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    获取所有冻结参数
    
    Args:
        model: 模型实例
    
    Returns:
        冻结参数列表
    """
    return [p for p in model.parameters() if not p.requires_grad]


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("freeze_backbone 函数测试")
    print("=" * 60)
    
    # 导入模型
    try:
        from src.model import SAM3_Medical
        
        # 创建模型
        model = SAM3_Medical()
        
        print("\n1. 测试 freeze_backbone 函数")
        stats = freeze_backbone(model, verbose=True)
        
        print("\n2. 验证冻结状态")
        is_valid = verify_frozen_state(model, verbose=True)
        
        print("\n3. 获取可训练参数")
        trainable_params = get_trainable_parameters(model)
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"可训练参数总数: {total_trainable:,}")
        
        print("\n4. 获取冻结参数")
        frozen_params = get_frozen_parameters(model)
        total_frozen = sum(p.numel() for p in frozen_params)
        print(f"冻结参数总数: {total_frozen:,}")
        
        print("\n5. 验证参数总和")
        total_all = total_trainable + total_frozen
        print(f"总参数: {total_all:,}")
        print(f"与统计信息一致: {total_all == stats['total_params']}")
        
    except ImportError as e:
        print(f"无法导入模型: {e}")
        print("请确保模型类已正确定义")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

