"""
串行训练辅助工具

此模块提供了在单GPU上实现多客户端串行训练的辅助函数。
通过状态卸载和重载，可以在有限的显存下模拟大规模联邦学习。

主要功能：
1. 可训练参数的提取和加载
2. 模型状态的CPU/GPU转移
3. 内存安全的状态管理
"""

import torch
import torch.nn as nn
from typing import Dict
from collections import OrderedDict


def get_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    提取模型中所有可训练参数的状态字典

    Args:
        model: PyTorch 模型

    Returns:
        包含可训练参数的字典
    """
    trainable_state = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.cpu().clone().detach()

    return trainable_state


def load_trainable_state_dict(
    model: nn.Module,
    state_dict,          # 允许 None，text_only 客户端可能不保存权重
    device: str = "cuda",
    strict: bool = False
) -> None:
    """
    将可训练参数的状态字典加载到模型

    ★ Bug5 Fix (2026-03-03): 增加对 state_dict 为 None 的判空保护。
    text_only 客户端在 Round 1 按设计跳过权重保存（weights=None），
    Round 2 开始时调用本函数会触发 AttributeError: 'NoneType'.items()。
    修复：直接在入口处 return，让该客户端沿用上一轮的全局模型权重（已由
    setup_environment 或上轮 aggregated_state 写入），不做任何覆盖。

    Args:
        model: 目标 PyTorch 模型
        state_dict: 可训练参数的状态字典，允许为 None
        device: 目标设备
        strict: 是否严格匹配参数名称
    """
    # ★ Bug5 Fix：state_dict 为 None 表示该客户端无本地权重缓存（如 text_only），
    # 直接跳过加载，保持模型当前（全局聚合后）的权重不变。
    if state_dict is None:
        print("  [load_state] ⚠ state_dict 为 None（可能是 text_only 客户端），跳过权重加载。")
        return

    full_state_dict = model.state_dict()
    skipped = []
    shape_fixed = []

    for name, param_tensor in state_dict.items():
        if name not in full_state_dict:
            continue

        model_shape = full_state_dict[name].shape
        ckpt_shape = param_tensor.shape

        if model_shape == ckpt_shape:
            # 形状匹配，正常加载
            full_state_dict[name] = param_tensor.to(device)
        else:
            # ★ Bug2 Fix (2026-03-03): 对 _output_conv 这类动态重建层，
            # 形状不匹配时不再静默跳过（会导致该层永远无法参与 FedAvg）。
            # 策略：若 ckpt 的形状与模型当前动态重建后的形状不匹配，
            # 记录日志并跳过（保留模型当前最新动态层），而不是静默覆盖为错误形状。
            # 根本修复在 integrated_model.py 的 _output_conv 初始化中（统一通道数）。
            skipped.append(
                f"{name}: model={tuple(model_shape)} vs ckpt={tuple(ckpt_shape)} → 跳过"
            )

    if skipped:
        print(f"  [load_state] 跳过形状不匹配的参数 ({len(skipped)} 个):")
        for s in skipped:
            print(f"    - {s}")

    model.load_state_dict(full_state_dict, strict=False)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_memory_footprint(model: nn.Module, in_mb: bool = True) -> float:
    """估算模型的内存占用"""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if in_mb:
        return total_bytes / (1024 ** 2)
    else:
        return total_bytes
