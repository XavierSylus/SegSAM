"""
模型模块
包含适配器和其他模型组件
"""

from .adapter import (
    Adapter,
    ParallelAdapter,
    SequentialAdapter,
    create_adapter,
)

from .freeze_utils import (
    freeze_backbone,
    verify_frozen_state,
    get_trainable_parameters,
    get_frozen_parameters,
)

__all__ = [
    # 适配器
    'Adapter',
    'ParallelAdapter',
    'SequentialAdapter',
    'create_adapter',
    # 冻结工具
    'freeze_backbone',
    'verify_frozen_state',
    'get_trainable_parameters',
    'get_frozen_parameters',
]

