"""
健壮的医学影像分割指标计算模块

问题诊断:
1. 当 Ground Truth 为空时，HD95 计算会报错或返回 N/A
2. 当 Ground Truth 为空但预测也为空时，Dice 被错误地计为 1.0
3. 当只有 GT 为空或只有 Pred 为空时，指标处理不一致

解决方案:
1. 统一的空掩码检测逻辑
2. 明确的返回值约定（使用 -1 表示"不适用"）
3. 提供详细的诊断信息
4. 支持 BraTS 多区域评估
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
import warnings

# 尝试导入 medpy（FedFMS 使用的方法）
try:
    from medpy import metric
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False

# 尝试导入 scipy（用于 HD95 计算，作为 medpy 的备选）
try:
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 尝试导入 MONAI
try:
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class RobustMetricsCalculator:
    """
    健壮的医学影像分割指标计算器

    特性:
    1. 处理空掩码（GT 为空、Pred 为空、两者都为空）
    2. 返回值约定：
       - 正常情况：返回实际指标值 [0, 1] 或 [0, +inf)
       - GT 为空且 Pred 为空：Dice=1.0, HD95=-1（完美匹配，但无意义）
       - GT 为空但 Pred 非空：Dice=0.0, HD95=-1（假阳性）
       - GT 非空但 Pred 为空：Dice=0.0, HD95=-1（假阴性）
    3. 提供详细的诊断信息
    """

    def __init__(self,
                 device: str = 'cuda',
                 percentile: int = 95,
                 empty_value: float = -1.0):
        """
        Args:
            device: 计算设备
            percentile: Hausdorff 距离百分位（默认 95）
            empty_value: 空掩码时的默认返回值（默认 -1.0，表示"不适用"）
        """
        self.device = device
        self.percentile = percentile
        self.empty_value = empty_value

    def is_empty(self, mask: Union[torch.Tensor, np.ndarray]) -> bool:
        """
        检查掩码是否为空（全为 0）

        Args:
            mask: 二值掩码

        Returns:
            是否为空
        """
        if isinstance(mask, torch.Tensor):
            return mask.sum().item() == 0
        else:
            return np.sum(mask) == 0

    def calculate_dice(self,
                       pred: torch.Tensor,
                       target: torch.Tensor,
                       smooth: float = 1e-6) -> Tuple[float, Dict]:
        """
        计算 Dice 系数（健壮版本）

        Args:
            pred: 预测掩码 (B, C, H, W) 或 (B, H, W)，logits 或概率
            target: 真实掩码 (B, C, H, W) 或 (B, H, W)，二值
            smooth: 平滑项

        Returns:
            (dice_score, info_dict)
        """
        # 转换为二值掩码
        pred_binary = self._to_binary(pred)
        target_binary = self._to_binary(target)

        # 检查空掩码
        pred_empty = self.is_empty(pred_binary)
        target_empty = self.is_empty(target_binary)

        info = {
            'pred_empty': pred_empty,
            'target_empty': target_empty,
            'pred_sum': pred_binary.sum().item(),
            'target_sum': target_binary.sum().item()
        }

        # 处理空掩码情况
        if target_empty and pred_empty:
            # 两者都为空：完美匹配，但无意义
            dice = 1.0
            info['status'] = 'both_empty'
        elif target_empty:
            # GT 为空，Pred 非空：假阳性
            dice = 0.0
            info['status'] = 'target_empty_pred_nonempty'
        elif pred_empty:
            # GT 非空，Pred 为空：假阴性
            dice = 0.0
            info['status'] = 'target_nonempty_pred_empty'
        else:
            # 正常情况：计算 Dice
            pred_flat = pred_binary.view(-1)
            target_flat = target_binary.view(-1)
            intersection = (pred_flat * target_flat).sum()
            dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
            dice = dice.item()
            info['status'] = 'normal'

        return dice, info

    def calculate_hd95(self,
                       pred: torch.Tensor,
                       target: torch.Tensor,
                       spacing: Optional[Tuple] = None) -> Tuple[float, Dict]:
        """
        计算 95% Hausdorff 距离（健壮版本）

        Args:
            pred: 预测掩码 (B, C, H, W) 或 (B, H, W)
            target: 真实掩码 (B, C, H, W) 或 (B, H, W)
            spacing: 像素间距（用于物理距离计算）

        Returns:
            (hd95, info_dict)
        """
        # 转换为二值掩码
        pred_binary = self._to_binary(pred)
        target_binary = self._to_binary(target)

        # 检查空掩码
        pred_empty = self.is_empty(pred_binary)
        target_empty = self.is_empty(target_binary)

        info = {
            'pred_empty': pred_empty,
            'target_empty': target_empty
        }

        # 处理空掩码情况
        if target_empty or pred_empty:
            # 任一为空：HD95 无意义
            hd95 = self.empty_value
            if target_empty and pred_empty:
                info['status'] = 'both_empty'
            elif target_empty:
                info['status'] = 'target_empty_pred_nonempty'
            else:
                info['status'] = 'target_nonempty_pred_empty'
            return hd95, info

        # 正常情况：计算 HD95
        info['status'] = 'normal'

        # 转换为 numpy
        if isinstance(pred_binary, torch.Tensor):
            pred_np = pred_binary.cpu().numpy()
        else:
            pred_np = pred_binary

        if isinstance(target_binary, torch.Tensor):
            target_np = target_binary.cpu().numpy()
        else:
            target_np = target_binary

        # 如果是 4D，取第一个 batch 和第一个 channel
        if pred_np.ndim == 4:
            pred_np = pred_np[0, 0]
            target_np = target_np[0, 0]
        elif pred_np.ndim == 3:
            pred_np = pred_np[0]
            target_np = target_np[0]

        try:
            if HAS_MEDPY:
                # 使用 medpy（FedFMS 方法）
                hd95 = metric.binary.hd95(pred_np, target_np, voxelspacing=spacing)
            elif HAS_SCIPY:
                # 使用 scipy 作为备选
                hd95 = self._compute_hd95_scipy(pred_np, target_np)
            else:
                warnings.warn("未安装 medpy 或 scipy，无法计算 HD95")
                hd95 = self.empty_value
                info['status'] = 'computation_unavailable'
        except Exception as e:
            warnings.warn(f"HD95 计算失败: {e}")
            hd95 = self.empty_value
            info['status'] = 'computation_error'
            info['error'] = str(e)

        return hd95, info

    def calculate_all_metrics(self,
                              pred: torch.Tensor,
                              target: torch.Tensor,
                              spacing: Optional[Tuple] = None) -> Dict:
        """
        计算所有指标

        Args:
            pred: 预测掩码
            target: 真实掩码
            spacing: 像素间距

        Returns:
            指标字典
        """
        dice, dice_info = self.calculate_dice(pred, target)
        hd95, hd95_info = self.calculate_hd95(pred, target, spacing)

        metrics = {
            'Dice': dice,
            'HD95': hd95,
            'dice_info': dice_info,
            'hd95_info': hd95_info
        }

        return metrics

    def _to_binary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        将掩码转换为二值掩码

        Args:
            mask: 输入掩码（可能是 logits、概率或二值）

        Returns:
            二值掩码
        """
        if mask.dtype == torch.float32 or mask.dtype == torch.float16:
            # 检查是否是 logits（范围不在 [0, 1]）
            if mask.min() < 0 or mask.max() > 1:
                # 应用 sigmoid
                mask = torch.sigmoid(mask)
            # 阈值化
            mask = (mask > 0.5).float()
        else:
            # 已经是整数类型，直接二值化
            mask = (mask > 0).float()

        return mask

    def _compute_hd95_scipy(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        使用 scipy 计算 HD95（备选方法）

        Args:
            pred: 预测二值掩码
            target: 真实二值掩码

        Returns:
            HD95
        """
        # 提取边界点
        pred_points = np.argwhere(pred > 0)
        target_points = np.argwhere(target > 0)

        if len(pred_points) == 0 or len(target_points) == 0:
            return self.empty_value

        # 计算双向 Hausdorff 距离
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]

        # 计算 95 百分位
        distances = np.concatenate([
            np.min(np.linalg.norm(pred_points[:, None] - target_points, axis=2), axis=1),
            np.min(np.linalg.norm(target_points[:, None] - pred_points, axis=2), axis=1)
        ])

        hd95 = np.percentile(distances, self.percentile)

        return hd95


class BraTSRobustMetricsCalculator:
    """
    BraTS 专用健壮指标计算器

    支持计算三个区域的指标：
    - WT (Whole Tumor): Label 1 + 2 + 4
    - TC (Tumor Core): Label 1 + 4
    - ET (Enhancing Tumor): Label 4

    处理空区域情况（如某些切片没有 ET）
    """

    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: 计算设备
        """
        self.device = device
        self.base_calculator = RobustMetricsCalculator(device=device)

    def calculate_brats_metrics(self,
                                 pred: torch.Tensor,
                                 target: torch.Tensor,
                                 spacing: Optional[Tuple] = None) -> Dict:
        """
        计算 BraTS 三个区域的指标

        Args:
            pred: 预测标签 (B, C, H, W)，C=4（背景 + 3 类）
            target: 真实标签 (B, C, H, W)，C=4（背景 + 3 类）
            spacing: 像素间距

        Returns:
            指标字典
        """
        # 转换为 BraTS 标签格式 [0, 1, 2, 4]
        pred_labels = self._convert_to_brats_labels(pred)
        target_labels = self._convert_to_brats_labels(target)

        # 定义三个区域
        regions = {
            'WT': [1, 2, 4],  # Whole Tumor
            'TC': [1, 4],     # Tumor Core
            'ET': [4]         # Enhancing Tumor
        }

        metrics = {}

        for region_name, region_labels in regions.items():
            # 创建区域掩码
            pred_region = self._create_region_mask(pred_labels, region_labels)
            target_region = self._create_region_mask(target_labels, region_labels)

            # 计算指标
            region_metrics = self.base_calculator.calculate_all_metrics(
                pred_region, target_region, spacing
            )

            # 添加到结果
            metrics[f'{region_name}_Dice'] = region_metrics['Dice']
            metrics[f'{region_name}_HD95'] = region_metrics['HD95']
            metrics[f'{region_name}_info'] = {
                'dice_info': region_metrics['dice_info'],
                'hd95_info': region_metrics['hd95_info']
            }

        # 计算平均 Dice 和 HD95（只计算有效值）
        valid_dices = [v for k, v in metrics.items() if 'Dice' in k and v >= 0]
        valid_hd95s = [v for k, v in metrics.items() if 'HD95' in k and v >= 0]

        metrics['Mean_Dice'] = np.mean(valid_dices) if valid_dices else -1.0
        metrics['Mean_HD95'] = np.mean(valid_hd95s) if valid_hd95s else -1.0

        return metrics

    def _convert_to_brats_labels(self, mask: torch.Tensor) -> torch.Tensor:
        """
        将模型输出转换为 BraTS 标签格式

        Args:
            mask: (B, C, H, W)，C=4（背景 + 3 类）或 (B, H, W)

        Returns:
            BraTS 标签 (B, H, W)，值为 [0, 1, 2, 4]
        """
        if mask.dim() == 4 and mask.shape[1] > 1:
            # 多通道：argmax
            labels = mask.argmax(dim=1)  # (B, H, W)

            # 转换为 BraTS 标签 [0, 1, 2, 4]
            brats_labels = torch.zeros_like(labels)
            brats_labels[labels == 1] = 1  # 坏死
            brats_labels[labels == 2] = 2  # 水肿
            brats_labels[labels == 3] = 4  # 增强肿瘤
        else:
            # 单通道：直接使用
            if mask.dim() == 4:
                labels = mask[:, 0]
            else:
                labels = mask
            brats_labels = labels

        return brats_labels

    def _create_region_mask(self, labels: torch.Tensor, region_labels: list) -> torch.Tensor:
        """
        根据区域定义创建二值掩码

        Args:
            labels: BraTS 标签 (B, H, W)
            region_labels: 区域包含的标签列表

        Returns:
            二值掩码 (B, 1, H, W)
        """
        region_mask = torch.zeros_like(labels).float()

        for label in region_labels:
            region_mask = region_mask | (labels == label).float()

        return region_mask.unsqueeze(1)  # (B, 1, H, W)


# ============================================================
# 使用示例和测试
# ============================================================

def test_robust_metrics():
    """测试健壮指标计算器"""
    print("=" * 80)
    print("健壮指标计算器测试")
    print("=" * 80)

    calculator = RobustMetricsCalculator()

    # 场景 1: 正常情况
    pred_normal = torch.zeros(1, 1, 256, 256)
    pred_normal[:, :, 100:150, 100:150] = 1
    target_normal = torch.zeros(1, 1, 256, 256)
    target_normal[:, :, 110:160, 110:160] = 1

    # 场景 2: GT 为空，Pred 非空（假阳性）
    pred_fp = torch.zeros(1, 1, 256, 256)
    pred_fp[:, :, 100:150, 100:150] = 1
    target_fp = torch.zeros(1, 1, 256, 256)

    # 场景 3: GT 非空，Pred 为空（假阴性）
    pred_fn = torch.zeros(1, 1, 256, 256)
    target_fn = torch.zeros(1, 1, 256, 256)
    target_fn[:, :, 100:150, 100:150] = 1

    # 场景 4: 两者都为空
    pred_empty = torch.zeros(1, 1, 256, 256)
    target_empty = torch.zeros(1, 1, 256, 256)

    scenarios = [
        ("正常情况", pred_normal, target_normal),
        ("假阳性 (GT空, Pred非空)", pred_fp, target_fp),
        ("假阴性 (GT非空, Pred空)", pred_fn, target_fn),
        ("两者都为空", pred_empty, target_empty)
    ]

    for scenario_name, pred, target in scenarios:
        print(f"\n{scenario_name}")
        print("-" * 80)
        metrics = calculator.calculate_all_metrics(pred, target)
        print(f"Dice: {metrics['Dice']:.4f}")
        print(f"HD95: {metrics['HD95']:.4f}")
        print(f"Dice Info: {metrics['dice_info']}")
        print(f"HD95 Info: {metrics['hd95_info']}")


if __name__ == "__main__":
    test_robust_metrics()
