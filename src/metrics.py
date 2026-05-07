"""医学图像分割评估指标：Dice、IoU、HD95，以及 BraTS 2020 专用评估器。"""

import torch
import numpy as np
from typing import Dict, Optional

try:
    from medpy import metric
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False
    print("警告: 未安装 medpy，HD95 计算将使用 scipy 实现。安装方法: pip install medpy")

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: 未安装 scipy，HD95 计算将不可用。安装方法: pip install scipy")


def select_best_mask(
    preds: torch.Tensor,
    iou_scores: torch.Tensor
) -> torch.Tensor:
    """
    SAM 架构专属：按 IoU 置信度选出最优掩码通道。

    ⚠️  仅限推理/验证阶段使用。训练阶段 IoU Head 输出为随机值，
    用随机分数选通道会给 Mask Decoder 引入混乱监督信号。

    Args:
        preds:      (B, 3, H, W) 模型输出掩码概率
        iou_scores: (B, 3) SAM 预测的 IoU 置信度
    Returns:
        (B, 1, H, W) 最优掩码
    """
    B = preds.shape[0]
    best_indices = torch.argmax(iou_scores, dim=1)  # (B,)
    best_preds = preds[
        torch.arange(B, device=preds.device),
        best_indices, :, :
    ].unsqueeze(1)  # (B, 1, H, W)
    return best_preds


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    use_medpy: bool = True
) -> float:
    """
    计算 HD95（95th percentile Hausdorff Distance）。

    优先使用 medpy，不可用时退回 scipy 逐点实现。
    预测或真实掩码全为背景时返回 inf。

    Args:
        pred:      (H, W) 二值预测掩码（或概率图，内部自动阈值化）
        target:    (H, W) 二值真实掩码
        use_medpy: 是否优先使用 medpy
    Returns:
        HD95 值（像素单位），或 float('inf')
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.bool_)
    target = (target > 0.5).astype(np.bool_)

    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    if use_medpy and HAS_MEDPY:
        try:
            return float(metric.binary.hd95(pred, target))
        except Exception:
            if not HAS_SCIPY:
                return float('inf')

    if not HAS_SCIPY:
        return float('inf')

    pred_u8 = pred.astype(np.uint8)
    target_u8 = target.astype(np.uint8)
    pred_edges = np.argwhere(pred_u8 > 0)
    target_edges = np.argwhere(target_u8 > 0)

    if len(pred_edges) == 0 or len(target_edges) == 0:
        return float('inf')

    distances = []
    for point in pred_edges:
        distances.append(np.sqrt(((target_edges - point) ** 2).sum(axis=1)).min())
    for point in target_edges:
        distances.append(np.sqrt(((pred_edges - point) ** 2).sum(axis=1)).min())

    return float(np.percentile(np.array(distances), 95))


def compute_metrics_from_binary(
    pred_binary: torch.Tensor,
    target: torch.Tensor,
    compute_hd95: bool = False
) -> Dict[str, float]:
    """
    SAM3 专用评估函数：接受已完成 sigmoid→select_best_mask→>0.5 二值化的预测掩码，
    直接计算 Dice/IoU/HD95，内部不再执行任何激活操作。

    Args:
        pred_binary : (B, 1, H, W)，值域严格为 {0.0, 1.0}
        target      : (B, C, H, W) 或 (B, 1, H, W)，值域 [0, 1]
        compute_hd95: 是否计算 HD95（较慢，默认 False）
    Returns:
        {'dice': float, 'iou': float} 以及可选的 {'hd95': float}
    """
    smooth = 1e-6
    batch_size = pred_binary.shape[0]

    with torch.no_grad():
        target_f = target.float()
        if target_f.shape[1] > 1:
            target_binary = (target_f.max(dim=1, keepdim=True).values > 0.5).float()
        else:
            target_binary = (target_f > 0.5).float()

        dice_scores: list = []
        iou_scores_list: list = []
        hd95_scores: list = []

        for i in range(batch_size):
            p = pred_binary[i, 0].float()  # (H, W)
            t = target_binary[i, 0]        # (H, W)

            intersection = (p * t).sum()
            dice_scores.append(((2. * intersection + smooth) / (p.sum() + t.sum() + smooth)).item())

            union = p.sum() + t.sum() - intersection
            iou_scores_list.append(((intersection + smooth) / (union + smooth)).item())

            if compute_hd95:
                try:
                    p_np = p.cpu().numpy().astype(np.bool_)
                    t_np = t.cpu().numpy().astype(np.bool_)
                    hd95_scores.append(hausdorff_distance_95(p_np, t_np, use_medpy=True))
                except Exception:
                    hd95_scores.append(float('inf'))

    results: Dict[str, float] = {
        'dice': float(np.mean(dice_scores)),
        'iou' : float(np.mean(iou_scores_list)),
    }
    if compute_hd95:
        valid_hd95 = [h for h in hd95_scores if h != float('inf')]
        results['hd95'] = float(np.mean(valid_hd95)) if valid_hd95 else float('inf')

    return results


# ==================== BraTS 2020 专用评估模块 ====================

try:
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.transforms import AsDiscrete
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    print("警告: 未安装 monai，MedicalMetricsCalculator 将不可用。安装方法: pip install monai")


class MedicalMetricsCalculator:
    """
    BraTS 2020 专用评估器，基于 MONAI 计算 WT/TC/ET 三区域 Dice 和 HD95。

    BraTS 2020 标签：0=背景, 1=NCR/NET, 2=ED, 4=ET
    区域定义：WT=1+2+4，TC=1+4，ET=4
    """

    def __init__(self, device: str = "cuda"):
        if not HAS_MONAI:
            raise ImportError("需要安装 monai。安装方法: pip install monai")

        self.device = device
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            ignore_empty=True
        )
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction="mean",
            directed=False
        )
        self.as_discrete = AsDiscrete(argmax=True, to_onehot=None)
        self.brats_labels = [0, 1, 2, 4]

    def _convert_to_brats_labels(self, pred: torch.Tensor) -> torch.Tensor:
        """将模型连续索引 [0,1,2,3] 映射到 BraTS 标签值 [0,1,2,4]。"""
        label_map = torch.tensor([0, 1, 2, 4], device=pred.device, dtype=pred.dtype)
        return label_map[torch.clamp(pred.long(), 0, 3)]

    def _create_region_mask(self, labels: torch.Tensor, region: str) -> torch.Tensor:
        """根据项目 mask 定义生成二值掩码（WT/ET）。"""
        if region == "WT":
            return ((labels == 2) | (labels == 4)).float()  # ch1=WT → label∈{2,4}
        elif region == "ET":
            return (labels == 4).float()
        else:
            raise ValueError(f"未知区域: {region}。支持: WT, ET")

    def _prepare_predictions(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        将模型输出转换为 BraTS 像素级标签 (B, H, W)。

        支持三种输入格式：
        - (B, 1, H, W)：单通道二分类 logits → sigmoid > 0.5 → label∈{0,1}
        - (B, 3, H, W)：多标签独立二分类 (WT/TC/ET) → 按区域嵌套规则反推像素标签
        - (B, C≥4, H, W)：多类分类 → softmax + argmax → BraTS 标签转换
        - (B, H, W)：已是类别索引 → BraTS 标签转换
        """
        if y_pred.dim() == 4 and y_pred.shape[1] == 1:
            pred_binary = torch.sigmoid(y_pred[:, 0]) > 0.5
            return pred_binary.long()

        elif y_pred.dim() == 4 and y_pred.shape[1] > 1:
            C = y_pred.shape[1]
            if C == 3:
                # 多标签输出：3通道格式 (BG/WT/ET)，与数据集 mask 完全对齐
                # channel 0 = BG（不参与肿瘤区域还原）
                # channel 1 = WT, channel 2 = ET
                pred_wt = torch.sigmoid(y_pred[:, 1]) > 0.5
                pred_et = torch.sigmoid(y_pred[:, 2]) > 0.5
                # 反推 BraTS 标签：ET(4) > 非ET前景(2/ED保守代入) > BG(0)
                brats_pred = torch.zeros_like(pred_wt, dtype=torch.long)
                brats_pred[pred_wt & ~pred_et] = 2
                brats_pred[pred_et] = 4
                return brats_pred
            else:
                y_pred = torch.softmax(y_pred, dim=1).argmax(dim=1)
                return self._convert_to_brats_labels(y_pred)

        elif y_pred.dim() == 3:
            return self._convert_to_brats_labels(torch.clamp(y_pred.long(), 0, 3))

        else:
            raise ValueError(f"不支持的预测张量维度: {y_pred.shape}")

    def _prepare_ground_truth(self, y: torch.Tensor) -> torch.Tensor:
        """将真实标签统一转为 BraTS 像素级标签 (B, H, W)。"""
        if y.dim() == 5:
            y = y.squeeze(2) if y.shape[2] == 1 else y[:, :, y.shape[2] // 2, :, :]

        if y.dim() == 4 and y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.dim() == 4 and y.shape[1] > 1:
            y = y.argmax(dim=1)

        if y.max() <= 3:
            y = self._convert_to_brats_labels(y)

        return y

    def calculate_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算 BraTS WT/ET 两区域的 Dice 和 HD95。

        Args:
            y_pred: 模型输出 (B, C, H, W) logits 或 (B, H, W) 索引
            y:      真实标签，支持 (B, H, W)/(B, C, H, W)/(B, C, D, H, W)
        Returns:
            {'WT_Dice', 'ET_Dice', 'WT_HD95', 'ET_HD95', 'Mean_Dice', 'Mean_HD95'}
        """
        y_pred = y_pred.to(self.device)
        y = y.to(self.device)

        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)

        pred_brats = self._prepare_predictions(y_pred)
        gt_brats = self._prepare_ground_truth(y)

        results = {}
        for region in ["WT", "ET"]:
            pred_mask = self._create_region_mask(pred_brats, region)
            gt_mask   = self._create_region_mask(gt_brats, region)

            pred_onehot = torch.stack([1 - pred_mask, pred_mask], dim=1)
            gt_onehot   = torch.stack([1 - gt_mask,   gt_mask],   dim=1)

            if pred_onehot.dim() == 3:
                pred_onehot = pred_onehot.unsqueeze(0)
                gt_onehot   = gt_onehot.unsqueeze(0)

            self.dice_metric.reset()
            self.dice_metric(y_pred=pred_onehot, y=gt_onehot)
            dice_score = self.dice_metric.aggregate().item()

            self.hd95_metric.reset()
            self.hd95_metric(y_pred=pred_onehot, y=gt_onehot)
            hd95_score = self.hd95_metric.aggregate().item()

            if not torch.any(gt_mask > 0):
                hd95_score = float('inf')

            results[f'{region}_Dice'] = dice_score
            results[f'{region}_HD95'] = hd95_score

        dice_scores = [results['WT_Dice'], results['ET_Dice']]
        hd95_scores = [h for h in [results['WT_HD95'], results['ET_HD95']]
                       if h != float('inf')]

        results['Mean_Dice'] = float(np.mean(dice_scores))
        results['Mean_HD95'] = float(np.mean(hd95_scores)) if hd95_scores else float('inf')

        return results
