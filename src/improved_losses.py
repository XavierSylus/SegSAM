"""
改进的损失函数 - 解决医学影像分割中的类别不均衡和模型坍塌问题

问题诊断:
1. Seg Loss 快速降到 0，但 Dice 始终为 0
   → 原因: BCELoss 在极度不均衡数据上会塌陷，模型学会只预测背景
2. Logits 异常低（-200 ~ -250）
   → 原因: 可能是 Cream Loss 权重过大，或者初始化问题
3. 空标签切片导致梯度消失
   → 原因: 当 target 全为 0 时，BCE Loss 梯度很小

解决方案:
1. 使用 DiceFocalLoss 替代纯 BCE/Focal Loss
2. 添加 sigmoid 参数支持
3. 添加空掩码检测和处理
4. 提供多种组合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss - 专门用于分割任务，对类别不均衡更鲁棒

    优势:
    - 直接优化 Dice 系数，与评估指标一致
    - 对类别不均衡不敏感
    - 在前景像素很少时也能提供有意义的梯度
    """

    def __init__(self,
                 smooth: float = 1.0,
                 sigmoid: bool = True,
                 squared_pred: bool = False,
                 reduction: str = 'mean'):
        """
        Args:
            smooth: 平滑项，避免除零（推荐 1.0）
            sigmoid: 是否对 logits 应用 sigmoid（推荐 True）
            squared_pred: 是否对预测平方（增加对困难样本的惩罚）
            reduction: 'mean', 'sum', 'none'
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid
        self.squared_pred = squared_pred
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits (B, C, H, W) 或 (B, 1, H, W)
            target: 真实标签 (B, C, H, W) 或 (B, 1, H, W)，值 0 或 1
        Returns:
            Dice Loss
        """
        # 应用 sigmoid
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        # 计算 Dice
        intersection = (pred * target).sum(dim=1)

        if self.squared_pred:
            denominator = (pred * pred).sum(dim=1) + target.sum(dim=1)
        else:
            denominator = pred.sum(dim=1) + target.sum(dim=1)

        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice_score

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 解决极度类别不均衡问题

    参考: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    优势:
    - 自动降低简单样本（背景）的权重
    - 聚焦于困难样本（边界、小目标）
    - 通过 gamma 参数控制聚焦程度
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 sigmoid: bool = True,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: 前景权重（推荐 0.25，背景权重为 1-alpha=0.75）
            gamma: 聚焦参数（推荐 2.0，越大越关注困难样本）
            sigmoid: 是否对 logits 应用 sigmoid
            reduction: 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = sigmoid
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits (B, C, H, W)
            target: 真实标签 (B, C, H, W)，值 0 或 1
        Returns:
            Focal Loss
        """
        # 计算 BCE（不做 reduction）
        if self.sigmoid:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        else:
            bce = F.binary_cross_entropy(pred, target, reduction='none')

        # 计算 p_t
        p_t = torch.exp(-bce)

        # 计算 Focal weight
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma

        # 应用权重
        focal_loss = focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceFocalLoss(nn.Module):
    """
    Dice + Focal 组合损失 - 推荐用于医学影像分割

    优势:
    - Dice Loss: 优化分割质量，对类别不均衡鲁棒
    - Focal Loss: 关注困难样本，提供像素级监督
    - 两者互补，避免模型坍塌

    适用场景:
    - 极度类别不均衡（前景 < 1%）
    - 小目标分割
    - 联邦学习中每轮只跑少量 Batch
    """

    def __init__(self,
                 dice_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 dice_smooth: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 sigmoid: bool = True):
        """
        Args:
            dice_weight: Dice Loss 权重（推荐 1.0）
            focal_weight: Focal Loss 权重（推荐 1.0）
            dice_smooth: Dice 平滑项（推荐 1.0）
            focal_alpha: Focal 前景权重（推荐 0.25）
            focal_gamma: Focal 聚焦参数（推荐 2.0）
            sigmoid: 是否对 logits 应用 sigmoid（推荐 True）
        """
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.sigmoid = sigmoid

        self.dice_loss = DiceLoss(smooth=dice_smooth, sigmoid=sigmoid)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, sigmoid=sigmoid)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: 预测 logits (B, C, H, W)
            target: 真实标签 (B, C, H, W)
        Returns:
            total_loss, loss_dict
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        total_loss = self.dice_weight * dice + self.focal_weight * focal

        loss_dict = {
            'dice_loss': dice.item(),
            'focal_loss': focal.item(),
            'total_seg_loss': total_loss.item()
        }

        return total_loss, loss_dict


class DiceCELoss(nn.Module):
    """
    Dice + CE 组合损失 - MONAI 推荐的组合

    参考: MONAI DiceCELoss

    优势:
    - Dice: 全局优化，关注区域重叠
    - CE: 局部优化，提供像素级监督
    - 比 DiceFocal 更稳定，但对极度不均衡数据效果稍差
    """

    def __init__(self,
                 dice_weight: float = 1.0,
                 ce_weight: float = 1.0,
                 dice_smooth: float = 1.0,
                 sigmoid: bool = True):
        """
        Args:
            dice_weight: Dice Loss 权重
            ce_weight: CE Loss 权重
            dice_smooth: Dice 平滑项
            sigmoid: 是否对 logits 应用 sigmoid
        """
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.sigmoid = sigmoid

        self.dice_loss = DiceLoss(smooth=dice_smooth, sigmoid=sigmoid)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: 预测 logits (B, C, H, W)
            target: 真实标签 (B, C, H, W)
        Returns:
            total_loss, loss_dict
        """
        dice = self.dice_loss(pred, target)

        # BCE with Logits
        if self.sigmoid:
            ce = F.binary_cross_entropy_with_logits(pred, target)
        else:
            ce = F.binary_cross_entropy(pred, target)

        total_loss = self.dice_weight * dice + self.ce_weight * ce

        loss_dict = {
            'dice_loss': dice.item(),
            'ce_loss': ce.item(),
            'total_seg_loss': total_loss.item()
        }

        return total_loss, loss_dict


class RobustSegmentationLoss(nn.Module):
    """
    健壮的分割损失 - 专门处理空标签和极端情况

    特性:
    1. 检测空标签（target 全为 0）
    2. 空标签时返回小的损失值，避免梯度爆炸
    3. 检测异常 logits（过大或过小）
    4. 提供详细的损失日志
    """

    def __init__(self,
                 base_loss: str = 'dice_focal',  # 'dice_focal', 'dice_ce', 'focal', 'dice'
                 dice_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 dice_smooth: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 sigmoid: bool = True,
                 empty_label_loss: float = 0.1,  # 空标签时返回的损失值
                 warn_logit_threshold: float = 100.0):  # Logit 异常阈值
        """
        Args:
            base_loss: 基础损失类型
            dice_weight: Dice 权重
            focal_weight: Focal 权重
            dice_smooth: Dice 平滑项
            focal_alpha: Focal alpha
            focal_gamma: Focal gamma
            sigmoid: 是否使用 sigmoid
            empty_label_loss: 空标签时返回的损失值（避免梯度消失）
            warn_logit_threshold: Logit 警告阈值
        """
        super(RobustSegmentationLoss, self).__init__()
        self.sigmoid = sigmoid
        self.empty_label_loss = empty_label_loss
        self.warn_logit_threshold = warn_logit_threshold

        # 选择基础损失
        if base_loss == 'dice_focal':
            self.base_criterion = DiceFocalLoss(
                dice_weight=dice_weight,
                focal_weight=focal_weight,
                dice_smooth=dice_smooth,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                sigmoid=sigmoid
            )
        elif base_loss == 'dice_ce':
            self.base_criterion = DiceCELoss(
                dice_weight=dice_weight,
                ce_weight=focal_weight,
                dice_smooth=dice_smooth,
                sigmoid=sigmoid
            )
        elif base_loss == 'focal':
            self.base_criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                sigmoid=sigmoid
            )
        elif base_loss == 'dice':
            self.base_criterion = DiceLoss(
                smooth=dice_smooth,
                sigmoid=sigmoid
            )
        else:
            raise ValueError(f"未知的损失类型: {base_loss}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: 预测 logits (B, C, H, W)
            target: 真实标签 (B, C, H, W)
        Returns:
            loss, info_dict
        """
        # 检查空标签
        foreground_pixels = target.sum().item()
        total_pixels = target.numel()
        foreground_ratio = foreground_pixels / total_pixels

        # 检测异常 logits
        logit_min = pred.min().item()
        logit_max = pred.max().item()
        logit_mean = pred.mean().item()

        info_dict = {
            'foreground_pixels': int(foreground_pixels),
            'total_pixels': int(total_pixels),
            'foreground_ratio': foreground_ratio,
            'logit_min': logit_min,
            'logit_max': logit_max,
            'logit_mean': logit_mean
        }

        # 异常检测
        if abs(logit_min) > self.warn_logit_threshold or abs(logit_max) > self.warn_logit_threshold:
            info_dict['warning'] = f"异常 Logits 检测: min={logit_min:.2f}, max={logit_max:.2f}"

        # 空标签处理
        if foreground_pixels == 0:
            info_dict['is_empty_label'] = True
            # 返回一个小的损失值，鼓励模型预测背景
            loss = torch.tensor(self.empty_label_loss, device=pred.device, requires_grad=True)
            info_dict['seg_loss'] = self.empty_label_loss
            return loss, info_dict

        # 正常计算损失
        info_dict['is_empty_label'] = False
        if isinstance(self.base_criterion, (DiceFocalLoss, DiceCELoss)):
            loss, loss_dict = self.base_criterion(pred, target)
            info_dict.update(loss_dict)
        else:
            loss = self.base_criterion(pred, target)
            info_dict['seg_loss'] = loss.item()

        return loss, info_dict


# ============================================================
# 使用示例和对比测试
# ============================================================

def test_losses():
    """测试不同损失函数在空标签和正常标签上的表现"""
    print("=" * 80)
    print("损失函数对比测试")
    print("=" * 80)

    # 创建测试数据
    batch_size = 2
    height, width = 256, 256

    # 场景 1: 正常标签（5% 前景）
    pred_normal = torch.randn(batch_size, 1, height, width) * 0.1  # 小的初始 logits
    target_normal = torch.zeros(batch_size, 1, height, width)
    target_normal[:, :, 100:150, 100:150] = 1  # 5% 前景

    # 场景 2: 空标签（0% 前景）
    pred_empty = torch.randn(batch_size, 1, height, width) * 0.1
    target_empty = torch.zeros(batch_size, 1, height, width)

    # 场景 3: 异常 logits（-200）
    pred_abnormal = torch.ones(batch_size, 1, height, width) * -200
    target_abnormal = torch.zeros(batch_size, 1, height, width)
    target_abnormal[:, :, 100:150, 100:150] = 1

    # 测试不同损失函数
    losses = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Focal': FocalLoss(sigmoid=True),
        'Dice': DiceLoss(sigmoid=True),
        'DiceFocal': DiceFocalLoss(sigmoid=True),
        'DiceCE': DiceCELoss(sigmoid=True),
        'Robust': RobustSegmentationLoss(base_loss='dice_focal', sigmoid=True)
    }

    scenarios = [
        ("正常标签 (5% 前景)", pred_normal, target_normal),
        ("空标签 (0% 前景)", pred_empty, target_empty),
        ("异常 Logits (-200)", pred_abnormal, target_abnormal)
    ]

    for scenario_name, pred, target in scenarios:
        print(f"\n{scenario_name}")
        print("-" * 80)
        for loss_name, criterion in losses.items():
            try:
                if isinstance(criterion, (DiceFocalLoss, DiceCELoss, RobustSegmentationLoss)):
                    loss, info = criterion(pred, target)
                    print(f"{loss_name:12s}: Loss={loss.item():.4f}, Info={info}")
                else:
                    loss = criterion(pred, target)
                    print(f"{loss_name:12s}: Loss={loss.item():.4f}")
            except Exception as e:
                print(f"{loss_name:12s}: ERROR - {e}")


if __name__ == "__main__":
    test_losses()
