"""
可视化工具：用于保存预测结果图像
将 MRI 底图、GT Mask 和 Pred Mask 叠加保存

参考：SAM3 sam3/agent/viz.py 和 sam3/visualization_utils.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Optional, Tuple
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb


def tensor_to_numpy(
    tensor: torch.Tensor,
    denormalize: bool = False,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    将 PyTorch Tensor 转换为 numpy 数组
    
    Args:
        tensor: PyTorch Tensor，形状可以是 (C, H, W) 或 (H, W)
        denormalize: 是否反归一化（默认: False）
        mean: 归一化均值（如果 denormalize=True）
        std: 归一化标准差（如果 denormalize=True）
    
    Returns:
        numpy 数组，形状 (H, W) 或 (H, W, C)
    """
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.array(tensor)
    
    # 处理维度：如果是 (C, H, W)，转换为 (H, W, C)
    if array.ndim == 3 and array.shape[0] == 1 or array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
        # 如果只有一个通道，去掉最后一个维度
        if array.shape[2] == 1:
            array = array.squeeze(2)
    
    # 反归一化
    if denormalize and mean is not None and std is not None:
        if array.ndim == 3:
            for c in range(3):
                array[:, :, c] = array[:, :, c] * std[c] + mean[c]
        else:
            # 单通道图像，假设使用 mean[0] 和 std[0]
            array = array * std[0] + mean[0]
    
    # 确保值在 [0, 1] 范围内
    if array.max() > 1.0:
        array = array / 255.0
    
    array = np.clip(array, 0, 1)
    
    return array


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # 红色
    alpha: float = 0.5
) -> np.ndarray:
    """
    在图像上叠加掩码
    
    Args:
        image: 背景图像，形状 (H, W) 或 (H, W, 3)，值在 [0, 1]
        mask: 掩码，形状 (H, W)，值为 0 或 1（或概率）
        color: 掩码颜色 RGB，值在 [0, 1]（默认: 红色）
        alpha: 透明度（默认: 0.5）
    
    Returns:
        叠加后的图像，形状 (H, W, 3)
    """
    # 确保图像是 3 通道
    if image.ndim == 2:
        # 灰度图转为 RGB
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    # 确保图像值在 [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0, 1)
    
    # 处理掩码：转换为二值掩码
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 处理掩码维度
    if mask.ndim == 3:
        if mask.shape[0] == 1 or mask.shape[2] == 1:
            mask = mask.squeeze()
    
    # 二值化掩码
    mask_binary = (mask > 0.5).astype(np.float32)
    
    # 确保掩码和图像尺寸一致
    if mask_binary.shape != image.shape[:2]:
                try:
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        image.shape[0] / mask_binary.shape[0],
                        image.shape[1] / mask_binary.shape[1]
                    )
                    mask_binary = zoom(mask_binary, zoom_factors, order=0)  # 最近邻插值
                    mask_binary = (mask_binary > 0.5).astype(np.float32)
                except ImportError:
                    # 如果没有 scipy，使用 torch 插值
                    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0).float()
                    mask_tensor = F.interpolate(
                        mask_tensor,
                        size=(image.shape[0], image.shape[1]),
                        mode='nearest'
                    )
                    mask_binary = mask_tensor.squeeze().numpy()
                    mask_binary = (mask_binary > 0.5).astype(np.float32)
    
    # 创建彩色掩码
    color_mask = np.zeros_like(image)
    for c in range(3):
        color_mask[:, :, c] = mask_binary * color[c]
    
    # 叠加：alpha 混合
    overlay = image * (1 - alpha * mask_binary[..., np.newaxis]) + color_mask * alpha
    
    return overlay


def save_prediction_image(
    mri_image: Union[torch.Tensor, np.ndarray],
    gt_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    pred_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Union[str, Path] = "prediction_result.png",
    gt_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),  # 绿色
    pred_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # 红色
    gt_alpha: float = 0.5,
    pred_alpha: float = 0.5,
    layout: str = "overlay",  # "overlay" 或 "side_by_side"
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 100,
    show_legend: bool = True
) -> None:
    """
    保存预测结果图像，将 MRI 底图、GT Mask 和 Pred Mask 叠加保存
    
    参考：SAM3 sam3/agent/viz.py 和 sam3/visualization_utils.py
    
    Args:
        mri_image: MRI 底图，形状 (C, H, W) 或 (H, W) 或 (H, W, C)
        gt_mask: Ground Truth 掩码，形状 (H, W) 或 (C, H, W)（可选）
        pred_mask: 预测掩码，形状 (H, W) 或 (C, H, W)（可选）
        save_path: 保存路径
        gt_color: GT 掩码颜色 RGB，值在 [0, 1]（默认: 绿色）
        pred_color: 预测掩码颜色 RGB，值在 [0, 1]（默认: 红色）
        gt_alpha: GT 掩码透明度（默认: 0.5）
        pred_alpha: 预测掩码透明度（默认: 0.5）
        layout: 布局方式："overlay"（叠加）或 "side_by_side"（并排）
        figsize: 图像大小（默认: 根据布局自动设置）
        dpi: 图像分辨率（默认: 100）
        show_legend: 是否显示图例（默认: True）
    """
    # 转换 MRI 图像为 numpy
    mri_np = tensor_to_numpy(mri_image)
    
    # 获取图像尺寸
    if mri_np.ndim == 2:
        h, w = mri_np.shape
    else:
        h, w = mri_np.shape[:2]
    
    # 设置默认 figsize
    if figsize is None:
        if layout == "overlay":
            figsize = (10, 10)
        else:
            figsize = (16, 8)
    
    if layout == "overlay":
        # 叠加模式：所有内容叠加在一张图上
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 先显示 MRI 底图
        if mri_np.ndim == 2:
            ax.imshow(mri_np, cmap='gray')
        else:
            ax.imshow(mri_np)
        
        # 叠加 GT Mask
        if gt_mask is not None:
            gt_np = tensor_to_numpy(gt_mask)
            # 处理掩码维度
            if gt_np.ndim == 3:
                if gt_np.shape[0] == 1 or gt_np.shape[2] == 1:
                    gt_np = gt_np.squeeze()
            
            # 确保尺寸一致
            if gt_np.shape != (h, w):
                try:
                    from scipy.ndimage import zoom
                    zoom_factors = (h / gt_np.shape[0], w / gt_np.shape[1])
                    gt_np = zoom(gt_np, zoom_factors, order=0)
                except ImportError:
                    # 如果没有 scipy，使用 torch 插值
                    gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float()
                    gt_tensor = F.interpolate(
                        gt_tensor,
                        size=(h, w),
                        mode='nearest'
                    )
                    gt_np = gt_tensor.squeeze().numpy()
            
            gt_binary = (gt_np > 0.5).astype(np.float32)
            mask_img_gt = np.zeros((h, w, 4), dtype=np.float32)
            mask_img_gt[..., :3] = to_rgb(gt_color)
            mask_img_gt[..., 3] = gt_binary * gt_alpha
            ax.imshow(mask_img_gt)
        
        # 叠加 Pred Mask
        if pred_mask is not None:
            pred_np = tensor_to_numpy(pred_mask)
            # 处理掩码维度
            if pred_np.ndim == 3:
                if pred_np.shape[0] == 1 or pred_np.shape[2] == 1:
                    pred_np = pred_np.squeeze()
            
            # 确保尺寸一致
            if pred_np.shape != (h, w):
                try:
                    from scipy.ndimage import zoom
                    zoom_factors = (h / pred_np.shape[0], w / pred_np.shape[1])
                    pred_np = zoom(pred_np, zoom_factors, order=0)
                except ImportError:
                    # 如果没有 scipy，使用 torch 插值
                    pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).float()
                    pred_tensor = F.interpolate(
                        pred_tensor,
                        size=(h, w),
                        mode='nearest'
                    )
                    pred_np = pred_tensor.squeeze().numpy()
            
            pred_binary = (pred_np > 0.5).astype(np.float32)
            mask_img_pred = np.zeros((h, w, 4), dtype=np.float32)
            mask_img_pred[..., :3] = to_rgb(pred_color)
            mask_img_pred[..., 3] = pred_binary * pred_alpha
            ax.imshow(mask_img_pred)
        
        ax.set_title("MRI Image with GT and Prediction Masks", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 添加图例
        if show_legend:
            legend_elements = []
            if gt_mask is not None:
                legend_elements.append(patches.Patch(facecolor=gt_color, label='Ground Truth', alpha=gt_alpha))
            if pred_mask is not None:
                legend_elements.append(patches.Patch(facecolor=pred_color, label='Prediction', alpha=pred_alpha))
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    else:  # side_by_side
        # 并排模式：显示多张图
        num_plots = 1  # MRI 底图
        if gt_mask is not None:
            num_plots += 1
        if pred_mask is not None:
            num_plots += 1
        
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 原始 MRI 图像
        ax = axes[plot_idx]
        if mri_np.ndim == 2:
            ax.imshow(mri_np, cmap='gray')
        else:
            ax.imshow(mri_np)
        ax.set_title("MRI Image", fontsize=12, fontweight='bold')
        ax.axis('off')
        plot_idx += 1
        
        # GT Mask 叠加
        if gt_mask is not None:
            ax = axes[plot_idx]
            overlay_gt = overlay_mask_on_image(mri_np, gt_mask, color=gt_color, alpha=gt_alpha)
            ax.imshow(overlay_gt)
            ax.set_title("Ground Truth", fontsize=12, fontweight='bold')
            ax.axis('off')
            plot_idx += 1
        
        # Pred Mask 叠加
        if pred_mask is not None:
            ax = axes[plot_idx]
            overlay_pred = overlay_mask_on_image(mri_np, pred_mask, color=pred_color, alpha=pred_alpha)
            ax.imshow(overlay_pred)
            ax.set_title("Prediction", fontsize=12, fontweight='bold')
            ax.axis('off')
            plot_idx += 1
    
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"预测结果图像已保存至: {save_path}")


if __name__ == "__main__":
    # 测试代码
    import torch
    
    print("=" * 60)
    print("测试 save_prediction_image 函数")
    print("=" * 60)
    
    # 创建虚拟数据
    h, w = 256, 256
    mri_image = torch.randn(3, h, w)  # RGB 图像
    mri_image = (mri_image - mri_image.min()) / (mri_image.max() - mri_image.min())
    
    gt_mask = torch.zeros(1, h, w)
    gt_mask[0, 50:150, 80:180] = 1.0
    
    pred_mask = torch.zeros(1, h, w)
    pred_mask[0, 60:160, 90:190] = 1.0
    
    # 测试叠加模式
    print("\n1. 测试叠加模式...")
    save_prediction_image(
        mri_image=mri_image,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        save_path="test_prediction_overlay.png",
        layout="overlay",
        show_legend=True
    )
    
    # 测试并排模式
    print("\n2. 测试并排模式...")
    save_prediction_image(
        mri_image=mri_image,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        save_path="test_prediction_side_by_side.png",
        layout="side_by_side",
        show_legend=False
    )
    
    # 测试只有预测掩码
    print("\n3. 测试只有预测掩码...")
    save_prediction_image(
        mri_image=mri_image,
        pred_mask=pred_mask,
        save_path="test_prediction_only.png",
        layout="overlay"
    )
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

