"""
预测结果保存模块
用于保存医学图像分割的预测结果和评估指标

功能：
1. 保存预测掩码为 NIfTI 格式 (.nii.gz)
2. 保存评估指标为 CSV 格式
3. 处理 BraTS 2020 标签映射（0, 1, 2, 4）
"""

import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

# 尝试导入 nibabel
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("nibabel 未安装，无法保存 NIfTI 文件。安装方法: pip install nibabel")


class PredictionSaver:
    """
    预测结果保存器
    
    用于将模型预测结果保存到硬盘，包括：
    - 预测掩码（NIfTI 格式）
    - 评估指标（CSV 格式）
    """
    
    def __init__(self, save_dir: Union[str, Path]):
        """
        初始化保存器
        
        Args:
            save_dir: 保存目录路径
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.pred_masks_dir = self.save_dir / "pred_masks"
        self.metrics_dir = self.save_dir / "metrics"
        self.pred_masks_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # BraTS 2020 标签映射
        # 模型输出索引 [0, 1, 2, 3] -> BraTS 标签值 [0, 1, 2, 4]
        self.label_map = {0: 0, 1: 1, 2: 2, 3: 4}
        self.brats_labels = [0, 1, 2, 4]
        
        if not HAS_NIBABEL:
            warnings.warn(
                "nibabel 未安装，save_batch_nifti 功能将不可用。"
                "安装方法: pip install nibabel"
            )
    
    def _convert_to_brats_labels(self, pred: np.ndarray) -> np.ndarray:
        """
        将模型输出的类别索引转换为 BraTS 2020 标签值
        
        模型输出可能是索引 [0, 1, 2, 3]，需要转换为 BraTS 标签 [0, 1, 2, 4]
        
        Args:
            pred: 预测的类别索引 (H, W) 或 (H, W, D)，值在 [0, 1, 2, 3]
        
        Returns:
            转换后的 BraTS 标签，值在 [0, 1, 2, 4]
        """
        # 创建标签映射数组
        brats_pred = np.zeros_like(pred, dtype=np.uint8)
        
        # 映射索引到 BraTS 标签值
        brats_pred[pred == 0] = 0  # 背景
        brats_pred[pred == 1] = 1  # NCR/NET
        brats_pred[pred == 2] = 2  # ED
        brats_pred[pred == 3] = 4  # ET（关键：索引 3 -> 标签 4）
        
        return brats_pred
    
    def _prepare_prediction_mask(
        self,
        pred: torch.Tensor,
        is_logits: bool = True
    ) -> np.ndarray:
        """
        准备预测掩码：将 logits 转换为 BraTS 标签
        
        Args:
            pred: 模型输出 (B, C, H, W) logits 或 (B, H, W) 索引
            is_logits: 是否为 logits（True）或已经是索引（False）
        
        Returns:
            BraTS 标签数组 (H, W) 或 (H, W, D)，值在 [0, 1, 2, 4]
        """
        # 转换为 numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        
        # 如果是批次数据，取第一个样本
        if pred.ndim == 4:
            pred = pred[0]  # (C, H, W) 或 (H, W, D)
        
        # 如果是 logits (C, H, W)，转换为类别索引
        if is_logits and pred.ndim == 3 and pred.shape[0] > 1:
            # 使用 argmax
            pred = pred.argmax(axis=0)  # (H, W)
        elif pred.ndim == 3 and pred.shape[0] == 1:
            # 单通道，取第一个通道
            pred = pred[0]  # (H, W)
        
        # 转换为 BraTS 标签值
        brats_pred = self._convert_to_brats_labels(pred)
        
        return brats_pred
    
    def save_batch_nifti(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        image_names: List[str],
        affine: Optional[np.ndarray] = None,
        is_logits: bool = True
    ) -> List[str]:
        """
        批量保存预测掩码为 NIfTI 格式
        
        Args:
            preds: 预测结果 (B, C, H, W) logits 或 (B, H, W) 索引
            image_names: 图像名称列表（用于生成文件名）
            affine: 仿射变换矩阵（可选，如果为 None 则使用单位矩阵）
            is_logits: 是否为 logits（True）或已经是索引（False）
        
        Returns:
            保存的文件路径列表
        """
        if not HAS_NIBABEL:
            raise ImportError(
                "需要安装 nibabel 来保存 NIfTI 文件。"
                "安装方法: pip install nibabel"
            )
        
        saved_paths = []
        batch_size = preds.shape[0] if preds.ndim >= 3 else 1
        
        # 如果没有提供 affine，使用单位矩阵
        if affine is None:
            affine = np.eye(4)
        
        for i in range(batch_size):
            # 提取单个样本的预测
            if preds.ndim == 4:
                pred_i = preds[i]  # (C, H, W)
            elif preds.ndim == 3:
                pred_i = preds[i] if i < preds.shape[0] else preds  # (H, W)
            else:
                pred_i = preds
            
            # 转换为 BraTS 标签
            brats_mask = self._prepare_prediction_mask(
                torch.from_numpy(pred_i) if isinstance(pred_i, np.ndarray) else pred_i,
                is_logits=is_logits
            )
            
            # 生成文件名
            image_name = image_names[i] if i < len(image_names) else f"pred_{i}"
            # 移除可能的扩展名，添加 _pred.nii.gz
            base_name = Path(image_name).stem.replace('.nii', '').replace('.gz', '')
            save_path = self.pred_masks_dir / f"{base_name}_pred.nii.gz"
            
            # 创建 NIfTI 图像
            nii_img = nib.Nifti1Image(brats_mask.astype(np.uint8), affine)
            
            # 保存文件
            nib.save(nii_img, str(save_path))
            saved_paths.append(str(save_path))
        
        return saved_paths
    
    def save_single_nifti(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        save_name: str,
        affine: Optional[np.ndarray] = None,
        is_logits: bool = True
    ) -> str:
        """
        保存单个预测掩码为 NIfTI 格式
        
        Args:
            pred: 预测结果 (C, H, W) logits 或 (H, W) 索引
            save_name: 保存文件名（不含扩展名）
            affine: 仿射变换矩阵（可选）
            is_logits: 是否为 logits
        
        Returns:
            保存的文件路径
        """
        if not HAS_NIBABEL:
            raise ImportError(
                "需要安装 nibabel 来保存 NIfTI 文件。"
                "安装方法: pip install nibabel"
            )
        
        # 转换为 BraTS 标签
        brats_mask = self._prepare_prediction_mask(
            torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred,
            is_logits=is_logits
        )
        
        # 生成文件路径
        base_name = Path(save_name).stem.replace('.nii', '').replace('.gz', '')
        save_path = self.pred_masks_dir / f"{base_name}_pred.nii.gz"
        
        # 如果没有提供 affine，使用单位矩阵
        if affine is None:
            affine = np.eye(4)
        
        # 创建并保存 NIfTI 图像
        nii_img = nib.Nifti1Image(brats_mask.astype(np.uint8), affine)
        nib.save(nii_img, str(save_path))
        
        return str(save_path)
    
    def save_metrics_to_csv(
        self,
        metrics_dict: Dict[str, Union[float, int]],
        filename: str = "metrics.csv",
        append: bool = False,
        include_timestamp: bool = True
    ) -> str:
        """
        保存评估指标到 CSV 文件
        
        Args:
            metrics_dict: 指标字典，例如：
                {
                    'WT_Dice': 0.85,
                    'TC_Dice': 0.78,
                    'ET_Dice': 0.72,
                    'WT_HD95': 5.2,
                    'TC_HD95': 6.1,
                    'ET_HD95': 4.8,
                    'Mean_Dice': 0.78,
                    'Mean_HD95': 5.37
                }
            filename: CSV 文件名（默认: "metrics.csv"）
            append: 是否追加写入（False 则覆盖）
            include_timestamp: 是否在字典中添加时间戳列
        
        Returns:
            保存的文件路径
        """
        import datetime
        
        # 准备数据
        data = metrics_dict.copy()
        
        # 添加时间戳（如果需要）
        if include_timestamp:
            data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 转换为 DataFrame
        df = pd.DataFrame([data])
        
        # 文件路径
        csv_path = self.metrics_dir / filename
        
        # 保存或追加
        if append and csv_path.exists():
            # 追加模式：读取现有文件并追加新行
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(csv_path, index=False)
        else:
            # 覆盖模式：直接保存
            df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def save_batch_metrics_to_csv(
        self,
        metrics_list: List[Dict[str, Union[float, int]]],
        filename: str = "batch_metrics.csv",
        include_image_names: bool = True,
        image_names: Optional[List[str]] = None
    ) -> str:
        """
        批量保存评估指标到 CSV
        
        Args:
            metrics_list: 指标字典列表，每个字典对应一个样本的指标
            filename: CSV 文件名
            include_image_names: 是否在 CSV 中包含图像名称列
            image_names: 图像名称列表（如果 include_image_names=True）
        
        Returns:
            保存的文件路径
        """
        # 准备数据
        data_list = []
        for i, metrics in enumerate(metrics_list):
            row = metrics.copy()
            if include_image_names and image_names is not None:
                row['image_name'] = image_names[i] if i < len(image_names) else f"sample_{i}"
            data_list.append(row)
        
        # 转换为 DataFrame
        df = pd.DataFrame(data_list)
        
        # 文件路径
        csv_path = self.metrics_dir / filename
        
        # 保存
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def load_metrics_from_csv(self, filename: str = "metrics.csv") -> pd.DataFrame:
        """
        从 CSV 文件加载评估指标
        
        Args:
            filename: CSV 文件名
        
        Returns:
            DataFrame 包含所有指标
        """
        csv_path = self.metrics_dir / filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"指标文件不存在: {csv_path}")
        
        df = pd.read_csv(csv_path)
        return df
    
    def get_summary_statistics(
        self,
        filename: str = "metrics.csv"
    ) -> Dict[str, Dict[str, float]]:
        """
        获取指标统计摘要（均值、标准差等）
        
        Args:
            filename: CSV 文件名
        
        Returns:
            统计摘要字典
        """
        df = self.load_metrics_from_csv(filename)
        
        # 计算数值列的统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {}
        for col in numeric_cols:
            summary[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        
        return summary

