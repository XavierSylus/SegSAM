"""
梯度余弦相似度监控模块

用于监控不同客户端之间的梯度方向，诊断跨模态参数污染问题。

核心指标:
- 余弦相似度 = 1.0: 梯度完全对齐（好）
- 余弦相似度 = 0.0: 梯度正交，90度夹角（中性）
- 余弦相似度 = -1.0: 梯度相反（坏，参数对抗）

对于异构联邦学习:
- Group B (image_only + multimodal): 预期梯度夹角 > 90度（负相似度），证明跨模态污染
- Group C (解耦聚合): 预期梯度相似度接近 1.0，因为不同模态参数物理隔离

Author: SegSAM Team
Date: 2026-03-28
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class GradientMonitor:
    """梯度余弦相似度监控器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化

        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.history = {
            'cosine_similarities': [],
            'angles': [],
            'norms': []
        }

    def compute_gradient_cosine_similarity(
        self,
        clients_weights: List[Dict],
        filter_patterns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算客户端之间的梯度余弦相似度

        Args:
            clients_weights: 客户端权重更新列表，每个元素是字典：
                             {'client_id': str, 'modality': str, 'weights': Dict[str, Tensor]}
            filter_patterns: 要分析的参数名称模式（如 ['adapter', 'lora']）

        Returns:
            余弦相似度字典，包含：
            - client_i_vs_client_j_cosine: 余弦相似度 [-1, 1]
            - client_i_vs_client_j_angle: 夹角（度）[0, 180]
            - client_i_vs_client_j_norm_i: 客户端 i 的梯度范数
            - client_i_vs_client_j_norm_j: 客户端 j 的梯度范数
        """
        if len(clients_weights) < 2:
            self.logger.warning("客户端数量少于2，无法计算梯度相似度")
            return {}

        similarities = {}

        # 提取所有客户端的权重更新
        grad_dicts = []
        client_ids = []
        modalities = []

        for client_data in clients_weights:
            weights = client_data.get('weights', {})
            if weights:
                grad_dicts.append(weights)
                client_ids.append(client_data.get('client_id', 'unknown'))
                modalities.append(client_data.get('modality', 'unknown'))

        if len(grad_dicts) < 2:
            self.logger.warning("有效客户端权重少于2，无法计算梯度相似度")
            return {}

        # 计算两两之间的余弦相似度
        for i in range(len(grad_dicts)):
            for j in range(i + 1, len(grad_dicts)):
                client_i_id = client_ids[i]
                client_j_id = client_ids[j]
                modality_i = modalities[i]
                modality_j = modalities[j]

                # 如果指定了过滤模式，只计算特定层的相似度
                if filter_patterns:
                    params_i = self._filter_params(grad_dicts[i], filter_patterns)
                    params_j = self._filter_params(grad_dicts[j], filter_patterns)
                    suffix = f"_{'_'.join(filter_patterns)}"
                else:
                    params_i = grad_dicts[i]
                    params_j = grad_dicts[j]
                    suffix = "_all"

                # 计算余弦相似度
                cos_sim, norm_i, norm_j = self._compute_param_cosine(params_i, params_j)

                # 计算夹角（度）
                angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi

                # 构造键名
                key_prefix = f"{client_i_id}_vs_{client_j_id}{suffix}"

                similarities[f"{key_prefix}_cosine"] = float(cos_sim)
                similarities[f"{key_prefix}_angle"] = float(angle)
                similarities[f"{key_prefix}_norm_{client_i_id}"] = float(norm_i)
                similarities[f"{key_prefix}_norm_{client_j_id}"] = float(norm_j)

                # 记录模态信息
                similarities[f"{key_prefix}_modality_{client_i_id}"] = modality_i
                similarities[f"{key_prefix}_modality_{client_j_id}"] = modality_j

                # 详细日志
                self.logger.info(
                    f"[GRADIENT] {client_i_id}({modality_i}) vs {client_j_id}({modality_j}){suffix}: "
                    f"cosine={cos_sim:.4f}, angle={angle:.2f}°, "
                    f"norm_i={norm_i:.6f}, norm_j={norm_j:.6f}"
                )

                # 诊断信息
                if angle > 90:
                    self.logger.warning(
                        f"⚠️  检测到梯度对抗！{client_i_id} 和 {client_j_id} 的梯度夹角 > 90度 ({angle:.2f}°)"
                    )
                elif cos_sim < 0.1:
                    self.logger.warning(
                        f"⚠️  梯度几乎正交！{client_i_id} 和 {client_j_id} 的余弦相似度接近 0 ({cos_sim:.4f})"
                    )

        # 特别关注 Adapter 层
        if 'adapter' not in (filter_patterns or []):
            adapter_similarities = self.compute_gradient_cosine_similarity(
                clients_weights,
                filter_patterns=['adapter']
            )
            similarities.update({f"adapter_{k}": v for k, v in adapter_similarities.items()})

        return similarities

    def _filter_params(self, params: Dict[str, torch.Tensor], patterns: List[str]) -> Dict[str, torch.Tensor]:
        """
        根据模式过滤参数

        Args:
            params: 参数字典
            patterns: 模式列表

        Returns:
            过滤后的参数字典
        """
        filtered = {}
        for key, value in params.items():
            if any(pattern.lower() in key.lower() for pattern in patterns):
                filtered[key] = value
        return filtered

    def _compute_param_cosine(
        self,
        params1: Dict[str, torch.Tensor],
        params2: Dict[str, torch.Tensor]
    ) -> Tuple[float, float, float]:
        """
        计算两个参数字典的余弦相似度

        Args:
            params1: 第一个参数字典
            params2: 第二个参数字典

        Returns:
            (余弦相似度, 范数1, 范数2)
        """
        # 找到共同的参数键
        common_keys = sorted(set(params1.keys()) & set(params2.keys()))

        if not common_keys:
            self.logger.warning("没有共同的参数键，无法计算余弦相似度")
            return 0.0, 0.0, 0.0

        # 展平参数为一维向量
        vec1 = []
        vec2 = []
        for key in common_keys:
            # 确保参数在同一设备上
            p1 = params1[key].detach().cpu().flatten()
            p2 = params2[key].detach().cpu().flatten()
            vec1.append(p1)
            vec2.append(p2)

        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)

        # 计算范数
        norm1 = torch.norm(vec1).item()
        norm2 = torch.norm(vec2).item()

        # 计算余弦相似度
        if norm1 < 1e-10 or norm2 < 1e-10:
            self.logger.warning("参数范数接近零，余弦相似度无意义")
            return 0.0, norm1, norm2

        cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

        return cos_sim, norm1, norm2

    def analyze_cross_modality_conflict(
        self,
        similarities: Dict[str, float]
    ) -> Dict[str, any]:
        """
        分析跨模态梯度冲突

        Args:
            similarities: compute_gradient_cosine_similarity() 的返回值

        Returns:
            分析结果
        """
        analysis = {
            'has_conflict': False,
            'conflict_pairs': [],
            'avg_angle': 0.0,
            'max_angle': 0.0,
            'min_cosine': 1.0
        }

        angles = []
        cosines = []

        for key, value in similarities.items():
            if 'angle' in key and isinstance(value, (int, float)):
                angles.append(value)
                if value > 90:
                    analysis['has_conflict'] = True
                    pair_name = key.replace('_angle', '')
                    analysis['conflict_pairs'].append({
                        'pair': pair_name,
                        'angle': value
                    })

            if 'cosine' in key and isinstance(value, (int, float)):
                cosines.append(value)

        if angles:
            analysis['avg_angle'] = np.mean(angles)
            analysis['max_angle'] = np.max(angles)

        if cosines:
            analysis['min_cosine'] = np.min(cosines)

        return analysis

    def log_summary(self, similarities: Dict[str, float]):
        """
        打印梯度相似度摘要

        Args:
            similarities: 相似度字典
        """
        print("\n" + "=" * 80)
        print("梯度余弦相似度监控摘要")
        print("=" * 80)

        # 提取主要指标
        main_metrics = {}
        for key, value in similarities.items():
            if 'cosine' in key or 'angle' in key:
                if 'adapter' in key:
                    main_metrics[key] = value

        if not main_metrics:
            print("未发现有效的梯度相似度指标")
            return

        print(f"\n发现 {len(main_metrics)} 个梯度对比指标:\n")

        for key, value in sorted(main_metrics.items()):
            if 'cosine' in key:
                print(f"  {key}: {value:.4f}")
            elif 'angle' in key:
                status = "⚠️ 对抗" if value > 90 else ("正交" if abs(value - 90) < 5 else "对齐")
                print(f"  {key}: {value:.2f}° ({status})")

        # 跨模态冲突分析
        analysis = self.analyze_cross_modality_conflict(similarities)
        print(f"\n跨模态梯度冲突分析:")
        print(f"  存在冲突: {'是' if analysis['has_conflict'] else '否'}")
        print(f"  平均夹角: {analysis['avg_angle']:.2f}°")
        print(f"  最大夹角: {analysis['max_angle']:.2f}°")
        print(f"  最小余弦相似度: {analysis['min_cosine']:.4f}")

        if analysis['conflict_pairs']:
            print(f"\n  冲突对 ({len(analysis['conflict_pairs'])}):")
            for pair in analysis['conflict_pairs']:
                print(f"    - {pair['pair']}: {pair['angle']:.2f}°")

        print("=" * 80)


def main():
    """测试梯度监控功能"""
    print("梯度余弦相似度监控模块测试")

    # 创建监控器
    monitor = GradientMonitor()

    # 模拟两个客户端的权重
    client1_weights = {
        'adapter.0.weight': torch.randn(64, 768),
        'adapter.0.bias': torch.randn(64),
        'adapter.1.weight': torch.randn(768, 64)
    }

    client2_weights = {
        'adapter.0.weight': torch.randn(64, 768),  # 随机初始化，应该与client1几乎正交
        'adapter.0.bias': torch.randn(64),
        'adapter.1.weight': torch.randn(768, 64)
    }

    # 模拟对抗梯度
    client3_weights = {
        'adapter.0.weight': -client1_weights['adapter.0.weight'],  # 完全相反
        'adapter.0.bias': -client1_weights['adapter.0.bias'],
        'adapter.1.weight': -client1_weights['adapter.1.weight']
    }

    clients_data = [
        {'client_id': 'client1', 'modality': 'image_only', 'weights': client1_weights},
        {'client_id': 'client2', 'modality': 'multimodal', 'weights': client2_weights},
        {'client_id': 'client3', 'modality': 'text_only', 'weights': client3_weights}
    ]

    # 计算相似度
    similarities = monitor.compute_gradient_cosine_similarity(clients_data)

    # 打印摘要
    monitor.log_summary(similarities)


if __name__ == "__main__":
    main()

