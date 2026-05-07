"""
Server Aggregator: 基于 CreamFL 的全局聚合逻辑。
实现基于客户端表示相似性的加权聚合，以及异构多模态的解耦聚合路由。
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import torch.optim

from src.model import SAM3_Medical, DEVICE
from src.contrastive_aggregation import ContrastiveWeightAggregation
from src.knowledge_distillation import KnowledgeDistillation


class CreamAggregator:
    """
    CreamFL 联邦聚合器。
    实现基于客户端表示相似性的加权聚合，以及异构模态间的物理隔离路由。
    """

    # ══════════════════════════════════════════════════════════════════
    # 五级白名单路由常量（唯一可信源，_is_vision_param 与路由器共享此源）
    #
    # 优先级  参数类型           准入模态                  关键词（子串匹配）
    # ──────  ─────────────────  ──────────────────────   ────────────────────────────────
    # 1       TEXT_ADAPTER       text_only + multimodal   'text_adapter'
    # 2       VISION_ADAPTER     image_only + multimodal  'adapters.', 'wrapped_blocks.', 'lora'
    # 3       TEXT_PARAMS        text_only + multimodal   'text_encoder', 'text_proj'
    # 4       IMAGE_PARAMS       image_only + multimodal  见下方常量
    # 5       COMPAT_FALLBACK    按实际上传模态决定        （未命中白名单）
    #
    # ★ 物理隔离三定律：
    #   ① text_only 客户端永久禁止参与 IMAGE_PARAMS / VISION_ADAPTER 聚合
    #   ② image_only 客户端永久禁止参与 TEXT_PARAMS / TEXT_ADAPTER 聚合
    #   ③ image_proj 归视觉侧：Client 1 无图像输入，其 image_proj 梯度为零/噪声
    # ══════════════════════════════════════════════════════════════════
    _TEXT_ADAPTER_KEYWORDS = ('text_adapter',)
    _VISION_ADAPTER_KEYWORDS = ('adapters.', 'wrapped_blocks.', 'lora')
    _TEXT_KEYWORDS = ('text_encoder', 'text_proj')
    _IMAGE_KEYWORDS = (
        'image_encoder', 'backbone', 'neck', 'sam3_model',
        'mask_decoder', 'segmentation_head', 'prompt_encoder',
        'image_proj', '_output_conv', 'medical_seg_head',
        'text_prompt_encoder',
    )

    def __init__(
        self,
        global_model: SAM3_Medical,
        device: str = DEVICE,
        aggregation_method: str = "contrastive_weighted",
        global_rep_alpha: float = 0.9,
        strict_aggregation_guard: bool = False,
        proxy_k_batches: int = 1,
    ):
        self.global_model = global_model.to(device)
        self.device = device
        self.aggregation_method = aggregation_method
        self.global_rep_alpha = global_rep_alpha
        self.strict_aggregation_guard = strict_aggregation_guard
        self.proxy_k_batches = max(1, int(proxy_k_batches))

        self.contrastive_aggregator = (
            ContrastiveWeightAggregation(device=device)
            if aggregation_method == "contrastive_weighted"
            else None
        )
        self.distiller = None

        embed_dim = global_model.embed_dim
        self.global_text_rep = nn.functional.normalize(
            torch.randn(embed_dim, device=device), p=2, dim=0
        )
        self.global_image_rep = nn.functional.normalize(
            torch.randn(embed_dim, device=device), p=2, dim=0
        )

        # 梯度冲突角度（由 federated_trainer 在聚合后读取写入 training_history）
        self._last_grad_conflict_deg: Optional[float] = None
        self._last_aggregation_audit: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────
    # 内部工具方法
    # ──────────────────────────────────────────────────────────────────

    def _is_vision_param(self, param_name: str) -> bool:
        """判断参数是否属于视觉侧（IMAGE_PARAMS 或 VISION_ADAPTER）。"""
        if any(kw in param_name for kw in self._TEXT_ADAPTER_KEYWORDS):
            return False
        if any(kw in param_name for kw in self._VISION_ADAPTER_KEYWORDS):
            return True
        return any(kw in param_name for kw in self._IMAGE_KEYWORDS)

    def _compute_similarity_weights(
        self,
        client_public_reps: List[torch.Tensor],
        consensus_rep: torch.Tensor
    ) -> torch.Tensor:
        """基于与共识表征的余弦相似度计算聚合权重。"""
        similarities = torch.stack([
            torch.dot(
                nn.functional.normalize(rep.to(self.device), p=2, dim=0),
                nn.functional.normalize(consensus_rep, p=2, dim=0)
            )
            for rep in client_public_reps
        ])
        weights = torch.softmax(similarities, dim=0)
        weights = torch.clamp(weights, min=0.0)
        return weights / (weights.sum() + 1e-8)

    def _safe_fill_missing_params(
        self,
        aggregated_state: Dict[str, torch.Tensor],
        location_tag: str = "",
        expected_param_names: Optional[set] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        防回归补齐 + 视觉参数守护。

        对缺失参数用 global_model 当前值（上轮聚合结果）补全。
        视觉侧参数缺失时打印 WARNING；已在 aggregated_state 中的参数绝不覆盖。
        """
        global_sd = self.global_model.state_dict()
        global_keys = set(global_sd.keys())
        aggregated_keys = set(aggregated_state.keys())
        missing = global_keys - aggregated_keys

        # strict guard should only audit keys expected to be aggregated in this round.
        if expected_param_names is None:
            expected_param_names = global_keys
        expected_missing = expected_param_names - aggregated_keys
        vision_missing = [k for k in expected_missing if self._is_vision_param(k)]
        non_vision_missing = [k for k in expected_missing if not self._is_vision_param(k)]

        vision_module_counts: Dict[str, int] = {}
        for k in vision_missing:
            mod = k.split('.')[0]
            vision_module_counts[mod] = vision_module_counts.get(mod, 0) + 1

        non_vision_module_counts: Dict[str, int] = {}
        for k in non_vision_missing:
            mod = k.split('.')[1] if k.count('.') >= 1 else k.split('.')[0]
            non_vision_module_counts[mod] = non_vision_module_counts.get(mod, 0) + 1

        self._last_aggregation_audit = {
            "location_tag": location_tag,
            "missing_total": len(missing),
            "expected_missing_total": len(expected_missing),
            "vision_missing": len(vision_missing),
            "non_vision_missing": len(non_vision_missing),
            "global_key_total": len(global_keys),
            "aggregated_key_total": len(aggregated_keys),
            "expected_key_total": len(expected_param_names),
            "vision_module_counts": vision_module_counts,
            "non_vision_module_counts": non_vision_module_counts,
            "expected_missing_sample": sorted(expected_missing)[:10],
        }
        print(
            f"  [{location_tag}][AggAudit] missing_total={len(missing)}, "
            f"expected_missing_total={len(expected_missing)}, "
            f"vision_missing={len(vision_missing)}, non_vision_missing={len(non_vision_missing)}"
        )
        print(
            f"  [{location_tag}][AggAudit] keyspace: global={len(global_keys)}, "
            f"aggregated={len(aggregated_keys)}, expected={len(expected_param_names)}"
        )
        if expected_missing:
            print(
                f"  [{location_tag}][AggAudit] expected_missing_sample="
                f"{sorted(expected_missing)[:10]}"
            )

        if vision_missing:
            print(
                f"  [{location_tag}][WARNING] 视觉参数未进入聚合池（{len(vision_missing)} 个，保留上轮值）："
                + "".join(f"\n    - {m}: {c} 个" for m, c in sorted(vision_module_counts.items()))
            )

            if self.strict_aggregation_guard:
                raise RuntimeError(
                    f"[{location_tag}] strict_aggregation_guard=True 且检测到视觉参数缺失: "
                    f"{len(vision_missing)} 个。请先修复聚合覆盖后再继续实验。"
                )

        if missing:
            for k in missing:
                aggregated_state[k] = global_sd[k].clone()

            if expected_missing:
                print(f"  [{location_tag}] 补齐 {len(missing)} 个参数（视觉: {len(vision_missing)}, 其他: {len(non_vision_missing)}）")
                for mod, cnt in sorted(non_vision_module_counts.items()):
                    print(f"    - {mod}: {cnt} 个（解耦聚合，由全局值保持）")

        still_missing = set(global_sd.keys()) - set(aggregated_state.keys())
        assert not still_missing, (
            f"[{location_tag}] 防回归断言失败：补齐后仍遗漏 {len(still_missing)} 个参数：\n"
            + "\n".join(f"  - {k}" for k in sorted(still_missing))
        )
        return aggregated_state

    def _get_participating_clients_dynamic(
        self,
        param_name: str,
        client_weights_list: List[Optional[Dict[str, torch.Tensor]]],
        client_modalities: Optional[List[str]] = None
    ) -> List[int]:
        """
        非对称解耦聚合路由器（Asymmetric Decoupled Aggregation Router）。

        按五级白名单优先级路由确定参与本参数聚合的客户端下标列表。
        无合格上传者时返回空列表，调用方收到空列表后必须 continue。
        """
        uploaded_indices = [
            i for i, w in enumerate(client_weights_list)
            if w is not None and param_name in w
        ]

        if client_modalities is None or len(client_modalities) != len(client_weights_list):
            return uploaded_indices

        # 五级优先白名单路由（先命中先出）
        is_text_adapter   = any(kw in param_name for kw in self._TEXT_ADAPTER_KEYWORDS)
        is_vision_adapter = any(kw in param_name for kw in self._VISION_ADAPTER_KEYWORDS)
        is_text           = any(kw in param_name for kw in self._TEXT_KEYWORDS)
        is_image          = any(kw in param_name for kw in self._IMAGE_KEYWORDS)

        if is_text_adapter:
            allowed_modalities = {'text_only', 'multimodal'}
            route_label = 'TEXT_ADAPTER'
        elif is_vision_adapter:
            allowed_modalities = {'image_only', 'multimodal'}
            route_label = 'VISION_ADAPTER'
        elif is_text:
            allowed_modalities = {'text_only', 'multimodal'}
            route_label = 'TEXT_PARAMS'
        elif is_image:
            allowed_modalities = {'image_only', 'multimodal'}
            route_label = 'IMAGE_PARAMS'
        else:
            allowed_modalities = {client_modalities[i] for i in uploaded_indices}
            route_label = 'COMPAT_FALLBACK'

        filtered_indices = [
            i for i in uploaded_indices if client_modalities[i] in allowed_modalities
        ]

        if len(filtered_indices) == 0 and len(uploaded_indices) > 0:
            print(
                f"[WARNING][ADA-Router] 参数 '{param_name}' "
                f"收到 {len(uploaded_indices)} 个客户端上传 "
                f"（上传方模态: {[client_modalities[i] for i in uploaded_indices]}），"
                f"但白名单强制路由后无合格客户端！"
                f"\n  路由判断: route={route_label}, "
                f"is_text_adapter={is_text_adapter}, is_vision_adapter={is_vision_adapter}, "
                f"is_text={is_text}, is_image={is_image}, allowed={allowed_modalities}"
                f"\n  → 跳过此参数，保留全局模型当前值。"
            )

        return filtered_indices

    def _apply_aggregated_state(
        self,
        aggregated_state: Dict[str, torch.Tensor],
        location_tag: str,
        expected_param_names: Optional[set] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        统一后处理：load_state_dict → RoPE 频率重置 → missing key 校验 → 防回归补齐。
        三条聚合分支共享此入口，消除重复代码。
        """
        def _rope_signature() -> Dict[str, Tuple[Tuple[int, ...], float, float]]:
            sig: Dict[str, Tuple[Tuple[int, ...], float, float]] = {}
            for key, value in self.global_model.state_dict().items():
                if any(tok in key for tok in ('freqs_cis', 'freqs_cos', 'freqs_sin', 'relative_coords')):
                    tensor = value.detach().float()
                    mean_val = float(tensor.mean().item()) if tensor.numel() > 0 else 0.0
                    std_val = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
                    sig[key] = (tuple(tensor.shape), mean_val, std_val)
            return sig

        rope_before = _rope_signature()
        missing_keys, _ = self.global_model.load_state_dict(aggregated_state, strict=False)
        rope_after_load = _rope_signature()

        if hasattr(self.global_model, 'reset_rope_frequencies'):
            count = self.global_model.reset_rope_frequencies(verbose=False)
            if count > 0:
                rope_after_reset = _rope_signature()
                changed_on_load = sum(
                    1 for key, value in rope_after_load.items()
                    if rope_before.get(key) != value
                )
                changed_on_reset = sum(
                    1 for key, value in rope_after_reset.items()
                    if rope_after_load.get(key) != value
                )
                reset_reason = "state_load_change" if changed_on_load > 0 else "forced_refresh_or_stale"
                print(
                    f"  [RoPE Fix] 重置了 {count} 个 Block 的 RoPE 频率 "
                    f"(reason={reset_reason}, rope_buffers={len(rope_after_reset)}, "
                    f"changed_on_load={changed_on_load}, changed_on_reset={changed_on_reset})"
                )

        trainable_names = {n for n, p in self.global_model.named_parameters() if p.requires_grad}
        missing_trainable = [k for k in missing_keys if k in trainable_names]
        if missing_trainable:
            print(f"[注意] 聚合状态缺失 {len(missing_trainable)} 个可训练参数（将由防回归补齐）")

        return self._safe_fill_missing_params(
            aggregated_state,
            location_tag=location_tag,
            expected_param_names=expected_param_names,
        )

    # ──────────────────────────────────────────────────────────────────
    # 梯度冲突探针
    # ──────────────────────────────────────────────────────────────────

    def _probe_gradient_conflict(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_modalities: List[str]
    ):
        """
        伪梯度冲突探针（多模态对比协议自动激活，SegSAM image-only 静默跳过）。
        计算 ImageOnly 与 Multimodal 客户端在 adapter 参数上的梯度夹角。
        结果写入 self._last_grad_conflict_deg，由 federated_trainer 读取后存入 JSON。
        """
        self._last_grad_conflict_deg = None
        try:
            idx_img   = client_modalities.index('image_only')
            idx_multi = client_modalities.index('multimodal')
        except ValueError:
            return

        global_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        delta_img = {
            k: (client_weights[idx_img][k] - global_state[k]).to(self.device)
            for k in client_weights[idx_img]
            if 'adapter' in k and k in global_state
        }
        delta_multi = {
            k: (client_weights[idx_multi][k] - global_state[k]).to(self.device)
            for k in client_weights[idx_multi]
            if 'adapter' in k and k in global_state
        }
        if delta_img and delta_multi:
            self._last_grad_conflict_deg = self.compute_gradient_conflict(delta_img, delta_multi)

    # ──────────────────────────────────────────────────────────────────
    # 核心聚合方法
    # ──────────────────────────────────────────────────────────────────

    def aggregate_weights(
        self,
        client_weights: List[Optional[Dict[str, torch.Tensor]]],
        client_public_reps: List[torch.Tensor],
        global_features_for_contrastive: Optional[torch.Tensor] = None,
        client_modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        使用 CreamFL 方法聚合客户端模型权重（含解耦聚合路由）。

        Args:
            client_weights:                  客户端模型状态字典列表（可含 None）
            client_public_reps:              客户端公共数据表征列表
            global_features_for_contrastive: 用于对比权重计算的全局特征（可选）
            client_modalities:               客户端模态列表，用于解耦聚合路由
        Returns:
            聚合后的全局模型状态字典
        """
        # ── 步骤 1: 安全过滤，剔除 weights 为 None 的客户端 ──
        valid = [
            (w, client_public_reps[i], client_modalities[i] if client_modalities else None)
            for i, w in enumerate(client_weights)
            if w is not None
        ]
        if not valid:
            print("⚠️ 警告: 所有客户端的 weights 为 None，跳过模型聚合")
            return self.global_model.state_dict()

        total_clients      = len(client_weights)
        client_weights     = [v[0] for v in valid]
        client_public_reps = [v[1] for v in valid]
        client_modalities  = [v[2] for v in valid] if valid[0][2] is not None else None
        num_clients        = len(client_weights)
        print(f"[Safe Filter] 有效客户端: {num_clients}/{total_clients}")

        # ── 步骤 2: 动态参数集合构建 ──
        all_param_names = set()
        for w in client_weights:
            all_param_names.update(w.keys())
        if not all_param_names:
            print("⚠️ 警告: 客户端上传的参数集合为空，跳过模型聚合")
            return self.global_model.state_dict()
        print(f"[Dynamic Aggregation] 检测到 {len(all_param_names)} 个唯一参数需要聚合")

        # ── 步骤 4: 计算聚合权重 ──
        use_contrastive = (
            self.aggregation_method == "contrastive_weighted"
            and self.contrastive_aggregator is not None
        )
        use_contrastive_decoupled = use_contrastive and (
            client_modalities is not None and len(client_modalities) == num_clients
        )

        if self.aggregation_method == "similarity_weighted":
            agg_weights = self._compute_similarity_weights(client_public_reps, self.global_image_rep)
        elif not use_contrastive:
            agg_weights = torch.ones(num_clients, device=self.device) / num_clients
        else:
            agg_weights = None  # contrastive 路径在参数循环内按子集计算

        # ── 步骤 5: 参数级解耦聚合循环 ──
        if global_features_for_contrastive is None:
            global_features_for_contrastive = self.global_image_rep.unsqueeze(0)

        aggregated_state: Dict[str, torch.Tensor] = {}
        decoupled_stats: Dict[str, List[int]] = {}

        if use_contrastive_decoupled:
            print("[Decoupled Contrastive Aggregation] 使用动态解耦对比权重聚合")

        for param_name in all_param_names:
            participating_indices = self._get_participating_clients_dynamic(
                param_name, client_weights, client_modalities
            )
            if not participating_indices:
                continue

            ptype = (
                'mask_decoder' if 'mask_decoder' in param_name else
                'text_encoder' if 'text_encoder' in param_name else
                'adapter'      if 'adapter'      in param_name else
                'other'
            )
            if ptype not in decoupled_stats:
                decoupled_stats[ptype] = [0, 0]  # [累计参与客户端数, 参数数量]
            decoupled_stats[ptype][0] += len(participating_indices)
            decoupled_stats[ptype][1] += 1

            if use_contrastive_decoupled:
                # 对比权重路径：按参与子集聚合
                p_weights = [client_weights[i] for i in participating_indices if param_name in client_weights[i]]
                p_reps    = [client_public_reps[i] for i in participating_indices if param_name in client_weights[i]]
                if not p_weights:
                    continue
                if len(p_weights) == 1:
                    aggregated_state[param_name] = p_weights[0][param_name].clone().to(self.device)
                else:
                    param_agg = self.contrastive_aggregator.aggregate_model_weights(
                        p_weights, p_reps, global_features_for_contrastive
                    )
                    if param_name in param_agg:
                        aggregated_state[param_name] = param_agg[param_name]
                    else:
                        gsd = self.global_model.state_dict()
                        if param_name in gsd:
                            aggregated_state[param_name] = gsd[param_name].clone()
                            if self._is_vision_param(param_name):
                                print(f"  [ContraAgg Fallback] 视觉参数 '{param_name}' 未在聚合器输出中，已保留全局当前值")
            else:
                # 加权/均匀聚合路径
                param_list = []
                weight_list = []
                for idx in participating_indices:
                    if idx < len(client_weights) and param_name in client_weights[idx]:
                        p = client_weights[idx][param_name].to(self.device)
                        param_list.append(p)
                        weight_list.append(agg_weights[idx])

                if not param_list:
                    continue

                w_tensor = torch.tensor(weight_list, device=self.device)
                w_sum = w_tensor.sum()
                w_tensor = w_tensor / w_sum if w_sum > 1e-8 else torch.full_like(w_tensor, 1.0 / len(param_list))

                stacked = torch.stack(param_list, dim=0)
                aggregated_state[param_name] = torch.sum(
                    stacked * w_tensor.view(-1, *([1] * (stacked.dim() - 1))), dim=0
                )

        if decoupled_stats:
            print("动态解耦聚合统计:")
            for ptype, (total_participants, param_count) in decoupled_stats.items():
                avg = total_participants / param_count if param_count > 0 else 0
                print(f"  - {ptype}: {param_count} 个参数，平均 {avg:.1f}/{num_clients} 客户端参与")

        return self._apply_aggregated_state(
            aggregated_state,
            location_tag="防回归补齐[aggregate_weights]",
            expected_param_names=all_param_names,
        )

    def _update_global_reps_decoupled(
        self,
        image_reps: List[torch.Tensor],
        text_reps: List[torch.Tensor]
    ):
        """
        解耦 EMA 更新全局表征。
        image_reps 来自 image_only + multimodal，text_reps 来自 text_only + multimodal。
        """
        alpha = self.global_rep_alpha

        def _align_and_avg(reps: List[torch.Tensor], target_dim: int) -> Optional[torch.Tensor]:
            aligned = []
            for rep in reps:
                rep = rep.to(self.device)
                if rep.dim() > 1:
                    rep = rep.mean(dim=0) if rep.shape[0] > 1 else rep.squeeze(0)
                curr = rep.shape[0]
                if curr > target_dim:
                    rep = rep[:target_dim]
                elif curr < target_dim:
                    rep = nn.functional.pad(rep, (0, target_dim - curr))
                aligned.append(rep)
            return torch.stack(aligned, dim=0).mean(dim=0) if aligned else None

        if image_reps:
            avg = _align_and_avg(image_reps, self.global_image_rep.shape[0])
            if avg is not None:
                self.global_image_rep = nn.functional.normalize(
                    alpha * self.global_image_rep + (1 - alpha) * avg, p=2, dim=0
                )

        if text_reps:
            avg = _align_and_avg(text_reps, self.global_text_rep.shape[0])
            if avg is not None:
                self.global_text_rep = nn.functional.normalize(
                    alpha * self.global_text_rep + (1 - alpha) * avg, p=2, dim=0
                )

    def aggregate_heterogeneous_clients(
        self,
        client_updates: List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]],
        global_features_for_contrastive: Optional[torch.Tensor] = None,
        client_modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        异构客户端聚合接口。

        处理三元组上传：(weights, image_rep, text_rep)
        - text_only  客户端: (None, None, text_rep)
        - image_only 客户端: (weights, image_rep, None)
        - multimodal 客户端: (weights, image_rep, text_rep)

        Args:
            client_updates:                  客户端上传的三元组列表
            global_features_for_contrastive: 用于对比权重计算的全局特征（可选）
            client_modalities:               客户端模态列表（用于解耦聚合）
        Returns:
            聚合后的全局模型状态字典
        """
        if not client_updates:
            print("警告: 没有客户端上传数据，保持全局模型不变")
            return self.global_model.state_dict()

        client_weights: List[Optional[Dict[str, torch.Tensor]]] = []
        image_reps: List[torch.Tensor] = []
        text_reps: List[torch.Tensor] = []
        participating_indices: List[int] = []

        for idx, (weights, img_rep, txt_rep) in enumerate(client_updates):
            modality = (
                client_modalities[idx]
                if client_modalities is not None and idx < len(client_modalities)
                else None
            )
            if weights is not None:
                client_weights.append(weights)
                participating_indices.append(idx)
            if img_rep is not None:
                image_reps.append(img_rep)
            if txt_rep is not None:
                if modality is None or modality in ('text_only', 'multimodal'):
                    text_reps.append(txt_rep)
                else:
                    # image_only 客户端意外上传了文本表征 → 拦截，防止随机视觉特征污染全局文本锚点
                    print(
                        f"[WARNING][Server] 客户端 {idx}（modality={modality}）"
                        f"意外上传了 txt_rep！已拦截，不纳入全局文本表征聚合池。"
                    )

        if not client_weights:
            print("⚠️ 警告: 没有客户端提供模型权重（所有客户端都是 text_only），跳过模型聚合")
            self._update_global_reps_decoupled(image_reps, text_reps)
            return self.global_model.state_dict()

        participating_reps = []
        for idx in participating_indices:
            _, img_rep, _ = client_updates[idx]
            if img_rep is not None:
                participating_reps.append(img_rep)
            else:
                print(f"警告: 客户端 {idx} 提供了权重但没有图像表征，使用零向量")
                participating_reps.append(torch.zeros(self.global_image_rep.shape[0]))

        participating_modalities = (
            [client_modalities[idx] for idx in participating_indices]
            if client_modalities is not None and len(client_modalities) == len(client_updates)
            else None
        )

        print(f"[Heterogeneous Aggregation] 参与模型聚合的客户端: {len(client_weights)}/{len(client_updates)}")
        print(f"[Heterogeneous Aggregation] 图像表征数量: {len(image_reps)}, 文本表征数量: {len(text_reps)}")

        aggregated_state = self.aggregate_weights(
            client_weights=client_weights,
            client_public_reps=participating_reps,
            global_features_for_contrastive=global_features_for_contrastive,
            client_modalities=participating_modalities
        )

        self._update_global_reps_decoupled(image_reps, text_reps)
        return aggregated_state

    def generate_and_dispatch_global_proxies(
        self,
        global_model,
        public_dataloader,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从公共数据取固定 K 个 Batch（proxy_k_batches），提取图像和文本全局表征（Global Reps）。

        设计约束：
          ① 在 torch.no_grad() 上下文内执行，禁止构建正向计算图；
          ② 返回张量强制 .detach()，物理截断跨客户端反向传播路径；
          ③ 推理完成后立即置 None 释放中间激活，防止显存泄漏。

        图像表征：调用 extract_features 提取，经 mean-pool 压缩为 1-D 向量后 EMA 更新。
        文本表征：使用 fusion_head.text_proj（若存在）投影当前 EMA 文本表征作为文本锚点。

        Returns:
            (image_proxy, text_proxy)，形状均为 (D,)，已 detach。
        """
        global_model.eval()

        data_iter = iter(public_dataloader)
        aggregated_img_vec: Optional[torch.Tensor] = None
        used_batches = 0

        with torch.no_grad():
            while used_batches < self.proxy_k_batches:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                pub_imgs = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
                raw_feats = global_model.extract_features(pub_imgs)
                pub_imgs = None

                if raw_feats.dim() == 3:
                    img_vec = raw_feats.mean(dim=(0, 1))
                elif raw_feats.dim() == 2:
                    img_vec = raw_feats.mean(dim=0)
                else:
                    img_vec = raw_feats.flatten()
                img_vec = nn.functional.normalize(img_vec, p=2, dim=0)

                if aggregated_img_vec is None:
                    aggregated_img_vec = torch.zeros_like(img_vec, device=self.device)
                aggregated_img_vec = aggregated_img_vec + img_vec.detach().to(self.device)
                used_batches += 1

        if used_batches == 0 or aggregated_img_vec is None:
            print("[WARNING][generate_proxies] public_dataloader 为空，返回当前 EMA 表征")
            return self.global_image_rep.detach().clone(), self.global_text_rep.detach().clone()

        image_proxy: torch.Tensor = nn.functional.normalize(
            aggregated_img_vec / float(used_batches), p=2, dim=0
        ).detach()
        aggregated_img_vec = None

        alpha = self.global_rep_alpha
        self.global_image_rep = nn.functional.normalize(
            alpha * self.global_image_rep + (1 - alpha) * image_proxy
            if self.global_image_rep.shape == image_proxy.shape
            else image_proxy.clone(),
            p=2, dim=0
        )

        with torch.no_grad():
            txt_src = self.global_text_rep.to(device)
            text_proj = getattr(getattr(global_model, 'fusion_head', None), 'text_proj', None)
            txt_projected = (
                nn.functional.normalize(text_proj(txt_src.unsqueeze(0)).squeeze(0), p=2, dim=0)
                if text_proj is not None
                else nn.functional.normalize(txt_src, p=2, dim=0)
            )

        text_proxy: torch.Tensor = txt_projected.detach().to(self.device)
        txt_src = txt_projected = None

        print(
            f"[generate_proxies] image_proxy.shape={image_proxy.shape}, "
            f"text_proxy.shape={text_proxy.shape}, used_batches={used_batches}/{self.proxy_k_batches}  "
            f"✓ (两张量已 detach，计算图已截断)"
        )
        return image_proxy, text_proxy

    @staticmethod
    def compute_gradient_conflict(
        grad_dict_img: Dict[str, torch.Tensor],
        grad_dict_multi: Dict[str, torch.Tensor],
    ) -> float:
        """
        梯度余弦相似度探针：诊断 ImageOnly 与 Multimodal 客户端的 adapter 梯度冲突。

        提取含 'adapter' 关键字的参数梯度，展平后拼接为 1-D 向量，计算余弦夹角（度）。
        0° = 完全同向（无冲突），90° = 正交，180° = 完全反向（严重毒化）。
        任一方无 adapter 梯度时，返回哨兵值 180.0。
        """
        img_grads = [
            v.detach().float().flatten()
            for k, v in grad_dict_img.items()
            if 'adapter' in k and v is not None
        ]
        multi_grads = [
            v.detach().float().flatten()
            for k, v in grad_dict_multi.items()
            if 'adapter' in k and v is not None
        ]

        if not img_grads or not multi_grads:
            print(
                "[WARNING][compute_gradient_conflict] "
                f"img_grads={len(img_grads)} 个, multi_grads={len(multi_grads)} 个，"
                "某一方无 adapter 梯度，无法计算冲突角，返回哨兵值 180.0°"
            )
            return 180.0

        vec_img   = torch.cat(img_grads, dim=0)
        vec_multi = torch.cat(multi_grads, dim=0)
        img_grads = multi_grads = None

        min_len   = min(vec_img.shape[0], vec_multi.shape[0])
        vec_img   = vec_img[:min_len]
        vec_multi = vec_multi[:min_len]

        img_norm = torch.linalg.norm(vec_img).item()
        multi_norm = torch.linalg.norm(vec_multi).item()
        dot_val = torch.dot(vec_img, vec_multi).item()
        img_nonzero_ratio = float((vec_img.abs() > 0).float().mean().item()) if min_len > 0 else 0.0
        multi_nonzero_ratio = float((vec_multi.abs() > 0).float().mean().item()) if min_len > 0 else 0.0
        print(
            "[GradConflictDiag] "
            f"img_norm={img_norm:.4e}, multi_norm={multi_norm:.4e}, "
            f"img_nonzero_ratio={img_nonzero_ratio:.4f}, "
            f"multi_nonzero_ratio={multi_nonzero_ratio:.4f}, "
            f"overlap_dim={min_len}, dot={dot_val:.4e}"
        )
        if img_norm <= 1e-12 or multi_norm <= 1e-12:
            print(
                "[WARNING][compute_gradient_conflict] "
                "one gradient vector norm is ~0; conflict angle is not reliable, return sentinel 180.0°"
            )
            return 180.0

        cos_sim = max(-1.0, min(1.0, nn.functional.cosine_similarity(
            vec_img.unsqueeze(0), vec_multi.unsqueeze(0)
        ).item()))
        angle_deg = math.degrees(math.acos(cos_sim))

        vec_img = vec_multi = None

        print(
            f"[GradConflict] adapter 梯度向量冲突角: {angle_deg:.2f}°  "
            f"(cos_sim={cos_sim:.4f}, adapter_params={min_len})"
        )
        if angle_deg > 90.0:
            print(
                f"  ⚠️ 冲突角 > 90°，ImageOnly 与 Multimodal 客户端 adapter 梯度方向相反！"
                f"建议检查 lambda_cream 权重或降低 multimodal 客户端学习率。"
            )
        return angle_deg

    # ──────────────────────────────────────────────────────────────────
    # 检查点接口
    # ──────────────────────────────────────────────────────────────────

    def get_global_reps(self) -> Dict[str, torch.Tensor]:
        return {
            'global_text_rep': self.global_text_rep.clone().cpu(),
            'global_image_rep': self.global_image_rep.clone().cpu(),
        }

    def get_global_model(self) -> SAM3_Medical:
        return self.global_model

    def set_global_model(self, model: SAM3_Medical):
        self.global_model = model.to(self.device)

    def get_state_dict(self) -> Dict[str, Any]:
        """
        获取服务器状态字典（用于检查点保存）。
        仅保留可训练参数及 Adapter/Decoder 相关 Buffer，防止检查点膨胀。
        """
        trainable_names = {name for name, p in self.global_model.named_parameters() if p.requires_grad}
        full_sd = self.global_model.state_dict()
        filtered_sd = {
            k: v.clone().cpu()
            for k, v in full_sd.items()
            if k in trainable_names or 'adapter' in k.lower() or 'decoder' in k.lower()
        }
        return {
            'model_state_dict': filtered_sd,
            'global_text_rep': self.global_text_rep.cpu().clone(),
            'global_image_rep': self.global_image_rep.cpu().clone(),
            'aggregation_method': self.aggregation_method,
            'global_rep_alpha': self.global_rep_alpha,
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """从状态字典加载服务器状态（用于检查点恢复）。"""
        if 'model_state_dict' in state_dict:
            missing_keys, unexpected_keys = self.global_model.load_state_dict(
                state_dict['model_state_dict'], strict=False
            )
            trainable_names = {n for n, p in self.global_model.named_parameters() if p.requires_grad}
            missing_trainable = [k for k in missing_keys if k in trainable_names]
            if missing_trainable:
                raise RuntimeError(f"致命错误: 服务器检查点缺失关键可训练参数! {missing_trainable}")
            if unexpected_keys:
                print(f"警告: 服务器检查点包含未知键: {unexpected_keys}")

        if 'global_text_rep' in state_dict:
            self.global_text_rep = state_dict['global_text_rep'].to(self.device)
        if 'global_image_rep' in state_dict:
            self.global_image_rep = state_dict['global_image_rep'].to(self.device)

        saved_method = state_dict.get('aggregation_method')
        if saved_method and saved_method != self.aggregation_method:
            print(f"警告: 检查点中的聚合方法 ({saved_method}) 与当前配置 ({self.aggregation_method}) 不一致")

        saved_alpha = state_dict.get('global_rep_alpha')
        if saved_alpha and abs(saved_alpha - self.global_rep_alpha) > 1e-6:
            print(f"警告: 检查点中的 global_rep_alpha ({saved_alpha}) 与当前配置 ({self.global_rep_alpha}) 不一致")

    def setup_distillation(
        self,
        optimizer: torch.optim.Optimizer,
        kd_weight: float = 1.0,
        use_fp16: bool = False,
        grad_clip: float = 0.0
    ):
        """初始化知识蒸馏器。"""
        self.distiller = KnowledgeDistillation(
            model=self.global_model,
            optimizer=optimizer,
            device=self.device,
            kd_weight=kd_weight,
            use_fp16=use_fp16,
            grad_clip=grad_clip
        )

    def distill_global_model(
        self,
        public_loader,
        aggregated_img_features: torch.Tensor,
        aggregated_txt_features: Optional[torch.Tensor] = None,
        distill_index: Optional[List[int]] = None,
        modality_types: List[str] = ["image", "text"],
        num_epochs: int = 1
    ) -> Dict[str, List[float]]:
        """
        在公共数据集上对全局模型进行知识蒸馏。

        Args:
            public_loader:           公共数据集 DataLoader
            aggregated_img_features: 聚合后的图像特征 (N, D)
            aggregated_txt_features: 聚合后的文本特征 (N, D)（可选）
            distill_index:           公共数据索引列表（用于映射到聚合特征）
            modality_types:          要蒸馏的模态类型列表
            num_epochs:              蒸馏轮数
        Returns:
            训练历史字典
        """
        if self.distiller is None:
            raise RuntimeError("Distiller not initialized. Call setup_distillation() first.")

        aggregated_features = {'img_vec': aggregated_img_features.to(self.device)}
        if aggregated_txt_features is not None:
            aggregated_features['txt_vec'] = aggregated_txt_features.to(self.device)

        if distill_index is None:
            distill_index = list(range(aggregated_img_features.shape[0]))

        return self.distiller.distill(
            public_loader=public_loader,
            aggregated_features=aggregated_features,
            distill_index=distill_index,
            modality_types=modality_types,
            num_epochs=num_epochs
        )

