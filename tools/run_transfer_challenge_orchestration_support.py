from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer
from orchestrator.transfer_sources import TransferSourceDNA
from tools.run_transfer_challenge_dataset_support import (
    _auto_transfer_mix_decay_steps,
    _build_transfer_dataset,
    _filter_transfer_dataset,
)
from tools.run_transfer_challenge_eval_support import _aggregate_curve, _episode_curve, _safe_float


_SourceDNA = TransferSourceDNA


def _mean_metric(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(
        sum(_safe_float(r.get(key, 0.0), 0.0) for r in rows) / float(max(1, len(rows)))
    )


def _build_filtered_transfer_dataset(
    args: Any,
    *,
    sources: List[_SourceDNA],
    bridge_label_cfg: Dict[str, Any],
    out_path: str,
    near_lane_enabled: bool,
    far_lane_enabled: bool,
) -> Dict[str, Any]:
    ds = _build_transfer_dataset(
        sources=sources,
        target_verse=str(args.target_verse),
        out_path=str(out_path),
        max_rows_per_source=max(0, int(args.max_rows_per_source)),
        near_lane_max_rows_per_source=max(0, int(args.near_lane_max_rows_per_source)),
        far_lane_max_rows_per_source=max(0, int(args.far_lane_max_rows_per_source)),
        near_lane_enabled=bool(near_lane_enabled),
        far_lane_enabled=bool(far_lane_enabled),
        universe_adapter_enabled=bool(args.universe_adapter_enabled),
        far_lane_score_weight_enabled=bool(args.far_lane_score_weight_enabled),
        far_lane_score_weight_strength=max(0.0, min(1.0, float(args.far_lane_score_weight_strength))),
        far_lane_min_universe_feature_score=max(
            0.0, min(1.0, float(args.far_lane_min_universe_feature_score))
        ),
        bridge_synthetic_reward_blend=max(0.0, min(1.0, float(args.bridge_synthetic_reward_blend))),
        bridge_synthetic_done_union=bool(args.bridge_synthetic_done_union),
        bridge_confidence_threshold=max(0.0, min(1.0, float(args.bridge_confidence_threshold))),
        bridge_label_cfg=bridge_label_cfg,
        bridge_behavioral_enabled=bool(args.bridge_behavioral_enabled),
        bridge_behavioral_score_weight=max(0.0, min(1.0, float(args.bridge_behavioral_score_weight))),
        bridge_behavioral_max_prototype_rows=max(1, int(args.bridge_behavioral_max_prototype_rows)),
    )
    if bool(args.transfer_filter):
        fs = _filter_transfer_dataset(
            path=str(ds["transfer_dataset_path"]),
            target_verse=str(args.target_verse),
            dedupe=bool(args.transfer_filter_dedupe),
            max_rows=max(0, int(args.transfer_filter_max_rows)),
            hazard_keep_ratio=max(0.0, min(1.0, float(args.transfer_filter_hazard_keep_ratio))),
            min_transfer_confidence=max(0.0, min(1.0, float(args.transfer_filter_min_confidence))),
        )
        ds["filter_stats"] = fs
        ds["transfer_dataset_rows"] = int(fs.get("kept_rows", ds.get("transfer_dataset_rows", 0)))
    return ds


def _refresh_transfer_mix_decay(args: Any, *, cfg: Dict[str, Any], transfer_rows: int) -> Dict[str, Any]:
    out = dict(cfg)
    if bool(out.get("dynamic_transfer_mix_enabled", False)) and int(args.transfer_mix_decay_steps) <= 0:
        out["transfer_mix_decay_steps"] = int(
            _auto_transfer_mix_decay_steps(
                episodes=max(1, int(args.episodes)),
                max_steps=max(1, int(args.max_steps)),
                transfer_rows=max(0, int(transfer_rows)),
                mix_start=max(0.0, min(1.0, float(args.transfer_mix_start))),
                mix_end=max(0.0, min(1.0, float(args.transfer_mix_end))),
            )
        )
    return out


def run_source_attribution(
    *,
    args: Any,
    sources: List[_SourceDNA],
    transfer_dataset: Dict[str, Any],
    transfer_cfg: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    bridge_label_cfg: Dict[str, Any],
    run_agent: Callable[..., str],
    source_id: Callable[[_SourceDNA], str],
    pick_sources_from_attribution: Callable[..., Dict[str, Any]],
    transfer_mode_utility: Callable[..., Dict[str, float]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "enabled": bool(args.source_attribution_enabled),
        "applied": False,
        "reason": "disabled",
        "evaluated_sources": [],
        "selection": {},
    }
    if not bool(args.source_attribution_enabled) or not sources or int(transfer_dataset.get("transfer_dataset_rows", 0)) <= 0:
        return {
            "report": report,
            "sources": list(sources),
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
        }

    attr_eps = max(1, min(int(args.episodes), int(max(1, args.source_attribution_pilot_episodes))))
    attr_max_steps = int(args.source_attribution_pilot_max_steps)
    if attr_max_steps <= 0:
        attr_max_steps = max(1, int(args.max_steps))
    attr_seed_count = max(1, int(args.source_attribution_num_seeds))
    attr_seed_stride = max(1, int(args.source_attribution_seed_stride))
    attr_seed_list = [int(args.seed) + (attr_seed_stride * (i + 1)) for i in range(attr_seed_count)]
    ranked_sources = sorted(
        list(sources),
        key=lambda s: (
            1 if str(s.source_lane or "") == "near_universe" else 0,
            1 if str(s.source_kind or "") == "dna_good" else 0,
            str(s.verse_name),
        ),
        reverse=True,
    )
    top_k = max(0, int(args.source_attribution_top_k_sources))
    eval_sources = ranked_sources[:top_k] if top_k > 0 else ranked_sources
    if not eval_sources:
        report.update({"enabled": True, "reason": "no_sources_selected"})
        return {
            "report": report,
            "sources": list(sources),
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
        }

    trainer_attr = Trainer(run_root=str(args.runs_root), schema_version="v1", auto_register_builtin=True)
    scratch_rows: List[Dict[str, Any]] = []
    for j, seed_j in enumerate(attr_seed_list):
        run_id = run_agent(
            trainer=trainer_attr,
            role="source_attr_scratch",
            verse_name=str(args.target_verse),
            episodes=int(attr_eps),
            max_steps=int(attr_max_steps),
            seed=int(seed_j),
            algo=str(args.baseline_algo),
            policy_id=f"source_attr_scratch_{args.baseline_algo}_{args.target_verse}_s{j}",
            cfg=dict(baseline_cfg),
        )
        run_dir = os.path.join(str(args.runs_root), run_id)
        st = evaluate_run(run_dir)
        agg = _aggregate_curve(_episode_curve(run_dir))
        scratch_rows.append(
            {
                "run_id": str(run_id),
                "run_dir": str(run_dir).replace("\\", "/"),
                "seed": int(seed_j),
                "success_rate": float(_safe_float(st.success_rate, 0.0)),
                "mean_return": float(_safe_float(st.mean_return, 0.0)),
                "hazard_per_1k": float(_safe_float(agg.get("hazard_events_per_1k_steps", 0.0), 0.0)),
            }
        )
    scratch_sr = _mean_metric(scratch_rows, "success_rate")
    scratch_ret = _mean_metric(scratch_rows, "mean_return")
    scratch_haz = _mean_metric(scratch_rows, "hazard_per_1k")

    attr_rows: List[Dict[str, Any]] = []
    ds_base = os.path.splitext(str(args.transfer_dataset_out))[0]
    next_transfer_cfg = dict(transfer_cfg)
    next_transfer_dataset = dict(transfer_dataset)
    next_sources = list(sources)
    for idx, src in enumerate(eval_sources):
        src_tag = f"{str(src.verse_name)}_{str(src.run_id)}_{idx}"
        src_out = f"{ds_base}.source_attr.{src_tag}.jsonl"
        src_ds = _build_filtered_transfer_dataset(
            args,
            sources=[src],
            bridge_label_cfg=bridge_label_cfg,
            out_path=str(src_out),
            near_lane_enabled=bool(args.near_lane_enabled),
            far_lane_enabled=bool(args.far_lane_enabled),
        )
        src_rows = int(src_ds.get("transfer_dataset_rows", 0))
        row: Dict[str, Any] = {
            "source_id": str(source_id(src)),
            "verse_name": str(src.verse_name),
            "run_id": str(src.run_id),
            "source_kind": str(src.source_kind),
            "source_lane": str(src.source_lane),
            "source_universe": str(src.source_universe),
            "dataset_path": str(src_ds.get("transfer_dataset_path", "")),
            "dataset_rows": int(src_rows),
            "evaluated": False,
            "gate_ok": False,
            "skip_reason": "",
        }
        if src_rows < max(1, int(args.source_attribution_min_rows)):
            row["skip_reason"] = "insufficient_rows"
            attr_rows.append(row)
            continue

        src_cfg = _refresh_transfer_mix_decay(
            args,
            cfg={**transfer_cfg, "dataset_path": str(src_ds["transfer_dataset_path"])},
            transfer_rows=int(src_rows),
        )
        src_seed_rows: List[Dict[str, Any]] = []
        for j, seed_j in enumerate(attr_seed_list):
            run_id = run_agent(
                trainer=trainer_attr,
                role=f"source_attr_{idx}",
                verse_name=str(args.target_verse),
                episodes=int(attr_eps),
                max_steps=int(attr_max_steps),
                seed=int(seed_j),
                algo=str(args.transfer_algo),
                policy_id=f"source_attr_{idx}_{args.transfer_algo}_{args.target_verse}_s{j}",
                cfg=dict(src_cfg),
            )
            run_dir = os.path.join(str(args.runs_root), run_id)
            st = evaluate_run(run_dir)
            agg = _aggregate_curve(_episode_curve(run_dir))
            src_seed_rows.append(
                {
                    "run_id": str(run_id),
                    "run_dir": str(run_dir).replace("\\", "/"),
                    "seed": int(seed_j),
                    "success_rate": float(_safe_float(st.success_rate, 0.0)),
                    "mean_return": float(_safe_float(st.mean_return, 0.0)),
                    "hazard_per_1k": float(_safe_float(agg.get("hazard_events_per_1k_steps", 0.0), 0.0)),
                }
            )
        cand_sr = _mean_metric(src_seed_rows, "success_rate")
        cand_ret = _mean_metric(src_seed_rows, "mean_return")
        cand_haz = _mean_metric(src_seed_rows, "hazard_per_1k")
        comp = transfer_mode_utility(
            candidate_success_rate=float(cand_sr),
            candidate_mean_return=float(cand_ret),
            candidate_hazard_per_1k=float(cand_haz),
            scratch_success_rate=float(scratch_sr),
            scratch_mean_return=float(scratch_ret),
            scratch_hazard_per_1k=float(scratch_haz),
            success_weight=float(args.source_attribution_success_weight),
            return_weight=float(args.source_attribution_return_weight),
            hazard_weight=float(args.source_attribution_hazard_weight),
        )
        hazard_regression_ok = bool(
            float(comp["hazard_gain_per_1k"]) >= -float(args.source_attribution_max_hazard_regression_per_1k)
        )
        gate_ok = bool(
            float(comp["utility"]) >= float(args.source_attribution_min_utility)
            and float(comp["success_delta"]) >= float(args.source_attribution_min_success_delta)
            and float(comp["hazard_gain_per_1k"]) >= float(args.source_attribution_min_hazard_gain_per_1k)
            and hazard_regression_ok
        )
        row.update(
            {
                "evaluated": True,
                "success_rate": float(cand_sr),
                "mean_return": float(cand_ret),
                "hazard_per_1k": float(cand_haz),
                "utility": float(comp["utility"]),
                "success_delta_vs_scratch": float(comp["success_delta"]),
                "return_delta_vs_scratch": float(comp["return_delta"]),
                "hazard_gain_vs_scratch_per_1k": float(comp["hazard_gain_per_1k"]),
                "hazard_regression_ok": bool(hazard_regression_ok),
                "gate_ok": bool(gate_ok),
                "per_seed": src_seed_rows,
            }
        )
        attr_rows.append(row)

    selection = pick_sources_from_attribution(
        all_sources=list(sources),
        attribution_rows=attr_rows,
        min_keep_sources=max(0, int(args.source_attribution_min_keep_sources)),
        keep_unscored=bool(args.source_attribution_keep_unscored_sources),
    )
    selected_sources = list(selection.get("kept_sources") or [])
    if selected_sources and len(selected_sources) < len(sources):
        pruned_ds = _build_filtered_transfer_dataset(
            args,
            sources=selected_sources,
            bridge_label_cfg=bridge_label_cfg,
            out_path=str(args.transfer_dataset_out),
            near_lane_enabled=bool(args.near_lane_enabled),
            far_lane_enabled=bool(args.far_lane_enabled),
        )
        if int(pruned_ds.get("transfer_dataset_rows", 0)) > 0:
            next_sources = list(selected_sources)
            next_transfer_dataset = dict(pruned_ds)
            next_transfer_cfg = _refresh_transfer_mix_decay(
                args,
                cfg={**transfer_cfg, "dataset_path": str(pruned_ds["transfer_dataset_path"])},
                transfer_rows=int(pruned_ds.get("transfer_dataset_rows", 0)),
            )
        else:
            selection["reason"] = "reverted_prune_empty_dataset"

    report = {
        "enabled": True,
        "applied": True,
        "reason": str(selection.get("reason", "evaluated")),
        "pilot": {
            "episodes": int(attr_eps),
            "max_steps": int(attr_max_steps),
            "num_seeds": int(attr_seed_count),
            "seed_stride": int(attr_seed_stride),
            "seed_list": [int(s) for s in attr_seed_list],
            "scratch_rows": scratch_rows,
            "scratch_mean": {
                "success_rate": float(scratch_sr),
                "mean_return": float(scratch_ret),
                "hazard_per_1k": float(scratch_haz),
            },
            "weights": {
                "success_weight": float(args.source_attribution_success_weight),
                "return_weight": float(args.source_attribution_return_weight),
                "hazard_weight": float(args.source_attribution_hazard_weight),
            },
            "gates": {
                "min_utility": float(args.source_attribution_min_utility),
                "min_success_delta": float(args.source_attribution_min_success_delta),
                "min_hazard_gain_per_1k": float(args.source_attribution_min_hazard_gain_per_1k),
                "max_hazard_regression_per_1k": float(args.source_attribution_max_hazard_regression_per_1k),
            },
        },
        "evaluated_sources": attr_rows,
        "selection": {
            "kept_source_ids": list(selection.get("kept_source_ids") or []),
            "dropped_source_ids": list(selection.get("dropped_source_ids") or []),
            "kept_count": int(len(selection.get("kept_source_ids") or [])),
            "dropped_count": int(len(selection.get("dropped_source_ids") or [])),
        },
    }
    return {
        "report": report,
        "sources": next_sources,
        "transfer_dataset": next_transfer_dataset,
        "transfer_cfg": next_transfer_cfg,
    }


def run_robust_selector(
    *,
    args: Any,
    sources: List[_SourceDNA],
    transfer_dataset: Dict[str, Any],
    transfer_cfg: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    bridge_label_cfg: Dict[str, Any],
    selected_transfer_algo: str,
    empty_transfer_dataset: bool,
    run_agent: Callable[..., str],
    pick_robust_transfer_mode: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "enabled": bool(args.robust_transfer_selector),
        "applied": False,
        "selected_algo": str(selected_transfer_algo),
        "selected_mode": ("scratch_control" if empty_transfer_dataset else "transfer_all_lanes"),
        "reason": ("empty_transfer_dataset" if empty_transfer_dataset else "not_run"),
        "pilot": {},
    }
    if not bool(args.robust_transfer_selector) or bool(empty_transfer_dataset):
        return {
            "report": report,
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
            "selected_transfer_algo": str(selected_transfer_algo),
        }

    pilot_eps = max(1, min(int(args.episodes), int(max(1, args.robust_selector_pilot_episodes))))
    pilot_max_steps = int(args.robust_selector_pilot_max_steps)
    if pilot_max_steps <= 0:
        pilot_max_steps = max(1, int(args.max_steps))
    pilot_seed_stride = max(1, int(args.robust_selector_seed_stride))
    pilot_candidates: List[Dict[str, Any]] = [
        {
            "mode": "scratch_control",
            "algo": str(args.baseline_algo),
            "cfg": dict(baseline_cfg),
            "dataset_path": "",
            "dataset_rows": 0,
        },
        {
            "mode": "transfer_all_lanes",
            "algo": str(args.transfer_algo),
            "cfg": dict(transfer_cfg),
            "dataset_path": str(transfer_dataset.get("transfer_dataset_path", "")),
            "dataset_rows": int(transfer_dataset.get("transfer_dataset_rows", 0)),
        },
    ]

    near_sources = [s for s in sources if str(s.source_lane or "") == "near_universe"]
    near_ds: Dict[str, Any] | None = None
    if bool(args.near_lane_enabled) and near_sources:
        near_out = os.path.splitext(str(args.transfer_dataset_out))[0] + ".near_lane_only.jsonl"
        near_ds = _build_filtered_transfer_dataset(
            args,
            sources=near_sources,
            bridge_label_cfg=bridge_label_cfg,
            out_path=str(near_out),
            near_lane_enabled=True,
            far_lane_enabled=False,
        )
        if int(near_ds.get("transfer_dataset_rows", 0)) > 0:
            pilot_candidates.append(
                {
                    "mode": "transfer_near_lane",
                    "algo": str(args.transfer_algo),
                    "cfg": _refresh_transfer_mix_decay(
                        args,
                        cfg={**transfer_cfg, "dataset_path": str(near_ds["transfer_dataset_path"])},
                        transfer_rows=int(near_ds.get("transfer_dataset_rows", 0)),
                    ),
                    "dataset_path": str(near_ds.get("transfer_dataset_path", "")),
                    "dataset_rows": int(near_ds.get("transfer_dataset_rows", 0)),
                }
            )

    if len(pilot_candidates) <= 1:
        report.update(
            {
                "enabled": True,
                "selected_algo": str(selected_transfer_algo),
                "selected_mode": "transfer_all_lanes",
                "reason": "insufficient_candidates",
                "pilot": {"candidate_count": int(len(pilot_candidates))},
            }
        )
        return {
            "report": report,
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
            "selected_transfer_algo": str(selected_transfer_algo),
        }

    trainer = Trainer(run_root=str(args.runs_root), schema_version="v1", auto_register_builtin=True)
    pilot_seed_count = max(1, int(args.robust_selector_num_seeds))
    pilot_seed_list = [int(args.seed) + (pilot_seed_stride * (j + 1)) for j in range(pilot_seed_count)]
    pilot_rows: List[Dict[str, Any]] = []
    for i, cand in enumerate(pilot_candidates):
        mode = str(cand.get("mode", "")).strip() or f"candidate_{i}"
        per_seed_rows: List[Dict[str, Any]] = []
        for j, mode_seed in enumerate(pilot_seed_list):
            pilot_algo = str(cand.get("algo", args.transfer_algo))
            run_id = run_agent(
                trainer=trainer,
                role=f"pilot_{mode}",
                verse_name=str(args.target_verse),
                episodes=int(pilot_eps),
                max_steps=int(pilot_max_steps),
                seed=int(mode_seed),
                algo=str(pilot_algo),
                policy_id=f"pilot_{mode}_{pilot_algo}_{args.target_verse}_s{j}",
                cfg=dict(cand.get("cfg") or {}),
            )
            run_dir = os.path.join(str(args.runs_root), run_id)
            st = evaluate_run(run_dir)
            agg = _aggregate_curve(_episode_curve(run_dir))
            per_seed_rows.append(
                {
                    "run_id": str(run_id),
                    "run_dir": str(run_dir).replace("\\", "/"),
                    "seed": int(mode_seed),
                    "algo": str(pilot_algo),
                    "success_rate": float(_safe_float(st.success_rate, 0.0)),
                    "mean_return": float(_safe_float(st.mean_return, 0.0)),
                    "hazard_per_1k": float(_safe_float(agg.get("hazard_events_per_1k_steps", 0.0), 0.0)),
                }
            )
        pilot_rows.append(
            {
                "mode": str(mode),
                "run_id": (str(per_seed_rows[0]["run_id"]) if per_seed_rows else ""),
                "run_dir": (str(per_seed_rows[0]["run_dir"]) if per_seed_rows else ""),
                "seed": (int(per_seed_rows[0]["seed"]) if per_seed_rows else int(args.seed)),
                "seed_list": [int(s) for s in pilot_seed_list],
                "num_seeds": int(len(per_seed_rows)),
                "algo": str(cand.get("algo", args.transfer_algo)),
                "episodes": int(pilot_eps),
                "max_steps": int(pilot_max_steps),
                "success_rate": _mean_metric(per_seed_rows, "success_rate"),
                "mean_return": _mean_metric(per_seed_rows, "mean_return"),
                "hazard_per_1k": _mean_metric(per_seed_rows, "hazard_per_1k"),
                "dataset_path": str(cand.get("dataset_path", "")),
                "dataset_rows": int(cand.get("dataset_rows", 0) or 0),
                "per_seed": per_seed_rows,
            }
        )

    mode_pick = pick_robust_transfer_mode(
        pilot_rows=pilot_rows,
        min_utility=float(args.robust_selector_min_utility),
        min_success_delta=float(args.robust_selector_min_success_delta),
        min_hazard_gain_per_1k=float(args.robust_selector_min_hazard_gain_per_1k),
        max_hazard_regression_per_1k=float(args.robust_selector_max_hazard_regression_per_1k),
        success_weight=float(args.robust_selector_success_weight),
        return_weight=float(args.robust_selector_return_weight),
        hazard_weight=float(args.robust_selector_hazard_weight),
        scratch_mode="scratch_control",
    )
    selected_mode = str(mode_pick.get("selected_mode", "scratch_control"))
    next_transfer_dataset = dict(transfer_dataset)
    next_transfer_cfg = dict(transfer_cfg)
    next_selected_transfer_algo = str(selected_transfer_algo)
    if selected_mode == "transfer_near_lane" and isinstance(near_ds, dict):
        next_transfer_dataset = dict(near_ds)
        next_transfer_cfg = _refresh_transfer_mix_decay(
            args,
            cfg={**transfer_cfg, "dataset_path": str(near_ds["transfer_dataset_path"])},
            transfer_rows=int(near_ds.get("transfer_dataset_rows", 0)),
        )
    elif selected_mode == "scratch_control":
        next_selected_transfer_algo = str(args.baseline_algo)
        next_transfer_cfg = dict(baseline_cfg)

    report = {
        "enabled": True,
        "applied": True,
        "selected_algo": str(next_selected_transfer_algo),
        "selected_mode": str(selected_mode),
        "reason": str(mode_pick.get("reason", "selected")),
        "pilot": {
            "episodes": int(pilot_eps),
            "max_steps": int(pilot_max_steps),
            "seed_stride": int(pilot_seed_stride),
            "num_seeds": int(pilot_seed_count),
            "seed_list": [int(s) for s in pilot_seed_list],
            "rows": pilot_rows,
            "decision": mode_pick,
            "selector_weights": {
                "success_weight": float(args.robust_selector_success_weight),
                "return_weight": float(args.robust_selector_return_weight),
                "hazard_weight": float(args.robust_selector_hazard_weight),
            },
            "selector_gates": {
                "min_utility": float(args.robust_selector_min_utility),
                "min_success_delta": float(args.robust_selector_min_success_delta),
                "min_hazard_gain_per_1k": float(args.robust_selector_min_hazard_gain_per_1k),
                "max_hazard_regression_per_1k": float(args.robust_selector_max_hazard_regression_per_1k),
            },
        },
    }
    return {
        "report": report,
        "transfer_dataset": next_transfer_dataset,
        "transfer_cfg": next_transfer_cfg,
        "selected_transfer_algo": next_selected_transfer_algo,
    }
