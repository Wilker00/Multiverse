from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List


def write_json_artifact(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_transfer_challenge_report(
    *,
    args: Any,
    sources: List[Any],
    source_selection: Dict[str, Any],
    transfer_dataset: Dict[str, Any],
    source_attribution_report: Dict[str, Any],
    robust_selector_report: Dict[str, Any],
    adt_prior_report: Dict[str, Any],
    transfer_score_diag: Dict[str, Any],
    bridge_label_cfg: Dict[str, Any],
    selected_transfer_algo: str,
    transfer_run: str,
    transfer_run_dir: str,
    transfer_stats: Any,
    transfer_early: Dict[str, Any],
    transfer_action_agreement: Dict[str, Any],
    transfer_td_diag: Dict[str, Any],
    agg_transfer: Dict[str, Any],
    transfer_safety_trend: Dict[str, Any],
    transfer_first: Any,
    baseline_run: str,
    baseline_run_dir: str,
    baseline_stats: Any,
    baseline_early: Dict[str, Any],
    baseline_action_agreement: Dict[str, Any],
    baseline_td_diag: Dict[str, Any],
    agg_baseline: Dict[str, Any],
    baseline_safety_trend: Dict[str, Any],
    baseline_first: Any,
    comparison: Dict[str, Any],
    health_scorecard: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "target_verse": str(args.target_verse),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "sources": [
            {
                "verse_name": s.verse_name,
                "path": s.path,
                "run_id": s.run_id,
                "source_kind": s.source_kind,
                "source_lane": s.source_lane,
                "source_universe": s.source_universe,
            }
            for s in sources
        ],
        "source_selection": dict(source_selection),
        "transfer_dataset": dict(transfer_dataset),
        "source_attribution": dict(source_attribution_report),
        "robust_selector": dict(robust_selector_report),
        "adt_prior": dict(adt_prior_report),
        "transfer_dataset_diagnostics": {
            "score_distribution": transfer_score_diag,
            "lane_summary": dict(transfer_dataset.get("lane_merge", {}))
            if isinstance(transfer_dataset.get("lane_merge"), dict)
            else {},
        },
        "bridge_tuning": {
            "synthetic_reward_blend": float(max(0.0, min(1.0, float(args.bridge_synthetic_reward_blend)))),
            "synthetic_done_union": bool(args.bridge_synthetic_done_union),
            "behavioral_bridge_enabled": bool(args.bridge_behavioral_enabled),
            "behavioral_bridge_score_weight": float(max(0.0, min(1.0, float(args.bridge_behavioral_score_weight)))),
            "behavioral_max_prototype_rows": int(max(1, int(args.bridge_behavioral_max_prototype_rows))),
            "lane_controls": {
                "near_lane_enabled": bool(args.near_lane_enabled),
                "far_lane_enabled": bool(args.far_lane_enabled),
                "max_rows_per_source": int(max(0, int(args.max_rows_per_source))),
                "near_lane_max_rows_per_source": int(max(0, int(args.near_lane_max_rows_per_source))),
                "far_lane_max_rows_per_source": int(max(0, int(args.far_lane_max_rows_per_source))),
            },
            "universe_adapter_enabled": bool(args.universe_adapter_enabled),
            "far_lane_weighting": {
                "enabled": bool(args.far_lane_score_weight_enabled),
                "strength": float(max(0.0, min(1.0, float(args.far_lane_score_weight_strength)))),
                "min_universe_feature_score": float(
                    max(0.0, min(1.0, float(args.far_lane_min_universe_feature_score)))
                ),
            },
            "label_cfg": dict(bridge_label_cfg),
        },
        "transfer_agent": {
            "algo": str(selected_transfer_algo),
            "requested_algo": str(args.transfer_algo),
            "selected_mode": str(robust_selector_report.get("selected_mode", "transfer_all_lanes")),
            "warehouse_obs_key_mode": str(args.transfer_q_warehouse_obs_key_mode),
            "config_overrides": {},
            "warmstart": {
                "reward_scale": float(args.transfer_warmstart_reward_scale),
                "max_rows": int(max(0, int(args.transfer_warmstart_max_rows))),
                "balance_actions": bool(args.transfer_warmstart_balance_actions),
                "action_balance_max_share": float(
                    max(0.0, min(1.0, float(args.transfer_warmstart_action_balance_max_share)))
                ),
                "use_transfer_score": bool(args.transfer_warmstart_use_transfer_score),
                "target_mode": str(args.transfer_warmstart_target),
                "target_gamma": float(max(0.0, min(1.0, float(args.transfer_warmstart_target_gamma)))),
                "transfer_score_min": float(args.transfer_warmstart_transfer_score_min),
                "transfer_score_max": float(args.transfer_warmstart_transfer_score_max),
            },
            "dynamic_mixing": {
                "enabled": bool(args.dynamic_transfer_mix),
                "mix_start": float(max(0.0, min(1.0, float(args.transfer_mix_start)))),
                "mix_end": float(max(0.0, min(1.0, float(args.transfer_mix_end)))),
                "mix_decay_steps": int(transfer_dataset.get("transfer_mix_decay_steps", 0))
                if False
                else None,
            },
            "adaptive_veto_schedule": {},
            "run_id": transfer_run,
            "run_dir": transfer_run_dir.replace("\\", "/"),
            "eval": {
                "episodes": int(transfer_stats.episodes),
                "mean_return": float(transfer_stats.mean_return),
                "success_rate": transfer_stats.success_rate,
                "mean_steps": float(transfer_stats.mean_steps),
            },
            "diagnostics": {
                "early_window": transfer_early,
                "action_agreement": transfer_action_agreement,
                "td_error": transfer_td_diag,
            },
            "curve_aggregate": agg_transfer,
            "safety_trend": transfer_safety_trend,
            "first_passable_episode": transfer_first,
        },
        "baseline_agent": {
            "algo": str(args.baseline_algo),
            "warehouse_obs_key_mode": str(args.baseline_q_warehouse_obs_key_mode),
            "config_overrides": {},
            "epsilon_schedule": {
                "epsilon_start": float(args.baseline_epsilon_start),
                "epsilon_min": float(args.baseline_epsilon_min),
                "epsilon_decay": float(args.baseline_epsilon_decay),
            },
            "run_id": baseline_run,
            "run_dir": baseline_run_dir.replace("\\", "/"),
            "eval": {
                "episodes": int(baseline_stats.episodes),
                "mean_return": float(baseline_stats.mean_return),
                "success_rate": baseline_stats.success_rate,
                "mean_steps": float(baseline_stats.mean_steps),
            },
            "diagnostics": {
                "early_window": baseline_early,
                "action_agreement": baseline_action_agreement,
                "td_error": baseline_td_diag,
            },
            "curve_aggregate": agg_baseline,
            "safety_trend": baseline_safety_trend,
            "first_passable_episode": baseline_first,
        },
        "comparison": comparison,
        "health_scorecard": health_scorecard,
        "production_summary": {
            "transfer_gain": {
                "speedup_ratio": comparison.get("transfer_speedup_ratio"),
                "hazard_improvement_pct": comparison.get("hazard_improvement_pct"),
                "transfer_wins_convergence": comparison.get("transfer_wins_convergence"),
            },
            "safety": {
                "transfer_mcts_veto_rate": float(agg_transfer.get("mcts_veto_rate", 0.0)),
                "baseline_mcts_veto_rate": float(agg_baseline.get("mcts_veto_rate", 0.0)),
                "transfer_hazard_per_1k": float(agg_transfer.get("hazard_events_per_1k_steps", 0.0)),
                "baseline_hazard_per_1k": float(agg_baseline.get("hazard_events_per_1k_steps", 0.0)),
                "transfer_veto_rate_improved_over_time": bool(transfer_safety_trend.get("veto_rate_improved", False)),
                "transfer_hazard_rate_improved_over_time": bool(transfer_safety_trend.get("hazard_rate_improved", False)),
            },
            "health": {
                "transfer": (health_scorecard.get("by_role", {}) or {}).get("transfer"),
                "baseline": (health_scorecard.get("by_role", {}) or {}).get("baseline"),
            },
        },
        "output_paths": {
            "report_out": str(args.report_out),
            "overlap_out": str(args.overlap_out),
        },
        "overlap_map_path": str(args.overlap_out),
    }


def enrich_report_agent_details(
    *,
    report: Dict[str, Any],
    args: Any,
    transfer_cfg: Dict[str, Any],
    safe_tune_info: Dict[str, Any],
    parse_cfg_overrides: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    out = dict(report)
    out["transfer_agent"]["config_overrides"] = parse_cfg_overrides(args.transfer_cfg)
    out["transfer_agent"]["dynamic_mixing"] = {
        "enabled": bool(args.dynamic_transfer_mix),
        "mix_start": float(max(0.0, min(1.0, float(args.transfer_mix_start)))),
        "mix_end": float(max(0.0, min(1.0, float(args.transfer_mix_end)))),
        "mix_decay_steps": int(transfer_cfg.get("transfer_mix_decay_steps", 0)),
        "mix_min_rows": int(transfer_cfg.get("transfer_mix_min_rows", 0)),
        "replay_reward_scale": float(transfer_cfg.get("transfer_replay_reward_scale", 0.0)),
        "auto_schedule_used": bool(int(args.transfer_mix_decay_steps) <= 0),
    }
    out["transfer_agent"]["adaptive_veto_schedule"] = {
        "enabled": bool((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_schedule_enabled", False)),
        "relax_start": float((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_relaxation_start", 0.0)),
        "relax_end": float((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_relaxation_end", 0.0)),
        "schedule_steps": int((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_schedule_steps", 0)),
        "schedule_power": float((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_schedule_power", 1.0)),
        "auto_schedule_used": bool(int(args.safe_adaptive_veto_schedule_steps) <= 0),
        "auto_tune_enabled": bool(args.safe_adaptive_veto_auto_tune),
        "auto_tune": dict(safe_tune_info),
    }
    out["baseline_agent"]["config_overrides"] = parse_cfg_overrides(args.baseline_cfg)
    return out


def print_transfer_challenge_summary(
    *,
    report: Dict[str, Any],
    transfer_run: str,
    baseline_run: str,
    transfer_stats: Any,
    baseline_stats: Any,
    agg_transfer: Dict[str, Any],
    agg_baseline: Dict[str, Any],
    safe_float: Callable[[Any, float], float],
    print_chart: Callable[..., None],
    transfer_curve: List[Dict[str, Any]],
    baseline_curve: List[Dict[str, Any]],
    chart_stride: int,
) -> None:
    print(f"transfer_run={transfer_run}")
    print(f"baseline_run={baseline_run}")
    print(
        f"transfer mean_return={float(transfer_stats.mean_return):.3f} "
        f"success={float(transfer_stats.success_rate or 0.0):.3f} "
        f"hazard/1k={float(agg_transfer['hazard_events_per_1k_steps']):.2f}"
    )
    print(
        f"baseline mean_return={float(baseline_stats.mean_return):.3f} "
        f"success={float(baseline_stats.success_rate or 0.0):.3f} "
        f"hazard/1k={float(agg_baseline['hazard_events_per_1k_steps']):.2f}"
    )
    print(
        f"speedup_ratio={report['comparison']['transfer_speedup_ratio']} "
        f"hazard_gain_pct={float(report['comparison']['hazard_improvement_pct']):.2f}"
    )
    print(
        f"robust_selector_mode={str((report.get('robust_selector', {}) or {}).get('selected_mode', 'transfer_all_lanes'))} "
        f"reason={str((report.get('robust_selector', {}) or {}).get('reason', 'n/a'))}"
    )
    hs = (
        report.get("health_scorecard", {}).get("by_role", {})
        if isinstance(report.get("health_scorecard"), dict)
        else {}
    )
    transfer_health = hs.get("transfer") if isinstance(hs, dict) else None
    if isinstance(transfer_health, dict):
        print(
            f"transfer_health_score={float(safe_float(transfer_health.get('total_score', 0.0), 0.0)):.1f} "
            f"status={str(transfer_health.get('status', 'n/a'))}"
        )
    print_chart(
        transfer_curve=transfer_curve,
        baseline_curve=baseline_curve,
        stride=max(1, int(chart_stride)),
    )
    print(f"report: {report['output_paths']['report_out']}")
    print(f"overlap_map: {report['output_paths']['overlap_out']}")
