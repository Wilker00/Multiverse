from __future__ import annotations

import argparse
import os
from typing import Any, Dict


def build_transfer_challenge_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--target_verse", type=str, default="warehouse_world")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--baseline_algo", type=str, default="q")
    ap.add_argument(
        "--baseline_cfg",
        action="append",
        default=None,
        help="Extra baseline agent config override in key=value form (repeatable).",
    )
    ap.add_argument("--baseline_epsilon_start", type=float, default=0.05)
    ap.add_argument("--baseline_epsilon_min", type=float, default=0.01)
    ap.add_argument("--baseline_epsilon_decay", type=float, default=0.999)
    ap.add_argument("--transfer_algo", type=str, default="q")
    ap.add_argument(
        "--transfer_cfg",
        action="append",
        default=None,
        help="Extra transfer agent config override in key=value form (repeatable).",
    )
    ap.add_argument("--source_dna", action="append", default=None)
    ap.add_argument("--source_verse", action="append", default=None)
    ap.add_argument("--max_source_runs_per_verse", type=int, default=2)
    ap.add_argument("--min_source_success_rate", type=float, default=0.55)
    ap.add_argument("--min_rows_per_source", type=int, default=50)
    ap.add_argument("--max_source_scan", type=int, default=200, help="Max strategy-verse runs to evaluate during discovery (0=unlimited)")
    ap.add_argument("--max_rows_per_source", type=int, default=2500)
    ap.add_argument("--near_lane_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--far_lane_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--near_lane_max_rows_per_source",
        type=int,
        default=0,
        help="0 => use --max_rows_per_source for near-universe sources",
    )
    ap.add_argument(
        "--far_lane_max_rows_per_source",
        type=int,
        default=0,
        help="0 => auto smaller cap for far-universe sources",
    )
    ap.add_argument(
        "--universe_adapter_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate translated rows with shared universe features (diagnostic metadata only).",
    )
    ap.add_argument(
        "--far_lane_score_weight_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downweight far-universe transfer rows using universe adapter feature quality.",
    )
    ap.add_argument(
        "--far_lane_score_weight_strength",
        type=float,
        default=0.35,
        help="Blend strength for far-lane score weighting (0..1).",
    )
    ap.add_argument(
        "--far_lane_min_universe_feature_score",
        type=float,
        default=0.0,
        help="Drop far-lane translated rows below this universe feature quality score (0..1).",
    )

    ap.add_argument("--bridge_synthetic_reward_blend", type=float, default=0.75)
    ap.add_argument("--bridge_synthetic_done_union", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bridge_confidence_threshold", type=float, default=0.35, help="Min confidence to keep a translated row (0..1)")
    ap.add_argument("--bridge_behavioral_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bridge_behavioral_score_weight", type=float, default=0.35)
    ap.add_argument("--bridge_behavioral_max_prototype_rows", type=int, default=4096)
    ap.add_argument("--bridge_warehouse_step_penalty", type=float, default=0.08)
    ap.add_argument("--bridge_warehouse_wall_penalty", type=float, default=0.40)
    ap.add_argument("--bridge_warehouse_obstacle_penalty", type=float, default=0.70)
    ap.add_argument("--bridge_warehouse_charge_reward", type=float, default=0.50)
    ap.add_argument("--bridge_warehouse_progress_bonus", type=float, default=0.35)
    ap.add_argument("--bridge_warehouse_regress_penalty", type=float, default=0.08)
    ap.add_argument("--bridge_warehouse_goal_reward", type=float, default=10.0)
    ap.add_argument("--bridge_warehouse_battery_fail_penalty", type=float, default=10.0)
    ap.add_argument("--transfer_filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--transfer_filter_dedupe", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--empty_transfer_dataset_policy",
        type=str,
        default="error",
        choices=["error", "continue"],
        help="Behavior when translated transfer dataset has zero rows.",
    )
    ap.add_argument(
        "--empty_transfer_dataset_diag_out",
        type=str,
        default="",
        help="Optional diagnostics JSON path written when transfer dataset is empty.",
    )
    ap.add_argument(
        "--transfer_filter_hazard_keep_ratio",
        type=float,
        default=1.0,
        help="Deterministic keep ratio for hazard-labeled synthetic rows (0..1).",
    )
    ap.add_argument(
        "--transfer_filter_min_confidence",
        type=float,
        default=0.0,
        help="Optional filter to drop translated rows below transfer_confidence (0..1).",
    )
    ap.add_argument("--transfer_filter_max_rows", type=int, default=0)
    ap.add_argument(
        "--transfer_dataset_out",
        type=str,
        default=os.path.join("models", "expert_datasets", "transfer_strategy_to_warehouse.jsonl"),
    )
    ap.add_argument("--transfer_warmstart_reward_scale", type=float, default=0.01)
    ap.add_argument("--transfer_warmstart_max_rows", type=int, default=192)
    ap.add_argument(
        "--transfer_warmstart_balance_actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--transfer_warmstart_action_balance_max_share",
        type=float,
        default=0.60,
    )
    ap.add_argument("--transfer_warmstart_use_transfer_score", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--transfer_warmstart_target",
        type=str,
        default="immediate",
        choices=["immediate", "return_to_go"],
    )
    ap.add_argument("--transfer_warmstart_target_gamma", type=float, default=0.99)
    ap.add_argument("--transfer_warmstart_transfer_score_min", type=float, default=0.0)
    ap.add_argument("--transfer_warmstart_transfer_score_max", type=float, default=2.0)
    ap.add_argument(
        "--transfer_q_warehouse_obs_key_mode",
        type=str,
        default="direction_only",
        choices=["direction_only", "task_lite"],
    )
    ap.add_argument("--transfer_epsilon_start", type=float, default=0.70)
    ap.add_argument("--transfer_epsilon_min", type=float, default=0.03)
    ap.add_argument("--transfer_epsilon_decay", type=float, default=0.996)
    ap.add_argument("--transfer_learn_hazard_penalty", type=float, default=0.0)
    ap.add_argument("--transfer_learn_success_bonus", type=float, default=0.0)
    ap.add_argument("--adt_prior_enabled", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--adt_prior_model_path",
        type=str,
        default=os.path.join("models", "dt_generalist.pt"),
    )
    ap.add_argument(
        "--adt_prior_cfg",
        action="append",
        default=None,
        help="Extra ADT pilot config override in key=value form (repeatable).",
    )
    ap.add_argument("--adt_prior_episodes", type=int, default=16)
    ap.add_argument("--adt_prior_max_steps", type=int, default=0, help="0 => match --max_steps")
    ap.add_argument(
        "--adt_prior_row_mode",
        type=str,
        default="",
        choices=["", "success_only", "top_return", "all_rows"],
        help="ADT pilot row selection policy. Empty preserves legacy --adt_prior_success_only behavior.",
    )
    ap.add_argument("--adt_prior_success_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--adt_prior_top_return_pct", type=float, default=0.25)
    ap.add_argument("--adt_prior_max_rows", type=int, default=96)
    ap.add_argument("--adt_prior_min_rows", type=int, default=8)
    ap.add_argument("--adt_prior_warmstart_reward_scale", type=float, default=0.01)
    ap.add_argument("--adt_prior_auto_rollback", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--adt_prior_rollback_min_success_delta", type=float, default=0.0)
    ap.add_argument("--adt_prior_rollback_min_return_delta", type=float, default=0.0)
    ap.add_argument("--adt_prior_rollback_max_hazard_regression_per_1k", type=float, default=25.0)
    ap.add_argument("--dynamic_transfer_mix", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--transfer_mix_start", type=float, default=1.0)
    ap.add_argument("--transfer_mix_end", type=float, default=0.0)
    ap.add_argument(
        "--transfer_mix_decay_steps",
        type=int,
        default=0,
        help="0 => auto: episodes*max_steps*0.6",
    )
    ap.add_argument("--transfer_mix_min_rows", type=int, default=32)
    ap.add_argument("--transfer_replay_reward_scale", type=float, default=0.8)
    ap.add_argument(
        "--robust_transfer_selector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run short pilot runs (scratch / near-lane / all-lanes) and keep transfer only "
            "when early utility indicates task-solving lift."
        ),
    )
    ap.add_argument("--robust_selector_pilot_episodes", type=int, default=24)
    ap.add_argument(
        "--robust_selector_num_seeds",
        type=int,
        default=2,
        help="Number of shared pilot seeds used for each selector candidate.",
    )
    ap.add_argument(
        "--robust_selector_pilot_max_steps",
        type=int,
        default=0,
        help="0 => match --max_steps for selector pilots",
    )
    ap.add_argument("--robust_selector_seed_stride", type=int, default=17)
    ap.add_argument("--robust_selector_success_weight", type=float, default=100.0)
    ap.add_argument("--robust_selector_return_weight", type=float, default=1.0)
    ap.add_argument("--robust_selector_hazard_weight", type=float, default=0.02)
    ap.add_argument("--robust_selector_min_utility", type=float, default=0.0)
    ap.add_argument("--robust_selector_min_success_delta", type=float, default=0.0)
    ap.add_argument("--robust_selector_min_hazard_gain_per_1k", type=float, default=30.0)
    ap.add_argument("--robust_selector_max_hazard_regression_per_1k", type=float, default=50.0)
    ap.add_argument(
        "--source_attribution_enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pilot each source independently and auto-prune sources that fail utility/safety gates.",
    )
    ap.add_argument("--source_attribution_top_k_sources", type=int, default=0, help="0 => evaluate all sources.")
    ap.add_argument("--source_attribution_pilot_episodes", type=int, default=20)
    ap.add_argument("--source_attribution_pilot_max_steps", type=int, default=0, help="0 => match --max_steps")
    ap.add_argument("--source_attribution_num_seeds", type=int, default=2)
    ap.add_argument("--source_attribution_seed_stride", type=int, default=29)
    ap.add_argument("--source_attribution_min_rows", type=int, default=64)
    ap.add_argument("--source_attribution_success_weight", type=float, default=100.0)
    ap.add_argument("--source_attribution_return_weight", type=float, default=1.0)
    ap.add_argument("--source_attribution_hazard_weight", type=float, default=0.02)
    ap.add_argument("--source_attribution_min_utility", type=float, default=0.0)
    ap.add_argument("--source_attribution_min_success_delta", type=float, default=0.0)
    ap.add_argument("--source_attribution_min_hazard_gain_per_1k", type=float, default=0.0)
    ap.add_argument("--source_attribution_max_hazard_regression_per_1k", type=float, default=50.0)
    ap.add_argument("--source_attribution_min_keep_sources", type=int, default=1)
    ap.add_argument("--source_attribution_keep_unscored_sources", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--safe_transfer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_baseline", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--safe_adaptive_veto", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_adaptive_veto_relaxation", type=float, default=0.35)
    ap.add_argument("--safe_adaptive_veto_schedule", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_adaptive_veto_relax_start", type=float, default=0.12)
    ap.add_argument("--safe_adaptive_veto_relax_end", type=float, default=0.35)
    ap.add_argument("--safe_adaptive_veto_schedule_steps", type=int, default=0)
    ap.add_argument("--safe_adaptive_veto_schedule_power", type=float, default=1.20)
    ap.add_argument("--safe_adaptive_veto_auto_tune", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_adaptive_veto_auto_tune_max_runs", type=int, default=8)
    ap.add_argument("--safe_adaptive_veto_auto_tune_min_runs", type=int, default=2)
    ap.add_argument("--safe_adaptive_veto_auto_tune_policy_prefix", type=str, default="transfer_")
    ap.add_argument("--safe_adaptive_veto_warmup_steps", type=int, default=20)
    ap.add_argument("--safe_adaptive_veto_failure_guard", type=float, default=0.20)
    ap.add_argument("--safe_danger_threshold", type=float, default=0.85)
    ap.add_argument("--safe_min_action_confidence", type=float, default=0.10)
    ap.add_argument("--safe_prefer_fallback_on_veto", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--safe_fallback_horizon_steps", type=int, default=0)
    ap.add_argument("--safe_fallback_algo", type=str, default="gateway")
    ap.add_argument(
        "--safe_fallback_manifest_path",
        type=str,
        default=os.path.join("models", "default_policy_set.json"),
    )
    ap.add_argument("--safe_fallback_manifest_section", type=str, default="deployment_ready_defaults")
    ap.add_argument("--enable_mcts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mcts_meta_model_path", type=str, default=os.path.join("models", "meta_transformer.pt"))
    ap.add_argument("--mcts_num_simulations", type=int, default=96)
    ap.add_argument("--mcts_max_depth", type=int, default=4)
    ap.add_argument("--mcts_loss_threshold", type=float, default=-0.98)
    ap.add_argument("--mcts_min_visits", type=int, default=12)
    ap.add_argument("--mcts_trigger_on_low_confidence", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mcts_value_confidence_threshold", type=float, default=0.30)
    ap.add_argument("--passable_success_rate", type=float, default=0.60)
    ap.add_argument("--passable_mean_return", type=float, default=1.50)
    ap.add_argument("--passable_window", type=int, default=20)
    ap.add_argument("--diagnostic_early_episodes", type=int, default=10)
    ap.add_argument("--diagnostic_action_agreement_first_k", type=int, default=20)
    ap.add_argument(
        "--baseline_q_warehouse_obs_key_mode",
        type=str,
        default="direction_only",
        choices=["direction_only", "task_lite"],
    )
    ap.add_argument("--chart_stride", type=int, default=5)
    ap.add_argument(
        "--health_trace_root",
        type=str,
        default=os.path.join("models", "expert_datasets"),
    )
    ap.add_argument("--health_kl_critical", type=float, default=0.25)
    ap.add_argument("--health_stale_kl_threshold", type=float, default=0.12)
    ap.add_argument("--health_unsafe_veto_rate", type=float, default=0.10)
    ap.add_argument("--health_incoherent_match_threshold", type=float, default=0.55)
    ap.add_argument("--health_memory_coherence_threshold", type=float, default=0.55)
    ap.add_argument("--health_max_trace_rows_per_file", type=int, default=20000)
    ap.add_argument(
        "--report_out",
        type=str,
        default=os.path.join("models", "tuning", "transfer_challenge_report.json"),
    )
    ap.add_argument(
        "--overlap_out",
        type=str,
        default=os.path.join("models", "tuning", "transfer_overlap_map.json"),
    )
    return ap


def build_transfer_bridge_label_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    if str(args.target_verse).strip().lower() != "warehouse_world":
        return {}
    return {
        "step_penalty": float(args.bridge_warehouse_step_penalty),
        "wall_penalty": float(args.bridge_warehouse_wall_penalty),
        "obstacle_penalty": float(args.bridge_warehouse_obstacle_penalty),
        "charge_reward": float(args.bridge_warehouse_charge_reward),
        "progress_bonus": float(args.bridge_warehouse_progress_bonus),
        "regress_penalty": float(args.bridge_warehouse_regress_penalty),
        "goal_reward": float(args.bridge_warehouse_goal_reward),
        "battery_fail_penalty": float(args.bridge_warehouse_battery_fail_penalty),
    }
