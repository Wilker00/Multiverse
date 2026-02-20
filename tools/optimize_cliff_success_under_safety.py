"""
tools/optimize_cliff_success_under_safety.py

Optimize hard-cliff success while enforcing a safety upper-bound constraint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from orchestrator.evaluator import evaluate_run
from tools.run_teacher_remediation import build_arg_parser, run_remediation


def _parse_float_grid(raw: str) -> List[float]:
    text = str(raw).strip()
    if not text:
        return []
    out: List[float] = []
    for part in text.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _report_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    cert = report.get("graduation_safety_certificate") or {}
    run = report.get("graduation_run") or {}
    final_holdout_run_id = str(run.get("final_holdout_eval_run_id", "")).strip()
    success_rate = None
    mean_return = None
    mean_steps = None
    if final_holdout_run_id:
        try:
            stats = evaluate_run(os.path.join("runs", final_holdout_run_id))
            success_rate = None if stats.success_rate is None else float(stats.success_rate)
            mean_return = float(stats.mean_return)
            mean_steps = float(stats.mean_steps)
        except Exception:
            pass
    return {
        "upper_bound_95": float(cert.get("upper_bound", 1.0)) if isinstance(cert, dict) and cert else 1.0,
        "observed_violation_rate": float(cert.get("observed_violation_rate", 1.0))
        if isinstance(cert, dict) and cert
        else 1.0,
        "observed_violations": int(cert.get("observed_violations", 0)) if isinstance(cert, dict) and cert else 0,
        "episodes": int(cert.get("episodes", 0)) if isinstance(cert, dict) and cert else 0,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_steps": mean_steps,
        "final_holdout_eval_run_id": final_holdout_run_id,
    }


def _rank_key(row: Dict[str, Any], safety_upper_max: float) -> Tuple[int, float, float, float]:
    feasible = bool(float(row.get("upper_bound_95", 1.0)) <= float(safety_upper_max))
    success = float(row.get("success_rate", -1.0) if row.get("success_rate", None) is not None else -1.0)
    upper = float(row.get("upper_bound_95", 1.0))
    vrate = float(row.get("observed_violation_rate", 1.0))
    return (0 if feasible else 1, -success, upper, vrate)


def _build_run_args(base: argparse.Namespace, *, seed: int) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base))
    args.seed = int(seed)
    return args


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize cliff hard-regime success under safety-bound constraints.")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--search_tutorial_episodes", type=int, default=120)
    ap.add_argument("--search_graduation_episodes", type=int, default=600)
    ap.add_argument("--search_stage_eval_episodes", type=int, default=100)
    ap.add_argument("--search_final_holdout_episodes", type=int, default=120)

    ap.add_argument("--confirm_best", action="store_true")
    ap.add_argument("--confirm_tutorial_episodes", type=int, default=120)
    ap.add_argument("--confirm_graduation_episodes", type=int, default=900)
    ap.add_argument("--confirm_stage_eval_episodes", type=int, default=120)
    ap.add_argument("--confirm_final_holdout_episodes", type=int, default=200)

    ap.add_argument("--q_success_bonus_grid", type=str, default="0.2,0.5,1.0")
    ap.add_argument("--q_hazard_penalty_grid", type=str, default="40,50,60")
    ap.add_argument("--q_epsilon_start_grid", type=str, default="0.2,0.3")
    ap.add_argument("--max_trials", type=int, default=0)

    ap.add_argument("--safety_upper_max", type=float, default=0.35)
    ap.add_argument("--out_json", type=str, default=os.path.join("models", "validation", "optimize_cliff_success_under_safety.json"))
    ap.add_argument(
        "--per_run_out_dir",
        type=str,
        default=os.path.join("models", "validation", "optimize_cliff_success_runs"),
    )
    args = ap.parse_args()

    success_grid = _parse_float_grid(args.q_success_bonus_grid)
    hazard_grid = _parse_float_grid(args.q_hazard_penalty_grid)
    eps_start_grid = _parse_float_grid(args.q_epsilon_start_grid)
    if not success_grid or not hazard_grid or not eps_start_grid:
        raise RuntimeError("All grids must be non-empty.")

    combos = list(itertools.product(success_grid, hazard_grid, eps_start_grid))
    if int(args.max_trials) > 0:
        combos = combos[: int(args.max_trials)]

    os.makedirs(str(args.per_run_out_dir), exist_ok=True)
    base_parser = build_arg_parser()
    base_defaults = base_parser.parse_args([])
    rows: List[Dict[str, Any]] = []

    for idx, (success_bonus, hazard_penalty, eps_start) in enumerate(combos, start=1):
        run_args = _build_run_args(base_defaults, seed=int(args.seed + idx))
        run_args.runs_root = str(args.runs_root)
        run_args.target_verse = "cliff_world"
        run_args.tutorial_verse = "wind_master_world"
        run_args.algo = "q"
        run_args.force_tutorial = True
        run_args.force_graduation = True
        run_args.staged_graduation = True

        run_args.tutorial_episodes = int(args.search_tutorial_episodes)
        run_args.graduation_episodes = int(args.search_graduation_episodes)
        run_args.stage_eval_episodes = int(args.search_stage_eval_episodes)
        run_args.final_holdout_episodes = int(args.search_final_holdout_episodes)
        run_args.eval_epsilon = 0.0

        run_args.stage_gate_enabled = True
        run_args.stage_gate_required_stages = 2
        run_args.stage_gate_use_upper_bound = True
        run_args.stage_gate_max_violation_rate = float(args.safety_upper_max)

        run_args.safe_guard_graduation = False
        run_args.teacher_risk_obs_key = "cliff_adjacent"
        run_args.teacher_risk_threshold = 1.0
        run_args.teacher_failure_rate_threshold = 0.20

        run_args.tutorial_target_margin = 4
        run_args.tutorial_gust_probability = 0.20
        run_args.tutorial_edge_penalty = -5.5
        run_args.tutorial_margin_reward_scale = 0.20

        run_args.q_lr = 0.08
        run_args.q_gamma = 0.995
        run_args.q_epsilon_start = float(eps_start)
        run_args.q_epsilon_min = 0.0
        run_args.q_epsilon_decay = 0.999
        run_args.q_learn_hazard_penalty = float(hazard_penalty)
        run_args.q_learn_success_bonus = float(success_bonus)
        run_args.q_warmstart_reward_scale = 3.0
        run_args.q_dataset_path = os.path.join("models", "expert_datasets", "cliff_world_combined_safety.jsonl")
        run_args.q_augment_cliff_dataset = False

        started = time.time()
        report = run_remediation(run_args)
        elapsed = float(time.time() - started)
        metrics = _report_metrics(report)

        row = {
            "trial_index": int(idx),
            "config": {
                "q_learn_success_bonus": float(success_bonus),
                "q_learn_hazard_penalty": float(hazard_penalty),
                "q_epsilon_start": float(eps_start),
            },
            "metrics": metrics,
            "elapsed_sec": float(elapsed),
            "tutorial_run_id": str(report.get("tutorial_run_id", "")),
            "graduation_run_id": str(report.get("graduation_run_id", "")),
            "stage_gate_status": report.get("stage_gate_status", {}),
        }
        out_path = os.path.join(str(args.per_run_out_dir), f"trial_{idx:03d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        row["report_path"] = out_path
        rows.append(row)

        print(
            f"[{idx}/{len(combos)}] bonus={success_bonus:.3f} hazard={hazard_penalty:.1f} eps0={eps_start:.3f} "
            f"success={metrics['success_rate']} upper={metrics['upper_bound_95']:.4f}"
        )

    ranked = sorted(
        rows,
        key=lambda r: _rank_key(r["metrics"], safety_upper_max=float(args.safety_upper_max)),
    )
    best = ranked[0] if ranked else None

    confirmation: Dict[str, Any] = {}
    if bool(args.confirm_best) and isinstance(best, dict):
        cfg = best.get("config", {})
        run_args = _build_run_args(base_defaults, seed=int(args.seed + 100000))
        run_args.runs_root = str(args.runs_root)
        run_args.target_verse = "cliff_world"
        run_args.tutorial_verse = "wind_master_world"
        run_args.algo = "q"
        run_args.force_tutorial = True
        run_args.force_graduation = True
        run_args.staged_graduation = True
        run_args.safe_guard_graduation = False

        run_args.tutorial_episodes = int(args.confirm_tutorial_episodes)
        run_args.graduation_episodes = int(args.confirm_graduation_episodes)
        run_args.stage_eval_episodes = int(args.confirm_stage_eval_episodes)
        run_args.final_holdout_episodes = int(args.confirm_final_holdout_episodes)
        run_args.eval_epsilon = 0.0

        run_args.stage_gate_enabled = True
        run_args.stage_gate_required_stages = 2
        run_args.stage_gate_use_upper_bound = True
        run_args.stage_gate_max_violation_rate = float(args.safety_upper_max)
        run_args.teacher_risk_obs_key = "cliff_adjacent"
        run_args.teacher_risk_threshold = 1.0

        run_args.tutorial_target_margin = 4
        run_args.tutorial_gust_probability = 0.20
        run_args.tutorial_edge_penalty = -5.5
        run_args.tutorial_margin_reward_scale = 0.20

        run_args.q_lr = 0.08
        run_args.q_gamma = 0.995
        run_args.q_epsilon_start = float(cfg.get("q_epsilon_start", 0.30))
        run_args.q_epsilon_min = 0.0
        run_args.q_epsilon_decay = 0.999
        run_args.q_learn_hazard_penalty = float(cfg.get("q_learn_hazard_penalty", 50.0))
        run_args.q_learn_success_bonus = float(cfg.get("q_learn_success_bonus", 0.2))
        run_args.q_warmstart_reward_scale = 3.0
        run_args.q_dataset_path = os.path.join("models", "expert_datasets", "cliff_world_combined_safety.jsonl")
        run_args.q_augment_cliff_dataset = False

        started = time.time()
        conf_report = run_remediation(run_args)
        conf_elapsed = float(time.time() - started)
        conf_metrics = _report_metrics(conf_report)
        conf_path = os.path.join(str(args.per_run_out_dir), "confirm_best.json")
        with open(conf_path, "w", encoding="utf-8") as f:
            json.dump(conf_report, f, ensure_ascii=False, indent=2)
        confirmation = {
            "config": dict(cfg),
            "metrics": conf_metrics,
            "elapsed_sec": conf_elapsed,
            "report_path": conf_path,
            "tutorial_run_id": str(conf_report.get("tutorial_run_id", "")),
            "graduation_run_id": str(conf_report.get("graduation_run_id", "")),
        }

    payload = {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "objective": {
            "maximize": "success_rate",
            "subject_to": f"upper_bound_95 <= {float(args.safety_upper_max):.4f}",
        },
        "search_space": {
            "q_success_bonus_grid": success_grid,
            "q_hazard_penalty_grid": hazard_grid,
            "q_epsilon_start_grid": eps_start_grid,
            "num_trials": int(len(combos)),
        },
        "ranked_trials": ranked,
        "best_trial": best,
        "confirmation": confirmation,
    }
    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"optimization_complete trials={len(combos)} out={args.out_json}")
    if isinstance(best, dict):
        m = best.get("metrics", {})
        print(
            "best_trial "
            f"success={m.get('success_rate')} "
            f"upper={float(m.get('upper_bound_95', 1.0)):.4f} "
            f"config={best.get('config', {})}"
        )


if __name__ == "__main__":
    main()
