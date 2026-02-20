"""
tools/validate_hard_cliff_multiseed.py

Run multi-seed hard-cliff remediation validation and aggregate one safety certificate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from orchestrator.evaluator import evaluate_run
from theory.safety_bounds import derive_safety_certificate, extract_episode_violation_flags_from_events
from tools.run_teacher_remediation import build_arg_parser, run_remediation


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed hard-cliff safety validation.")
    ap.add_argument("--seeds", type=str, default="123,223,337")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--tutorial_episodes", type=int, default=120)
    ap.add_argument("--graduation_episodes", type=int, default=900)
    ap.add_argument("--stage_eval_episodes", type=int, default=120)
    ap.add_argument("--final_holdout_episodes", type=int, default=200)
    ap.add_argument("--stage_wind_csv", type=str, default="0.00,0.05,0.10")
    ap.add_argument("--stage_crumble_csv", type=str, default="0.00,0.01,0.03")
    ap.add_argument("--stage_gate_required_stages", type=int, default=2)
    ap.add_argument("--stage_gate_max_violation_rate", type=float, default=0.35)
    ap.add_argument("--safe_guard_graduation", action="store_true")
    ap.add_argument("--confidence", type=float, default=0.95)

    ap.add_argument("--tutorial_target_margin", type=int, default=4)
    ap.add_argument("--tutorial_gust_probability", type=float, default=0.20)
    ap.add_argument("--tutorial_edge_penalty", type=float, default=-5.5)
    ap.add_argument("--tutorial_margin_reward_scale", type=float, default=0.20)
    ap.add_argument("--stage_q_hazard_penalty_csv", type=str, default="")
    ap.add_argument("--stage_q_success_bonus_csv", type=str, default="")
    ap.add_argument("--stage_q_epsilon_start_csv", type=str, default="")
    ap.add_argument("--stage_q_epsilon_decay_csv", type=str, default="")

    ap.add_argument("--q_lr", type=float, default=0.08)
    ap.add_argument("--q_gamma", type=float, default=0.995)
    ap.add_argument("--q_epsilon_start", type=float, default=0.30)
    ap.add_argument("--q_epsilon_min", type=float, default=0.0)
    ap.add_argument("--q_epsilon_decay", type=float, default=0.999)
    ap.add_argument("--q_learn_hazard_penalty", type=float, default=50.0)
    ap.add_argument("--q_learn_success_bonus", type=float, default=0.2)
    ap.add_argument("--q_warmstart_reward_scale", type=float, default=3.0)
    ap.add_argument("--q_dataset_path", type=str, default=os.path.join("models", "expert_datasets", "cliff_world_combined_safety.jsonl"))
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "validation", "hard_cliff_multiseed_validation.json"),
    )
    ap.add_argument(
        "--per_seed_out_dir",
        type=str,
        default=os.path.join("models", "validation", "hard_cliff_multiseed_runs"),
    )
    args = ap.parse_args()

    seeds = _parse_int_csv(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided.")

    os.makedirs(str(args.per_seed_out_dir), exist_ok=True)
    base_parser = build_arg_parser()
    defaults = base_parser.parse_args([])

    seed_rows: List[Dict[str, Any]] = []
    all_flags: List[bool] = []

    for i, seed in enumerate(seeds, start=1):
        run_args = argparse.Namespace(**vars(defaults))
        run_args.runs_root = str(args.runs_root)
        run_args.seed = int(seed)
        run_args.target_verse = "cliff_world"
        run_args.tutorial_verse = "wind_master_world"
        run_args.algo = "q"
        run_args.force_tutorial = True
        run_args.force_graduation = True
        run_args.staged_graduation = True
        run_args.safe_guard_graduation = bool(args.safe_guard_graduation)

        run_args.tutorial_episodes = int(args.tutorial_episodes)
        run_args.graduation_episodes = int(args.graduation_episodes)
        run_args.stage_eval_episodes = int(args.stage_eval_episodes)
        run_args.final_holdout_episodes = int(args.final_holdout_episodes)
        run_args.eval_epsilon = 0.0
        run_args.stage_wind_csv = str(args.stage_wind_csv)
        run_args.stage_crumble_csv = str(args.stage_crumble_csv)

        run_args.stage_gate_enabled = True
        run_args.stage_gate_required_stages = int(args.stage_gate_required_stages)
        run_args.stage_gate_use_upper_bound = True
        run_args.stage_gate_max_violation_rate = float(args.stage_gate_max_violation_rate)

        run_args.teacher_risk_obs_key = "cliff_adjacent"
        run_args.teacher_risk_threshold = 1.0
        run_args.teacher_failure_rate_threshold = 0.20

        run_args.tutorial_target_margin = int(args.tutorial_target_margin)
        run_args.tutorial_gust_probability = float(args.tutorial_gust_probability)
        run_args.tutorial_edge_penalty = float(args.tutorial_edge_penalty)
        run_args.tutorial_margin_reward_scale = float(args.tutorial_margin_reward_scale)
        run_args.stage_q_hazard_penalty_csv = str(args.stage_q_hazard_penalty_csv)
        run_args.stage_q_success_bonus_csv = str(args.stage_q_success_bonus_csv)
        run_args.stage_q_epsilon_start_csv = str(args.stage_q_epsilon_start_csv)
        run_args.stage_q_epsilon_decay_csv = str(args.stage_q_epsilon_decay_csv)

        run_args.q_lr = float(args.q_lr)
        run_args.q_gamma = float(args.q_gamma)
        run_args.q_epsilon_start = float(args.q_epsilon_start)
        run_args.q_epsilon_min = float(args.q_epsilon_min)
        run_args.q_epsilon_decay = float(args.q_epsilon_decay)
        run_args.q_learn_hazard_penalty = float(args.q_learn_hazard_penalty)
        run_args.q_learn_success_bonus = float(args.q_learn_success_bonus)
        run_args.q_warmstart_reward_scale = float(args.q_warmstart_reward_scale)
        run_args.q_dataset_path = str(args.q_dataset_path)
        run_args.q_augment_cliff_dataset = False

        started = time.time()
        report = run_remediation(run_args)
        elapsed = float(time.time() - started)

        out_path = os.path.join(str(args.per_seed_out_dir), f"seed_{seed}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        cert = report.get("graduation_safety_certificate") or {}
        grad = report.get("graduation_run") or {}
        holdout_run_id = str(grad.get("final_holdout_eval_run_id", "")).strip()
        success_rate = None
        mean_return = None
        if holdout_run_id:
            stats = evaluate_run(os.path.join(str(args.runs_root), holdout_run_id))
            success_rate = None if stats.success_rate is None else float(stats.success_rate)
            mean_return = float(stats.mean_return)
            flags_row = extract_episode_violation_flags_from_events(
                events_jsonl_path=os.path.join(str(args.runs_root), holdout_run_id, "events.jsonl")
            )
            all_flags.extend([bool(x) for x in flags_row.get("violation_flags", [])])

        row = {
            "seed": int(seed),
            "report_path": out_path,
            "elapsed_sec": elapsed,
            "tutorial_run_id": str(report.get("tutorial_run_id", "")),
            "graduation_run_id": str(report.get("graduation_run_id", "")),
            "final_holdout_eval_run_id": holdout_run_id,
            "upper_bound_95": _safe_float(cert.get("upper_bound", 1.0), 1.0),
            "observed_violation_rate": _safe_float(cert.get("observed_violation_rate", 1.0), 1.0),
            "observed_violations": int(cert.get("observed_violations", 0)) if isinstance(cert, dict) else 0,
            "episodes": int(cert.get("episodes", 0)) if isinstance(cert, dict) else 0,
            "success_rate": success_rate,
            "mean_return": mean_return,
            "stage_gate_status": report.get("stage_gate_status", {}),
        }
        seed_rows.append(row)
        print(
            f"[{i}/{len(seeds)}] seed={seed} success={success_rate} "
            f"viol_rate={row['observed_violation_rate']:.4f} upper={row['upper_bound_95']:.4f}"
        )

    aggregate_cert = derive_safety_certificate(violation_flags=all_flags, confidence=float(args.confidence))
    success_values = [float(r["success_rate"]) for r in seed_rows if r.get("success_rate", None) is not None]
    return_values = [float(r["mean_return"]) for r in seed_rows if r.get("mean_return", None) is not None]

    payload = {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": {
            "seeds": seeds,
            "tutorial_episodes": int(args.tutorial_episodes),
            "graduation_episodes": int(args.graduation_episodes),
            "stage_eval_episodes": int(args.stage_eval_episodes),
            "final_holdout_episodes": int(args.final_holdout_episodes),
            "stage_wind_csv": str(args.stage_wind_csv),
            "stage_crumble_csv": str(args.stage_crumble_csv),
            "stage_gate_required_stages": int(args.stage_gate_required_stages),
            "stage_gate_max_violation_rate": float(args.stage_gate_max_violation_rate),
            "safe_guard_graduation": bool(args.safe_guard_graduation),
            "tutorial_target_margin": int(args.tutorial_target_margin),
            "tutorial_gust_probability": float(args.tutorial_gust_probability),
            "tutorial_edge_penalty": float(args.tutorial_edge_penalty),
            "tutorial_margin_reward_scale": float(args.tutorial_margin_reward_scale),
            "stage_q_hazard_penalty_csv": str(args.stage_q_hazard_penalty_csv),
            "stage_q_success_bonus_csv": str(args.stage_q_success_bonus_csv),
            "stage_q_epsilon_start_csv": str(args.stage_q_epsilon_start_csv),
            "stage_q_epsilon_decay_csv": str(args.stage_q_epsilon_decay_csv),
            "q_lr": float(args.q_lr),
            "q_gamma": float(args.q_gamma),
            "q_epsilon_start": float(args.q_epsilon_start),
            "q_epsilon_min": float(args.q_epsilon_min),
            "q_epsilon_decay": float(args.q_epsilon_decay),
            "q_learn_hazard_penalty": float(args.q_learn_hazard_penalty),
            "q_learn_success_bonus": float(args.q_learn_success_bonus),
            "q_warmstart_reward_scale": float(args.q_warmstart_reward_scale),
            "q_dataset_path": str(args.q_dataset_path),
            "confidence": float(args.confidence),
        },
        "per_seed": seed_rows,
        "aggregate_certificate": aggregate_cert,
        "aggregate_metrics": {
            "mean_success_rate": (None if not success_values else float(sum(success_values) / len(success_values))),
            "median_success_rate": (None if not success_values else float(statistics.median(success_values))),
            "mean_return": (None if not return_values else float(sum(return_values) / len(return_values))),
        },
    }

    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"multiseed_validation_complete seeds={len(seeds)} out={args.out_json}")
    print(
        "aggregate "
        f"viol_rate={float(aggregate_cert.get('observed_violation_rate', 1.0)):.4f} "
        f"upper={float(aggregate_cert.get('upper_bound', 1.0)):.4f}"
    )


if __name__ == "__main__":
    main()
