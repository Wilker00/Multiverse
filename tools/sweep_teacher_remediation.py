"""
tools/sweep_teacher_remediation.py

Grid-sweep Teacher remediation settings and rank by final safety bound.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

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


def _parse_int_grid(raw: str) -> List[int]:
    text = str(raw).strip()
    if not text:
        return []
    out: List[int] = []
    for part in text.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _rank_key(row: Dict[str, Any]) -> Any:
    # Prefer runs that reached the final stage, then lower bounds/rates.
    return (
        0 if bool(row.get("reached_final_stage", False)) else 1,
        0 if not bool(row.get("stage_gate_blocked", False)) else 1,
        float(row.get("upper_bound_95", 1.0)),
        float(row.get("observed_violation_rate", 1.0)),
        -int(row.get("completed_stages", 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep Teacher remediation configs and rank by safety bound.")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--target_verse", type=str, default="cliff_world")
    ap.add_argument("--tutorial_verse", type=str, default="wind_master_world")
    ap.add_argument("--algo", type=str, default="q")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--tutorial_episodes", type=int, default=80)
    ap.add_argument("--graduation_episodes", type=int, default=120)
    ap.add_argument("--tutorial_max_steps", type=int, default=80)
    ap.add_argument("--graduation_max_steps", type=int, default=100)
    ap.add_argument("--stage_wind_csv", type=str, default="0.00,0.05,0.10")
    ap.add_argument("--stage_crumble_csv", type=str, default="0.00,0.01,0.03")
    ap.add_argument("--safe_guard_graduation", action="store_true")
    ap.add_argument("--safety_confidence", type=float, default=0.95)

    ap.add_argument("--tutorial_margin_reward_scale_grid", type=str, default="0.10,0.14")
    ap.add_argument("--tutorial_edge_penalty_grid", type=str, default="-3.5,-4.5")
    ap.add_argument("--tutorial_target_margin_grid", type=str, default="3")
    ap.add_argument("--tutorial_gust_probability_grid", type=str, default="0.25")
    ap.add_argument("--stage_gate_max_violation_rate_grid", type=str, default="0.30,0.40")
    ap.add_argument("--stage_gate_required_stages", type=int, default=2)
    ap.add_argument("--stage_gate_use_upper_bound", action="store_true")
    ap.add_argument("--max_trials", type=int, default=0, help="0 means run all combinations.")

    ap.add_argument("--force_tutorial", action="store_true")
    ap.add_argument("--force_graduation", action="store_true")
    ap.add_argument("--teacher_risk_obs_key", type=str, default="cliff_adjacent")
    ap.add_argument("--teacher_risk_threshold", type=float, default=1.0)
    ap.add_argument("--teacher_failure_rate_threshold", type=float, default=0.20)

    ap.add_argument("--per_run_out_dir", type=str, default=os.path.join("models", "validation", "sweeps"))
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "validation", "teacher_wind_remediation_sweep.json"),
    )
    args = ap.parse_args()

    margin_grid = _parse_float_grid(args.tutorial_margin_reward_scale_grid)
    edge_grid = _parse_float_grid(args.tutorial_edge_penalty_grid)
    target_grid = _parse_int_grid(args.tutorial_target_margin_grid)
    gust_grid = _parse_float_grid(args.tutorial_gust_probability_grid)
    gate_grid = _parse_float_grid(args.stage_gate_max_violation_rate_grid)
    if not margin_grid or not edge_grid or not target_grid or not gust_grid or not gate_grid:
        raise RuntimeError("All sweep grids must be non-empty.")

    combos = list(itertools.product(margin_grid, edge_grid, target_grid, gust_grid, gate_grid))
    if int(args.max_trials) > 0:
        combos = combos[: int(args.max_trials)]

    base_parser = build_arg_parser()
    default_run_args = base_parser.parse_args([])

    os.makedirs(str(args.per_run_out_dir), exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for idx, (margin, edge, target_margin, gust, gate_thr) in enumerate(combos, start=1):
        run_args = argparse.Namespace(**vars(default_run_args))
        run_args.runs_root = str(args.runs_root)
        run_args.target_verse = str(args.target_verse)
        run_args.tutorial_verse = str(args.tutorial_verse)
        run_args.algo = str(args.algo)
        run_args.seed = int(args.seed + idx)
        run_args.tutorial_episodes = int(args.tutorial_episodes)
        run_args.graduation_episodes = int(args.graduation_episodes)
        run_args.tutorial_max_steps = int(args.tutorial_max_steps)
        run_args.graduation_max_steps = int(args.graduation_max_steps)
        run_args.staged_graduation = True
        run_args.stage_wind_csv = str(args.stage_wind_csv)
        run_args.stage_crumble_csv = str(args.stage_crumble_csv)
        run_args.safe_guard_graduation = bool(args.safe_guard_graduation)
        run_args.safety_confidence = float(args.safety_confidence)

        run_args.stage_gate_enabled = True
        run_args.stage_gate_required_stages = int(args.stage_gate_required_stages)
        run_args.stage_gate_max_violation_rate = float(gate_thr)
        run_args.stage_gate_use_upper_bound = bool(args.stage_gate_use_upper_bound)

        run_args.force_tutorial = bool(args.force_tutorial) or True
        run_args.force_graduation = bool(args.force_graduation) or True
        run_args.teacher_risk_obs_key = str(args.teacher_risk_obs_key)
        run_args.teacher_risk_threshold = float(args.teacher_risk_threshold)
        run_args.teacher_failure_rate_threshold = float(args.teacher_failure_rate_threshold)
        run_args.lesson_log_path = os.path.join(str(args.per_run_out_dir), "teacher_lessons_sweep.json")

        run_args.tutorial_margin_reward_scale = float(margin)
        run_args.tutorial_edge_penalty = float(edge)
        run_args.tutorial_target_margin = int(target_margin)
        run_args.tutorial_gust_probability = float(gust)

        started = time.time()
        report = run_remediation(run_args)
        elapsed = float(time.time() - started)

        run_report_path = os.path.join(
            str(args.per_run_out_dir),
            f"teacher_wind_sweep_run_{idx:03d}.json",
        )
        with open(run_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        stage_runs = report.get("graduation_stage_runs") or []
        max_stage = 0
        for row in stage_runs:
            try:
                max_stage = max(max_stage, int(row.get("stage_index", 0)))
            except Exception:
                pass
        cert = report.get("graduation_safety_certificate") or {}
        gate_status = report.get("stage_gate_status") or {}
        rows.append(
            {
                "run_index": int(idx),
                "report_path": run_report_path,
                "tutorial_margin_reward_scale": float(margin),
                "tutorial_edge_penalty": float(edge),
                "tutorial_target_margin": int(target_margin),
                "tutorial_gust_probability": float(gust),
                "stage_gate_max_violation_rate": float(gate_thr),
                "completed_stages": int(max_stage),
                "reached_final_stage": bool(max_stage >= 3),
                "stage_gate_blocked": bool(gate_status.get("blocked_before_stage", None) is not None),
                "stage_gate_blocked_before_stage": gate_status.get("blocked_before_stage", None),
                "upper_bound_95": float(cert.get("upper_bound", 1.0)) if isinstance(cert, dict) and cert else 1.0,
                "observed_violation_rate": float(cert.get("observed_violation_rate", 1.0))
                if isinstance(cert, dict) and cert
                else 1.0,
                "observed_violations": int(cert.get("observed_violations", 0)) if isinstance(cert, dict) and cert else 0,
                "episodes": int(cert.get("episodes", 0)) if isinstance(cert, dict) and cert else 0,
                "elapsed_sec": float(elapsed),
                "tutorial_run_id": str(report.get("tutorial_run_id", "")),
                "graduation_run_id": str(report.get("graduation_run_id", "")),
            }
        )
        print(
            f"[{idx}/{len(combos)}] margin={margin:.3f} edge={edge:.3f} target={target_margin} gate={gate_thr:.3f} "
            f"stages={max_stage} upper={rows[-1]['upper_bound_95']:.4f} blocked={rows[-1]['stage_gate_blocked']}"
        )

    ranked = sorted(rows, key=_rank_key)
    summary = {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": {
            "target_verse": str(args.target_verse),
            "tutorial_verse": str(args.tutorial_verse),
            "algo": str(args.algo),
            "seed": int(args.seed),
            "runs_root": str(args.runs_root),
            "num_trials": int(len(rows)),
        },
        "grids": {
            "tutorial_margin_reward_scale_grid": margin_grid,
            "tutorial_edge_penalty_grid": edge_grid,
            "tutorial_target_margin_grid": target_grid,
            "tutorial_gust_probability_grid": gust_grid,
            "stage_gate_max_violation_rate_grid": gate_grid,
        },
        "best": ranked[0] if ranked else None,
        "ranked_results": ranked,
    }

    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"sweep_complete trials={len(rows)} out={args.out_json}")
    if ranked:
        best = ranked[0]
        print(
            "best "
            f"margin={best['tutorial_margin_reward_scale']:.3f} "
            f"edge={best['tutorial_edge_penalty']:.3f} "
            f"target={best['tutorial_target_margin']} "
            f"gate={best['stage_gate_max_violation_rate']:.3f} "
            f"upper={best['upper_bound_95']:.4f} "
            f"blocked={best['stage_gate_blocked']}"
        )


if __name__ == "__main__":
    main()
