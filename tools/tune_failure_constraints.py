"""
tools/tune_failure_constraints.py

Grid-search tuner for cliff failure-aware constraints:
- death_similarity_threshold (Behavioral Surgeon pruning aggressiveness)
- danger_temperature (FailureAware soft mask strength)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from orchestrator.evaluator import evaluate_run


def _parse_list_floats(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


@dataclass
class TrialResult:
    death_similarity: float
    danger_temperature: float
    run_id: str
    mean_return: float
    success_rate: float
    mean_steps: float
    utility: float
    feasible: bool


def _new_run_id(before: set[str], runs_root: str) -> str:
    after = set(os.listdir(runs_root)) if os.path.isdir(runs_root) else set()
    new_runs = sorted(list(after - before))
    if not new_runs:
        raise RuntimeError(f"Could not detect new run under {runs_root}")
    return str(new_runs[-1])


def _run_behavioral_surgeon(*, death_similarity: float, cliff_lookback_runs: int, behavior_top_percent: float) -> None:
    cmd = [
        sys.executable,
        os.path.join("tools", "behavioral_surgeon.py"),
        "--cliff_lookback_runs",
        str(max(1, int(cliff_lookback_runs))),
        "--behavior_top_percent",
        str(float(behavior_top_percent)),
        "--death_similarity_threshold",
        str(float(death_similarity)),
    ]
    subprocess.run(cmd, check=True)


def _eval_failure_aware(
    *,
    runs_root: str,
    episodes: int,
    max_steps: int,
    seed: int,
    good_dataset: str,
    bad_dataset: str,
    danger_temperature: float,
    hard_block_threshold: float,
    caution_penalty_scale: float,
    danger_prior: float,
) -> Tuple[str, float, float, float]:
    before = set(os.listdir(runs_root)) if os.path.isdir(runs_root) else set()
    cmd = [
        sys.executable,
        os.path.join("tools", "train_agent.py"),
        "--algo",
        "failure_aware",
        "--verse",
        "cliff_world",
        "--episodes",
        str(int(episodes)),
        "--max_steps",
        str(int(max_steps)),
        "--seed",
        str(int(seed)),
        "--dataset",
        str(good_dataset),
        "--bad_dna",
        str(bad_dataset),
        "--aconfig",
        f"danger_temperature={float(danger_temperature)}",
        "--aconfig",
        f"hard_block_threshold={float(hard_block_threshold)}",
        "--aconfig",
        f"caution_penalty_scale={float(caution_penalty_scale)}",
        "--aconfig",
        f"danger_prior={float(danger_prior)}",
        "--eval",
    ]
    subprocess.run(cmd, check=True)
    run_id = _new_run_id(before, runs_root)
    stats = evaluate_run(os.path.join(runs_root, run_id))
    return run_id, float(stats.mean_return), float(stats.success_rate or 0.0), float(stats.mean_steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--good_dataset", type=str, default=os.path.join("models", "expert_datasets", "cliff_world.jsonl"))
    ap.add_argument(
        "--bad_dataset",
        type=str,
        default=os.path.join("models", "expert_datasets", "cliff_death_transitions.jsonl"),
    )
    ap.add_argument("--death_similarities", type=str, default="0.85,0.9,0.95")
    ap.add_argument("--danger_temperatures", type=str, default="1.0,1.25,1.5,1.75,2.0")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cliff_lookback_runs", type=int, default=10)
    ap.add_argument("--behavior_top_percent", type=float, default=20.0)
    ap.add_argument("--hard_block_threshold", type=float, default=0.97)
    ap.add_argument("--caution_penalty_scale", type=float, default=0.7)
    ap.add_argument("--danger_prior", type=float, default=1.0)
    ap.add_argument("--min_success_rate", type=float, default=0.35)
    ap.add_argument("--min_mean_return", type=float, default=-250.0)
    ap.add_argument("--success_weight", type=float, default=250.0, help="Utility = mean_return + success_weight*success_rate")
    ap.add_argument("--out_json", type=str, default=os.path.join("models", "tuning", "failure_constraints.json"))
    args = ap.parse_args()

    sims = _parse_list_floats(args.death_similarities)
    temps = _parse_list_floats(args.danger_temperatures)
    results: List[TrialResult] = []

    for sim in sims:
        _run_behavioral_surgeon(
            death_similarity=float(sim),
            cliff_lookback_runs=int(args.cliff_lookback_runs),
            behavior_top_percent=float(args.behavior_top_percent),
        )
        if not os.path.isfile(args.bad_dataset):
            raise FileNotFoundError(f"Bad dataset missing after surgeon run: {args.bad_dataset}")
        for temp in temps:
            run_id, mean_return, success_rate, mean_steps = _eval_failure_aware(
                runs_root=args.runs_root,
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                seed=int(args.seed),
                good_dataset=args.good_dataset,
                bad_dataset=args.bad_dataset,
                danger_temperature=float(temp),
                hard_block_threshold=float(args.hard_block_threshold),
                caution_penalty_scale=float(args.caution_penalty_scale),
                danger_prior=float(args.danger_prior),
            )
            feasible = (success_rate >= float(args.min_success_rate)) and (mean_return >= float(args.min_mean_return))
            utility = float(mean_return) + (float(args.success_weight) * float(success_rate))
            results.append(
                TrialResult(
                    death_similarity=float(sim),
                    danger_temperature=float(temp),
                    run_id=run_id,
                    mean_return=float(mean_return),
                    success_rate=float(success_rate),
                    mean_steps=float(mean_steps),
                    utility=float(utility),
                    feasible=bool(feasible),
                )
            )
            print(
                f"trial sim={sim:.3f} temp={temp:.3f} run={run_id} "
                f"ret={mean_return:.3f} succ={success_rate:.3f} steps={mean_steps:.2f} feasible={feasible}"
            )

    feasible = [r for r in results if r.feasible]
    pool = feasible if feasible else results
    if not pool:
        raise RuntimeError("No tuning trials completed.")
    best = max(pool, key=lambda r: r.utility)

    payload: Dict[str, Any] = {
        "settings": {
            "death_similarities": sims,
            "danger_temperatures": temps,
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "hard_block_threshold": float(args.hard_block_threshold),
            "caution_penalty_scale": float(args.caution_penalty_scale),
            "danger_prior": float(args.danger_prior),
            "min_success_rate": float(args.min_success_rate),
            "min_mean_return": float(args.min_mean_return),
            "success_weight": float(args.success_weight),
        },
        "best": {
            "death_similarity": float(best.death_similarity),
            "danger_temperature": float(best.danger_temperature),
            "run_id": best.run_id,
            "mean_return": float(best.mean_return),
            "success_rate": float(best.success_rate),
            "mean_steps": float(best.mean_steps),
            "utility": float(best.utility),
            "feasible": bool(best.feasible),
        },
        "trials": [
            {
                "death_similarity": float(r.death_similarity),
                "danger_temperature": float(r.danger_temperature),
                "run_id": str(r.run_id),
                "mean_return": float(r.mean_return),
                "success_rate": float(r.success_rate),
                "mean_steps": float(r.mean_steps),
                "utility": float(r.utility),
                "feasible": bool(r.feasible),
            }
            for r in results
        ],
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print("Tuning complete")
    print(f"best_death_similarity  : {best.death_similarity:.3f}")
    print(f"best_danger_temperature: {best.danger_temperature:.3f}")
    print(f"best_run_id            : {best.run_id}")
    print(f"best_mean_return       : {best.mean_return:.3f}")
    print(f"best_success_rate      : {best.success_rate:.3f}")
    print(f"results_json           : {args.out_json}")


if __name__ == "__main__":
    main()

