"""
tools/compare_runs.py

Compare multiple runs under a root directory (default: runs/).

It uses orchestrator/evaluator.py to compute stats per run, then prints:
- leaderboard by mean_return
- success_rate if available
- mean_steps
- run id and optional hints from events

Usage:
  python tools/compare_runs.py
  python tools/compare_runs.py --runs_root runs --top 10
  python tools/compare_runs.py --filter_verse line_world
  python tools/compare_runs.py --filter_policy ppo
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from orchestrator.evaluator import evaluate_run, RunStats


@dataclass
class RunRow:
    run_id: str
    verse_name: str
    policy_id: str
    mean_return: float
    success_rate: Optional[float]
    mean_steps: float
    episodes: int


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _peek_first_event(run_dir: str, filename: str = "events.jsonl") -> Tuple[str, str]:
    """
    Reads the first non-empty line of events.jsonl and extracts verse_name and policy_id.
    Returns ("unknown","unknown") if anything fails.
    """
    path = os.path.join(run_dir, filename)
    if not os.path.isfile(path):
        return ("unknown", "unknown")

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                verse = _safe_str(ev.get("verse_name"), "unknown")
                policy = _safe_str(ev.get("policy_id"), "unknown")
                return (verse, policy)
    except Exception:
        return ("unknown", "unknown")

    return ("unknown", "unknown")


def _list_run_dirs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[str] = []
    for name in os.listdir(runs_root):
        p = os.path.join(runs_root, name)
        if os.path.isdir(p):
            # run dir must contain events.jsonl
            if os.path.isfile(os.path.join(p, "events.jsonl")):
                out.append(p)
    out.sort()
    return out


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--sort", type=str, default="mean_return", choices=["mean_return", "success_rate", "mean_steps"])
    ap.add_argument("--filter_verse", type=str, default=None)
    ap.add_argument("--filter_policy", type=str, default=None)
    args = ap.parse_args()

    run_dirs = _list_run_dirs(args.runs_root)
    if not run_dirs:
        print(f"No run directories found under: {args.runs_root}")
        return

    rows: List[RunRow] = []

    for rd in run_dirs:
        run_id = os.path.basename(os.path.normpath(rd))
        verse_name, policy_id = _peek_first_event(rd)

        if args.filter_verse and verse_name != args.filter_verse:
            continue
        if args.filter_policy and policy_id != args.filter_policy:
            continue

        try:
            stats: RunStats = evaluate_run(rd)
        except Exception as e:
            print(f"Skipping {run_id} due to error: {e}")
            continue

        rows.append(
            RunRow(
                run_id=run_id,
                verse_name=verse_name,
                policy_id=policy_id,
                mean_return=float(stats.mean_return),
                success_rate=stats.success_rate,
                mean_steps=float(stats.mean_steps),
                episodes=int(stats.episodes),
            )
        )

    if not rows:
        print("No runs matched filters.")
        return

    # Sorting
    if args.sort == "mean_return":
        rows.sort(key=lambda r: r.mean_return, reverse=True)
    elif args.sort == "success_rate":
        rows.sort(key=lambda r: (-1.0 if r.success_rate is None else r.success_rate), reverse=True)
    else:
        rows.sort(key=lambda r: r.mean_steps)

    top = rows[: max(1, int(args.top))]

    print("")
    print("Run leaderboard")
    print(f"root: {args.runs_root}")
    if args.filter_verse:
        print(f"filter verse : {args.filter_verse}")
    if args.filter_policy:
        print(f"filter policy: {args.filter_policy}")
    print("")

    header = f"{'rank':>4}  {'run_id':<20}  {'verse':<12}  {'policy':<16}  {'mean_return':>11}  {'success':>8}  {'mean_steps':>10}  {'eps':>4}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(top, start=1):
        print(
            f"{i:>4}  {r.run_id:<20}  {r.verse_name:<12}  {r.policy_id:<16}  "
            f"{r.mean_return:>11.3f}  {_format_pct(r.success_rate):>8}  {r.mean_steps:>10.2f}  {r.episodes:>4}"
        )

    print("")
    print("Tip: run full report on one run with:")
    print("  python orchestrator/evaluator.py runs/<run_id>")


if __name__ == "__main__":
    main()




