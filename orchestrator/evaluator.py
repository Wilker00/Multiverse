"""
orchestrator/evaluator.py

Minimal evaluator that reads events.jsonl from a run directory and computes:
- per-episode return
- per-episode length
- success rate (if "reached_goal" exists in info)
- simple aggregate stats

This is intentionally lightweight and works with the StepEvent schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EpisodeStats:
    episode_id: str
    steps: int
    return_sum: float
    reached_goal: Optional[bool] = None


@dataclass
class RunStats:
    run_id: str
    episodes: int
    total_steps: int
    mean_return: float
    mean_steps: float
    success_rate: Optional[float]
    episode_stats: List[EpisodeStats]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _load_events(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def evaluate_run(run_dir: str, filename: str = "events.jsonl") -> RunStats:
    """
    Evaluate a run directory that contains events.jsonl.

    Expected layout:
      runs/<run_id>/events.jsonl
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_id = os.path.basename(os.path.normpath(run_dir))
    events_path = os.path.join(run_dir, filename)

    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")

    events = _load_events(events_path)

    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ep = str(e.get("episode_id"))
        by_ep.setdefault(ep, []).append(e)

    episode_stats: List[EpisodeStats] = []
    successes: List[bool] = []

    for ep_id, ep_events in by_ep.items():
        # Sort by step index in case logs are out of order
        ep_events.sort(key=lambda x: int(x.get("step_idx", 0)))

        ret = 0.0
        steps = 0
        reached_goal: Optional[bool] = None

        for ev in ep_events:
            ret += _safe_float(ev.get("reward", 0.0))
            steps += 1

            info = ev.get("info") or {}
            if isinstance(info, dict) and "reached_goal" in info:
                # If any step marks reached_goal True, count as success
                rg = info.get("reached_goal")
                if isinstance(rg, bool) and rg:
                    reached_goal = True

        if reached_goal is None:
            # If reached_goal key exists but never True, mark as False
            # This avoids None if the key exists but is always False
            for ev in ep_events:
                info = ev.get("info") or {}
                if isinstance(info, dict) and "reached_goal" in info:
                    reached_goal = False
                    break

        if reached_goal is not None:
            successes.append(bool(reached_goal))

        episode_stats.append(
            EpisodeStats(
                episode_id=ep_id,
                steps=steps,
                return_sum=ret,
                reached_goal=reached_goal,
            )
        )

    episodes = len(episode_stats)
    total_steps = sum(es.steps for es in episode_stats)

    mean_return = (sum(es.return_sum for es in episode_stats) / episodes) if episodes else 0.0
    mean_steps = (sum(es.steps for es in episode_stats) / episodes) if episodes else 0.0

    success_rate: Optional[float] = None
    if successes:
        success_rate = sum(1 for s in successes if s) / len(successes)

    return RunStats(
        run_id=run_id,
        episodes=episodes,
        total_steps=total_steps,
        mean_return=mean_return,
        mean_steps=mean_steps,
        success_rate=success_rate,
        episode_stats=sorted(episode_stats, key=lambda x: x.episode_id),
    )


def print_report(stats: RunStats) -> None:
    print("Evaluation report")
    print(f"run_id       : {stats.run_id}")
    print(f"episodes     : {stats.episodes}")
    print(f"total_steps  : {stats.total_steps}")
    print(f"mean_return  : {stats.mean_return:.3f}")
    print(f"mean_steps   : {stats.mean_steps:.2f}")
    if stats.success_rate is not None:
        print(f"success_rate : {stats.success_rate * 100:.1f}%")
    else:
        print("success_rate : n/a")

    print("")
    print("Per-episode:")
    for es in stats.episode_stats:
        rg = "n/a" if es.reached_goal is None else ("yes" if es.reached_goal else "no")
        print(f"  {es.episode_id}  steps={es.steps:3d}  return={es.return_sum:8.3f}  reached_goal={rg}")


if __name__ == "__main__":
    # Example:
    # python orchestrator/evaluator.py runs/<run_id>
    import sys

    if len(sys.argv) < 2:
        print("Usage: python orchestrator/evaluator.py runs/<run_id>")
        raise SystemExit(2)

    run_dir = sys.argv[1]
    stats = evaluate_run(run_dir)
    print_report(stats)
