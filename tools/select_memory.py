"""
tools/select_memory.py

Prune and select useful memories from events.jsonl.
"""

from __future__ import annotations

import argparse

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.selection import SelectionConfig, prune_events_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--min_reward", type=float, default=-1e9)
    ap.add_argument("--max_reward", type=float, default=1e9)
    ap.add_argument("--keep_successes", action="store_true")
    ap.add_argument("--success_key", type=str, default="reached_goal")
    ap.add_argument("--top_k_per_episode", type=int, default=0)
    ap.add_argument("--top_k_episodes", type=int, default=0)
    ap.add_argument("--novelty_bonus", type=float, default=0.0)
    ap.add_argument("--creative_failure_bonus", type=float, default=0.0)
    ap.add_argument("--creative_failure_min_return", type=float, default=0.0)
    ap.add_argument("--creative_failure_max_steps", type=int, default=0)
    ap.add_argument("--recency_half_life_ms", type=int, default=0)
    ap.add_argument("--input", type=str, default="events.jsonl")
    ap.add_argument("--output", type=str, default="events.pruned.jsonl")
    args = ap.parse_args()

    cfg = SelectionConfig(
        min_reward=args.min_reward,
        max_reward=args.max_reward,
        keep_successes=args.keep_successes,
        success_key=args.success_key,
        keep_top_k_per_episode=args.top_k_per_episode,
        keep_top_k_episodes=args.top_k_episodes,
        novelty_bonus=args.novelty_bonus,
        creative_failure_bonus=args.creative_failure_bonus,
        creative_failure_min_return=args.creative_failure_min_return,
        creative_failure_max_steps=args.creative_failure_max_steps,
        recency_half_life_ms=args.recency_half_life_ms,
    )

    out = prune_events_jsonl(
        run_dir=args.run_dir,
        input_filename=args.input,
        output_filename=args.output,
        cfg=cfg,
    )
    print(f"Pruned events written to: {out}")


if __name__ == "__main__":
    main()




