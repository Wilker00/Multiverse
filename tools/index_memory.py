"""
tools/index_memory.py

Index and query vector memory from a run directory.
"""

from __future__ import annotations

import argparse
import json

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.vector_memory import VectorMemoryConfig, build_inmemory_index, query_memory
from memory.selection import SelectionConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--query_obs", type=str, default=None, help="JSON obs to query, e.g. '{\"pos\": 2, \"goal\": 8, \"t\": 3}'")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_reward", type=float, default=-1e9)
    ap.add_argument("--max_reward", type=float, default=1e9)
    ap.add_argument("--top_k_per_episode", type=int, default=0)
    ap.add_argument("--top_k_episodes", type=int, default=0)
    ap.add_argument("--novelty_bonus", type=float, default=0.0)
    ap.add_argument("--creative_failure_bonus", type=float, default=0.0)
    ap.add_argument("--creative_failure_min_return", type=float, default=0.0)
    ap.add_argument("--creative_failure_max_steps", type=int, default=0)
    ap.add_argument("--recency_half_life_ms", type=int, default=0)
    ap.add_argument("--encoder", type=str, default="raw")
    ap.add_argument("--encoder_model", type=str, default=None)
    args = ap.parse_args()

    selection = SelectionConfig(
        min_reward=args.min_reward,
        max_reward=args.max_reward,
        keep_top_k_per_episode=args.top_k_per_episode,
        keep_top_k_episodes=args.top_k_episodes,
        novelty_bonus=args.novelty_bonus,
        creative_failure_bonus=args.creative_failure_bonus,
        creative_failure_min_return=args.creative_failure_min_return,
        creative_failure_max_steps=args.creative_failure_max_steps,
        recency_half_life_ms=args.recency_half_life_ms,
    )
    store = build_inmemory_index(
        VectorMemoryConfig(run_dir=args.run_dir, encoder=args.encoder, encoder_model=args.encoder_model),
        selection=selection,
    )
    print("Indexed vector memory (in-memory).")

    if args.query_obs:
        obs = json.loads(args.query_obs)
        matches = query_memory(store, obs=obs, top_k=args.top_k, encoder=args.encoder, encoder_model=args.encoder_model)
        print(f"Top {len(matches)} matches:")
        for m in matches:
            print(f"{m.vector_id}  score={m.score:.3f}  meta={m.metadata}")


if __name__ == "__main__":
    main()




