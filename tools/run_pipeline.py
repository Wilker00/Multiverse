"""
tools/run_pipeline.py

One-command pipeline demo for u.ai:

1) Run an explorer (random for now) to generate experience
2) Build episode index
3) Retrieve a subset of episodes (successes if available, otherwise top return)
4) Build apprentice dataset from those episodes
5) Run an imitation agent using that dataset

This proves the Off World -> Memory -> Native transfer loop in a single script.

Usage:
  python tools/run_pipeline.py

Optional:
  python tools/run_pipeline.py --episodes 30 --goal 8 --max_steps 40

Notes:
- Keeps everything minimal and explicit.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict, Any

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec, AgentSpec
from orchestrator.trainer import Trainer

from memory.episode_index import EpisodeIndexConfig, build_episode_index
from memory.retrieval import RetrievalConfig, EpisodeFilter, filter_episodes
from memory.apprentice_dataset import ApprenticeDatasetConfig, build_apprentice_dataset


def _pick_episode_ids(episode_summaries: List[Dict[str, Any]], limit: int) -> List[str]:
    # Prefer successes first, else sort by return_sum desc
    successes = [e for e in episode_summaries if e.get("reached_goal") is True]
    if successes:
        successes = sorted(successes, key=lambda x: float(x.get("return_sum", 0.0)), reverse=True)
        return [str(e["episode_id"]) for e in successes[:limit]]

    # No success signal, pick top returns
    ranked = sorted(episode_summaries, key=lambda x: float(x.get("return_sum", 0.0)), reverse=True)
    return [str(e["episode_id"]) for e in ranked[:limit]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--goal", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--select", type=int, default=10, help="how many episodes to keep for apprenticeship")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    trainer = Trainer(run_root="runs", schema_version="v1", auto_register_builtin=True)

    # -------------------------
    # 1) Explorer run
    # -------------------------
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name="line_world",
        verse_version="0.1",
        seed=args.seed,
        tags=["pipeline"],
        params={
            "goal_pos": args.goal,
            "max_steps": args.max_steps,
            "step_penalty": -0.02,
        },
    )

    explorer_spec = AgentSpec(
        spec_version="v1",
        policy_id="explorer_random",
        policy_version="0.0",
        algo="random",
        seed=args.seed,
    )

    # Trainer prints run_id, but we need it programmatically.
    # Quick workaround: we infer "latest run dir" after the run.
    before = set(os.listdir("runs")) if os.path.isdir("runs") else set()

    trainer.run(
        verse_spec=verse_spec,
        agent_spec=explorer_spec,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    after = set(os.listdir("runs"))
    new_runs = sorted(list(after - before))
    if not new_runs:
        raise RuntimeError("Could not detect new run directory under runs/")
    run_id = new_runs[-1]
    run_dir = os.path.join("runs", run_id)

    print("")
    print(f"Pipeline detected run_dir: {run_dir}")

    # -------------------------
    # 2) Build episode index
    # -------------------------
    idx_path = build_episode_index(EpisodeIndexConfig(run_dir=run_dir))
    print(f"Built episode index: {idx_path}")

    # -------------------------
    # 3) Retrieve episodes
    # -------------------------
    r_cfg = RetrievalConfig(run_dir=run_dir)

    # Try for successes, else just grab top returns
    summaries = filter_episodes(
        r_cfg,
        EpisodeFilter(
            verse_name="line_world",
            reached_goal=True,
            limit=args.select,
        ),
    )
    if not summaries:
        summaries = filter_episodes(
            r_cfg,
            EpisodeFilter(
                verse_name="line_world",
                limit=max(args.select, 20),
            ),
        )

    chosen_ids = _pick_episode_ids(summaries, limit=args.select)
    print(f"Selected {len(chosen_ids)} episode(s) for apprenticeship")

    # -------------------------
    # 4) Build apprentice dataset
    # -------------------------
    dataset_path = build_apprentice_dataset(
        cfg=ApprenticeDatasetConfig(run_dir=run_dir),
        episode_ids=chosen_ids,
    )
    print(f"Built apprentice dataset: {dataset_path}")

    # -------------------------
    # 5) Run imitation agent on same verse
    # -------------------------
    imit_spec = AgentSpec(
        spec_version="v1",
        policy_id="native_imitation",
        policy_version="0.0",
        algo="imitation_lookup",
        seed=args.seed,
        config={"dataset_path": dataset_path},
    )

    trainer.run(
        verse_spec=verse_spec,
        agent_spec=imit_spec,
        episodes=5,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    print("")
    print("Pipeline complete")
    print("Next move: replace Random explorer with PPO and imitation_lookup with a small neural policy.")


if __name__ == "__main__":
    main()



