"""
tools/make_apprentice.py

Build an apprentice dataset from one run.

Pipeline:
1) Build episode index (episodes.jsonl) from events.jsonl
2) Retrieve episodes using simple filters (prefer successes if available)
3) Build apprentice_dataset.jsonl (obs -> action pairs, plus optional next_obs/reward/done)

Usage:
  python tools/make_apprentice.py runs/<run_id>

Common:
  python tools/make_apprentice.py runs/<run_id> --select 20
  python tools/make_apprentice.py runs/<run_id> --only_success
  python tools/make_apprentice.py runs/<run_id> --min_return 0.5 --select 50
  python tools/make_apprentice.py runs/<run_id> --verse line_world --policy explorer_random

Outputs (inside run_dir):
- episodes.jsonl
- apprentice_dataset.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.episode_index import EpisodeIndexConfig, build_episode_index
from memory.retrieval import RetrievalConfig, EpisodeFilter, filter_episodes
from memory.apprentice_dataset import ApprenticeDatasetConfig, build_apprentice_dataset


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _pick_episode_ids(
    summaries: List[Dict[str, Any]],
    select: int,
    prefer_success: bool = True,
) -> List[str]:
    """
    Choose episodes to convert into an apprentice dataset.

    Strategy:
    - if prefer_success: pick reached_goal True episodes first (if any)
    - else pick by return_sum descending
    """
    if not summaries:
        return []

    if prefer_success:
        succ = [s for s in summaries if s.get("reached_goal") is True]
        if succ:
            succ.sort(key=lambda s: _safe_float(s.get("return_sum", 0.0)), reverse=True)
            return [str(s["episode_id"]) for s in succ[:select]]

    ranked = sorted(summaries, key=lambda s: _safe_float(s.get("return_sum", 0.0)), reverse=True)
    return [str(s["episode_id"]) for s in ranked[:select]]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("run_dir", type=str, help="Path like runs/<run_id>")

    ap.add_argument("--verse", type=str, default=None, help="Filter episodes by verse_name")
    ap.add_argument("--policy", type=str, default=None, help="Filter episodes by policy_id")

    ap.add_argument("--select", type=int, default=20, help="Number of episodes to include")
    ap.add_argument("--only_success", action="store_true", help="Only use reached_goal True episodes")
    ap.add_argument("--prefer_success", action="store_true", help="Prefer reached_goal episodes if available")

    ap.add_argument("--min_return", type=float, default=None)
    ap.add_argument("--max_return", type=float, default=None)
    ap.add_argument("--min_steps", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)

    ap.add_argument("--tag_in", action="append", default=None, help="Require tag (repeatable)")
    ap.add_argument("--tag_out", action="append", default=None, help="Exclude tag (repeatable)")

    ap.add_argument("--output", type=str, default="apprentice_dataset.jsonl")

    ap.add_argument("--include_next_obs", action="store_true", help="Include next_obs column")
    ap.add_argument("--include_reward", action="store_true", help="Include reward column")
    ap.add_argument("--include_done", action="store_true", help="Include done column")

    args = ap.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    # 1) Build episode index
    idx_path = build_episode_index(EpisodeIndexConfig(run_dir=run_dir))
    print(f"Built episode index: {idx_path}")

    # 2) Retrieve episodes
    r_cfg = RetrievalConfig(run_dir=run_dir)

    flt = EpisodeFilter(
        verse_name=args.verse,
        policy_id=args.policy,
        tag_includes=args.tag_in if args.tag_in else None,
        tag_excludes=args.tag_out if args.tag_out else None,
        reached_goal=True if args.only_success else None,
        min_return=args.min_return,
        max_return=args.max_return,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        limit=None,  # we pick after ranking
    )

    summaries = filter_episodes(r_cfg, flt)

    if not summaries:
        print("No episodes matched filters.")
        return

    prefer_success = args.prefer_success or args.only_success
    chosen_ids = _pick_episode_ids(
        summaries=summaries,
        select=max(1, int(args.select)),
        prefer_success=prefer_success,
    )

    if args.only_success:
        chosen_ids = [eid for eid in chosen_ids if any(s.get("episode_id") == eid and s.get("reached_goal") is True for s in summaries)]

    if not chosen_ids:
        print("No episodes selected after ranking.")
        return

    print(f"Selected {len(chosen_ids)} episode(s)")

    # 3) Build apprentice dataset
    ds_cfg = ApprenticeDatasetConfig(
        run_dir=run_dir,
        output_filename=args.output,
        include_next_obs=bool(args.include_next_obs),
        include_reward=bool(args.include_reward),
        include_done=bool(args.include_done),
    )

    out_path = build_apprentice_dataset(cfg=ds_cfg, episode_ids=chosen_ids)
    print(f"Apprentice dataset written: {out_path}")
    print("Next: run imitation agent with:")
    print(f"  python tools/train_agent.py --algo imitation_lookup --verse {args.verse or '<your_verse>'} --episodes 10 --max_steps 40 --dataset {out_path}")


if __name__ == "__main__":
    main()
