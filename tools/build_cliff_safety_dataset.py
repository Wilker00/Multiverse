"""
tools/build_cliff_safety_dataset.py

Generate a synthetic safety-first expert dataset for cliff_world under stochastic settings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


def _safe_zig_action(*, x: int, y: int, goal_x: int, goal_y: int) -> int:
    # Keep two rows away from cliff while progressing right.
    if y > 1:
        return 0  # up
    if x < goal_x:
        return 3  # right
    if y < goal_y:
        return 1  # down
    return 3


def _top_row_action(*, x: int, y: int, goal_x: int, goal_y: int) -> int:
    # Very conservative: prioritize staying at top row.
    if y > 0:
        return 0
    if x < goal_x:
        return 3
    if y < goal_y:
        return 1
    return 3


def _build_spec(*, seed: int, max_steps: int, wind_probability: float, crumble_probability: float) -> VerseSpec:
    return VerseSpec(
        spec_version="v1",
        verse_name="cliff_world",
        verse_version="0.1",
        seed=int(seed),
        tags=["synthetic", "expert", "cliff", "safety_first"],
        params={
            "max_steps": int(max_steps),
            "width": 12,
            "height": 4,
            "step_penalty": -1.0,
            "cliff_penalty": -100.0,
            "end_on_cliff": False,
            "wind_probability": float(wind_probability),
            "crumble_probability": float(crumble_probability),
            "adr_enabled": False,
        },
    )


def _as_obs(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return dict(v)
    return {}


def _collect_rows_for_regime(
    *,
    wind_probability: float,
    crumble_probability: float,
    episodes: int,
    max_steps: int,
    seed: int,
    conservative_ratio: float,
) -> List[Dict[str, Any]]:
    spec = _build_spec(
        seed=int(seed),
        max_steps=int(max_steps),
        wind_probability=float(wind_probability),
        crumble_probability=float(crumble_probability),
    )
    verse = create_verse(spec)
    rng = random.Random(int(seed))
    rows: List[Dict[str, Any]] = []
    try:
        for ep in range(max(1, int(episodes))):
            verse.seed(int(seed + ep))
            rr = verse.reset()
            obs = _as_obs(rr.obs)
            done = False
            truncated = False
            step_idx = 0
            while not done and not truncated and step_idx < int(max_steps):
                x = int(obs.get("x", 0))
                y = int(obs.get("y", 0))
                goal_x = int(obs.get("goal_x", 11))
                goal_y = int(obs.get("goal_y", 3))
                if rng.random() < float(conservative_ratio):
                    action = _top_row_action(x=x, y=y, goal_x=goal_x, goal_y=goal_y)
                else:
                    action = _safe_zig_action(x=x, y=y, goal_x=goal_x, goal_y=goal_y)
                st = verse.step(action)
                next_obs = _as_obs(st.obs)
                rows.append(
                    {
                        "obs": obs,
                        "action": int(action),
                        "reward": float(st.reward),
                        "next_obs": next_obs,
                        "done": bool(st.done),
                        "truncated": bool(st.truncated),
                        "info": st.info if isinstance(st.info, dict) else {},
                        "episode_id": f"{wind_probability:.2f}_{crumble_probability:.2f}_{ep}",
                        "step_idx": int(step_idx),
                    }
                )
                obs = next_obs
                done = bool(st.done)
                truncated = bool(st.truncated)
                step_idx += 1
    finally:
        verse.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build synthetic cliff safety-first expert dataset.")
    ap.add_argument("--episodes_stage2", type=int, default=400)
    ap.add_argument("--episodes_hard", type=int, default=400)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--conservative_ratio", type=float, default=0.60)
    ap.add_argument(
        "--out_path",
        type=str,
        default=os.path.join("models", "expert_datasets", "cliff_world_safetyfirst_stochastic.jsonl"),
    )
    args = ap.parse_args()

    register_builtin()
    rows: List[Dict[str, Any]] = []
    rows.extend(
        _collect_rows_for_regime(
            wind_probability=0.05,
            crumble_probability=0.01,
            episodes=int(args.episodes_stage2),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            conservative_ratio=float(args.conservative_ratio),
        )
    )
    rows.extend(
        _collect_rows_for_regime(
            wind_probability=0.10,
            crumble_probability=0.03,
            episodes=int(args.episodes_hard),
            max_steps=int(args.max_steps),
            seed=int(args.seed + 10000),
            conservative_ratio=min(1.0, float(args.conservative_ratio) + 0.15),
        )
    )
    os.makedirs(os.path.dirname(str(args.out_path)) or ".", exist_ok=True)
    with open(str(args.out_path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"dataset_rows={len(rows)} out_path={args.out_path}")


if __name__ == "__main__":
    main()
