"""
tools/build_labyrinth_expert.py

Build a high-quality expert dataset for labyrinth_world by planning in the
actual environment dynamics (including moving lasers and battery).
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import JSONValue, VerseSpec
from verses.registry import create_verse, register_builtin


def _state_key(state: Dict[str, JSONValue]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _goal_from_reset_info(reset_info: Dict[str, JSONValue]) -> Tuple[int, int]:
    return int(reset_info.get("goal_x", 0)), int(reset_info.get("goal_y", 0))


def _manhattan(x: int, y: int, gx: int, gy: int) -> int:
    return abs(x - gx) + abs(y - gy)


def find_plan(
    *,
    seed: int,
    max_expansions: int,
    verse_params: Dict[str, JSONValue],
) -> List[int]:
    """
    Finds a safe action sequence to goal using A* on environment states.
    """
    register_builtin()
    spec = VerseSpec(
        spec_version="v1",
        verse_name="labyrinth_world",
        verse_version="0.1",
        seed=int(seed),
        tags=["planner"],
        params=dict(verse_params),
    )
    verse = create_verse(spec)
    verse.seed(seed)
    reset = verse.reset()
    goal_x, goal_y = _goal_from_reset_info(reset.info)

    start_state = verse.export_state()  # type: ignore[attr-defined]
    start_key = _state_key(start_state)
    start_obs = reset.obs if isinstance(reset.obs, dict) else {}
    start_h = _manhattan(int(start_obs.get("x", 0)), int(start_obs.get("y", 0)), goal_x, goal_y)

    frontier: List[Tuple[int, int, str]] = []
    heapq.heappush(frontier, (int(start_h), 0, start_key))

    state_by_key: Dict[str, Dict[str, JSONValue]] = {start_key: start_state}
    g_best: Dict[str, int] = {start_key: 0}
    parent: Dict[str, Tuple[str, int]] = {}

    expanded = 0
    solved_key: Optional[str] = None
    solved_done_info: Optional[Dict[str, Any]] = None

    while frontier and expanded < int(max_expansions):
        _, g, cur_key = heapq.heappop(frontier)
        if g > g_best.get(cur_key, 10**9):
            continue
        cur_state = state_by_key[cur_key]

        for action in range(5):
            verse.import_state(cur_state)  # type: ignore[attr-defined]
            step = verse.step(action)
            nxt_state = verse.export_state()  # type: ignore[attr-defined]
            nxt_key = _state_key(nxt_state)
            expanded += 1

            reached_goal = isinstance(step.info, dict) and bool(step.info.get("reached_goal", False))
            is_failure_terminal = bool(step.done) and not reached_goal
            if is_failure_terminal:
                continue
            if bool(step.truncated):
                continue

            ng = g + 1
            if ng >= g_best.get(nxt_key, 10**9):
                continue

            g_best[nxt_key] = ng
            state_by_key[nxt_key] = nxt_state
            parent[nxt_key] = (cur_key, int(action))

            x = int(step.info.get("x", 0)) if isinstance(step.info, dict) else 0
            y = int(step.info.get("y", 0)) if isinstance(step.info, dict) else 0
            h = _manhattan(x, y, goal_x, goal_y)
            heapq.heappush(frontier, (ng + h, ng, nxt_key))

            if reached_goal:
                solved_key = nxt_key
                solved_done_info = dict(step.info or {})
                frontier = []
                break

    verse.close()

    if solved_key is None:
        raise RuntimeError(f"No plan found within {max_expansions} expansions.")

    actions_rev: List[int] = []
    cur = solved_key
    while cur != start_key:
        prev, action = parent[cur]
        actions_rev.append(int(action))
        cur = prev
    actions_rev.reverse()

    if not actions_rev:
        raise RuntimeError("Planner returned empty plan.")

    if solved_done_info is not None and not bool(solved_done_info.get("reached_goal", False)):
        raise RuntimeError("Planner ended without reached_goal signal.")

    return actions_rev


def build_dataset(
    *,
    out_path: str,
    actions: List[int],
    episodes: int,
    seed: int,
    verse_params: Dict[str, JSONValue],
) -> Dict[str, Any]:
    register_builtin()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rows = 0
    successes = 0
    failures = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ep in range(int(episodes)):
            ep_seed = int(seed) + ep
            spec = VerseSpec(
                spec_version="v1",
                verse_name="labyrinth_world",
                verse_version="0.1",
                seed=ep_seed,
                tags=["planner_dataset"],
                params=dict(verse_params),
            )
            verse = create_verse(spec)
            verse.seed(ep_seed)
            reset = verse.reset()
            obs = reset.obs
            ep_ok = False

            for action in actions:
                row = {
                    "obs": obs,
                    "action": int(action),
                    "source": "labyrinth_planner",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1

                step = verse.step(int(action))
                obs = step.obs
                if bool(step.done) or bool(step.truncated):
                    if isinstance(step.info, dict) and bool(step.info.get("reached_goal", False)):
                        ep_ok = True
                    break

            if ep_ok:
                successes += 1
            else:
                failures += 1
            verse.close()

    return {
        "rows": int(rows),
        "episodes": int(episodes),
        "successes": int(successes),
        "failures": int(failures),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="models/expert_datasets/labyrinth_world.jsonl")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_steps", type=int, default=180)
    ap.add_argument("--width", type=int, default=15)
    ap.add_argument("--height", type=int, default=11)
    ap.add_argument("--battery_capacity", type=int, default=80)
    ap.add_argument("--action_noise", type=float, default=0.0, help="Use 0.0 for deterministic expert extraction.")
    ap.add_argument("--max_expansions", type=int, default=250000)
    args = ap.parse_args()

    verse_params: Dict[str, JSONValue] = {
        "adr_enabled": False,
        "width": int(args.width),
        "height": int(args.height),
        "max_steps": int(args.max_steps),
        "battery_capacity": int(args.battery_capacity),
        "action_noise": float(args.action_noise),
    }

    actions = find_plan(
        seed=int(args.seed),
        max_expansions=int(args.max_expansions),
        verse_params=verse_params,
    )
    stats = build_dataset(
        out_path=args.out,
        actions=actions,
        episodes=int(args.episodes),
        seed=int(args.seed),
        verse_params=verse_params,
    )

    print("Labyrinth expert dataset built")
    print(f"out         : {args.out}")
    print(f"plan_len    : {len(actions)}")
    print(f"rows        : {stats['rows']}")
    print(f"episodes    : {stats['episodes']}")
    print(f"successes   : {stats['successes']}")
    print(f"failures    : {stats['failures']}")


if __name__ == "__main__":
    main()

