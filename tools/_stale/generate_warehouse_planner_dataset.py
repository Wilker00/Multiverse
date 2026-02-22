#!/usr/bin/env python3
"""
Generate warehouse expert trajectories using an internal planner.
This creates warehouse-specific behavior DNA for unseen-feature transfer.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


Action = int
Pos = Tuple[int, int]


def _neighbors(p: Pos, w: int, h: int) -> Iterable[Tuple[Pos, Action]]:
    x, y = p
    cand = [((x, y - 1), 0), ((x, y + 1), 1), ((x - 1, y), 2), ((x + 1, y), 3)]
    for (nx, ny), a in cand:
        if 0 <= nx < w and 0 <= ny < h:
            yield (nx, ny), a


def _bfs_path(start: Pos, goal: Pos, blocked: set[Pos], w: int, h: int) -> List[Action]:
    if start == goal:
        return []
    q: deque[Pos] = deque([start])
    parent: Dict[Pos, Optional[Pos]] = {start: None}
    parent_a: Dict[Pos, Action] = {}
    while q:
        cur = q.popleft()
        for nxt, a in _neighbors(cur, w, h):
            if nxt in blocked or nxt in parent:
                continue
            parent[nxt] = cur
            parent_a[nxt] = a
            if nxt == goal:
                out: List[Action] = []
                n = nxt
                while parent[n] is not None:
                    out.append(parent_a[n])
                    n = parent[n]  # type: ignore[index]
                out.reverse()
                return out
            q.append(nxt)
    return []


def _nearest_charger(src: Pos, chargers: List[Pos], blocked: set[Pos], w: int, h: int) -> Optional[Tuple[Pos, List[Action]]]:
    best: Optional[Tuple[Pos, List[Action]]] = None
    for c in chargers:
        p = _bfs_path(src, c, blocked, w, h)
        if not p:
            continue
        if best is None or len(p) < len(best[1]):
            best = (c, p)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--out", type=str, default=os.path.join("models", "expert_datasets", "warehouse_world_real.jsonl"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--obstacle_count", type=int, default=10)
    ap.add_argument("--battery_capacity", type=int, default=28)
    ap.add_argument("--charge_rate", type=int, default=6)
    ap.add_argument("--step_penalty", type=float, default=-0.06)
    ap.add_argument("--goal_reward", type=float, default=20.0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    register_builtin()
    rows = 0
    successes = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for ep in range(int(args.episodes)):
            spec = VerseSpec(
                spec_version="v1",
                verse_name="warehouse_world",
                verse_version="0.1",
                seed=int(args.seed) + ep,
                params={
                    "adr_enabled": False,
                    "width": int(args.width),
                    "height": int(args.height),
                    "obstacle_count": int(args.obstacle_count),
                    "battery_capacity": int(args.battery_capacity),
                    "charge_rate": int(args.charge_rate),
                    "step_penalty": float(args.step_penalty),
                    "goal_reward": float(args.goal_reward),
                    "max_steps": int(args.max_steps),
                },
            )
            verse = create_verse(spec)
            rr = verse.reset()
            obs = rr.obs
            done = False
            step_idx = 0

            while not done and step_idx < int(args.max_steps):
                state = {}
                if hasattr(verse, "export_state"):
                    state = dict(verse.export_state())  # type: ignore[assignment]
                x = int(state.get("x", obs.get("x", 0) if isinstance(obs, dict) else 0))
                y = int(state.get("y", obs.get("y", 0) if isinstance(obs, dict) else 0))
                gx = int(obs.get("goal_x", 0) if isinstance(obs, dict) else 0)
                gy = int(obs.get("goal_y", 0) if isinstance(obs, dict) else 0)
                batt = int(obs.get("battery", 0) if isinstance(obs, dict) else 0)
                blocked = set((int(p[0]), int(p[1])) for p in (state.get("obstacles") or []))
                chargers = [(int(p[0]), int(p[1])) for p in (state.get("chargers") or [])]

                cur = (x, y)
                goal = (gx, gy)
                to_goal = _bfs_path(cur, goal, blocked, int(args.width), int(args.height))
                reserve = 2
                action: Action = 4
                if to_goal and batt >= len(to_goal) + reserve:
                    action = int(to_goal[0]) if to_goal else 4
                else:
                    best_ch = _nearest_charger(cur, chargers, blocked, int(args.width), int(args.height))
                    if best_ch is not None and best_ch[1]:
                        action = int(best_ch[1][0])
                    elif to_goal:
                        action = int(to_goal[0])

                sr = verse.step(action)
                out_row: Dict[str, Any] = {
                    "episode_id": f"ep_wh_{ep:06d}",
                    "step_idx": step_idx,
                    "obs": obs,
                    "action": action,
                    "reward": float(sr.reward),
                    "done": bool(sr.done or sr.truncated),
                    "next_obs": sr.obs,
                    "verse_name": "warehouse_world",
                    "source": "planner_expert",
                }
                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                rows += 1

                obs = sr.obs
                done = bool(sr.done or sr.truncated)
                if isinstance(sr.info, dict) and bool(sr.info.get("reached_goal", False)):
                    successes += 1
                step_idx += 1

            verse.close()

    print("warehouse_planner_dataset")
    print(f"out={args.out}")
    print(f"rows={rows}")
    print(f"success_events={successes}")


if __name__ == "__main__":
    main()

