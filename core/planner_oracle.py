"""
core/planner_oracle.py

Lightweight environment-state planner used for runtime assistance.
"""

from __future__ import annotations

import heapq
import json
from typing import Any, Dict, List, Optional, Tuple

from core.types import JSONValue


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _state_key(state: Dict[str, JSONValue]) -> str:
    # Planning abstraction to keep search tractable:
    # - ignore monotonically increasing time
    # - ignore done flags
    # - bucket battery to reduce duplicate near-identical states
    compact: Dict[str, JSONValue] = {}
    for k, v in state.items():
        ks = str(k)
        if ks in ("t", "done"):
            continue
        if ks == "battery":
            compact[ks] = int(_safe_int(v, 0) // 2)
            continue
        compact[ks] = v
    return json.dumps(compact, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _heuristic(
    info: Optional[Dict[str, Any]],
    goal_hint: Optional[Tuple[int, int]],
) -> int:
    if not isinstance(info, dict):
        return 0
    x = info.get("x")
    y = info.get("y")
    gx = info.get("goal_x")
    gy = info.get("goal_y")
    if x is None or y is None:
        return 0
    if gx is None or gy is None:
        if goal_hint is None:
            return 0
        gx, gy = goal_hint
    return abs(_safe_int(x) - _safe_int(gx)) + abs(_safe_int(y) - _safe_int(gy))


def plan_actions_from_current_state(
    *,
    verse: Any,
    horizon: int = 5,
    max_expansions: int = 8000,
    avoid_terminal_failures: bool = True,
) -> List[int]:
    """
    A* over the verse transition function from current exported state.

    Returns up to `horizon` actions toward the nearest discovered goal.
    """
    if not hasattr(verse, "export_state") or not hasattr(verse, "import_state"):
        return []
    if not hasattr(verse, "action_space"):
        return []

    n_actions = _safe_int(getattr(verse.action_space, "n", 0), 0)
    if n_actions <= 0:
        return []

    original_state = verse.export_state()
    if not isinstance(original_state, dict):
        return []

    # Disable env stochastic action noise during planning if supported.
    old_action_noise = None
    had_action_noise = False
    try:
        if hasattr(verse, "params") and hasattr(verse.params, "action_noise"):
            old_action_noise = float(getattr(verse.params, "action_noise"))
            setattr(verse.params, "action_noise", 0.0)
            had_action_noise = True
    except Exception:
        had_action_noise = False

    try:
        start_key = _state_key(original_state)
        g_best: Dict[str, int] = {start_key: 0}
        state_by_key: Dict[str, Dict[str, JSONValue]] = {start_key: dict(original_state)}
        parent: Dict[str, Tuple[str, int]] = {}
        goal_hint: Optional[Tuple[int, int]] = None

        frontier: List[Tuple[int, int, str]] = []
        heapq.heappush(frontier, (0, 0, start_key))

        solved_key: Optional[str] = None
        expansions = 0

        while frontier and expansions < int(max_expansions):
            _, g, cur_key = heapq.heappop(frontier)
            if g > g_best.get(cur_key, 10**9):
                continue

            cur_state = state_by_key[cur_key]

            for a in range(n_actions):
                verse.import_state(cur_state)
                step = verse.step(a)
                expansions += 1
                next_state = verse.export_state()
                if not isinstance(next_state, dict):
                    continue
                next_key = _state_key(next_state)
                info = step.info if isinstance(step.info, dict) else {}

                if "goal_x" in info and "goal_y" in info:
                    goal_hint = (_safe_int(info.get("goal_x")), _safe_int(info.get("goal_y")))

                reached_goal = bool(info.get("reached_goal") is True)
                if reached_goal:
                    parent[next_key] = (cur_key, int(a))
                    solved_key = next_key
                    frontier = []
                    break

                if bool(step.truncated):
                    continue
                if bool(step.done) and avoid_terminal_failures:
                    continue

                ng = g + 1
                if ng >= g_best.get(next_key, 10**9):
                    continue

                g_best[next_key] = ng
                state_by_key[next_key] = next_state
                parent[next_key] = (cur_key, int(a))
                h = _heuristic(info, goal_hint)
                heapq.heappush(frontier, (ng + h, ng, next_key))

        if solved_key is None:
            return []

        actions_rev: List[int] = []
        cur = solved_key
        while cur != start_key:
            prev, act = parent[cur]
            actions_rev.append(int(act))
            cur = prev
        actions_rev.reverse()
        if not actions_rev:
            return []
        return actions_rev[: max(1, int(horizon))]
    finally:
        # Always restore environment state and planning toggles.
        try:
            verse.import_state(original_state)
        except Exception:
            pass
        if had_action_noise:
            try:
                setattr(verse.params, "action_noise", old_action_noise)
            except Exception:
                pass
