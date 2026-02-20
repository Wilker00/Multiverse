"""
verses/memory_vault_world.py

Diagnostic verse for sequential short-term memory:
- Goal location is only shown briefly at reset.
- Agent must remember the hint while navigating a procedural maze.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class MemoryVaultParams:
    width: int = 9
    height: int = 9
    max_steps: int = 120
    wall_density: float = 0.16
    step_penalty: float = -0.02
    wall_penalty: float = -0.12
    goal_reward: float = 8.0
    hint_visible_steps: int = 1


class MemoryVaultWorldVerse:
    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in MemoryVaultWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        cfg = self.spec.params
        self.params = MemoryVaultParams(
            width=max(5, int(cfg.get("width", 9))),
            height=max(5, int(cfg.get("height", 9))),
            max_steps=max(10, int(cfg.get("max_steps", 120))),
            wall_density=max(0.0, min(0.45, float(cfg.get("wall_density", 0.16)))),
            step_penalty=float(cfg.get("step_penalty", -0.02)),
            wall_penalty=float(cfg.get("wall_penalty", -0.12)),
            goal_reward=float(cfg.get("goal_reward", 8.0)),
            hint_visible_steps=max(1, int(cfg.get("hint_visible_steps", 1))),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=["x", "y", "t", "goal_hint_visible", "goal_hint_x", "goal_hint_y", "walls_up", "walls_down", "walls_left", "walls_right"],
            subspaces={
                "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_hint_visible": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_hint_x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_hint_y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "walls_up": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "walls_down": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "walls_left": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "walls_right": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="Hint is only visible in the opening frame(s).",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=5,
            notes="0=up,1=right,2=down,3=left,4=wait",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._grid: List[List[int]] = []
        self._agent: Tuple[int, int] = (0, 0)
        self._goal: Tuple[int, int] = (0, 0)
        self._hint_visible_steps_left = 0
        self._t = 0
        self._done = False

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)
        self._t = 0
        self._done = False
        self._agent = (0, 0)
        self._hint_visible_steps_left = int(self.params.hint_visible_steps)
        self._generate_layout()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "goal_hint_visible": True,
            "hint_visible_steps": int(self.params.hint_visible_steps),
            "stm_diagnostic": True,
        }
        return ResetResult(obs=self._make_obs(), info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(obs=self._make_obs(), reward=0.0, done=True, truncated=False, info={"warning": "step() called after done"})

        a = max(0, min(4, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": int(a)}

        x, y = self._agent
        nx, ny = x, y
        if a == 0:
            ny -= 1
        elif a == 1:
            nx += 1
        elif a == 2:
            ny += 1
        elif a == 3:
            nx -= 1

        moved = True
        if a != 4:
            if not self._is_free(nx, ny):
                moved = False
                reward += float(self.params.wall_penalty)
                info["hit_wall"] = True
            else:
                self._agent = (nx, ny)
        else:
            moved = False
            info["waited"] = True
        info["moved"] = bool(moved)

        reached_goal = bool(self._agent == self._goal)
        if reached_goal:
            reward += float(self.params.goal_reward)
            info["reached_goal"] = True
            self._done = True

        if self._hint_visible_steps_left > 0:
            self._hint_visible_steps_left -= 1
        info["goal_hint_visible"] = bool(self._hint_visible_steps_left > 0)
        info["hint_frame_consumed"] = bool(self._t <= int(self.params.hint_visible_steps))
        info["t"] = int(self._t)

        truncated = bool(self._t >= int(self.params.max_steps) and not self._done)
        self._done = bool(self._done or truncated)
        return StepResult(
            obs=self._make_obs(),
            reward=float(reward),
            done=bool(reached_goal),
            truncated=bool(truncated),
            info=info,
        )

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        rows: List[str] = []
        for y in range(self.params.height):
            chars: List[str] = []
            for x in range(self.params.width):
                if (x, y) == self._agent:
                    chars.append("A")
                elif (x, y) == self._goal:
                    chars.append("G" if self._hint_visible_steps_left > 0 else "?")
                elif self._grid[y][x] == 1:
                    chars.append("#")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        return "\n".join(rows)

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "grid": [list(r) for r in self._grid],
            "agent_x": int(self._agent[0]),
            "agent_y": int(self._agent[1]),
            "goal_x": int(self._goal[0]),
            "goal_y": int(self._goal[1]),
            "hint_visible_steps_left": int(self._hint_visible_steps_left),
            "t": int(self._t),
            "done": bool(self._done),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        grid = state.get("grid")
        if isinstance(grid, list) and len(grid) == self.params.height:
            rows: List[List[int]] = []
            ok = True
            for row in grid:
                if not isinstance(row, list) or len(row) != self.params.width:
                    ok = False
                    break
                rows.append([1 if int(v) != 0 else 0 for v in row])
            if ok:
                self._grid = rows
        ax = max(0, min(self.params.width - 1, int(state.get("agent_x", self._agent[0]))))
        ay = max(0, min(self.params.height - 1, int(state.get("agent_y", self._agent[1]))))
        gx = max(0, min(self.params.width - 1, int(state.get("goal_x", self._goal[0]))))
        gy = max(0, min(self.params.height - 1, int(state.get("goal_y", self._goal[1]))))
        self._agent = (ax, ay)
        self._goal = (gx, gy)
        self._hint_visible_steps_left = max(0, int(state.get("hint_visible_steps_left", self._hint_visible_steps_left)))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.params.width and 0 <= y < self.params.height

    def _is_free(self, x: int, y: int) -> bool:
        if not self._in_bounds(x, y):
            return False
        return bool(self._grid[y][x] == 0)

    def _neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if self._is_free(nx, ny):
                out.append((nx, ny))
        return out

    def _reachable(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        if start == goal:
            return True
        frontier = [start]
        seen: Set[Tuple[int, int]] = {start}
        while frontier:
            cur = frontier.pop()
            for nb in self._neighbors(cur[0], cur[1]):
                if nb in seen:
                    continue
                if nb == goal:
                    return True
                seen.add(nb)
                frontier.append(nb)
        return False

    def _generate_layout(self) -> None:
        w = self.params.width
        h = self.params.height
        start = (0, 0)
        for _ in range(64):
            grid = [[0 for _ in range(w)] for _ in range(h)]
            gx = self._rng.randrange(0, w)
            gy = self._rng.randrange(0, h)
            if (gx, gy) == start:
                gx = min(w - 1, gx + 1)
            for y in range(h):
                for x in range(w):
                    if (x, y) == start or (x, y) == (gx, gy):
                        continue
                    if self._rng.random() < float(self.params.wall_density):
                        grid[y][x] = 1
            self._grid = grid
            self._goal = (gx, gy)
            if self._reachable(start, self._goal):
                return
        # Fallback to open world to guarantee reachability.
        self._grid = [[0 for _ in range(w)] for _ in range(h)]
        self._goal = (w - 1, h - 1)

    def _make_obs(self) -> JSONValue:
        x, y = self._agent
        hint_visible = bool(self._hint_visible_steps_left > 0)
        gx, gy = self._goal if hint_visible else (-1, -1)
        up = 0 if self._is_free(x, y - 1) else 1
        down = 0 if self._is_free(x, y + 1) else 1
        left = 0 if self._is_free(x - 1, y) else 1
        right = 0 if self._is_free(x + 1, y) else 1
        return {
            "x": int(x),
            "y": int(y),
            "t": int(self._t),
            "goal_hint_visible": 1 if hint_visible else 0,
            "goal_hint_x": int(gx),
            "goal_hint_y": int(gy),
            "walls_up": int(up),
            "walls_down": int(down),
            "walls_left": int(left),
            "walls_right": int(right),
        }


class MemoryVaultWorldFactory:
    @property
    def tags(self) -> List[str]:
        return [
            "navigation",
            "memory_diagnostics",
            "sequential_memory",
            "partial_observable",
            "risk_sensitive",
        ]

    def create(self, spec: VerseSpec) -> Verse:
        return MemoryVaultWorldVerse(spec)

