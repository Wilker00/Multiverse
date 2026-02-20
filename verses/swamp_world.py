"""
verses/swamp_world.py

Dynamic-terrain survival verse on a 2D grid.
A rising flood slowly submerges the map from the bottom up.
The agent must reach high ground (the goal) before being swallowed.
Safe havens provide temporary flood immunity. Mud tiles slow movement.
"""

from __future__ import annotations

import dataclasses
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class SwampParams:
    width: int = 10
    height: int = 10
    max_steps: int = 80
    step_penalty: float = -0.05
    goal_reward: float = 10.0
    flood_penalty: float = -8.0
    mud_count: int = 6
    haven_count: int = 2
    # Flood rises every N steps (lower = faster)
    flood_rate: int = 5
    # Mud slows: 50% chance your move fails on mud
    mud_slip_prob: float = 0.50
    # Havens grant immunity for N steps after visiting
    haven_immunity_steps: int = 3


class SwampWorldVerse:
    """
    Dynamic terrain survival:
    - Flood rises from bottom row upward every flood_rate steps
    - Agent standing on a flooded cell (and not immune) => penalty + episode end
    - Mud tiles: movement may fail (slip), wasting the step
    - Safe havens: visiting grants flood immunity for a few steps
    - Goal: reach the top of the map (high ground) before drowning
    """

    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in SwampWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        cfg = self.spec.params
        self.params = SwampParams(
            width=max(5, int(cfg.get("width", 10))),
            height=max(5, int(cfg.get("height", 10))),
            max_steps=max(10, int(cfg.get("max_steps", 80))),
            step_penalty=float(cfg.get("step_penalty", -0.05)),
            goal_reward=float(cfg.get("goal_reward", 10.0)),
            flood_penalty=float(cfg.get("flood_penalty", -8.0)),
            mud_count=max(0, int(cfg.get("mud_count", 6))),
            haven_count=max(0, int(cfg.get("haven_count", 2))),
            flood_rate=max(1, int(cfg.get("flood_rate", 5))),
            mud_slip_prob=max(0.0, min(1.0, float(cfg.get("mud_slip_prob", 0.50)))),
            haven_immunity_steps=max(0, int(cfg.get("haven_immunity_steps", 3))),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "x", "y", "flood_level", "on_mud", "immunity_left",
                "nearest_haven_dist", "goal_dist", "t",
            ],
            subspaces={
                "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "flood_level": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "on_mud": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "immunity_left": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearest_haven_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="SwampWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=4,
            notes="0=up,1=down,2=left,3=right",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._x = 0
        self._y = 0
        self._t = 0
        self._done = False
        self._flood_level = 0  # rows from bottom that are flooded (0 = none)
        self._immunity = 0  # steps of flood immunity remaining
        self._mud: Set[Tuple[int, int]] = set()
        self._havens: Set[Tuple[int, int]] = set()
        self._goal: Tuple[int, int] = (0, 0)

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
        self._flood_level = 0
        self._immunity = 0
        self._build_layout()

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "width": self.params.width,
            "height": self.params.height,
            "goal_x": self._goal[0],
            "goal_y": self._goal[1],
        }
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = max(0, min(3, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": int(a)}
        done = False

        # Try to move
        nx, ny = self._x, self._y
        if a == 0:
            ny -= 1
        elif a == 1:
            ny += 1
        elif a == 2:
            nx -= 1
        elif a == 3:
            nx += 1

        # Bounds check
        if 0 <= nx < self.params.width and 0 <= ny < self.params.height:
            # Mud slip check: if target cell is mud, chance of slip
            if (nx, ny) in self._mud and self._rng.random() < self.params.mud_slip_prob:
                info["slipped_mud"] = True
                # Movement fails — stay in place
            else:
                self._x, self._y = nx, ny
        else:
            info["hit_wall"] = True

        # Check for haven visit
        if (self._x, self._y) in self._havens:
            self._immunity = self.params.haven_immunity_steps
            info["activated_haven"] = True

        # Advance flood
        if self._t > 0 and self._t % self.params.flood_rate == 0:
            self._flood_level = min(self.params.height, self._flood_level + 1)
            info["flood_rose"] = True

        # Check if agent is flooded
        agent_row_from_bottom = self.params.height - 1 - self._y
        if agent_row_from_bottom < self._flood_level:
            if self._immunity > 0:
                self._immunity -= 1
                info["immunity_saved"] = True
            else:
                reward += float(self.params.flood_penalty)
                info["drowned"] = True
                done = True

        # Decrement immunity if not used
        if not done and self._immunity > 0 and "immunity_saved" not in info and "activated_haven" not in info:
            self._immunity -= 1

        # Check goal
        if (self._x, self._y) == self._goal and not done:
            reward += float(self.params.goal_reward)
            info["reached_goal"] = True
            done = True

        truncated = bool(self._t >= self.params.max_steps and not done)
        self._done = bool(done or truncated)
        info["t"] = int(self._t)
        info["flood_level"] = int(self._flood_level)
        info["immunity"] = int(self._immunity)

        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        rows: List[str] = []
        for y in range(self.params.height):
            row: List[str] = []
            row_from_bottom = self.params.height - 1 - y
            flooded = row_from_bottom < self._flood_level
            for x in range(self.params.width):
                ch = "~" if flooded else "."
                if (x, y) in self._mud:
                    ch = "M" if not flooded else "~"
                if (x, y) in self._havens:
                    ch = "H" if not flooded else "~"
                if (x, y) == self._goal:
                    ch = "G"
                if x == self._x and y == self._y:
                    ch = "A"
                row.append(ch)
            rows.append("".join(row))
        return "\n".join(rows)

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "x": int(self._x),
            "y": int(self._y),
            "t": int(self._t),
            "done": bool(self._done),
            "flood_level": int(self._flood_level),
            "immunity": int(self._immunity),
            "mud": [[x, y] for x, y in sorted(self._mud)],
            "havens": [[x, y] for x, y in sorted(self._havens)],
            "goal": list(self._goal),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._x = max(0, min(self.params.width - 1, int(state.get("x", self._x))))
        self._y = max(0, min(self.params.height - 1, int(state.get("y", self._y))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        self._flood_level = max(0, int(state.get("flood_level", self._flood_level)))
        self._immunity = max(0, int(state.get("immunity", self._immunity)))
        mud = state.get("mud")
        if isinstance(mud, list):
            self._mud = set((int(p[0]), int(p[1])) for p in mud if isinstance(p, list) and len(p) == 2)
        havens = state.get("havens")
        if isinstance(havens, list):
            self._havens = set((int(p[0]), int(p[1])) for p in havens if isinstance(p, list) and len(p) == 2)
        g = state.get("goal")
        if isinstance(g, list) and len(g) == 2:
            self._goal = (int(g[0]), int(g[1]))

    def _make_obs(self) -> JSONValue:
        on_mud = 1 if (self._x, self._y) in self._mud else 0
        # Nearest haven
        nearest_haven = -1
        if self._havens:
            nearest_haven = min(abs(self._x - hx) + abs(self._y - hy) for hx, hy in self._havens)
        # Goal distance
        goal_dist = abs(self._x - self._goal[0]) + abs(self._y - self._goal[1])
        return {
            "x": int(self._x),
            "y": int(self._y),
            "flood_level": int(self._flood_level),
            "on_mud": int(on_mud),
            "immunity_left": int(self._immunity),
            "nearest_haven_dist": int(nearest_haven),
            "goal_dist": int(goal_dist),
            "t": int(self._t),
        }

    def _build_layout(self) -> None:
        self._mud = set()
        self._havens = set()

        # Start: bottom-left corner
        self._x = 0
        self._y = self.params.height - 1

        # Goal: random cell in top two rows (high ground)
        gx = self._rng.randint(0, self.params.width - 1)
        gy = self._rng.randint(0, 1)  # top two rows
        self._goal = (gx, gy)

        start = (self._x, self._y)
        reserved = {start, self._goal}

        # Place mud tiles (in bottom 2/3 of map)
        open_cells = [
            (x, y) for x in range(self.params.width)
            for y in range(self.params.height // 3, self.params.height)
            if (x, y) not in reserved
        ]
        self._rng.shuffle(open_cells)
        for i in range(min(self.params.mud_count, len(open_cells))):
            self._mud.add(open_cells[i])

        # Place havens (in middle band of map)
        mid_band = [
            (x, y) for x in range(self.params.width)
            for y in range(self.params.height // 4, 3 * self.params.height // 4)
            if (x, y) not in reserved and (x, y) not in self._mud
        ]
        self._rng.shuffle(mid_band)
        for i in range(min(self.params.haven_count, len(mid_band))):
            self._havens.add(mid_band[i])


class SwampWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "2d", "dynamic_terrain", "survival", "risk_sensitive", "time_pressure"]

    def create(self, spec: VerseSpec) -> Verse:
        return SwampWorldVerse(spec)
