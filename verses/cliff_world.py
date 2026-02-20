"""
verses/cliff_world.py

Risky 2D navigation verse with a cliff strip.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


def hash_spec(spec: VerseSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class CliffWorldParams:
    width: int = 12
    height: int = 4
    start_x: int = 0
    start_y: int = -1
    goal_x: int = -1
    goal_y: int = -1
    max_steps: int = 100
    step_penalty: float = -1.0
    cliff_penalty: float = -100.0
    end_on_cliff: bool = False
    # Wind: probability per step that a lateral gust pushes the agent toward the cliff
    wind_probability: float = 0.10
    # Crumbling: cells adjacent to the cliff can crumble, expanding the danger zone
    crumble_probability: float = 0.03


class CliffWorldVerse:
    """
    Classic cliff world:
    - Start near bottom-left
    - Goal near bottom-right
    - Bottom strip between start and goal is the cliff
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in CliffWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = CliffWorldParams(
            width=int(self.spec.params.get("width", 12)),
            height=int(self.spec.params.get("height", 4)),
            start_x=int(self.spec.params.get("start_x", 0)),
            start_y=int(self.spec.params.get("start_y", -1)),
            goal_x=int(self.spec.params.get("goal_x", -1)),
            goal_y=int(self.spec.params.get("goal_y", -1)),
            max_steps=int(self.spec.params.get("max_steps", 100)),
            step_penalty=float(self.spec.params.get("step_penalty", -1.0)),
            cliff_penalty=float(self.spec.params.get("cliff_penalty", -100.0)),
            end_on_cliff=bool(self.spec.params.get("end_on_cliff", False)),
            wind_probability=max(0.0, min(1.0, float(self.spec.params.get("wind_probability", 0.10)))),
            crumble_probability=max(0.0, min(1.0, float(self.spec.params.get("crumble_probability", 0.03)))),
        )
        self.params.width = max(3, int(self.params.width))
        self.params.height = max(2, int(self.params.height))
        self.params.max_steps = max(1, int(self.params.max_steps))

        obs_keys = ["x", "y", "t", "cliff_adjacent", "wind_active", "crumbled_count"]
        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=obs_keys,
            subspaces={
                "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "cliff_adjacent": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "wind_active": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "crumbled_count": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="CliffWorld obs dict with wind and crumbling",
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
        self._crumbled: Set[Tuple[int, int]] = set()
        self._wind_active = False

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        self._x = self._start_x()
        self._y = self._start_y()
        self._t = 0
        self._done = False
        self._crumbled = set()
        self._wind_active = False

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "width": self.params.width,
            "height": self.params.height,
            "start_x": self._start_x(),
            "start_y": self._start_y(),
            "goal_x": self._goal_x(),
            "goal_y": self._goal_y(),
        }
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(
                obs=self._make_obs(),
                reward=0.0,
                done=True,
                truncated=False,
                info={"warning": "step() called after done"},
            )

        a = int(action)
        if a == 0:
            self._y = max(0, self._y - 1)
        elif a == 1:
            self._y = min(self.params.height - 1, self._y + 1)
        elif a == 2:
            self._x = max(0, self._x - 1)
        elif a == 3:
            self._x = min(self.params.width - 1, self._x + 1)
        else:
            raise ValueError("CliffWorld action must be 0..3")

        self._t += 1

        # Wind gust: push agent one cell toward the cliff (downward)
        self._wind_active = False
        if self.params.wind_probability > 0.0 and self._rng.random() < self.params.wind_probability:
            self._wind_active = True
            self._y = min(self.params.height - 1, self._y + 1)

        # Crumbling: cells adjacent to cliff may become cliff
        if self.params.crumble_probability > 0.0:
            cliff_row = self.params.height - 1
            for cx in range(self.params.width):
                if self._is_base_cliff(cx, cliff_row):
                    # Check cell above the cliff
                    above = (cx, cliff_row - 1)
                    if above not in self._crumbled and self._rng.random() < self.params.crumble_probability:
                        # Don't crumble start or goal positions
                        if above != (self._start_x(), self._start_y()) and above != (self._goal_x(), self._goal_y()):
                            self._crumbled.add(above)
                # Also crumble from already-crumbled cells
                for cy in range(cliff_row - 1, 0, -1):
                    if (cx, cy) in self._crumbled:
                        above2 = (cx, cy - 1)
                        if above2 not in self._crumbled and self._rng.random() < self.params.crumble_probability * 0.5:
                            if above2 != (self._start_x(), self._start_y()) and above2 != (self._goal_x(), self._goal_y()):
                                self._crumbled.add(above2)

        fell_cliff = self._is_cliff(self._x, self._y)
        reached_goal = (self._x == self._goal_x()) and (self._y == self._goal_y())

        reward = float(self.params.step_penalty)
        done = False
        if fell_cliff:
            reward = float(self.params.cliff_penalty)
            if bool(self.params.end_on_cliff):
                done = True
            else:
                self._x = self._start_x()
                self._y = self._start_y()
        elif reached_goal:
            reward = 0.0
            done = True

        truncated = self._t >= self.params.max_steps and not done
        self._done = bool(done or truncated)

        info: Dict[str, JSONValue] = {
            "t": self._t,
            "x": self._x,
            "y": self._y,
            "fell_cliff": bool(fell_cliff),
            "reached_goal": bool(reached_goal),
            "goal_x": self._goal_x(),
            "goal_y": self._goal_y(),
            "wind_pushed": bool(self._wind_active),
            "crumbled_count": len(self._crumbled),
        }

        return StepResult(
            obs=self._make_obs(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode == "human":
            frame = self.render(mode="ansi")
            if frame is not None:
                print(frame)
            return None
        if mode == "rgb_array":
            return self._render_rgb_array()
        if mode != "ansi":
            return None
        rows: List[str] = []
        for y in range(self.params.height):
            row: List[str] = []
            for x in range(self.params.width):
                if x == self._x and y == self._y:
                    row.append("A")
                elif x == self._goal_x() and y == self._goal_y():
                    row.append("G")
                elif x == self._start_x() and y == self._start_y():
                    row.append("S")
                elif self._is_cliff(x, y):
                    row.append("C")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)

    def _render_rgb_array(self) -> List[List[List[int]]]:
        """
        Lightweight RGB frame (H x W x 3) using Python lists.
        Colors:
        - background: white
        - start: gray
        - goal: green
        - cliff: red
        - agent: blue
        """
        bg = [255, 255, 255]
        start = [180, 180, 180]
        goal = [60, 200, 80]
        cliff = [220, 60, 60]
        agent = [60, 120, 240]
        frame: List[List[List[int]]] = []
        for y in range(self.params.height):
            row: List[List[int]] = []
            for x in range(self.params.width):
                px = list(bg)
                if x == self._start_x() and y == self._start_y():
                    px = list(start)
                if x == self._goal_x() and y == self._goal_y():
                    px = list(goal)
                if self._is_cliff(x, y):
                    px = list(cliff)
                if x == self._x and y == self._y:
                    px = list(agent)
                row.append(px)
            frame.append(row)
        return frame

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "x": int(self._x),
            "y": int(self._y),
            "t": int(self._t),
            "done": bool(self._done),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._x = max(0, min(self.params.width - 1, int(state.get("x", self._x))))
        self._y = max(0, min(self.params.height - 1, int(state.get("y", self._y))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))

    def _make_obs(self) -> JSONValue:
        # Count adjacent cliff cells
        cliff_adj = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self._x + dx, self._y + dy
            if 0 <= nx < self.params.width and 0 <= ny < self.params.height:
                if self._is_cliff(nx, ny):
                    cliff_adj += 1
        return {
            "x": self._x,
            "y": self._y,
            "t": self._t,
            "cliff_adjacent": cliff_adj,
            "wind_active": 1 if self._wind_active else 0,
            "crumbled_count": len(self._crumbled),
        }

    def _start_x(self) -> int:
        return max(0, min(self.params.width - 1, int(self.params.start_x)))

    def _start_y(self) -> int:
        y = int(self.params.start_y)
        if y < 0:
            y = self.params.height - 1
        return max(0, min(self.params.height - 1, y))

    def _goal_x(self) -> int:
        x = int(self.params.goal_x)
        if x < 0:
            x = self.params.width - 1
        return max(0, min(self.params.width - 1, x))

    def _goal_y(self) -> int:
        y = int(self.params.goal_y)
        if y < 0:
            y = self.params.height - 1
        return max(0, min(self.params.height - 1, y))

    def _is_base_cliff(self, x: int, y: int) -> bool:
        """Check if (x, y) is part of the original fixed cliff strip."""
        if y != self.params.height - 1:
            return False
        if x <= min(self._start_x(), self._goal_x()):
            return False
        if x >= max(self._start_x(), self._goal_x()):
            return False
        return True

    def _is_cliff(self, x: int, y: int) -> bool:
        """Check if (x, y) is cliff (base cliff or crumbled)."""
        if self._is_base_cliff(x, y):
            return True
        if (x, y) in self._crumbled:
            return True
        return False


class CliffWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "grid", "risky", "cliff"]

    def create(self, spec: VerseSpec) -> Verse:
        return CliffWorldVerse(spec)
