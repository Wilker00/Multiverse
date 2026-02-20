"""
verses/harvest_world.py

Resource-collection verse on a 2D grid.
The agent gathers scattered fruit, carries them (limited capacity),
and deposits at a home base for reward. Obstacles block movement.
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
class HarvestParams:
    width: int = 8
    height: int = 8
    max_steps: int = 120
    num_fruit: int = 6
    obstacle_count: int = 8
    carry_capacity: int = 3
    step_penalty: float = -0.05
    fruit_reward: float = 0.5
    deposit_reward: float = 3.0
    obstacle_penalty: float = -0.5
    wall_penalty: float = -0.3
    # Fruit spoilage: probability each uncollected fruit disappears per step
    spoil_probability: float = 0.01


class HarvestWorldVerse:
    """
    2D grid resource-collection:
    - Collect fruit scattered on the grid
    - Limited carry capacity forces deposit trips
    - Deposit at home base (0, 0) for big reward
    - Obstacles block movement (BFS-validated placement)
    - Fruit can spoil over time
    """

    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in HarvestWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        cfg = self.spec.params
        self.params = HarvestParams(
            width=max(5, int(cfg.get("width", 8))),
            height=max(5, int(cfg.get("height", 8))),
            max_steps=max(10, int(cfg.get("max_steps", 120))),
            num_fruit=max(1, int(cfg.get("num_fruit", 6))),
            obstacle_count=max(0, int(cfg.get("obstacle_count", 8))),
            carry_capacity=max(1, int(cfg.get("carry_capacity", 3))),
            step_penalty=float(cfg.get("step_penalty", -0.05)),
            fruit_reward=float(cfg.get("fruit_reward", 0.5)),
            deposit_reward=float(cfg.get("deposit_reward", 3.0)),
            obstacle_penalty=float(cfg.get("obstacle_penalty", -0.5)),
            wall_penalty=float(cfg.get("wall_penalty", -0.3)),
            spoil_probability=max(0.0, min(1.0, float(cfg.get("spoil_probability", 0.01)))),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "x", "y", "carrying", "deposited", "fruit_remaining",
                "nearby_fruit", "nearest_fruit_dist", "nearest_base_dist", "t",
            ],
            subspaces={
                "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "carrying": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "deposited": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "fruit_remaining": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearby_fruit": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearest_fruit_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearest_base_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="HarvestWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=5,
            notes="0=up,1=down,2=left,3=right,4=deposit",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._x = 0
        self._y = 0
        self._t = 0
        self._done = False
        self._carrying = 0
        self._deposited = 0
        self._obstacles: Set[Tuple[int, int]] = set()
        self._fruit: Set[Tuple[int, int]] = set()
        self._base = (0, 0)

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        self._x = 0
        self._y = 0
        self._t = 0
        self._done = False
        self._carrying = 0
        self._deposited = 0
        self._build_layout()

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "width": self.params.width,
            "height": self.params.height,
            "num_fruit": len(self._fruit),
            "obstacle_count": len(self._obstacles),
        }
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = max(0, min(4, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": int(a)}
        done = False

        if a == 4:
            # Deposit at base
            if (self._x, self._y) == self._base and self._carrying > 0:
                reward += float(self.params.deposit_reward) * self._carrying
                self._deposited += self._carrying
                info["deposited"] = self._carrying
                self._carrying = 0
        else:
            nx, ny = self._x, self._y
            if a == 0:
                ny -= 1
            elif a == 1:
                ny += 1
            elif a == 2:
                nx -= 1
            elif a == 3:
                nx += 1

            if not (0 <= nx < self.params.width and 0 <= ny < self.params.height):
                reward += float(self.params.wall_penalty)
                info["hit_wall"] = True
            elif (nx, ny) in self._obstacles:
                reward += float(self.params.obstacle_penalty)
                info["hit_obstacle"] = True
            else:
                self._x, self._y = nx, ny

                # Auto-pick fruit if on a fruit cell and have capacity
                if (self._x, self._y) in self._fruit and self._carrying < self.params.carry_capacity:
                    self._fruit.discard((self._x, self._y))
                    self._carrying += 1
                    reward += float(self.params.fruit_reward)
                    info["picked_fruit"] = True

        # Spoilage: each remaining fruit has a chance to disappear
        if self.params.spoil_probability > 0.0 and self._fruit:
            spoiled = set()
            for f in self._fruit:
                if self._rng.random() < self.params.spoil_probability:
                    spoiled.add(f)
            if spoiled:
                self._fruit -= spoiled
                info["spoiled"] = len(spoiled)

        # Done conditions
        total_fruit = len(self._fruit) + self._carrying
        if total_fruit == 0 and self._deposited > 0:
            # All fruit deposited or gone
            done = True
            info["all_done"] = True

        truncated = bool(self._t >= self.params.max_steps and not done)
        self._done = bool(done or truncated)
        info["t"] = int(self._t)
        info["carrying"] = int(self._carrying)
        info["deposited"] = int(self._deposited)
        info["fruit_remaining"] = len(self._fruit)

        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        rows: List[str] = []
        for y in range(self.params.height):
            row: List[str] = []
            for x in range(self.params.width):
                ch = "."
                if (x, y) in self._obstacles:
                    ch = "#"
                if (x, y) in self._fruit:
                    ch = "F"
                if (x, y) == self._base:
                    ch = "B"
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
            "carrying": int(self._carrying),
            "deposited": int(self._deposited),
            "t": int(self._t),
            "done": bool(self._done),
            "obstacles": [[x, y] for (x, y) in sorted(self._obstacles)],
            "fruit": [[x, y] for (x, y) in sorted(self._fruit)],
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._x = max(0, min(self.params.width - 1, int(state.get("x", self._x))))
        self._y = max(0, min(self.params.height - 1, int(state.get("y", self._y))))
        self._carrying = max(0, int(state.get("carrying", self._carrying)))
        self._deposited = max(0, int(state.get("deposited", self._deposited)))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        obs = state.get("obstacles")
        if isinstance(obs, list):
            self._obstacles = set((int(p[0]), int(p[1])) for p in obs if isinstance(p, list) and len(p) == 2)
        fr = state.get("fruit")
        if isinstance(fr, list):
            self._fruit = set((int(p[0]), int(p[1])) for p in fr if isinstance(p, list) and len(p) == 2)

    def _make_obs(self) -> JSONValue:
        # Count nearby fruit in cardinal directions
        nearby = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (self._x + dx, self._y + dy) in self._fruit:
                nearby += 1
        # Nearest fruit Manhattan distance
        nearest_fruit = -1
        if self._fruit:
            nearest_fruit = min(abs(self._x - fx) + abs(self._y - fy) for fx, fy in self._fruit)
        # Distance to base
        base_dist = abs(self._x - self._base[0]) + abs(self._y - self._base[1])
        return {
            "x": int(self._x),
            "y": int(self._y),
            "carrying": int(self._carrying),
            "deposited": int(self._deposited),
            "fruit_remaining": len(self._fruit),
            "nearby_fruit": int(nearby),
            "nearest_fruit_dist": int(nearest_fruit),
            "nearest_base_dist": int(base_dist),
            "t": int(self._t),
        }

    def _path_exists(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        if start == goal:
            return True
        visited: Set[Tuple[int, int]] = {start}
        queue: deque[Tuple[int, int]] = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) == goal:
                    return True
                if (0 <= nx < self.params.width and 0 <= ny < self.params.height
                        and (nx, ny) not in self._obstacles and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def _build_layout(self) -> None:
        self._obstacles = set()
        self._fruit = set()
        self._base = (0, 0)

        reserved = {self._base}

        # Place obstacles with path validation to all potential fruit spots
        attempts = 0
        max_attempts = self.params.obstacle_count * 6
        while len(self._obstacles) < self.params.obstacle_count and attempts < max_attempts:
            attempts += 1
            x = self._rng.randint(0, self.params.width - 1)
            y = self._rng.randint(0, self.params.height - 1)
            p = (x, y)
            if p in reserved:
                continue
            self._obstacles.add(p)
            # Verify base is still reachable from center-ish area
            test_goal = (self.params.width - 1, self.params.height - 1)
            if not self._path_exists(self._base, test_goal):
                self._obstacles.discard(p)

        # Place fruit on open cells
        open_cells = [
            (x, y) for x in range(self.params.width) for y in range(self.params.height)
            if (x, y) not in self._obstacles and (x, y) != self._base
        ]
        self._rng.shuffle(open_cells)
        for i in range(min(self.params.num_fruit, len(open_cells))):
            self._fruit.add(open_cells[i])


class HarvestWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "2d", "resource_collection", "obstacle_navigation", "carry_capacity"]

    def create(self, spec: VerseSpec) -> Verse:
        return HarvestWorldVerse(spec)
