"""
verses/escape_world.py

Stealth / evasion verse on a 2D grid.
The agent must reach an exit while avoiding patrol guards that follow
predictable routes. Hiding spots grant temporary invisibility. Getting
spotted by a guard resets the agent to the start with a penalty.
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
class EscapeParams:
    width: int = 10
    height: int = 10
    max_steps: int = 100
    num_guards: int = 3
    num_hiding_spots: int = 4
    guard_vision: int = 2          # Manhattan vision radius
    hiding_cooldown: int = 5       # steps of invisibility per hide
    step_penalty: float = -0.05
    spotted_penalty: float = -5.0
    exit_reward: float = 10.0
    wall_count: int = 8


@dataclass
class Guard:
    x: int
    y: int
    route: List[Tuple[int, int]]
    route_idx: int = 0
    direction: int = 1  # +1 forward, -1 backward along route


class EscapeWorldVerse:
    """
    Stealth navigation:
    - Guards patrol fixed routes; they reverse when hitting route ends
    - Guard vision: Manhattan distance <= guard_vision => spotted
    - Hiding spots: entering one grants hiding_cooldown steps of invisibility
    - Walls block movement but not guard vision
    - Goal: reach the exit without being spotted
    """

    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in EscapeWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        cfg = self.spec.params
        self.params = EscapeParams(
            width=max(6, int(cfg.get("width", 10))),
            height=max(6, int(cfg.get("height", 10))),
            max_steps=max(10, int(cfg.get("max_steps", 100))),
            num_guards=max(1, int(cfg.get("num_guards", 3))),
            num_hiding_spots=max(0, int(cfg.get("num_hiding_spots", 4))),
            guard_vision=max(1, int(cfg.get("guard_vision", 2))),
            hiding_cooldown=max(1, int(cfg.get("hiding_cooldown", 5))),
            step_penalty=float(cfg.get("step_penalty", -0.05)),
            spotted_penalty=float(cfg.get("spotted_penalty", -5.0)),
            exit_reward=float(cfg.get("exit_reward", 10.0)),
            wall_count=max(0, int(cfg.get("wall_count", 8))),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "x", "y", "exit_dist", "nearest_guard_dist", "hidden_steps_left",
                "guards_in_vision", "on_hiding_spot", "t",
            ],
            subspaces={
                "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "exit_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearest_guard_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "hidden_steps_left": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "guards_in_vision": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "on_hiding_spot": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="EscapeWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=5,
            notes="0=up,1=down,2=left,3=right,4=hide",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._x = 0
        self._y = 0
        self._t = 0
        self._done = False
        self._hidden_steps = 0
        self._guards: List[Guard] = []
        self._walls: Set[Tuple[int, int]] = set()
        self._hiding_spots: Set[Tuple[int, int]] = set()
        self._exit: Tuple[int, int] = (0, 0)
        self._times_spotted = 0

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
        self._hidden_steps = 0
        self._times_spotted = 0
        self._build_layout()

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "width": self.params.width,
            "height": self.params.height,
            "exit_x": self._exit[0],
            "exit_y": self._exit[1],
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
            # Hide action: only works on a hiding spot
            if (self._x, self._y) in self._hiding_spots:
                self._hidden_steps = self.params.hiding_cooldown
                info["activated_hide"] = True
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

            if 0 <= nx < self.params.width and 0 <= ny < self.params.height and (nx, ny) not in self._walls:
                self._x, self._y = nx, ny
            else:
                info["blocked"] = True

        # Advance guards
        for g in self._guards:
            if len(g.route) > 1:
                g.route_idx += g.direction
                if g.route_idx >= len(g.route):
                    g.direction = -1
                    g.route_idx = max(0, len(g.route) - 2)
                elif g.route_idx < 0:
                    g.direction = 1
                    g.route_idx = min(1, len(g.route) - 1)
                g.x, g.y = g.route[g.route_idx]

        # Check spotted
        if self._hidden_steps > 0:
            self._hidden_steps -= 1
            info["hidden"] = True
        else:
            for g in self._guards:
                dist = abs(self._x - g.x) + abs(self._y - g.y)
                if dist <= self.params.guard_vision:
                    reward += float(self.params.spotted_penalty)
                    self._times_spotted += 1
                    info["spotted"] = True
                    # Reset to start
                    self._x = 0
                    self._y = self.params.height - 1
                    break

        # Check exit
        if (self._x, self._y) == self._exit:
            reward += float(self.params.exit_reward)
            info["reached_goal"] = True
            done = True

        truncated = bool(self._t >= self.params.max_steps and not done)
        self._done = bool(done or truncated)
        info["t"] = int(self._t)
        info["times_spotted"] = int(self._times_spotted)

        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        rows: List[str] = []
        guard_pos = {(g.x, g.y) for g in self._guards}
        for y in range(self.params.height):
            row: List[str] = []
            for x in range(self.params.width):
                ch = "."
                if (x, y) in self._walls:
                    ch = "#"
                if (x, y) in self._hiding_spots:
                    ch = "H"
                if (x, y) == self._exit:
                    ch = "E"
                if (x, y) in guard_pos:
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
            "x": int(self._x), "y": int(self._y),
            "t": int(self._t), "done": bool(self._done),
            "hidden_steps": int(self._hidden_steps),
            "times_spotted": int(self._times_spotted),
            "guards": [{"x": g.x, "y": g.y, "route_idx": g.route_idx, "direction": g.direction} for g in self._guards],
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._x = max(0, min(self.params.width - 1, int(state.get("x", self._x))))
        self._y = max(0, min(self.params.height - 1, int(state.get("y", self._y))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        self._hidden_steps = max(0, int(state.get("hidden_steps", 0)))
        self._times_spotted = max(0, int(state.get("times_spotted", 0)))
        gs = state.get("guards")
        if isinstance(gs, list):
            for i, gd in enumerate(gs):
                if i < len(self._guards) and isinstance(gd, dict):
                    self._guards[i].x = int(gd.get("x", self._guards[i].x))
                    self._guards[i].y = int(gd.get("y", self._guards[i].y))
                    self._guards[i].route_idx = int(gd.get("route_idx", 0))
                    self._guards[i].direction = int(gd.get("direction", 1))

    def _make_obs(self) -> JSONValue:
        # Nearest guard
        nearest_guard = 99
        guards_in_range = 0
        for g in self._guards:
            d = abs(self._x - g.x) + abs(self._y - g.y)
            nearest_guard = min(nearest_guard, d)
            if d <= self.params.guard_vision + 1:
                guards_in_range += 1
        exit_dist = abs(self._x - self._exit[0]) + abs(self._y - self._exit[1])
        on_hiding = 1 if (self._x, self._y) in self._hiding_spots else 0
        return {
            "x": int(self._x),
            "y": int(self._y),
            "exit_dist": int(exit_dist),
            "nearest_guard_dist": int(nearest_guard),
            "hidden_steps_left": int(self._hidden_steps),
            "guards_in_vision": int(guards_in_range),
            "on_hiding_spot": int(on_hiding),
            "t": int(self._t),
        }

    def _build_layout(self) -> None:
        self._walls = set()
        self._hiding_spots = set()
        self._guards = []

        # Start: bottom-left
        self._x = 0
        self._y = self.params.height - 1
        # Exit: top-right
        self._exit = (self.params.width - 1, 0)

        start = (self._x, self._y)
        reserved = {start, self._exit}

        # Place walls
        for _ in range(self.params.wall_count * 3):
            if len(self._walls) >= self.params.wall_count:
                break
            wx = self._rng.randint(0, self.params.width - 1)
            wy = self._rng.randint(0, self.params.height - 1)
            if (wx, wy) not in reserved:
                self._walls.add((wx, wy))
                if not self._path_exists(start, self._exit):
                    self._walls.discard((wx, wy))

        # Place hiding spots
        open_cells = [
            (x, y) for x in range(self.params.width) for y in range(self.params.height)
            if (x, y) not in reserved and (x, y) not in self._walls
        ]
        self._rng.shuffle(open_cells)
        for i in range(min(self.params.num_hiding_spots, len(open_cells))):
            self._hiding_spots.add(open_cells[i])

        # Create patrol guards with routes
        guard_cells = [c for c in open_cells if c not in self._hiding_spots]
        self._rng.shuffle(guard_cells)
        for i in range(min(self.params.num_guards, len(guard_cells))):
            gx, gy = guard_cells[i]
            # Build a short patrol route (3-5 waypoints in a line)
            route_len = self._rng.randint(3, 5)
            horizontal = self._rng.random() < 0.5
            route: List[Tuple[int, int]] = [(gx, gy)]
            for _ in range(route_len - 1):
                lx, ly = route[-1]
                if horizontal:
                    nlx = lx + self._rng.choice([-1, 1])
                    nlx = max(0, min(self.params.width - 1, nlx))
                    if (nlx, ly) not in self._walls and (nlx, ly) not in reserved:
                        route.append((nlx, ly))
                else:
                    nly = ly + self._rng.choice([-1, 1])
                    nly = max(0, min(self.params.height - 1, nly))
                    if (lx, nly) not in self._walls and (lx, nly) not in reserved:
                        route.append((lx, nly))
            self._guards.append(Guard(x=gx, y=gy, route=route, route_idx=0, direction=1))

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
                        and (nx, ny) not in self._walls and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False


class EscapeWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "2d", "stealth", "evasion", "opponent_modeling", "risk_sensitive"]

    def create(self, spec: VerseSpec) -> Verse:
        return EscapeWorldVerse(spec)
