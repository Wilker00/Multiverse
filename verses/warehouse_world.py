"""
verses/warehouse_world.py

Warehouse navigation with obstacles, battery management, chargers,
conveyor belts, and a patrol robot.
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
class WarehouseParams:
    width: int = 8
    height: int = 8
    start_x: int = 0
    start_y: int = 0
    goal_x: int = -1
    goal_y: int = -1
    max_steps: int = 100
    obstacle_count: int = 14
    battery_capacity: int = 20
    battery_drain: int = 1
    charge_rate: int = 5
    step_penalty: float = -0.10
    obstacle_penalty: float = -1.00
    wall_penalty: float = -0.50
    battery_fail_penalty: float = -10.0
    goal_reward: float = 10.0
    charge_reward: float = 0.5
    # Conveyor belts: tiles that push the agent one cell in a fixed direction
    conveyor_count: int = 3
    # Patrol robot: moves along a fixed route and resets agent on contact
    patrol_robot: bool = True
    patrol_penalty: float = -5.0
    lidar_range: int = 4


class WarehouseWorldVerse:
    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in WarehouseWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)
        self.params = WarehouseParams(
            width=int(self.spec.params.get("width", 8)),
            height=int(self.spec.params.get("height", 8)),
            start_x=int(self.spec.params.get("start_x", 0)),
            start_y=int(self.spec.params.get("start_y", 0)),
            goal_x=int(self.spec.params.get("goal_x", -1)),
            goal_y=int(self.spec.params.get("goal_y", -1)),
            max_steps=int(self.spec.params.get("max_steps", 100)),
            obstacle_count=int(self.spec.params.get("obstacle_count", 14)),
            battery_capacity=int(self.spec.params.get("battery_capacity", 20)),
            battery_drain=int(self.spec.params.get("battery_drain", 1)),
            charge_rate=int(self.spec.params.get("charge_rate", 5)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.10)),
            obstacle_penalty=float(self.spec.params.get("obstacle_penalty", -1.00)),
            wall_penalty=float(self.spec.params.get("wall_penalty", -0.50)),
            battery_fail_penalty=float(self.spec.params.get("battery_fail_penalty", -10.0)),
            goal_reward=float(self.spec.params.get("goal_reward", 10.0)),
            charge_reward=float(self.spec.params.get("charge_reward", 0.5)),
            conveyor_count=max(0, int(self.spec.params.get("conveyor_count", 3))),
            patrol_robot=bool(self.spec.params.get("patrol_robot", True)),
            patrol_penalty=float(self.spec.params.get("patrol_penalty", -5.0)),
            lidar_range=int(self.spec.params.get("lidar_range", 4)),
        )

        self.params.width = max(5, int(self.params.width))
        self.params.height = max(5, int(self.params.height))
        self.params.max_steps = max(10, int(self.params.max_steps))
        self.params.obstacle_count = max(1, int(self.params.obstacle_count))
        self.params.battery_capacity = max(1, int(self.params.battery_capacity))
        self.params.battery_drain = max(0, int(self.params.battery_drain))
        self.params.charge_rate = max(0, int(self.params.charge_rate))
        self.params.lidar_range = max(1, int(self.params.lidar_range))

        self._enable_ego_grid = bool(self.spec.params.get("enable_ego_grid", False))
        ego_size = int(self.spec.params.get("ego_grid_size", 5))
        if ego_size < 3:
            ego_size = 3
        if ego_size % 2 == 0:
            ego_size += 1
        self._ego_grid_size = int(ego_size)

        obs_keys = [
            "x", "y", "goal_x", "goal_y", "battery",
            "nearby_obstacles", "nearest_charger_dist", "t",
            "on_conveyor", "patrol_dist", "lidar", "flat",
        ]
        obs_subspaces = {
            "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "goal_x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "goal_y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "battery": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "nearby_obstacles": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "nearest_charger_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "on_conveyor": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "patrol_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "lidar": SpaceSpec(type="vector", shape=(8,), dtype="int32"),
            "flat": SpaceSpec(type="vector", shape=(17,), dtype="float32"),
        }
        if self._enable_ego_grid:
            obs_keys.append("ego_grid")
            obs_subspaces["ego_grid"] = SpaceSpec(
                type="vector",
                shape=(self._ego_grid_size * self._ego_grid_size,),
                dtype="int32",
            )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=obs_keys,
            subspaces=obs_subspaces,
            notes="WarehouseWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=5,
            notes="0=up,1=down,2=left,3=right,4=wait/charge",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._x = 0
        self._y = 0
        self._battery = int(self.params.battery_capacity)
        self._t = 0
        self._done = False
        self._obstacles: Set[Tuple[int, int]] = set()
        self._chargers: Set[Tuple[int, int]] = set()
        # Conveyors: dict mapping (x,y) -> direction index (0=up,1=down,2=left,3=right)
        self._conveyors: Dict[Tuple[int, int], int] = {}
        # Patrol robot state
        self._patrol_route: List[Tuple[int, int]] = []
        self._patrol_idx: int = 0
        self._patrol_pos: Tuple[int, int] = (-1, -1)

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
        self._battery = int(self.params.battery_capacity)
        self._t = 0
        self._done = False
        self._build_layout()
        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": self._seed,
                "width": int(self.params.width),
                "height": int(self.params.height),
                "goal_x": int(self._goal_x()),
                "goal_y": int(self._goal_y()),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = max(0, min(4, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": int(a), "reached_goal": False}
        done = False

        if a == 4:
            if (self._x, self._y) in self._chargers:
                old = self._battery
                self._battery = min(int(self.params.battery_capacity), self._battery + int(self.params.charge_rate))
                if self._battery > old:
                    reward += float(self.params.charge_reward)
                    info["charged"] = True
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
                self._battery = max(0, self._battery - int(self.params.battery_drain))

        # Conveyor belt: push agent one cell in the belt's direction
        pos = (self._x, self._y)
        if pos in self._conveyors:
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            cdx, cdy = directions[self._conveyors[pos]]
            cx, cy = self._x + cdx, self._y + cdy
            if (0 <= cx < self.params.width and 0 <= cy < self.params.height
                    and (cx, cy) not in self._obstacles):
                self._x, self._y = cx, cy
                info["conveyed"] = True

        # Patrol robot: advance along route and check collision
        if self._patrol_route:
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_route)
            self._patrol_pos = self._patrol_route[self._patrol_idx]
            if (self._x, self._y) == self._patrol_pos:
                reward += float(self.params.patrol_penalty)
                info["hit_patrol"] = True
                # Reset to start
                self._x = self._start_x()
                self._y = self._start_y()

        if self._battery <= 0 and (self._x, self._y) != (self._goal_x(), self._goal_y()):
            reward += float(self.params.battery_fail_penalty)
            info["battery_death"] = True
            done = True
        elif (self._x, self._y) == (self._goal_x(), self._goal_y()):
            reward += float(self.params.goal_reward)
            info["reached_goal"] = True
            done = True

        truncated = bool(self._t >= int(self.params.max_steps) and not done)
        self._done = bool(done or truncated)
        info["t"] = int(self._t)
        info["battery"] = int(self._battery)
        info["x"] = int(self._x)
        info["y"] = int(self._y)
        info["goal_x"] = int(self._goal_x())
        info["goal_y"] = int(self._goal_y())

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
                if (x, y) in self._chargers:
                    ch = "C"
                if x == self._goal_x() and y == self._goal_y():
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
            "battery": int(self._battery),
            "t": int(self._t),
            "done": bool(self._done),
            "obstacles": [[x, y] for (x, y) in sorted(self._obstacles)],
            "chargers": [[x, y] for (x, y) in sorted(self._chargers)],
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._x = max(0, min(self.params.width - 1, int(state.get("x", self._x))))
        self._y = max(0, min(self.params.height - 1, int(state.get("y", self._y))))
        self._battery = max(0, min(int(self.params.battery_capacity), int(state.get("battery", self._battery))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        obs = state.get("obstacles")
        if isinstance(obs, list):
            self._obstacles = set((int(p[0]), int(p[1])) for p in obs if isinstance(p, list) and len(p) == 2)
        ch = state.get("chargers")
        if isinstance(ch, list):
            self._chargers = set((int(p[0]), int(p[1])) for p in ch if isinstance(p, list) and len(p) == 2)

    def _make_obs(self) -> JSONValue:
        nearby = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (self._x + dx, self._y + dy) in self._obstacles:
                nearby += 1
        nearest = 999
        for cx, cy in self._chargers:
            d = abs(self._x - cx) + abs(self._y - cy)
            nearest = min(nearest, d)
        # Conveyor awareness
        on_conv = 1 if (self._x, self._y) in self._conveyors else 0
        # Patrol robot distance
        patrol_dist = -1
        if self._patrol_route:
            patrol_dist = abs(self._x - self._patrol_pos[0]) + abs(self._y - self._patrol_pos[1])

        # Lidar implementation
        lidar = []
        scanner_directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        rng = int(self.params.lidar_range)
        for dx, dy in scanner_directions:
            dist = rng
            for k in range(1, rng + 1):
                tx, ty = self._x + dx * k, self._y + dy * k
                if not (0 <= tx < self.params.width and 0 <= ty < self.params.height):
                    dist = k
                    break
                if (tx, ty) in self._obstacles:
                    dist = k
                    break
            lidar.append(dist)

        flat = [
                float(self._x), float(self._y), float(self._goal_x()),
                float(self._goal_y()), float(self._battery), float(nearby),
                float(self._t), float(on_conv), float(patrol_dist),
        ] + [float(d) for d in lidar]

        out = {
            "x": int(self._x),
            "y": int(self._y),
            "goal_x": int(self._goal_x()),
            "goal_y": int(self._goal_y()),
            "battery": int(self._battery),
            "nearby_obstacles": int(nearby),
            "nearest_charger_dist": int(nearest if nearest < 999 else -1),
            "t": int(self._t),
            "on_conveyor": int(on_conv),
            "patrol_dist": int(patrol_dist),
            "lidar": lidar,
            "flat": flat,
        }
        if self._enable_ego_grid:
            out["ego_grid"] = self._ego_grid_from_lidar(lidar=lidar)
        return out

    def _ego_grid_from_lidar(self, *, lidar: List[int]) -> List[int]:
        """
        Egocentric occupancy inferred from lidar rays.
        Cell encoding:
          0 = free/unknown
          1 = obstacle/wall
          2 = goal (if goal is inside local window)
        """
        size = int(self._ego_grid_size)
        radius = size // 2
        grid = [[0 for _ in range(size)] for _ in range(size)]
        dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ]
        for i, (dx, dy) in enumerate(dirs):
            dist = max(1, int(lidar[i] if i < len(lidar) else (radius + 1)))
            for k in range(1, radius + 1):
                lx = radius + dx * k
                ly = radius + dy * k
                if lx < 0 or lx >= size or ly < 0 or ly >= size:
                    continue
                if k >= dist:
                    grid[ly][lx] = 1
                    break
        dgx = int(self._goal_x()) - int(self._x)
        dgy = int(self._goal_y()) - int(self._y)
        if -radius <= dgx <= radius and -radius <= dgy <= radius:
            gx = dgx + radius
            gy = dgy + radius
            if grid[gy][gx] == 0:
                grid[gy][gx] = 2
        return [int(v) for row in grid for v in row]

    def _path_exists(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """BFS check that a walkable path exists from start to goal."""
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
        self._chargers = set()
        self._conveyors = {}
        self._patrol_route = []
        self._patrol_idx = 0
        self._patrol_pos = (-1, -1)

        for pos in [(2, 2), (5, 5), (2, 5), (5, 2)]:
            if 0 <= pos[0] < self.params.width and 0 <= pos[1] < self.params.height:
                self._chargers.add(pos)
        start = (self._start_x(), self._start_y())
        goal = (self._goal_x(), self._goal_y())

        # Place obstacles with BFS path validation
        attempts = 0
        max_attempts = self.params.obstacle_count * 5
        while len(self._obstacles) < int(self.params.obstacle_count) and attempts < max_attempts:
            attempts += 1
            x = self._rng.randint(0, self.params.width - 1)
            y = self._rng.randint(0, self.params.height - 1)
            p = (x, y)
            if p == start or p == goal or p in self._chargers:
                continue
            self._obstacles.add(p)
            if not self._path_exists(start, goal):
                self._obstacles.discard(p)

        # Place conveyor belts on open cells
        open_cells = [
            (x, y) for x in range(self.params.width) for y in range(self.params.height)
            if (x, y) not in self._obstacles and (x, y) not in self._chargers
            and (x, y) != start and (x, y) != goal
        ]
        self._rng.shuffle(open_cells)
        for i in range(min(self.params.conveyor_count, len(open_cells))):
            direction = self._rng.randint(0, 3)  # 0=up,1=down,2=left,3=right
            self._conveyors[open_cells[i]] = direction

        # Build patrol robot route (L-shaped path through the grid)
        if self.params.patrol_robot:
            mid_x = self.params.width // 2
            mid_y = self.params.height // 2
            route: List[Tuple[int, int]] = []
            # Horizontal sweep
            for x in range(1, self.params.width - 1):
                if (x, mid_y) not in self._obstacles:
                    route.append((x, mid_y))
            # Vertical sweep
            for y in range(1, self.params.height - 1):
                if (mid_x, y) not in self._obstacles and (mid_x, y) not in route:
                    route.append((mid_x, y))
            if len(route) >= 2:
                self._patrol_route = route
                self._patrol_idx = 0
                self._patrol_pos = route[0]

    def _start_x(self) -> int:
        return max(0, min(self.params.width - 1, int(self.params.start_x)))

    def _start_y(self) -> int:
        return max(0, min(self.params.height - 1, int(self.params.start_y)))

    def _goal_x(self) -> int:
        gx = int(self.params.goal_x)
        if gx < 0:
            gx = self.params.width - 1
        return max(0, min(self.params.width - 1, gx))

    def _goal_y(self) -> int:
        gy = int(self.params.goal_y)
        if gy < 0:
            gy = self.params.height - 1
        return max(0, min(self.params.height - 1, gy))


class WarehouseWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "grid", "resource", "obstacles", "warehouse"]

    def create(self, spec: VerseSpec) -> Verse:
        return WarehouseWorldVerse(spec)
