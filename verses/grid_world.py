"""
verses/grid_world.py

2D grid world with random obstacles, ice patches, and teleporter pads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


def hash_spec(spec: VerseSpec) -> str:
    # ... (existing implementation)
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class GridWorldParams:
    start_x: int = 0
    start_y: int = 0
    goal_x: int = -1
    goal_y: int = -1
    width: int = 5
    height: int = 5
    max_steps: int = 50
    step_penalty: float = -0.01
    # Obstacles: static blockers the agent cannot walk through
    obstacle_count: int = 4
    obstacle_penalty: float = -0.15
    # Ice patches: agent slides in move direction until hitting a wall/obstacle
    ice_count: int = 2
    # Teleporter pads: warp agent to a paired location
    teleporter_pairs: int = 1


class GridWorldVerse:
    """
    2D grid with a fixed goal at bottom-right.
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in GridWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = GridWorldParams(
            start_x=int(self.spec.params.get("start_x", 0)),
            start_y=int(self.spec.params.get("start_y", 0)),
            goal_x=int(self.spec.params.get("goal_x", -1)),
            goal_y=int(self.spec.params.get("goal_y", -1)),
            width=max(3, int(self.spec.params.get("width", 5))),
            height=max(3, int(self.spec.params.get("height", 5))),
            max_steps=max(5, int(self.spec.params.get("max_steps", 50))),
            step_penalty=float(self.spec.params.get("step_penalty", -0.01)),
            obstacle_count=max(0, int(self.spec.params.get("obstacle_count", 4))),
            obstacle_penalty=float(self.spec.params.get("obstacle_penalty", -0.15)),
            ice_count=max(0, int(self.spec.params.get("ice_count", 2))),
            teleporter_pairs=max(0, int(self.spec.params.get("teleporter_pairs", 1))),
        )

        self._enable_ego_grid = bool(self.spec.params.get("enable_ego_grid", False))
        ego_size = int(self.spec.params.get("ego_grid_size", 5))
        if ego_size < 3:
            ego_size = 3
        if ego_size % 2 == 0:
            ego_size += 1
        self._ego_grid_size = int(ego_size)

        obs_keys = ["x", "y", "goal_x", "goal_y", "t", "nearby_obstacles",
                    "on_ice", "on_teleporter"]
        obs_subspaces = {
            "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "goal_x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "goal_y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "nearby_obstacles": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "on_ice": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "on_teleporter": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
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
            notes="GridWorld obs dict with obstacles, ice, and teleporters",
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
        self._obstacles: Set[Tuple[int, int]] = set()
        self._ice: Set[Tuple[int, int]] = set()
        # Teleporters: dict mapping position to its paired destination
        self._teleporters: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    # ... (rest of the methods are unchanged)
    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        self._x = max(0, min(self.params.width - 1, int(self.params.start_x)))
        self._y = max(0, min(self.params.height - 1, int(self.params.start_y)))
        self._t = 0
        self._done = False

        start = (self._x, self._y)
        goal = (self._goal_x(), self._goal_y())

        # Generate random obstacles ensuring path to goal exists
        self._obstacles = set()
        all_cells = [
            (x, y) for x in range(self.params.width)
            for y in range(self.params.height)
            if (x, y) != start and (x, y) != goal
        ]
        self._rng.shuffle(all_cells)
        for cell in all_cells:
            if len(self._obstacles) >= self.params.obstacle_count:
                break
            self._obstacles.add(cell)
            if not self._path_exists(start, goal):
                self._obstacles.discard(cell)

        # Place ice patches on open cells
        self._ice = set()
        open_cells = [c for c in all_cells if c not in self._obstacles and c != start and c != goal]
        self._rng.shuffle(open_cells)
        for i in range(min(self.params.ice_count, len(open_cells))):
            self._ice.add(open_cells[i])

        # Place teleporter pairs on remaining open cells
        self._teleporters = {}
        tp_candidates = [c for c in open_cells if c not in self._ice]
        self._rng.shuffle(tp_candidates)
        for i in range(self.params.teleporter_pairs):
            if len(tp_candidates) >= 2:
                a = tp_candidates.pop()
                b = tp_candidates.pop()
                self._teleporters[a] = b
                self._teleporters[b] = a

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "width": self.params.width,
            "height": self.params.height,
            "start_x": self.params.start_x,
            "start_y": self.params.start_y,
            "goal_x": self._goal_x(),
            "goal_y": self._goal_y(),
            "obstacle_count": len(self._obstacles),
            "ice_count": len(self._ice),
            "teleporter_pairs": len(self._teleporters) // 2,
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
        dx, dy = 0, 0
        if a == 0:
            dy = -1
        elif a == 1:
            dy = 1
        elif a == 2:
            dx = -1
        elif a == 3:
            dx = 1
        else:
            raise ValueError("GridWorld action must be 0..3")

        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"reached_goal": False}

        # Try to move
        nx, ny = self._x + dx, self._y + dy
        if not (0 <= nx < self.params.width and 0 <= ny < self.params.height):
            # Hit boundary — stay in place
            info["hit_wall"] = True
        elif (nx, ny) in self._obstacles:
            reward += float(self.params.obstacle_penalty)
            info["hit_obstacle"] = True
        else:
            self._x, self._y = nx, ny

        # Ice patch: slide in move direction until hitting wall/obstacle/edge
        if (self._x, self._y) in self._ice:
            info["slid_on_ice"] = True
            while True:
                sx, sy = self._x + dx, self._y + dy
                if not (0 <= sx < self.params.width and 0 <= sy < self.params.height):
                    break
                if (sx, sy) in self._obstacles:
                    break
                self._x, self._y = sx, sy
                if (self._x, self._y) not in self._ice:
                    break  # Stop sliding when leaving ice

        # Teleporter: warp to paired location
        pos = (self._x, self._y)
        if pos in self._teleporters:
            dest = self._teleporters[pos]
            self._x, self._y = dest
            info["teleported"] = True

        goal_x = self._goal_x()
        goal_y = self._goal_y()
        reached = (self._x == goal_x and self._y == goal_y)
        truncated = self._t >= self.params.max_steps and not reached

        if reached:
            reward = 1.0
        done = bool(reached)
        self._done = bool(done or truncated)

        info["t"] = self._t
        info["x"] = self._x
        info["y"] = self._y
        info["reached_goal"] = bool(reached)

        return StepResult(
            obs=self._make_obs(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def _path_exists(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """BFS check that a path exists from start to goal avoiding obstacles."""
        if start == goal:
            return True
        visited: Set[Tuple[int, int]] = {start}
        queue: deque[Tuple[int, int]] = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nnx, nny = cx + ddx, cy + ddy
                if (nnx, nny) == goal:
                    return True
                if (0 <= nnx < self.params.width and 0 <= nny < self.params.height
                        and (nnx, nny) not in self._obstacles and (nnx, nny) not in visited):
                    visited.add((nnx, nny))
                    queue.append((nnx, nny))
        return False

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
        w = self.params.width
        h = self.params.height
        goal_x = self._goal_x()
        goal_y = self._goal_y()
        rows = []
        for y in range(h):
            row = []
            for x in range(w):
                if x == self._x and y == self._y:
                    row.append("A")
                elif x == goal_x and y == goal_y:
                    row.append("G")
                elif (x, y) in self._obstacles:
                    row.append("#")
                elif (x, y) in self._ice:
                    row.append("~")
                elif (x, y) in self._teleporters:
                    row.append("T")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)

    def _render_rgb_array(self) -> List[List[List[int]]]:
        """
        Lightweight RGB frame (H x W x 3) using Python lists.
        Colors: bg=white, goal=green, agent=blue, obstacle=dark gray,
        ice=cyan, teleporter=purple.
        """
        w = self.params.width
        h = self.params.height
        goal_x = self._goal_x()
        goal_y = self._goal_y()
        bg = [255, 255, 255]
        goal_c = [60, 200, 80]
        agent_c = [60, 120, 240]
        obs_c = [80, 80, 80]
        ice_c = [160, 230, 255]
        tp_c = [180, 100, 220]
        frame: List[List[List[int]]] = []
        for y in range(h):
            row: List[List[int]] = []
            for x in range(w):
                px = list(bg)
                if (x, y) in self._obstacles:
                    px = list(obs_c)
                elif (x, y) in self._ice:
                    px = list(ice_c)
                elif (x, y) in self._teleporters:
                    px = list(tp_c)
                if x == goal_x and y == goal_y:
                    px = list(goal_c)
                if x == self._x and y == self._y:
                    px = list(agent_c)
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
        # Count adjacent obstacles
        nearby_obs = 0
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self._x + ddx, self._y + ddy
            if not (0 <= nx < self.params.width and 0 <= ny < self.params.height):
                nearby_obs += 1  # Boundary counts as obstacle
            elif (nx, ny) in self._obstacles:
                nearby_obs += 1
        out = {
            "x": self._x,
            "y": self._y,
            "goal_x": self._goal_x(),
            "goal_y": self._goal_y(),
            "t": self._t,
            "nearby_obstacles": nearby_obs,
            "on_ice": 1 if (self._x, self._y) in self._ice else 0,
            "on_teleporter": 1 if (self._x, self._y) in self._teleporters else 0,
        }
        if self._enable_ego_grid:
            out["ego_grid"] = self._ego_grid_flat()
        return out

    def _ego_grid_flat(self) -> List[int]:
        """
        Egocentric occupancy grid flattened row-major.
        Cell encoding:
          0 = free
          1 = obstacle / out of bounds
          2 = goal
        """
        size = int(self._ego_grid_size)
        radius = size // 2
        gx = self._goal_x()
        gy = self._goal_y()
        out: List[int] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                wx = self._x + dx
                wy = self._y + dy
                val = 0
                if not (0 <= wx < self.params.width and 0 <= wy < self.params.height):
                    val = 1
                elif (wx, wy) in self._obstacles:
                    val = 1
                elif wx == gx and wy == gy:
                    val = 2
                out.append(int(val))
        return out

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


class GridWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "grid"]

    def create(self, spec: VerseSpec) -> Verse:
        return GridWorldVerse(spec)
