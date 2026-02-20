"""
verses/labyrinth_world.py

Complex labyrinth with procedurally generated maze walls, pit hazards,
sweeping laser beams, battery management, action noise, and partial
observability via limited vision radius.
"""

from __future__ import annotations

import dataclasses
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class LabyrinthParams:
    width: int = 15
    height: int = 11
    max_steps: int = 200
    step_penalty: float = -0.05
    wall_penalty: float = -0.20
    pit_penalty: float = -25.0
    laser_penalty: float = -18.0
    goal_reward: float = 10.0
    battery_capacity: int = 80
    battery_drain: int = 1
    charge_rate: int = 8
    charge_reward: float = 0.3
    # Maze generation: passage_ratio controls openness (0.0 = full maze, 1.0 = no walls)
    passage_ratio: float = 0.30
    # Number of pit cells and laser emitters
    pit_count: int = 6
    laser_count: int = 3
    # Charger count
    charger_count: int = 3
    # Action noise: probability that the executed action is randomly perturbed
    action_noise: float = 0.08
    # Vision radius for partial observability (-1 = full observability)
    vision_radius: int = 3
    # Whether lasers sweep (change position each step)
    lasers_sweep: bool = True
    # ADR (automatic domain randomization) — not used here, but kept for compat
    adr_enabled: bool = False


def _generate_maze(
    width: int, height: int, rng: random.Random, passage_ratio: float
) -> Set[Tuple[int, int]]:
    """
    Generate maze walls using randomized DFS (recursive backtracker).
    Returns a set of (x, y) wall positions on a grid.

    The maze is generated on a sub-grid where odd cells are passages and even
    cells are walls, then mapped onto the full grid.  `passage_ratio` controls
    how many extra walls are removed after generation to create shortcuts.
    """
    # Work in maze coordinates (half-size grid)
    mw = max(2, (width - 1) // 2)
    mh = max(2, (height - 1) // 2)

    # All cells start as walls
    walls: Set[Tuple[int, int]] = set()
    for x in range(width):
        for y in range(height):
            walls.add((x, y))

    # Carve passages from maze cells
    visited: Set[Tuple[int, int]] = set()
    stack: List[Tuple[int, int]] = []

    start_mx, start_my = 0, 0
    visited.add((start_mx, start_my))
    stack.append((start_mx, start_my))
    # Map maze cell to grid cell
    gx0, gy0 = 1 + start_mx * 2, 1 + start_my * 2
    if (gx0, gy0) in walls:
        walls.discard((gx0, gy0))

    while stack:
        mx, my = stack[-1]
        neighbors = []
        for dmx, dmy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nmx, nmy = mx + dmx, my + dmy
            if 0 <= nmx < mw and 0 <= nmy < mh and (nmx, nmy) not in visited:
                neighbors.append((nmx, nmy, dmx, dmy))

        if not neighbors:
            stack.pop()
            continue

        nmx, nmy, dmx, dmy = rng.choice(neighbors)
        visited.add((nmx, nmy))
        stack.append((nmx, nmy))

        # Carve the passage cell and the wall between
        gx_new = 1 + nmx * 2
        gy_new = 1 + nmy * 2
        gx_between = 1 + mx * 2 + dmx
        gy_between = 1 + my * 2 + dmy
        walls.discard((gx_new, gy_new))
        walls.discard((gx_between, gy_between))

    # Remove extra walls to create shortcuts based on passage_ratio
    interior_walls = [
        (x, y) for x, y in walls if 1 <= x < width - 1 and 1 <= y < height - 1
    ]
    rng.shuffle(interior_walls)
    remove_count = int(len(interior_walls) * max(0.0, min(1.0, passage_ratio)))
    for i in range(remove_count):
        walls.discard(interior_walls[i])

    # Ensure border stays as walls
    border: Set[Tuple[int, int]] = set()
    for x in range(width):
        border.add((x, 0))
        border.add((x, height - 1))
    for y in range(height):
        border.add((0, y))
        border.add((width - 1, y))
    walls |= border

    return walls


def _find_path_exists(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    blocked: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> bool:
    """BFS to check if a path exists from start to goal avoiding walls and blocked cells."""
    if start == goal:
        return True
    visited: Set[Tuple[int, int]] = {start}
    queue: deque[Tuple[int, int]] = deque([start])
    all_blocked = walls | blocked
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) == goal:
                return True
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in all_blocked and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


class LabyrinthWorldVerse:
    """
    Complex labyrinth environment with:
    - Procedurally generated maze walls (recursive backtracker + shortcut removal)
    - Pit hazards (instant high penalty, episode ends)
    - Sweeping laser beams (move each step, high penalty on contact)
    - Battery management with charger stations
    - Action noise (stochastic slip to adjacent direction)
    - Partial observability (limited vision radius)
    - 5 actions: up, down, left, right, wait/charge
    """

    def __init__(self, spec: VerseSpec):
        self.spec = spec
        cfg = spec.params
        self.params = LabyrinthParams(
            width=max(7, int(cfg.get("width", 15))),
            height=max(7, int(cfg.get("height", 11))),
            max_steps=max(20, int(cfg.get("max_steps", 200))),
            step_penalty=float(cfg.get("step_penalty", -0.05)),
            wall_penalty=float(cfg.get("wall_penalty", -0.20)),
            pit_penalty=float(cfg.get("pit_penalty", -25.0)),
            laser_penalty=float(cfg.get("laser_penalty", -18.0)),
            goal_reward=float(cfg.get("goal_reward", 10.0)),
            battery_capacity=max(10, int(cfg.get("battery_capacity", 80))),
            battery_drain=max(0, int(cfg.get("battery_drain", 1))),
            charge_rate=max(0, int(cfg.get("charge_rate", 8))),
            charge_reward=float(cfg.get("charge_reward", 0.3)),
            passage_ratio=max(0.0, min(1.0, float(cfg.get("passage_ratio", 0.30)))),
            pit_count=max(0, int(cfg.get("pit_count", 6))),
            laser_count=max(0, int(cfg.get("laser_count", 3))),
            charger_count=max(0, int(cfg.get("charger_count", 3))),
            action_noise=max(0.0, min(1.0, float(cfg.get("action_noise", 0.08)))),
            vision_radius=int(cfg.get("vision_radius", 3)),
            lasers_sweep=bool(cfg.get("lasers_sweep", True)),
            adr_enabled=bool(cfg.get("adr_enabled", False)),
        )

        obs_keys = [
            "x", "y", "goal_x", "goal_y", "battery", "t",
            "nearby_walls", "nearby_pits", "nearest_charger_dist",
            "laser_nearby", "vision_cells",
        ]
        self.observation_space = SpaceSpec(
            type="dict",
            keys=obs_keys,
            subspaces={
                "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "battery": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearby_walls": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearby_pits": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "nearest_charger_dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "laser_nearby": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "vision_cells": SpaceSpec(type="vector", shape=(1,), dtype="int32",
                                          notes="Number of visible open cells within vision radius"),
            },
            notes="LabyrinthWorld obs dict with partial observability",
        )
        self.action_space = SpaceSpec(
            type="discrete", n=5,
            notes="0=up,1=down,2=left,3=right,4=wait/charge",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None

        # Layout state (regenerated on reset)
        self._walls: Set[Tuple[int, int]] = set()
        self._pits: Set[Tuple[int, int]] = set()
        self._chargers: Set[Tuple[int, int]] = set()
        # Lasers: list of (emitter_pos, direction_index, current_beam_cells)
        self._laser_emitters: List[Tuple[Tuple[int, int], int]] = []
        self._laser_cells: Set[Tuple[int, int]] = set()

        # Agent state
        self._x = 1
        self._y = 1
        self._battery = self.params.battery_capacity
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
        self._battery = self.params.battery_capacity

        # Generate maze
        self._walls = _generate_maze(
            self.params.width, self.params.height, self._rng, self.params.passage_ratio
        )

        # Start and goal positions (always in open space)
        self._x = 1
        self._y = 1
        self._walls.discard((1, 1))
        goal_x = self.params.width - 2
        goal_y = self.params.height - 2
        self._walls.discard((goal_x, goal_y))
        # Ensure path carving around start and goal
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            sx, sy = 1 + dx, 1 + dy
            if 0 < sx < self.params.width - 1 and 0 < sy < self.params.height - 1:
                self._walls.discard((sx, sy))
            gx2, gy2 = goal_x + dx, goal_y + dy
            if 0 < gx2 < self.params.width - 1 and 0 < gy2 < self.params.height - 1:
                self._walls.discard((gx2, gy2))

        # Place pits in open cells (not near start or goal)
        self._pits = set()
        open_cells = self._get_open_cells(exclude_near={(1, 1), (goal_x, goal_y)}, min_dist=3)
        self._rng.shuffle(open_cells)
        pit_count = min(self.params.pit_count, len(open_cells) // 3)
        for i in range(pit_count):
            candidate = open_cells[i]
            # Don't place if it would block the only path
            if _find_path_exists((1, 1), (goal_x, goal_y), self._walls, self._pits | {candidate},
                                 self.params.width, self.params.height):
                self._pits.add(candidate)

        # Place chargers
        self._chargers = set()
        remaining = [c for c in open_cells if c not in self._pits]
        self._rng.shuffle(remaining)
        for i in range(min(self.params.charger_count, len(remaining))):
            self._chargers.add(remaining[i])

        # Place laser emitters along walls with a direction
        self._laser_emitters = []
        wall_adjacent = self._find_wall_emitter_spots()
        self._rng.shuffle(wall_adjacent)
        for i in range(min(self.params.laser_count, len(wall_adjacent))):
            pos, direction = wall_adjacent[i]
            self._laser_emitters.append((pos, direction))
        self._update_laser_beams()

        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "width": self.params.width,
            "height": self.params.height,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "pit_count": len(self._pits),
            "laser_count": len(self._laser_emitters),
            "charger_count": len(self._chargers),
            "wall_count": len(self._walls),
        }
        return ResetResult(obs=self._make_obs(), info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() after done"})

        a = max(0, min(4, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": a, "reached_goal": False}
        done = False

        goal_x = self.params.width - 2
        goal_y = self.params.height - 2

        # Action noise: with some probability, slip to an adjacent direction
        if a < 4 and self.params.action_noise > 0.0 and self._rng.random() < self.params.action_noise:
            # Slip to perpendicular direction
            if a in (0, 1):
                a = self._rng.choice([2, 3])
            else:
                a = self._rng.choice([0, 1])
            info["slipped"] = True

        if a == 4:
            # Wait / charge
            if (self._x, self._y) in self._chargers:
                old = self._battery
                self._battery = min(self.params.battery_capacity, self._battery + self.params.charge_rate)
                if self._battery > old:
                    reward += float(self.params.charge_reward)
                    info["charged"] = True
            # Standing still still drains battery
            self._battery = max(0, self._battery - self.params.battery_drain)
        else:
            # Movement
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
            elif (nx, ny) in self._walls:
                reward += float(self.params.wall_penalty)
                info["hit_wall"] = True
            else:
                self._x, self._y = nx, ny
                self._battery = max(0, self._battery - self.params.battery_drain)

        # Check pit
        if (self._x, self._y) in self._pits:
            reward += float(self.params.pit_penalty)
            info["fell_pit"] = True
            done = True

        # Sweep lasers (they move every step if enabled)
        if self.params.lasers_sweep:
            self._sweep_lasers()
        self._update_laser_beams()

        # Check laser hit
        if not done and (self._x, self._y) in self._laser_cells:
            reward += float(self.params.laser_penalty)
            info["hit_laser"] = True
            done = True

        # Battery death
        if not done and self._battery <= 0 and (self._x, self._y) != (goal_x, goal_y):
            info["battery_depleted"] = True
            done = True

        # Goal reached
        if not done and self._x == goal_x and self._y == goal_y:
            reward += float(self.params.goal_reward)
            info["reached_goal"] = True
            done = True

        truncated = bool(self._t >= self.params.max_steps and not done)
        self._done = bool(done or truncated)

        info["t"] = self._t
        info["x"] = self._x
        info["y"] = self._y
        info["battery"] = self._battery
        info["goal_x"] = goal_x
        info["goal_y"] = goal_y

        return StepResult(
            obs=self._make_obs(),
            reward=float(reward),
            done=bool(done),
            truncated=bool(truncated),
            info=info,
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _goal_x(self) -> int:
        return self.params.width - 2

    def _goal_y(self) -> int:
        return self.params.height - 2

    def _get_open_cells(
        self,
        exclude_near: Set[Tuple[int, int]] | None = None,
        min_dist: int = 2,
    ) -> List[Tuple[int, int]]:
        """Return open (non-wall, non-start/goal) cells, excluding those near given positions."""
        open_cells: List[Tuple[int, int]] = []
        for x in range(1, self.params.width - 1):
            for y in range(1, self.params.height - 1):
                if (x, y) in self._walls:
                    continue
                if exclude_near:
                    too_close = False
                    for ex, ey in exclude_near:
                        if abs(x - ex) + abs(y - ey) < min_dist:
                            too_close = True
                            break
                    if too_close:
                        continue
                open_cells.append((x, y))
        return open_cells

    def _find_wall_emitter_spots(self) -> List[Tuple[Tuple[int, int], int]]:
        """Find wall cells adjacent to open space that can emit a laser beam."""
        spots: List[Tuple[Tuple[int, int], int]] = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for wx, wy in self._walls:
            if wx == 0 or wx == self.params.width - 1 or wy == 0 or wy == self.params.height - 1:
                continue  # Skip border walls
            for di, (dx, dy) in enumerate(directions):
                ax, ay = wx + dx, wy + dy
                if 0 <= ax < self.params.width and 0 <= ay < self.params.height:
                    if (ax, ay) not in self._walls:
                        spots.append(((wx, wy), di))
                        break  # One direction per wall cell
        return spots

    def _update_laser_beams(self) -> None:
        """Compute current laser beam cells from emitters."""
        self._laser_cells = set()
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for (ex, ey), di in self._laser_emitters:
            dx, dy = directions[di % 4]
            cx, cy = ex + dx, ey + dy
            for _ in range(max(self.params.width, self.params.height)):
                if not (0 <= cx < self.params.width and 0 <= cy < self.params.height):
                    break
                if (cx, cy) in self._walls:
                    break
                self._laser_cells.add((cx, cy))
                cx += dx
                cy += dy

    def _sweep_lasers(self) -> None:
        """Rotate laser directions periodically to create sweeping hazards."""
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_emitters: List[Tuple[Tuple[int, int], int]] = []
        for (ex, ey), di in self._laser_emitters:
            # Rotate direction every 4 steps
            if self._t % 4 == 0:
                # Try next direction; skip if it immediately hits a wall
                for offset in range(1, 5):
                    new_di = (di + offset) % 4
                    dx, dy = directions[new_di]
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < self.params.width and 0 <= ny < self.params.height and (nx, ny) not in self._walls:
                        di = new_di
                        break
            new_emitters.append(((ex, ey), di))
        self._laser_emitters = new_emitters

    def _make_obs(self) -> Dict[str, Any]:
        goal_x = self._goal_x()
        goal_y = self._goal_y()

        # Count adjacent walls and pits
        nearby_walls = 0
        nearby_pits = 0
        laser_nearby = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self._x + dx, self._y + dy
            if not (0 <= nx < self.params.width and 0 <= ny < self.params.height) or (nx, ny) in self._walls:
                nearby_walls += 1
            if (nx, ny) in self._pits:
                nearby_pits += 1
            if (nx, ny) in self._laser_cells:
                laser_nearby += 1

        # Also check diagonals for pits and lasers (wider awareness)
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = self._x + dx, self._y + dy
            if (nx, ny) in self._pits:
                nearby_pits += 1
            if (nx, ny) in self._laser_cells:
                laser_nearby += 1

        # Nearest charger distance
        nearest_charger = -1
        for cx, cy in self._chargers:
            d = abs(self._x - cx) + abs(self._y - cy)
            if nearest_charger < 0 or d < nearest_charger:
                nearest_charger = d

        # Vision: count open cells within vision radius (partial observability proxy)
        vision_cells = 0
        vr = self.params.vision_radius
        if vr < 0:
            # Full observability
            vision_cells = (self.params.width * self.params.height) - len(self._walls)
        else:
            for vdx in range(-vr, vr + 1):
                for vdy in range(-vr, vr + 1):
                    if abs(vdx) + abs(vdy) > vr:
                        continue
                    vx, vy = self._x + vdx, self._y + vdy
                    if 0 <= vx < self.params.width and 0 <= vy < self.params.height:
                        if (vx, vy) not in self._walls:
                            vision_cells += 1

        return {
            "x": int(self._x),
            "y": int(self._y),
            "goal_x": int(goal_x),
            "goal_y": int(goal_y),
            "battery": int(self._battery),
            "t": int(self._t),
            "nearby_walls": int(nearby_walls),
            "nearby_pits": int(nearby_pits),
            "nearest_charger_dist": int(nearest_charger),
            "laser_nearby": int(laser_nearby),
            "vision_cells": int(vision_cells),
        }


class LabyrinthWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "grid", "labyrinth", "maze", "dynamic_hazards"]

    def create(self, spec: VerseSpec) -> Verse:
        return LabyrinthWorldVerse(spec)
