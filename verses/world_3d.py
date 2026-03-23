"""
verses/world_3d.py

3-D Terrain Navigation World.

An agent navigates a procedurally generated heightmap from a start cell to a
goal cell.  Movement is blocked by cliffs (height change > max_climb) and pits
(impassable cells).  Optional patrol adversaries roam the terrain.

Two repo systems are wired in where they genuinely belong:

    core.communication  ->  A MessageBus can be injected via spec.params["event_bus"]
                            to receive world events without modifying this file.
                            Cost when no bus is attached: one None-check per event.
    core.taxonomy       ->  Tags, universe, and memory type are read at render time
                            so the ANSI header stays in sync with the live taxonomy.
                            Cost at training time: zero (render is never called
                            in a training loop).

If you want oracle path-hints, call plan_actions_from_current_state(verse=v)
from core.planner_oracle externally.  The verse already implements export_state
/ import_state, so the oracle works out of the box without baking it in here.

Params
------
width, depth      : grid dimensions (default 10, 10)
max_height        : terrain elevation range 0 .. max_height-1 (default 5)
max_climb         : maximum elevation change per move (default 1)
start_x, start_y  : agent start position (default 0, 0)
goal_x, goal_y    : goal position; -1 -> bottom-right corner (default -1, -1)
max_steps         : episode cap (default 120)
step_penalty      : reward each step (default -0.04)
goal_reward       : reward on reaching goal (default 1.0)
roughness         : terrain ruggedness 0..1 (default 0.4)
pit_density       : fraction of cells that are impassable pits (default 0.05)
stamina_capacity  : stamina pool size, 0 = disabled (default 0)
patrol_count      : number of NPC patrols, 0 = none (default 0)
patrol_penalty    : reward on patrol collision (default -1.0)

Actions (discrete, 4)
---------------------
  0: +x  (east)
  1: -x  (west)
  2: +y  (north)
  3: -y  (south)

Observation dict
----------------
  x, y, z           : agent position (z = terrain height)
  goal_x, goal_y, goal_z
  dx, dy, dz        : signed delta to goal
  nearby_z          : [E, W, N, S] neighbour elevations (-1 = pit/oob)
  t                 : timestep
  stamina           : current stamina (present only when stamina_capacity > 0)
  patrol_nearby     : [E, W, N, S] patrol presence (present only when patrol_count > 0)

Events (MessageBus topics, fired only when a bus is attached)
-------------------------------------------------------------
  world_3d.reset          : {verse, start, goal, terrain_features}
  world_3d.cliff_blocked  : {x, y, z, target_x, target_y, target_z, t}
  world_3d.pit_warning    : {x, y, nearby_pit_direction, t}
  world_3d.goal_reached   : {x, y, steps, total_reward}
  world_3d.stamina_critical : {stamina, t}  -- fires when stamina drops to <= 1
  world_3d.patrol_caught  : {x, y, patrol_idx, t}

ANSI render
-----------
  Line 1: live taxonomy tags + universe + memory type (via core.taxonomy)
  Line 2: stamina bar (when enabled) and patrol count (when enabled)
  Grid:   top-down elevation map; right margin shows avg-z bar per row
  Footer: terrain-feature summary (peaks, valleys, cliff-edges) + step counter
"""

from __future__ import annotations

import dataclasses
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse

from core.communication import Message, MessageBus
from core.taxonomy import memory_type_for_verse, tags_for_verse, universe_for_verse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ELEV_CHARS = [".", "-", "=", "^", "*"]
_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))   # E, W, N, S


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class World3DParams:
    # Terrain
    width: int = 10
    depth: int = 10
    max_height: int = 5
    max_climb: int = 1
    # Positions
    start_x: int = 0
    start_y: int = 0
    goal_x: int = -1     # -1 -> width - 1
    goal_y: int = -1     # -1 -> depth - 1
    # Episode
    max_steps: int = 120
    step_penalty: float = -0.04
    goal_reward: float = 1.0
    # Terrain generation
    roughness: float = 0.4
    pit_density: float = 0.05
    # Optional mechanics (default off)
    stamina_capacity: int = 0      # 0 = disabled
    patrol_count: int = 0          # 0 = no adversaries
    patrol_penalty: float = -1.0   # reward on patrol collision


# ---------------------------------------------------------------------------
# Terrain generation
# ---------------------------------------------------------------------------

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _generate_heightmap(
    rng: random.Random,
    width: int,
    depth: int,
    max_height: int,
    roughness: float,
) -> List[List[int]]:
    """Smoothed random terrain — no numpy required."""
    h = [[rng.randint(0, max_height - 1) for _ in range(depth)] for _ in range(width)]
    smooth_passes = max(1, int(round(roughness * 4)))
    for _ in range(smooth_passes):
        nh = [[0] * depth for _ in range(width)]
        for x in range(width):
            for y in range(depth):
                vals = [h[x][y]]
                for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = x + ddx, y + ddy
                    if 0 <= nx < width and 0 <= ny < depth:
                        vals.append(h[nx][ny])
                nh[x][y] = int(round(sum(vals) / len(vals)))
        h = nh
    return h


def _generate_pits(
    rng: random.Random,
    width: int,
    depth: int,
    pit_density: float,
    exclude: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    pits: Set[Tuple[int, int]] = set()
    for x in range(width):
        for y in range(depth):
            if (x, y) not in exclude and rng.random() < pit_density:
                pits.add((x, y))
    return pits


def _is_reachable(
    heights: List[List[int]],
    pits: Set[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_climb: int,
    width: int,
    depth: int,
) -> bool:
    visited: Set[Tuple[int, int]] = {start}
    queue: deque[Tuple[int, int]] = deque([start])
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            return True
        for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + ddx, cy + ddy
            if (nx, ny) in visited or (nx, ny) in pits:
                continue
            if 0 <= nx < width and 0 <= ny < depth:
                if abs(heights[nx][ny] - heights[cx][cy]) <= max_climb:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False


def _carve_path(
    heights: List[List[int]],
    pits: Set[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_climb: int,
    width: int,
    depth: int,
    max_height: int,
) -> None:
    """BFS ignoring height constraints, then flatten height diffs along the path."""
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    queue: deque[Tuple[int, int]] = deque([start])
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            break
        for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + ddx, cy + ddy
            if (nx, ny) not in parent and (nx, ny) not in pits:
                if 0 <= nx < width and 0 <= ny < depth:
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))
    cell: Optional[Tuple[int, int]] = goal
    while cell is not None and cell != start:
        prev = parent.get(cell)
        if prev is not None:
            px, py = prev
            cx, cy = cell
            diff = heights[cx][cy] - heights[px][py]
            if abs(diff) > max_climb:
                step = max_climb if diff > 0 else -max_climb
                heights[cx][cy] = _clamp(heights[px][py] + step, 0, max_height - 1)
        cell = prev


# ---------------------------------------------------------------------------
# Terrain feature analysis
# ---------------------------------------------------------------------------

@dataclass
class TerrainFeatures:
    peaks: List[Tuple[int, int]]        # local maxima
    valleys: List[Tuple[int, int]]      # local minima
    cliff_edges: List[Tuple[int, int]]  # cells with >= 1 neighbour blocked by max_climb

    def summary(self) -> str:
        parts = []
        if self.peaks:
            n = len(self.peaks)
            parts.append(f"{n} peak{'s' if n != 1 else ''}")
        if self.valleys:
            n = len(self.valleys)
            parts.append(f"{n} valley{'s' if n != 1 else ''}")
        if self.cliff_edges:
            n = len(self.cliff_edges)
            parts.append(f"{n} cliff-edge{'s' if n != 1 else ''}")
        return " · ".join(parts) if parts else "flat terrain"


def _analyze_terrain(
    heights: List[List[int]],
    pits: Set[Tuple[int, int]],
    max_climb: int,
    width: int,
    depth: int,
) -> TerrainFeatures:
    peaks: List[Tuple[int, int]] = []
    valleys: List[Tuple[int, int]] = []
    cliff_edges: List[Tuple[int, int]] = []
    for x in range(width):
        for y in range(depth):
            if (x, y) in pits:
                continue
            z = heights[x][y]
            neighbour_z: List[int] = []
            has_cliff = False
            for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + ddx, y + ddy
                if not (0 <= nx < width and 0 <= ny < depth) or (nx, ny) in pits:
                    continue
                nz = heights[nx][ny]
                neighbour_z.append(nz)
                if abs(nz - z) > max_climb:
                    has_cliff = True
            if has_cliff:
                cliff_edges.append((x, y))
            if neighbour_z:
                if all(nz <= z for nz in neighbour_z):
                    peaks.append((x, y))
                elif all(nz >= z for nz in neighbour_z):
                    valleys.append((x, y))
    return TerrainFeatures(peaks=peaks, valleys=valleys, cliff_edges=cliff_edges)


# ---------------------------------------------------------------------------
# Verse
# ---------------------------------------------------------------------------

class World3DVerse:
    """3-D terrain navigation verse."""

    def __init__(self, spec: VerseSpec) -> None:
        tags = list(spec.tags)
        for t in World3DFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        p = self.spec.params
        self.params = World3DParams(
            width=max(3, int(p.get("width", 10))),
            depth=max(3, int(p.get("depth", 10))),
            max_height=max(2, int(p.get("max_height", 5))),
            max_climb=max(1, int(p.get("max_climb", 1))),
            start_x=int(p.get("start_x", 0)),
            start_y=int(p.get("start_y", 0)),
            goal_x=int(p.get("goal_x", -1)),
            goal_y=int(p.get("goal_y", -1)),
            max_steps=max(1, int(p.get("max_steps", 120))),
            step_penalty=float(p.get("step_penalty", -0.04)),
            goal_reward=float(p.get("goal_reward", 1.0)),
            roughness=max(0.0, min(1.0, float(p.get("roughness", 0.4)))),
            pit_density=max(0.0, min(0.5, float(p.get("pit_density", 0.05)))),
            stamina_capacity=max(0, int(p.get("stamina_capacity", 0))),
            patrol_count=max(0, int(p.get("patrol_count", 0))),
            patrol_penalty=float(p.get("patrol_penalty", -1.0)),
        )

        # Optional MessageBus — inject via spec.params["event_bus"].
        bus_raw = p.get("event_bus")
        self._bus: Optional[MessageBus] = bus_raw if isinstance(bus_raw, MessageBus) else None

        obs_keys: List[str] = [
            "x", "y", "z", "goal_x", "goal_y", "goal_z",
            "dx", "dy", "dz", "nearby_z", "t",
        ]
        obs_subspaces: Dict[str, SpaceSpec] = {
            "x":        SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "y":        SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "z":        SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "goal_x":   SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "goal_y":   SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "goal_z":   SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "dx":       SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "dy":       SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "dz":       SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "nearby_z": SpaceSpec(type="vector", shape=(4,), dtype="int32"),
            "t":        SpaceSpec(type="vector", shape=(1,), dtype="int32"),
        }
        if self.params.stamina_capacity > 0:
            obs_keys.append("stamina")
            obs_subspaces["stamina"] = SpaceSpec(type="vector", shape=(1,), dtype="int32")
        if self.params.patrol_count > 0:
            obs_keys.append("patrol_nearby")
            obs_subspaces["patrol_nearby"] = SpaceSpec(type="vector", shape=(4,), dtype="int32")

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=obs_keys,
            subspaces=obs_subspaces,
            notes="World3D: position, goal, delta, 4-neighbour elevations, timestep",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete", n=4,
            notes="0=+x(E), 1=-x(W), 2=+y(N), 3=-y(S)",
        )

        self._rng = random.Random()
        self._heights: List[List[int]] = []
        self._pits: Set[Tuple[int, int]] = set()
        self._ax = _clamp(self.params.start_x, 0, self.params.width - 1)
        self._ay = _clamp(self.params.start_y, 0, self.params.depth - 1)
        self._t = 0
        self._done = False
        self._stamina = self.params.stamina_capacity
        self._patrols: List[Tuple[int, int]] = []
        self._features = TerrainFeatures([], [], [])
        self._total_reward: float = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _gx(self) -> int:
        return (self.params.width - 1) if self.params.goal_x < 0 else _clamp(
            self.params.goal_x, 0, self.params.width - 1)

    def _gy(self) -> int:
        return (self.params.depth - 1) if self.params.goal_y < 0 else _clamp(
            self.params.goal_y, 0, self.params.depth - 1)

    def _elevation(self, x: int, y: int) -> int:
        """Return elevation at (x, y), or -1 for pit / out-of-bounds."""
        if not (0 <= x < self.params.width and 0 <= y < self.params.depth):
            return -1
        if (x, y) in self._pits:
            return -1
        return int(self._heights[x][y])

    def _agent_z(self) -> int:
        return int(self._heights[self._ax][self._ay])

    def _can_step(self, fx: int, fy: int, tx: int, ty: int, effective_climb: int) -> bool:
        if not (0 <= tx < self.params.width and 0 <= ty < self.params.depth):
            return False
        if (tx, ty) in self._pits:
            return False
        return abs(self._heights[tx][ty] - self._heights[fx][fy]) <= effective_climb

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self._bus is not None:
            self._bus.publish(Message(sender_id="world_3d", topic=topic, payload=payload))

    def _place_patrols(self, exclude: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        candidates = [
            (x, y)
            for x in range(self.params.width)
            for y in range(self.params.depth)
            if (x, y) not in exclude and (x, y) not in self._pits
        ]
        self._rng.shuffle(candidates)
        return candidates[: self.params.patrol_count]

    def _move_patrol(self, px: int, py: int) -> Tuple[int, int]:
        options = [
            (px + ddx, py + ddy)
            for ddx, ddy in _DIRECTIONS
            if self._can_step(px, py, px + ddx, py + ddy, self.params.max_climb)
        ]
        return self._rng.choice(options) if options else (px, py)

    # ------------------------------------------------------------------ #
    # Verse contract
    # ------------------------------------------------------------------ #

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2 ** 31 - 1)
        self._rng = random.Random(int(seed))

    def reset(self) -> ResetResult:
        if self.spec.seed is not None:
            self.seed(self.spec.seed)

        gx, gy = self._gx(), self._gy()
        sx = _clamp(self.params.start_x, 0, self.params.width - 1)
        sy = _clamp(self.params.start_y, 0, self.params.depth - 1)
        exclude: Set[Tuple[int, int]] = {(sx, sy), (gx, gy)}

        for _ in range(8):
            self._heights = _generate_heightmap(
                self._rng, self.params.width, self.params.depth,
                self.params.max_height, self.params.roughness,
            )
            self._pits = _generate_pits(
                self._rng, self.params.width, self.params.depth,
                self.params.pit_density, exclude,
            )
            if _is_reachable(
                self._heights, self._pits, (sx, sy), (gx, gy),
                self.params.max_climb, self.params.width, self.params.depth,
            ):
                break
        else:
            _carve_path(
                self._heights, self._pits, (sx, sy), (gx, gy),
                self.params.max_climb, self.params.width, self.params.depth,
                self.params.max_height,
            )

        self._ax, self._ay = sx, sy
        self._t = 0
        self._done = False
        self._total_reward = 0.0
        self._stamina = self.params.stamina_capacity
        self._features = _analyze_terrain(
            self._heights, self._pits, self.params.max_climb,
            self.params.width, self.params.depth,
        )
        self._patrols = self._place_patrols(exclude)

        self._publish("world_3d.reset", {
            "verse": "world_3d",
            "start": [sx, sy, self._agent_z()],
            "goal": [gx, gy, self._elevation(gx, gy)],
            "terrain_features": self._features.summary(),
        })

        return ResetResult(obs=self._make_obs(), info={
            "start":            [sx, sy, self._agent_z()],
            "goal":             [gx, gy, self._elevation(gx, gy)],
            "width":            self.params.width,
            "depth":            self.params.depth,
            "max_height":       self.params.max_height,
            "terrain_features": self._features.summary(),
        })

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(
                obs=self._make_obs(), reward=0.0, done=True, truncated=False,
                info={"warning": "step() called after done"},
            )

        a = int(action)
        if a not in (0, 1, 2, 3):
            raise ValueError(f"World3D action must be 0-3, got {a}")

        ddx, ddy = _DIRECTIONS[a]
        nx, ny = self._ax + ddx, self._ay + ddy

        # Stamina: exhausted agent cannot climb.
        effective_climb = self.params.max_climb
        if self.params.stamina_capacity > 0 and self._stamina <= 0:
            effective_climb = 0

        info: Dict[str, JSONValue] = {}
        reward = float(self.params.step_penalty)

        if not (0 <= nx < self.params.width and 0 <= ny < self.params.depth):
            info["hit_boundary"] = True
            if self.params.stamina_capacity > 0:
                self._stamina = min(self.params.stamina_capacity, self._stamina + 1)
        elif (nx, ny) in self._pits:
            info["hit_pit"] = True
            self._publish("world_3d.pit_warning", {
                "x": int(self._ax), "y": int(self._ay),
                "nearby_pit_direction": int(a), "t": int(self._t),
            })
            if self.params.stamina_capacity > 0:
                self._stamina = min(self.params.stamina_capacity, self._stamina + 1)
        else:
            new_z = int(self._heights[nx][ny])
            cur_z = self._agent_z()
            if abs(new_z - cur_z) > effective_climb:
                info["blocked_cliff"] = True
                self._publish("world_3d.cliff_blocked", {
                    "x": int(self._ax), "y": int(self._ay), "z": int(cur_z),
                    "target_x": int(nx), "target_y": int(ny), "target_z": int(new_z),
                    "t": int(self._t),
                })
                if self.params.stamina_capacity > 0:
                    self._stamina = min(self.params.stamina_capacity, self._stamina + 1)
            else:
                self._ax, self._ay = nx, ny
                info["moved"] = True
                if self.params.stamina_capacity > 0:
                    cost = 2 if new_z > cur_z else 1
                    self._stamina = max(0, self._stamina - cost)

        self._t += 1

        if self.params.patrol_count > 0:
            self._patrols = [self._move_patrol(px, py) for px, py in self._patrols]

        caught_by_patrol = False
        if self.params.patrol_count > 0 and (self._ax, self._ay) in set(self._patrols):
            caught_by_patrol = True
            patrol_idx = next(
                (i for i, (px, py) in enumerate(self._patrols)
                 if px == self._ax and py == self._ay), 0,
            )
            reward += float(self.params.patrol_penalty)
            info["caught_by_patrol"] = True
            self._publish("world_3d.patrol_caught", {
                "x": int(self._ax), "y": int(self._ay),
                "patrol_idx": int(patrol_idx), "t": int(self._t),
            })

        gx, gy = self._gx(), self._gy()
        reached = (self._ax == gx and self._ay == gy)
        truncated = not reached and not caught_by_patrol and self._t >= self.params.max_steps

        if reached:
            reward = float(self.params.goal_reward)
            info["reached_goal"] = True
            self._total_reward += reward
            self._publish("world_3d.goal_reached", {
                "x": int(self._ax), "y": int(self._ay),
                "steps": int(self._t), "total_reward": float(self._total_reward),
            })
        else:
            self._total_reward += reward

        if self.params.stamina_capacity > 0 and self._stamina <= 1 and not reached:
            self._publish("world_3d.stamina_critical", {
                "stamina": int(self._stamina), "t": int(self._t),
            })

        self._done = reached or truncated or caught_by_patrol
        return StepResult(
            obs=self._make_obs(),
            reward=reward,
            done=bool(reached or caught_by_patrol),
            truncated=bool(truncated),
            info=info,
        )

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode == "human":
            frame = self.render(mode="ansi")
            if frame is not None:
                print(frame)
            return None
        if mode == "rgb_array":
            return self._render_rgb()
        if mode != "ansi":
            return None
        return self._render_ansi()

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        state: Dict[str, JSONValue] = {
            "ax":   int(self._ax),
            "ay":   int(self._ay),
            "t":    int(self._t),
            "done": bool(self._done),
        }
        if self.params.stamina_capacity > 0:
            state["stamina"] = int(self._stamina)
        if self.params.patrol_count > 0:
            state["patrols"] = [[int(px), int(py)] for px, py in self._patrols]
        return state

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._ax = _clamp(int(state.get("ax", self._ax)), 0, self.params.width - 1)
        self._ay = _clamp(int(state.get("ay", self._ay)), 0, self.params.depth - 1)
        self._t  = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        if self.params.stamina_capacity > 0 and "stamina" in state:
            self._stamina = _clamp(int(state["stamina"]), 0, self.params.stamina_capacity)
        if self.params.patrol_count > 0 and "patrols" in state:
            raw = state["patrols"]
            if isinstance(raw, list):
                self._patrols = [
                    (int(entry[0]), int(entry[1]))
                    for entry in raw
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2
                ]

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _make_obs(self) -> JSONValue:
        gx, gy = self._gx(), self._gy()
        az = self._agent_z()
        gz = self._elevation(gx, gy)
        nearby = [self._elevation(self._ax + ddx, self._ay + ddy) for ddx, ddy in _DIRECTIONS]
        obs: Dict[str, Any] = {
            "x":        int(self._ax),
            "y":        int(self._ay),
            "z":        int(az),
            "goal_x":   int(gx),
            "goal_y":   int(gy),
            "goal_z":   int(gz),
            "dx":       int(gx - self._ax),
            "dy":       int(gy - self._ay),
            "dz":       int(gz - az),
            "nearby_z": list(nearby),
            "t":        int(self._t),
        }
        if self.params.stamina_capacity > 0:
            obs["stamina"] = int(self._stamina)
        if self.params.patrol_count > 0:
            patrol_set = set(self._patrols)
            obs["patrol_nearby"] = [
                int((self._ax + ddx, self._ay + ddy) in patrol_set)
                for ddx, ddy in _DIRECTIONS
            ]
        return obs

    # ------------------------------------------------------------------ #
    # ANSI render
    # ------------------------------------------------------------------ #

    def _elev_char(self, x: int, y: int) -> str:
        if (x, y) in self._pits:
            return "X"
        z = int(self._heights[x][y])
        return _ELEV_CHARS[min(z, len(_ELEV_CHARS) - 1)]

    def _stamina_bar(self, bar_width: int = 8) -> str:
        cap = self.params.stamina_capacity
        filled = _clamp(int(round(self._stamina * bar_width / max(1, cap))), 0, bar_width)
        return "#" * filled + "." * (bar_width - filled) + f" {self._stamina}/{cap}"

    def _render_ansi(self) -> str:
        gx, gy = self._gx(), self._gy()
        az = self._agent_z()
        gz = self._elevation(gx, gy)
        patrol_set = set(self._patrols)

        # Line 1: live taxonomy header (render-only, no training cost).
        try:
            vtags = tags_for_verse("world_3d")
            tag_str = "·".join(
                t for t in vtags
                if not t.startswith("universe:") and not t.startswith("memory_")
            )
            uni = universe_for_verse("world_3d")
            mem = memory_type_for_verse("world_3d")
        except Exception:
            tag_str = "navigation·3d·terrain·risk_sensitive"
            uni = "navigation_risk"
            mem = "spatial_procedural"
        line1 = f"World3D  [{tag_str}]  universe:{uni}  memory:{mem}"

        # Line 2: optional mechanics status.
        line2_parts: List[str] = []
        if self.params.stamina_capacity > 0:
            line2_parts.append(f"Stamina:[{self._stamina_bar()}]")
        if self.params.patrol_count > 0:
            line2_parts.append(f"Patrols:{len(self._patrols)}")
        line2 = "   ".join(line2_parts) if line2_parts else ""

        legend = ".=0  -=1  ==2  ^=3  *=4  X=pit  A=agent  G=goal"
        if self.params.patrol_count > 0:
            legend += "  P=patrol"

        lines = [line1]
        if line2:
            lines.append(line2)
        lines.append("+" + "-" * self.params.width + "+  " + legend)

        for y in range(self.params.depth):
            row_chars: List[str] = []
            for x in range(self.params.width):
                on_agent  = (x == self._ax and y == self._ay)
                on_goal   = (x == gx and y == gy)
                on_patrol = (x, y) in patrol_set
                if on_agent and on_goal:
                    row_chars.append("@")
                elif on_agent:
                    row_chars.append("A")
                elif on_goal:
                    row_chars.append("G")
                elif on_patrol:
                    row_chars.append("P")
                else:
                    row_chars.append(self._elev_char(x, y))
            row_elevs = [
                self._heights[x][y] if (x, y) not in self._pits else -1
                for x in range(self.params.width)
            ]
            valid = [z for z in row_elevs if z >= 0]
            avg_z = (sum(valid) // len(valid)) if valid else 0
            lines.append("|" + "".join(row_chars) + f"|  z~{avg_z} {'#' * (avg_z + 1)}")

        lines.append("+" + "-" * self.params.width + "+")
        lines.append(
            f"Terrain: {self._features.summary()}"
            f"   step: {self._t}/{self.params.max_steps}"
            f"   pos:({self._ax},{self._ay},z={az})"
            f"   goal:({gx},{gy},z={gz})"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # rgb_array render
    # ------------------------------------------------------------------ #

    def _render_rgb(self) -> List[List[List[int]]]:
        """Depth x Width x 3.  Blue->teal->green->white gradient; agent=orange, goal=green, pit=black, patrol=red."""
        W, D = self.params.width, self.params.depth
        MH = max(1, self.params.max_height - 1)
        gx, gy = self._gx(), self._gy()
        patrol_set = set(self._patrols)

        def _elev_color(z: int) -> List[int]:
            t = float(z) / float(MH)
            if t < 0.33:
                f = t / 0.33
                return [int(20 + 60 * f), int(60 + 120 * f), int(180 - 80 * f)]
            elif t < 0.66:
                f = (t - 0.33) / 0.33
                return [int(80 + 100 * f), int(180 + 40 * f), int(100 - 60 * f)]
            else:
                f = (t - 0.66) / 0.34
                return [int(180 + 75 * f), int(220 + 35 * f), int(40 + 215 * f)]

        frame: List[List[List[int]]] = []
        for y in range(D):
            row: List[List[int]] = []
            for x in range(W):
                px = [10, 10, 10] if (x, y) in self._pits else _elev_color(int(self._heights[x][y]))
                if x == self._ax and y == self._ay and x == gx and y == gy:
                    px = [255, 220, 50]
                elif x == self._ax and y == self._ay:
                    px = [255, 120, 20]
                elif x == gx and y == gy:
                    px = [30, 230, 80]
                elif (x, y) in patrol_set:
                    px = [220, 30, 30]
                row.append(px)
            frame.append(row)
        return frame


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class World3DFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "3d", "terrain", "spatial", "physics", "risk_sensitive", "sim2real"]

    def create(self, spec: VerseSpec) -> World3DVerse:
        return World3DVerse(spec)
