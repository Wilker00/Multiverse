"""
verses/maze_world.py

Maze Runner Verse — procedurally generated maze using recursive backtracking.
Every maze is guaranteed to be solvable (perfect maze, exactly one path exists).

Agent starts at (0, 0) top-left corner.
Exit is at (width-1, height-1) bottom-right corner.

Observations (dict):
  - x, y            : agent position (cell coordinates)
  - exit_x, exit_y  : exit position
  - dx, dy          : signed distance to exit (direction hint)
  - t               : step counter
  - wall_n           : 1 if wall to the north,  0 if open
  - wall_s           : 1 if wall to the south,  0 if open
  - wall_w           : 1 if wall to the west,   0 if open
  - wall_e           : 1 if wall to the east,   0 if open
  - visited          : number of unique cells visited so far
  - steps_since_new  : steps since agent last visited a new cell (exploration stagnation)

Actions:  0=north  1=south  2=west  3=east

Rewards:
  +1.0   reaching the exit
  -0.01  each step (time pressure)
  -0.05  running into a wall (bump penalty)
  +0.10  visiting a brand-new cell (exploration bonus, configurable)

Difficulty params (all configurable via VerseSpec.params):
  width              int   default 7
  height             int   default 7
  max_steps          int   default 200
  step_penalty       float default -0.01
  bump_penalty       float default -0.05
  explore_bonus      float default  0.10   (0 to disable)
  fog_of_war         bool  default False   (only reveals cells agent has visited)
  hazard_count       int   default 0       (random moving hazard tiles)
  hazard_penalty     float default -0.20   (penalty for stepping onto a hazard)

ANSI render legend:
  A = agent
  E = exit
  # = wall
  . = open corridor (visited)
  ? = unvisited (when fog_of_war=True)
  X = hazard
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_spec(spec: VerseSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Maze generation — Recursive Backtracking (Depth-First Search)
# ---------------------------------------------------------------------------

def _generate_maze(width: int, height: int, rng: random.Random) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Returns a set of open PASSAGES as frozenset pairs of adjacent cells.
    A passage between (x1,y1) and (x2,y2) means the wall between them is removed.
    All cells not connected by a passage are walled off.
    """
    passages: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    visited: Set[Tuple[int, int]] = set()

    def neighbors(x: int, y: int) -> List[Tuple[int, int]]:
        result = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                result.append((nx, ny))
        return result

    def carve(x: int, y: int) -> None:
        visited.add((x, y))
        nbrs = neighbors(x, y)
        rng.shuffle(nbrs)
        for nx, ny in nbrs:
            if (nx, ny) not in visited:
                # Remove wall between (x,y) and (nx,ny)
                passages.add(((x, y), (nx, ny)))
                passages.add(((nx, ny), (x, y)))
                carve(nx, ny)

    # Use iterative DFS to avoid Python recursion limits on large mazes
    stack = [(0, 0)]
    visited.add((0, 0))
    while stack:
        x, y = stack[-1]
        nbrs = [n for n in neighbors(x, y) if n not in visited]
        if not nbrs:
            stack.pop()
        else:
            nx, ny = rng.choice(nbrs)
            passages.add(((x, y), (nx, ny)))
            passages.add(((nx, ny), (x, y)))
            visited.add((nx, ny))
            stack.append((nx, ny))

    return passages


def _bfs_path_length(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    passages: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    width: int,
    height: int,
) -> int:
    """Returns shortest path length (in steps) from start to goal. -1 if unreachable."""
    if start == goal:
        return 0
    visited: Set[Tuple[int, int]] = {start}
    queue: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and ((x, y), (nx, ny)) in passages:
                if (nx, ny) == goal:
                    return dist + 1
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return -1


# ---------------------------------------------------------------------------
# Params & Verse
# ---------------------------------------------------------------------------

@dataclass
class MazeWorldParams:
    width: int = 7
    height: int = 7
    max_steps: int = 200
    step_penalty: float = -0.01
    bump_penalty: float = -0.05
    explore_bonus: float = 0.10
    fog_of_war: bool = False
    hazard_count: int = 0
    hazard_penalty: float = -0.20


class MazeWorldVerse:
    """
    Procedurally generated maze-runner environment.
    The agent must find the exit in a perfect maze (unique path, always solvable).
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in MazeWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        p = self.spec.params
        self.params = MazeWorldParams(
            width=max(3, int(p.get("width", 7))),
            height=max(3, int(p.get("height", 7))),
            max_steps=max(10, int(p.get("max_steps", 200))),
            step_penalty=float(p.get("step_penalty", -0.01)),
            bump_penalty=float(p.get("bump_penalty", -0.05)),
            explore_bonus=float(p.get("explore_bonus", 0.10)),
            fog_of_war=bool(p.get("fog_of_war", False)),
            hazard_count=max(0, int(p.get("hazard_count", 0))),
            hazard_penalty=float(p.get("hazard_penalty", -0.20)),
        )

        # Build observation space
        obs_keys = [
            "x", "y", "exit_x", "exit_y", "dx", "dy", "t",
            "wall_n", "wall_s", "wall_w", "wall_e",
            "visited", "steps_since_new",
        ]
        obs_subspaces = {
            "x":              SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "y":              SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "exit_x":         SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "exit_y":         SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "dx":             SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "dy":             SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "t":              SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "wall_n":         SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "wall_s":         SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "wall_w":         SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "wall_e":         SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "visited":        SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "steps_since_new":SpaceSpec(type="vector", shape=(1,), dtype="int32"),
        }
        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=obs_keys,
            subspaces=obs_subspaces,
            notes="MazeWorld: agent position, wall sensors, exploration stats",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=4,
            notes="0=north(up) 1=south(down) 2=west(left) 3=east(right)",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._x = 0
        self._y = 0
        self._t = 0
        self._done = False
        self._passages: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        self._visited_cells: Set[Tuple[int, int]] = set()
        self._steps_since_new = 0
        self._hazards: List[Tuple[int, int]] = []  # current hazard positions
        self._optimal_steps = 0  # BFS-optimal path length (for info)

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2 ** 31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        # Generate fresh maze
        self._passages = _generate_maze(self.params.width, self.params.height, self._rng)

        # Agent always starts top-left, exit always bottom-right
        self._x = 0
        self._y = 0
        self._t = 0
        self._done = False
        self._visited_cells = {(0, 0)}
        self._steps_since_new = 0

        # Place hazards on random non-start, non-exit open cells
        self._hazards = []
        if self.params.hazard_count > 0:
            ex, ey = self._exit_pos()
            candidates = [
                (cx, cy)
                for cx in range(self.params.width)
                for cy in range(self.params.height)
                if (cx, cy) != (0, 0) and (cx, cy) != (ex, ey)
            ]
            self._rng.shuffle(candidates)
            self._hazards = candidates[:self.params.hazard_count]

        # Compute optimal path length (for diagnostics)
        ex, ey = self._exit_pos()
        self._optimal_steps = _bfs_path_length((0, 0), (ex, ey), self._passages, self.params.width, self.params.height)

        obs = self._make_obs()
        return ResetResult(
            obs=obs,
            info={
                "seed": self._seed,
                "width": self.params.width,
                "height": self.params.height,
                "exit_x": self._exit_pos()[0],
                "exit_y": self._exit_pos()[1],
                "total_cells": self.params.width * self.params.height,
                "optimal_steps": self._optimal_steps,
                "fog_of_war": self.params.fog_of_war,
                "hazard_count": len(self._hazards),
            },
        )

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
        deltas = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        if a not in deltas:
            raise ValueError(f"MazeWorld action must be 0..3, got {a}")

        dx, dy = deltas[a]
        nx, ny = self._x + dx, self._y + dy

        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {
            "t": self._t,
            "reached_goal": False,
            "bumped_wall": False,
            "hit_hazard": False,
            "new_cell": False,
        }

        # Check if move is valid (passage exists)
        if ((self._x, self._y), (nx, ny)) in self._passages:
            self._x, self._y = nx, ny

            # Exploration bonus for new cells
            if (self._x, self._y) not in self._visited_cells:
                self._visited_cells.add((self._x, self._y))
                reward += float(self.params.explore_bonus)
                self._steps_since_new = 0
                info["new_cell"] = True
            else:
                self._steps_since_new += 1
        else:
            # Hit a wall — stay in place, apply bump penalty
            reward += float(self.params.bump_penalty)
            self._steps_since_new += 1
            info["bumped_wall"] = True

        # Hazard check
        if (self._x, self._y) in self._hazards:
            reward += float(self.params.hazard_penalty)
            info["hit_hazard"] = True

        # Move hazards randomly (they roam the maze)
        if self._hazards:
            new_hazards: List[Tuple[int, int]] = []
            for hx, hy in self._hazards:
                ex, ey = self._exit_pos()
                dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                valid_moves = [
                    (hx + ddx, hy + ddy)
                    for ddx, ddy in dirs
                    if ((hx, hy), (hx + ddx, hy + ddy)) in self._passages
                    and (hx + ddx, hy + ddy) != (0, 0)
                    and (hx + ddx, hy + ddy) != (ex, ey)
                ]
                if valid_moves:
                    new_hazards.append(self._rng.choice(valid_moves))
                else:
                    new_hazards.append((hx, hy))
            self._hazards = new_hazards

        # Check exit
        ex, ey = self._exit_pos()
        reached_goal = (self._x == ex and self._y == ey)
        truncated = (self._t >= self.params.max_steps and not reached_goal)

        if reached_goal:
            reward = 1.0
            info["reached_goal"] = True
            info["episode_success"] = True

        done = bool(reached_goal)
        self._done = bool(done or truncated)

        info["x"] = self._x
        info["y"] = self._y
        info["visited_cells"] = len(self._visited_cells)
        info["total_cells"] = self.params.width * self.params.height
        info["coverage_pct"] = round(100.0 * len(self._visited_cells) / max(1, self.params.width * self.params.height), 2)

        return StepResult(
            obs=self._make_obs(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

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
        return self._render_ansi()

    def _render_ansi(self) -> str:
        """
        Render the maze as ASCII art.
        Each cell is 3 chars wide and 2 lines tall to show walls clearly.

        Wall representation:
          +---+  horizontal wall above cell
          |   |  vertical walls on sides
        """
        w = self.params.width
        h = self.params.height
        ex, ey = self._exit_pos()
        fog = bool(self.params.fog_of_war)

        lines: List[str] = []

        for y in range(h):
            # Top border of cell row
            top_line = ""
            for x in range(w):
                top_line += "+"
                # North wall: wall if no passage going north from (x, y)
                has_north = ((x, y), (x, y - 1)) in self._passages
                revealed = not fog or (x, y) in self._visited_cells or (x, y - 1) in self._visited_cells
                if y == 0:
                    top_line += "---"
                elif has_north and revealed:
                    top_line += "   "
                else:
                    top_line += "---"
            top_line += "+"
            lines.append(top_line)

            # Cell content row
            mid_line = ""
            for x in range(w):
                # West wall
                has_west = ((x, y), (x - 1, y)) in self._passages
                revealed = not fog or (x, y) in self._visited_cells or (x - 1, y) in self._visited_cells
                if x == 0:
                    mid_line += "|"
                elif has_west and revealed:
                    mid_line += " "
                else:
                    mid_line += "|"

                # Cell content
                revealed_cell = not fog or (x, y) in self._visited_cells
                if x == self._x and y == self._y:
                    mid_line += " A "
                elif x == ex and y == ey:
                    mid_line += " E "
                elif (x, y) in self._hazards and revealed_cell:
                    mid_line += " X "
                elif not revealed_cell:
                    mid_line += " ? "
                elif (x, y) in self._visited_cells:
                    mid_line += " . "
                else:
                    mid_line += "   "

            mid_line += "|"
            lines.append(mid_line)

        # Bottom border
        bottom = ("+" + "---") * w + "+"
        lines.append(bottom)

        ex, ey = self._exit_pos()
        stats = (
            f"Step {self._t}/{self.params.max_steps}  "
            f"Pos({self._x},{self._y})  "
            f"Exit({ex},{ey})  "
            f"Visited:{len(self._visited_cells)}/{self.params.width * self.params.height}"
        )
        lines.append(stats)
        return "\n".join(lines)

    def _render_rgb_array(self) -> List[List[List[int]]]:
        """
        Render as RGB pixel grid (3px per maze cell for visibility).
        Colors: wall=dark, open=white, agent=blue, exit=green, hazard=orange, visited=light-blue
        """
        w, h = self.params.width, self.params.height
        ex, ey = self._exit_pos()
        scale = 3  # pixels per cell
        pw = w * scale + 1
        ph = h * scale + 1

        # Canvas filled with wall color
        wall_c = [40, 40, 40]
        open_c = [240, 240, 240]
        visited_c = [200, 225, 255]
        agent_c = [30, 100, 240]
        exit_c = [40, 200, 80]
        hazard_c = [240, 140, 40]
        fog_c = [80, 80, 80]

        frame: List[List[List[int]]] = [[list(wall_c) for _ in range(pw)] for _ in range(ph)]

        fog = bool(self.params.fog_of_war)

        def cell_px(cx: int, cy: int) -> Tuple[int, int]:
            return cy * scale + 1, cx * scale + 1  # (row, col) of top-left pixel of cell

        def fill_cell(cx: int, cy: int, color: List[int]) -> None:
            r0, c0 = cell_px(cx, cy)
            for dr in range(scale - 1):
                for dc in range(scale - 1):
                    frame[r0 + dr][c0 + dc] = list(color)

        def open_passage(cx1: int, cy1: int, cx2: int, cy2: int, color: List[int]) -> None:
            """Open the wall pixel between two adjacent cells."""
            r1, c1 = cell_px(cx1, cy1)
            r2, c2 = cell_px(cx2, cy2)
            mr = (r1 + r2 + scale - 2) // 2
            mc = (c1 + c2 + scale - 2) // 2
            frame[mr][mc] = list(color)

        for y in range(h):
            for x in range(w):
                revealed = not fog or (x, y) in self._visited_cells
                if not revealed:
                    fill_cell(x, y, fog_c)
                elif (x, y) in self._visited_cells:
                    fill_cell(x, y, visited_c)
                else:
                    fill_cell(x, y, open_c)

        # Passages
        for y in range(h):
            for x in range(w):
                for dx, dy in [(1, 0), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if ((x, y), (nx, ny)) in self._passages:
                            fog_both = fog and (x, y) not in self._visited_cells and (nx, ny) not in self._visited_cells
                            c = fog_c if fog_both else visited_c if ((x, y) in self._visited_cells or (nx, ny) in self._visited_cells) else open_c
                            open_passage(x, y, nx, ny, c)

        # Hazards, exit, agent
        for hx, hy in self._hazards:
            if not fog or (hx, hy) in self._visited_cells:
                fill_cell(hx, hy, hazard_c)
        fill_cell(ex, ey, exit_c)
        fill_cell(self._x, self._y, agent_c)

        return frame

    # ------------------------------------------------------------------
    # State checkpointing (for SafeExecutor)
    # ------------------------------------------------------------------

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "x": int(self._x),
            "y": int(self._y),
            "t": int(self._t),
            "done": bool(self._done),
            "visited_cells": [[int(cx), int(cy)] for cx, cy in self._visited_cells],
            "steps_since_new": int(self._steps_since_new),
            "hazards": [[int(hx), int(hy)] for hx, hy in self._hazards],
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._x = max(0, min(self.params.width - 1, int(state.get("x", self._x))))
        self._y = max(0, min(self.params.height - 1, int(state.get("y", self._y))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        self._steps_since_new = int(state.get("steps_since_new", 0))
        vc = state.get("visited_cells", [])
        self._visited_cells = {(int(c[0]), int(c[1])) for c in (vc or []) if isinstance(c, (list, tuple)) and len(c) == 2}
        hazards_raw = state.get("hazards", [])
        self._hazards = [(int(h[0]), int(h[1])) for h in (hazards_raw or []) if isinstance(h, (list, tuple)) and len(h) == 2]

    def close(self) -> None:
        return

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _exit_pos(self) -> Tuple[int, int]:
        return (self.params.width - 1, self.params.height - 1)

    def _wall_sensor(self, dx: int, dy: int) -> int:
        """1 if wall in direction (dx,dy), 0 if open passage."""
        nx, ny = self._x + dx, self._y + dy
        if not (0 <= nx < self.params.width and 0 <= ny < self.params.height):
            return 1  # boundary = wall
        return 0 if ((self._x, self._y), (nx, ny)) in self._passages else 1

    def _make_obs(self) -> JSONValue:
        ex, ey = self._exit_pos()
        return {
            "x": int(self._x),
            "y": int(self._y),
            "exit_x": int(ex),
            "exit_y": int(ey),
            "dx": int(ex - self._x),
            "dy": int(ey - self._y),
            "t": int(self._t),
            "wall_n": self._wall_sensor(0, -1),
            "wall_s": self._wall_sensor(0, 1),
            "wall_w": self._wall_sensor(-1, 0),
            "wall_e": self._wall_sensor(1, 0),
            "visited": int(len(self._visited_cells)),
            "steps_since_new": int(self._steps_since_new),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class MazeWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "maze", "exploration", "spatial"]

    def create(self, spec: VerseSpec) -> Verse:
        return MazeWorldVerse(spec)
