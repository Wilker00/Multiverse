"""
verses/line_world.py

1D line-world: agent moves left/right along a lane to reach a goal.

Params:
  lane_len    : total length of the lane (default 10)
  start_pos   : agent starting position (default 0)
  goal_pos    : goal position (default lane_len - 1)
  max_steps   : maximum steps per episode (default 30)
  step_penalty: reward applied each step (default -0.02)
  goal_reward : reward on reaching goal (default 1.0)
  friction    : probability that an action is ignored / agent stays (default 0.0)
  gravity     : cells drifted left per step (default 0)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


def hash_spec(spec: VerseSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class LineWorldParams:
    lane_len: int = 10
    start_pos: int = 0
    goal_pos: int = -1       # -1 → lane_len - 1
    max_steps: int = 30
    step_penalty: float = -0.02
    goal_reward: float = 1.0
    friction: float = 0.0    # probability action is ignored (agent stays)
    gravity: int = 0         # cells drifted toward position 0 each step


class LineWorldVerse:
    """
    Minimal 1-D navigation verse.

    Observation dict:
      pos      : current agent position  (int)
      goal     : goal position           (int)
      dist     : abs(goal - pos)         (int)
      t        : timestep                (int)

    Actions:
      0 = move left  (-1)
      1 = move right (+1)

    ANSI render:  |A...G...|
    """

    def __init__(self, spec: VerseSpec) -> None:
        tags = list(spec.tags)
        for t in LineWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        lane_len = max(2, int(self.spec.params.get("lane_len", 10)))
        goal_pos_raw = int(self.spec.params.get("goal_pos", -1))
        goal_pos = (lane_len - 1) if goal_pos_raw < 0 else max(0, min(lane_len - 1, goal_pos_raw))

        self.params = LineWorldParams(
            lane_len=lane_len,
            start_pos=max(0, min(lane_len - 1, int(self.spec.params.get("start_pos", 0)))),
            goal_pos=goal_pos,
            max_steps=max(1, int(self.spec.params.get("max_steps", 30))),
            step_penalty=float(self.spec.params.get("step_penalty", -0.02)),
            goal_reward=float(self.spec.params.get("goal_reward", 1.0)),
            friction=max(0.0, min(1.0, float(self.spec.params.get("friction", 0.0)))),
            gravity=max(0, int(self.spec.params.get("gravity", 0))),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=["pos", "goal", "dist", "t"],
            subspaces={
                "pos":  SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "dist": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t":    SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="LineWorld obs: position, goal, distance, timestep",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=2,
            notes="0=left, 1=right",
        )

        self._rng = random.Random()
        self._pos: int = int(self.params.start_pos)
        self._t: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Verse contract
    # ------------------------------------------------------------------

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._rng = random.Random(int(seed))

    def reset(self) -> ResetResult:
        if self.spec.seed is not None:
            self.seed(self.spec.seed)
        self._pos = int(self.params.start_pos)
        self._t = 0
        self._done = False
        return ResetResult(obs=self._make_obs(), info=self._make_info())

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
        if a not in (0, 1):
            raise ValueError(f"LineWorld action must be 0 or 1, got {a}")

        # Apply friction (action may be ignored)
        if self.params.friction > 0.0 and self._rng.random() < self.params.friction:
            pass  # slip — agent stays
        else:
            dx = -1 if a == 0 else 1
            self._pos = max(0, min(self.params.lane_len - 1, self._pos + dx))

        # Apply gravity (drift toward 0)
        if self.params.gravity > 0:
            self._pos = max(0, self._pos - self.params.gravity)

        self._t += 1
        reached = (self._pos == self.params.goal_pos)
        truncated = not reached and self._t >= self.params.max_steps
        reward = self.params.goal_reward if reached else float(self.params.step_penalty)
        self._done = reached or truncated

        info = self._make_info()
        info["reached_goal"] = bool(reached)

        return StepResult(
            obs=self._make_obs(),
            reward=reward,
            done=bool(reached),
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
            return self._render_rgb_array()
        if mode != "ansi":
            return None
        return self._render_ansi()

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "pos": int(self._pos),
            "t": int(self._t),
            "done": bool(self._done),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._pos = max(0, min(self.params.lane_len - 1, int(state.get("pos", self._pos))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_obs(self) -> JSONValue:
        return {
            "pos":  int(self._pos),
            "goal": int(self.params.goal_pos),
            "dist": abs(int(self.params.goal_pos) - int(self._pos)),
            "t":    int(self._t),
        }

    def _make_info(self) -> Dict[str, JSONValue]:
        return {
            "pos":      int(self._pos),
            "goal":     int(self.params.goal_pos),
            "lane_len": int(self.params.lane_len),
            "t":        int(self._t),
        }

    def _render_ansi(self) -> str:
        cells: List[str] = []
        for i in range(self.params.lane_len):
            if i == self._pos and i == self.params.goal_pos:
                cells.append("X")   # agent on goal
            elif i == self._pos:
                cells.append("A")   # agent
            elif i == self.params.goal_pos:
                cells.append("G")   # goal
            else:
                cells.append(".")
        return "|" + "".join(cells) + "|"

    def _render_rgb_array(self) -> List[List[List[int]]]:
        """1-row RGB frame (1 x lane_len x 3)."""
        bg      = [255, 255, 255]
        agent_c = [60, 120, 240]
        goal_c  = [60, 200, 80]
        both_c  = [220, 180, 60]
        row: List[List[int]] = []
        for i in range(self.params.lane_len):
            on_agent = (i == self._pos)
            on_goal  = (i == self.params.goal_pos)
            if on_agent and on_goal:
                row.append(list(both_c))
            elif on_agent:
                row.append(list(agent_c))
            elif on_goal:
                row.append(list(goal_c))
            else:
                row.append(list(bg))
        return [row]


class LineWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "1d", "simple", "deterministic"]

    def create(self, spec: VerseSpec) -> LineWorldVerse:
        return LineWorldVerse(spec)
