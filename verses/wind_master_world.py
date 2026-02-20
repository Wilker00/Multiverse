"""
verses/wind_master_world.py

Teacher tutorial verse for safety-margin planning under wind perturbations.
The objective is to reach the goal while maintaining distance from the top/bottom
edges, not merely avoiding direct edge collisions.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


@dataclass
class WindMasterParams:
    width: int = 14
    height: int = 7
    start_x: int = 0
    start_y: int = -1
    goal_x: int = -1
    goal_y: int = -1
    max_steps: int = 80

    gust_probability: float = 0.12
    edge_bias: float = 0.75

    target_margin: int = 2
    step_penalty: float = -0.02
    progress_reward: float = 0.08
    regress_penalty: float = -0.04
    margin_reward_scale: float = 0.05
    edge_penalty: float = -1.50
    goal_reward: float = 3.00
    unsafe_goal_penalty: float = -1.00


class WindMasterWorldVerse:
    """
    Action space:
    0=up, 1=down, 2=left, 3=right
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in WindMasterWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = WindMasterParams(
            width=int(self.spec.params.get("width", 14)),
            height=int(self.spec.params.get("height", 7)),
            start_x=int(self.spec.params.get("start_x", 0)),
            start_y=int(self.spec.params.get("start_y", -1)),
            goal_x=int(self.spec.params.get("goal_x", -1)),
            goal_y=int(self.spec.params.get("goal_y", -1)),
            max_steps=int(self.spec.params.get("max_steps", 80)),
            gust_probability=float(self.spec.params.get("gust_probability", 0.12)),
            edge_bias=float(self.spec.params.get("edge_bias", 0.75)),
            target_margin=int(self.spec.params.get("target_margin", 2)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.02)),
            progress_reward=float(self.spec.params.get("progress_reward", 0.08)),
            regress_penalty=float(self.spec.params.get("regress_penalty", -0.04)),
            margin_reward_scale=float(self.spec.params.get("margin_reward_scale", 0.05)),
            edge_penalty=float(self.spec.params.get("edge_penalty", -1.50)),
            goal_reward=float(self.spec.params.get("goal_reward", 3.00)),
            unsafe_goal_penalty=float(self.spec.params.get("unsafe_goal_penalty", -1.00)),
        )

        self.params.width = max(5, int(self.params.width))
        self.params.height = max(5, int(self.params.height))
        self.params.max_steps = max(10, int(self.params.max_steps))
        self.params.gust_probability = max(0.0, min(1.0, float(self.params.gust_probability)))
        self.params.edge_bias = max(0.0, min(1.0, float(self.params.edge_bias)))
        self.params.target_margin = _clip(int(self.params.target_margin), 1, max(1, (self.params.height - 1) // 2))

        obs_keys = ["x", "y", "goal_x", "goal_y", "safety_margin", "wind_active", "t"]
        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=obs_keys,
            subspaces={
                "x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_x": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal_y": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "safety_margin": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "wind_active": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="Tutorial obs emphasizing lateral safety margin under wind.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=4,
            notes="0=up,1=down,2=left,3=right",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._done = False
        self._t = 0
        self._x = 0
        self._y = 0
        self._wind_active = False

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)
        self._done = False
        self._t = 0
        self._wind_active = False
        self._x = int(self._start_x())
        self._y = int(self._start_y())
        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": self._seed,
                "lesson": "safety_margin_navigation",
                "target_margin": int(self.params.target_margin),
                "goal_x": int(self._goal_x()),
                "goal_y": int(self._goal_y()),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = _clip(int(action), 0, 3)
        prev_x = int(self._x)
        if a == 0:
            self._y = max(0, int(self._y - 1))
        elif a == 1:
            self._y = min(int(self.params.height - 1), int(self._y + 1))
        elif a == 2:
            self._x = max(0, int(self._x - 1))
        else:
            self._x = min(int(self.params.width - 1), int(self._x + 1))

        self._wind_active = False
        if self._rng.random() < float(self.params.gust_probability):
            self._wind_active = True
            mid = int((self.params.height - 1) // 2)
            toward_edge = -1 if self._y <= mid else 1
            if self._rng.random() < float(self.params.edge_bias):
                self._y = _clip(self._y + toward_edge, 0, int(self.params.height - 1))
            else:
                self._y = _clip(self._y + (1 if self._rng.random() < 0.5 else -1), 0, int(self.params.height - 1))

        self._t += 1
        margin = int(self._margin(self._y))
        reached_goal = bool(self._x == self._goal_x())
        unsafe_finish = bool(reached_goal and margin < int(self.params.target_margin))
        high_risk_failure = bool(margin == 0)

        reward = float(self.params.step_penalty)
        if self._x > prev_x:
            reward += float(self.params.progress_reward)
        elif self._x < prev_x:
            reward += float(self.params.regress_penalty)
        reward += float(self.params.margin_reward_scale) * float(margin)

        if high_risk_failure:
            reward += float(self.params.edge_penalty)

        done = False
        if reached_goal and not unsafe_finish:
            reward += float(self.params.goal_reward)
            done = True
        elif unsafe_finish:
            reward += float(self.params.unsafe_goal_penalty)
            done = True

        truncated = bool(self._t >= int(self.params.max_steps) and not done)
        self._done = bool(done or truncated)
        info: Dict[str, JSONValue] = {
            "reached_goal": bool(done and not unsafe_finish),
            "unsafe_finish": bool(unsafe_finish),
            "high_risk_failure": bool(high_risk_failure),
            "safety_margin": int(margin),
            "target_margin": int(self.params.target_margin),
            "wind_active": bool(self._wind_active),
            "lesson": "safety_margin_navigation",
            "t": int(self._t),
        }
        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[int]:
        return [] if self._done else [0, 1, 2, 3]

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode != "ansi":
            return None
        return (
            f"WindMaster t={self._t} x={self._x} y={self._y} "
            f"margin={self._margin(self._y)} wind={int(self._wind_active)}"
        )

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "done": bool(self._done),
            "t": int(self._t),
            "x": int(self._x),
            "y": int(self._y),
            "wind_active": bool(self._wind_active),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._done = bool(state.get("done", False))
        self._t = _clip(int(state.get("t", self._t)), 0, int(self.params.max_steps))
        self._x = _clip(int(state.get("x", self._x)), 0, int(self.params.width - 1))
        self._y = _clip(int(state.get("y", self._y)), 0, int(self.params.height - 1))
        self._wind_active = bool(state.get("wind_active", False))

    def _start_x(self) -> int:
        return _clip(int(self.params.start_x), 0, int(self.params.width - 1))

    def _start_y(self) -> int:
        y = int(self.params.start_y)
        if y < 0:
            y = int((self.params.height - 1) // 2)
        return _clip(y, 0, int(self.params.height - 1))

    def _goal_x(self) -> int:
        gx = int(self.params.goal_x)
        if gx < 0:
            gx = int(self.params.width - 1)
        return _clip(gx, 0, int(self.params.width - 1))

    def _goal_y(self) -> int:
        gy = int(self.params.goal_y)
        if gy < 0:
            gy = int((self.params.height - 1) // 2)
        return _clip(gy, 0, int(self.params.height - 1))

    def _margin(self, y: int) -> int:
        return int(min(int(y), int(self.params.height - 1 - y)))

    def _make_obs(self) -> JSONValue:
        return {
            "x": int(self._x),
            "y": int(self._y),
            "goal_x": int(self._goal_x()),
            "goal_y": int(self._goal_y()),
            "safety_margin": int(self._margin(self._y)),
            "wind_active": 1 if self._wind_active else 0,
            "t": int(self._t),
        }


class WindMasterWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["tutorial", "teacher_generated", "risk_sensitive", "navigation", "wind"]

    def create(self, spec: VerseSpec) -> Verse:
        return WindMasterWorldVerse(spec)

