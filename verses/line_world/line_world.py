"""
verses/line_world/line_world.py
...
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
    # ... (existing implementation)
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class LineWorldParams:
    # ... (existing implementation)
    start_pos: int = 0
    goal_pos: int = 10
    max_steps: int = 50
    step_penalty: float = -0.01


class LineWorldVerse:
    """
    Minimal Verse implementation.
    """

    def __init__(self, spec: VerseSpec):
        # Add tags to the instance's spec without mutating a frozen dataclass payload.
        merged_tags = list(spec.tags)
        for t in LineWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = LineWorldParams(
            start_pos=int(self.spec.params.get("start_pos", 0)),
            goal_pos=int(self.spec.params.get("goal_pos", 10)),
            max_steps=int(self.spec.params.get("max_steps", 50)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.01)),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=["pos", "goal", "t"],
            subspaces={
                "pos": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="LineWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=2,
            notes="0=left, 1=right",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._pos = 0
        self._t = 0
        self._done = False

    # ... (rest of the methods are unchanged)
    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        self._pos = max(0, min(int(self.params.goal_pos), int(self.params.start_pos)))
        self._t = 0
        self._done = False

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "start_pos": self.params.start_pos,
            "goal_pos": self.params.goal_pos,
            "max_steps": self.params.max_steps,
            "step_penalty": self.params.step_penalty,
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
        if a not in (0, 1):
            raise ValueError("LineWorld action must be 0 (left) or 1 (right)")

        if a == 0:
            self._pos = max(0, self._pos - 1)
        else:
            self._pos = min(self.params.goal_pos, self._pos + 1)

        self._t += 1

        reached = self._pos >= self.params.goal_pos
        truncated = self._t >= self.params.max_steps and not reached

        reward = 1.0 if reached else float(self.params.step_penalty)
        done = bool(reached)

        self._done = bool(done or truncated)

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "t": self._t,
            "pos": self._pos,
            "reached_goal": reached,
        }

        return StepResult(
            obs=obs,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode == "ansi":
            # Example: "|A....G|"
            goal = self.params.goal_pos
            cells = ["." for _ in range(goal + 1)]
            cells[self._pos] = "A"
            cells[goal] = "G"
            return "|" + "".join(cells) + "|"
        return None

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "pos": int(self._pos),
            "t": int(self._t),
            "done": bool(self._done),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._pos = max(0, min(int(self.params.goal_pos), int(state.get("pos", self._pos))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))

    def _make_obs(self) -> JSONValue:
        return {"pos": self._pos, "goal": self.params.goal_pos, "t": self._t}


class LineWorldFactory:
    """
    Factory so orchestrator code can instantiate by spec.
    """
    @property
    def tags(self) -> List[str]:
        return ["navigation", "simple"]

    def create(self, spec: VerseSpec) -> Verse:
        return LineWorldVerse(spec)
