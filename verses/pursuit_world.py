"""
verses/pursuit_world.py

Minimal pursuit Verse on a 1D line.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


def hash_spec(spec: VerseSpec) -> str:
    # ... (existing implementation)
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class PursuitWorldParams:
    # ... (existing implementation)
    lane_len: int = 9
    start_agent: int = 0
    start_target: int = -1
    max_steps: int = 60
    step_penalty: float = -0.01
    catch_reward: float = 1.0


class PursuitWorldVerse:
    """
    Agent tries to catch a target moving randomly on a 1D line.
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in PursuitWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = PursuitWorldParams(
            lane_len=int(self.spec.params.get("lane_len", 9)),
            start_agent=int(self.spec.params.get("start_agent", 0)),
            start_target=int(self.spec.params.get("start_target", -1)),
            max_steps=int(self.spec.params.get("max_steps", 60)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.01)),
            catch_reward=float(self.spec.params.get("catch_reward", 1.0)),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=["agent", "target", "t"],
            subspaces={
                "agent": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "target": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="PursuitWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=2,
            notes="0=left,1=right",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._agent = 0
        self._target = 0
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

        self._agent = max(0, min(self.params.lane_len - 1, int(self.params.start_agent)))
        start_target = int(self.params.start_target)
        if start_target < 0:
            start_target = self.params.lane_len - 1
        self._target = max(0, min(self.params.lane_len - 1, start_target))
        self._t = 0
        self._done = False
        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "lane_len": self.params.lane_len,
            "start_agent": self.params.start_agent,
            "start_target": start_target,
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
        if a == 0:
            self._agent = max(0, self._agent - 1)
        elif a == 1:
            self._agent = min(self.params.lane_len - 1, self._agent + 1)
        else:
            raise ValueError("PursuitWorld action must be 0 or 1")

        # Target moves randomly after agent
        if self._rng.random() < 0.5:
            self._target = max(0, self._target - 1)
        else:
            self._target = min(self.params.lane_len - 1, self._target + 1)

        self._t += 1

        caught = self._agent == self._target
        truncated = self._t >= self.params.max_steps and not caught

        reward = float(self.params.catch_reward) if caught else float(self.params.step_penalty)
        done = bool(caught)
        self._done = bool(done or truncated)

        info: Dict[str, JSONValue] = {
            "t": self._t,
            "agent": self._agent,
            "target": self._target,
            "reached_goal": caught,
        }

        return StepResult(
            obs=self._make_obs(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        cells = ["." for _ in range(self.params.lane_len)]
        cells[self._target] = "T"
        cells[self._agent] = "A"
        return "|" + "".join(cells) + "|"

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "agent": int(self._agent),
            "target": int(self._target),
            "t": int(self._t),
            "done": bool(self._done),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._agent = max(0, min(self.params.lane_len - 1, int(state.get("agent", self._agent))))
        self._target = max(0, min(self.params.lane_len - 1, int(state.get("target", self._target))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))

    def _make_obs(self) -> JSONValue:
        return {"agent": self._agent, "target": self._target, "t": self._t}


class PursuitWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["multi-agent", "pursuit"]

    def create(self, spec: VerseSpec) -> Verse:
        return PursuitWorldVerse(spec)
