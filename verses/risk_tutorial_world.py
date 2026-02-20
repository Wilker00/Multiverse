"""
verses/risk_tutorial_world.py

A focused teaching Verse that trains one abstract concept: risk calibration.
The agent must lower risk before attempting a high-commit conversion move.
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
class RiskTutorialParams:
    max_steps: int = 50
    risk_floor_start: int = 8
    target_control: int = 8
    all_in_threshold: int = 3
    stabilize_gain: int = 2
    probe_gain: int = 2
    probe_risk: int = 1
    ambient_risk_noise: float = 0.20
    success_reward: float = 2.5
    fail_penalty: float = -2.0
    step_penalty: float = -0.01


class RiskTutorialWorldVerse:
    """
    Action space (3):
    0 = stabilize_risk
    1 = measured_probe
    2 = high_commit_push
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in RiskTutorialWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = RiskTutorialParams(
            max_steps=int(self.spec.params.get("max_steps", 50)),
            risk_floor_start=int(self.spec.params.get("risk_floor_start", 8)),
            target_control=int(self.spec.params.get("target_control", 8)),
            all_in_threshold=int(self.spec.params.get("all_in_threshold", 3)),
            stabilize_gain=int(self.spec.params.get("stabilize_gain", 2)),
            probe_gain=int(self.spec.params.get("probe_gain", 2)),
            probe_risk=int(self.spec.params.get("probe_risk", 1)),
            ambient_risk_noise=float(self.spec.params.get("ambient_risk_noise", 0.20)),
            success_reward=float(self.spec.params.get("success_reward", 2.5)),
            fail_penalty=float(self.spec.params.get("fail_penalty", -2.0)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.01)),
        )
        self.params.max_steps = max(10, int(self.params.max_steps))
        self.params.risk_floor_start = _clip(self.params.risk_floor_start, 2, 12)
        self.params.target_control = _clip(self.params.target_control, 4, 12)
        self.params.all_in_threshold = _clip(self.params.all_in_threshold, 1, 8)
        self.params.ambient_risk_noise = max(0.0, min(1.0, float(self.params.ambient_risk_noise)))

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=["risk", "control", "pressure", "confidence", "lesson_phase", "t"],
            subspaces={
                "risk": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "control": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "pressure": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "confidence": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "lesson_phase": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="Teacher verse for risk-aware timing and conversion.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=3,
            notes="0=stabilize,1=probe,2=high_commit",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._done = False
        self._t = 0
        self._risk = 0
        self._control = 0
        self._confidence = 0

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
        self._risk = int(self.params.risk_floor_start)
        self._control = 2
        self._confidence = 3
        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": self._seed,
                "lesson": "risk_calibration",
                "target_control": int(self.params.target_control),
                "all_in_threshold": int(self.params.all_in_threshold),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = _clip(int(action), 0, 2)
        self._t += 1
        reward = float(self.params.step_penalty)
        high_risk_failure = False

        if a == 0:
            self._risk = _clip(self._risk - int(self.params.stabilize_gain), 0, 12)
            self._control = _clip(self._control + 1, 0, 12)
            self._confidence = _clip(self._confidence + 1, 0, 12)
            reward += 0.08
        elif a == 1:
            self._control = _clip(self._control + int(self.params.probe_gain), 0, 12)
            self._risk = _clip(self._risk + int(self.params.probe_risk), 0, 12)
            self._confidence = _clip(self._confidence + 1, 0, 12)
            reward += 0.06 if self._risk <= 5 else -0.08
        else:
            ready = self._risk <= int(self.params.all_in_threshold) and self._control >= int(self.params.target_control)
            if ready:
                reward += float(self.params.success_reward)
                self._done = True
                return StepResult(
                    obs=self._make_obs(),
                    reward=float(reward),
                    done=True,
                    truncated=False,
                    info={
                        "reached_goal": True,
                        "converted_advantage": True,
                        "high_risk_failure": False,
                        "lesson": "risk_calibration",
                        "t": int(self._t),
                    },
                )
            high_risk_failure = True
            self._risk = _clip(self._risk + 2, 0, 12)
            self._control = _clip(self._control - 1, 0, 12)
            self._confidence = _clip(self._confidence - 2, 0, 12)
            reward += float(self.params.fail_penalty)

        if self._rng.random() < float(self.params.ambient_risk_noise):
            self._risk = _clip(self._risk + 1, 0, 12)

        truncated = bool(self._t >= int(self.params.max_steps))
        self._done = bool(truncated)
        info: Dict[str, JSONValue] = {
            "reached_goal": False,
            "high_risk_failure": bool(high_risk_failure),
            "lesson": "risk_calibration",
            "risk": int(self._risk),
            "control": int(self._control),
            "t": int(self._t),
        }
        return StepResult(self._make_obs(), float(reward), False, truncated, info)

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[int]:
        return [] if self._done else [0, 1, 2]

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode != "ansi":
            return None
        return (
            f"RiskTutorial t={self._t} risk={self._risk} "
            f"control={self._control} confidence={self._confidence}"
        )

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "done": bool(self._done),
            "t": int(self._t),
            "risk": int(self._risk),
            "control": int(self._control),
            "confidence": int(self._confidence),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._done = bool(state.get("done", False))
        self._t = _clip(int(state.get("t", self._t)), 0, int(self.params.max_steps))
        self._risk = _clip(int(state.get("risk", self._risk)), 0, 12)
        self._control = _clip(int(state.get("control", self._control)), 0, 12)
        self._confidence = _clip(int(state.get("confidence", self._confidence)), 0, 12)

    def _lesson_phase(self) -> int:
        if self._risk > int(self.params.all_in_threshold):
            return 0  # stabilize
        if self._control < int(self.params.target_control):
            return 1  # build control
        return 2  # ready to convert

    def _make_obs(self) -> JSONValue:
        pressure = _clip(self._risk + max(0, self._t // 8), 0, 12)
        return {
            "risk": int(self._risk),
            "control": int(self._control),
            "pressure": int(pressure),
            "confidence": int(self._confidence),
            "lesson_phase": int(self._lesson_phase()),
            "t": int(self._t),
        }


class RiskTutorialWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["tutorial", "risk_sensitive", "teacher_generated", "strategy"]

    def create(self, spec: VerseSpec) -> Verse:
        return RiskTutorialWorldVerse(spec)

