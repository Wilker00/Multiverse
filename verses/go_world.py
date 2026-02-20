"""
verses/go_world.py

Lightweight strategy verse inspired by go.
This is an abstract tactical control game, not a full go rules implementation.
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


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


@dataclass
class GoWorldParams:
    max_steps: int = 90
    target_territory: int = 10
    random_swing: float = 0.25
    convert_bonus: float = 1.4
    step_penalty: float = -0.01


class GoWorldVerse:
    """
    Strategic abstraction with 6 actions:
    0 expand_territory
    1 apply_pressure
    2 tactical_capture
    3 defend_shape
    4 tempo_play (pass)
    5 convert_advantage
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in GoWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = GoWorldParams(
            max_steps=int(self.spec.params.get("max_steps", 90)),
            target_territory=int(self.spec.params.get("target_territory", 10)),
            random_swing=float(self.spec.params.get("random_swing", 0.25)),
            convert_bonus=float(self.spec.params.get("convert_bonus", 1.4)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.01)),
        )
        self.params.max_steps = max(10, int(self.params.max_steps))
        self.params.target_territory = max(4, int(self.params.target_territory))
        self.params.random_swing = max(0.0, min(0.9, float(self.params.random_swing)))

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "territory_delta",
                "liberties_delta",
                "influence",
                "capture_threat",
                "ko_risk",
                "score_delta",
                "pressure",
                "risk",
                "tempo",
                "control",
                "resource",
                "consecutive_passes",
                "t",
            ],
            subspaces={
                "territory_delta": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "liberties_delta": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "influence": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "capture_threat": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "ko_risk": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "score_delta": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "pressure": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "risk": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "tempo": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "control": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "resource": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "consecutive_passes": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="Abstract go-like territorial factors.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=6,
            notes="0=expand,1=pressure,2=capture,3=defend,4=tempo_pass,5=convert",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._done = False
        self._t = 0

        self._territory = 0
        self._liberties = 2
        self._influence = 1
        self._capture_threat = 1
        self._ko_risk = 1
        self._tempo = 2
        self._passes = 0

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
        self._territory = 0
        self._liberties = 2
        self._influence = 1
        self._capture_threat = 1
        self._ko_risk = 1
        self._tempo = 2
        self._passes = 0
        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": self._seed,
                "max_steps": int(self.params.max_steps),
                "target_territory": int(self.params.target_territory),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = _clip(int(action), 0, 5)
        self._t += 1
        reward = float(self.params.step_penalty)

        if a == 0:
            self._territory = _clip(self._territory + 1, -20, 20)
            self._influence = _clip(self._influence + 1, -10, 10)
            self._tempo = _clip(self._tempo + 1, 0, 10)
            self._passes = 0
            reward += 0.05
        elif a == 1:
            self._influence = _clip(self._influence + 2, -10, 10)
            self._capture_threat = _clip(self._capture_threat + 1, 0, 12)
            self._ko_risk = _clip(self._ko_risk + 1, 0, 12)
            self._passes = 0
            reward += 0.03
        elif a == 2:
            p = (
                0.30
                + 0.04 * float(self._capture_threat)
                + 0.03 * float(self._influence)
                - 0.05 * float(max(0, self._ko_risk - self._liberties))
            )
            if self._rng.random() < max(0.05, min(0.95, p)):
                self._territory = _clip(self._territory + 2, -20, 20)
                self._capture_threat = _clip(self._capture_threat - 1, 0, 12)
                self._tempo = _clip(self._tempo + 1, 0, 10)
                reward += 0.20
            else:
                self._territory = _clip(self._territory - 1, -20, 20)
                self._liberties = _clip(self._liberties - 1, -10, 12)
                self._ko_risk = _clip(self._ko_risk + 1, 0, 12)
                reward -= 0.16
            self._passes = 0
        elif a == 3:
            self._liberties = _clip(self._liberties + 2, -10, 12)
            self._ko_risk = _clip(self._ko_risk - 2, 0, 12)
            self._capture_threat = _clip(self._capture_threat - 1, 0, 12)
            self._tempo = _clip(self._tempo - 1, 0, 10)
            self._passes = 0
            reward += 0.06
        elif a == 4:
            self._passes = _clip(self._passes + 1, 0, 3)
            self._tempo = _clip(self._tempo + 1, 0, 10)
            reward += 0.03 if self._territory > 0 else -0.03
        else:
            if self._territory >= int(self.params.target_territory) and self._ko_risk <= 2:
                reward += float(self.params.convert_bonus)
                self._done = True
                return StepResult(
                    obs=self._make_obs(),
                    reward=float(reward),
                    done=True,
                    truncated=False,
                    info={
                        "reached_goal": True,
                        "converted_advantage": True,
                        "territory_delta": int(self._territory),
                        "t": int(self._t),
                    },
                )
            reward -= 0.10
            self._ko_risk = _clip(self._ko_risk + 1, 0, 12)
            self._passes = 0

        self._opponent_response()

        reached_goal = False
        lost_game = False
        done = False
        if self._passes >= 2:
            done = True
            if self._territory >= 0:
                reward += 0.45
                reached_goal = True
            else:
                reward -= 0.45
                lost_game = True
        elif self._territory >= int(self.params.target_territory) + 4:
            reward += 1.05
            done = True
            reached_goal = True
        elif self._territory <= -int(self.params.target_territory):
            reward -= 1.05
            done = True
            lost_game = True

        truncated = self._t >= int(self.params.max_steps) and not done
        if truncated:
            reward += 0.40 if self._territory > 0 else (-0.40 if self._territory < 0 else 0.0)

        self._done = bool(done or truncated)
        info: Dict[str, JSONValue] = {
            "reached_goal": bool(reached_goal),
            "lost_game": bool(lost_game),
            "territory_delta": int(self._territory),
            "liberties_delta": int(self._liberties),
            "capture_threat": int(self._capture_threat),
            "ko_risk": int(self._ko_risk),
            "consecutive_passes": int(self._passes),
            "t": int(self._t),
            "action_used": int(a),
        }
        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[int]:
        if self._done:
            return []
        legal: List[int] = [0, 1, 2, 3]
        # Tempo pass is mostly useful when ahead or near late game.
        if self._territory >= 0 or self._t >= max(1, int(self.params.max_steps) - 8):
            legal.append(4)
        # Convert is legal only when close enough to a winning conversion state.
        if self._territory >= int(self.params.target_territory) - 1 and self._ko_risk <= 4:
            legal.append(5)
        return legal if legal else [0]

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode != "ansi":
            return None
        return (
            f"GoWorld t={self._t} territory={self._territory} liberties={self._liberties} "
            f"influence={self._influence} threat={self._capture_threat} ko={self._ko_risk}"
        )

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "done": bool(self._done),
            "t": int(self._t),
            "territory": int(self._territory),
            "liberties": int(self._liberties),
            "influence": int(self._influence),
            "capture_threat": int(self._capture_threat),
            "ko_risk": int(self._ko_risk),
            "tempo": int(self._tempo),
            "passes": int(self._passes),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._done = bool(state.get("done", False))
        self._t = max(0, int(state.get("t", self._t)))
        self._territory = _clip(int(state.get("territory", self._territory)), -20, 20)
        self._liberties = _clip(int(state.get("liberties", self._liberties)), -10, 12)
        self._influence = _clip(int(state.get("influence", self._influence)), -10, 10)
        self._capture_threat = _clip(int(state.get("capture_threat", self._capture_threat)), 0, 12)
        self._ko_risk = _clip(int(state.get("ko_risk", self._ko_risk)), 0, 12)
        self._tempo = _clip(int(state.get("tempo", self._tempo)), 0, 10)
        self._passes = _clip(int(state.get("passes", self._passes)), 0, 3)

    def _opponent_response(self) -> None:
        if self._rng.random() < float(self.params.random_swing):
            self._territory = _clip(self._territory - 1, -20, 20)
        if self._rng.random() < 0.5:
            self._capture_threat = _clip(self._capture_threat + 1, 0, 12)
        self._influence = _clip(self._influence + int(round(self._rng.uniform(-1.0, 1.0))), -10, 10)
        self._tempo = _clip(self._tempo - 1, 0, 10)
        if self._liberties <= 0:
            self._territory = _clip(self._territory - 2, -20, 20)
            self._ko_risk = _clip(self._ko_risk + 1, 0, 12)

    def _make_obs(self) -> JSONValue:
        pressure = _clip(self._capture_threat + self._influence, -16, 16)
        risk = _clip(self._ko_risk - max(0, self._liberties), 0, 16)
        return {
            "territory_delta": int(self._territory),
            "liberties_delta": int(self._liberties),
            "influence": int(self._influence),
            "capture_threat": int(self._capture_threat),
            "ko_risk": int(self._ko_risk),
            "score_delta": int(self._territory),
            "pressure": int(pressure),
            "risk": int(risk),
            "tempo": int(self._tempo),
            "control": int(self._influence),
            "resource": int(self._liberties),
            "consecutive_passes": int(self._passes),
            "t": int(self._t),
        }


class GoWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["strategy", "board_game", "go_like", "transferable_logic"]

    def create(self, spec: VerseSpec) -> Verse:
        return GoWorldVerse(spec)
