"""
verses/chess_world.py

Lightweight strategy verse inspired by chess.
This is not a full rules engine; it is an abstract tactical game with
chess-like state factors (material, king safety, center control, tempo).
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
class ChessWorldParams:
    max_steps: int = 80
    win_material: int = 8
    lose_material: int = -8
    random_swing: float = 0.20
    convert_bonus: float = 1.5
    step_penalty: float = -0.01


class ChessWorldVerse:
    """
    Strategic abstraction with 6 actions:
    0 build_position
    1 apply_pressure
    2 tactical_capture
    3 defend_king
    4 tempo_play
    5 convert_advantage
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in ChessWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = ChessWorldParams(
            max_steps=int(self.spec.params.get("max_steps", 80)),
            win_material=int(self.spec.params.get("win_material", 8)),
            lose_material=int(self.spec.params.get("lose_material", -8)),
            random_swing=float(self.spec.params.get("random_swing", 0.20)),
            convert_bonus=float(self.spec.params.get("convert_bonus", 1.5)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.01)),
        )
        self.params.max_steps = max(10, int(self.params.max_steps))
        self.params.win_material = max(3, int(self.params.win_material))
        self.params.lose_material = min(-3, int(self.params.lose_material))
        self.params.random_swing = max(0.0, min(0.9, float(self.params.random_swing)))

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "material_delta",
                "development",
                "king_safety",
                "center_control",
                "score_delta",
                "pressure",
                "risk",
                "tempo",
                "control",
                "resource",
                "phase",
                "t",
            ],
            subspaces={
                "material_delta": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "development": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "king_safety": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "center_control": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "score_delta": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "pressure": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "risk": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "tempo": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "control": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "resource": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "phase": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="Abstract chess-like strategic factors.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=6,
            notes="0=build,1=pressure,2=capture,3=defend,4=tempo,5=convert",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._done = False
        self._t = 0

        self._material = 0
        self._development = 2
        self._king_safety = 5
        self._center_control = 0
        self._tempo = 3
        self._opp_threat = 3
        self._castled = False

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
        self._material = 0
        self._development = 2
        self._king_safety = 5
        self._center_control = 0
        self._tempo = 3
        self._opp_threat = 3
        self._castled = False
        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": self._seed,
                "max_steps": int(self.params.max_steps),
                "win_material": int(self.params.win_material),
                "lose_material": int(self.params.lose_material),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = _clip(int(action), 0, 5)
        self._t += 1
        reward = float(self.params.step_penalty)

        if a == 0:
            self._development = _clip(self._development + 2, 0, 12)
            self._center_control = _clip(self._center_control + 1, -8, 8)
            self._tempo = _clip(self._tempo + 1, 0, 10)
            reward += 0.05
        elif a == 1:
            self._center_control = _clip(self._center_control + 2, -8, 8)
            self._tempo = _clip(self._tempo + 1, 0, 10)
            self._opp_threat = _clip(self._opp_threat + 1, 0, 12)
            reward += 0.03
        elif a == 2:
            p = (
                0.35
                + 0.03 * float(self._development)
                + 0.04 * float(self._center_control)
                - 0.05 * float(max(0, self._opp_threat - self._king_safety))
            )
            if self._rng.random() < max(0.05, min(0.95, p)):
                self._material = _clip(self._material + 2, -16, 16)
                self._tempo = _clip(self._tempo + 1, 0, 10)
                reward += 0.22
            else:
                self._material = _clip(self._material - 1, -16, 16)
                self._king_safety = _clip(self._king_safety - 1, 0, 12)
                reward -= 0.18
        elif a == 3:
            self._king_safety = _clip(self._king_safety + 2, 0, 12)
            self._opp_threat = _clip(self._opp_threat - 2, 0, 12)
            self._tempo = _clip(self._tempo - 1, 0, 10)
            reward += 0.05
        elif a == 4:
            if not self._castled and self._t <= max(2, self.params.max_steps // 2):
                self._castled = True
                self._king_safety = _clip(self._king_safety + 3, 0, 12)
                self._tempo = _clip(self._tempo + 2, 0, 10)
                reward += 0.12
            else:
                self._center_control = _clip(self._center_control + 1, -8, 8)
                reward -= 0.03
        else:
            if self._material >= 3 and self._king_safety >= 4:
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
                        "material_delta": int(self._material),
                        "t": int(self._t),
                    },
                )
            reward -= 0.20
            self._opp_threat = _clip(self._opp_threat + 1, 0, 12)

        self._opponent_response()

        reached_goal = False
        lost_game = False
        done = False
        if self._material >= int(self.params.win_material):
            reward += 1.10
            done = True
            reached_goal = True
        elif self._material <= int(self.params.lose_material):
            reward -= 1.10
            done = True
            lost_game = True

        truncated = self._t >= int(self.params.max_steps) and not done
        if truncated:
            terminal_score = self._score_delta()
            reward += 0.40 if terminal_score > 0 else (-0.40 if terminal_score < 0 else 0.0)

        self._done = bool(done or truncated)
        info: Dict[str, JSONValue] = {
            "reached_goal": bool(reached_goal),
            "lost_game": bool(lost_game),
            "material_delta": int(self._material),
            "king_safety": int(self._king_safety),
            "center_control": int(self._center_control),
            "t": int(self._t),
            "action_used": int(a),
        }
        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[int]:
        if self._done:
            return []
        legal: List[int] = [0, 1, 2, 3, 4]
        # Converting is only legal when there is meaningful advantage to convert.
        if self._material >= 2:
            legal.append(5)
        # Tactical capture is disallowed in extreme king danger states.
        if self._opp_threat >= (self._king_safety + 5) and 2 in legal:
            legal.remove(2)
        # Keep at least one action available.
        return legal if legal else [0]

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode != "ansi":
            return None
        return (
            f"ChessWorld t={self._t} material={self._material} "
            f"dev={self._development} king={self._king_safety} "
            f"center={self._center_control} tempo={self._tempo} opp_threat={self._opp_threat}"
        )

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "done": bool(self._done),
            "t": int(self._t),
            "material": int(self._material),
            "development": int(self._development),
            "king_safety": int(self._king_safety),
            "center_control": int(self._center_control),
            "tempo": int(self._tempo),
            "opp_threat": int(self._opp_threat),
            "castled": bool(self._castled),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._done = bool(state.get("done", False))
        self._t = max(0, int(state.get("t", self._t)))
        self._material = _clip(int(state.get("material", self._material)), -16, 16)
        self._development = _clip(int(state.get("development", self._development)), 0, 12)
        self._king_safety = _clip(int(state.get("king_safety", self._king_safety)), 0, 12)
        self._center_control = _clip(int(state.get("center_control", self._center_control)), -8, 8)
        self._tempo = _clip(int(state.get("tempo", self._tempo)), 0, 10)
        self._opp_threat = _clip(int(state.get("opp_threat", self._opp_threat)), 0, 12)
        self._castled = bool(state.get("castled", self._castled))

    def _opponent_response(self) -> None:
        swing = int(round(self._rng.uniform(-1.0, 1.0)))
        self._center_control = _clip(self._center_control + swing, -8, 8)
        if self._rng.random() < float(self.params.random_swing):
            self._opp_threat = _clip(self._opp_threat + 1, 0, 12)
        if self._opp_threat > self._king_safety + 2:
            self._material = _clip(self._material - 1, -16, 16)
        self._tempo = _clip(self._tempo - 1, 0, 10)

    def _score_delta(self) -> int:
        return int(
            self._material
            + self._center_control // 2
            + self._development // 3
            - max(0, self._opp_threat - self._king_safety)
        )

    def _phase(self) -> int:
        if self._t <= max(5, self.params.max_steps // 3):
            return 0
        if self._t <= max(10, (2 * self.params.max_steps) // 3):
            return 1
        return 2

    def _make_obs(self) -> JSONValue:
        score_delta = self._score_delta()
        pressure = _clip(self._development + self._center_control + self._tempo // 2, -16, 16)
        risk = _clip(self._opp_threat - self._king_safety, 0, 16)
        return {
            "material_delta": int(self._material),
            "development": int(self._development),
            "king_safety": int(self._king_safety),
            "center_control": int(self._center_control),
            "score_delta": int(score_delta),
            "pressure": int(pressure),
            "risk": int(risk),
            "tempo": int(self._tempo),
            "control": int(self._center_control),
            "resource": int(self._development),
            "phase": int(self._phase()),
            "t": int(self._t),
        }


class ChessWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["strategy", "board_game", "chess_like", "transferable_logic"]

    def create(self, spec: VerseSpec) -> Verse:
        return ChessWorldVerse(spec)
