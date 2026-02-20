"""
verses/uno_world.py

Lightweight strategy verse inspired by UNO.
This is an abstract card-pressure game, not a full UNO rules engine.
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
class UnoWorldParams:
    start_hand: int = 7
    opp_start_hand: int = 7
    max_steps: int = 70
    random_swing: float = 0.25
    convert_bonus: float = 1.6
    step_penalty: float = -0.01


class UnoWorldVerse:
    """
    Strategic abstraction with 6 actions:
    0 play_safe
    1 apply_pressure
    2 tactical_combo
    3 defend
    4 tempo_play
    5 convert_finish
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in UnoWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = UnoWorldParams(
            start_hand=int(self.spec.params.get("start_hand", 7)),
            opp_start_hand=int(self.spec.params.get("opp_start_hand", 7)),
            max_steps=int(self.spec.params.get("max_steps", 70)),
            random_swing=float(self.spec.params.get("random_swing", 0.25)),
            convert_bonus=float(self.spec.params.get("convert_bonus", 1.6)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.01)),
        )
        self.params.start_hand = max(2, int(self.params.start_hand))
        self.params.opp_start_hand = max(2, int(self.params.opp_start_hand))
        self.params.max_steps = max(10, int(self.params.max_steps))
        self.params.random_swing = max(0.0, min(0.9, float(self.params.random_swing)))

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "my_cards",
                "opp_cards",
                "color_control",
                "action_charge",
                "draw_pressure",
                "uno_ready",
                "score_delta",
                "pressure",
                "risk",
                "tempo",
                "control",
                "resource",
                "t",
            ],
            subspaces={
                "my_cards": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "opp_cards": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "color_control": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "action_charge": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "draw_pressure": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "uno_ready": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "score_delta": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "pressure": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "risk": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "tempo": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "control": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "resource": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="Abstract UNO-like card pressure factors.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=6,
            notes="0=safe,1=pressure,2=combo,3=defend,4=tempo,5=convert",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._done = False
        self._t = 0

        self._my_cards = int(self.params.start_hand)
        self._opp_cards = int(self.params.opp_start_hand)
        self._color_control = 0
        self._action_charge = 1
        self._draw_pressure = 1
        self._tempo = 2

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
        self._my_cards = int(self.params.start_hand)
        self._opp_cards = int(self.params.opp_start_hand)
        self._color_control = 0
        self._action_charge = 1
        self._draw_pressure = 1
        self._tempo = 2
        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": self._seed,
                "max_steps": int(self.params.max_steps),
                "start_hand": int(self.params.start_hand),
                "opp_start_hand": int(self.params.opp_start_hand),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = _clip(int(action), 0, 5)
        self._t += 1
        reward = float(self.params.step_penalty)

        skip_opponent = False
        if a == 0:
            if self._color_control >= -1 and self._my_cards > 0:
                self._my_cards = max(0, self._my_cards - 1)
                reward += 0.05
            else:
                self._my_cards += 1
                reward -= 0.04
            self._action_charge = _clip(self._action_charge + 1, 0, 4)
        elif a == 1:
            if self._action_charge >= 1 and self._my_cards > 0:
                self._action_charge -= 1
                self._my_cards = max(0, self._my_cards - 1)
                self._opp_cards += 2
                self._draw_pressure = _clip(self._draw_pressure + 1, 0, 10)
                reward += 0.12
            else:
                self._my_cards += 1
                reward -= 0.08
        elif a == 2:
            p = (
                0.35
                + 0.09 * float(self._color_control)
                + 0.07 * float(self._action_charge)
                - 0.08 * float(self._draw_pressure)
            )
            if self._rng.random() < max(0.05, min(0.95, p)):
                self._my_cards = max(0, self._my_cards - 2)
                self._opp_cards += 1
                reward += 0.18
            else:
                self._my_cards += 1
                self._draw_pressure = _clip(self._draw_pressure + 1, 0, 10)
                reward -= 0.12
        elif a == 3:
            self._draw_pressure = _clip(self._draw_pressure - 2, 0, 10)
            self._color_control = _clip(self._color_control + 1, -5, 5)
            self._tempo = _clip(self._tempo - 1, 0, 10)
            reward += 0.04
        elif a == 4:
            if self._action_charge >= 1 and self._my_cards > 0:
                self._action_charge -= 1
                self._my_cards = max(0, self._my_cards - 1)
                self._tempo = _clip(self._tempo + 2, 0, 10)
                skip_opponent = True
                reward += 0.09
            else:
                reward -= 0.05
        else:
            if self._my_cards <= 1 and self._color_control >= 0:
                self._my_cards = 0
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
                        "my_cards": int(self._my_cards),
                        "opp_cards": int(self._opp_cards),
                        "t": int(self._t),
                    },
                )
            self._my_cards += 2
            self._draw_pressure = _clip(self._draw_pressure + 1, 0, 10)
            reward -= 0.20

        self._opponent_response(skip_opponent=bool(skip_opponent))

        reached_goal = False
        lost_game = False
        done = False
        if self._my_cards <= 0:
            reward += 1.05
            done = True
            reached_goal = True
        elif self._opp_cards <= 0:
            reward -= 1.05
            done = True
            lost_game = True

        truncated = self._t >= int(self.params.max_steps) and not done
        if truncated:
            if self._my_cards < self._opp_cards:
                reward += 0.40
            elif self._my_cards > self._opp_cards:
                reward -= 0.40

        self._done = bool(done or truncated)
        info: Dict[str, JSONValue] = {
            "reached_goal": bool(reached_goal),
            "lost_game": bool(lost_game),
            "my_cards": int(self._my_cards),
            "opp_cards": int(self._opp_cards),
            "draw_pressure": int(self._draw_pressure),
            "action_charge": int(self._action_charge),
            "color_control": int(self._color_control),
            "t": int(self._t),
            "action_used": int(a),
        }
        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[int]:
        if self._done:
            return []
        legal: List[int] = [0, 2, 3]
        # Pressure and tempo require action charge.
        if self._action_charge >= 1 and self._my_cards > 0:
            legal.append(1)
            legal.append(4)
        # Convert finish is only legal when near UNO state.
        if self._my_cards <= 2:
            legal.append(5)
        legal = sorted(set(int(a) for a in legal))
        return legal if legal else [0]

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode != "ansi":
            return None
        return (
            f"UnoWorld t={self._t} my={self._my_cards} opp={self._opp_cards} "
            f"control={self._color_control} charge={self._action_charge} pressure={self._draw_pressure}"
        )

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "done": bool(self._done),
            "t": int(self._t),
            "my_cards": int(self._my_cards),
            "opp_cards": int(self._opp_cards),
            "color_control": int(self._color_control),
            "action_charge": int(self._action_charge),
            "draw_pressure": int(self._draw_pressure),
            "tempo": int(self._tempo),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._done = bool(state.get("done", False))
        self._t = max(0, int(state.get("t", self._t)))
        self._my_cards = _clip(int(state.get("my_cards", self._my_cards)), 0, 200)
        self._opp_cards = _clip(int(state.get("opp_cards", self._opp_cards)), 0, 200)
        self._color_control = _clip(int(state.get("color_control", self._color_control)), -5, 5)
        self._action_charge = _clip(int(state.get("action_charge", self._action_charge)), 0, 4)
        self._draw_pressure = _clip(int(state.get("draw_pressure", self._draw_pressure)), 0, 10)
        self._tempo = _clip(int(state.get("tempo", self._tempo)), 0, 10)

    def _opponent_response(self, *, skip_opponent: bool) -> None:
        if not skip_opponent and self._opp_cards > 0:
            if self._draw_pressure > 0 and self._rng.random() < 0.50:
                self._opp_cards += 1
            elif self._rng.random() < 0.65:
                self._opp_cards = max(0, self._opp_cards - 1)

        if self._rng.random() < float(self.params.random_swing):
            self._color_control = _clip(self._color_control + int(round(self._rng.uniform(-1.0, 1.0))), -5, 5)
        self._draw_pressure = _clip(self._draw_pressure - 1, 0, 10)
        self._tempo = _clip(self._tempo - 1, 0, 10)

    def _make_obs(self) -> JSONValue:
        score_delta = _clip(self._opp_cards - self._my_cards, -200, 200)
        risk = _clip(self._draw_pressure - self._color_control, 0, 20)
        return {
            "my_cards": int(self._my_cards),
            "opp_cards": int(self._opp_cards),
            "color_control": int(self._color_control),
            "action_charge": int(self._action_charge),
            "draw_pressure": int(self._draw_pressure),
            "uno_ready": int(self._my_cards <= 1),
            "score_delta": int(score_delta),
            "pressure": int(self._draw_pressure),
            "risk": int(risk),
            "tempo": int(self._tempo),
            "control": int(self._color_control),
            "resource": int(self._action_charge),
            "t": int(self._t),
        }


class UnoWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["strategy", "card_game", "uno_like", "transferable_logic"]

    def create(self, spec: VerseSpec) -> Verse:
        return UnoWorldVerse(spec)
