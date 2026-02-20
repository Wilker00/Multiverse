"""
verses/bridge_world.py

Construction-planning verse: build a bridge across a chasm.
The agent places segments left-to-right across a gap. Each segment
can be cheap (weak) or expensive (strong). Wind gusts test structural
integrity — weak segments may collapse, costing repair time. Agent
must balance speed vs. durability to cross before the deadline.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class BridgeParams:
    bridge_length: int = 8
    max_steps: int = 60
    step_penalty: float = -0.1
    weak_cost: float = 0.0
    strong_cost: float = -0.3
    wind_probability: float = 0.15
    collapse_prob_weak: float = 0.35
    collapse_prob_strong: float = 0.05
    repair_penalty: float = -1.5
    cross_reward: float = 10.0
    partial_reward_per_segment: float = 0.3


# Segment types: 0=empty, 1=weak, 2=strong
EMPTY, WEAK, STRONG = 0, 1, 2


class BridgeWorldVerse:
    """
    Construction-planning verse:
    - Agent builds a bridge of N segments from left to right
    - Each step: place weak (fast/cheap) or strong (slow/expensive) segment, or wait
    - Wind gusts can collapse weak segments (high prob) or strong ones (low prob)
    - Agent crosses once all segments are placed and intact
    - Reward = crossing bonus - costs - collapse penalties
    """

    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in BridgeWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        cfg = self.spec.params
        self.params = BridgeParams(
            bridge_length=max(3, int(cfg.get("bridge_length", 8))),
            max_steps=max(10, int(cfg.get("max_steps", 60))),
            step_penalty=float(cfg.get("step_penalty", -0.1)),
            weak_cost=float(cfg.get("weak_cost", 0.0)),
            strong_cost=float(cfg.get("strong_cost", -0.3)),
            wind_probability=max(0.0, min(1.0, float(cfg.get("wind_probability", 0.15)))),
            collapse_prob_weak=max(0.0, min(1.0, float(cfg.get("collapse_prob_weak", 0.35)))),
            collapse_prob_strong=max(0.0, min(1.0, float(cfg.get("collapse_prob_strong", 0.05)))),
            repair_penalty=float(cfg.get("repair_penalty", -1.5)),
            cross_reward=float(cfg.get("cross_reward", 10.0)),
            partial_reward_per_segment=float(cfg.get("partial_reward_per_segment", 0.3)),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "cursor", "segments_placed", "segments_intact", "weak_count",
                "strong_count", "wind_active", "bridge_complete", "t",
            ],
            subspaces={
                "cursor": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "segments_placed": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "segments_intact": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "weak_count": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "strong_count": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "wind_active": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "bridge_complete": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="BridgeWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=4,
            notes="0=place_weak,1=place_strong,2=wait,3=cross",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._segments: List[int] = []
        self._cursor = 0
        self._t = 0
        self._done = False
        self._wind_active = False
        self._crossed = False

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        self._segments = [EMPTY] * self.params.bridge_length
        self._cursor = 0
        self._t = 0
        self._done = False
        self._wind_active = False
        self._crossed = False

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "bridge_length": self.params.bridge_length,
        }
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = max(0, min(3, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": int(a)}
        done = False

        if a == 0:
            # Place weak segment at cursor
            if self._cursor < self.params.bridge_length and self._segments[self._cursor] == EMPTY:
                self._segments[self._cursor] = WEAK
                reward += float(self.params.weak_cost)
                reward += float(self.params.partial_reward_per_segment)
                info["placed"] = "weak"
                self._cursor += 1
        elif a == 1:
            # Place strong segment at cursor
            if self._cursor < self.params.bridge_length and self._segments[self._cursor] == EMPTY:
                self._segments[self._cursor] = STRONG
                reward += float(self.params.strong_cost)
                reward += float(self.params.partial_reward_per_segment)
                info["placed"] = "strong"
                self._cursor += 1
        elif a == 2:
            # Wait — pure step penalty, but can repair collapsed segments
            # Find first collapsed (empty) segment before cursor and repair it as weak
            for i in range(self._cursor):
                if self._segments[i] == EMPTY:
                    self._segments[i] = WEAK
                    reward += float(self.params.repair_penalty)
                    info["repaired"] = i
                    break
        elif a == 3:
            # Attempt to cross — only succeeds if all segments intact
            intact = all(s != EMPTY for s in self._segments)
            if intact:
                self._crossed = True
                reward += float(self.params.cross_reward)
                info["crossed"] = True
                done = True
            else:
                # Fell through! Big penalty
                reward += float(self.params.repair_penalty) * 2
                info["fell"] = True

        # Wind: may collapse segments
        self._wind_active = False
        if self.params.wind_probability > 0.0 and self._rng.random() < self.params.wind_probability:
            self._wind_active = True
            collapsed = 0
            for i in range(len(self._segments)):
                if self._segments[i] == WEAK and self._rng.random() < self.params.collapse_prob_weak:
                    self._segments[i] = EMPTY
                    collapsed += 1
                elif self._segments[i] == STRONG and self._rng.random() < self.params.collapse_prob_strong:
                    self._segments[i] = EMPTY
                    collapsed += 1
            if collapsed > 0:
                info["wind_collapsed"] = collapsed

        truncated = bool(self._t >= self.params.max_steps and not done)
        self._done = bool(done or truncated)
        info["t"] = int(self._t)
        info["crossed"] = bool(self._crossed)

        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        seg_chars = []
        for i, s in enumerate(self._segments):
            if s == EMPTY:
                ch = "_"
            elif s == WEAK:
                ch = "w"
            elif s == STRONG:
                ch = "S"
            else:
                ch = "?"
            if i == self._cursor and self._cursor < self.params.bridge_length:
                ch = "^"
            seg_chars.append(ch)
        return f"[{''.join(seg_chars)}] cursor={self._cursor} t={self._t}"

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "segments": list(self._segments),
            "cursor": int(self._cursor),
            "t": int(self._t),
            "done": bool(self._done),
            "crossed": bool(self._crossed),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        segs = state.get("segments")
        if isinstance(segs, list) and len(segs) == self.params.bridge_length:
            self._segments = [max(0, min(2, int(s))) for s in segs]
        self._cursor = max(0, min(self.params.bridge_length, int(state.get("cursor", self._cursor))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        self._crossed = bool(state.get("crossed", False))

    def _make_obs(self) -> JSONValue:
        placed = sum(1 for s in self._segments if s != EMPTY)
        intact = placed
        weak_cnt = sum(1 for s in self._segments if s == WEAK)
        strong_cnt = sum(1 for s in self._segments if s == STRONG)
        complete = 1 if all(s != EMPTY for s in self._segments) else 0
        return {
            "cursor": int(self._cursor),
            "segments_placed": int(placed),
            "segments_intact": int(intact),
            "weak_count": int(weak_cnt),
            "strong_count": int(strong_cnt),
            "wind_active": 1 if self._wind_active else 0,
            "bridge_complete": int(complete),
            "t": int(self._t),
        }


class BridgeWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["planning", "construction", "risk_sensitive", "sequential_decision"]

    def create(self, spec: VerseSpec) -> Verse:
        return BridgeWorldVerse(spec)
