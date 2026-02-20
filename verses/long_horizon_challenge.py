"""
verses/long_horizon_challenge.py

Long-horizon sparse-reward challenge:
- Find key
- Unlock door
- Navigate checkpoints
- Collect treasure

Reward is only provided at final completion.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class LongHorizonParams:
    width: int = 30
    max_steps: int = 120
    key_pos: int = 6
    door_pos: int = 14
    treasure_pos: int = 29
    checkpoint_1: int = 18
    checkpoint_2: int = 22
    checkpoint_3: int = 26
    final_reward: float = 100.0


class LongHorizonChallengeVerse:
    """
    Actions:
    - 0: move left
    - 1: move right
    - 2: interact
    """

    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in LongHorizonChallengeFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)
        cfg = self.spec.params
        width = max(12, int(cfg.get("width", 30)))
        key_pos = max(1, min(width - 6, int(cfg.get("key_pos", 6))))
        door_pos = max(key_pos + 2, min(width - 4, int(cfg.get("door_pos", 14))))
        cp1 = max(door_pos + 1, min(width - 3, int(cfg.get("checkpoint_1", 18))))
        cp2 = max(cp1 + 1, min(width - 2, int(cfg.get("checkpoint_2", 22))))
        cp3 = max(cp2 + 1, min(width - 1, int(cfg.get("checkpoint_3", 26))))
        treasure = max(cp3 + 1, min(width - 1, int(cfg.get("treasure_pos", width - 1))))
        self.params = LongHorizonParams(
            width=width,
            max_steps=max(40, int(cfg.get("max_steps", 120))),
            key_pos=key_pos,
            door_pos=door_pos,
            treasure_pos=treasure,
            checkpoint_1=cp1,
            checkpoint_2=cp2,
            checkpoint_3=cp3,
            final_reward=float(cfg.get("final_reward", 100.0)),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "pos",
                "t",
                "has_key",
                "door_unlocked",
                "checkpoint_idx",
                "checkpoint_total",
                "treasure_visible",
            ],
            subspaces={
                "pos": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "has_key": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "door_unlocked": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "checkpoint_idx": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "checkpoint_total": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "treasure_visible": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="Sparse-reward long-horizon chain with staged prerequisites.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=3,
            notes="0=left, 1=right, 2=interact",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._pos = 0
        self._t = 0
        self._done = False
        self._has_key = False
        self._door_unlocked = False
        self._checkpoint_idx = 0
        self._checkpoints: List[int] = [
            int(self.params.checkpoint_1),
            int(self.params.checkpoint_2),
            int(self.params.checkpoint_3),
        ]

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)
        self._pos = 0
        self._t = 0
        self._done = False
        self._has_key = False
        self._door_unlocked = False
        self._checkpoint_idx = 0
        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": self._seed,
                "subtasks": ["find_key", "unlock_door", "navigate_maze", "collect_treasure"],
                "horizon": int(self.params.max_steps),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = max(0, min(2, int(action)))
        self._t += 1

        # movement
        if a == 0:
            self._pos = max(0, int(self._pos - 1))
        elif a == 1:
            if (not self._door_unlocked) and int(self._pos) >= int(self.params.door_pos):
                # locked door blocks progress past door tile
                self._pos = int(self.params.door_pos)
            else:
                self._pos = min(int(self.params.width - 1), int(self._pos + 1))
        else:
            # interact
            if (not self._has_key) and int(self._pos) == int(self.params.key_pos):
                self._has_key = True
            elif self._has_key and (not self._door_unlocked) and int(self._pos) == int(self.params.door_pos):
                self._door_unlocked = True
            elif (
                self._door_unlocked
                and self._checkpoint_idx >= len(self._checkpoints)
                and int(self._pos) == int(self.params.treasure_pos)
            ):
                # Treasure collection only possible after all checkpoints.
                self._done = True
                return StepResult(
                    obs=self._make_obs(),
                    reward=float(self.params.final_reward),
                    done=True,
                    truncated=False,
                    info={
                        "reached_goal": True,
                        "current_subtask": "collect_treasure",
                        "steps": int(self._t),
                    },
                )

        # checkpoint progression (automatically when position reached)
        if self._door_unlocked and self._checkpoint_idx < len(self._checkpoints):
            if int(self._pos) >= int(self._checkpoints[self._checkpoint_idx]):
                self._checkpoint_idx += 1

        done = False
        truncated = bool(self._t >= int(self.params.max_steps))
        self._done = bool(done or truncated)
        info: Dict[str, JSONValue] = {
            "reached_goal": False,
            "has_key": bool(self._has_key),
            "door_unlocked": bool(self._door_unlocked),
            "checkpoint_idx": int(self._checkpoint_idx),
            "checkpoint_total": int(len(self._checkpoints)),
            "current_subtask": self._current_subtask(),
            "steps": int(self._t),
        }
        return StepResult(self._make_obs(), 0.0, bool(done), bool(truncated), info)

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode != "ansi":
            return None
        cells = ["." for _ in range(int(self.params.width))]
        cells[int(self.params.key_pos)] = "K"
        cells[int(self.params.door_pos)] = "D" if not self._door_unlocked else "d"
        cells[int(self.params.treasure_pos)] = "T"
        for i, cp in enumerate(self._checkpoints):
            label = str(i + 1)
            if 0 <= int(cp) < len(cells):
                cells[int(cp)] = label
        cells[int(self._pos)] = "A"
        return (
            "|" + "".join(cells) + "| "
            f"t={self._t} key={int(self._has_key)} door={int(self._door_unlocked)} "
            f"cp={self._checkpoint_idx}/{len(self._checkpoints)}"
        )

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "pos": int(self._pos),
            "t": int(self._t),
            "done": bool(self._done),
            "has_key": bool(self._has_key),
            "door_unlocked": bool(self._door_unlocked),
            "checkpoint_idx": int(self._checkpoint_idx),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._pos = max(0, min(int(self.params.width - 1), int(state.get("pos", self._pos))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        self._has_key = bool(state.get("has_key", self._has_key))
        self._door_unlocked = bool(state.get("door_unlocked", self._door_unlocked))
        self._checkpoint_idx = max(0, min(len(self._checkpoints), int(state.get("checkpoint_idx", self._checkpoint_idx))))

    def _current_subtask(self) -> str:
        if not self._has_key:
            return "find_key"
        if not self._door_unlocked:
            return "unlock_door"
        if self._checkpoint_idx < len(self._checkpoints):
            return "navigate_maze"
        return "collect_treasure"

    def _make_obs(self) -> JSONValue:
        return {
            "pos": int(self._pos),
            "t": int(self._t),
            "has_key": 1 if self._has_key else 0,
            "door_unlocked": 1 if self._door_unlocked else 0,
            "checkpoint_idx": int(self._checkpoint_idx),
            "checkpoint_total": int(len(self._checkpoints)),
            "treasure_visible": 1 if self._checkpoint_idx >= len(self._checkpoints) else 0,
        }


class LongHorizonChallengeFactory:
    @property
    def tags(self) -> List[str]:
        return [
            "long_horizon",
            "sparse_reward",
            "credit_assignment",
            "sequential_decision",
        ]

    def create(self, spec: VerseSpec) -> Verse:
        return LongHorizonChallengeVerse(spec)
