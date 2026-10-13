"""Batch-oriented extension of the stable scalar Verse contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from core.types import JSONValue
from core.verse_base import Verse


@dataclass
class VectorResetResult:
    observations: List[JSONValue]
    infos: List[Dict[str, JSONValue]]


@dataclass
class VectorStepResult:
    observations: List[JSONValue]
    rewards: List[float]
    terminated: List[bool]
    truncated: List[bool]
    infos: List[Dict[str, JSONValue]]

    @property
    def done(self) -> List[bool]:
        return [bool(term or trunc) for term, trunc in zip(self.terminated, self.truncated)]


class VectorizedVerse(Protocol):
    """Optional vector API; scalar Verse remains the compatibility contract."""

    num_envs: int

    def reset_batch(self, seeds: Sequence[Optional[int]]) -> VectorResetResult:
        ...

    def reset_lanes(self, lane_seeds: Dict[int, Optional[int]]) -> VectorResetResult:
        ...

    def step_batch(self, actions: Sequence[JSONValue]) -> VectorStepResult:
        ...

    def close(self) -> None:
        ...


class SerialVectorizedVerse:
    """Compatibility adapter for any collection of scalar Verses."""

    def __init__(self, verses: Sequence[Verse]) -> None:
        if not verses:
            raise ValueError("at least one Verse lane is required")
        self.verses = list(verses)
        self.num_envs = len(self.verses)
        self.spec = self.verses[0].spec
        self.observation_space = self.verses[0].observation_space
        self.action_space = self.verses[0].action_space

    def reset_batch(self, seeds: Sequence[Optional[int]]) -> VectorResetResult:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, received {len(seeds)}")
        observations = []
        infos = []
        for verse, seed in zip(self.verses, seeds):
            if seed is not None:
                verse.seed(int(seed))
            result = verse.reset()
            observations.append(result.obs)
            infos.append(dict(result.info or {}))
        return VectorResetResult(observations, infos)

    def reset_lanes(self, lane_seeds: Dict[int, Optional[int]]) -> VectorResetResult:
        observations: List[JSONValue] = [None] * self.num_envs
        infos: List[Dict[str, JSONValue]] = [{} for _ in range(self.num_envs)]
        for lane, seed in lane_seeds.items():
            index = int(lane)
            if not 0 <= index < self.num_envs:
                raise IndexError(f"lane index out of range: {index}")
            verse = self.verses[index]
            if seed is not None:
                verse.seed(int(seed))
            result = verse.reset()
            observations[index] = result.obs
            infos[index] = dict(result.info or {})
        return VectorResetResult(observations, infos)

    def step_batch(self, actions: Sequence[JSONValue]) -> VectorStepResult:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, received {len(actions)}")
        results = [verse.step(action) for verse, action in zip(self.verses, actions)]
        return VectorStepResult(
            observations=[result.obs for result in results],
            rewards=[float(result.reward) for result in results],
            terminated=[bool(result.done) for result in results],
            truncated=[bool(result.truncated) for result in results],
            infos=[dict(result.info or {}) for result in results],
        )

    def close(self) -> None:
        errors = []
        for verse in self.verses:
            try:
                verse.close()
            except Exception as exc:  # Close every lane before surfacing failures.
                errors.append(exc)
        if errors:
            raise RuntimeError(f"failed to close {len(errors)} vectorized Verse lane(s)") from errors[0]


__all__ = ["VectorResetResult", "VectorStepResult", "VectorizedVerse", "SerialVectorizedVerse"]
