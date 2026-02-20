"""
core/gym_adapter.py

Gym-like adapter over Verse so tools can use the familiar:
- reset(seed, options) -> (obs, info)
- step(action) -> (obs, reward, terminated, truncated, info)
- render(), close()

This module does not require gymnasium to be installed. If gymnasium is
available, observation_space/action_space are exposed as gym spaces.
Otherwise, lightweight fallback space objects are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from core.types import JSONValue, SpaceSpec
from core.verse_base import Verse


@dataclass(frozen=True)
class DiscreteSpace:
    n: int

    def sample(self) -> int:
        import random

        return int(random.randrange(max(1, int(self.n))))

    def contains(self, x: Any) -> bool:
        if not isinstance(x, int):
            return False
        return 0 <= x < int(self.n)


@dataclass(frozen=True)
class BoxSpace:
    low: list[float]
    high: list[float]
    shape: tuple[int, ...]
    dtype: str = "float32"

    def sample(self) -> list[float]:
        import random

        if len(self.low) != len(self.high):
            return []
        return [float(random.uniform(self.low[i], self.high[i])) for i in range(len(self.low))]

    def contains(self, x: Any) -> bool:
        if not isinstance(x, (list, tuple)):
            return False
        if len(x) != len(self.low):
            return False
        for i, v in enumerate(x):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return False
            if fv < float(self.low[i]) or fv > float(self.high[i]):
                return False
        return True


@dataclass(frozen=True)
class DictSpace:
    spaces: Dict[str, Any]

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        for k, sub in self.spaces.items():
            if k not in x:
                return False
            if hasattr(sub, "contains") and not sub.contains(x[k]):
                return False
        return True


def _to_fallback_space(spec: SpaceSpec) -> Any:
    if spec.type == "discrete":
        return DiscreteSpace(n=int(spec.n or 1))
    if spec.type == "continuous":
        low = [float(v) for v in (spec.low or [])]
        high = [float(v) for v in (spec.high or [])]
        shape = tuple(int(v) for v in (spec.shape or (len(low),)))
        return BoxSpace(low=low, high=high, shape=shape, dtype=str(spec.dtype or "float32"))
    if spec.type in ("vector", "image"):
        dim = int(spec.shape[0]) if spec.shape else 1
        return BoxSpace(
            low=[-1.0e9] * dim,
            high=[1.0e9] * dim,
            shape=tuple(int(v) for v in (spec.shape or (dim,))),
            dtype=str(spec.dtype or "float32"),
        )
    if spec.type == "dict":
        sub = spec.subspaces or {}
        return DictSpace(spaces={k: _to_fallback_space(v) for k, v in sub.items()})
    # Fallback for unknown types
    return DictSpace(spaces={})


def to_gym_space(spec: SpaceSpec) -> Any:
    """
    Convert SpaceSpec to gymnasium space when available.
    Falls back to lightweight internal space types otherwise.
    """
    try:
        from gymnasium import spaces  # type: ignore
    except Exception:
        spaces = None

    if spaces is None:
        return _to_fallback_space(spec)

    if spec.type == "discrete":
        return spaces.Discrete(int(spec.n or 1))
    if spec.type == "continuous":
        import numpy as np  # type: ignore

        low = np.array(spec.low or [], dtype=np.float32)
        high = np.array(spec.high or [], dtype=np.float32)
        shape = tuple(int(v) for v in (spec.shape or (len(low),)))
        if low.size == 0:
            low = np.full(shape, -1.0, dtype=np.float32)
            high = np.full(shape, 1.0, dtype=np.float32)
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    if spec.type in ("vector", "image"):
        import numpy as np  # type: ignore

        shape = tuple(int(v) for v in (spec.shape or (1,)))
        return spaces.Box(
            low=np.full(shape, -1.0e9, dtype=np.float32),
            high=np.full(shape, 1.0e9, dtype=np.float32),
            dtype=np.float32,
        )
    if spec.type == "dict":
        sub = spec.subspaces or {}
        return spaces.Dict({k: to_gym_space(v) for k, v in sub.items()})
    return spaces.Dict({})


class VerseGymAdapter:
    """
    Adapter that exposes a Gymnasium-style API over a Verse instance.
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"]}

    def __init__(self, verse: Verse):
        self.verse = verse
        self.observation_space = to_gym_space(verse.observation_space)
        self.action_space = to_gym_space(verse.action_space)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, JSONValue]] = None,
    ) -> tuple[JSONValue, Dict[str, JSONValue]]:
        del options  # Reserved for compatibility.
        if seed is not None:
            self.verse.seed(seed)
        rr = self.verse.reset()
        return rr.obs, dict(rr.info or {})

    def step(self, action: JSONValue) -> tuple[JSONValue, float, bool, bool, Dict[str, JSONValue]]:
        sr = self.verse.step(action)
        return sr.obs, float(sr.reward), bool(sr.done), bool(sr.truncated), dict(sr.info or {})

    def render(self, mode: str = "ansi") -> Optional[Any]:
        return self.verse.render(mode=mode)

    def close(self) -> None:
        self.verse.close()

    @property
    def unwrapped(self) -> Verse:
        return self.verse

