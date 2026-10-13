"""Seeded, opt-in task-level physics and appearance randomization."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DomainRandomizationSample:
    actuator_gain: float = 1.0
    goal_offset: Tuple[float, float] = (0.0, 0.0)
    visual_tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actuator_gain": float(self.actuator_gain),
            "goal_offset": [float(v) for v in self.goal_offset],
            "visual_tint": [float(v) for v in self.visual_tint],
        }


def _range(value: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = float(value[0]), float(value[1])
        return (min(lower, upper), max(lower, upper))
    return fallback


def sample_domain_randomization(seed: Optional[int], config: Any) -> DomainRandomizationSample:
    """Sample one deterministic lane's actuator, target, and visual variation.

    Example configuration: ``{"enabled": true, "actuator_gain": [0.9, 1.1],
    "goal_jitter": 0.04, "visual_color_jitter": 0.06}``.
    """
    if isinstance(config, bool):
        enabled = config
        options: Dict[str, Any] = {}
    elif isinstance(config, dict):
        options = dict(config)
        enabled = bool(options.get("enabled", True))
    else:
        enabled = False
        options = {}
    if not enabled:
        return DomainRandomizationSample()

    rng = random.Random(seed)
    gain_low, gain_high = _range(options.get("actuator_gain"), (0.9, 1.1))
    goal_jitter = max(0.0, float(options.get("goal_jitter", 0.04)))
    tint_jitter = max(0.0, min(0.5, float(options.get("visual_color_jitter", 0.06))))
    return DomainRandomizationSample(
        actuator_gain=float(rng.uniform(gain_low, gain_high)),
        goal_offset=(rng.uniform(-goal_jitter, goal_jitter), rng.uniform(-goal_jitter, goal_jitter)),
        visual_tint=tuple(float(rng.uniform(1.0 - tint_jitter, 1.0 + tint_jitter)) for _ in range(3)),
    )


__all__ = ["DomainRandomizationSample", "sample_domain_randomization"]
