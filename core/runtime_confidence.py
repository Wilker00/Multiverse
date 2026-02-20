"""
core/runtime_confidence.py

Lightweight online confidence monitor used by SafeExecutor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class ConfidenceStatus:
    window_size: int
    competence_rate: float
    mean_confidence: float
    mean_danger: float
    mean_regret: float
    failure_rate: float
    in_incompetence_zone: bool


class RuntimeConfidenceMonitor:
    def __init__(
        self,
        *,
        window_size: int,
        min_competence_rate: float,
        min_action_confidence: float,
        danger_threshold: float,
    ):
        self.window_size = max(1, int(window_size))
        self.min_competence_rate = max(0.0, min(1.0, float(min_competence_rate)))
        self.min_action_confidence = max(0.0, min(1.0, float(min_action_confidence)))
        self.danger_threshold = max(0.0, min(1.0, float(danger_threshold)))
        self._competence: Deque[int] = deque(maxlen=self.window_size)
        self._confidence: Deque[float] = deque(maxlen=self.window_size)
        self._danger: Deque[float] = deque(maxlen=self.window_size)
        self._regret: Deque[float] = deque(maxlen=self.window_size)
        self._failure: Deque[int] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._competence.clear()
        self._confidence.clear()
        self._danger.clear()
        self._regret.clear()
        self._failure.clear()

    def observe(
        self,
        *,
        confidence: float,
        danger: float,
        dangerous_outcome: bool,
        reward: float = 0.0,
    ) -> None:
        c = max(0.0, min(1.0, float(confidence)))
        d = max(0.0, min(1.0, float(danger)))
        # Regret proxy for sparse RL loops: negative reward magnitude.
        reg = max(0.0, -float(reward))
        self._confidence.append(c)
        self._danger.append(d)
        self._competence.append(0 if bool(dangerous_outcome) else 1)
        self._regret.append(reg)
        self._failure.append(1 if bool(dangerous_outcome) else 0)

    def status(self) -> ConfidenceStatus:
        n = max(1, len(self._competence))
        competence_rate = float(sum(self._competence)) / float(n)
        mean_confidence = float(sum(self._confidence) / float(max(1, len(self._confidence))))
        mean_danger = float(sum(self._danger) / float(max(1, len(self._danger))))
        mean_regret = float(sum(self._regret) / float(max(1, len(self._regret))))
        failure_rate = float(sum(self._failure) / float(max(1, len(self._failure))))
        zone = False
        if len(self._competence) >= self.window_size:
            zone = (
                (competence_rate < self.min_competence_rate)
                or (mean_confidence < self.min_action_confidence)
                or (mean_danger >= self.danger_threshold)
            )
        return ConfidenceStatus(
            window_size=self.window_size,
            competence_rate=competence_rate,
            mean_confidence=mean_confidence,
            mean_danger=mean_danger,
            mean_regret=mean_regret,
            failure_rate=failure_rate,
            in_incompetence_zone=bool(zone),
        )

    def adaptive_planner_threshold(self, *, base_threshold: float, regret_adaptation: float) -> float:
        s = self.status()
        t = float(base_threshold) + float(regret_adaptation) * min(1.0, float(s.mean_regret) / 10.0)
        t += float(regret_adaptation) * 0.5 * float(s.failure_rate)
        return max(0.01, min(0.99, t))
