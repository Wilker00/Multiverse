"""
core/planning_budget.py

Planner budget manager:
- Planning is treated as a scarce resource.
- Dynamic invoke threshold adapts using recent regret/failure trends per verse.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class PlanningBudgetConfig:
    enabled: bool = True
    base_threshold: float = 0.12
    regret_adaptation: float = 0.30
    budget_per_episode: int = 6
    budget_per_minute: int = 120
    history_window: int = 64


class PlanningBudget:
    def __init__(self, cfg: Optional[PlanningBudgetConfig] = None):
        self.cfg = cfg or PlanningBudgetConfig()
        self._episode_calls_used = 0
        self._global_calls_t_ms: Deque[int] = collections.deque()
        self._verse_regret: Dict[str, Deque[float]] = {}
        self._verse_failures: Dict[str, Deque[int]] = {}

    def reset_episode(self) -> None:
        self._episode_calls_used = 0

    def record_outcome(self, *, verse_name: str, reward: float, failed: bool) -> None:
        v = str(verse_name).strip().lower() or "unknown"
        reg = self._verse_regret.setdefault(v, collections.deque(maxlen=max(8, int(self.cfg.history_window))))
        flg = self._verse_failures.setdefault(v, collections.deque(maxlen=max(8, int(self.cfg.history_window))))
        # Regret proxy for sparse RL: negative reward magnitude.
        regret = max(0.0, -_safe_float(reward, 0.0))
        reg.append(float(regret))
        flg.append(1 if bool(failed) else 0)

    def dynamic_threshold(self, *, verse_name: str, base_threshold: Optional[float] = None) -> float:
        base = _safe_float(base_threshold, self.cfg.base_threshold)
        v = str(verse_name).strip().lower() or "unknown"
        reg = self._verse_regret.get(v)
        flg = self._verse_failures.get(v)
        if not reg:
            return max(0.01, min(0.99, base))

        mean_regret = float(sum(reg) / float(max(1, len(reg))))
        failure_rate = float(sum(flg or []) / float(max(1, len(flg or []))))
        adapt = float(self.cfg.regret_adaptation)
        # More regret/failure => higher threshold => planner triggers earlier.
        t = base + adapt * min(1.0, (mean_regret / 10.0)) + adapt * failure_rate * 0.5
        return max(0.01, min(0.99, float(t)))

    def can_invoke(self, *, verse_name: str, confidence: float, base_threshold: Optional[float] = None) -> bool:
        if not bool(self.cfg.enabled):
            return False
        if self._episode_calls_used >= int(self.cfg.budget_per_episode):
            return False
        now = int(time.time() * 1000)
        cutoff = now - 60_000
        while self._global_calls_t_ms and self._global_calls_t_ms[0] < cutoff:
            self._global_calls_t_ms.popleft()
        if len(self._global_calls_t_ms) >= int(self.cfg.budget_per_minute):
            return False
        dyn_t = self.dynamic_threshold(verse_name=verse_name, base_threshold=base_threshold)
        return bool(_safe_float(confidence, 1.0) < dyn_t)

    def consume(self) -> None:
        self._episode_calls_used += 1
        self._global_calls_t_ms.append(int(time.time() * 1000))

    def snapshot(self, *, verse_name: str) -> Dict[str, float]:
        v = str(verse_name).strip().lower() or "unknown"
        reg = self._verse_regret.get(v) or []
        flg = self._verse_failures.get(v) or []
        mean_regret = float(sum(reg) / float(max(1, len(reg)))) if reg else 0.0
        failure_rate = float(sum(flg) / float(max(1, len(flg)))) if flg else 0.0
        return {
            "episode_calls_used": float(self._episode_calls_used),
            "episode_budget": float(self.cfg.budget_per_episode),
            "minute_calls_used": float(len(self._global_calls_t_ms)),
            "minute_budget": float(self.cfg.budget_per_minute),
            "mean_regret": float(mean_regret),
            "failure_rate": float(failure_rate),
            "dynamic_threshold": float(self.dynamic_threshold(verse_name=v)),
        }

