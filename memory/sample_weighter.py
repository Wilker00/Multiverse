"""
memory/sample_weighter.py

Sample weighting for offline/imitation training batches.

Weights account for:
- staleness (older samples down-weighted)
- policy age (stale actor-learner lag)
- risky overconfidence (high confidence but bad outcome)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@dataclass
class ReplayWeightConfig:
    enabled: bool = False
    staleness_half_life: float = 200_000.0
    confidence_penalty: float = 0.4
    policy_age_half_life_steps: float = 50_000.0

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "ReplayWeightConfig":
        c = cfg if isinstance(cfg, dict) else {}
        return ReplayWeightConfig(
            enabled=bool(c.get("enabled", False)),
            staleness_half_life=max(1.0, _safe_float(c.get("staleness_half_life", 200_000.0), 200_000.0)),
            confidence_penalty=max(0.0, min(1.0, _safe_float(c.get("confidence_penalty", 0.4), 0.4))),
            policy_age_half_life_steps=max(
                1.0, _safe_float(c.get("policy_age_half_life_steps", 50_000.0), 50_000.0)
            ),
        )


def _exp_half_life_decay(age: float, half_life: float) -> float:
    age = max(0.0, float(age))
    half_life = max(1e-9, float(half_life))
    return float(0.5 ** (age / half_life))


def _extract_runtime_fields(row: Dict[str, Any]) -> Dict[str, float]:
    info = row.get("info")
    info = info if isinstance(info, dict) else {}
    se = info.get("safe_executor")
    se = se if isinstance(se, dict) else {}
    conf = _safe_float(se.get("confidence", 1.0), 1.0)
    danger = _safe_float(se.get("danger", 0.0), 0.0)
    rewound = bool(se.get("rewound", False))
    severe = bool(se.get("severe_penalty", False))
    return {
        "confidence": max(0.0, min(1.0, conf)),
        "danger": max(0.0, min(1.0, danger)),
        "rewound": 1.0 if rewound else 0.0,
        "severe": 1.0 if severe else 0.0,
    }


def compute_sample_weight(
    row: Dict[str, Any],
    *,
    cfg: ReplayWeightConfig,
    now_ms: Optional[int] = None,
) -> float:
    if not bool(cfg.enabled):
        return 1.0

    now = int(now_ms if now_ms is not None else int(time.time() * 1000))
    t_ms = _safe_int(row.get("t_ms", now), now)
    age_ms = max(0.0, float(now - t_ms))
    w_time = _exp_half_life_decay(age_ms, cfg.staleness_half_life)

    policy_age_steps = _safe_float(row.get("policy_age_steps", 0.0), 0.0)
    w_policy_age = _exp_half_life_decay(policy_age_steps, cfg.policy_age_half_life_steps)

    rt = _extract_runtime_fields(row)
    overconf_risk = max(0.0, rt["confidence"] - (1.0 - rt["danger"]))
    if rt["rewound"] > 0:
        overconf_risk = max(overconf_risk, 0.5)
    if rt["severe"] > 0:
        overconf_risk = max(overconf_risk, 0.75)
    w_conf = max(0.05, 1.0 - cfg.confidence_penalty * overconf_risk)

    out = float(w_time * w_policy_age * w_conf)
    return max(0.01, min(2.0, out))

