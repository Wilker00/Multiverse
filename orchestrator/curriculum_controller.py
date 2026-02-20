"""
orchestrator/curriculum_controller.py

Adaptive curriculum controller driven by monitoring signals.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm(x: Any) -> str:
    return str(x).strip().lower()


@dataclass
class CurriculumConfig:
    enabled: bool = True
    state_path: str = os.path.join("models", "curriculum_adjustments.json")
    plateau_window: int = 5
    step_size: float = 0.05
    collapse_threshold: float = 0.20
    min_noise: float = 0.0
    max_noise: float = 0.35
    min_partial_obs: float = 0.0
    max_partial_obs: float = 0.75
    min_distractors: int = 0
    max_distractors: int = 6

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "CurriculumConfig":
        c = cfg if isinstance(cfg, dict) else {}
        return CurriculumConfig(
            enabled=bool(c.get("enabled", True)),
            state_path=str(c.get("state_path", os.path.join("models", "curriculum_adjustments.json"))),
            plateau_window=max(3, int(c.get("plateau_window", 5))),
            step_size=max(0.01, min(0.5, _safe_float(c.get("step_size", 0.05), 0.05))),
            collapse_threshold=max(0.01, min(0.99, _safe_float(c.get("collapse_threshold", 0.20), 0.20))),
        )


class CurriculumController:
    def __init__(self, cfg: Optional[CurriculumConfig] = None):
        self.cfg = cfg or CurriculumConfig()
        os.makedirs(os.path.dirname(self.cfg.state_path) or ".", exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        if not os.path.isfile(self.cfg.state_path):
            return {"version": "v1", "verses": {}, "updated_at_ms": 0}
        with open(self.cfg.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": "v1", "verses": {}, "updated_at_ms": 0}
        if not isinstance(data.get("verses"), dict):
            data["verses"] = {}
        return data

    def _save(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["updated_at_ms"] = int(time.time() * 1000)
        with open(self.cfg.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def update_from_signal(self, *, verse_name: str, success_rate: float, mean_return: float) -> Dict[str, Any]:
        payload = self._load()
        verses = payload.get("verses", {})
        v = _norm(verse_name)
        rec = verses.get(v)
        if not isinstance(rec, dict):
            rec = {
                "history": [],
                "noise": 0.0,
                "stochasticity": 0.0,
                "partial_observability": 0.0,
                "distractors": 0,
            }

        hist = rec.get("history")
        hist = hist if isinstance(hist, list) else []
        hist.append(
            {
                "t_ms": int(time.time() * 1000),
                "success_rate": float(success_rate),
                "mean_return": float(mean_return),
            }
        )
        hist = hist[-max(8, self.cfg.plateau_window * 4) :]
        rec["history"] = hist

        plateau = self._is_plateau(hist)
        collapse = bool(float(success_rate) < float(self.cfg.collapse_threshold))
        step = float(self.cfg.step_size)

        if collapse:
            rec["noise"] = max(self.cfg.min_noise, _safe_float(rec.get("noise", 0.0), 0.0) - step * 1.5)
            rec["stochasticity"] = max(0.0, _safe_float(rec.get("stochasticity", 0.0), 0.0) - step * 1.5)
            rec["partial_observability"] = max(
                self.cfg.min_partial_obs, _safe_float(rec.get("partial_observability", 0.0), 0.0) - step
            )
            rec["distractors"] = max(self.cfg.min_distractors, int(rec.get("distractors", 0)) - 1)
            rec["mode"] = "collapse_backoff"
        elif plateau:
            rec["noise"] = min(self.cfg.max_noise, _safe_float(rec.get("noise", 0.0), 0.0) + step)
            rec["stochasticity"] = min(1.0, _safe_float(rec.get("stochasticity", 0.0), 0.0) + step)
            rec["partial_observability"] = min(
                self.cfg.max_partial_obs, _safe_float(rec.get("partial_observability", 0.0), 0.0) + step * 0.5
            )
            rec["distractors"] = min(self.cfg.max_distractors, int(rec.get("distractors", 0)) + 1)
            rec["mode"] = "plateau_harder"
        else:
            rec["mode"] = "stable"

        verses[v] = rec
        payload["verses"] = verses
        self._save(payload)
        return rec

    def get_adjustment(self, *, verse_name: str) -> Dict[str, Any]:
        payload = self._load()
        rec = payload.get("verses", {}).get(_norm(verse_name), {})
        if not isinstance(rec, dict):
            return {}
        return {
            "noise": float(_safe_float(rec.get("noise", 0.0), 0.0)),
            "stochasticity": float(_safe_float(rec.get("stochasticity", 0.0), 0.0)),
            "partial_observability": float(_safe_float(rec.get("partial_observability", 0.0), 0.0)),
            "distractors": int(rec.get("distractors", 0) or 0),
            "mode": str(rec.get("mode", "stable")),
        }

    def _is_plateau(self, history: List[Dict[str, Any]]) -> bool:
        w = int(self.cfg.plateau_window)
        if len(history) < w:
            return False
        win = history[-w:]
        s = [_safe_float(x.get("success_rate", 0.0), 0.0) for x in win]
        r = [_safe_float(x.get("mean_return", 0.0), 0.0) for x in win]
        # Plateau means low variance + low trend.
        s_var = statistics.pvariance(s) if len(s) > 1 else 0.0
        r_var = statistics.pvariance(r) if len(r) > 1 else 0.0
        s_trend = abs(s[-1] - s[0]) if len(s) > 1 else 0.0
        r_trend = abs(r[-1] - r[0]) if len(r) > 1 else 0.0
        return bool(s_var < 0.001 and r_var < 0.05 and s_trend < 0.03 and r_trend < 0.10)


def load_curriculum_adjustments(path: str = os.path.join("models", "curriculum_adjustments.json")) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        verses = data.get("verses", {})
        return verses if isinstance(verses, dict) else {}
    except Exception:
        return {}


def apply_curriculum_params(*, verse_name: str, params: Dict[str, Any], adjustments: Dict[str, Any]) -> Dict[str, Any]:
    v = _norm(verse_name)
    rec = adjustments.get(v)
    if not isinstance(rec, dict):
        return dict(params)
    out = dict(params)
    # Map generic curriculum controls to existing verse params where possible.
    if "noise" in rec and "action_noise" in out:
        out["action_noise"] = max(0.0, _safe_float(out.get("action_noise", 0.0), 0.0) + _safe_float(rec["noise"], 0.0))
    if "stochasticity" in rec and "adr_jitter" in out:
        out["adr_jitter"] = max(0.0, _safe_float(out.get("adr_jitter", 0.0), 0.0) + 0.5 * _safe_float(rec["stochasticity"], 0.0))
    if "partial_observability" in rec:
        # Generic signal; verses can consume this when implemented.
        out["partial_observability"] = _safe_float(rec["partial_observability"], 0.0)
    if "distractors" in rec:
        out["distractors"] = int(rec["distractors"])
    return out

