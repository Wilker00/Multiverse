"""
core/skill_contracts.py

PathNet-style skill contracts:
- Contracts are immutable by default.
- Updates are allowed only via strict improvement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm(x: Any) -> str:
    return str(x).strip().lower()


def _validate_allowed_keys(*, payload: Dict[str, Any], allowed: List[str], label: str) -> None:
    allow = {str(k) for k in allowed}
    unknown = sorted(str(k) for k in payload.keys() if str(k) not in allow)
    if not unknown:
        return
    raise ValueError(f"{label} has unknown keys: {unknown}. Allowed keys: {sorted(allow)}")


@dataclass
class ContractConfig:
    enabled: bool = True
    path: str = os.path.join("models", "skill_contracts.json")
    strict_improvement_delta: float = 0.01
    strict_config_validation: bool = True
    robustness_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "min_success_rate": 0.5,
            "min_mean_return": 0.0,
            "max_safety_violation_rate": 0.5,
        }
    )

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "ContractConfig":
        data = cfg if isinstance(cfg, dict) else {}
        strict_cfg = bool(data.get("strict_config_validation", True))
        if strict_cfg:
            _validate_allowed_keys(
                payload=data,
                allowed=[
                    "enabled",
                    "path",
                    "strict_improvement_delta",
                    "robustness_thresholds",
                    "strict_config_validation",
                ],
                label="skill_contract config",
            )
        thresholds_raw = data.get("robustness_thresholds")
        if strict_cfg and thresholds_raw is not None and not isinstance(thresholds_raw, dict):
            raise ValueError("skill_contract.robustness_thresholds must be a dict")
        thresholds = (
            dict(thresholds_raw)
            if isinstance(thresholds_raw, dict)
            else {
                "min_success_rate": 0.5,
                "min_mean_return": 0.0,
                "max_safety_violation_rate": 0.5,
            }
        )
        if strict_cfg:
            _validate_allowed_keys(
                payload=thresholds,
                allowed=[
                    "min_success_rate",
                    "min_mean_return",
                    "max_safety_violation_rate",
                ],
                label="skill_contract.robustness_thresholds",
            )
        return cls(
            enabled=bool(data.get("enabled", True)),
            path=str(data.get("path", os.path.join("models", "skill_contracts.json"))),
            strict_improvement_delta=float(data.get("strict_improvement_delta", 0.01)),
            strict_config_validation=bool(strict_cfg),
            robustness_thresholds={
                "min_success_rate": _safe_float(thresholds.get("min_success_rate", 0.5), 0.5),
                "min_mean_return": _safe_float(thresholds.get("min_mean_return", 0.0), 0.0),
                "max_safety_violation_rate": _safe_float(
                    thresholds.get("max_safety_violation_rate", 0.5), 0.5
                ),
            },
        )


@dataclass
class SkillContract:
    verse_name: str
    skill_tag: str
    required_metrics: Dict[str, float]
    allowed_action_bounds: Dict[str, float]
    safety_invariants: Dict[str, Any]
    created_at_ms: int
    updated_at_ms: int
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_name": self.verse_name,
            "skill_tag": self.skill_tag,
            "required_metrics": dict(self.required_metrics),
            "allowed_action_bounds": dict(self.allowed_action_bounds),
            "safety_invariants": dict(self.safety_invariants),
            "created_at_ms": int(self.created_at_ms),
            "updated_at_ms": int(self.updated_at_ms),
            "version": int(self.version),
        }


class SkillContractManager:
    def __init__(self, cfg: Optional[ContractConfig] = None):
        self.cfg = cfg or ContractConfig()
        os.makedirs(os.path.dirname(self.cfg.path) or ".", exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        if not os.path.isfile(self.cfg.path):
            return {"version": "v1", "contracts": {}, "history": []}
        with open(self.cfg.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": "v1", "contracts": {}, "history": []}
        if not isinstance(data.get("contracts"), dict):
            data["contracts"] = {}
        if not isinstance(data.get("history"), list):
            data["history"] = []
        return data

    def _save(self, payload: Dict[str, Any]) -> None:
        with open(self.cfg.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _key(self, verse_name: str, skill_tag: str) -> str:
        return f"{_norm(verse_name)}::{_norm(skill_tag)}"

    def get(self, *, verse_name: str, skill_tag: str) -> Optional[SkillContract]:
        payload = self._load()
        item = payload.get("contracts", {}).get(self._key(verse_name, skill_tag))
        if not isinstance(item, dict):
            return None
        return SkillContract(
            verse_name=str(item.get("verse_name", verse_name)),
            skill_tag=str(item.get("skill_tag", skill_tag)),
            required_metrics=dict(item.get("required_metrics") or {}),
            allowed_action_bounds=dict(item.get("allowed_action_bounds") or {}),
            safety_invariants=dict(item.get("safety_invariants") or {}),
            created_at_ms=int(item.get("created_at_ms", 0) or 0),
            updated_at_ms=int(item.get("updated_at_ms", 0) or 0),
            version=int(item.get("version", 1) or 1),
        )

    def check_satisfied(
        self,
        *,
        verse_name: str,
        skill_tag: str,
        metrics: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        c = self.get(verse_name=verse_name, skill_tag=skill_tag)
        if c is None:
            # No contract exists yet.
            return True, []
        reasons: List[str] = []
        req = c.required_metrics
        if _safe_float(metrics.get("success_rate", 0.0), 0.0) < _safe_float(req.get("min_success_rate", 0.0), 0.0):
            reasons.append("success_rate_below_contract")
        if _safe_float(metrics.get("mean_return", -1e18), -1e18) < _safe_float(req.get("min_mean_return", -1e18), -1e18):
            reasons.append("mean_return_below_contract")
        max_safety = req.get("max_safety_violation_rate")
        if max_safety is not None:
            if _safe_float(metrics.get("safety_violation_rate", 1.0), 1.0) > _safe_float(max_safety, 1.0):
                reasons.append("safety_violation_above_contract")
        return (len(reasons) == 0), reasons

    def register_or_update(
        self,
        *,
        verse_name: str,
        skill_tag: str,
        metrics: Dict[str, Any],
        allowed_action_bounds: Optional[Dict[str, Any]] = None,
        safety_invariants: Optional[Dict[str, Any]] = None,
        strict_improvement_delta: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload = self._load()
        key = self._key(verse_name, skill_tag)
        contracts = payload.get("contracts", {})
        old = contracts.get(key) if isinstance(contracts.get(key), dict) else None

        req = {
            "min_success_rate": max(
                _safe_float(self.cfg.robustness_thresholds.get("min_success_rate", 0.0), 0.0),
                _safe_float(metrics.get("success_rate", 0.0), 0.0),
            ),
            "min_mean_return": max(
                _safe_float(self.cfg.robustness_thresholds.get("min_mean_return", -1e18), -1e18),
                _safe_float(metrics.get("mean_return", -1e18), -1e18),
            ),
            "max_safety_violation_rate": min(
                _safe_float(self.cfg.robustness_thresholds.get("max_safety_violation_rate", 1.0), 1.0),
                _safe_float(metrics.get("safety_violation_rate", 1.0), 1.0),
            ),
        }
        now = int(time.time() * 1000)
        delta = float(strict_improvement_delta if strict_improvement_delta is not None else self.cfg.strict_improvement_delta)

        if not isinstance(old, dict):
            c = SkillContract(
                verse_name=str(verse_name),
                skill_tag=str(skill_tag),
                required_metrics=req,
                allowed_action_bounds={k: _safe_float(v) for k, v in dict(allowed_action_bounds or {}).items()},
                safety_invariants=dict(safety_invariants or {}),
                created_at_ms=now,
                updated_at_ms=now,
                version=1,
            )
            contracts[key] = c.to_dict()
            payload["contracts"] = contracts
            payload.setdefault("history", []).append({"t_ms": now, "event": "created", "key": key})
            self._save(payload)
            return {"updated": True, "created": True, "reason": "created"}

        # strict improvement gate
        old_sr = _safe_float(old.get("required_metrics", {}).get("min_success_rate", 0.0), 0.0)
        old_ret = _safe_float(old.get("required_metrics", {}).get("min_mean_return", -1e18), -1e18)
        old_sv = _safe_float(old.get("required_metrics", {}).get("max_safety_violation_rate", 1.0), 1.0)

        improved = (
            (_safe_float(req.get("min_success_rate", old_sr), old_sr) >= old_sr + delta)
            or (_safe_float(req.get("min_mean_return", old_ret), old_ret) >= old_ret + delta)
            or (_safe_float(req.get("max_safety_violation_rate", old_sv), old_sv) <= old_sv - delta)
        )
        if not improved:
            return {"updated": False, "created": False, "reason": "strict_improvement_not_met"}

        old["required_metrics"] = req
        if allowed_action_bounds:
            old["allowed_action_bounds"] = {k: _safe_float(v) for k, v in dict(allowed_action_bounds).items()}
        if safety_invariants:
            old["safety_invariants"] = dict(safety_invariants)
        old["updated_at_ms"] = now
        old["version"] = int(old.get("version", 1) or 1) + 1
        contracts[key] = old
        payload["contracts"] = contracts
        payload.setdefault("history", []).append({"t_ms": now, "event": "updated", "key": key})
        self._save(payload)
        return {"updated": True, "created": False, "reason": "strict_improvement"}
