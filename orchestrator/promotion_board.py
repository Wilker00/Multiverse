"""
orchestrator/promotion_board.py

Multi-critic promotion board:
- critic_1: regression + safety gate (existing eval harness A/B)
- critic_2: long-horizon stress + adversarial stress checks
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.types import AgentSpec
from orchestrator.eval_harness import (
    BenchmarkCase,
    evaluate_agent_case,
    run_ab_gate,
    gate_to_dict,
)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _validate_allowed_keys(*, payload: Dict[str, Any], allowed: List[str], label: str) -> None:
    allowed_set = {str(k) for k in allowed}
    unknown = sorted(str(k) for k in payload.keys() if str(k) not in allowed_set)
    if unknown:
        raise ValueError(
            f"{label} has unknown keys: {unknown}. Allowed keys: {sorted(allowed_set)}"
        )


@dataclass
class PromotionConfig:
    enabled: bool = True
    require_multi_critic: bool = True
    disagreement_policy: str = "quarantine"  # quarantine | reject | allow
    quarantine_dir: str = os.path.join("models", "quarantine")
    long_horizon_episodes_per_seed: int = 3
    bootstrap_samples: int = 600
    alpha: float = 0.05
    external_benchmark_json: str = ""
    human_decision_path: str = os.path.join("models", "promotion_board_human_decisions.json")
    require_human_bless: bool = False
    decision_ttl_hours: float = 168.0
    memory_health_path: str = os.path.join("models", "memory_health", "latest.json")
    max_memory_slope_mb_per_hour: float = 48.0
    require_stability_verses: int = 3

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "PromotionConfig":
        c = cfg if isinstance(cfg, dict) else {}
        strict_cfg = bool(c.get("strict_config_validation", True))
        if strict_cfg:
            _validate_allowed_keys(
                payload=c,
                allowed=[
                    "enabled",
                    "require_multi_critic",
                    "disagreement_policy",
                    "quarantine_dir",
                    "long_horizon_episodes_per_seed",
                    "bootstrap_samples",
                    "alpha",
                    "external_benchmark_json",
                    "human_decision_path",
                    "require_human_bless",
                    "decision_ttl_hours",
                    "memory_health_path",
                    "max_memory_slope_mb_per_hour",
                    "require_stability_verses",
                    "strict_config_validation",
                ],
                label="promotion board config",
            )
        return PromotionConfig(
            enabled=bool(c.get("enabled", True)),
            require_multi_critic=bool(c.get("require_multi_critic", True)),
            disagreement_policy=str(c.get("disagreement_policy", "quarantine")),
            quarantine_dir=str(c.get("quarantine_dir", os.path.join("models", "quarantine"))),
            long_horizon_episodes_per_seed=max(1, int(c.get("long_horizon_episodes_per_seed", 3))),
            bootstrap_samples=max(100, int(c.get("bootstrap_samples", 600))),
            alpha=max(0.001, min(0.2, _safe_float(c.get("alpha", 0.05), 0.05))),
            external_benchmark_json=str(c.get("external_benchmark_json", "")),
            human_decision_path=str(c.get("human_decision_path", os.path.join("models", "promotion_board_human_decisions.json"))),
            require_human_bless=bool(c.get("require_human_bless", False)),
            decision_ttl_hours=max(1.0, _safe_float(c.get("decision_ttl_hours", 168.0), 168.0)),
            memory_health_path=str(c.get("memory_health_path", os.path.join("models", "memory_health", "latest.json"))),
            max_memory_slope_mb_per_hour=max(
                0.0, _safe_float(c.get("max_memory_slope_mb_per_hour", 48.0), 48.0)
            ),
            require_stability_verses=max(1, int(c.get("require_stability_verses", 3))),
        )


def _default_human_decisions() -> Dict[str, Any]:
    return {
        "version": "v1",
        "updated_at_ms": 0,
        "decisions": [],
    }


def _load_human_decisions(path: str) -> Dict[str, Any]:
    p = str(path or "").strip()
    if not p or not os.path.isfile(p):
        return _default_human_decisions()
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return _default_human_decisions()
        if not isinstance(obj.get("decisions"), list):
            obj["decisions"] = []
        return obj
    except Exception:
        return _default_human_decisions()


def record_human_decision(
    *,
    path: str,
    target_verse: str,
    candidate_policy_id: str,
    decision: str,
    actor: str = "operator",
    note: str = "",
) -> Dict[str, Any]:
    d = str(decision).strip().lower()
    if d not in ("bless", "veto"):
        raise ValueError(f"decision must be bless|veto, got: {decision}")
    payload = _load_human_decisions(path)
    rows = payload.get("decisions")
    rows = rows if isinstance(rows, list) else []
    row = {
        "t_ms": int(time.time() * 1000),
        "target_verse": str(target_verse).strip().lower(),
        "candidate_policy_id": str(candidate_policy_id).strip(),
        "decision": d,
        "actor": str(actor).strip() or "operator",
        "note": str(note),
    }
    rows.append(row)
    payload["version"] = "v1"
    payload["updated_at_ms"] = int(time.time() * 1000)
    payload["decisions"] = rows
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return row


def _latest_human_decision(
    *,
    path: str,
    target_verse: str,
    candidate_policy_id: str,
    ttl_hours: float,
) -> Optional[Dict[str, Any]]:
    payload = _load_human_decisions(path)
    rows = payload.get("decisions")
    rows = rows if isinstance(rows, list) else []
    now_ms = int(time.time() * 1000)
    ttl_ms = int(max(1.0, float(ttl_hours)) * 3600.0 * 1000.0)
    tv = str(target_verse).strip().lower()
    cp = str(candidate_policy_id).strip()
    latest: Optional[Dict[str, Any]] = None
    latest_t = -1
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("target_verse", "")).strip().lower() != tv:
            continue
        if str(row.get("candidate_policy_id", "")).strip() != cp:
            continue
        t_ms = int(row.get("t_ms", 0) or 0)
        if t_ms <= 0 or (now_ms - t_ms) > ttl_ms:
            continue
        if t_ms > latest_t:
            latest_t = t_ms
            latest = row
    return latest


def _read_external_benchmark(path: str) -> Optional[Dict[str, Any]]:
    p = str(path or "").strip()
    if not p or not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _read_json_obj(path: str) -> Optional[Dict[str, Any]]:
    p = str(path or "").strip()
    if not p or not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _stability_gate(critic2: Dict[str, Any], *, min_passed_verses: int) -> Dict[str, Any]:
    by_verse = critic2.get("by_verse")
    by_verse = by_verse if isinstance(by_verse, dict) else {}
    total = 0
    passed = 0
    failed: List[str] = []
    for verse, row in by_verse.items():
        if not isinstance(row, dict):
            continue
        total += 1
        ok = bool(row.get("passed", False))
        if ok:
            passed += 1
        else:
            failed.append(str(verse))
    enough_coverage = bool(total >= int(min_passed_verses))
    stable = bool(enough_coverage and passed >= int(min_passed_verses))
    reason = "stable"
    if not enough_coverage:
        reason = "insufficient_verse_coverage"
    elif passed < int(min_passed_verses):
        reason = "insufficient_stable_verses"
    return {
        "passed": bool(stable),
        "reason": str(reason),
        "required_verses": int(min_passed_verses),
        "total_verses": int(total),
        "passed_verses": int(passed),
        "failed_verses": failed,
    }


def _memory_health_gate(*, path: str, max_slope_mb_per_hour: float) -> Dict[str, Any]:
    obj = _read_json_obj(path)
    if obj is None:
        return {
            "passed": True,
            "skipped": True,
            "path": str(path),
            "reason": "missing_memory_health_report",
        }
    slope = _safe_float(
        obj.get(
            "rss_slope_mb_per_hour",
            obj.get("memory_slope_mb_per_hour", 0.0),
        ),
        0.0,
    )
    leak = bool(obj.get("leak_detected", False))
    max_slope = max(0.0, float(max_slope_mb_per_hour))
    passed = bool((not leak) and slope <= max_slope)
    reason = "ok"
    if leak:
        reason = "leak_detected"
    elif slope > max_slope:
        reason = "memory_slope_too_high"
    return {
        "passed": bool(passed),
        "skipped": False,
        "path": str(path),
        "rss_slope_mb_per_hour": float(slope),
        "max_allowed_mb_per_hour": float(max_slope),
        "leak_detected": bool(leak),
        "reason": str(reason),
    }


def _long_horizon_cases(eps_per_seed: int) -> List[BenchmarkCase]:
    return [
        BenchmarkCase(
            verse_name="labyrinth_world",
            verse_version="0.1",
            params={
                "adr_enabled": False,
                "width": 17,
                "height": 13,
                "max_steps": 220,
                "action_noise": 0.18,
                "battery_capacity": 80,
                "step_penalty": -0.05,
            },
            seeds=[111, 223],
            episodes_per_seed=int(eps_per_seed),
            max_steps=220,
        ),
        BenchmarkCase(
            verse_name="cliff_world",
            verse_version="0.1",
            params={
                "adr_enabled": False,
                "width": 12,
                "height": 4,
                "max_steps": 120,
                "step_penalty": -1.0,
                "cliff_penalty": -100.0,
                "end_on_cliff": False,
            },
            seeds=[111, 223],
            episodes_per_seed=int(eps_per_seed),
            max_steps=120,
        ),
        BenchmarkCase(
            verse_name="pursuit_world",
            verse_version="0.1",
            params={"adr_enabled": False, "max_steps": 100},
            seeds=[111, 223],
            episodes_per_seed=int(eps_per_seed),
            max_steps=100,
        ),
    ]


def _critic_long_horizon(candidate_spec: AgentSpec, baseline_spec: AgentSpec, *, eps_per_seed: int) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    passed = True
    for case in _long_horizon_cases(eps_per_seed):
        b = evaluate_agent_case(agent_spec=baseline_spec, case=case)
        c = evaluate_agent_case(agent_spec=candidate_spec, case=case)
        # Candidate must not regress heavily on long-horizon success and safety.
        min_success = float(b.success_rate) * 0.90
        max_safety = max(float(b.safety_violation_rate) * 1.20, 0.25)
        ok = bool(c.success_rate >= min_success and c.safety_violation_rate <= max_safety)
        rows[case.verse_name] = {
            "passed": ok,
            "candidate": {
                "mean_return": float(c.mean_return),
                "success_rate": float(c.success_rate),
                "safety_violation_rate": float(c.safety_violation_rate),
            },
            "baseline": {
                "mean_return": float(b.mean_return),
                "success_rate": float(b.success_rate),
                "safety_violation_rate": float(b.safety_violation_rate),
            },
            "constraints": {
                "min_success_rate": float(min_success),
                "max_safety_violation_rate": float(max_safety),
            },
        }
        passed = bool(passed and ok)
    return {"passed": bool(passed), "by_verse": rows}


def run_promotion_board(
    *,
    baseline_spec: AgentSpec,
    candidate_spec: AgentSpec,
    target_verse: str,
    cfg: PromotionConfig,
) -> Dict[str, Any]:
    if not bool(cfg.enabled):
        return {"passed": True, "skipped": True, "reason": "promotion_board_disabled"}

    critic1 = run_ab_gate(
        baseline_spec=baseline_spec,
        candidate_spec=candidate_spec,
        suite_mode="target",
        target_verse=str(target_verse),
        bootstrap_samples=int(cfg.bootstrap_samples),
        alpha=float(cfg.alpha),
    )
    critic1_pass = bool(critic1.passed)
    critic2 = _critic_long_horizon(candidate_spec, baseline_spec, eps_per_seed=int(cfg.long_horizon_episodes_per_seed))
    ext = _read_external_benchmark(cfg.external_benchmark_json)
    if isinstance(ext, dict):
        ext_pass = bool(ext.get("overall_pass", False))
        critic2 = {
            **critic2,
            "external_benchmark": {
                "path": str(cfg.external_benchmark_json),
                "overall_pass": bool(ext_pass),
                "summary": ext.get("summary"),
            },
            "passed": bool(critic2.get("passed", False) and ext_pass),
        }
    critic2_pass = bool(critic2.get("passed", False))
    stability = _stability_gate(critic2, min_passed_verses=int(cfg.require_stability_verses))
    stability_pass = bool(stability.get("passed", False))
    mem_health = _memory_health_gate(
        path=str(cfg.memory_health_path),
        max_slope_mb_per_hour=float(cfg.max_memory_slope_mb_per_hour),
    )
    mem_health_pass = bool(mem_health.get("passed", False))

    if bool(cfg.require_multi_critic):
        passed = bool(critic1_pass and critic2_pass and stability_pass and mem_health_pass)
    else:
        passed = bool((critic1_pass or critic2_pass) and stability_pass and mem_health_pass)

    disagreed = bool(critic1_pass != critic2_pass)
    hard_veto = bool(not mem_health_pass)
    result = {
        "passed": bool(passed),
        "disagreed": disagreed,
        "target_verse": str(target_verse),
        "candidate_policy_id": str(candidate_spec.policy_id),
        "baseline_policy_id": str(baseline_spec.policy_id),
        "critic_1": gate_to_dict(critic1),
        "critic_2": critic2,
        "stability_gate": stability,
        "memory_health_gate": mem_health,
        "policy": str(cfg.disagreement_policy),
        "hard_veto": bool(hard_veto),
    }
    if disagreed and str(cfg.disagreement_policy).strip().lower() == "quarantine":
        os.makedirs(cfg.quarantine_dir, exist_ok=True)
        out = os.path.join(cfg.quarantine_dir, f"promotion_disagreement_{int(time.time())}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["quarantine_report"] = out
        result["passed"] = False
    elif disagreed and str(cfg.disagreement_policy).strip().lower() == "reject":
        result["passed"] = False

    human = _latest_human_decision(
        path=str(cfg.human_decision_path),
        target_verse=str(target_verse),
        candidate_policy_id=str(candidate_spec.policy_id),
        ttl_hours=float(cfg.decision_ttl_hours),
    )
    if human is not None:
        decision = str(human.get("decision", "")).strip().lower()
        result["human_decision"] = human
        if decision == "veto":
            result["passed"] = False
            result["human_override"] = "veto"
        elif decision == "bless":
            if hard_veto:
                result["passed"] = False
                result["human_override"] = "bless_blocked_by_hard_veto"
            else:
                result["passed"] = True
                result["human_override"] = "bless"
    elif bool(cfg.require_human_bless):
        result["passed"] = False
        result["human_override"] = "pending_bless"
    return result
