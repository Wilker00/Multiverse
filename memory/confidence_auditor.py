"""
memory/confidence_auditor.py

Tracks direct-vs-semantic reward deltas and derives bridge trust weights.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from memory.task_taxonomy import primary_task_tag


@dataclass
class ConfidenceAuditConfig:
    root_dir: str = "central_memory"
    filename: str = "bridge_audit.json"
    min_samples: int = 20
    learning_rate: float = 0.5
    min_weight: float = 0.2
    max_weight: float = 1.5
    smoothing: float = 0.3


@dataclass
class OutcomeSample:
    strategy: str
    target_verse: str
    reward: float
    source_verse: Optional[str] = None
    task_tag: Optional[str] = None


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


def _path(cfg: ConfidenceAuditConfig) -> str:
    return os.path.join(cfg.root_dir, cfg.filename)


def _default_state() -> Dict[str, Any]:
    return {"version": "v1", "updated_at_ms": 0, "tasks": {}}


def load_state(cfg: Optional[ConfidenceAuditConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = ConfidenceAuditConfig()
    os.makedirs(cfg.root_dir, exist_ok=True)
    p = _path(cfg)
    if not os.path.isfile(p):
        return _default_state()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return _default_state()
    if "tasks" not in data or not isinstance(data.get("tasks"), dict):
        return _default_state()
    return data


def save_state(state: Dict[str, Any], cfg: Optional[ConfidenceAuditConfig] = None) -> str:
    if cfg is None:
        cfg = ConfidenceAuditConfig()
    os.makedirs(cfg.root_dir, exist_ok=True)
    p = _path(cfg)
    state = dict(state)
    state["updated_at_ms"] = int(time.time() * 1000)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return p


def _task_key(*, target_verse: str, task_tag: Optional[str]) -> str:
    t = str(task_tag or "").strip().lower()
    if not t:
        t = primary_task_tag(target_verse)
    return f"{str(target_verse).strip().lower()}::{t}"


def _bridge_key(source_verse: str, target_verse: str) -> str:
    return f"{str(source_verse).strip().lower()}->{str(target_verse).strip().lower()}"


def _mean_update(bucket: Dict[str, Any], reward: float) -> None:
    c_old = _safe_int(bucket.get("count", 0))
    m_old = _safe_float(bucket.get("mean_reward", 0.0))
    c_new = c_old + 1
    m_new = m_old + ((float(reward) - m_old) / float(max(1, c_new)))
    bucket["count"] = int(c_new)
    bucket["mean_reward"] = float(m_new)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _update_weights_for_task(
    *,
    task_bucket: Dict[str, Any],
    target_verse: str,
    cfg: ConfidenceAuditConfig,
) -> None:
    direct = task_bucket.get("direct")
    if not isinstance(direct, dict):
        direct = {"count": 0, "mean_reward": 0.0}
        task_bucket["direct"] = direct
    direct_count = _safe_int(direct.get("count", 0))
    direct_mean = _safe_float(direct.get("mean_reward", 0.0))

    bridges = task_bucket.get("bridges")
    if not isinstance(bridges, dict):
        bridges = {}
        task_bucket["bridges"] = bridges

    for bkey, b in bridges.items():
        if not isinstance(b, dict):
            continue
        b_count = _safe_int(b.get("count", 0))
        b_mean = _safe_float(b.get("mean_reward", 0.0))
        old_w = _safe_float(b.get("weight", 1.0), 1.0)

        if direct_count < int(cfg.min_samples) or b_count < max(1, int(cfg.min_samples // 2)):
            b["weight"] = float(old_w)
            continue

        delta = float(b_mean - direct_mean)
        target_w = _clamp(
            1.0 + (float(cfg.learning_rate) * delta),
            float(cfg.min_weight),
            float(cfg.max_weight),
        )
        smooth = _clamp(float(cfg.smoothing), 0.0, 1.0)
        new_w = ((1.0 - smooth) * old_w) + (smooth * target_w)
        b["weight"] = float(_clamp(new_w, float(cfg.min_weight), float(cfg.max_weight)))

        b["delta_vs_direct"] = float(delta)
        b["target_verse"] = str(target_verse).strip().lower()


def record_outcomes_batch(
    samples: Iterable[OutcomeSample],
    cfg: Optional[ConfidenceAuditConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = ConfidenceAuditConfig()
    state = load_state(cfg)
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        tasks = {}
        state["tasks"] = tasks

    changed = 0
    for s in samples:
        strategy = str(s.strategy or "").strip().lower()
        target_verse = str(s.target_verse or "").strip().lower()
        if not strategy or not target_verse:
            continue
        reward = _safe_float(s.reward, 0.0)
        key = _task_key(target_verse=target_verse, task_tag=s.task_tag)
        bucket = tasks.get(key)
        if not isinstance(bucket, dict):
            bucket = {
                "target_verse": target_verse,
                "task_tag": key.split("::", 1)[1],
                "direct": {"count": 0, "mean_reward": 0.0},
                "semantic": {"count": 0, "mean_reward": 0.0},
                "bridges": {},
            }
            tasks[key] = bucket

        if strategy == "direct":
            _mean_update(bucket["direct"], reward)
            changed += 1
            continue

        if strategy in ("semantic_bridge", "hybrid_low_confidence"):
            _mean_update(bucket["semantic"], reward)
            src = str(s.source_verse or "").strip().lower()
            if src:
                bk = _bridge_key(src, target_verse)
                bridges = bucket.get("bridges")
                if not isinstance(bridges, dict):
                    bridges = {}
                    bucket["bridges"] = bridges
                b = bridges.get(bk)
                if not isinstance(b, dict):
                    b = {"count": 0, "mean_reward": 0.0, "weight": 1.0}
                    bridges[bk] = b
                _mean_update(b, reward)
            changed += 1
            continue

        # Unknown strategy: ignore

    if changed > 0:
        for key, task_bucket in tasks.items():
            if not isinstance(task_bucket, dict):
                continue
            target_verse = str(task_bucket.get("target_verse", key.split("::", 1)[0]))
            _update_weights_for_task(task_bucket=task_bucket, target_verse=target_verse, cfg=cfg)
        save_state(state, cfg)
    return state


def get_bridge_weight(
    *,
    source_verse: str,
    target_verse: str,
    task_tag: Optional[str] = None,
    cfg: Optional[ConfidenceAuditConfig] = None,
    state: Optional[Dict[str, Any]] = None,
) -> float:
    if cfg is None:
        cfg = ConfidenceAuditConfig()
    if state is None:
        state = load_state(cfg)

    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        return 1.0
    key = _task_key(target_verse=target_verse, task_tag=task_tag)
    task_bucket = tasks.get(key)
    if not isinstance(task_bucket, dict):
        return 1.0
    bridges = task_bucket.get("bridges")
    if not isinstance(bridges, dict):
        return 1.0
    bk = _bridge_key(source_verse, target_verse)
    b = bridges.get(bk)
    if not isinstance(b, dict):
        return 1.0
    return _clamp(_safe_float(b.get("weight", 1.0), 1.0), float(cfg.min_weight), float(cfg.max_weight))


def summarize_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for key, tb in tasks.items():
        if not isinstance(tb, dict):
            continue
        direct = tb.get("direct", {})
        semantic = tb.get("semantic", {})
        bridges = tb.get("bridges", {})
        if not isinstance(bridges, dict):
            bridges = {}
        for bk, b in bridges.items():
            if not isinstance(b, dict):
                continue
            rows.append(
                {
                    "task": key,
                    "bridge": bk,
                    "direct_count": _safe_int(direct.get("count", 0)),
                    "direct_mean_reward": _safe_float(direct.get("mean_reward", 0.0)),
                    "semantic_count": _safe_int(semantic.get("count", 0)),
                    "semantic_mean_reward": _safe_float(semantic.get("mean_reward", 0.0)),
                    "bridge_count": _safe_int(b.get("count", 0)),
                    "bridge_mean_reward": _safe_float(b.get("mean_reward", 0.0)),
                    "weight": _safe_float(b.get("weight", 1.0)),
                    "delta_vs_direct": _safe_float(b.get("delta_vs_direct", 0.0)),
                }
            )
    rows.sort(key=lambda r: (r["task"], r["bridge"]))
    return rows
