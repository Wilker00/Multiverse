"""
memory/decay_manager.py

Temporal knowledge decay utilities for retrieval freshness control.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DecayConfig:
    decay_lambda: float = 0.0
    stale_threshold: float = 0.1
    current_time_ms: Optional[int] = None


@dataclass
class ArchiveStats:
    input_rows: int
    active_rows: int
    archived_rows: int
    archive_path: str


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def now_ms() -> int:
    return int(time.time() * 1000)


def recency_weight(
    *,
    t_ms: int,
    decay_lambda: float,
    current_time_ms: Optional[int] = None,
) -> float:
    lam = max(0.0, float(decay_lambda))
    if lam <= 0.0:
        return 1.0
    now = int(current_time_ms) if current_time_ms is not None else now_ms()
    age_ms = max(0.0, float(now - int(t_ms)))
    return float(math.exp(-lam * age_ms))


def apply_temporal_decay(
    *,
    score: float,
    t_ms: int,
    decay_lambda: float,
    current_time_ms: Optional[int] = None,
) -> Tuple[float, float]:
    w = recency_weight(t_ms=t_ms, decay_lambda=decay_lambda, current_time_ms=current_time_ms)
    return float(score) * float(w), float(w)


def archive_stale_memories(
    *,
    memory_dir: str = "central_memory",
    memories_filename: str = "memories.jsonl",
    archive_filename: str = "memories.archive.jsonl",
    cfg: Optional[DecayConfig] = None,
) -> ArchiveStats:
    if cfg is None:
        cfg = DecayConfig()

    mem_path = os.path.join(memory_dir, memories_filename)
    archive_path = os.path.join(memory_dir, archive_filename)
    if not os.path.isfile(mem_path):
        raise FileNotFoundError(f"Memory file not found: {mem_path}")

    active_rows: List[Dict[str, Any]] = []
    archived_rows: List[Dict[str, Any]] = []
    total = 0
    with open(mem_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            total += 1
            t_ms = _safe_int(row.get("t_ms", 0))
            w = recency_weight(
                t_ms=t_ms,
                decay_lambda=float(cfg.decay_lambda),
                current_time_ms=cfg.current_time_ms,
            )
            row["recency_weight"] = float(w)
            tier = str(row.get("memory_tier", "stm")).strip().lower()
            # LTM is sovereign memory and is never archived by temporal decay.
            if tier != "ltm" and w < float(cfg.stale_threshold):
                archived_rows.append(row)
            else:
                active_rows.append(row)

    os.makedirs(memory_dir, exist_ok=True)
    with open(mem_path, "w", encoding="utf-8") as out:
        for row in active_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(archive_path, "a", encoding="utf-8") as out:
        for row in archived_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return ArchiveStats(
        input_rows=total,
        active_rows=len(active_rows),
        archived_rows=len(archived_rows),
        archive_path=archive_path,
    )
