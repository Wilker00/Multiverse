"""
memory/value_baseline.py

Build a simple value baseline V(s) from event logs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ValueBaselineConfig:
    run_dir: str
    events_filename: str = "events.jsonl"
    output_filename: str = "value_baseline.jsonl"
    gamma: float = 0.99


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _obs_key(obs: Any) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(obs)


def _load_events(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_value_baseline(cfg: ValueBaselineConfig) -> str:
    if not os.path.isdir(cfg.run_dir):
        raise FileNotFoundError(f"run_dir not found: {cfg.run_dir}")
    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"events file not found: {events_path}")

    events = _load_events(events_path)
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ep = str(e.get("episode_id"))
        by_ep.setdefault(ep, []).append(e)

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for ep_events in by_ep.values():
        ep_events.sort(key=lambda e: int(e.get("step_idx", 0)))
        G = 0.0
        for e in reversed(ep_events):
            r = _safe_float(e.get("reward", 0.0))
            G = r + cfg.gamma * G
            key = _obs_key(e.get("obs"))
            sums[key] = sums.get(key, 0.0) + G
            counts[key] = counts.get(key, 0) + 1

    out_path = os.path.join(cfg.run_dir, cfg.output_filename)
    with open(out_path, "w", encoding="utf-8") as out:
        for key, total in sums.items():
            c = counts.get(key, 1)
            row = {"obs_key": key, "value": total / float(c), "count": c}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_path
