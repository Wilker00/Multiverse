"""
memory/advantage.py

Compute advantage-style signals from event logs and extract "DNA" sets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _load_events(path: str) -> List[Dict[str, Any]]:
    # ... (implementation unchanged)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _obs_key(obs: Any) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(obs)


def _resolve_value_baseline_path(run_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.isfile(path):
        return path
    return os.path.join(run_dir, path)


def _load_value_baseline(path: str) -> Dict[str, float]:
    baseline: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("obs_key", ""))
            if key:
                baseline[key] = _safe_float(row.get("value", 0.0))
    return baseline


def calculate_advantages(
    events: List[Dict[str, Any]],
    gamma: float,
    baseline_by: str,
    value_baseline: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Calculates episode-level advantage for each event and returns a new list
    of events with an 'advantage' key.
    """
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ep = str(e.get("episode_id"))
        by_ep.setdefault(ep, []).append(e)

    # Build discounted returns per event.
    returns_by_event: Dict[Tuple[str, int], float] = {}
    for ep, ep_events in by_ep.items():
        ep_events.sort(key=lambda ev: int(ev.get("step_idx", 0)))
        G = 0.0
        for ev in reversed(ep_events):
            r = _safe_float(ev.get("reward", 0.0))
            G = r + gamma * G
            step_idx = int(ev.get("step_idx", 0))
            returns_by_event[(ep, step_idx)] = G

    out_events: List[Dict[str, Any]] = []
    if value_baseline:
        for e in events:
            ep = str(e.get("episode_id"))
            step_idx = int(e.get("step_idx", 0))
            G = returns_by_event.get((ep, step_idx), 0.0)
            b = value_baseline.get(_obs_key(e.get("obs")), 0.0)
            row = dict(e)
            row["advantage"] = float(G - b)
            out_events.append(row)
        return out_events

    # Fallback: group baseline by field (old behavior style).
    returns_by_ep: Dict[str, float] = {}
    for ep, ep_events in by_ep.items():
        returns_by_ep[ep] = sum(_safe_float(ev.get("reward", 0.0)) for ev in ep_events)

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for e in events:
        ep = str(e.get("episode_id"))
        key = str(e.get(baseline_by))
        sums[key] = sums.get(key, 0.0) + returns_by_ep.get(ep, 0.0)
        counts[key] = counts.get(key, 0) + 1
    baselines: Dict[str, float] = {k: v / float(counts.get(k, 1)) for k, v in sums.items()}

    for e in events:
        ep = str(e.get("episode_id"))
        key = str(e.get(baseline_by))
        baseline = baselines.get(key, 0.0)
        adv = returns_by_ep.get(ep, 0.0) - baseline
        row = dict(e)
        row["advantage"] = float(adv)
        out_events.append(row)
    return out_events

@dataclass
class AdvantageConfig:
    run_dir: str
    events_filename: str = "events.jsonl"
    good_output: str = "dna_good.jsonl"
    bad_output: str = "dna_bad.jsonl"
    advantage_threshold: float = 0.0
    baseline_by: str = "verse_name"
    value_baseline_path: Optional[str] = None
    gamma: float = 0.99

def extract_dna(cfg: AdvantageConfig) -> Tuple[str, str]:
    """
    Loads events, calculates advantages, and writes good/bad DNA files.
    """
    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")

    events = _load_events(events_path)

    value_baseline: Optional[Dict[str, float]] = None
    if cfg.value_baseline_path:
        vb_path = _resolve_value_baseline_path(cfg.run_dir, cfg.value_baseline_path)
        if not os.path.isfile(vb_path):
            raise FileNotFoundError(f"Value baseline file not found: {vb_path}")
        value_baseline = _load_value_baseline(vb_path)

    events_with_adv = calculate_advantages(
        events,
        cfg.gamma,
        cfg.baseline_by,
        value_baseline=value_baseline,
    )

    good_path = os.path.join(cfg.run_dir, cfg.good_output)
    bad_path = os.path.join(cfg.run_dir, cfg.bad_output)

    with open(good_path, "w", encoding="utf-8") as good, open(bad_path, "w", encoding="utf-8") as bad:
        for e in events_with_adv:
            adv = e.get("advantage", 0.0)
            if adv >= cfg.advantage_threshold:
                good.write(json.dumps(e, ensure_ascii=False) + "\n")
            elif adv <= -cfg.advantage_threshold:
                bad.write(json.dumps(e, ensure_ascii=False) + "\n")

    return good_path, bad_path
