"""
orchestrator/curriculum_gen.py

Automatic curriculum generation (ACED-style) from collected trajectories.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.types import VerseSpec


@dataclass
class CurriculumItem:
    verse_name: str
    distance_bucket: float
    success_rate: float
    samples: int
    params: Dict[str, Any]


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


def _infer_verse(obs: Any) -> Optional[str]:
    if not isinstance(obs, dict):
        return None
    ks = set(obs.keys())
    if {"pos", "goal", "t"}.issubset(ks):
        return "line_world"
    if {"x", "y", "goal_x", "goal_y", "t"}.issubset(ks):
        return "grid_world"
    if {"x", "y", "t"}.issubset(ks):
        return "cliff_world"
    if {"agent", "target", "t"}.issubset(ks):
        return "pursuit_world"
    return None


def _distance_to_goal(obs: Any, info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    if not isinstance(obs, dict):
        return None
    if "pos" in obs and "goal" in obs:
        return abs(_safe_float(obs.get("goal")) - _safe_float(obs.get("pos")))
    if "x" in obs and "y" in obs and "goal_x" in obs and "goal_y" in obs:
        dx = abs(_safe_float(obs.get("goal_x")) - _safe_float(obs.get("x")))
        dy = abs(_safe_float(obs.get("goal_y")) - _safe_float(obs.get("y")))
        return dx + dy
    if "x" in obs and "y" in obs and isinstance(info, dict) and "goal_x" in info and "goal_y" in info:
        dx = abs(_safe_float(info.get("goal_x")) - _safe_float(obs.get("x")))
        dy = abs(_safe_float(info.get("goal_y")) - _safe_float(obs.get("y")))
        return dx + dy
    if "agent" in obs and "target" in obs:
        return abs(_safe_float(obs.get("target")) - _safe_float(obs.get("agent")))
    return None


def _iter_events_from_runs(runs_root: str) -> Iterable[Dict[str, Any]]:
    if os.path.isfile(os.path.join(runs_root, "events.jsonl")):
        files = [os.path.join(runs_root, "events.jsonl")]
    elif os.path.isdir(runs_root):
        files = []
        for name in os.listdir(runs_root):
            p = os.path.join(runs_root, name, "events.jsonl")
            if os.path.isfile(p):
                files.append(p)
    else:
        files = []

    for path in sorted(files):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row


def _episode_outcomes(events: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], bool]:
    out: Dict[Tuple[str, str], bool] = {}
    for e in events:
        run = str(e.get("run_id", ""))
        ep = str(e.get("episode_id", ""))
        key = (run, ep)
        info = e.get("info")
        success = False
        if isinstance(info, dict) and info.get("reached_goal") is True:
            success = True
        if key not in out:
            out[key] = success
        else:
            out[key] = bool(out[key] or success)
    return out


def _state_params_for_verse(verse_name: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    v = str(verse_name).strip().lower()
    if v in ("line_world", "park_world"):
        return {
            "start_pos": _safe_int(obs.get("pos", 0)),
            "goal_pos": _safe_int(obs.get("goal", obs.get("goal_pos", 8))),
        }
    if v == "grid_world":
        return {
            "start_x": _safe_int(obs.get("x", 0)),
            "start_y": _safe_int(obs.get("y", 0)),
            "goal_x": _safe_int(obs.get("goal_x", 4)),
            "goal_y": _safe_int(obs.get("goal_y", 4)),
        }
    if v == "cliff_world":
        return {
            "start_x": _safe_int(obs.get("x", 0)),
            "start_y": _safe_int(obs.get("y", 0)),
        }
    if v == "pursuit_world":
        return {
            "start_agent": _safe_int(obs.get("agent", 0)),
            "start_target": _safe_int(obs.get("target", 8)),
        }
    return {}


def _advance_params(params: Dict[str, Any], ratio: float) -> Dict[str, Any]:
    ratio = max(0.0, min(1.0, float(ratio)))
    out = dict(params)
    for key in ("start_pos", "start_x", "start_y", "start_agent"):
        if key not in out:
            continue
        goal_key = {
            "start_pos": "goal_pos",
            "start_x": "goal_x",
            "start_y": "goal_y",
            "start_agent": "start_target",  # pursuit target as dynamic "goal"
        }.get(key)
        if goal_key is None or goal_key not in out:
            continue
        cur = _safe_float(out.get(key, 0.0))
        goal = _safe_float(out.get(goal_key, 0.0))
        # Move 10% further from goal each stage.
        out[key] = int(round(cur + (cur - goal) * ratio))
    return out


def generate_task_sequence(
    memory_dir: str,
    *,
    verse_name: Optional[str] = None,
    max_specs: int = 8,
    success_target: float = 0.8,
    advance_ratio: float = 0.10,
) -> List[VerseSpec]:
    """
    Generate a curriculum starting from boundary states (success ~50%).
    """
    events = list(_iter_events_from_runs(memory_dir))
    if not events:
        return []

    outcomes = _episode_outcomes(events)
    # Re-iterate from cached events.
    by_verse_bucket: Dict[Tuple[str, int], Dict[str, Any]] = {}
    reps: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for e in events:
        obs = e.get("obs")
        info = e.get("info")
        if not isinstance(obs, dict):
            continue
        vv = str(e.get("verse_name", "")).strip().lower() or (_infer_verse(obs) or "")
        if not vv:
            continue
        if verse_name and vv != str(verse_name).strip().lower():
            continue
        dist = _distance_to_goal(obs, info if isinstance(info, dict) else None)
        if dist is None:
            continue
        bucket = int(round(dist))
        k = (vv, bucket)
        run = str(e.get("run_id", ""))
        ep = str(e.get("episode_id", ""))
        success = bool(outcomes.get((run, ep), False))
        b = by_verse_bucket.get(k)
        if not isinstance(b, dict):
            b = {"n": 0, "succ": 0}
            by_verse_bucket[k] = b
        b["n"] = int(b["n"]) + 1
        b["succ"] = int(b["succ"]) + (1 if success else 0)
        reps.setdefault(k, obs)

    per_verse: Dict[str, List[CurriculumItem]] = {}
    for (vv, bucket), stats in by_verse_bucket.items():
        n = int(stats["n"])
        if n <= 0:
            continue
        sr = float(int(stats["succ"]) / float(n))
        params = _state_params_for_verse(vv, reps[(vv, bucket)])
        if vv == "cliff_world":
            params.setdefault("goal_x", -1)
            params.setdefault("goal_y", -1)
        per_verse.setdefault(vv, []).append(
            CurriculumItem(
                verse_name=vv,
                distance_bucket=float(bucket),
                success_rate=sr,
                samples=n,
                params=params,
            )
        )

    out_specs: List[VerseSpec] = []
    for vv, items in per_verse.items():
        items.sort(key=lambda x: abs(x.success_rate - 0.5))
        boundary = items[0]
        params = dict(boundary.params)
        stages = min(max_specs, max(1, len(items)))
        for _ in range(stages):
            out_specs.append(
                VerseSpec(
                    spec_version="v1",
                    verse_name=vv,
                    verse_version="0.1",
                    seed=None,
                    tags=["curriculum", "aced"],
                    params=dict(params),
                )
            )
            if boundary.success_rate >= float(success_target):
                params = _advance_params(params, advance_ratio)

    return out_specs[: max(1, int(max_specs))]
