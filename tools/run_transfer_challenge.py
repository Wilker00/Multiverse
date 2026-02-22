"""
tools/run_transfer_challenge.py

Cross-verse transfer challenge:
- Transfer agent: strategy DNA warm-start + optional SafeExecutor/MCTS
- Baseline agent: naive training from scratch

The target verse defaults to warehouse_world.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from core.universe_registry import (
    build_transfer_source_plan,
    primary_universe_for_verse,
    source_transfer_lane,
)
from memory.universe_adapters import transfer_row_universe_metadata
from memory.semantic_bridge import bridge_reason, infer_verse_from_obs, translate_dna
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer


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


def _parse_cfg_scalar(raw: str) -> Any:
    s = str(raw).strip()
    lo = s.lower()
    if lo in {"true", "t", "yes", "y", "on", "1"}:
        return True
    if lo in {"false", "f", "no", "n", "off", "0"}:
        return False
    if lo in {"none", "null"}:
        return None
    try:
        if "." not in s and "e" not in lo:
            return int(s)
        return float(s)
    except Exception:
        return s


def _parse_cfg_overrides(tokens: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tok in (tokens or []):
        s = str(tok).strip()
        if not s:
            continue
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        k = str(key).strip()
        if not k:
            continue
        out[k] = _parse_cfg_scalar(val)
    return out


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
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


def _peek_first_event(events_path: str) -> Dict[str, Any]:
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
    except Exception:
        return {}
    return {}


def _list_run_dirs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[str] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        if os.path.isdir(run_dir) and os.path.isfile(os.path.join(run_dir, "events.jsonl")):
            out.append(run_dir)
    return out


def _extract_success_dna_from_events(
    *,
    run_dir: str,
    out_path: str,
    max_rows: int = 5000,
) -> int:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return 0
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for ev in _iter_jsonl(events_path):
        ep = str(ev.get("episode_id", "")).strip()
        if not ep:
            continue
        if ep not in by_ep:
            by_ep[ep] = []
            order.append(ep)
        by_ep[ep].append(ev)

    success_eps = set()
    for ep in order:
        rows = by_ep.get(ep, [])
        if any(bool((r.get("info") or {}).get("reached_goal", False)) for r in rows if isinstance(r, dict)):
            success_eps.add(ep)
    if not success_eps:
        return 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ep in order:
            if ep not in success_eps:
                continue
            rows = by_ep.get(ep, [])
            rows.sort(key=lambda r: _safe_int(r.get("step_idx", 0), 0))
            for i, ev in enumerate(rows):
                if written >= int(max_rows):
                    return written
                obs = ev.get("obs")
                action = ev.get("action")
                try:
                    a = int(action)
                except Exception:
                    continue
                next_obs = rows[i + 1].get("obs") if i + 1 < len(rows) else obs
                row = {
                    "episode_id": str(ev.get("episode_id", "")),
                    "step_idx": _safe_int(ev.get("step_idx", i), i),
                    "verse_name": str(ev.get("verse_name", "")),
                    "obs": obs,
                    "action": int(a),
                    "reward": _safe_float(ev.get("reward", 0.0), 0.0),
                    "done": bool(ev.get("done", False) or ev.get("truncated", False)),
                    "next_obs": next_obs,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    return written


@dataclass
class _SourceDNA:
    verse_name: str
    path: str
    run_id: str
    source_kind: str
    source_lane: str = "far_universe"
    source_universe: str = ""


def _discover_transfer_sources(
    *,
    target_verse: str,
    runs_root: str,
    max_runs_per_verse: int,
    min_success_rate: float,
    min_rows_per_source: int,
    max_source_scan: int = 200,
) -> List[_SourceDNA]:
    plan = build_transfer_source_plan(str(target_verse))
    ordered_sources = [str(v).strip().lower() for v in (plan.get("ordered_sources") or []) if str(v).strip()]
    if not ordered_sources:
        ordered_sources = ["chess_world", "go_world", "trade_world", "uno_world"]
    near_set = set(str(v).strip().lower() for v in (plan.get("near_sources") or []) if str(v).strip())
    all_targets = tuple(ordered_sources)
    candidates_by_verse: Dict[str, List[Tuple[str, float, float]]] = {v: [] for v in all_targets}
    scanned = 0
    for run_dir in _list_run_dirs(runs_root):
        events_path = os.path.join(run_dir, "events.jsonl")
        first = _peek_first_event(events_path)
        verse_name = str(first.get("verse_name", "")).strip().lower()
        if verse_name not in all_targets:
            continue
        scanned += 1
        if int(max_source_scan) > 0 and scanned > int(max_source_scan):
            break
        try:
            st = evaluate_run(run_dir)
        except Exception:
            continue
        success_rate = float(st.success_rate or 0.0)
        if success_rate < float(min_success_rate):
            continue
        mtime = _safe_float(os.path.getmtime(events_path), 0.0)
        candidates_by_verse[verse_name].append((run_dir, success_rate, mtime))

    out: List[_SourceDNA] = []
    for verse_name in all_targets:
        cands = sorted(
            candidates_by_verse.get(verse_name, []),
            key=lambda t: (float(t[1]), float(t[2])),
            reverse=True,
        )[: max(1, int(max_runs_per_verse))]
        for run_dir, _, _ in cands:
            run_id = os.path.basename(run_dir)
            dna_good = os.path.join(run_dir, "dna_good.jsonl")
            if os.path.isfile(dna_good):
                rows = sum(1 for _ in _iter_jsonl(dna_good))
                if rows >= int(min_rows_per_source):
                    out.append(
                        _SourceDNA(
                            verse_name=verse_name,
                            path=dna_good,
                            run_id=run_id,
                            source_kind="dna_good",
                            source_lane=("near_universe" if verse_name in near_set else "far_universe"),
                            source_universe=(primary_universe_for_verse(verse_name) or ""),
                        )
                    )
                    continue
            succ = os.path.join(run_dir, "dna_success_only.jsonl")
            rows = _extract_success_dna_from_events(run_dir=run_dir, out_path=succ, max_rows=12000)
            if rows >= int(min_rows_per_source):
                out.append(
                    _SourceDNA(
                        verse_name=verse_name,
                        path=succ,
                        run_id=run_id,
                        source_kind="success_events",
                        source_lane=("near_universe" if verse_name in near_set else "far_universe"),
                        source_universe=(primary_universe_for_verse(verse_name) or ""),
                    )
                )

    # Fallback to curated expert datasets if run discovery is empty.
    if not out:
        for verse_name in all_targets:
            p = os.path.join("models", "expert_datasets", f"{verse_name}.jsonl")
            if os.path.isfile(p):
                out.append(
                    _SourceDNA(
                        verse_name=verse_name,
                        path=p,
                        run_id="expert_dataset",
                        source_kind="fallback",
                        source_lane=("near_universe" if verse_name in near_set else "far_universe"),
                        source_universe=(primary_universe_for_verse(verse_name) or ""),
                    )
                )
    return out


def _discover_strategy_sources(
    *,
    runs_root: str,
    max_runs_per_verse: int,
    min_success_rate: float,
    min_rows_per_source: int,
    max_source_scan: int = 200,
) -> List[_SourceDNA]:
    # Backward-compatible wrapper for tests/tools that still call the legacy helper.
    return _discover_transfer_sources(
        target_verse="warehouse_world",
        runs_root=runs_root,
        max_runs_per_verse=max_runs_per_verse,
        min_success_rate=min_success_rate,
        min_rows_per_source=min_rows_per_source,
        max_source_scan=max_source_scan,
    )


def _merge_jsonl(paths: List[str], out_path: str, *, max_rows_per_file: int = 0) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            if not os.path.isfile(p):
                continue
            taken = 0
            for row in _iter_jsonl(p):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                taken += 1
                if int(max_rows_per_file) > 0 and taken >= int(max_rows_per_file):
                    break
    return rows


def _default_far_lane_cap(*, base_cap: int) -> int:
    if int(base_cap) <= 0:
        return 0
    return max(25, int(base_cap) // 2)


def _universe_feature_score_from_row(row: Dict[str, Any]) -> Optional[float]:
    ut = row.get("universe_transfer")
    if not isinstance(ut, dict):
        return None
    adapter = ut.get("adapter")
    if not isinstance(adapter, dict):
        return None
    feats = adapter.get("features")
    if not isinstance(feats, dict) or not feats:
        return None

    core_keys = (
        "goal_progress",
        "hazard_proximity",
        "resource_level",
        "queue_pressure",
        "throughput",
        "congestion",
        "time_pressure",
    )
    vals: List[float] = []
    for k in core_keys:
        v = feats.get(k)
        if isinstance(v, (int, float)):
            fv = max(0.0, min(1.0, float(v)))
            vals.append(fv)
    if not vals:
        return None

    mean_core = float(sum(vals) / float(len(vals)))
    # Reward non-degenerate transitions slightly when mechanics deltas are present.
    prog_delta = abs(_safe_float(feats.get("mechanics_progress_delta", 0.0), 0.0))
    comp_delta = abs(_safe_float(feats.get("mechanics_completed_delta", 0.0), 0.0))
    motion_bonus = 0.0
    if (prog_delta + comp_delta) > 0.0:
        motion_bonus = min(0.15, 0.05 + 0.05 * prog_delta + 0.03 * comp_delta)
    score = max(0.0, min(1.0, mean_core + motion_bonus))
    return float(score)


def _augment_translated_file_with_lane_metadata(
    *,
    path: str,
    source: _SourceDNA,
    target_verse: str,
    universe_adapter_enabled: bool,
    far_lane_score_weight_enabled: bool = True,
    far_lane_score_weight_strength: float = 0.35,
    far_lane_min_universe_feature_score: float = 0.0,
) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {
            "path": str(path),
            "updated_rows": 0,
            "adapter_rows": 0,
            "lane": str(source.source_lane),
            "dropped_rows": 0,
        }
    tmp = path + ".lane.tmp"
    updated = 0
    adapter_rows = 0
    dropped_rows = 0
    weighted_rows = 0
    universe_feature_scores: List[float] = []
    with open(path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
        for line in src:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            row["source_lane"] = str(source.source_lane)
            row["source_universe"] = str(source.source_universe)
            row["target_universe"] = str(primary_universe_for_verse(str(target_verse)) or "")
            if bool(universe_adapter_enabled):
                meta = transfer_row_universe_metadata(
                    source_verse=str(source.verse_name),
                    target_verse=str(target_verse),
                    translated_obs=row.get("obs"),
                    translated_next_obs=row.get("next_obs"),
                )
                if isinstance(meta, dict):
                    row["universe_transfer"] = meta
                    adapter_rows += 1
            uf_score = _universe_feature_score_from_row(row)
            if uf_score is not None:
                row["universe_feature_score"] = float(uf_score)
                universe_feature_scores.append(float(uf_score))
            lane = str(source.source_lane or "")
            if (
                lane == "far_universe"
                and float(max(0.0, min(1.0, far_lane_min_universe_feature_score))) > 0.0
                and uf_score is not None
                and float(uf_score) < float(max(0.0, min(1.0, far_lane_min_universe_feature_score)))
            ):
                dropped_rows += 1
                continue
            if (
                lane == "far_universe"
                and bool(far_lane_score_weight_enabled)
                and uf_score is not None
                and isinstance(row.get("transfer_score"), (int, float))
            ):
                strength = max(0.0, min(1.0, float(far_lane_score_weight_strength)))
                mult = (1.0 - strength) + (strength * float(uf_score))
                base_ts = float(_safe_float(row.get("transfer_score", 0.0), 0.0))
                row["transfer_score_pre_lane_weight"] = float(base_ts)
                row["far_lane_weight_multiplier"] = float(mult)
                row["transfer_score"] = float(base_ts * mult)
                weighted_rows += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            updated += 1
    os.replace(tmp, path)
    qs = _quantiles(universe_feature_scores)
    return {
        "path": str(path),
        "updated_rows": int(updated),
        "adapter_rows": int(adapter_rows),
        "lane": str(source.source_lane),
        "dropped_rows": int(dropped_rows),
        "far_lane_weighted_rows": int(weighted_rows),
        "universe_feature_score": {
            "mean": (float(sum(universe_feature_scores) / float(len(universe_feature_scores))) if universe_feature_scores else None),
            "p10": qs.get("p10"),
            "p50": qs.get("p50"),
            "p90": qs.get("p90"),
        },
    }


def _merge_transfer_files_by_lane(
    *,
    sources: List[_SourceDNA],
    translated_paths_by_source: List[Tuple[_SourceDNA, str]],
    out_path: str,
    max_rows_per_source: int,
    near_lane_max_rows_per_source: int,
    far_lane_max_rows_per_source: int,
    far_lane_enabled: bool,
    near_lane_enabled: bool,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lane_file_counts: Dict[str, int] = {}
    lane_rows_written: Dict[str, int] = {}
    per_source_rows_written: List[Dict[str, Any]] = []
    total_rows = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for src, p in translated_paths_by_source:
            if not os.path.isfile(p):
                continue
            lane = str(src.source_lane or "unknown")
            if lane == "near_universe" and not bool(near_lane_enabled):
                continue
            if lane == "far_universe" and not bool(far_lane_enabled):
                continue

            lane_file_counts[lane] = int(lane_file_counts.get(lane, 0)) + 1

            if lane == "near_universe":
                cap = int(near_lane_max_rows_per_source) if int(near_lane_max_rows_per_source) > 0 else int(max_rows_per_source)
            elif lane == "far_universe":
                if int(far_lane_max_rows_per_source) > 0:
                    cap = int(far_lane_max_rows_per_source)
                else:
                    cap = _default_far_lane_cap(base_cap=int(max_rows_per_source))
            else:
                cap = int(max_rows_per_source)

            taken = 0
            for row in _iter_jsonl(p):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1
                taken += 1
                lane_rows_written[lane] = int(lane_rows_written.get(lane, 0)) + 1
                if int(cap) > 0 and taken >= int(cap):
                    break
            per_source_rows_written.append(
                {
                    "verse_name": str(src.verse_name),
                    "run_id": str(src.run_id),
                    "source_kind": str(src.source_kind),
                    "source_lane": str(src.source_lane),
                    "source_universe": str(src.source_universe),
                    "rows_written": int(taken),
                    "cap_used": int(cap),
                    "path": str(p),
                }
            )

    return {
        "rows_written": int(total_rows),
        "lane_file_counts": lane_file_counts,
        "lane_rows_written": lane_rows_written,
        "per_source_rows_written": per_source_rows_written,
        "lane_controls": {
            "near_lane_enabled": bool(near_lane_enabled),
            "far_lane_enabled": bool(far_lane_enabled),
            "max_rows_per_source": int(max_rows_per_source),
            "near_lane_max_rows_per_source": int(near_lane_max_rows_per_source),
            "far_lane_max_rows_per_source": int(far_lane_max_rows_per_source),
            "far_lane_default_cap_if_zero": int(_default_far_lane_cap(base_cap=int(max_rows_per_source))),
        },
    }


def _target_action_count(target_verse: str) -> int:
    v = str(target_verse).strip().lower()
    if v in {"warehouse_world", "labyrinth_world", "escape_world"}:
        return 5
    if v in {"chess_world", "go_world", "uno_world"}:
        return 6
    if v == "bridge_world":
        return 4
    if v == "trade_world":
        return 3
    if v == "factory_world":
        return 7  # 3 machines * 2 + 1 idle
    return 5


def _normalize_target_obs(target_verse: str, obs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obs, dict):
        return None
    v = str(target_verse).strip().lower()
    if v == "warehouse_world":
        x = max(0, min(7, _safe_int(obs.get("x", 0), 0)))
        y = max(0, min(7, _safe_int(obs.get("y", 0), 0)))
        gx = max(0, min(7, _safe_int(obs.get("goal_x", 7), 7)))
        gy = max(0, min(7, _safe_int(obs.get("goal_y", 7), 7)))
        battery = max(0, min(100, _safe_int(obs.get("battery", 20), 20)))
        nearby = max(0, min(4, _safe_int(obs.get("nearby_obstacles", 0), 0)))
        nearest = max(-1, min(25, _safe_int(obs.get("nearest_charger_dist", -1), -1)))
        t = max(0, _safe_int(obs.get("t", 0), 0))
        on_conveyor = max(0, min(1, _safe_int(obs.get("on_conveyor", 0), 0)))
        patrol_dist = max(0, min(25, _safe_int(obs.get("patrol_dist", 4), 4)))
        raw_lidar = obs.get("lidar")
        lidar: List[int] = []
        if isinstance(raw_lidar, list):
            for i in range(8):
                v_i = raw_lidar[i] if i < len(raw_lidar) else 1
                lidar.append(max(0, min(25, _safe_int(v_i, 1))))
        else:
            lidar = [1, 1, 4, 4, 1, 1, 1, 1]
        flat = [
            float(x),
            float(y),
            float(gx),
            float(gy),
            float(battery),
            float(nearby),
            float(t),
            float(on_conveyor),
            float(patrol_dist),
        ] + [float(v_i) for v_i in lidar]
        return {
            "x": int(x),
            "y": int(y),
            "goal_x": int(gx),
            "goal_y": int(gy),
            "battery": int(battery),
            "nearby_obstacles": int(nearby),
            "nearest_charger_dist": int(nearest),
            "t": int(t),
            "on_conveyor": int(on_conveyor),
            "patrol_dist": int(patrol_dist),
            "lidar": [int(v_i) for v_i in lidar],
            "flat": flat,
        }
    if v == "labyrinth_world":
        x = max(0, min(20, _safe_int(obs.get("x", 1), 1)))
        y = max(0, min(20, _safe_int(obs.get("y", 1), 1)))
        t = max(0, _safe_int(obs.get("t", 0), 0))
        battery = max(0, min(200, _safe_int(obs.get("battery", 50), 50)))
        out = {
            "x": int(x),
            "y": int(y),
            "t": int(t),
            "battery": int(battery),
            "goal_visible": max(0, min(1, _safe_int(obs.get("goal_visible", 0), 0))),
            "goal_dx": _safe_int(obs.get("goal_dx", 0), 0),
            "goal_dy": _safe_int(obs.get("goal_dy", 0), 0),
            "hazard_up": max(0, min(1, _safe_int(obs.get("hazard_up", 0), 0))),
            "hazard_down": max(0, min(1, _safe_int(obs.get("hazard_down", 0), 0))),
            "hazard_left": max(0, min(1, _safe_int(obs.get("hazard_left", 0), 0))),
            "hazard_right": max(0, min(1, _safe_int(obs.get("hazard_right", 0), 0))),
            "near_pits": max(0, min(12, _safe_int(obs.get("near_pits", 0), 0))),
            "near_lasers": max(0, min(12, _safe_int(obs.get("near_lasers", 0), 0))),
        }
        return out
    
    if v == "escape_world":
        # Escape: 10x10, user at x,y
        # Normalize to relevant keys
        x = max(0, min(9, _safe_int(obs.get("x", 0), 0)))
        y = max(0, min(9, _safe_int(obs.get("y", 0), 0)))
        t = max(0, _safe_int(obs.get("t", 0), 0))
        return {
            "x": int(x),
            "y": int(y),
            "exit_dist": _safe_int(obs.get("exit_dist", 0), 0),
            "nearest_guard_dist": _safe_int(obs.get("nearest_guard_dist", 0), 0),
            "hidden_steps_left": _safe_int(obs.get("hidden_steps_left", 0), 0),
            "guards_in_vision": _safe_int(obs.get("guards_in_vision", 0), 0),
            "on_hiding_spot": _safe_int(obs.get("on_hiding_spot", 0), 0),
            "t": int(t),
        }

    if v == "bridge_world":
        # Bridge: 1D cursor 0..8
        cursor = max(0, min(8, _safe_int(obs.get("cursor", 0), 0)))
        placed = max(0, min(8, _safe_int(obs.get("segments_placed", 0), 0)))
        return {
            "cursor": int(cursor),
            "segments_placed": int(placed),
            "weak_count": _safe_int(obs.get("weak_count", 0), 0),
            "strong_count": _safe_int(obs.get("strong_count", 0), 0),
            "wind_active": _safe_int(obs.get("wind_active", 0), 0),
            "bridge_complete": _safe_int(obs.get("bridge_complete", 0), 0),
            "t": _safe_int(obs.get("t", 0), 0),
        }

    if v == "factory_world":
        # Factory: machines
        out = {
            "t": _safe_int(obs.get("t", 0), 0),
            "completed": _safe_int(obs.get("completed", 0), 0),
            "total_arrived": _safe_int(obs.get("total_arrived", 0), 0),
            "output_buf": _safe_int(obs.get("output_buf", 0), 0),
        }
        for i in range(3):
            out[f"buf_{i}"] = _safe_int(obs.get(f"buf_{i}", 0), 0)
            out[f"broken_{i}"] = _safe_int(obs.get(f"broken_{i}", 0), 0)
            out[f"repair_{i}"] = _safe_int(obs.get(f"repair_{i}", 0), 0)
        return out

    if v == "trade_world":
        return {
            "price": _safe_float(obs.get("price", 0.0), 0.0),
            "cash": _safe_float(obs.get("cash", 0.0), 0.0),
            "inventory": _safe_int(obs.get("inventory", 0), 0),
            "portfolio_value": _safe_float(obs.get("portfolio_value", 0.0), 0.0),
            "avg_buy_price": _safe_float(obs.get("avg_buy_price", 0.0), 0.0),
            "t": _safe_int(obs.get("t", 0), 0),
        }

    if v == "chess_world":
        return {
            "material_delta": _safe_int(obs.get("material_delta", 0), 0),
            "development": _safe_int(obs.get("development", 0), 0),
            "king_safety": _safe_int(obs.get("king_safety", 5), 5),
            "center_control": _safe_int(obs.get("center_control", 0), 0),
            "phase": _safe_int(obs.get("phase", 0), 0),
            "score_delta": _safe_int(obs.get("score_delta", 0), 0),
            "pressure": _safe_int(obs.get("pressure", 0), 0),
            "risk": _safe_int(obs.get("risk", 0), 0),
            "tempo": _safe_int(obs.get("tempo", 0), 0),
            "control": _safe_int(obs.get("control", 0), 0),
            "resource": _safe_int(obs.get("resource", 0), 0),
            "t": _safe_int(obs.get("t", 0), 0),
        }

    if v == "go_world":
        return {
            "territory_delta": _safe_int(obs.get("territory_delta", 0), 0),
            "liberties_delta": _safe_int(obs.get("liberties_delta", 0), 0),
            "influence": _safe_int(obs.get("influence", 0), 0),
            "capture_threat": _safe_int(obs.get("capture_threat", 0), 0),
            "ko_risk": _safe_int(obs.get("ko_risk", 0), 0),
            "consecutive_passes": _safe_int(obs.get("consecutive_passes", 0), 0),
            "score_delta": _safe_int(obs.get("score_delta", 0), 0),
            "pressure": _safe_int(obs.get("pressure", 0), 0),
            "risk": _safe_int(obs.get("risk", 0), 0),
            "tempo": _safe_int(obs.get("tempo", 0), 0),
            "control": _safe_int(obs.get("control", 0), 0),
            "resource": _safe_int(obs.get("resource", 0), 0),
            "t": _safe_int(obs.get("t", 0), 0),
        }

    if v == "uno_world":
        return {
            "my_cards": _safe_int(obs.get("my_cards", 7), 7),
            "opp_cards": _safe_int(obs.get("opp_cards", 7), 7),
            "color_control": _safe_int(obs.get("color_control", 0), 0),
            "action_charge": _safe_int(obs.get("action_charge", 0), 0),
            "draw_pressure": _safe_int(obs.get("draw_pressure", 0), 0),
            "uno_ready": _safe_int(obs.get("uno_ready", 0), 0),
            "score_delta": _safe_int(obs.get("score_delta", 0), 0),
            "pressure": _safe_int(obs.get("pressure", 0), 0),
            "risk": _safe_int(obs.get("risk", 0), 0),
            "tempo": _safe_int(obs.get("tempo", 0), 0),
            "control": _safe_int(obs.get("control", 0), 0),
            "resource": _safe_int(obs.get("resource", 0), 0),
            "t": _safe_int(obs.get("t", 0), 0),
        }

    return obs


def _filter_transfer_dataset(
    *,
    path: str,
    target_verse: str,
    dedupe: bool,
    max_rows: int,
    hazard_keep_ratio: float = 1.0,
) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {
            "enabled": True,
            "path": path,
            "input_rows": 0,
            "kept_rows": 0,
            "dropped_invalid": 0,
            "dropped_duplicates": 0,
        }
    tmp = path + ".filtered.tmp"
    action_n = max(1, _target_action_count(target_verse))
    seen: set[str] = set()
    input_rows = 0
    kept_rows = 0
    dropped_invalid = 0
    dropped_dup = 0
    dropped_hazard = 0
    hz_ratio = max(0.0, min(1.0, float(hazard_keep_ratio)))
    hazard_keys = {
        "hit_wall",
        "hit_obstacle",
        "battery_death",
        "battery_depleted",
        "fell_cliff",
        "fell_pit",
        "hit_laser",
    }
    with open(path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
        for line in src:
            s = line.strip()
            if not s:
                continue
            input_rows += 1
            try:
                row = json.loads(s)
            except Exception:
                dropped_invalid += 1
                continue
            if not isinstance(row, dict):
                dropped_invalid += 1
                continue
            try:
                action = int(row.get("action"))
            except Exception:
                dropped_invalid += 1
                continue
            if action < 0 or action >= int(action_n):
                dropped_invalid += 1
                continue
            reward = _safe_float(row.get("reward", 0.0), 0.0)
            if not math.isfinite(float(reward)) or abs(float(reward)) > 1e6:
                dropped_invalid += 1
                continue
            obs_raw = row.get("obs")
            nxt_raw = row.get("next_obs")
            if nxt_raw is None:
                nxt_raw = obs_raw
            obs = _normalize_target_obs(str(target_verse), obs_raw)
            nxt = _normalize_target_obs(str(target_verse), nxt_raw)
            if obs is None or nxt is None:
                dropped_invalid += 1
                continue
            out_row = dict(row)
            out_row["action"] = int(action)
            out_row["reward"] = float(reward)
            out_row["obs"] = obs
            out_row["next_obs"] = nxt
            out_row["done"] = bool(row.get("done", False))
            out_row["truncated"] = bool(row.get("truncated", False))
            info = row.get("info")
            info = info if isinstance(info, dict) else {}
            out_row["info"] = info
            dedupe_key = ""
            obs_key = json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            nxt_key = json.dumps(nxt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            dedupe_key = (
                obs_key
                + f"|{int(action)}|"
                + nxt_key
                + f"|{int(bool(out_row['done'] or out_row['truncated']))}"
            )
            is_hazard = any(bool(info.get(k, False)) for k in hazard_keys)
            if is_hazard and hz_ratio < 1.0:
                digest = hashlib.md5(dedupe_key.encode("utf-8")).hexdigest()
                frac = int(digest[:8], 16) / float(16**8 - 1)
                if float(frac) > float(hz_ratio):
                    dropped_hazard += 1
                    continue
            if bool(dedupe):
                if dedupe_key in seen:
                    dropped_dup += 1
                    continue
                seen.add(dedupe_key)
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept_rows += 1
            if int(max_rows) > 0 and kept_rows >= int(max_rows):
                break
    os.replace(tmp, path)
    return {
        "enabled": True,
        "path": path,
        "input_rows": int(input_rows),
        "kept_rows": int(kept_rows),
        "dropped_invalid": int(dropped_invalid),
        "dropped_duplicates": int(dropped_dup),
        "dropped_hazard": int(dropped_hazard),
        "hazard_keep_ratio": float(hz_ratio),
        "dedupe": bool(dedupe),
        "action_space_n": int(action_n),
    }


def _auto_transfer_mix_decay_steps(
    *,
    episodes: int,
    max_steps: int,
    transfer_rows: int,
    mix_start: float,
    mix_end: float,
) -> int:
    total_online = max(1, int(episodes) * int(max_steps))
    if float(mix_end) >= float(mix_start):
        return max(1, int(round(0.50 * float(total_online))))
    data_ratio = max(0.0, min(2.0, float(transfer_rows) / float(max(1, total_online))))
    # Aggressive decay: transfer influence fades within first 15-30% of training.
    # This limits the window during which noisy cross-domain data can interfere
    # with online learning. The agent gets a brief boost, then relies on experience.
    frac = 0.15 + 0.10 * min(1.0, data_ratio) + 0.05 * min(1.0, float(mix_start) - float(mix_end))
    frac = max(0.10, min(0.30, frac))
    return max(1, int(round(float(total_online) * float(frac))))


def _auto_safe_veto_schedule_steps(
    *,
    episodes: int,
    max_steps: int,
    transfer_rows: int,
) -> int:
    total_online = max(1, int(episodes) * int(max_steps))
    data_ratio = max(0.0, min(2.0, float(transfer_rows) / float(max(1, total_online))))
    frac = 0.25 + 0.20 * min(1.0, data_ratio)
    frac = max(0.20, min(0.60, frac))
    return max(1, int(round(float(total_online) * float(frac))))


def _recent_hazard_trend_for_target(
    *,
    runs_root: str,
    target_verse: str,
    policy_prefix: str,
    max_runs: int,
) -> Dict[str, Any]:
    rows: List[Tuple[float, str]] = []
    for run_dir in _list_run_dirs(str(runs_root)):
        events_path = os.path.join(run_dir, "events.jsonl")
        first = _peek_first_event(events_path)
        if not first:
            continue
        verse_name = str(first.get("verse_name", "")).strip().lower()
        if verse_name != str(target_verse).strip().lower():
            continue
        policy_id = str(first.get("policy_id", "")).strip().lower()
        prefix = str(policy_prefix).strip().lower()
        if prefix and not policy_id.startswith(prefix):
            continue
        rows.append((_safe_float(os.path.getmtime(events_path), 0.0), run_dir))
    rows.sort(key=lambda t: float(t[0]), reverse=True)
    pick = rows[: max(1, int(max_runs))]
    if not pick:
        return {
            "num_runs": 0,
            "mean_hazard_trend_ratio": None,
            "mean_hazard_per_1k": None,
            "mean_mcts_veto_rate": None,
            "improving_share": None,
        }

    ratios: List[float] = []
    hazard_levels: List[float] = []
    veto_rates: List[float] = []
    improving = 0
    for _, run_dir in pick:
        curve = _episode_curve(run_dir)
        if not curve:
            continue
        trend = _safety_trend(curve)
        first_h = _safe_float((trend.get("first_half") or {}).get("hazard_per_1k_steps", 0.0), 0.0)
        second_h = _safe_float((trend.get("second_half") or {}).get("hazard_per_1k_steps", 0.0), 0.0)
        ratio = (float(second_h) / float(max(1e-9, first_h))) if first_h > 0.0 else (2.0 if second_h > 0.0 else 1.0)
        ratios.append(float(ratio))
        agg = _aggregate_curve(curve)
        hazard_levels.append(float(_safe_float(agg.get("hazard_events_per_1k_steps", 0.0), 0.0)))
        veto_rates.append(float(_safe_float(agg.get("mcts_veto_rate", 0.0), 0.0)))
        if bool(trend.get("hazard_rate_improved", False)):
            improving += 1

    if not ratios:
        return {
            "num_runs": 0,
            "mean_hazard_trend_ratio": None,
            "mean_hazard_per_1k": None,
            "mean_mcts_veto_rate": None,
            "improving_share": None,
        }

    return {
        "num_runs": int(len(ratios)),
        "mean_hazard_trend_ratio": float(sum(ratios) / float(len(ratios))),
        "mean_hazard_per_1k": float(sum(hazard_levels) / float(max(1, len(hazard_levels)))),
        "mean_mcts_veto_rate": float(sum(veto_rates) / float(max(1, len(veto_rates)))),
        "improving_share": float(improving / float(max(1, len(ratios)))),
    }


def _auto_tune_safe_veto_schedule(
    *,
    base_relax_start: float,
    base_relax_end: float,
    base_schedule_steps: int,
    base_schedule_power: float,
    trend: Dict[str, Any],
) -> Dict[str, Any]:
    relax_start = max(0.0, min(1.0, float(base_relax_start)))
    relax_end = max(0.0, min(1.0, float(base_relax_end)))
    schedule_steps = max(1, int(base_schedule_steps))
    schedule_power = max(0.10, float(base_schedule_power))
    num_runs = int(_safe_int(trend.get("num_runs", 0), 0))
    if num_runs <= 0:
        return {
            "applied": False,
            "reason": "no_history",
            "relax_start": float(relax_start),
            "relax_end": float(relax_end),
            "schedule_steps": int(schedule_steps),
            "schedule_power": float(schedule_power),
            "conservative_factor": 1.0,
        }

    ratio = _safe_float(trend.get("mean_hazard_trend_ratio", 1.0), 1.0)
    hazard = _safe_float(trend.get("mean_hazard_per_1k", 0.0), 0.0)
    improving_share = _safe_float(trend.get("improving_share", 0.5), 0.5)

    conservative = 1.0
    if ratio > 1.05:
        conservative *= min(1.80, 1.0 + (ratio - 1.0))
    elif ratio < 0.90:
        conservative *= max(0.70, 1.0 - (0.90 - ratio) * 0.60)

    if hazard >= 500.0:
        conservative *= 1.30
    elif hazard >= 350.0:
        conservative *= 1.15
    elif hazard <= 220.0:
        conservative *= 0.90

    if improving_share >= 0.70:
        conservative *= 0.90
    elif improving_share <= 0.30:
        conservative *= 1.10

    conservative = max(0.60, min(2.20, conservative))

    tuned_steps = max(1, int(round(float(schedule_steps) * float(conservative))))
    if conservative >= 1.0:
        tuned_end = max(0.01, min(1.0, float(relax_end) / float(max(1.0, conservative * 0.90))))
    else:
        tuned_end = max(0.01, min(1.0, float(relax_end) * (1.0 + (1.0 - conservative) * 0.60)))
    tuned_start = max(0.0, min(float(tuned_end), float(relax_start)))
    tuned_power = max(0.10, min(4.0, float(schedule_power) * (1.0 + max(0.0, conservative - 1.0) * 0.60)))

    return {
        "applied": True,
        "reason": "hazard_trend_auto_tune",
        "relax_start": float(tuned_start),
        "relax_end": float(tuned_end),
        "schedule_steps": int(tuned_steps),
        "schedule_power": float(tuned_power),
        "conservative_factor": float(conservative),
        "history_used": dict(trend),
    }


def _build_transfer_dataset(
    *,
    sources: List[_SourceDNA],
    target_verse: str,
    out_path: str,
    max_rows_per_source: int,
    near_lane_max_rows_per_source: int = 0,
    far_lane_max_rows_per_source: int = 0,
    near_lane_enabled: bool = True,
    far_lane_enabled: bool = True,
    universe_adapter_enabled: bool = True,
    far_lane_score_weight_enabled: bool = True,
    far_lane_score_weight_strength: float = 0.35,
    far_lane_min_universe_feature_score: float = 0.0,
    bridge_synthetic_reward_blend: float = 0.75,
    bridge_synthetic_done_union: bool = True,
    bridge_confidence_threshold: float = 0.35,
    bridge_label_cfg: Optional[Dict[str, Any]] = None,
    bridge_behavioral_enabled: bool = False,
    bridge_behavioral_score_weight: float = 0.35,
    bridge_behavioral_max_prototype_rows: int = 4096,
) -> Dict[str, Any]:
    translated_paths: List[str] = []
    translated_pairs: List[Tuple[_SourceDNA, str]] = []
    bridge_stats: List[Dict[str, Any]] = []
    translated_lane_annotation_stats: List[Dict[str, Any]] = []
    for src in sources:
        base = os.path.splitext(os.path.basename(src.path))[0]
        tmp_out = os.path.join(
            os.path.dirname(out_path) or ".",
            f"synthetic_transfer_{src.verse_name}_to_{target_verse}_{src.run_id}_{base}.jsonl",
        )
        st = translate_dna(
            source_dna_path=src.path,
            target_verse_name=target_verse,
            output_path=tmp_out,
            source_verse_name=src.verse_name,
            synthetic_reward_blend=float(bridge_synthetic_reward_blend),
            synthetic_done_union=bool(bridge_synthetic_done_union),
            confidence_threshold=max(0.0, min(1.0, float(bridge_confidence_threshold))),
            target_label_cfg=(dict(bridge_label_cfg) if isinstance(bridge_label_cfg, dict) else None),
            behavioral_bridge_enabled=bool(bridge_behavioral_enabled),
            behavioral_bridge_score_weight=max(0.0, min(1.0, float(bridge_behavioral_score_weight))),
            behavioral_max_prototype_rows=max(1, int(bridge_behavioral_max_prototype_rows)),
        )
        if st.translated_rows > 0:
            translated_paths.append(tmp_out)
            translated_pairs.append((src, tmp_out))
            translated_lane_annotation_stats.append(
                _augment_translated_file_with_lane_metadata(
                    path=tmp_out,
                    source=src,
                    target_verse=str(target_verse),
                    universe_adapter_enabled=bool(universe_adapter_enabled),
                    far_lane_score_weight_enabled=bool(far_lane_score_weight_enabled),
                    far_lane_score_weight_strength=max(0.0, min(1.0, float(far_lane_score_weight_strength))),
                    far_lane_min_universe_feature_score=max(
                        0.0, min(1.0, float(far_lane_min_universe_feature_score))
                    ),
                )
            )
        bridge_stats.append(
            {
                "source_verse": src.verse_name,
                "source_path": src.path,
                "source_kind": src.source_kind,
                "source_lane": str(src.source_lane),
                "source_universe": str(src.source_universe),
                "run_id": src.run_id,
                "output_path": tmp_out,
                "input_rows": int(st.input_rows),
                "translated_rows": int(st.translated_rows),
                "dropped_rows": int(st.dropped_rows),
                "learned_bridge_enabled": bool(st.learned_bridge_enabled),
                "learned_bridge_model_path": st.learned_bridge_model_path,
                "learned_scored_rows": int(st.learned_scored_rows),
                "behavioral_bridge_enabled": bool(st.behavioral_bridge_enabled),
                "behavioral_scored_rows": int(st.behavioral_scored_rows),
                "behavioral_prototype_rows": int(st.behavioral_prototype_rows),
                "bridge_reason": bridge_reason(src.verse_name, target_verse),
            }
        )
    lane_merge = _merge_transfer_files_by_lane(
        sources=sources,
        translated_paths_by_source=translated_pairs,
        out_path=out_path,
        max_rows_per_source=max(0, int(max_rows_per_source)),
        near_lane_max_rows_per_source=max(0, int(near_lane_max_rows_per_source)),
        far_lane_max_rows_per_source=max(0, int(far_lane_max_rows_per_source)),
        near_lane_enabled=bool(near_lane_enabled),
        far_lane_enabled=bool(far_lane_enabled),
    )
    merged_rows = int(lane_merge.get("rows_written", 0))
    return {
        "transfer_dataset_path": out_path,
        "transfer_dataset_rows": int(merged_rows),
        "translated_files": [str(p) for p in translated_paths],
        "translated_lane_annotation_stats": translated_lane_annotation_stats,
        "lane_merge": lane_merge,
        "lane_weighting": {
            "far_lane_score_weight_enabled": bool(far_lane_score_weight_enabled),
            "far_lane_score_weight_strength": float(max(0.0, min(1.0, float(far_lane_score_weight_strength)))),
            "far_lane_min_universe_feature_score": float(
                max(0.0, min(1.0, float(far_lane_min_universe_feature_score)))
            ),
            "universe_adapter_enabled": bool(universe_adapter_enabled),
        },
        "bridge_stats": bridge_stats,
    }


def _default_target_params(target_verse: str, *, max_steps: int) -> Dict[str, Any]:
    v = str(target_verse).strip().lower()
    if v == "warehouse_world":
        return {
            "max_steps": int(max_steps),
            "width": 8,
            "height": 8,
            # Slightly easier default curriculum to unblock early transfer adaptation.
            "obstacle_count": 10,
            "battery_capacity": 24,
            "battery_drain": 1,
            "charge_rate": 5,
        }
    if v == "labyrinth_world":
        return {
            "max_steps": int(max_steps),
            "width": 15,
            "height": 11,
            "battery_capacity": 80,
            "battery_drain": 1,
            "action_noise": 0.08,
        }
    if v == "escape_world":
        return {"max_steps": int(max_steps), "width": 10, "height": 10}
    if v == "bridge_world":
        return {"max_steps": int(max_steps)}
    if v == "factory_world":
        return {"max_steps": int(max_steps), "num_machines": 3}
    if v == "trade_world":
        return {"max_steps": int(max_steps)}
    return {"max_steps": int(max_steps)}


def _run_agent(
    *,
    trainer: Trainer,
    role: str,
    verse_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    algo: str,
    policy_id: str,
    cfg: Dict[str, Any],
) -> str:
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=str(verse_name),
        verse_version="0.1",
        seed=int(seed),
        tags=["transfer_challenge", str(role)],
        params=_default_target_params(verse_name, max_steps=max_steps),
    )
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=str(policy_id),
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
        tags=["transfer_challenge", str(role)],
        config=dict(cfg),
    )
    out = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
    )
    run_id = str(out.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError(f"missing run_id for {role}")
    return run_id


def _episode_curve(run_dir: str) -> List[Dict[str, Any]]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return []

    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for ev in _iter_jsonl(events_path):
        ep = str(ev.get("episode_id", "")).strip()
        if not ep:
            continue
        if ep not in by_ep:
            by_ep[ep] = []
            order.append(ep)
        by_ep[ep].append(ev)

    hazard_keys = {
        "hit_obstacle",
        "hit_wall",
        "battery_death",
        "battery_depleted",
        "fell_cliff",
        "fell_pit",
        "hit_laser",
        "collision",
        "crash",
        "unsafe",
    }

    out: List[Dict[str, Any]] = []
    for idx, ep in enumerate(order, start=1):
        rows = by_ep.get(ep, [])
        rows.sort(key=lambda r: _safe_int(r.get("step_idx", 0), 0))
        ret = 0.0
        steps = 0
        success = False
        hazards = 0
        mcts_veto_steps = 0
        shield_veto_steps = 0
        mcts_queries = 0
        mcts_vetoes = 0
        shield_vetoes_total = 0
        for r in rows:
            steps += 1
            ret += _safe_float(r.get("reward", 0.0), 0.0)
            info = r.get("info")
            info = info if isinstance(info, dict) else {}
            if bool(info.get("reached_goal", False)):
                success = True
            if any(bool(info.get(k, False)) for k in hazard_keys):
                hazards += 1
            se = info.get("safe_executor")
            se = se if isinstance(se, dict) else {}
            mode = str(se.get("mode", "")).strip().lower()
            if mode == "mcts_veto":
                mcts_veto_steps += 1
            if mode == "shield_veto":
                shield_veto_steps += 1
            mcts_stats = se.get("mcts_stats")
            mcts_stats = mcts_stats if isinstance(mcts_stats, dict) else {}
            mcts_queries = max(mcts_queries, max(0, _safe_int(mcts_stats.get("queries", 0), 0)))
            mcts_vetoes = max(mcts_vetoes, max(0, _safe_int(mcts_stats.get("vetoes", 0), 0)))
            counters = se.get("counters")
            counters = counters if isinstance(counters, dict) else {}
            shield_vetoes_total = max(shield_vetoes_total, max(0, _safe_int(counters.get("shield_vetoes", 0), 0)))
        out.append(
            {
                "episode_idx": int(idx),
                "episode_id": ep,
                "steps": int(steps),
                "return_sum": float(ret),
                "success": bool(success),
                "hazard_events": int(hazards),
                "mcts_veto_steps": int(mcts_veto_steps),
                "shield_veto_steps": int(shield_veto_steps),
                "mcts_queries": int(mcts_queries),
                "mcts_vetoes": int(mcts_vetoes),
                "shield_vetoes_total": int(shield_vetoes_total),
            }
        )
    return out


def _first_passable_episode(
    curve: List[Dict[str, Any]],
    *,
    window: int,
    passable_success_rate: float,
    passable_mean_return: float,
) -> Optional[int]:
    if not curve:
        return None
    w = max(1, int(window))
    for i in range(len(curve)):
        lo = max(0, i - w + 1)
        seg = curve[lo : i + 1]
        mean_ret = sum(_safe_float(r.get("return_sum", 0.0), 0.0) for r in seg) / float(len(seg))
        mean_succ = (
            sum(1 for r in seg if bool(r.get("success", False))) / float(len(seg))
        )
        if mean_succ >= float(passable_success_rate) or mean_ret >= float(passable_mean_return):
            return int(i + 1)
    return None


def _aggregate_curve(curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    episodes = len(curve)
    total_steps = sum(_safe_int(r.get("steps", 0), 0) for r in curve)
    total_return = sum(_safe_float(r.get("return_sum", 0.0), 0.0) for r in curve)
    successes = sum(1 for r in curve if bool(r.get("success", False)))
    hazards = sum(_safe_int(r.get("hazard_events", 0), 0) for r in curve)
    mcts_veto = sum(_safe_int(r.get("mcts_veto_steps", 0), 0) for r in curve)
    shield_veto = sum(_safe_int(r.get("shield_veto_steps", 0), 0) for r in curve)
    mcts_queries = sum(_safe_int(r.get("mcts_queries", 0), 0) for r in curve)
    mcts_vetoes = sum(_safe_int(r.get("mcts_vetoes", 0), 0) for r in curve)
    return {
        "episodes": int(episodes),
        "total_steps": int(total_steps),
        "mean_return": (float(total_return) / float(max(1, episodes))),
        "success_rate": float(successes / float(max(1, episodes))),
        "hazard_events": int(hazards),
        "hazard_events_per_1k_steps": float(1000.0 * float(hazards) / float(max(1, total_steps))),
        "mcts_veto_steps": int(mcts_veto),
        "shield_veto_steps": int(shield_veto),
        "mcts_queries": int(mcts_queries),
        "mcts_vetoes": int(mcts_vetoes),
        "mcts_veto_rate": float(mcts_vetoes / float(max(1, mcts_queries))),
    }


def _quantiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}
    vs = sorted(float(v) for v in values)

    def _q(p: float) -> float:
        idx = int(round(float(max(0.0, min(1.0, p))) * float(len(vs) - 1)))
        return float(vs[idx])

    return {
        "p10": float(_q(0.10)),
        "p50": float(_q(0.50)),
        "p90": float(_q(0.90)),
    }


def _transfer_score_diagnostics(dataset_path: str) -> Dict[str, Any]:
    if not os.path.isfile(dataset_path):
        return {
            "rows": 0,
            "transfer_score": {"mean": None, "min": None, "max": None, "p10": None, "p50": None, "p90": None},
            "universe_feature_score": {"mean": None, "p10": None, "p50": None, "p90": None},
            "by_lane": {},
        }
    vals: List[float] = []
    feature_vals: List[float] = []
    by_lane_scores: Dict[str, List[float]] = {}
    by_lane_feature_scores: Dict[str, List[float]] = {}
    by_lane_rows: Dict[str, int] = {}
    weighted_rows = 0
    dropped_far_in_premerge = 0
    for row in _iter_jsonl(dataset_path):
        if not isinstance(row, dict):
            continue
        score = float(_safe_float(row.get("transfer_score", 0.0), 0.0))
        vals.append(score)
        lane = str(row.get("source_lane", "unknown")).strip() or "unknown"
        by_lane_rows[lane] = int(by_lane_rows.get(lane, 0)) + 1
        by_lane_scores.setdefault(lane, []).append(score)
        if isinstance(row.get("far_lane_weight_multiplier"), (int, float)):
            weighted_rows += 1
        fs = row.get("universe_feature_score")
        if isinstance(fs, (int, float)):
            fsv = max(0.0, min(1.0, float(fs)))
            feature_vals.append(fsv)
            by_lane_feature_scores.setdefault(lane, []).append(fsv)
        # Kept for future compatibility if row-level drops are tracked inside merged files later.
        if bool(row.get("_far_lane_dropped_premerge", False)):
            dropped_far_in_premerge += 1
    qs = _quantiles(vals)
    fqs = _quantiles(feature_vals)
    by_lane: Dict[str, Any] = {}
    for lane, lane_vals in by_lane_scores.items():
        lqs = _quantiles(lane_vals)
        lf = by_lane_feature_scores.get(lane, [])
        lfqs = _quantiles(lf)
        by_lane[lane] = {
            "rows": int(by_lane_rows.get(lane, 0)),
            "transfer_score": {
                "mean": (float(sum(lane_vals) / float(max(1, len(lane_vals)))) if lane_vals else None),
                "min": (float(min(lane_vals)) if lane_vals else None),
                "max": (float(max(lane_vals)) if lane_vals else None),
                "p10": lqs.get("p10"),
                "p50": lqs.get("p50"),
                "p90": lqs.get("p90"),
            },
            "universe_feature_score": {
                "mean": (float(sum(lf) / float(max(1, len(lf)))) if lf else None),
                "p10": lfqs.get("p10"),
                "p50": lfqs.get("p50"),
                "p90": lfqs.get("p90"),
            },
        }
    return {
        "rows": int(len(vals)),
        "transfer_score": {
            "mean": (float(sum(vals) / float(max(1, len(vals)))) if vals else None),
            "min": (float(min(vals)) if vals else None),
            "max": (float(max(vals)) if vals else None),
            "p10": qs.get("p10"),
            "p50": qs.get("p50"),
            "p90": qs.get("p90"),
        },
        "universe_feature_score": {
            "mean": (float(sum(feature_vals) / float(max(1, len(feature_vals)))) if feature_vals else None),
            "p10": fqs.get("p10"),
            "p50": fqs.get("p50"),
            "p90": fqs.get("p90"),
        },
        "weighted_far_lane_rows": int(weighted_rows),
        "dropped_far_rows_flagged_in_rows": int(dropped_far_in_premerge),
        "by_lane": by_lane,
    }


def _early_window(curve: List[Dict[str, Any]], *, episodes: int) -> Dict[str, Any]:
    n = max(1, int(episodes))
    seg = list(curve[:n])
    if not seg:
        return {
            "episodes": 0,
            "mean_return": None,
            "success_rate": None,
            "hazard_events_per_1k_steps": None,
        }
    total_steps = sum(_safe_int(r.get("steps", 0), 0) for r in seg)
    total_return = sum(_safe_float(r.get("return_sum", 0.0), 0.0) for r in seg)
    successes = sum(1 for r in seg if bool(r.get("success", False)))
    hazards = sum(_safe_int(r.get("hazard_events", 0), 0) for r in seg)
    return {
        "episodes": int(len(seg)),
        "mean_return": float(total_return / float(max(1, len(seg)))),
        "success_rate": float(successes / float(max(1, len(seg)))),
        "hazard_events_per_1k_steps": float(1000.0 * float(hazards) / float(max(1, total_steps))),
    }


def _action_agreement_diagnostics(run_dir: str, *, first_k_steps: int) -> Dict[str, Any]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return {"rows": 0, "agreement_rate": None, "exploit_rate": None}
    k = max(1, int(first_k_steps))
    rows = 0
    matches = 0
    exploit_rows = 0
    by_ep_seen: Dict[str, int] = {}
    for ev in _iter_jsonl(events_path):
        ep = str(ev.get("episode_id", "")).strip()
        if not ep:
            continue
        seen = int(by_ep_seen.get(ep, 0))
        step_idx = _safe_int(ev.get("step_idx", seen), seen)
        if step_idx >= k:
            continue
        by_ep_seen[ep] = max(seen, step_idx + 1)
        info = ev.get("info")
        info = info if isinstance(info, dict) else {}
        action_info = info.get("action_info")
        action_info = action_info if isinstance(action_info, dict) else {}
        if "greedy_action" not in action_info:
            continue
        rows += 1
        chosen = _safe_int(ev.get("action"), -1)
        greedy = _safe_int(action_info.get("greedy_action"), -2)
        if chosen == greedy:
            matches += 1
        mode = str(action_info.get("mode", "")).strip().lower()
        if mode == "exploit":
            exploit_rows += 1
    return {
        "rows": int(rows),
        "agreement_rate": (float(matches / float(max(1, rows))) if rows > 0 else None),
        "exploit_rate": (float(exploit_rows / float(max(1, rows))) if rows > 0 else None),
        "first_k_steps_per_episode": int(k),
    }


def _train_td_diagnostics(run_dir: str, *, early_episodes: int) -> Dict[str, Any]:
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        return {
            "episodes_logged": 0,
            "early_episodes": 0,
            "native_td_abs_mean_early": None,
            "transfer_td_abs_mean_early": None,
            "transfer_td_score_corr_early": None,
            "transfer_replay_sampled_score_mean_early": None,
            "transfer_replay_weighted_sampling_enabled": None,
        }
    rows = [r for r in _iter_jsonl(metrics_path) if isinstance(r, dict)]
    n = max(1, int(early_episodes))
    early = rows[:n]

    def _mean(rows_: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals: List[float] = []
        for r in rows_:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return None
        return float(sum(vals) / float(max(1, len(vals))))

    def _bool_summary(rows_: List[Dict[str, Any]], key: str) -> Optional[bool]:
        vals: List[bool] = []
        for r in rows_:
            v = r.get(key)
            if isinstance(v, bool):
                vals.append(bool(v))
            elif isinstance(v, (int, float)):
                vals.append(bool(v))
        if not vals:
            return None
        # `any` is useful here: report whether weighted replay sampling was active
        # in at least one early episode metric payload.
        return bool(any(vals))

    return {
        "episodes_logged": int(len(rows)),
        "early_episodes": int(len(early)),
        "native_td_abs_mean_early": _mean(early, "native_td_abs_mean"),
        "native_td_abs_p90_early": _mean(early, "native_td_abs_p90"),
        "transfer_td_abs_mean_early": _mean(early, "transfer_td_abs_mean"),
        "transfer_td_abs_p90_early": _mean(early, "transfer_td_abs_p90"),
        "transfer_td_score_corr_early": _mean(early, "transfer_td_score_corr"),
        "transfer_replay_sampled_score_mean_early": _mean(early, "transfer_replay_sampled_score_mean"),
        "transfer_replay_weighted_sampling_enabled": _bool_summary(early, "transfer_replay_weighted_sampling"),
    }


def _speedup_summary(
    *,
    transfer_first_passable: Optional[int],
    baseline_first_passable: Optional[int],
    transfer_hazard_rate: float,
    baseline_hazard_rate: float,
) -> Dict[str, Any]:
    speedup = None
    transfer_wins_convergence = False
    if isinstance(transfer_first_passable, int) and isinstance(baseline_first_passable, int):
        speedup = float(baseline_first_passable) / float(max(1, transfer_first_passable))
        transfer_wins_convergence = transfer_first_passable < baseline_first_passable
    elif isinstance(transfer_first_passable, int) and baseline_first_passable is None:
        transfer_wins_convergence = True
        speedup = None

    hazard_improvement_abs = float(baseline_hazard_rate - transfer_hazard_rate)
    hazard_improvement_pct = (
        float(100.0 * hazard_improvement_abs / float(max(1e-9, baseline_hazard_rate)))
        if baseline_hazard_rate > 0.0
        else 0.0
    )
    return {
        "transfer_first_passable_episode": transfer_first_passable,
        "baseline_first_passable_episode": baseline_first_passable,
        "transfer_speedup_ratio": speedup,
        "transfer_wins_convergence": bool(transfer_wins_convergence),
        "hazard_improvement_per_1k_steps": float(hazard_improvement_abs),
        "hazard_improvement_pct": float(hazard_improvement_pct),
    }


def _safety_trend(curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not curve:
        return {
            "episodes": 0,
            "first_half": {"mcts_veto_rate": 0.0, "hazard_per_1k_steps": 0.0},
            "second_half": {"mcts_veto_rate": 0.0, "hazard_per_1k_steps": 0.0},
            "veto_rate_improved": False,
            "hazard_rate_improved": False,
        }
    mid = max(1, int(len(curve) // 2))
    first = curve[:mid]
    second = curve[mid:]
    if not second:
        second = curve

    def _agg(seg: List[Dict[str, Any]]) -> Dict[str, float]:
        steps = sum(_safe_int(r.get("steps", 0), 0) for r in seg)
        hazards = sum(_safe_int(r.get("hazard_events", 0), 0) for r in seg)
        mcts_q = sum(_safe_int(r.get("mcts_queries", 0), 0) for r in seg)
        mcts_v = sum(_safe_int(r.get("mcts_vetoes", 0), 0) for r in seg)
        return {
            "mcts_veto_rate": float(mcts_v / float(max(1, mcts_q))),
            "hazard_per_1k_steps": float(1000.0 * float(hazards) / float(max(1, steps))),
        }

    f = _agg(first)
    s = _agg(second)
    return {
        "episodes": int(len(curve)),
        "first_half": f,
        "second_half": s,
        "veto_rate_improved": bool(float(s["mcts_veto_rate"]) <= float(f["mcts_veto_rate"])),
        "hazard_rate_improved": bool(float(s["hazard_per_1k_steps"]) <= float(f["hazard_per_1k_steps"])),
    }


@dataclass
class _RunTraceProxy:
    verse_name: str
    rows: int
    mean_kl: Optional[float]
    prior_top1_match: Optional[float]
    high_quality_rate: Optional[float]


def _collect_run_trace_proxy(
    *,
    run_dir: str,
    max_rows: int,
) -> Optional[_RunTraceProxy]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return None

    verse_name = ""
    rows = 0
    query_rows = 0
    match_n = 0
    match_sum = 0.0
    disagree_n = 0
    disagree_sum = 0.0
    hq_n = 0
    hq_sum = 0.0

    for ev in _iter_jsonl(events_path):
        if not verse_name:
            verse_name = str(ev.get("verse_name", "")).strip().lower()
        rows += 1

        info = ev.get("info")
        info = info if isinstance(info, dict) else {}
        se = info.get("safe_executor")
        se = se if isinstance(se, dict) else {}
        mcts_stats = se.get("mcts_stats")
        mcts_stats = mcts_stats if isinstance(mcts_stats, dict) else {}
        last_query = mcts_stats.get("last_query")
        last_query = last_query if isinstance(last_query, dict) else {}
        if last_query:
            query_rows += 1
            action = _safe_int(ev.get("action", -1), -1)
            best = _safe_int(last_query.get("best_action", -1), -1)
            if action >= 0 and best >= 0:
                m = 1.0 if int(action) == int(best) else 0.0
                match_sum += float(m)
                match_n += 1
                disagree_sum += float(1.0 - m)
                disagree_n += 1

            forced_loss = bool(last_query.get("forced_loss_detected", False))
            hq_sum += 0.0 if forced_loss else 1.0
            hq_n += 1

        if int(max_rows) > 0 and rows >= int(max_rows):
            break

    if rows <= 0:
        return None
    disagree_rate = None if disagree_n <= 0 else float(disagree_sum / float(disagree_n))
    # Keep proxy KL on the same rough scale as trace-quality KL used elsewhere (~0.0-0.25+).
    proxy_kl = None if disagree_rate is None else float(0.25 * max(0.0, min(1.0, disagree_rate)))
    return _RunTraceProxy(
        verse_name=str(verse_name),
        rows=int(rows),
        mean_kl=proxy_kl,
        prior_top1_match=(None if match_n <= 0 else float(match_sum / float(match_n))),
        high_quality_rate=(None if hq_n <= 0 else float(hq_sum / float(hq_n))),
    )


def _build_health_scorecard(
    *,
    run_dirs_by_role: Dict[str, str],
    trace_root: str,
    kl_critical: float,
    stale_kl_threshold: float,
    unsafe_veto_rate: float,
    incoherent_match_threshold: float,
    memory_coherence_threshold: float,
    max_trace_rows_per_file: int,
) -> Dict[str, Any]:
    try:
        from tools.agent_health_monitor import (
            _collect_event_run_metrics,
            _collect_trace_metrics,
            _discover_trace_paths,
            _score_row,
        )
    except Exception as e:
        return {"enabled": False, "error": f"health_import_failed: {e}", "rows": [], "by_role": {}}

    try:
        trace_paths = _discover_trace_paths(str(trace_root))
        trace_by_verse = _collect_trace_metrics(
            trace_paths,
            max_rows_per_file=max(0, int(max_trace_rows_per_file)),
        )
    except Exception as e:
        return {"enabled": False, "error": f"health_trace_failed: {e}", "rows": [], "by_role": {}}

    rows: List[Dict[str, Any]] = []
    by_role: Dict[str, Dict[str, Any]] = {}
    local_trace_used = 0
    fallback_trace_used = 0
    for role, run_dir in run_dirs_by_role.items():
        rm = _collect_event_run_metrics(run_dir)
        if rm is None:
            continue
        try:
            st = evaluate_run(run_dir)
            mean_return = float(st.mean_return)
            success_rate = st.success_rate
        except Exception:
            mean_return = 0.0
            success_rate = None
        local_trace = _collect_run_trace_proxy(
            run_dir=run_dir,
            max_rows=max(0, int(max_trace_rows_per_file)),
        )
        trace_source = "run_events_proxy"
        trace = local_trace
        if trace is None or (
            trace.mean_kl is None and trace.prior_top1_match is None and int(getattr(trace, "rows", 0)) <= 0
        ):
            trace = trace_by_verse.get(str(rm.verse_name))
            trace_source = "global_trace_fallback" if trace is not None else "none"
            if trace is not None:
                fallback_trace_used += 1
        else:
            local_trace_used += 1
        row = _score_row(
            run_metrics=rm,
            trace_metrics=trace,
            mean_return=mean_return,
            success_rate=success_rate,
            market_reputation=None,
            kl_critical=float(kl_critical),
            unsafe_veto_rate=float(unsafe_veto_rate),
            incoherent_match_threshold=float(incoherent_match_threshold),
            memory_coherence_threshold=float(memory_coherence_threshold),
            stale_kl_threshold=float(stale_kl_threshold),
        )
        payload = {
            "role": str(role),
            "agent_id": row.agent_id,
            "run_id": row.run_id,
            "run_dir": row.run_dir,
            "verse_name": row.verse_name,
            "policy_id": row.policy_id,
            "mean_return": float(row.mean_return),
            "success_rate": row.success_rate,
            "intuition_match": row.intuition_match,
            "memory_coherence": row.memory_coherence,
            "search_regret_kl": row.search_regret_kl,
            "veto_rate": float(row.veto_rate),
            "shield_veto_rate": float(row.shield_veto_rate),
            "total_score": float(row.total_score),
            "status": str(row.status),
            "issues": list(row.issues),
            "recommended_actions": list(row.recommended_actions),
            "trace_source": str(trace_source),
            "trace_rows": int(getattr(trace, "rows", 0) if trace is not None else 0),
        }
        rows.append(payload)
        by_role[str(role)] = payload
    return {
        "enabled": True,
        "trace_paths": [str(p) for p in trace_paths],
        "trace_sources": {
            "run_events_proxy_used": int(local_trace_used),
            "global_trace_fallback_used": int(fallback_trace_used),
        },
        "rows": rows,
        "by_role": by_role,
    }


def _build_overlap_map(
    *,
    target_verse: str,
    sources: List[_SourceDNA],
    transfer_dataset_rows: int,
) -> Dict[str, Any]:
    return {
        "target_verse": str(target_verse),
        "bridge_family": "semantic_projection_v2",
        "bridge_reason": bridge_reason("chess_world", target_verse),
        "mappings": [
            {
                "source_feature": "score_delta (territory/material edge)",
                "target_feature": "path_progress (x,y toward goal)",
                "intuition": "Advantage conversion maps to route completion pressure.",
            },
            {
                "source_feature": "risk (ko danger / king exposure)",
                "target_feature": "shelf_safety (nearby_obstacles / hazard proxy)",
                "intuition": "Defensive board awareness maps to collision avoidance bias.",
            },
            {
                "source_feature": "resource + tempo",
                "target_feature": "battery + charger urgency",
                "intuition": "Resource pacing maps to energy-aware logistics control.",
            },
        ],
        "sources": [
            {
                "verse_name": str(s.verse_name),
                "run_id": str(s.run_id),
                "path": str(s.path),
                "source_kind": str(s.source_kind),
                "source_lane": str(s.source_lane),
                "source_universe": str(s.source_universe),
            }
            for s in sources
        ],
        "transfer_dataset_rows": int(transfer_dataset_rows),
    }


def _source_selection_summary(*, target_verse: str, sources: List[_SourceDNA]) -> Dict[str, Any]:
    plan = build_transfer_source_plan(str(target_verse))
    counts_by_lane: Dict[str, int] = {}
    counts_by_verse: Dict[str, int] = {}
    for s in sources:
        lane = str(s.source_lane or "unknown")
        counts_by_lane[lane] = int(counts_by_lane.get(lane, 0)) + 1
        vv = str(s.verse_name or "")
        if vv:
            counts_by_verse[vv] = int(counts_by_verse.get(vv, 0)) + 1
    return {
        "target_verse": str(target_verse),
        "target_universe": plan.get("target_universe"),
        "planned_near_sources": list(plan.get("near_sources") or []),
        "planned_far_sources": list(plan.get("far_sources") or []),
        "counts_by_lane": counts_by_lane,
        "counts_by_verse": counts_by_verse,
    }


def _print_chart(
    *,
    transfer_curve: List[Dict[str, Any]],
    baseline_curve: List[Dict[str, Any]],
    stride: int,
) -> None:
    s = max(1, int(stride))
    n = max(len(transfer_curve), len(baseline_curve))
    print("")
    print("Transfer Speedup Chart")
    print(f"{'ep':>4}  {'transfer_ret':>12}  {'baseline_ret':>12}  {'transfer_hz':>11}  {'baseline_hz':>11}")
    for i in range(0, n, s):
        t = transfer_curve[i] if i < len(transfer_curve) else {}
        b = baseline_curve[i] if i < len(baseline_curve) else {}
        print(
            f"{i+1:>4d}  "
            f"{_safe_float(t.get('return_sum', 0.0), 0.0):>12.3f}  "
            f"{_safe_float(b.get('return_sum', 0.0), 0.0):>12.3f}  "
            f"{_safe_int(t.get('hazard_events', 0), 0):>11d}  "
            f"{_safe_int(b.get('hazard_events', 0), 0):>11d}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--target_verse", type=str, default="warehouse_world")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--baseline_algo", type=str, default="q")
    ap.add_argument(
        "--baseline_cfg",
        action="append",
        default=None,
        help="Extra baseline agent config override in key=value form (repeatable).",
    )
    ap.add_argument("--baseline_epsilon_start", type=float, default=0.05)
    ap.add_argument("--baseline_epsilon_min", type=float, default=0.01)
    ap.add_argument("--baseline_epsilon_decay", type=float, default=0.999)
    ap.add_argument("--transfer_algo", type=str, default="q")
    ap.add_argument(
        "--transfer_cfg",
        action="append",
        default=None,
        help="Extra transfer agent config override in key=value form (repeatable).",
    )
    ap.add_argument("--source_dna", action="append", default=None)
    ap.add_argument("--source_verse", action="append", default=None)
    ap.add_argument("--max_source_runs_per_verse", type=int, default=2)
    ap.add_argument("--min_source_success_rate", type=float, default=0.55)
    ap.add_argument("--min_rows_per_source", type=int, default=50)
    ap.add_argument("--max_source_scan", type=int, default=200, help="Max strategy-verse runs to evaluate during discovery (0=unlimited)")
    ap.add_argument("--max_rows_per_source", type=int, default=2500)
    ap.add_argument("--near_lane_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--far_lane_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--near_lane_max_rows_per_source",
        type=int,
        default=0,
        help="0 => use --max_rows_per_source for near-universe sources",
    )
    ap.add_argument(
        "--far_lane_max_rows_per_source",
        type=int,
        default=0,
        help="0 => auto smaller cap for far-universe sources",
    )
    ap.add_argument(
        "--universe_adapter_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate translated rows with shared universe features (diagnostic metadata only).",
    )
    ap.add_argument(
        "--far_lane_score_weight_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downweight far-universe transfer rows using universe adapter feature quality.",
    )
    ap.add_argument(
        "--far_lane_score_weight_strength",
        type=float,
        default=0.35,
        help="Blend strength for far-lane score weighting (0..1).",
    )
    ap.add_argument(
        "--far_lane_min_universe_feature_score",
        type=float,
        default=0.0,
        help="Drop far-lane translated rows below this universe feature quality score (0..1).",
    )

    ap.add_argument("--bridge_synthetic_reward_blend", type=float, default=0.75)
    ap.add_argument("--bridge_synthetic_done_union", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bridge_confidence_threshold", type=float, default=0.35, help="Min confidence to keep a translated row (0..1)")
    ap.add_argument("--bridge_behavioral_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bridge_behavioral_score_weight", type=float, default=0.35)
    ap.add_argument("--bridge_behavioral_max_prototype_rows", type=int, default=4096)
    ap.add_argument("--bridge_warehouse_step_penalty", type=float, default=0.08)
    ap.add_argument("--bridge_warehouse_wall_penalty", type=float, default=0.40)
    ap.add_argument("--bridge_warehouse_obstacle_penalty", type=float, default=0.70)
    ap.add_argument("--bridge_warehouse_charge_reward", type=float, default=0.50)
    ap.add_argument("--bridge_warehouse_progress_bonus", type=float, default=0.35)
    ap.add_argument("--bridge_warehouse_regress_penalty", type=float, default=0.08)
    ap.add_argument("--bridge_warehouse_goal_reward", type=float, default=10.0)
    ap.add_argument("--bridge_warehouse_battery_fail_penalty", type=float, default=10.0)
    ap.add_argument("--transfer_filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--transfer_filter_dedupe", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--empty_transfer_dataset_policy",
        type=str,
        default="error",
        choices=["error", "continue"],
        help="Behavior when translated transfer dataset has zero rows.",
    )
    ap.add_argument(
        "--empty_transfer_dataset_diag_out",
        type=str,
        default="",
        help="Optional diagnostics JSON path written when transfer dataset is empty.",
    )
    ap.add_argument(
        "--transfer_filter_hazard_keep_ratio",
        type=float,
        default=1.0,
        help="Deterministic keep ratio for hazard-labeled synthetic rows (0..1).",
    )
    ap.add_argument("--transfer_filter_max_rows", type=int, default=0)
    ap.add_argument(
        "--transfer_dataset_out",
        type=str,
        default=os.path.join("models", "expert_datasets", "transfer_strategy_to_warehouse.jsonl"),
    )
    ap.add_argument("--transfer_warmstart_reward_scale", type=float, default=0.01)
    ap.add_argument("--transfer_warmstart_use_transfer_score", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--transfer_warmstart_transfer_score_min", type=float, default=0.0)
    ap.add_argument("--transfer_warmstart_transfer_score_max", type=float, default=2.0)
    ap.add_argument(
        "--transfer_q_warehouse_obs_key_mode",
        type=str,
        default="direction_only",
        choices=["direction_only"],
    )
    ap.add_argument("--transfer_epsilon_start", type=float, default=0.70)
    ap.add_argument("--transfer_epsilon_min", type=float, default=0.03)
    ap.add_argument("--transfer_epsilon_decay", type=float, default=0.996)
    ap.add_argument("--transfer_learn_hazard_penalty", type=float, default=0.0)
    ap.add_argument("--transfer_learn_success_bonus", type=float, default=0.0)
    ap.add_argument("--dynamic_transfer_mix", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--transfer_mix_start", type=float, default=1.0)
    ap.add_argument("--transfer_mix_end", type=float, default=0.0)
    ap.add_argument(
        "--transfer_mix_decay_steps",
        type=int,
        default=0,
        help="0 => auto: episodes*max_steps*0.6",
    )
    ap.add_argument("--transfer_mix_min_rows", type=int, default=32)
    ap.add_argument("--transfer_replay_reward_scale", type=float, default=0.8)
    ap.add_argument("--safe_transfer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_baseline", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--safe_adaptive_veto", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_adaptive_veto_relaxation", type=float, default=0.35)
    ap.add_argument("--safe_adaptive_veto_schedule", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_adaptive_veto_relax_start", type=float, default=0.12)
    ap.add_argument("--safe_adaptive_veto_relax_end", type=float, default=0.35)
    ap.add_argument("--safe_adaptive_veto_schedule_steps", type=int, default=0)
    ap.add_argument("--safe_adaptive_veto_schedule_power", type=float, default=1.20)
    ap.add_argument("--safe_adaptive_veto_auto_tune", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_adaptive_veto_auto_tune_max_runs", type=int, default=8)
    ap.add_argument("--safe_adaptive_veto_auto_tune_min_runs", type=int, default=2)
    ap.add_argument("--safe_adaptive_veto_auto_tune_policy_prefix", type=str, default="transfer_")
    ap.add_argument("--safe_adaptive_veto_warmup_steps", type=int, default=20)
    ap.add_argument("--safe_adaptive_veto_failure_guard", type=float, default=0.20)
    ap.add_argument("--safe_danger_threshold", type=float, default=0.85)
    ap.add_argument("--safe_min_action_confidence", type=float, default=0.10)
    ap.add_argument("--safe_prefer_fallback_on_veto", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--safe_fallback_algo", type=str, default="gateway")
    ap.add_argument(
        "--safe_fallback_manifest_path",
        type=str,
        default=os.path.join("models", "default_policy_set.json"),
    )
    ap.add_argument("--safe_fallback_manifest_section", type=str, default="deployment_ready_defaults")
    ap.add_argument("--enable_mcts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mcts_meta_model_path", type=str, default=os.path.join("models", "meta_transformer.pt"))
    ap.add_argument("--mcts_num_simulations", type=int, default=96)
    ap.add_argument("--mcts_max_depth", type=int, default=4)
    ap.add_argument("--mcts_loss_threshold", type=float, default=-0.98)
    ap.add_argument("--mcts_min_visits", type=int, default=12)
    ap.add_argument("--mcts_trigger_on_low_confidence", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mcts_value_confidence_threshold", type=float, default=0.30)
    ap.add_argument("--passable_success_rate", type=float, default=0.60)
    ap.add_argument("--passable_mean_return", type=float, default=1.50)
    ap.add_argument("--passable_window", type=int, default=20)
    ap.add_argument("--diagnostic_early_episodes", type=int, default=10)
    ap.add_argument("--diagnostic_action_agreement_first_k", type=int, default=20)
    ap.add_argument(
        "--baseline_q_warehouse_obs_key_mode",
        type=str,
        default="direction_only",
        choices=["direction_only"],
    )
    ap.add_argument("--chart_stride", type=int, default=5)
    ap.add_argument(
        "--health_trace_root",
        type=str,
        default=os.path.join("models", "expert_datasets"),
    )
    ap.add_argument("--health_kl_critical", type=float, default=0.25)
    ap.add_argument("--health_stale_kl_threshold", type=float, default=0.12)
    ap.add_argument("--health_unsafe_veto_rate", type=float, default=0.10)
    ap.add_argument("--health_incoherent_match_threshold", type=float, default=0.55)
    ap.add_argument("--health_memory_coherence_threshold", type=float, default=0.55)
    ap.add_argument("--health_max_trace_rows_per_file", type=int, default=20000)
    ap.add_argument(
        "--report_out",
        type=str,
        default=os.path.join("models", "tuning", "transfer_challenge_report.json"),
    )
    ap.add_argument(
        "--overlap_out",
        type=str,
        default=os.path.join("models", "tuning", "transfer_overlap_map.json"),
    )
    args = ap.parse_args()

    sources: List[_SourceDNA] = []
    if args.source_dna:
        hints = [str(v).strip().lower() for v in (args.source_verse or [])]
        for i, p in enumerate(args.source_dna):
            path = str(p).strip()
            if not path:
                continue
            verse_hint = hints[i] if i < len(hints) and hints[i] else ""
            if not verse_hint:
                verse_hint = infer_verse_from_obs(next(_iter_jsonl(path), {}).get("obs")) or ""
            verse_hint = str(verse_hint).strip().lower()
            if not verse_hint:
                continue
            sources.append(
                _SourceDNA(
                    verse_name=verse_hint,
                    path=path,
                    run_id=os.path.basename(os.path.dirname(path)) or "manual",
                    source_kind="explicit",
                    source_lane=source_transfer_lane(verse_hint, str(args.target_verse)),
                    source_universe=(primary_universe_for_verse(verse_hint) or ""),
                )
            )
    else:
        sources = _discover_transfer_sources(
            target_verse=str(args.target_verse),
            runs_root=str(args.runs_root),
            max_runs_per_verse=max(1, int(args.max_source_runs_per_verse)),
            min_success_rate=float(args.min_source_success_rate),
            min_rows_per_source=max(1, int(args.min_rows_per_source)),
            max_source_scan=max(0, int(args.max_source_scan)),
        )

    if not sources:
        raise RuntimeError(
            "No transfer DNA sources found. Provide --source_dna or generate runs/expert datasets "
            "for same-universe or far-transfer source verses."
        )

    bridge_label_cfg: Dict[str, Any] = {}
    if str(args.target_verse).strip().lower() == "warehouse_world":
        bridge_label_cfg = {
            "step_penalty": float(args.bridge_warehouse_step_penalty),
            "wall_penalty": float(args.bridge_warehouse_wall_penalty),
            "obstacle_penalty": float(args.bridge_warehouse_obstacle_penalty),
            "charge_reward": float(args.bridge_warehouse_charge_reward),
            "progress_bonus": float(args.bridge_warehouse_progress_bonus),
            "regress_penalty": float(args.bridge_warehouse_regress_penalty),
            "goal_reward": float(args.bridge_warehouse_goal_reward),
            "battery_fail_penalty": float(args.bridge_warehouse_battery_fail_penalty),
        }

    ds = _build_transfer_dataset(
        sources=sources,
        target_verse=str(args.target_verse),
        out_path=str(args.transfer_dataset_out),
        max_rows_per_source=max(0, int(args.max_rows_per_source)),
        near_lane_max_rows_per_source=max(0, int(args.near_lane_max_rows_per_source)),
        far_lane_max_rows_per_source=max(0, int(args.far_lane_max_rows_per_source)),
        near_lane_enabled=bool(args.near_lane_enabled),
        far_lane_enabled=bool(args.far_lane_enabled),
        universe_adapter_enabled=bool(args.universe_adapter_enabled),
        far_lane_score_weight_enabled=bool(args.far_lane_score_weight_enabled),
        far_lane_score_weight_strength=max(0.0, min(1.0, float(args.far_lane_score_weight_strength))),
        far_lane_min_universe_feature_score=max(0.0, min(1.0, float(args.far_lane_min_universe_feature_score))),
        bridge_synthetic_reward_blend=max(0.0, min(1.0, float(args.bridge_synthetic_reward_blend))),
        bridge_synthetic_done_union=bool(args.bridge_synthetic_done_union),
        bridge_confidence_threshold=max(0.0, min(1.0, float(args.bridge_confidence_threshold))),
        bridge_label_cfg=bridge_label_cfg,
        bridge_behavioral_enabled=bool(args.bridge_behavioral_enabled),
        bridge_behavioral_score_weight=max(0.0, min(1.0, float(args.bridge_behavioral_score_weight))),
        bridge_behavioral_max_prototype_rows=max(1, int(args.bridge_behavioral_max_prototype_rows)),
    )
    if bool(args.transfer_filter):
        fs = _filter_transfer_dataset(
            path=str(ds["transfer_dataset_path"]),
            target_verse=str(args.target_verse),
            dedupe=bool(args.transfer_filter_dedupe),
            max_rows=max(0, int(args.transfer_filter_max_rows)),
            hazard_keep_ratio=max(0.0, min(1.0, float(args.transfer_filter_hazard_keep_ratio))),
        )
        ds["filter_stats"] = fs
        ds["transfer_dataset_rows"] = int(fs.get("kept_rows", ds.get("transfer_dataset_rows", 0)))
    empty_transfer_dataset = int(ds.get("transfer_dataset_rows", 0)) <= 0
    if empty_transfer_dataset:
        diag = {
            "target_verse": str(args.target_verse),
            "reason": "transfer_dataset_empty_after_semantic_bridge_translation",
            "source_count": int(len(sources)),
            "sources": [
                {
                    "verse_name": s.verse_name,
                    "path": s.path,
                    "run_id": s.run_id,
                    "source_kind": s.source_kind,
                    "source_lane": s.source_lane,
                    "source_universe": s.source_universe,
                }
                for s in sources
            ],
            "transfer_dataset": dict(ds),
            "hint": (
                "Provide compatible --source_dna/--source_verse, lower --bridge_confidence_threshold, "
                "or run with --empty_transfer_dataset_policy continue to execute without warmstart."
            ),
        }
        diag_out = str(args.empty_transfer_dataset_diag_out or "").strip()
        if not diag_out:
            diag_out = str(args.report_out) + ".preflight.json"
        os.makedirs(os.path.dirname(diag_out) or ".", exist_ok=True)
        with open(diag_out, "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)
        ds["empty_dataset_diagnostics_path"] = str(diag_out)
        policy = str(args.empty_transfer_dataset_policy).strip().lower()
        if policy == "error":
            raise RuntimeError(
                "Transfer dataset is empty after semantic bridge translation. "
                f"Diagnostics: {diag_out}"
            )
        print(
            "warning: transfer dataset is empty after semantic bridge translation; "
            "continuing without warmstart transfer rows."
        )
        ds["empty_dataset_policy"] = "continue"

    transfer_safe_cfg: Dict[str, Any] = {}
    safe_tune_info: Dict[str, Any] = {"applied": False, "reason": "safe_transfer_disabled"}
    if bool(args.safe_transfer):
        mcts_model_path = str(args.mcts_meta_model_path or "")
        if not os.path.isfile(mcts_model_path):
            mcts_model_path = ""
        veto_schedule_steps = int(args.safe_adaptive_veto_schedule_steps)
        if veto_schedule_steps <= 0:
            veto_schedule_steps = _auto_safe_veto_schedule_steps(
                episodes=max(1, int(args.episodes)),
                max_steps=max(1, int(args.max_steps)),
                transfer_rows=max(0, int(ds.get("transfer_dataset_rows", 0))),
            )
        relax_start = max(0.0, min(1.0, float(args.safe_adaptive_veto_relax_start)))
        relax_end = max(0.0, min(1.0, float(args.safe_adaptive_veto_relax_end)))
        schedule_power = max(0.10, float(args.safe_adaptive_veto_schedule_power))
        if bool(args.safe_adaptive_veto_auto_tune):
            trend = _recent_hazard_trend_for_target(
                runs_root=str(args.runs_root),
                target_verse=str(args.target_verse),
                policy_prefix=str(args.safe_adaptive_veto_auto_tune_policy_prefix),
                max_runs=max(1, int(args.safe_adaptive_veto_auto_tune_max_runs)),
            )
            if int(_safe_int(trend.get("num_runs", 0), 0)) >= int(max(1, int(args.safe_adaptive_veto_auto_tune_min_runs))):
                safe_tune_info = _auto_tune_safe_veto_schedule(
                    base_relax_start=float(relax_start),
                    base_relax_end=float(relax_end),
                    base_schedule_steps=int(veto_schedule_steps),
                    base_schedule_power=float(schedule_power),
                    trend=trend,
                )
                relax_start = float(_safe_float(safe_tune_info.get("relax_start", relax_start), relax_start))
                relax_end = float(_safe_float(safe_tune_info.get("relax_end", relax_end), relax_end))
                veto_schedule_steps = int(_safe_int(safe_tune_info.get("schedule_steps", veto_schedule_steps), veto_schedule_steps))
                schedule_power = float(_safe_float(safe_tune_info.get("schedule_power", schedule_power), schedule_power))
            else:
                safe_tune_info = {
                    "applied": False,
                    "reason": "insufficient_history",
                    "history_used": trend,
                    "required_min_runs": int(max(1, int(args.safe_adaptive_veto_auto_tune_min_runs))),
                }
        else:
            safe_tune_info = {"applied": False, "reason": "auto_tune_disabled"}
        fallback_algo = str(args.safe_fallback_algo or "").strip().lower()
        fallback_cfg: Dict[str, Any] = {}
        fallback_manifest_path = str(args.safe_fallback_manifest_path or "").strip()
        fallback_manifest_section = str(args.safe_fallback_manifest_section or "").strip()
        if fallback_manifest_path:
            fallback_cfg["manifest_path"] = fallback_manifest_path
        if fallback_manifest_section:
            fallback_cfg["manifest_section"] = fallback_manifest_section
        if fallback_algo in ("gateway", "special_moe", "adaptive_moe"):
            fallback_cfg.setdefault("verse_name", str(args.target_verse))
        transfer_safe_cfg = {
            "enabled": True,
            "danger_threshold": max(0.0, min(1.0, float(args.safe_danger_threshold))),
            "min_action_confidence": max(0.0, min(1.0, float(args.safe_min_action_confidence))),
            "adaptive_veto_enabled": bool(args.safe_adaptive_veto),
            # Backward-compatible scalar knob kept as the terminal schedule strength.
            "adaptive_veto_relaxation": max(0.0, min(1.0, float(args.safe_adaptive_veto_relaxation))),
            "adaptive_veto_schedule_enabled": bool(args.safe_adaptive_veto_schedule),
            "adaptive_veto_relaxation_start": float(relax_start),
            "adaptive_veto_relaxation_end": float(relax_end),
            "adaptive_veto_schedule_steps": int(veto_schedule_steps),
            "adaptive_veto_schedule_power": float(schedule_power),
            "adaptive_veto_warmup_steps": max(0, int(args.safe_adaptive_veto_warmup_steps)),
            "adaptive_veto_failure_guard": max(1e-6, float(args.safe_adaptive_veto_failure_guard)),
            "prefer_fallback_on_veto": bool(args.safe_prefer_fallback_on_veto),
            "fallback_algo": str(fallback_algo),
            "fallback_config": dict(fallback_cfg),
            "planner_enabled": True,
            "planner_trigger_on_block": True,
            "planner_trigger_on_high_danger": True,
            "mcts_enabled": bool(args.enable_mcts),
            "mcts_num_simulations": max(8, int(args.mcts_num_simulations)),
            "mcts_max_depth": max(2, int(args.mcts_max_depth)),
            "mcts_loss_threshold": float(args.mcts_loss_threshold),
            "mcts_min_visits": max(1, int(args.mcts_min_visits)),
            "mcts_trigger_on_low_confidence": bool(args.mcts_trigger_on_low_confidence),
            "mcts_trigger_on_high_danger": True,
            "mcts_trigger_on_block": True,
            "mcts_meta_model_path": mcts_model_path,
            "mcts_value_confidence_threshold": max(0.0, min(1.0, float(args.mcts_value_confidence_threshold))),
        }

    baseline_safe_cfg: Dict[str, Any] = {}
    if bool(args.safe_baseline):
        baseline_safe_cfg = {"enabled": True}

    baseline_cfg: Dict[str, Any] = {
        "train": True,
        "epsilon_start": float(args.baseline_epsilon_start),
        "epsilon_min": float(args.baseline_epsilon_min),
        "epsilon_decay": float(args.baseline_epsilon_decay),
        "diag_temperature": 1.0,
        "warehouse_obs_key_mode": str(args.baseline_q_warehouse_obs_key_mode),
    }
    baseline_cfg.update(_parse_cfg_overrides(args.baseline_cfg))
    if baseline_safe_cfg:
        baseline_cfg["safe_executor"] = baseline_safe_cfg

    transfer_cfg: Dict[str, Any] = {
        "train": True,
        "dataset_path": str(ds["transfer_dataset_path"]),
        "warmstart_reward_scale": float(args.transfer_warmstart_reward_scale),
        "warmstart_use_transfer_score": bool(args.transfer_warmstart_use_transfer_score),
        "warmstart_transfer_score_min": float(args.transfer_warmstart_transfer_score_min),
        "warmstart_transfer_score_max": float(args.transfer_warmstart_transfer_score_max),
        "warehouse_obs_key_mode": str(args.transfer_q_warehouse_obs_key_mode),
        "epsilon_start": float(args.transfer_epsilon_start),
        "epsilon_min": float(args.transfer_epsilon_min),
        "epsilon_decay": float(args.transfer_epsilon_decay),
        "learn_hazard_penalty": max(0.0, float(args.transfer_learn_hazard_penalty)),
        "learn_success_bonus": max(0.0, float(args.transfer_learn_success_bonus)),
        "diag_temperature": 0.75,
    }
    transfer_cfg.update(_parse_cfg_overrides(args.transfer_cfg))
    if bool(args.dynamic_transfer_mix):
        mix_decay_steps = int(args.transfer_mix_decay_steps)
        if mix_decay_steps <= 0:
            mix_decay_steps = _auto_transfer_mix_decay_steps(
                episodes=max(1, int(args.episodes)),
                max_steps=max(1, int(args.max_steps)),
                transfer_rows=max(0, int(ds.get("transfer_dataset_rows", 0))),
                mix_start=max(0.0, min(1.0, float(args.transfer_mix_start))),
                mix_end=max(0.0, min(1.0, float(args.transfer_mix_end))),
            )
        transfer_cfg.update(
            {
                "dynamic_transfer_mix_enabled": True,
                "transfer_mix_start": max(0.0, min(1.0, float(args.transfer_mix_start))),
                "transfer_mix_end": max(0.0, min(1.0, float(args.transfer_mix_end))),
                "transfer_mix_decay_steps": int(mix_decay_steps),
                "transfer_mix_min_rows": max(1, int(args.transfer_mix_min_rows)),
                "transfer_replay_reward_scale": max(0.0, float(args.transfer_replay_reward_scale)),
            }
        )
    if empty_transfer_dataset and str(args.empty_transfer_dataset_policy).strip().lower() == "continue":
        # Continue mode turns the transfer run into a no-warmstart control.
        transfer_cfg["warmstart_reward_scale"] = 0.0
        transfer_cfg["dynamic_transfer_mix_enabled"] = False
        transfer_cfg["transfer_mix_start"] = 0.0
        transfer_cfg["transfer_mix_end"] = 0.0
        transfer_cfg["transfer_mix_decay_steps"] = 1
        transfer_cfg["transfer_mix_min_rows"] = max(1, int(args.transfer_mix_min_rows))
        transfer_cfg["transfer_replay_reward_scale"] = 0.0
    if transfer_safe_cfg:
        transfer_cfg["safe_executor"] = transfer_safe_cfg

    trainer = Trainer(run_root=str(args.runs_root), schema_version="v1", auto_register_builtin=True)
    transfer_run = _run_agent(
        trainer=trainer,
        role="transfer",
        verse_name=str(args.target_verse),
        episodes=max(1, int(args.episodes)),
        max_steps=max(1, int(args.max_steps)),
        seed=int(args.seed),
        algo=str(args.transfer_algo),
        policy_id=f"transfer_{args.transfer_algo}_{args.target_verse}",
        cfg=transfer_cfg,
    )
    baseline_run = _run_agent(
        trainer=trainer,
        role="baseline",
        verse_name=str(args.target_verse),
        episodes=max(1, int(args.episodes)),
        max_steps=max(1, int(args.max_steps)),
        seed=int(args.seed),
        algo=str(args.baseline_algo),
        policy_id=f"baseline_{args.baseline_algo}_{args.target_verse}",
        cfg=baseline_cfg,
    )

    transfer_run_dir = os.path.join(str(args.runs_root), transfer_run)
    baseline_run_dir = os.path.join(str(args.runs_root), baseline_run)

    transfer_stats = evaluate_run(transfer_run_dir)
    baseline_stats = evaluate_run(baseline_run_dir)
    transfer_curve = _episode_curve(transfer_run_dir)
    baseline_curve = _episode_curve(baseline_run_dir)
    early_episodes = max(1, int(args.diagnostic_early_episodes))
    action_first_k = max(1, int(args.diagnostic_action_agreement_first_k))

    transfer_first = _first_passable_episode(
        transfer_curve,
        window=max(1, int(args.passable_window)),
        passable_success_rate=float(args.passable_success_rate),
        passable_mean_return=float(args.passable_mean_return),
    )
    baseline_first = _first_passable_episode(
        baseline_curve,
        window=max(1, int(args.passable_window)),
        passable_success_rate=float(args.passable_success_rate),
        passable_mean_return=float(args.passable_mean_return),
    )

    agg_transfer = _aggregate_curve(transfer_curve)
    agg_baseline = _aggregate_curve(baseline_curve)
    transfer_safety_trend = _safety_trend(transfer_curve)
    baseline_safety_trend = _safety_trend(baseline_curve)
    transfer_early = _early_window(transfer_curve, episodes=early_episodes)
    baseline_early = _early_window(baseline_curve, episodes=early_episodes)
    transfer_action_agreement = _action_agreement_diagnostics(
        transfer_run_dir, first_k_steps=action_first_k
    )
    baseline_action_agreement = _action_agreement_diagnostics(
        baseline_run_dir, first_k_steps=action_first_k
    )
    transfer_td_diag = _train_td_diagnostics(transfer_run_dir, early_episodes=early_episodes)
    baseline_td_diag = _train_td_diagnostics(baseline_run_dir, early_episodes=early_episodes)
    transfer_score_diag = _transfer_score_diagnostics(str(ds["transfer_dataset_path"]))
    comparison = _speedup_summary(
        transfer_first_passable=transfer_first,
        baseline_first_passable=baseline_first,
        transfer_hazard_rate=float(agg_transfer["hazard_events_per_1k_steps"]),
        baseline_hazard_rate=float(agg_baseline["hazard_events_per_1k_steps"]),
    )

    overlap_map = _build_overlap_map(
        target_verse=str(args.target_verse),
        sources=sources,
        transfer_dataset_rows=int(ds.get("transfer_dataset_rows", 0)),
    )
    os.makedirs(os.path.dirname(str(args.overlap_out)) or ".", exist_ok=True)
    with open(str(args.overlap_out), "w", encoding="utf-8") as f:
        json.dump(overlap_map, f, ensure_ascii=False, indent=2)

    health_scorecard = _build_health_scorecard(
        run_dirs_by_role={
            "transfer": transfer_run_dir,
            "baseline": baseline_run_dir,
        },
        trace_root=str(args.health_trace_root),
        kl_critical=float(args.health_kl_critical),
        stale_kl_threshold=float(args.health_stale_kl_threshold),
        unsafe_veto_rate=float(args.health_unsafe_veto_rate),
        incoherent_match_threshold=float(args.health_incoherent_match_threshold),
        memory_coherence_threshold=float(args.health_memory_coherence_threshold),
        max_trace_rows_per_file=max(0, int(args.health_max_trace_rows_per_file)),
    )

    report = {
        "target_verse": str(args.target_verse),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "sources": [
            {
                "verse_name": s.verse_name,
                "path": s.path,
                "run_id": s.run_id,
                "source_kind": s.source_kind,
                "source_lane": s.source_lane,
                "source_universe": s.source_universe,
            }
            for s in sources
        ],
        "source_selection": _source_selection_summary(
            target_verse=str(args.target_verse),
            sources=sources,
        ),
        "transfer_dataset": ds,
        "transfer_dataset_diagnostics": {
            "score_distribution": transfer_score_diag,
            "lane_summary": dict(ds.get("lane_merge", {})) if isinstance(ds.get("lane_merge"), dict) else {},
        },
        "bridge_tuning": {
            "synthetic_reward_blend": float(max(0.0, min(1.0, float(args.bridge_synthetic_reward_blend)))),
            "synthetic_done_union": bool(args.bridge_synthetic_done_union),
            "behavioral_bridge_enabled": bool(args.bridge_behavioral_enabled),
            "behavioral_bridge_score_weight": float(max(0.0, min(1.0, float(args.bridge_behavioral_score_weight)))),
            "behavioral_max_prototype_rows": int(max(1, int(args.bridge_behavioral_max_prototype_rows))),
            "lane_controls": {
                "near_lane_enabled": bool(args.near_lane_enabled),
                "far_lane_enabled": bool(args.far_lane_enabled),
                "max_rows_per_source": int(max(0, int(args.max_rows_per_source))),
                "near_lane_max_rows_per_source": int(max(0, int(args.near_lane_max_rows_per_source))),
                "far_lane_max_rows_per_source": int(max(0, int(args.far_lane_max_rows_per_source))),
            },
            "universe_adapter_enabled": bool(args.universe_adapter_enabled),
            "far_lane_weighting": {
                "enabled": bool(args.far_lane_score_weight_enabled),
                "strength": float(max(0.0, min(1.0, float(args.far_lane_score_weight_strength)))),
                "min_universe_feature_score": float(
                    max(0.0, min(1.0, float(args.far_lane_min_universe_feature_score)))
                ),
            },
            "label_cfg": dict(bridge_label_cfg),
        },
        "transfer_agent": {
            "algo": str(args.transfer_algo),
            "warehouse_obs_key_mode": str(args.transfer_q_warehouse_obs_key_mode),
            "config_overrides": _parse_cfg_overrides(args.transfer_cfg),
            "warmstart": {
                "reward_scale": float(args.transfer_warmstart_reward_scale),
                "use_transfer_score": bool(args.transfer_warmstart_use_transfer_score),
                "transfer_score_min": float(args.transfer_warmstart_transfer_score_min),
                "transfer_score_max": float(args.transfer_warmstart_transfer_score_max),
            },
            "dynamic_mixing": {
                "enabled": bool(args.dynamic_transfer_mix),
                "mix_start": float(max(0.0, min(1.0, float(args.transfer_mix_start)))),
                "mix_end": float(max(0.0, min(1.0, float(args.transfer_mix_end)))),
                "mix_decay_steps": int(transfer_cfg.get("transfer_mix_decay_steps", 0)),
                "mix_min_rows": int(transfer_cfg.get("transfer_mix_min_rows", 0)),
                "replay_reward_scale": float(transfer_cfg.get("transfer_replay_reward_scale", 0.0)),
                "auto_schedule_used": bool(int(args.transfer_mix_decay_steps) <= 0),
            },
            "adaptive_veto_schedule": {
                "enabled": bool((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_schedule_enabled", False)),
                "relax_start": float((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_relaxation_start", 0.0)),
                "relax_end": float((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_relaxation_end", 0.0)),
                "schedule_steps": int((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_schedule_steps", 0)),
                "schedule_power": float((transfer_cfg.get("safe_executor", {}) or {}).get("adaptive_veto_schedule_power", 1.0)),
                "auto_schedule_used": bool(int(args.safe_adaptive_veto_schedule_steps) <= 0),
                "auto_tune_enabled": bool(args.safe_adaptive_veto_auto_tune),
                "auto_tune": dict(safe_tune_info),
            },
            "run_id": transfer_run,
            "run_dir": transfer_run_dir.replace("\\", "/"),
            "eval": {
                "episodes": int(transfer_stats.episodes),
                "mean_return": float(transfer_stats.mean_return),
                "success_rate": transfer_stats.success_rate,
                "mean_steps": float(transfer_stats.mean_steps),
            },
            "diagnostics": {
                "early_window": transfer_early,
                "action_agreement": transfer_action_agreement,
                "td_error": transfer_td_diag,
            },
            "curve_aggregate": agg_transfer,
            "safety_trend": transfer_safety_trend,
            "first_passable_episode": transfer_first,
        },
        "baseline_agent": {
            "algo": str(args.baseline_algo),
            "warehouse_obs_key_mode": str(args.baseline_q_warehouse_obs_key_mode),
            "config_overrides": _parse_cfg_overrides(args.baseline_cfg),
            "epsilon_schedule": {
                "epsilon_start": float(args.baseline_epsilon_start),
                "epsilon_min": float(args.baseline_epsilon_min),
                "epsilon_decay": float(args.baseline_epsilon_decay),
            },
            "run_id": baseline_run,
            "run_dir": baseline_run_dir.replace("\\", "/"),
            "eval": {
                "episodes": int(baseline_stats.episodes),
                "mean_return": float(baseline_stats.mean_return),
                "success_rate": baseline_stats.success_rate,
                "mean_steps": float(baseline_stats.mean_steps),
            },
            "diagnostics": {
                "early_window": baseline_early,
                "action_agreement": baseline_action_agreement,
                "td_error": baseline_td_diag,
            },
            "curve_aggregate": agg_baseline,
            "safety_trend": baseline_safety_trend,
            "first_passable_episode": baseline_first,
        },
        "comparison": comparison,
        "health_scorecard": health_scorecard,
        "production_summary": {
            "transfer_gain": {
                "speedup_ratio": comparison.get("transfer_speedup_ratio"),
                "hazard_improvement_pct": comparison.get("hazard_improvement_pct"),
                "transfer_wins_convergence": comparison.get("transfer_wins_convergence"),
            },
            "safety": {
                "transfer_mcts_veto_rate": float(agg_transfer.get("mcts_veto_rate", 0.0)),
                "baseline_mcts_veto_rate": float(agg_baseline.get("mcts_veto_rate", 0.0)),
                "transfer_hazard_per_1k": float(agg_transfer.get("hazard_events_per_1k_steps", 0.0)),
                "baseline_hazard_per_1k": float(agg_baseline.get("hazard_events_per_1k_steps", 0.0)),
                "transfer_veto_rate_improved_over_time": bool(transfer_safety_trend.get("veto_rate_improved", False)),
                "transfer_hazard_rate_improved_over_time": bool(transfer_safety_trend.get("hazard_rate_improved", False)),
            },
            "health": {
                "transfer": (health_scorecard.get("by_role", {}) or {}).get("transfer"),
                "baseline": (health_scorecard.get("by_role", {}) or {}).get("baseline"),
            },
        },
        "overlap_map_path": str(args.overlap_out),
    }

    os.makedirs(os.path.dirname(str(args.report_out)) or ".", exist_ok=True)
    with open(str(args.report_out), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"transfer_run={transfer_run}")
    print(f"baseline_run={baseline_run}")
    print(
        f"transfer mean_return={float(transfer_stats.mean_return):.3f} "
        f"success={float(transfer_stats.success_rate or 0.0):.3f} "
        f"hazard/1k={float(agg_transfer['hazard_events_per_1k_steps']):.2f}"
    )
    print(
        f"baseline mean_return={float(baseline_stats.mean_return):.3f} "
        f"success={float(baseline_stats.success_rate or 0.0):.3f} "
        f"hazard/1k={float(agg_baseline['hazard_events_per_1k_steps']):.2f}"
    )
    print(
        f"speedup_ratio={report['comparison']['transfer_speedup_ratio']} "
        f"hazard_gain_pct={float(report['comparison']['hazard_improvement_pct']):.2f}"
    )
    hs = (report.get("health_scorecard", {}).get("by_role", {}) if isinstance(report.get("health_scorecard"), dict) else {})
    transfer_health = hs.get("transfer") if isinstance(hs, dict) else None
    if isinstance(transfer_health, dict):
        print(
            f"transfer_health_score={float(_safe_float(transfer_health.get('total_score', 0.0), 0.0)):.1f} "
            f"status={str(transfer_health.get('status', 'n/a'))}"
        )
    _print_chart(
        transfer_curve=transfer_curve,
        baseline_curve=baseline_curve,
        stride=max(1, int(args.chart_stride)),
    )
    print(f"report: {args.report_out}")
    print(f"overlap_map: {args.overlap_out}")


if __name__ == "__main__":
    main()
