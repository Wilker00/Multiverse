from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from core.universe_registry import primary_universe_for_verse
from memory.universe_adapters import transfer_row_universe_metadata
from memory.semantic_bridge import translate_dna
from orchestrator.transfer_sources import TransferSourceDNA, iter_jsonl, list_run_dirs, peek_first_event
from tools.run_transfer_challenge_eval_support import (
    _aggregate_curve,
    _episode_curve,
    _quantiles,
    _safe_float,
    _safe_int,
    _safety_trend,
)


_SourceDNA = TransferSourceDNA


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
            "mean": (
                float(sum(universe_feature_scores) / float(len(universe_feature_scores)))
                if universe_feature_scores
                else None
            ),
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
                cap = (
                    int(near_lane_max_rows_per_source)
                    if int(near_lane_max_rows_per_source) > 0
                    else int(max_rows_per_source)
                )
            elif lane == "far_universe":
                if int(far_lane_max_rows_per_source) > 0:
                    cap = int(far_lane_max_rows_per_source)
                else:
                    cap = _default_far_lane_cap(base_cap=int(max_rows_per_source))
            else:
                cap = int(max_rows_per_source)

            taken = 0
            for row in iter_jsonl(p):
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
        return 7
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
        patrol_dist = max(-1, min(25, _safe_int(obs.get("patrol_dist", 4), 4)))
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
        return {
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
    if v == "escape_world":
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
    min_transfer_confidence: float = 0.0,
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
    dropped_low_confidence = 0
    hz_ratio = max(0.0, min(1.0, float(hazard_keep_ratio)))
    min_conf = max(0.0, min(1.0, float(min_transfer_confidence)))
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
            conf = row.get("transfer_confidence")
            if conf is not None:
                conf_f = _safe_float(conf, -1.0)
                if conf_f < float(min_conf):
                    dropped_low_confidence += 1
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
        "dropped_low_confidence": int(dropped_low_confidence),
        "hazard_keep_ratio": float(hz_ratio),
        "min_transfer_confidence": float(min_conf),
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
    for run_dir in list_run_dirs(str(runs_root)):
        events_path = os.path.join(run_dir, "events.jsonl")
        first = peek_first_event(events_path)
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
    from memory.semantic_bridge import bridge_reason

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
