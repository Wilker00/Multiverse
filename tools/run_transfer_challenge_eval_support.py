from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from orchestrator.evaluator import RunStats, evaluate_run
from orchestrator.transfer_sources import iter_jsonl


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


def _episode_curve(run_dir: str) -> List[Dict[str, Any]]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return []

    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for ev in iter_jsonl(events_path):
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
        mean_succ = sum(1 for r in seg if bool(r.get("success", False))) / float(len(seg))
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
    for row in iter_jsonl(dataset_path):
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
    for ev in iter_jsonl(events_path):
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
    rows = [r for r in iter_jsonl(metrics_path) if isinstance(r, dict)]
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


def _collect_run_eval(
    run_dir: str,
    *,
    early_episodes: int,
    action_first_k: int,
) -> Dict[str, Any]:
    if (not os.path.isdir(run_dir)) or (not os.path.isfile(os.path.join(run_dir, "events.jsonl"))):
        stats = RunStats(
            run_id=os.path.basename(os.path.normpath(run_dir)),
            episodes=0,
            total_steps=0,
            mean_return=0.0,
            mean_steps=0.0,
            success_rate=None,
            episode_stats=[],
        )
        curve: List[Dict[str, Any]] = []
    else:
        stats = evaluate_run(run_dir)
        curve = _episode_curve(run_dir)
    return {
        "stats": stats,
        "curve": curve,
        "aggregate": _aggregate_curve(curve),
        "safety_trend": _safety_trend(curve),
        "early_window": _early_window(curve, episodes=early_episodes),
        "action_agreement": _action_agreement_diagnostics(run_dir, first_k_steps=action_first_k),
        "td_error": _train_td_diagnostics(run_dir, early_episodes=early_episodes),
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

    for ev in iter_jsonl(events_path):
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
    proxy_kl = None if disagree_rate is None else float(0.25 * max(0.0, min(1.0, disagree_rate)))
    return _RunTraceProxy(
        verse_name=str(verse_name),
        rows=int(rows),
        mean_kl=proxy_kl,
        prior_top1_match=(None if match_n <= 0 else float(match_sum / float(match_n))),
        high_quality_rate=(None if hq_n <= 0 else float(hq_sum / float(hq_n))),
    )
