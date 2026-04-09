"""
tools/run_transfer_challenge.py

Cross-verse transfer challenge:
- Transfer agent: strategy DNA warm-start + optional SafeExecutor/MCTS
- Baseline agent: naive training from scratch

The target verse defaults to warehouse_world.
"""

from __future__ import annotations

import json
import math
import os
import sys
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
from memory.semantic_bridge import bridge_reason, infer_verse_from_obs
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer
from orchestrator.transfer_sources import (
    TransferSourceDNA,
    discover_transfer_sources,
    extract_success_dna_from_events,
    iter_jsonl,
    list_run_dirs,
    merge_jsonl,
    peek_first_event,
)
from tools.run_transfer_challenge_cli_support import (
    build_transfer_bridge_label_cfg,
    build_transfer_challenge_arg_parser,
)
from tools.run_transfer_challenge_adt_support import (
    apply_adt_prior_rollback,
    prepare_adt_prior,
)
from tools.run_transfer_challenge_dataset_support import (
    _augment_translated_file_with_lane_metadata,
    _auto_safe_veto_schedule_steps,
    _auto_transfer_mix_decay_steps,
    _auto_tune_safe_veto_schedule,
    _build_transfer_dataset,
    _default_far_lane_cap,
    _filter_transfer_dataset,
    _normalize_target_obs,
    _recent_hazard_trend_for_target,
    _target_action_count,
)
from tools.run_transfer_challenge_eval_support import (
    _RunTraceProxy,
    _action_agreement_diagnostics,
    _aggregate_curve,
    _collect_run_eval,
    _collect_run_trace_proxy,
    _early_window,
    _episode_curve,
    _first_passable_episode,
    _quantiles,
    _safety_trend,
    _speedup_summary,
    _train_td_diagnostics,
    _transfer_score_diagnostics,
)
from tools.run_transfer_challenge_orchestration_support import (
    run_robust_selector,
    run_source_attribution,
)
from tools.run_transfer_challenge_reporting_support import (
    build_transfer_challenge_report,
    enrich_report_agent_details,
    print_transfer_challenge_summary,
    write_json_artifact,
)


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


_iter_jsonl = iter_jsonl
_peek_first_event = peek_first_event
_list_run_dirs = list_run_dirs
_extract_success_dna_from_events = extract_success_dna_from_events
_SourceDNA = TransferSourceDNA
_discover_transfer_sources = discover_transfer_sources
_merge_jsonl = merge_jsonl


def _extract_top_return_dna_from_events(
    *,
    run_dir: str,
    out_path: str,
    max_rows: int = 5000,
    top_return_pct: float = 0.25,
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
    if not order:
        return 0

    episode_scores: List[Tuple[float, int, str]] = []
    for idx, ep in enumerate(order):
        rows = by_ep.get(ep, [])
        ret = float(sum(_safe_float(r.get("reward", 0.0), 0.0) for r in rows if isinstance(r, dict)))
        success = 1 if any(bool((r.get("info") or {}).get("reached_goal", False)) for r in rows if isinstance(r, dict)) else 0
        # Prefer higher-return episodes; break ties toward successful episodes and earlier order.
        episode_scores.append((float(ret), int(success), str(ep)))
    episode_scores.sort(key=lambda x: (float(x[0]), int(x[1])), reverse=True)
    keep_pct = max(0.0, min(1.0, float(top_return_pct)))
    keep_eps = max(1, int(math.ceil(float(len(episode_scores)) * keep_pct)))
    selected = set(str(ep) for _, _, ep in episode_scores[:keep_eps])
    if not selected:
        return 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ep in order:
            if ep not in selected:
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


def _disable_transfer_warmstart(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["warmstart_reward_scale"] = 0.0
    out["dynamic_transfer_mix_enabled"] = False
    out["transfer_mix_start"] = 0.0
    out["transfer_mix_end"] = 0.0
    out["transfer_mix_decay_steps"] = 1
    out["transfer_replay_reward_scale"] = 0.0
    return out


def _adt_prior_rollback_decision(
    *,
    candidate_success_rate: float,
    candidate_mean_return: float,
    candidate_hazard_per_1k: float,
    baseline_success_rate: float,
    baseline_mean_return: float,
    baseline_hazard_per_1k: float,
    min_success_delta: float,
    min_return_delta: float,
    max_hazard_regression_per_1k: float,
    success_weight: float = 100.0,
    return_weight: float = 1.0,
    hazard_weight: float = 0.02,
) -> Dict[str, Any]:
    comp = _transfer_mode_utility(
        candidate_success_rate=float(candidate_success_rate),
        candidate_mean_return=float(candidate_mean_return),
        candidate_hazard_per_1k=float(candidate_hazard_per_1k),
        scratch_success_rate=float(baseline_success_rate),
        scratch_mean_return=float(baseline_mean_return),
        scratch_hazard_per_1k=float(baseline_hazard_per_1k),
        success_weight=float(success_weight),
        return_weight=float(return_weight),
        hazard_weight=float(hazard_weight),
    )
    hazard_regression_ok = bool(
        float(comp.get("hazard_gain_per_1k", 0.0)) >= -float(max_hazard_regression_per_1k)
    )
    keep_prior = bool(
        float(comp.get("success_delta", 0.0)) >= float(min_success_delta)
        and float(comp.get("return_delta", 0.0)) >= float(min_return_delta)
        and hazard_regression_ok
    )
    return {
        "keep_prior": bool(keep_prior),
        "rollback": bool(not keep_prior),
        "hazard_regression_ok": bool(hazard_regression_ok),
        "thresholds": {
            "min_success_delta": float(min_success_delta),
            "min_return_delta": float(min_return_delta),
            "max_hazard_regression_per_1k": float(max_hazard_regression_per_1k),
        },
        "comparison": comp,
    }


def _align_with_baseline_scratch_schedule(
    cfg: Dict[str, Any], baseline_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    out = dict(cfg)
    if isinstance(baseline_cfg, dict):
        for key in ("epsilon_start", "epsilon_min", "epsilon_decay", "warehouse_obs_key_mode"):
            if key in baseline_cfg:
                out[key] = baseline_cfg.get(key)
    return out


def _transfer_mode_utility(
    *,
    candidate_success_rate: float,
    candidate_mean_return: float,
    candidate_hazard_per_1k: float,
    scratch_success_rate: float,
    scratch_mean_return: float,
    scratch_hazard_per_1k: float,
    success_weight: float,
    return_weight: float,
    hazard_weight: float,
) -> Dict[str, float]:
    ds = float(candidate_success_rate - scratch_success_rate)
    dr = float(candidate_mean_return - scratch_mean_return)
    hg = float(scratch_hazard_per_1k - candidate_hazard_per_1k)
    utility = (
        float(success_weight) * float(ds)
        + float(return_weight) * float(dr)
        + float(hazard_weight) * float(hg)
    )
    return {
        "success_delta": float(ds),
        "return_delta": float(dr),
        "hazard_gain_per_1k": float(hg),
        "utility": float(utility),
    }


def _pick_robust_transfer_mode(
    *,
    pilot_rows: List[Dict[str, Any]],
    min_utility: float,
    min_success_delta: float,
    min_hazard_gain_per_1k: float,
    max_hazard_regression_per_1k: float,
    success_weight: float,
    return_weight: float,
    hazard_weight: float,
    scratch_mode: str = "scratch_control",
) -> Dict[str, Any]:
    rows = [dict(r) for r in pilot_rows if isinstance(r, dict)]
    scratch = next((r for r in rows if str(r.get("mode")) == str(scratch_mode)), None)
    if scratch is None:
        return {
            "selected_mode": str(scratch_mode),
            "reason": "missing_scratch_control",
            "scored": [],
        }
    s_sr = _safe_float(scratch.get("success_rate", 0.0), 0.0)
    s_ret = _safe_float(scratch.get("mean_return", 0.0), 0.0)
    s_haz = _safe_float(scratch.get("hazard_per_1k", 0.0), 0.0)

    scored: List[Dict[str, Any]] = []
    for row in rows:
        mode = str(row.get("mode", "")).strip()
        if not mode or mode == str(scratch_mode):
            continue
        comp = _transfer_mode_utility(
            candidate_success_rate=_safe_float(row.get("success_rate", 0.0), 0.0),
            candidate_mean_return=_safe_float(row.get("mean_return", 0.0), 0.0),
            candidate_hazard_per_1k=_safe_float(row.get("hazard_per_1k", 0.0), 0.0),
            scratch_success_rate=float(s_sr),
            scratch_mean_return=float(s_ret),
            scratch_hazard_per_1k=float(s_haz),
            success_weight=float(success_weight),
            return_weight=float(return_weight),
            hazard_weight=float(hazard_weight),
        )
        hazard_regression_ok = bool(
            float(comp["hazard_gain_per_1k"]) >= -float(max_hazard_regression_per_1k)
        )
        gate_ok = bool(
            float(comp["utility"]) >= float(min_utility)
            and float(comp["success_delta"]) >= float(min_success_delta)
            and float(comp["hazard_gain_per_1k"]) >= float(min_hazard_gain_per_1k)
            and hazard_regression_ok
        )
        merged = dict(row)
        merged.update(
            {
                "utility": float(comp["utility"]),
                "success_delta_vs_scratch": float(comp["success_delta"]),
                "return_delta_vs_scratch": float(comp["return_delta"]),
                "hazard_gain_vs_scratch_per_1k": float(comp["hazard_gain_per_1k"]),
                "gate_ok": bool(gate_ok),
                "hazard_regression_ok": bool(hazard_regression_ok),
            }
        )
        scored.append(merged)

    if not scored:
        return {
            "selected_mode": str(scratch_mode),
            "reason": "no_transfer_candidates",
            "scored": [],
        }

    scored.sort(
        key=lambda r: (
            1 if bool(r.get("gate_ok", False)) else 0,
            float(_safe_float(r.get("utility", 0.0), 0.0)),
            float(_safe_float(r.get("success_delta_vs_scratch", 0.0), 0.0)),
            float(_safe_float(r.get("hazard_gain_vs_scratch_per_1k", 0.0), 0.0)),
            float(_safe_float(r.get("return_delta_vs_scratch", 0.0), 0.0)),
        ),
        reverse=True,
    )
    best = scored[0]
    if bool(best.get("gate_ok", False)):
        return {
            "selected_mode": str(best.get("mode", scratch_mode)),
            "reason": "best_gated_utility",
            "selected": best,
            "scored": scored,
            "scratch_control": {
                "mode": str(scratch_mode),
                "success_rate": float(s_sr),
                "mean_return": float(s_ret),
                "hazard_per_1k": float(s_haz),
            },
        }
    return {
        "selected_mode": str(scratch_mode),
        "reason": "no_candidate_passed_gate",
        "selected": best,
        "scored": scored,
        "scratch_control": {
            "mode": str(scratch_mode),
            "success_rate": float(s_sr),
            "mean_return": float(s_ret),
            "hazard_per_1k": float(s_haz),
        },
    }


def _source_id(src: _SourceDNA) -> str:
    return f"{str(src.verse_name)}|{str(src.run_id)}|{str(src.source_kind)}|{str(src.path)}"


def _pick_sources_from_attribution(
    *,
    all_sources: List[_SourceDNA],
    attribution_rows: List[Dict[str, Any]],
    min_keep_sources: int,
    keep_unscored: bool,
) -> Dict[str, Any]:
    row_by_source: Dict[str, Dict[str, Any]] = {}
    for row in attribution_rows:
        sid = str(row.get("source_id", "")).strip()
        if sid:
            row_by_source[sid] = dict(row)

    scored_ids = set(row_by_source.keys())
    keep_ids: List[str] = []
    for sid, row in row_by_source.items():
        if bool(row.get("gate_ok", False)):
            keep_ids.append(sid)

    ranked = sorted(
        [dict(r) for r in attribution_rows if str(r.get("source_id", "")).strip()],
        key=lambda r: (
            1 if bool(r.get("gate_ok", False)) else 0,
            float(_safe_float(r.get("utility", 0.0), 0.0)),
            float(_safe_float(r.get("success_delta_vs_scratch", 0.0), 0.0)),
            float(_safe_float(r.get("hazard_gain_vs_scratch_per_1k", 0.0), 0.0)),
            float(_safe_float(r.get("return_delta_vs_scratch", 0.0), 0.0)),
        ),
        reverse=True,
    )
    keep_set = set(keep_ids)
    for row in ranked:
        if len(keep_set) >= max(0, int(min_keep_sources)):
            break
        sid = str(row.get("source_id", "")).strip()
        if sid and sid not in keep_set:
            keep_set.add(sid)

    kept_sources: List[_SourceDNA] = []
    dropped_sources: List[_SourceDNA] = []
    for src in all_sources:
        sid = _source_id(src)
        if sid in keep_set:
            kept_sources.append(src)
            continue
        if bool(keep_unscored) and sid not in scored_ids:
            kept_sources.append(src)
            continue
        dropped_sources.append(src)

    if not kept_sources:
        kept_sources = list(all_sources)
        dropped_sources = []
        reason = "fallback_keep_all_no_kept_sources"
    elif len(kept_sources) == len(all_sources):
        reason = "no_pruning_effective"
    else:
        reason = "pruned_by_source_attribution"

    return {
        "kept_sources": kept_sources,
        "dropped_sources": dropped_sources,
        "kept_source_ids": [str(_source_id(s)) for s in kept_sources],
        "dropped_source_ids": [str(_source_id(s)) for s in dropped_sources],
        "reason": str(reason),
    }


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
    args = build_transfer_challenge_arg_parser().parse_args()

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

    bridge_label_cfg = build_transfer_bridge_label_cfg(args)

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
            min_transfer_confidence=max(0.0, min(1.0, float(args.transfer_filter_min_confidence))),
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
            "fallback_horizon_steps": max(0, int(args.safe_fallback_horizon_steps)),
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
        "warmstart_max_rows": max(0, int(args.transfer_warmstart_max_rows)),
        "warmstart_balance_actions": bool(args.transfer_warmstart_balance_actions),
        "warmstart_action_balance_max_share": max(
            0.0, min(1.0, float(args.transfer_warmstart_action_balance_max_share))
        ),
        "warmstart_use_transfer_score": bool(args.transfer_warmstart_use_transfer_score),
        "warmstart_target": str(args.transfer_warmstart_target),
        "warmstart_target_gamma": max(0.0, min(1.0, float(args.transfer_warmstart_target_gamma))),
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
        transfer_cfg = _align_with_baseline_scratch_schedule(transfer_cfg, baseline_cfg)
    if transfer_safe_cfg:
        transfer_cfg["safe_executor"] = transfer_safe_cfg

    source_attr_result = run_source_attribution(
        args=args,
        sources=list(sources),
        transfer_dataset=dict(ds),
        transfer_cfg=dict(transfer_cfg),
        baseline_cfg=dict(baseline_cfg),
        bridge_label_cfg=dict(bridge_label_cfg),
        run_agent=_run_agent,
        source_id=_source_id,
        pick_sources_from_attribution=_pick_sources_from_attribution,
        transfer_mode_utility=_transfer_mode_utility,
    )
    source_attribution_report = dict(source_attr_result["report"])
    sources = list(source_attr_result["sources"])
    ds = dict(source_attr_result["transfer_dataset"])
    transfer_cfg = dict(source_attr_result["transfer_cfg"])

    selected_transfer_algo = str(args.transfer_algo)
    robust_selector_result = run_robust_selector(
        args=args,
        sources=list(sources),
        transfer_dataset=dict(ds),
        transfer_cfg=dict(transfer_cfg),
        baseline_cfg=dict(baseline_cfg),
        bridge_label_cfg=dict(bridge_label_cfg),
        selected_transfer_algo=str(selected_transfer_algo),
        empty_transfer_dataset=bool(empty_transfer_dataset),
        run_agent=_run_agent,
        pick_robust_transfer_mode=_pick_robust_transfer_mode,
    )
    robust_selector_report = dict(robust_selector_result["report"])
    ds = dict(robust_selector_result["transfer_dataset"])
    transfer_cfg = dict(robust_selector_result["transfer_cfg"])
    selected_transfer_algo = str(robust_selector_result["selected_transfer_algo"])

    base_transfer_cfg = dict(transfer_cfg)
    base_selected_transfer_algo = str(selected_transfer_algo)
    adt_prior_result = prepare_adt_prior(
        args=args,
        transfer_dataset=dict(ds),
        transfer_cfg=dict(transfer_cfg),
        selected_transfer_algo=str(base_selected_transfer_algo),
        run_agent=_run_agent,
        parse_cfg_overrides=_parse_cfg_overrides,
        collect_run_eval=_collect_run_eval,
        extract_success_dna_from_events=_extract_success_dna_from_events,
        extract_top_return_dna_from_events=_extract_top_return_dna_from_events,
        merge_jsonl=_merge_jsonl,
        auto_transfer_mix_decay_steps=_auto_transfer_mix_decay_steps,
    )
    adt_prior_report = dict(adt_prior_result["report"])
    ds = dict(adt_prior_result["transfer_dataset"])
    transfer_cfg = dict(adt_prior_result["transfer_cfg"])

    trainer = Trainer(run_root=str(args.runs_root), schema_version="v1", auto_register_builtin=True)
    transfer_run = _run_agent(
        trainer=trainer,
        role="transfer",
        verse_name=str(args.target_verse),
        episodes=max(1, int(args.episodes)),
        max_steps=max(1, int(args.max_steps)),
        seed=int(args.seed),
        algo=str(selected_transfer_algo),
        policy_id=f"transfer_{selected_transfer_algo}_{args.target_verse}",
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

    early_episodes = max(1, int(args.diagnostic_early_episodes))
    action_first_k = max(1, int(args.diagnostic_action_agreement_first_k))
    baseline_eval = _collect_run_eval(
        baseline_run_dir,
        early_episodes=early_episodes,
        action_first_k=action_first_k,
    )
    transfer_eval = _collect_run_eval(
        transfer_run_dir,
        early_episodes=early_episodes,
        action_first_k=action_first_k,
    )

    adt_rollback_result = apply_adt_prior_rollback(
        args=args,
        adt_prior_report=dict(adt_prior_report),
        transfer_eval=dict(transfer_eval),
        baseline_eval=dict(baseline_eval),
        trainer=trainer,
        transfer_run=str(transfer_run),
        transfer_run_dir=str(transfer_run_dir),
        transfer_dataset=dict(ds),
        transfer_cfg=dict(transfer_cfg),
        base_transfer_cfg=dict(base_transfer_cfg),
        base_selected_transfer_algo=str(base_selected_transfer_algo),
        run_agent=_run_agent,
        adt_prior_rollback_decision=_adt_prior_rollback_decision,
        collect_run_eval=_collect_run_eval,
        safe_float=_safe_float,
        safe_int=_safe_int,
    )
    adt_prior_report = dict(adt_rollback_result["report"])
    transfer_run = str(adt_rollback_result["transfer_run"])
    transfer_run_dir = str(adt_rollback_result["transfer_run_dir"])
    transfer_eval = dict(adt_rollback_result["transfer_eval"])
    ds = dict(adt_rollback_result["transfer_dataset"])
    transfer_cfg = dict(adt_rollback_result["transfer_cfg"])

    transfer_stats = transfer_eval["stats"]
    baseline_stats = baseline_eval["stats"]
    transfer_curve = transfer_eval["curve"]
    baseline_curve = baseline_eval["curve"]
    agg_transfer = transfer_eval["aggregate"]
    agg_baseline = baseline_eval["aggregate"]
    transfer_safety_trend = transfer_eval["safety_trend"]
    baseline_safety_trend = baseline_eval["safety_trend"]
    transfer_early = transfer_eval["early_window"]
    baseline_early = baseline_eval["early_window"]
    transfer_action_agreement = transfer_eval["action_agreement"]
    baseline_action_agreement = baseline_eval["action_agreement"]
    transfer_td_diag = transfer_eval["td_error"]
    baseline_td_diag = baseline_eval["td_error"]
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
    write_json_artifact(str(args.overlap_out), overlap_map)

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

    source_selection = _source_selection_summary(
        target_verse=str(args.target_verse),
        sources=sources,
    )
    report = build_transfer_challenge_report(
        args=args,
        sources=sources,
        source_selection=source_selection,
        transfer_dataset=ds,
        source_attribution_report=source_attribution_report,
        robust_selector_report=robust_selector_report,
        adt_prior_report=adt_prior_report,
        transfer_score_diag=transfer_score_diag,
        bridge_label_cfg=bridge_label_cfg,
        selected_transfer_algo=str(selected_transfer_algo),
        transfer_run=str(transfer_run),
        transfer_run_dir=str(transfer_run_dir),
        transfer_stats=transfer_stats,
        transfer_early=transfer_early,
        transfer_action_agreement=transfer_action_agreement,
        transfer_td_diag=transfer_td_diag,
        agg_transfer=agg_transfer,
        transfer_safety_trend=transfer_safety_trend,
        transfer_first=transfer_first,
        baseline_run=str(baseline_run),
        baseline_run_dir=str(baseline_run_dir),
        baseline_stats=baseline_stats,
        baseline_early=baseline_early,
        baseline_action_agreement=baseline_action_agreement,
        baseline_td_diag=baseline_td_diag,
        agg_baseline=agg_baseline,
        baseline_safety_trend=baseline_safety_trend,
        baseline_first=baseline_first,
        comparison=comparison,
        health_scorecard=health_scorecard,
    )
    report = enrich_report_agent_details(
        report=report,
        args=args,
        transfer_cfg=transfer_cfg,
        safe_tune_info=safe_tune_info,
        parse_cfg_overrides=_parse_cfg_overrides,
    )
    write_json_artifact(str(args.report_out), report)

    print_transfer_challenge_summary(
        report=report,
        transfer_run=str(transfer_run),
        baseline_run=str(baseline_run),
        transfer_stats=transfer_stats,
        baseline_stats=baseline_stats,
        agg_transfer=agg_transfer,
        agg_baseline=agg_baseline,
        safe_float=_safe_float,
        print_chart=_print_chart,
        transfer_curve=transfer_curve,
        baseline_curve=baseline_curve,
        chart_stride=max(1, int(args.chart_stride)),
    )


if __name__ == "__main__":
    main()
