"""
Support helpers for rollout memory recall, telemetry, and warnings.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from core.types import AgentRef, JSONValue, RunRef, VerseRef, now_ms


def _memory_pointer(*, run_id: str, episode_id: str, step_idx: int) -> str:
    rid = str(run_id).strip()
    ep = str(episode_id).strip()
    si = int(step_idx)
    return f"runs/{rid}/events.jsonl#episode_id={ep};step_idx={si}"


def _as_set(raw: Any) -> Optional[set[str]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        out = set(str(x).strip().lower() for x in raw if str(x).strip())
        return out if out else None
    s = str(raw).strip().lower()
    if not s:
        return None
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    return set(parts) if parts else None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _build_memory_bundle(*, req: Dict[str, Any], matches: List[Any], step_idx: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for m in matches:
        row = {
            "score": float(getattr(m, "score", 0.0)),
            "run_id": str(getattr(m, "run_id", "")),
            "episode_id": str(getattr(m, "episode_id", "")),
            "step_idx": int(getattr(m, "step_idx", 0)),
            "verse_name": str(getattr(m, "verse_name", "")),
            "action": getattr(m, "action", None),
            "source_greedy_action": getattr(m, "source_greedy_action", None),
            "source_action_matches_greedy": getattr(m, "source_action_matches_greedy", None),
            "reward": float(getattr(m, "reward", 0.0)),
            "pointer_path": _memory_pointer(
                run_id=str(getattr(m, "run_id", "")),
                episode_id=str(getattr(m, "episode_id", "")),
                step_idx=int(getattr(m, "step_idx", 0)),
            ),
        }
        if getattr(m, "trajectory", None) is not None:
            row["trajectory"] = list(m.trajectory)
        rows.append(row)
    return {
        "mode": "on_demand",
        "reason": str(req.get("reason", "agent_request")),
        "query_step_idx": int(step_idx),
        "query": {
            "top_k": int(req.get("top_k", 3)),
            "min_score": float(req.get("min_score", -1.0)),
            "verse_name": (None if req.get("verse_name") in (None, "") else str(req.get("verse_name"))),
            "memory_families": sorted(list(_as_set(req.get("memory_families")) or set())),
            "memory_types": sorted(list(_as_set(req.get("memory_types")) or set())),
        },
        "matches": rows,
        "match_count": int(len(rows)),
    }


def _obs_hash(obs: JSONValue) -> str:
    try:
        raw = json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except Exception:
        raw = str(obs).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def _record_runtime_warning(
    *,
    counters: Dict[str, int],
    warnings: List[Dict[str, Any]],
    code: str,
    component: str,
    step_idx: Optional[int],
    exc: Exception,
) -> None:
    key = str(code or "").strip() or "unknown_error"
    counters[key] = int(counters.get(key, 0)) + 1
    warnings.append(
        {
            "code": key,
            "component": str(component),
            "step_idx": (None if step_idx is None else int(step_idx)),
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
        }
    )


def _selector_routing_telemetry(
    *,
    action_info: Optional[Dict[str, Any]],
    obs: JSONValue,
    verse_name: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(action_info, dict):
        return None

    selector_active = bool(action_info.get("selector_active", False))
    experts_raw = action_info.get("experts")
    weights_raw = action_info.get("weights")
    if (not isinstance(experts_raw, list)) or (not isinstance(weights_raw, list)) or (not experts_raw) or (not weights_raw):
        selected = str(action_info.get("selected_expert", "")).strip()
        conf = _safe_float(action_info.get("selector_confidence", 0.0), 0.0)
        if not selected:
            return None
        return {
            "timestamp_ms": int(now_ms()),
            "verse": str(verse_name),
            "obs_hash": _obs_hash(obs),
            "selected_expert": selected,
            "confidence": float(conf),
            "selector_active": bool(selector_active),
            "top_experts": [{"expert": selected, "weight": float(conf)}],
        }

    pairs: List[Dict[str, Any]] = []
    for i in range(min(len(experts_raw), len(weights_raw))):
        expert = str(experts_raw[i]).strip()
        if not expert:
            continue
        try:
            weight = float(weights_raw[i])
        except Exception:
            continue
        pairs.append({"expert": expert, "weight": float(weight)})
    if not pairs:
        return None

    pairs.sort(key=lambda x: float(x.get("weight", 0.0)), reverse=True)
    selected = str(pairs[0]["expert"])
    conf = float(pairs[0]["weight"])
    if not selector_active:
        mode = str(action_info.get("mode", "")).strip().lower()
        selector_active = ("moe" in mode) or ("selector" in mode)

    return {
        "timestamp_ms": int(now_ms()),
        "verse": str(verse_name),
        "obs_hash": _obs_hash(obs),
        "selected_expert": selected,
        "confidence": float(conf),
        "selector_active": bool(selector_active),
        "top_experts": pairs[:5],
    }


def _universe_relation_safe(source_verse: str, target_verse: str) -> str:
    try:
        from core.taxonomy import universe_relation

        return str(universe_relation(source_verse, target_verse))
    except Exception:
        return "unknown"


def _memory_recall_transfer_decision_record(
    *,
    step_idx: int,
    episode_id: str,
    run: RunRef,
    agent_ref: AgentRef,
    verse_ref: VerseRef,
    action_info: Optional[Dict[str, Any]],
    hint: Optional[Dict[str, Any]],
    memory_query_state: Dict[str, Any],
    recall_ablation_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(action_info, dict):
        return None

    has_recall_keys = any(str(k).startswith("memory_recall_") for k in action_info.keys())
    raw_bundle = hint.get("memory_recall") if isinstance(hint, dict) else None
    bundle = raw_bundle if isinstance(raw_bundle, dict) else None
    if (not has_recall_keys) and (not isinstance(bundle, dict)):
        return None

    matches = bundle.get("matches") if isinstance(bundle, dict) else None
    top_match = matches[0] if isinstance(matches, list) and matches and isinstance(matches[0], dict) else {}
    if not isinstance(top_match, dict):
        top_match = {}

    source_verse = str(top_match.get("verse_name", "")).strip().lower()
    target_verse = str(verse_ref.verse_name).strip().lower()
    query_verse_name = ""
    if isinstance(bundle, dict):
        q = bundle.get("query")
        if isinstance(q, dict) and isinstance(q.get("verse_name"), str):
            query_verse_name = str(q.get("verse_name", "")).strip().lower()
    if not source_verse:
        source_verse = "unknown"

    eligible = bool(action_info.get("memory_recall_eligible", False))
    used = bool(action_info.get("memory_recall_used", False))
    gate_passed = bool(action_info.get("memory_recall_gate_passed", False))
    disabled_for_ablation = bool(action_info.get("memory_recall_disabled_for_ablation", False))
    greedy_changed = bool(action_info.get("memory_recall_greedy_changed", False))
    match_count = int(action_info.get("memory_recall_match_count", 0) or 0)

    if used:
        decision = "accept_transfer"
        decision_reason = "memory_recall_applied"
    elif disabled_for_ablation and eligible:
        decision = "fallback_scratch"
        decision_reason = "ablation_disabled_apply"
    elif eligible and (not gate_passed):
        decision = "fallback_scratch"
        decision_reason = "gate_rejected"
    elif eligible and gate_passed and (not used):
        decision = "fallback_scratch"
        decision_reason = "not_applied_after_gate"
    elif (match_count > 0) and (not eligible):
        decision = "fallback_scratch"
        decision_reason = "no_valid_recall_prior"
    elif bool(recall_ablation_state.get("eligible", False)):
        decision = "fallback_scratch"
        decision_reason = "recall_eligible_not_used"
    else:
        decision = "no_transfer_candidate"
        decision_reason = "no_recall_context"

    record: Dict[str, Any] = {
        "schema_version": "transfer_decision_record.v1",
        "transfer_mode": "memory_recall",
        "timestamp_ms": int(now_ms()),
        "run_id": str(run.run_id),
        "episode_id": str(episode_id),
        "step_idx": int(step_idx),
        "agent": {
            "agent_id": str(agent_ref.agent_id),
            "policy_id": str(agent_ref.policy_id),
            "policy_version": str(agent_ref.policy_version),
        },
        "target": {
            "verse_id": str(verse_ref.verse_id),
            "verse_name": str(verse_ref.verse_name),
            "verse_version": str(verse_ref.verse_version),
        },
        "source": {
            "verse_name": str(source_verse),
            "run_id": str(top_match.get("run_id", "")),
            "episode_id": str(top_match.get("episode_id", "")),
            "step_idx": (None if top_match.get("step_idx") is None else int(top_match.get("step_idx", 0))),
            "pointer_path": str(top_match.get("pointer_path", "")),
            "score": _safe_float(top_match.get("score", 0.0), 0.0),
            "action": top_match.get("action"),
            "source_greedy_action": top_match.get("source_greedy_action"),
            "source_action_matches_greedy": top_match.get("source_action_matches_greedy"),
        },
        "universe_relation": _universe_relation_safe(source_verse=source_verse, target_verse=target_verse),
        "decision": str(decision),
        "decision_reason": str(decision_reason),
        "gate_inputs": {
            "epsilon": _safe_float(action_info.get("epsilon", 0.0), 0.0),
            "memory_recall_consensus_margin": _safe_float(
                action_info.get("memory_recall_consensus_margin", 0.0), 0.0
            ),
            "memory_recall_q_margin_base": _safe_float(action_info.get("memory_recall_q_margin_base", 0.0), 0.0),
            "memory_recall_q_margin_with_recall": _safe_float(
                action_info.get("memory_recall_q_margin_with_recall", 0.0), 0.0
            ),
            "memory_recall_base_greedy_action": (
                None
                if action_info.get("memory_recall_base_greedy_action") is None
                else int(action_info.get("memory_recall_base_greedy_action"))
            ),
            "memory_recall_recall_greedy_action": (
                None
                if action_info.get("memory_recall_recall_greedy_action") is None
                else int(action_info.get("memory_recall_recall_greedy_action"))
            ),
            "memory_recall_greedy_changed": bool(greedy_changed),
        },
        "gate_config": {
            "memory_recall_gate_change_required": bool(
                action_info.get("memory_recall_gate_change_required", False)
            ),
            "memory_recall_gate_visible_wall_reject": bool(
                action_info.get("memory_recall_gate_visible_wall_reject", False)
            ),
            "memory_recall_gate_min_epsilon": _safe_float(
                action_info.get("memory_recall_gate_min_epsilon", 0.0), 0.0
            ),
        },
        "gate_outcome": {
            "eligible": bool(eligible),
            "gate_passed": bool(gate_passed),
            "used": bool(used),
            "disabled_for_ablation": bool(disabled_for_ablation),
            "match_count": int(match_count),
        },
        "memory_query": {
            "enabled": bool(memory_query_state.get("enabled", False)),
            "query_requested": bool(memory_query_state.get("query_requested", False)),
            "query_executed": bool(memory_query_state.get("query_executed", False)),
            "block_reason": str(memory_query_state.get("block_reason", "")),
            "query_reason": str(memory_query_state.get("query_reason", "")),
            "query_verse_name": str(query_verse_name),
            "match_count": int(memory_query_state.get("match_count", 0) or 0),
        },
        "ablation": {
            "enabled": bool(recall_ablation_state.get("enabled", False)),
            "eligible": bool(recall_ablation_state.get("eligible", False)),
            "randomized": bool(recall_ablation_state.get("randomized", False)),
            "disabled_apply": bool(recall_ablation_state.get("disabled_apply", False)),
            "reason": str(recall_ablation_state.get("reason", "")),
            "prob": _safe_float(recall_ablation_state.get("prob", 0.0), 0.0),
        },
    }
    if isinstance(bundle, dict):
        record["transfer_request"] = {
            "reason": str(bundle.get("reason", "")),
            "query_step_idx": int(bundle.get("query_step_idx", step_idx) or step_idx),
            "match_count": int(bundle.get("match_count", 0) or 0),
            "query": dict(bundle.get("query", {})) if isinstance(bundle.get("query"), dict) else {},
        }
    return record
