"""
Policy-flow helpers for SafeExecutor class internals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.agent_base import ActionResult
from core.types import JSONValue
from core.safe_executor_support import _obs_key, _safe_float, _safe_int, _default_mcts_overrides


def record_runtime_error_support(executor: Any, *, code: str, exc: Exception, context: str = "") -> None:
    key = str(code or "").strip() or "unknown_error"
    executor._runtime_error_counts[key] = int(executor._runtime_error_counts.get(key, 0)) + 1
    if isinstance(getattr(executor, "_episode_counters", None), dict):
        executor._episode_counters["runtime_errors"] = int(executor._episode_counters.get("runtime_errors", 0)) + 1
    event = {
        "code": key,
        "context": str(context),
        "error_type": type(exc).__name__,
        "error": str(exc)[:240],
    }
    executor._runtime_error_recent.append(event)
    if len(executor._runtime_error_recent) > 20:
        executor._runtime_error_recent = executor._runtime_error_recent[-20:]


def reset_episode_support(executor: Any, seed: Optional[int]) -> None:
    executor._fallback_remaining = 0
    executor._rewinds_used = 0
    executor._monitor.reset()
    executor._checkpoint_state = None
    executor._checkpoint_obs = None
    executor._checkpoint_agent_state = None
    executor._blocked_actions.clear()
    executor._planner_buffer = []
    executor._planner_takeover_remaining = 0
    executor._last_mcts_result = {}
    executor._runtime_error_counts = {}
    executor._runtime_error_recent = []
    executor._episode_counters = {
        "shield_vetoes": 0,
        "fallback_actions": 0,
        "rewinds": 0,
        "dangerous_outcomes": 0,
        "planner_actions": 0,
        "planner_queries": 0,
        "mcts_queries": 0,
        "mcts_vetoes": 0,
        "runtime_errors": 0,
    }
    executor._steps_observed = 0
    executor._planning_budget.reset_episode()
    if executor._mcts is not None:
        try:
            executor._mcts.seed(seed)
        except Exception as exc:
            executor._record_runtime_error(
                code="mcts_seed_error",
                exc=exc,
                context="safe_executor.mcts.seed",
            )
    if executor.fallback_agent is not None:
        try:
            executor.fallback_agent.seed(seed)
        except Exception as exc:
            executor._record_runtime_error(
                code="fallback_seed_error",
                exc=exc,
                context="safe_executor.fallback.seed",
            )


def apply_mcts_overrides_for_verse_support(executor: Any, verse_name: str) -> None:
    merged: Dict[str, Any] = {}
    merged.update(_default_mcts_overrides(verse_name))
    if isinstance(executor.config.mcts_verse_overrides, dict):
        merged.update(dict(executor.config.mcts_verse_overrides.get(str(verse_name).strip().lower(), {}) or {}))
    if not merged:
        return
    if "mcts_enabled" in merged:
        executor.config.mcts_enabled = bool(merged.get("mcts_enabled"))
    if "mcts_num_simulations" in merged:
        executor.config.mcts_num_simulations = max(
            8,
            _safe_int(merged.get("mcts_num_simulations"), executor.config.mcts_num_simulations),
        )
    if "mcts_max_depth" in merged:
        executor.config.mcts_max_depth = max(2, _safe_int(merged.get("mcts_max_depth"), executor.config.mcts_max_depth))
    if "mcts_c_puct" in merged:
        executor.config.mcts_c_puct = max(0.1, _safe_float(merged.get("mcts_c_puct"), executor.config.mcts_c_puct))
    if "mcts_discount" in merged:
        executor.config.mcts_discount = max(
            0.0,
            min(1.0, _safe_float(merged.get("mcts_discount"), executor.config.mcts_discount)),
        )
    if "mcts_loss_threshold" in merged:
        executor.config.mcts_loss_threshold = max(
            -1.0,
            min(0.0, _safe_float(merged.get("mcts_loss_threshold"), executor.config.mcts_loss_threshold)),
        )
    if "mcts_min_visits" in merged:
        executor.config.mcts_min_visits = max(1, _safe_int(merged.get("mcts_min_visits"), executor.config.mcts_min_visits))
    if "mcts_value_confidence_threshold" in merged:
        executor.config.mcts_value_confidence_threshold = max(
            0.0,
            min(
                1.0,
                _safe_float(
                    merged.get("mcts_value_confidence_threshold"),
                    executor.config.mcts_value_confidence_threshold,
                ),
            ),
        )


def effective_veto_thresholds_support(executor: Any) -> Tuple[float, float, float]:
    min_conf = float(executor.config.min_action_confidence)
    danger_thr = float(executor.config.danger_threshold)
    adaptation = 0.0
    if not bool(executor.config.adaptive_veto_enabled):
        return min_conf, danger_thr, adaptation
    if int(executor._steps_observed) < int(executor.config.adaptive_veto_warmup_steps):
        return min_conf, danger_thr, adaptation
    st = executor._monitor.status()
    if st.window_size <= 0:
        return min_conf, danger_thr, adaptation
    relax_strength = float(executor.config.adaptive_veto_relaxation)
    if bool(executor.config.adaptive_veto_schedule_enabled):
        sched_steps = max(1, int(executor.config.adaptive_veto_schedule_steps))
        sched_power = max(0.10, float(executor.config.adaptive_veto_schedule_power))
        progress_steps = max(0, int(executor._steps_observed) - int(executor.config.adaptive_veto_warmup_steps))
        progress = max(0.0, min(1.0, float(progress_steps) / float(sched_steps)))
        shaped = float(progress) ** float(sched_power)
        start = max(0.0, min(1.0, float(executor.config.adaptive_veto_relaxation_start)))
        end = max(0.0, min(1.0, float(executor.config.adaptive_veto_relaxation_end)))
        relax_strength = float(start + (end - start) * shaped)
    denom = max(1e-6, 1.0 - float(executor.config.min_competence_rate))
    competence_gain = max(
        0.0,
        min(1.0, (float(st.competence_rate) - float(executor.config.min_competence_rate)) / denom),
    )
    failure_guard = 1.0 - max(
        0.0,
        min(1.0, float(st.failure_rate) / float(max(1e-6, executor.config.adaptive_veto_failure_guard))),
    )
    adaptation = max(
        0.0,
        min(1.0, float(relax_strength) * float(competence_gain) * float(failure_guard)),
    )
    eff_min_conf = max(0.01, min(1.0, float(min_conf) * (1.0 - float(adaptation))))
    eff_danger = max(
        0.0,
        min(0.999, float(danger_thr) + (1.0 - float(danger_thr)) * float(adaptation)),
    )
    return eff_min_conf, eff_danger, adaptation


def is_action_blocked_support(executor: Any, obs: JSONValue, action: int) -> bool:
    if action < 0:
        return False
    return action in executor._blocked_actions.get(_obs_key(obs), set())


def block_action_support(executor: Any, obs: JSONValue, action: int) -> None:
    if action < 0:
        return
    k = _obs_key(obs)
    bucket = executor._blocked_actions.setdefault(k, set())
    bucket.add(int(action))


def estimate_action_risk_support(executor: Any, agent: Any, obs: JSONValue, action: int) -> Dict[str, float]:
    out: Dict[str, float] = {"danger": 0.0, "confidence": 1.0}
    if action < 0:
        return out
    try:
        if hasattr(agent, "action_diagnostics"):
            diag = agent.action_diagnostics(obs)  # type: ignore[attr-defined]
            dangers = diag.get("danger_scores") if isinstance(diag, dict) else None
            probs = diag.get("sample_probs") if isinstance(diag, dict) else None
            if isinstance(dangers, list) and 0 <= action < len(dangers):
                out["danger"] = max(0.0, min(1.0, _safe_float(dangers[action], 0.0)))
            if isinstance(probs, list) and 0 <= action < len(probs):
                out["confidence"] = max(0.0, min(1.0, _safe_float(probs[action], 1.0)))
    except Exception as exc:
        executor._record_runtime_error(
            code="risk_diag_error",
            exc=exc,
            context="safe_executor.risk.action_diagnostics",
        )
    model_danger = executor._predict_confidence_model_danger(obs=obs, action=int(action))
    if model_danger is not None:
        w = max(0.0, min(1.0, float(executor.config.confidence_model_weight)))
        diag_danger = max(0.0, min(1.0, _safe_float(out.get("danger", 0.0), 0.0)))
        diag_conf = max(0.0, min(1.0, _safe_float(out.get("confidence", 1.0), 1.0)))
        fused_danger = ((1.0 - w) * float(diag_danger)) + (w * float(model_danger))
        fused_conf = ((1.0 - w) * float(diag_conf)) + (w * (1.0 - float(model_danger)))
        out["danger"] = max(0.0, min(1.0, float(fused_danger)))
        out["confidence"] = max(0.0, min(1.0, float(fused_conf)))
        out["confidence_model_danger"] = max(0.0, min(1.0, float(model_danger)))
        out["confidence_model_blend_weight"] = float(w)
    return out


def select_action_support(executor: Any, primary_agent: Any, obs: JSONValue) -> ActionResult:
    if not executor.config.enabled:
        return primary_agent.act(obs)

    if executor._planner_active and executor._planner_takeover_remaining > 0 and executor._planner_buffer:
        chosen = int(executor._planner_buffer.pop(0))
        executor._planner_takeover_remaining = max(0, int(executor._planner_takeover_remaining - 1))
        executor._episode_counters["planner_actions"] = int(executor._episode_counters["planner_actions"]) + 1
        return ActionResult(
            action=chosen,
            info={
                "safe_executor": {
                    "mode": "planner_takeover",
                    "planner_remaining": int(executor._planner_takeover_remaining),
                }
            },
        )

    force_fallback = bool(executor._fallback_remaining > 0 and executor.fallback_agent is not None)
    if force_fallback:
        executor._fallback_remaining = max(0, int(executor._fallback_remaining - 1))
        executor._episode_counters["fallback_actions"] = int(executor._episode_counters["fallback_actions"]) + 1
        fres = executor.fallback_agent.act(obs)
        finfo = dict(fres.info or {})
        finfo["safe_executor"] = {
            "mode": "fallback",
            "fallback_remaining": int(executor._fallback_remaining),
            "shield_veto": False,
        }
        return ActionResult(action=fres.action, info=finfo)

    pres = primary_agent.act(obs)
    p_action = _safe_int(pres.action, -1)
    blocked = executor._is_action_blocked(obs, p_action)
    risk = executor._estimate_action_risk(primary_agent, obs, p_action)
    eff_conf, eff_danger, veto_adaptation = executor._effective_veto_thresholds()
    low_conf = bool(risk["confidence"] < float(eff_conf))
    high_danger = bool(risk["danger"] >= float(eff_danger))
    failure_signature_match = executor._get_failure_signature_match(obs, p_action)
    memory_forced_veto = bool(failure_signature_match is not None)
    if memory_forced_veto:
        blocked = True
        high_danger = True
    planner_low_conf = bool(risk["confidence"] < float(executor.config.planner_confidence_threshold))
    mcts_trigger = bool(
        executor._mcts_active
        and executor._mcts is not None
        and (
            (blocked and bool(executor.config.mcts_trigger_on_block))
            or (low_conf and bool(executor.config.mcts_trigger_on_low_confidence))
            or (high_danger and bool(executor.config.mcts_trigger_on_high_danger))
            or (memory_forced_veto and bool(executor.config.force_mcts_on_failure_signature))
        )
    )
    if mcts_trigger:
        mcts_alt = executor._mcts_takeover(
            primary_agent=primary_agent,
            obs=obs,
            denied_action=p_action,
            risk=risk,
            blocked=blocked,
            low_conf=low_conf,
            high_danger=high_danger,
            effective_min_confidence=float(eff_conf),
            effective_danger_threshold=float(eff_danger),
            veto_adaptation=float(veto_adaptation),
            force_alternative=bool(memory_forced_veto),
            failure_signature_match=failure_signature_match,
        )
        if mcts_alt is not None:
            return mcts_alt

    if executor.config.shield_enabled and executor.config.shield_model_path:
        shield_danger = executor._check_shield(obs, p_action)
        if shield_danger >= executor.config.shield_threshold:
            alt = executor._choose_safe_alternative(primary_agent=primary_agent, obs=obs, denied_action=p_action)
            if alt:
                info = dict(alt.info or {})
                info["safe_executor"] = {
                    "mode": "shield_reflex",
                    "denied_action": int(p_action),
                    "shield_danger": float(shield_danger),
                }
                return ActionResult(action=alt.action, info=info)

    if blocked or low_conf or high_danger:
        planner_failed = False
        danger_map_match = executor._get_danger_map_match(obs)
        planner_allowed = bool(
            executor._planner_active
            and (
                planner_low_conf
                or (blocked and executor.config.planner_trigger_on_block)
                or (high_danger and executor.config.planner_trigger_on_high_danger)
            )
        )
        if planner_allowed:
            dyn_threshold = executor._monitor.adaptive_planner_threshold(
                base_threshold=float(executor.config.planner_confidence_threshold),
                regret_adaptation=float(executor.config.planning_regret_adaptation),
            )
            planner_allowed = bool(
                executor._planning_budget.can_invoke(
                    verse_name=str(getattr(getattr(executor.verse, "spec", None), "verse_name", "unknown")),
                    confidence=float(risk["confidence"]),
                    base_threshold=float(dyn_threshold),
                )
            )
        if planner_allowed:
            planned = executor._planner_takeover(obs=obs, denied_action=p_action)
            if planned is not None:
                executor._planning_budget.consume()
                return planned
            planner_failed = True
        alt = executor._choose_safe_alternative(primary_agent=primary_agent, obs=obs, denied_action=p_action)
        if alt is None and memory_forced_veto and getattr(getattr(executor.verse, "action_space", None), "type", "") == "discrete":
            n = _safe_int(getattr(getattr(executor.verse, "action_space", None), "n", 0), 0)
            for cand in range(max(0, n)):
                if cand == int(p_action):
                    continue
                if executor._is_action_blocked(obs, int(cand)):
                    continue
                alt = ActionResult(action=int(cand), info={"safe_executor_alt": "memory_forced_fallback"})
                break
        if alt is not None:
            executor._episode_counters["shield_vetoes"] = int(executor._episode_counters["shield_vetoes"]) + 1
            explanation = executor._generate_veto_explanation(
                low_conf=low_conf,
                high_danger=high_danger,
                blocked=blocked,
                danger_map_match=danger_map_match,
                failure_signature_match=failure_signature_match,
            )
            meta = dict(alt.info or {})
            meta["safe_executor"] = {
                "mode": "shield_veto",
                "denied_action": int(p_action),
                "danger": float(risk["danger"]),
                "confidence": float(risk["confidence"]),
                "confidence_model_danger": _safe_float(risk.get("confidence_model_danger", -1.0), -1.0),
                "confidence_model_blend_weight": _safe_float(risk.get("confidence_model_blend_weight", 0.0), 0.0),
                "effective_min_confidence": float(eff_conf),
                "effective_danger_threshold": float(eff_danger),
                "veto_adaptation": float(veto_adaptation),
                "blocked": bool(blocked),
                "low_confidence": bool(low_conf),
                "high_danger": bool(high_danger),
                "planner_attempted": bool(planner_allowed),
                "planner_failed": bool(planner_failed),
                "explanation": explanation,
                "danger_map_match": danger_map_match,
                "declarative_failure_signature": bool(memory_forced_veto),
                "failure_signature_match": failure_signature_match,
            }
            return ActionResult(action=alt.action, info=meta)

    info = dict(pres.info or {})
    info["safe_executor"] = {
        "mode": "primary",
        "shield_veto": False,
        "danger": float(risk["danger"]),
        "confidence": float(risk["confidence"]),
        "confidence_model_danger": _safe_float(risk.get("confidence_model_danger", -1.0), -1.0),
        "confidence_model_blend_weight": _safe_float(risk.get("confidence_model_blend_weight", 0.0), 0.0),
        "effective_min_confidence": float(eff_conf),
        "effective_danger_threshold": float(eff_danger),
        "veto_adaptation": float(veto_adaptation),
        "declarative_failure_signature": bool(memory_forced_veto),
        "failure_signature_match": failure_signature_match,
    }
    return ActionResult(action=pres.action, info=info)
