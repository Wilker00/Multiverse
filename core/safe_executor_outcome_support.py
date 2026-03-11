"""
core/safe_executor_outcome_support.py

Support functions for post-step outcome handling in SafeExecutor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from core.agent_base import ActionResult
from core.verse_base import StepResult
from core.safe_executor_support import (
    _danger_label_from_info,
    _infer_failure_mode,
    _is_nan_action,
    _is_safety_violation,
    _safe_float,
    _safe_int,
)


def process_post_step_outcome_support(
    *,
    executor: Any,
    obs: Any,
    action_result: ActionResult,
    step_result: StepResult,
    step_idx: int,
    primary_agent: Any,
) -> StepResult:
    executor._steps_observed = int(executor._steps_observed) + 1
    info = dict(step_result.info or {})
    se_meta = dict(info.get("safe_executor") or {})
    se_meta.update(dict(action_result.info or {}).get("safe_executor") or {})

    safety_violation = _is_safety_violation(info)
    severe_penalty = float(step_result.reward) <= float(executor.config.severe_reward_threshold)
    danger_label, danger_label_source = _danger_label_from_info(info)
    dangerous_outcome = (
        bool(danger_label)
        if danger_label is not None
        else bool(safety_violation or severe_penalty)
    )
    action_meta = dict(action_result.info or {}).get("safe_executor") or {}
    action_conf = _safe_float(action_meta.get("confidence", 1.0), 1.0)
    action_danger = _safe_float(action_meta.get("danger", 0.0), 0.0)
    eff_conf = _safe_float(
        action_meta.get("effective_min_confidence", executor.config.min_action_confidence),
        float(executor.config.min_action_confidence),
    )
    eff_danger = _safe_float(
        action_meta.get("effective_danger_threshold", executor.config.danger_threshold),
        float(executor.config.danger_threshold),
    )
    low_confidence = bool(action_conf < float(eff_conf))
    high_danger = bool(action_danger >= float(eff_danger))
    planner_failed = bool(action_meta.get("planner_failed", False))
    policy_nan = _is_nan_action(action_result.action)
    executor._monitor.observe(
        confidence=float(action_conf),
        danger=float(action_danger),
        dangerous_outcome=bool(dangerous_outcome),
        reward=float(step_result.reward),
    )
    monitor_status = executor._monitor.status()
    executor._planning_budget.record_outcome(
        verse_name=str(getattr(getattr(executor.verse, "spec", None), "verse_name", "unknown")),
        reward=float(step_result.reward),
        failed=bool(dangerous_outcome),
    )

    if dangerous_outcome:
        executor._episode_counters["dangerous_outcomes"] = int(executor._episode_counters["dangerous_outcomes"]) + 1
        if executor.config.block_repeated_fail_action:
            executor._block_action(obs, _safe_int(action_result.action, -1))

    if monitor_status.in_incompetence_zone:
        executor._fallback_remaining = max(int(executor._fallback_remaining), int(executor.config.fallback_horizon_steps))

    rewound = False
    if dangerous_outcome and executor._can_rewind():
        rewound = executor._rewind_to_checkpoint(agent=primary_agent)
        if rewound:
            executor._rewinds_used += 1
            executor._episode_counters["rewinds"] = int(executor._episode_counters["rewinds"]) + 1

    if not dangerous_outcome and executor._can_checkpoint() and (int(step_idx) % int(executor.config.checkpoint_interval) == 0):
        executor._save_checkpoint(obs=step_result.obs, agent=primary_agent)

    se_meta["dangerous_outcome"] = bool(dangerous_outcome)
    se_meta["danger_label"] = (
        None if danger_label is None else bool(danger_label)
    )
    se_meta["danger_label_source"] = str(danger_label_source)
    se_meta["safety_violation"] = bool(safety_violation)
    se_meta["severe_penalty"] = bool(severe_penalty)
    se_meta["low_confidence"] = bool(low_confidence)
    se_meta["high_danger"] = bool(high_danger)
    se_meta["effective_min_confidence"] = float(eff_conf)
    se_meta["effective_danger_threshold"] = float(eff_danger)
    se_meta["planner_failed"] = bool(planner_failed)
    se_meta["policy_nan"] = bool(policy_nan)
    se_meta["rewound"] = bool(rewound)
    se_meta["rewinds_used"] = int(executor._rewinds_used)
    se_meta["fallback_remaining"] = int(executor._fallback_remaining)
    se_meta["confidence_status"] = {
        "window_size": int(monitor_status.window_size),
        "competence_rate": float(monitor_status.competence_rate),
        "mean_confidence": float(monitor_status.mean_confidence),
        "mean_danger": float(monitor_status.mean_danger),
        "mean_regret": float(monitor_status.mean_regret),
        "failure_rate": float(monitor_status.failure_rate),
        "in_incompetence_zone": bool(monitor_status.in_incompetence_zone),
    }
    se_meta["planning_budget"] = executor._planning_budget.snapshot(
        verse_name=str(getattr(getattr(executor.verse, "spec", None), "verse_name", "unknown"))
    )
    mcts_queries = int(executor._episode_counters.get("mcts_queries", 0))
    mcts_vetoes = int(executor._episode_counters.get("mcts_vetoes", 0))
    se_meta["mcts_stats"] = {
        "enabled": bool(executor._mcts_active and executor._mcts is not None),
        "queries": mcts_queries,
        "vetoes": mcts_vetoes,
        "veto_rate": float(mcts_vetoes / float(max(1, mcts_queries))),
        "last_query": dict(executor._last_mcts_result or {}),
    }
    se_meta["runtime_errors"] = {
        "total": int(sum(int(v) for v in executor._runtime_error_counts.values())),
        "by_code": dict(executor._runtime_error_counts),
        "recent": list(executor._runtime_error_recent[-5:]),
    }
    se_meta["blocked_action_count"] = int(sum(len(v) for v in executor._blocked_actions.values()))
    se_meta["counters"] = dict(executor._episode_counters)
    failure_mode = _infer_failure_mode(
        info=info,
        dangerous_outcome=bool(dangerous_outcome),
        severe_penalty=bool(severe_penalty),
        safety_violation=bool(safety_violation),
        low_confidence=bool(low_confidence),
        high_danger=bool(high_danger),
        planner_failed=bool(planner_failed),
        policy_nan=bool(policy_nan),
        done=bool(step_result.done),
        truncated=bool(step_result.truncated),
    )
    failure_signals: List[str] = []
    if bool(severe_penalty):
        failure_signals.append("severe_penalty")
    if bool(safety_violation):
        failure_signals.append("safety_violation")
    if bool(low_confidence):
        failure_signals.append("low_confidence")
    if bool(high_danger):
        failure_signals.append("high_danger")
    if bool(action_meta.get("declarative_failure_signature", False)):
        failure_signals.append("declarative_failure_signature")
    if bool(planner_failed):
        failure_signals.append("planner_failed")
    if bool(policy_nan):
        failure_signals.append("policy_nan")
    if bool(rewound):
        failure_signals.append("rewound")
    if "warning" in info or "error" in info:
        failure_signals.append("env_warning")
    if int(sum(int(v) for v in executor._runtime_error_counts.values())) > 0:
        failure_signals.append("runtime_error")
    se_meta["failure_mode"] = str(failure_mode)
    se_meta["failure_signals"] = failure_signals
    info["safe_executor"] = se_meta

    if rewound and executor._checkpoint_obs is not None:
        return StepResult(
            obs=executor._checkpoint_obs,
            reward=float(step_result.reward),
            done=False,
            truncated=False,
            info=info,
        )

    return StepResult(
        obs=step_result.obs,
        reward=float(step_result.reward),
        done=bool(step_result.done),
        truncated=bool(step_result.truncated),
        info=info,
    )
