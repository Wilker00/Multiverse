"""
Runtime helper functions for SafeExecutor takeover and recovery paths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from core.agent_base import ActionResult
from core.mcts_search import AgentPolicyPrior
from core.planner_oracle import plan_actions_from_current_state
from core.safe_executor_support import _safe_float, _safe_int
from core.types import JSONValue


def choose_safe_alternative(executor: Any, *, primary_agent: Any, obs: JSONValue, denied_action: int) -> Optional[ActionResult]:
    if executor.config.prefer_fallback_on_veto and executor.fallback_agent is not None:
        executor._episode_counters["fallback_actions"] = int(executor._episode_counters["fallback_actions"]) + 1
        fres = executor.fallback_agent.act(obs)
        info = dict(fres.info or {})
        info["safe_executor_alt"] = "fallback"
        return ActionResult(action=fres.action, info=info)

    try:
        if hasattr(primary_agent, "action_diagnostics"):
            diag = primary_agent.action_diagnostics(obs)  # type: ignore[attr-defined]
            dangers = diag.get("danger_scores") if isinstance(diag, dict) else None
            probs = diag.get("sample_probs") if isinstance(diag, dict) else None
            if isinstance(dangers, list) and len(dangers) > 0:
                candidates: List[Tuple[float, float, int]] = []
                for i in range(len(dangers)):
                    if i == int(denied_action):
                        continue
                    if executor._is_action_blocked(obs, int(i)):
                        continue
                    d = max(0.0, min(1.0, _safe_float(dangers[i], 1.0)))
                    p = 0.0
                    if isinstance(probs, list) and i < len(probs):
                        p = max(0.0, min(1.0, _safe_float(probs[i], 0.0)))
                    candidates.append((d, -p, int(i)))
                if candidates:
                    candidates.sort()
                    chosen = int(candidates[0][2])
                    return ActionResult(action=chosen, info={"safe_executor_alt": "primary_diag"})
    except Exception as exc:
        executor._record_runtime_error(
            code="alt_diag_error",
            exc=exc,
            context="safe_executor.alt.action_diagnostics",
        )

    if executor.fallback_agent is not None:
        executor._episode_counters["fallback_actions"] = int(executor._episode_counters["fallback_actions"]) + 1
        fres = executor.fallback_agent.act(obs)
        info = dict(fres.info or {})
        info["safe_executor_alt"] = "fallback"
        return ActionResult(action=fres.action, info=info)
    return None


def can_checkpoint(executor: Any) -> bool:
    return bool(hasattr(executor.verse, "export_state") and callable(getattr(executor.verse, "export_state")))


def can_rewind(executor: Any) -> bool:
    if executor._rewinds_used >= int(executor.config.max_rewinds_per_episode):
        return False
    if executor._checkpoint_state is None or executor._checkpoint_obs is None:
        return False
    return bool(hasattr(executor.verse, "import_state") and callable(getattr(executor.verse, "import_state")))


def save_checkpoint(executor: Any, obs: JSONValue, agent: Optional[Any] = None) -> None:
    try:
        executor._checkpoint_state = executor.verse.export_state()
        executor._checkpoint_obs = obs
        if agent and hasattr(agent, "get_state"):
            executor._checkpoint_agent_state = agent.get_state()
    except Exception as exc:
        executor._record_runtime_error(
            code="checkpoint_save_error",
            exc=exc,
            context="safe_executor.checkpoint.save",
        )


def rewind_to_checkpoint(executor: Any, agent: Optional[Any] = None) -> bool:
    if executor._checkpoint_state is None:
        return False
    try:
        executor.verse.import_state(executor._checkpoint_state)
        if agent and hasattr(agent, "set_state") and executor._checkpoint_agent_state is not None:
            agent.set_state(executor._checkpoint_agent_state)
        return True
    except Exception as exc:
        executor._record_runtime_error(
            code="checkpoint_rewind_error",
            exc=exc,
            context="safe_executor.checkpoint.rewind",
        )
        return False


def check_shield(executor: Any, obs: JSONValue, action: int) -> float:
    if not executor._shield:
        from models.safety_shield import SafetyShield
        try:
            from agents.ppo_agent import _obs_to_tensor

            vec = _obs_to_tensor(obs).numpy()
            dim = len(vec) + 1
            executor._shield = SafetyShield(input_dim=dim, checkpoint_path=executor.config.shield_model_path)
        except Exception as exc:
            executor._record_runtime_error(
                code="shield_init_error",
                exc=exc,
                context="safe_executor.shield.init",
            )
            return 0.0

    try:
        from agents.ppo_agent import _obs_to_tensor

        vec = _obs_to_tensor(obs).numpy()
        return executor._shield.predict_danger(vec, action)
    except Exception as exc:
        executor._record_runtime_error(
            code="shield_infer_error",
            exc=exc,
            context="safe_executor.shield.infer",
        )
        return 0.0


def mcts_takeover(
    executor: Any,
    *,
    primary_agent: Any,
    obs: JSONValue,
    denied_action: int,
    risk: Dict[str, float],
    blocked: bool,
    low_conf: bool,
    high_danger: bool,
    effective_min_confidence: float,
    effective_danger_threshold: float,
    veto_adaptation: float,
    force_alternative: bool = False,
    failure_signature_match: Optional[Dict[str, Any]] = None,
) -> Optional[ActionResult]:
    if executor._mcts is None:
        return None
    if denied_action < 0:
        return None
    try:
        executor._episode_counters["mcts_queries"] = int(executor._episode_counters["mcts_queries"]) + 1
        policy_net = AgentPolicyPrior(
            agent=primary_agent,
            action_count=max(1, _safe_int(getattr(getattr(executor.verse, "action_space", None), "n", 0), 0)),
        )
        result = executor._mcts.search(
            root_obs=obs,
            policy_net=policy_net,
            value_net=executor._mcts_value_net,
        )
        executor._last_mcts_result = {
            "best_action": int(result.best_action),
            "root_value": float(result.root_value),
            "avg_leaf_value": float(result.avg_leaf_value),
            "simulations": int(result.simulations),
            "forced_loss_detected": bool(result.forced_loss_detected),
            "forced_loss_actions": [int(a) for a in (result.forced_loss_actions or [])],
            "principal_variation": [int(a) for a in (result.principal_variation or [])],
        }
        forced = set(int(a) for a in (result.forced_loss_actions or []))
        if denied_action not in forced and not bool(force_alternative):
            return None

        candidates = [
            int(a)
            for a in range(len(result.action_probs))
            if int(a) != int(denied_action)
            and int(a) not in forced
            and not executor._is_action_blocked(obs, int(a))
        ]
        if not candidates:
            candidates = [
                int(a)
                for a in range(len(result.action_probs))
                if int(a) != int(denied_action) and not executor._is_action_blocked(obs, int(a))
            ]
        if not candidates:
            return None

        chosen = max(
            candidates,
            key=lambda a: (
                _safe_float(result.action_probs[a] if a < len(result.action_probs) else 0.0, 0.0),
                _safe_float(result.action_values[a] if a < len(result.action_values) else -1.0, -1.0),
                _safe_int(result.visit_counts[a] if a < len(result.visit_counts) else 0, 0),
            ),
        )
        executor._episode_counters["mcts_vetoes"] = int(executor._episode_counters["mcts_vetoes"]) + 1
        executor._episode_counters["shield_vetoes"] = int(executor._episode_counters["shield_vetoes"]) + 1
        return ActionResult(
            action=int(chosen),
            info={
                "safe_executor": {
                    "mode": "mcts_veto",
                    "denied_action": int(denied_action),
                    "chosen_action": int(chosen),
                    "danger": float(risk.get("danger", 0.0)),
                    "confidence": float(risk.get("confidence", 1.0)),
                    "confidence_model_danger": _safe_float(risk.get("confidence_model_danger", -1.0), -1.0),
                    "confidence_model_blend_weight": _safe_float(risk.get("confidence_model_blend_weight", 0.0), 0.0),
                    "effective_min_confidence": float(effective_min_confidence),
                    "effective_danger_threshold": float(effective_danger_threshold),
                    "veto_adaptation": float(veto_adaptation),
                    "blocked": bool(blocked),
                    "low_confidence": bool(low_conf),
                    "high_danger": bool(high_danger),
                    "forced_loss_actions": [int(a) for a in sorted(forced)],
                    "mcts_best_action": int(result.best_action),
                    "mcts_principal_variation": [int(a) for a in (result.principal_variation or [])],
                    "declarative_failure_signature": bool(failure_signature_match is not None),
                    "failure_signature_match": failure_signature_match,
                }
            },
        )
    except Exception as exc:
        executor._record_runtime_error(
            code="mcts_takeover_error",
            exc=exc,
            context="safe_executor.mcts.takeover",
        )
        return None


def planner_takeover(executor: Any, *, obs: JSONValue, denied_action: int) -> Optional[ActionResult]:
    try:
        executor._episode_counters["planner_queries"] = int(executor._episode_counters["planner_queries"]) + 1
        actions = plan_actions_from_current_state(
            verse=executor.verse,
            horizon=max(1, int(executor.config.planner_horizon)),
            max_expansions=max(100, int(executor.config.planner_max_expansions)),
            avoid_terminal_failures=True,
        )
        if not actions:
            return None
        filtered = [int(a) for a in actions if int(a) != int(denied_action) and not executor._is_action_blocked(obs, int(a))]
        if not filtered:
            return None
        executor._planner_buffer = list(filtered)
        executor._planner_takeover_remaining = max(0, min(len(executor._planner_buffer), int(executor.config.planner_horizon)))
        chosen = int(executor._planner_buffer.pop(0))
        executor._planner_takeover_remaining = max(0, int(executor._planner_takeover_remaining - 1))
        executor._episode_counters["planner_actions"] = int(executor._episode_counters["planner_actions"]) + 1
        return ActionResult(
            action=chosen,
            info={"safe_executor": {"mode": "planner_takeover", "planner_remaining": int(executor._planner_takeover_remaining)}},
        )
    except Exception as exc:
        executor._record_runtime_error(
            code="planner_takeover_error",
            exc=exc,
            context="safe_executor.planner.takeover",
        )
        return None
