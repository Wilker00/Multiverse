"""
core/safe_executor.py

Runtime safety wrapper with three controls:
1) Competence Shield: veto high-risk/low-confidence actions.
2) Recursive Fallback: temporarily route control to a safer fallback policy.
3) Checkpoint Recovery: rewind to last safe checkpoint after dangerous outcomes.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from core.agent_base import ActionResult
from core.mcts_search import AgentPolicyPrior, MCTSConfig, MCTSSearch, MetaTransformerValue
from core.planning_budget import PlanningBudget, PlanningBudgetConfig
from core.runtime_confidence import RuntimeConfidenceMonitor
from core.safe_executor_policy_support import (
    apply_mcts_overrides_for_verse_support,
    block_action_support,
    effective_veto_thresholds_support,
    estimate_action_risk_support,
    is_action_blocked_support,
    record_runtime_error_support,
    reset_episode_support,
    select_action_support,
)
from core.safe_executor_runtime_support import (
    can_checkpoint,
    can_rewind,
    check_shield,
    choose_safe_alternative,
    mcts_takeover,
    planner_takeover,
    rewind_to_checkpoint,
    save_checkpoint,
)
from core.safe_executor_risk_support import (
    load_confidence_model_support,
    predict_confidence_model_danger_support,
    load_danger_map_support,
    get_danger_map_match_support,
    load_failure_signatures_support,
    get_failure_signature_match_support,
    generate_veto_explanation_support,
)
from core.safe_executor_outcome_support import process_post_step_outcome_support
from core.safe_executor_support import (
    SafeExecutorConfig,
)
from core.types import JSONValue
from core.verse_base import StepResult


class SafeExecutor:
    def __init__(
        self,
        *,
        config: SafeExecutorConfig,
        verse: Any,
        fallback_agent: Optional[Any] = None,
    ):
        self.config = config
        self.verse = verse
        self.fallback_agent = fallback_agent

        self._fallback_remaining = 0
        self._rewinds_used = 0
        self._confidence_torch = None
        self._confidence_model = None
        self._monitor = RuntimeConfidenceMonitor(
            window_size=max(1, int(config.competence_window)),
            min_competence_rate=float(config.min_competence_rate),
            min_action_confidence=float(config.min_action_confidence),
            danger_threshold=float(config.danger_threshold),
        )
        self._planning_budget = PlanningBudget(
            PlanningBudgetConfig(
                enabled=bool(config.planner_enabled),
                base_threshold=float(config.planner_confidence_threshold),
                regret_adaptation=float(config.planning_regret_adaptation),
                budget_per_episode=int(config.planning_budget_per_episode),
                budget_per_minute=int(config.planning_budget_per_minute),
                history_window=max(8, int(config.competence_window) * 4),
            )
        )

        self._checkpoint_state: Optional[Dict[str, JSONValue]] = None
        self._checkpoint_obs: Optional[JSONValue] = None
        self._checkpoint_agent_state: Optional[Dict[str, Any]] = None
        self._blocked_actions: Dict[str, Set[int]] = {}
        self._planner_buffer: List[int] = []
        self._planner_takeover_remaining: int = 0
        self._mcts: Optional[MCTSSearch] = None
        self._mcts_value_net: Optional[Any] = None
        self._last_mcts_result: Dict[str, JSONValue] = {}
        self._shield: Optional[Any] = None # Will lazy init
        self._runtime_error_counts: Dict[str, int] = {}
        self._runtime_error_recent: List[Dict[str, Any]] = []
        self._danger_clusters: List[Dict[str, Any]] = []
        self._danger_map_embedding_dim: int = 0
        if self.config.danger_map_path and os.path.isfile(self.config.danger_map_path):
            self._danger_clusters, self._danger_map_embedding_dim = load_danger_map_support(
                self.config.danger_map_path, self._record_runtime_error
            )
        self._failure_signatures: List[Dict[str, Any]] = []
        if self.config.failure_signature_path and os.path.isfile(self.config.failure_signature_path):
            self._failure_signatures = self._load_failure_signatures(self.config.failure_signature_path)
        self._load_confidence_model_if_available(self.config.confidence_model_path)

        vname = ""
        try:
            vname = str(getattr(getattr(self.verse, "spec", None), "verse_name", "")).strip().lower()
        except Exception as exc:
            vname = ""
            self._record_runtime_error(
                code="verse_name_resolution_error",
                exc=exc,
                context="safe_executor.verse_name",
            )
        self._apply_mcts_overrides_for_verse(vname)
        allow = set(str(x).strip().lower() for x in (self.config.planner_verse_allowlist or []))
        self._planner_active = bool(self.config.planner_enabled) and (not allow or vname in allow)
        self._mcts_active = bool(self.config.mcts_enabled)
        if self._mcts_active:
            try:
                self._mcts = MCTSSearch(
                    verse=self.verse,
                    config=MCTSConfig(
                        num_simulations=int(self.config.mcts_num_simulations),
                        max_depth=int(self.config.mcts_max_depth),
                        c_puct=float(self.config.mcts_c_puct),
                        discount=float(self.config.mcts_discount),
                        forced_loss_threshold=float(self.config.mcts_loss_threshold),
                        forced_loss_min_visits=int(self.config.mcts_min_visits),
                        value_confidence_threshold=float(self.config.mcts_value_confidence_threshold),
                    ),
                )
            except Exception as exc:
                self._mcts = None
                self._mcts_active = False
                self._record_runtime_error(
                    code="mcts_init_error",
                    exc=exc,
                    context="safe_executor.mcts.init",
                )
        if self._mcts_active and str(self.config.mcts_meta_model_path).strip():
            try:
                self._mcts_value_net = MetaTransformerValue(
                    checkpoint_path=str(self.config.mcts_meta_model_path).strip(),
                    history_len=int(self.config.mcts_meta_history_len),
                )
            except Exception as exc:
                self._mcts_value_net = None
                self._record_runtime_error(
                    code="mcts_value_model_load_error",
                    exc=exc,
                    context="safe_executor.mcts.value_model",
                )

        self._episode_counters: Dict[str, int] = {
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
        self._steps_observed: int = 0

    def _record_runtime_error(self, *, code: str, exc: Exception, context: str = "") -> None:
        record_runtime_error_support(self, code=code, exc=exc, context=context)

    def _load_confidence_model_if_available(self, model_path: str) -> None:
        self._confidence_torch, self._confidence_model = load_confidence_model_support(
            model_path, self._record_runtime_error
        )

    def _predict_confidence_model_danger(self, *, obs: JSONValue, action: int) -> Optional[float]:
        return predict_confidence_model_danger_support(
            self._confidence_torch,
            self._confidence_model,
            int(self.config.confidence_model_obs_dim),
            obs,
            action,
            self._record_runtime_error,
        )

    def _get_danger_map_match(self, obs: JSONValue) -> Optional[Dict[str, Any]]:
        return get_danger_map_match_support(
            self._danger_clusters,
            self._danger_map_embedding_dim,
            self.config.danger_map_similarity_threshold,
            obs,
            self._record_runtime_error,
        )

    def _load_failure_signatures(self, path: str) -> List[Dict[str, Any]]:
        return load_failure_signatures_support(
            path,
            int(self.config.failure_signature_embedding_dim),
            self._record_runtime_error,
        )

    def _get_failure_signature_match(self, obs: JSONValue, action: int) -> Optional[Dict[str, Any]]:
        return get_failure_signature_match_support(
            self._failure_signatures,
            int(self.config.failure_signature_embedding_dim),
            float(self.config.failure_signature_similarity_threshold),
            obs,
            action,
            self._record_runtime_error,
        )

    def _generate_veto_explanation(
        self,
        *,
        low_conf: bool,
        high_danger: bool,
        blocked: bool,
        danger_map_match: Optional[Dict[str, Any]],
        failure_signature_match: Optional[Dict[str, Any]] = None,
    ) -> str:
        return generate_veto_explanation_support(
            low_conf=low_conf,
            high_danger=high_danger,
            blocked=blocked,
            danger_map_match=danger_map_match,
            failure_signature_match=failure_signature_match,
        )

    def reset_episode(self, seed: Optional[int]) -> None:
        reset_episode_support(self, seed)

    def select_action(self, primary_agent: Any, obs: JSONValue) -> ActionResult:
        return select_action_support(self, primary_agent, obs)

    def post_step(
        self,
        *,
        obs: JSONValue,
        action_result: ActionResult,
        step_result: StepResult,
        step_idx: int,
        primary_agent: Any = None,
    ) -> StepResult:
        if not self.config.enabled:
            return step_result
        return process_post_step_outcome_support(
            executor=self,
            obs=obs,
            action_result=action_result,
            step_result=step_result,
            step_idx=step_idx,
            primary_agent=primary_agent,
        )

    def close(self) -> None:
        if self.fallback_agent is not None:
            try:
                self.fallback_agent.close()
            except Exception as exc:
                self._record_runtime_error(
                    code="fallback_close_error",
                    exc=exc,
                    context="safe_executor.fallback.close",
                )

    def _apply_mcts_overrides_for_verse(self, verse_name: str) -> None:
        apply_mcts_overrides_for_verse_support(self, verse_name)

    def _effective_veto_thresholds(self) -> Tuple[float, float, float]:
        return effective_veto_thresholds_support(self)

    def _is_action_blocked(self, obs: JSONValue, action: int) -> bool:
        return is_action_blocked_support(self, obs, action)

    def _block_action(self, obs: JSONValue, action: int) -> None:
        block_action_support(self, obs, action)

    def _estimate_action_risk(self, agent: Any, obs: JSONValue, action: int) -> Dict[str, float]:
        return estimate_action_risk_support(self, agent, obs, action)

    def _choose_safe_alternative(self, *, primary_agent: Any, obs: JSONValue, denied_action: int) -> Optional[ActionResult]:
        return choose_safe_alternative(self, primary_agent=primary_agent, obs=obs, denied_action=denied_action)

    def _can_checkpoint(self) -> bool:
        return can_checkpoint(self)

    def _can_rewind(self) -> bool:
        return can_rewind(self)

    def _save_checkpoint(self, obs: JSONValue, agent: Optional[Any] = None) -> None:
        save_checkpoint(self, obs, agent)

    def _rewind_to_checkpoint(self, agent: Optional[Any] = None) -> bool:
        return rewind_to_checkpoint(self, agent)

    def _check_shield(self, obs: JSONValue, action: int) -> float:
        return check_shield(self, obs, action)

    def _mcts_takeover(
        self,
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
        return mcts_takeover(
            self,
            primary_agent=primary_agent,
            obs=obs,
            denied_action=denied_action,
            risk=risk,
            blocked=blocked,
            low_conf=low_conf,
            high_danger=high_danger,
            effective_min_confidence=effective_min_confidence,
            effective_danger_threshold=effective_danger_threshold,
            veto_adaptation=veto_adaptation,
            force_alternative=force_alternative,
            failure_signature_match=failure_signature_match,
        )

    def _planner_takeover(self, *, obs: JSONValue, denied_action: int) -> Optional[ActionResult]:
        return planner_takeover(self, obs=obs, denied_action=denied_action)
