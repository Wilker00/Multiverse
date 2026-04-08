"""
core/rollout.py

The first runnable loop in u.ai.

This file:
- Wires Verse + Agent together
- Produces StepEvent records
- Does NOT train by default (learning hooks are optional)
- Is intentionally boring and explicit

If this file is solid, everything else is an extension.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.memory_runtime import request_memory_payload, resolve_memory_request
from core.rollout_support import (
    _memory_recall_transfer_decision_record,
    _record_runtime_warning,
    _selector_routing_telemetry,
)
from core.types import (
    AgentRef,
    RunRef,
    StepEvent,
    VerseRef,
    make_step_event,
)
from core.types import JSONValue
from core.verse_base import Verse, ResetResult, StepResult, validate_reset_result, validate_step_result
from core.agent_base import Agent, ActionResult, ExperienceBatch, Transition


class RolloutConfig:
    """
    Simple config object so this does not turn into a function with 20 params.
    """

    def __init__(
        self,
        schema_version: str,
        max_steps: int,
        train: bool = False,
        collect_transitions: bool = False,
        safe_executor: Optional[Any] = None,
        retriever: Optional[Any] = None,
        retrieval_interval: Optional[int] = None,
        on_demand_memory_enabled: bool = False,
        bootstrap_memory_enabled: bool = False,
        on_demand_memory_root: str = "central_memory",
        on_demand_query_budget: Optional[int] = None,
        on_demand_min_interval: Optional[int] = None,
        on_demand_recall_ablation_prob: float = 0.0,
        on_demand_recall_ablation_seed: Optional[int] = None,
    ):
        self.schema_version = schema_version
        self.max_steps = max_steps
        self.train = train
        self.collect_transitions = collect_transitions
        self.safe_executor = safe_executor
        self.retriever = retriever
        # Use environment variables as defaults if not explicitly provided
        self.retrieval_interval = (
            retrieval_interval if retrieval_interval is not None
            else int(os.environ.get("MULTIVERSE_ROLLOUT_RETRIEVAL_INTERVAL", "10"))
        )
        self.on_demand_memory_enabled = bool(on_demand_memory_enabled)
        self.bootstrap_memory_enabled = bool(bootstrap_memory_enabled)
        self.on_demand_memory_root = str(on_demand_memory_root)
        self.on_demand_query_budget = max(0, int(
            on_demand_query_budget if on_demand_query_budget is not None
            else int(os.environ.get("MULTIVERSE_ROLLOUT_QUERY_BUDGET", "8"))
        ))
        self.on_demand_min_interval = max(1, int(
            on_demand_min_interval if on_demand_min_interval is not None
            else int(os.environ.get("MULTIVERSE_ROLLOUT_MIN_INTERVAL", "2"))
        ))
        self.on_demand_recall_ablation_prob = max(0.0, min(1.0, float(on_demand_recall_ablation_prob)))
        self.on_demand_recall_ablation_seed = (
            None if on_demand_recall_ablation_seed is None else int(on_demand_recall_ablation_seed)
        )

class RolloutResult:
    """
    What a rollout returns after one episode.
    """

    def __init__(
        self,
        events: List[StepEvent],
        episode_id: str,
        steps: int,
        return_sum: float,
        train_metrics: Optional[Dict[str, float]] = None,
    ):
        self.events = events
        self.episode_id = episode_id
        self.steps = steps
        self.return_sum = return_sum
        self.train_metrics = train_metrics or {}


def run_episode(
    *,
    verse: Verse,
    verse_ref: VerseRef,
    agent: Agent,
    agent_ref: AgentRef,
    run: RunRef,
    config: RolloutConfig,
    seed: Optional[int] = None,
    on_step: Optional[Callable[[StepEvent], None]] = None,
) -> RolloutResult:
    """
    Run a single episode.

    on_step(event) can be used to stream events to a logger
    without holding everything in memory.
    """

    if seed is not None:
        verse.seed(seed)
        agent.seed(seed)
    ablation_seed_payload = f"{config.on_demand_recall_ablation_seed}|{seed}|{run.run_id}|{agent_ref.agent_id}|{verse_ref.verse_id}"
    ablation_seed_int = int(hashlib.sha1(ablation_seed_payload.encode("utf-8")).hexdigest()[:16], 16)
    recall_ablation_rng = random.Random(ablation_seed_int)
    runtime_error_counters: Dict[str, int] = {}
    pending_runtime_warnings: List[Dict[str, Any]] = []
    if config.safe_executor is not None:
        try:
            config.safe_executor.reset_episode(seed)
        except Exception as exc:
            _record_runtime_warning(
                counters=runtime_error_counters,
                warnings=pending_runtime_warnings,
                code="safe_executor_reset_error",
                component="rollout.safe_executor.reset_episode",
                step_idx=None,
                exc=exc,
            )

    reset_result: ResetResult = verse.reset()
    validate_reset_result(reset_result)
    reset_info = dict(reset_result.info or {}) if isinstance(reset_result.info, dict) else {}
    pending_terminal_autoresolve = bool(reset_info.get("blackjack_pending", False))

    episode_id = StepEvent.new_episode_id()
    obs = reset_result.obs

    events: List[StepEvent] = []
    transitions: List[Transition] = []

    step_idx = 0
    done = False
    return_sum = 0.0
    memory_queries_used = 0
    last_memory_query_step = -10**9
    retrieval_interval = max(1, int(config.retrieval_interval))
    safe_executor_verbose = str(os.environ.get("MULTIVERSE_SAFE_EXECUTOR_VERBOSE", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    on_demand_find_similar = None
    on_demand_mem_cfg = None
    if bool(config.on_demand_memory_enabled) or bool(config.bootstrap_memory_enabled):
        try:
            from memory.central_repository import CentralMemoryConfig, find_similar

            on_demand_find_similar = find_similar
            on_demand_mem_cfg = CentralMemoryConfig(root_dir=str(config.on_demand_memory_root))
        except Exception as exc:
            on_demand_find_similar = None
            on_demand_mem_cfg = None
            _record_runtime_warning(
                counters=runtime_error_counters,
                warnings=pending_runtime_warnings,
                code="memory_lookup_init_error",
                component="rollout.memory.init",
                step_idx=None,
                exc=exc,
            )

    bootstrap_memory_state: Dict[str, Any] = {
        "enabled": bool(config.bootstrap_memory_enabled),
        "query_requested": False,
        "query_executed": False,
        "match_count": 0,
        "block_reason": "",
        "query_reason": "",
        "active_on_step": False,
    }
    bootstrap_hint: Optional[Dict[str, Any]] = None
    if bool(config.bootstrap_memory_enabled):
        if on_demand_find_similar is None or on_demand_mem_cfg is None:
            bootstrap_memory_state["block_reason"] = "lookup_unavailable"
        elif not hasattr(agent, "memory_bootstrap_request"):
            bootstrap_memory_state["block_reason"] = "agent_no_bootstrap_api"
        else:
            bootstrap_req = request_memory_payload(
                agent=agent,
                method_name="memory_bootstrap_request",
                request_obs=obs,
                request_step_idx=0,
                on_error=lambda code, component, step_idx_local, exc: _record_runtime_warning(
                    counters=runtime_error_counters,
                    warnings=pending_runtime_warnings,
                    code=code,
                    component=component,
                    step_idx=step_idx_local,
                    exc=exc,
                ),
                error_code="memory_bootstrap_request_error",
                component="rollout.memory.bootstrap_request",
            )
            if isinstance(bootstrap_req, dict):
                bootstrap_memory_state["query_requested"] = True
                bootstrap_memory_state["query_reason"] = str(bootstrap_req.get("reason", "bootstrap"))
                bootstrap_resolved = resolve_memory_request(
                    agent=agent,
                    req=bootstrap_req,
                    default_obs=obs,
                    request_step_idx=0,
                    find_similar_fn=on_demand_find_similar,
                    memory_cfg=on_demand_mem_cfg,
                    on_error=lambda code, component, step_idx_local, exc: _record_runtime_warning(
                        counters=runtime_error_counters,
                        warnings=pending_runtime_warnings,
                        code=code,
                        component=component,
                        step_idx=step_idx_local,
                        exc=exc,
                    ),
                    lookup_error_code="memory_bootstrap_lookup_error",
                    lookup_component="rollout.memory.bootstrap_lookup",
                    response_error_code="memory_bootstrap_response_error",
                )
                if isinstance(bootstrap_resolved, dict):
                    bootstrap_memory_state["query_executed"] = True
                    bootstrap_memory_state["match_count"] = int(bootstrap_resolved.get("match_count", 0))
                    bootstrap_memory_state["block_reason"] = "executed"
                    bootstrap_hint = {"memory_recall": bootstrap_resolved["bundle"]}
                else:
                    bootstrap_memory_state["block_reason"] = "lookup_error"
            else:
                bootstrap_memory_state["block_reason"] = "agent_declined"

    while not done and step_idx < config.max_steps:
        step_runtime_warnings: List[Dict[str, Any]] = list(pending_runtime_warnings)
        pending_runtime_warnings.clear()
        query_budget = max(0, int(config.on_demand_query_budget))
        memory_query_state: Dict[str, Any] = {
            "enabled": bool(config.on_demand_memory_enabled),
            "used": int(memory_queries_used),
            "budget": int(query_budget),
            "remaining": int(max(0, query_budget - int(memory_queries_used))),
            "can_query": False,
            "query_requested": False,
            "query_executed": False,
            "block_reason": "",
            "last_query_step_idx": int(last_memory_query_step),
        }
        recall_ablation_state: Dict[str, Any] = {
            "enabled": bool(float(config.on_demand_recall_ablation_prob) > 0.0),
            "eligible": False,
            "randomized": False,
            "prob": float(config.on_demand_recall_ablation_prob),
            "disabled_apply": False,
            "reason": "",
        }
        hint = None
        bootstrap_active_on_step = bool(
            step_idx == 0
            and isinstance(bootstrap_hint, dict)
            and isinstance(bootstrap_hint.get("memory_recall"), dict)
        )
        if bootstrap_active_on_step:
            hint = dict(bootstrap_hint)
        if config.retriever and step_idx % retrieval_interval == 0:
            try:
                from memory.retrieval import EpisodeFilter
                # 1. Try local verse match first
                flt = EpisodeFilter(verse_name=verse_ref.verse_name, reached_goal=True, limit=1)
                matches = config.retriever.filter_episodes(flt)
                
                # 2. If no local success, try Strategic Bridge match
                if not matches:
                    from memory.semantic_bridge import _strategy_signature
                    sig = _strategy_signature(obs, verse_ref.verse_name)
                    if sig:
                        # Search for ANY verse with a similar strategic signature
                        # This allows a Chess agent to 'recall' a high-pressure scenario from Go.
                        flt_strat = EpisodeFilter(strategic_match=sig, reached_goal=True, limit=1)
                        matches = config.retriever.filter_episodes(flt_strat)
                
                if matches:
                    m = matches[0]
                    hint = {
                        "successful_episode": m["episode_id"],
                        "source_verse": m.get("verse_name"),
                        "strategic_signature": m.get("strategic_signature"),
                    }
            except Exception as exc:
                _record_runtime_warning(
                    counters=runtime_error_counters,
                    warnings=step_runtime_warnings,
                    code="retrieval_hint_error",
                    component="rollout.retriever",
                    step_idx=step_idx,
                    exc=exc,
                )

        if bootstrap_active_on_step:
            memory_query_state["block_reason"] = "bootstrap_prefetched"
        elif bool(config.on_demand_memory_enabled) and hasattr(agent, "memory_query_request"):
            can_budget = bool(memory_queries_used < int(query_budget))
            can_cooldown = bool((step_idx - last_memory_query_step) >= int(config.on_demand_min_interval))
            can_query = bool(can_budget and can_cooldown)
            memory_query_state["can_query"] = bool(can_query)
            if not can_budget:
                memory_query_state["block_reason"] = "budget_exhausted"
            elif not can_cooldown:
                memory_query_state["block_reason"] = "cooldown"
            else:
                memory_query_state["block_reason"] = "ready"
            if can_query:
                req = request_memory_payload(
                    agent=agent,
                    method_name="memory_query_request",
                    request_obs=obs,
                    request_step_idx=step_idx,
                    on_error=lambda code, component, step_idx_local, exc: _record_runtime_warning(
                        counters=runtime_error_counters,
                        warnings=step_runtime_warnings,
                        code=code,
                        component=component,
                        step_idx=step_idx_local,
                        exc=exc,
                    ),
                    error_code="memory_query_request_error",
                    component="rollout.memory.request",
                )
                if isinstance(req, dict):
                    memory_query_state["query_requested"] = True
                    memory_query_state["query_reason"] = str(req.get("reason", "agent_request"))
                    resolved = resolve_memory_request(
                        agent=agent,
                        req=req,
                        default_obs=obs,
                        request_step_idx=step_idx,
                        find_similar_fn=on_demand_find_similar,
                        memory_cfg=on_demand_mem_cfg,
                        on_error=lambda code, component, step_idx_local, exc: _record_runtime_warning(
                            counters=runtime_error_counters,
                            warnings=step_runtime_warnings,
                            code=code,
                            component=component,
                            step_idx=step_idx_local,
                            exc=exc,
                        ),
                        lookup_error_code="memory_lookup_error",
                        lookup_component="rollout.memory.lookup",
                        response_error_code="memory_query_response_error",
                    )
                    if isinstance(resolved, dict):
                        memory_queries_used += 1
                        last_memory_query_step = int(step_idx)
                        memory_query_state["query_executed"] = True
                        memory_query_state["match_count"] = int(resolved.get("match_count", 0))
                        memory_query_state["block_reason"] = "executed"
                        memory_query_state["used"] = int(memory_queries_used)
                        memory_query_state["remaining"] = int(max(0, query_budget - int(memory_queries_used)))
                        memory_query_state["last_query_step_idx"] = int(last_memory_query_step)
                        if not isinstance(hint, dict):
                            hint = {}
                        hint["memory_recall"] = resolved["bundle"]
                    else:
                        memory_query_state["block_reason"] = "lookup_error"
                else:
                    memory_query_state["block_reason"] = "agent_declined"
        elif bool(config.on_demand_memory_enabled) and not hasattr(agent, "memory_query_request"):
            memory_query_state["block_reason"] = "agent_no_query_api"

        synthetic_action = False
        env_action = None
        if pending_terminal_autoresolve and step_idx == 0:
            synthetic_action = True
            env_action = 1
            action_result = ActionResult(
                action=None,
                info={"synthetic_action": True, "reason": "pending_terminal_autoresolve"},
            )
        elif config.safe_executor is not None:
            action_result = config.safe_executor.select_action(agent, obs)
        else:
            if (
                isinstance(hint, dict)
                and isinstance(hint.get("memory_recall"), dict)
                and hasattr(agent, "act_with_hint")
            ):
                recall_ablation_state["eligible"] = True
                if float(config.on_demand_recall_ablation_prob) > 0.0:
                    u = float(recall_ablation_rng.random())
                    disable_apply = bool(u < float(config.on_demand_recall_ablation_prob))
                    recall_ablation_state["randomized"] = True
                    recall_ablation_state["disabled_apply"] = bool(disable_apply)
                    recall_ablation_state["reason"] = "randomized_ablation"
                    if disable_apply:
                        hint = dict(hint)
                        hint["_memory_recall_control"] = {
                            "disable_apply": True,
                            "policy": "randomized_ablation",
                            "u": float(u),
                            "prob": float(config.on_demand_recall_ablation_prob),
                        }
                else:
                    recall_ablation_state["reason"] = "eligible_no_ablation"
            # Pass hint if the agent supports it
            if hasattr(agent, "act_with_hint"):
                action_result = agent.act_with_hint(obs, hint)
            else:
                action_result = agent.act(obs)

        try:
            explanation = (action_result.info or {}).get("safe_executor", {}).get("explanation")
            if explanation and safe_executor_verbose:
                print(f"[SafeExecutor] {explanation}")
        except Exception as exc:
            _record_runtime_warning(
                counters=runtime_error_counters,
                warnings=step_runtime_warnings,
                code="safe_executor_explanation_error",
                component="rollout.safe_executor.explanation",
                step_idx=step_idx,
                exc=exc,
            )


        action = action_result.action
        if env_action is None:
            env_action = action

        step_result: StepResult = verse.step(env_action)
        if config.safe_executor is not None:
            step_result = config.safe_executor.post_step(
                obs=obs,
                action_result=action_result,
                step_result=step_result,
                step_idx=step_idx,
                primary_agent=agent,
            )
        validate_step_result(step_result)

        event_info = dict(step_result.info or {})
        action_info = None
        if isinstance(action_result.info, dict) and action_result.info:
            action_info = dict(action_result.info)
            selector_route = _selector_routing_telemetry(
                action_info=action_info,
                obs=obs,
                verse_name=verse_ref.verse_name,
            )
            if selector_route is not None:
                action_info["selector_routing"] = dict(selector_route)
                event_info["selector_routing"] = dict(selector_route)
            event_info["action_info"] = action_info
        transfer_decision_records: List[Dict[str, Any]] = []
        try:
            tdr = _memory_recall_transfer_decision_record(
                step_idx=step_idx,
                episode_id=episode_id,
                run=run,
                agent_ref=agent_ref,
                verse_ref=verse_ref,
                action_info=action_info,
                hint=hint if isinstance(hint, dict) else None,
                memory_query_state=memory_query_state,
                recall_ablation_state=recall_ablation_state,
            )
            if isinstance(tdr, dict):
                transfer_decision_records.append(tdr)
        except Exception as exc:
            _record_runtime_warning(
                counters=runtime_error_counters,
                warnings=step_runtime_warnings,
                code="transfer_decision_record_error",
                component="rollout.transfer_decision_record",
                step_idx=step_idx,
                exc=exc,
            )
        memory_query_state["used"] = int(memory_queries_used)
        memory_query_state["remaining"] = int(max(0, query_budget - int(memory_queries_used)))
        bootstrap_event_state = dict(bootstrap_memory_state)
        bootstrap_event_state["active_on_step"] = bool(bootstrap_active_on_step)
        event_info["memory_bootstrap"] = bootstrap_event_state
        event_info["memory_query"] = memory_query_state
        event_info["memory_recall_ablation"] = recall_ablation_state
        if transfer_decision_records:
            event_info["transfer_decision_records"] = transfer_decision_records
        event_info["runtime_errors"] = {
            "counters": dict(runtime_error_counters),
            "warnings": list(step_runtime_warnings),
        }
        if step_idx == 0 and hasattr(verse, "spec"):
            event_info["verse_params"] = verse.spec.params

        event = make_step_event(
            schema_version=config.schema_version,
            run=run,
            episode_id=episode_id,
            step_idx=step_idx,
            agent=agent_ref,
            verse=verse_ref,
            obs=obs,
            action=action,
            reward=step_result.reward,
            done=step_result.done,
            truncated=step_result.truncated,
            seed=seed,
            info=event_info,
        )

        if on_step:
            on_step(event)
        else:
            events.append(event)

        if config.collect_transitions and not synthetic_action:
            transitions.append(
                Transition(
                    obs=obs,
                    action=action,
                    reward=step_result.reward,
                    next_obs=step_result.obs,
                    done=step_result.done,
                    truncated=step_result.truncated,
                    info={
                        "env_info": step_result.info,
                        "action_info": action_result.info,
                    },
                )
            )

        obs = step_result.obs
        return_sum += float(step_result.reward)
        done = bool(step_result.done or step_result.truncated)
        step_idx += 1

    train_metrics: Dict[str, float] = {}

    if config.train and config.collect_transitions:
        batch = ExperienceBatch(
            transitions=transitions,
            meta={
                "episode_id": episode_id,
                "steps": step_idx,
                "return_sum": return_sum,
            },
        )
        try:
            metrics = agent.learn(batch)
            if metrics:
                casted: Dict[str, float] = {}
                for k, v in metrics.items():
                    try:
                        casted[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                train_metrics = casted
        except NotImplementedError:
            pass

    return RolloutResult(
        events=events,
        episode_id=episode_id,
        steps=step_idx,
        return_sum=return_sum,
        train_metrics=train_metrics,
    )


def run_episodes(
    *,
    verse: Verse,
    verse_ref: VerseRef,
    agent: Agent,
    agent_ref: AgentRef,
    run: RunRef,
    config: RolloutConfig,
    episodes: int,
    seed: Optional[int] = None,
    on_step: Optional[Callable[[StepEvent], None]] = None,
) -> List[RolloutResult]:
    """
    Convenience helper for running multiple episodes back to back.
    """

    results: List[RolloutResult] = []

    for ep in range(episodes):
        ep_seed = None if seed is None else seed + ep
        result = run_episode(
            verse=verse,
            verse_ref=verse_ref,
            agent=agent,
            agent_ref=agent_ref,
            run=run,
            config=config,
            seed=ep_seed,
            on_step=on_step,
        )
        results.append(result)

    return results
