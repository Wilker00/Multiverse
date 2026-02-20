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
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.types import (
    AgentRef,
    JSONValue,
    RunRef,
    StepEvent,
    VerseRef,
    make_step_event,
    now_ms,
)
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
        retrieval_interval: int = 10,
        on_demand_memory_enabled: bool = False,
        on_demand_memory_root: str = "central_memory",
        on_demand_query_budget: int = 8,
        on_demand_min_interval: int = 2,
    ):
        self.schema_version = schema_version
        self.max_steps = max_steps
        self.train = train
        self.collect_transitions = collect_transitions
        self.safe_executor = safe_executor
        self.retriever = retriever
        self.retrieval_interval = retrieval_interval
        self.on_demand_memory_enabled = bool(on_demand_memory_enabled)
        self.on_demand_memory_root = str(on_demand_memory_root)
        self.on_demand_query_budget = max(0, int(on_demand_query_budget))
        self.on_demand_min_interval = max(1, int(on_demand_min_interval))


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
        rows.append(
            {
                "score": float(getattr(m, "score", 0.0)),
                "run_id": str(getattr(m, "run_id", "")),
                "episode_id": str(getattr(m, "episode_id", "")),
                "step_idx": int(getattr(m, "step_idx", 0)),
                "verse_name": str(getattr(m, "verse_name", "")),
                "action": getattr(m, "action", None),
                "reward": float(getattr(m, "reward", 0.0)),
                "pointer_path": _memory_pointer(
                    run_id=str(getattr(m, "run_id", "")),
                    episode_id=str(getattr(m, "episode_id", "")),
                    step_idx=int(getattr(m, "step_idx", 0)),
                ),
            }
        )
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
    if bool(config.on_demand_memory_enabled):
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
        hint = None
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

        if bool(config.on_demand_memory_enabled) and hasattr(agent, "memory_query_request"):
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
                req = None
                try:
                    req = agent.memory_query_request(obs=obs, step_idx=step_idx)  # type: ignore[attr-defined]
                except TypeError:
                    try:
                        req = agent.memory_query_request(obs)  # type: ignore[attr-defined]
                    except Exception as exc:
                        req = None
                        _record_runtime_warning(
                            counters=runtime_error_counters,
                            warnings=step_runtime_warnings,
                            code="memory_query_request_error",
                            component="rollout.memory.request",
                            step_idx=step_idx,
                            exc=exc,
                        )
                except Exception as exc:
                    req = None
                    _record_runtime_warning(
                        counters=runtime_error_counters,
                        warnings=step_runtime_warnings,
                        code="memory_query_request_error",
                        component="rollout.memory.request",
                        step_idx=step_idx,
                        exc=exc,
                    )

                if isinstance(req, dict):
                    memory_query_state["query_requested"] = True
                    memory_query_state["query_reason"] = str(req.get("reason", "agent_request"))
                    try:
                        if on_demand_find_similar is None or on_demand_mem_cfg is None:
                            raise RuntimeError("on-demand memory lookup unavailable")
                        query_obs = req.get("query_obs", obs)
                        top_k = max(1, int(req.get("top_k", 3)))
                        min_score = float(req.get("min_score", -1.0))
                        verse_name = req.get("verse_name")
                        verse_name = None if verse_name in (None, "") else str(verse_name).strip().lower()
                        memory_families = _as_set(req.get("memory_families"))
                        memory_types = _as_set(req.get("memory_types"))
                        matches = on_demand_find_similar(
                            obs=query_obs,
                            cfg=on_demand_mem_cfg,
                            top_k=top_k,
                            verse_name=verse_name,
                            min_score=min_score,
                            memory_families=memory_families,
                            memory_types=memory_types,
                        )
                        memory_queries_used += 1
                        last_memory_query_step = int(step_idx)
                        memory_query_state["query_executed"] = True
                        memory_query_state["match_count"] = int(len(matches))
                        memory_query_state["block_reason"] = "executed"
                        memory_query_state["used"] = int(memory_queries_used)
                        memory_query_state["remaining"] = int(max(0, query_budget - int(memory_queries_used)))
                        memory_query_state["last_query_step_idx"] = int(last_memory_query_step)
                        bundle = _build_memory_bundle(req=req, matches=matches, step_idx=step_idx)
                        if not isinstance(hint, dict):
                            hint = {}
                        hint["memory_recall"] = bundle
                        if hasattr(agent, "on_memory_response"):
                            try:
                                agent.on_memory_response(bundle)  # type: ignore[attr-defined]
                            except Exception as exc:
                                _record_runtime_warning(
                                    counters=runtime_error_counters,
                                    warnings=step_runtime_warnings,
                                    code="memory_query_response_error",
                                    component="rollout.memory.on_memory_response",
                                    step_idx=step_idx,
                                    exc=exc,
                                )
                    except Exception as exc:
                        memory_query_state["block_reason"] = "lookup_error"
                        _record_runtime_warning(
                            counters=runtime_error_counters,
                            warnings=step_runtime_warnings,
                            code="memory_lookup_error",
                            component="rollout.memory.lookup",
                            step_idx=step_idx,
                            exc=exc,
                        )
                else:
                    memory_query_state["block_reason"] = "agent_declined"
        elif bool(config.on_demand_memory_enabled) and not hasattr(agent, "memory_query_request"):
            memory_query_state["block_reason"] = "agent_no_query_api"

        if config.safe_executor is not None:
            action_result = config.safe_executor.select_action(agent, obs)
        else:
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

        step_result: StepResult = verse.step(action)
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
        memory_query_state["used"] = int(memory_queries_used)
        memory_query_state["remaining"] = int(max(0, query_budget - int(memory_queries_used)))
        event_info["memory_query"] = memory_query_state
        event_info["runtime_errors"] = {
            "counters": dict(runtime_error_counters),
            "warnings": list(step_runtime_warnings),
        }

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

        if config.collect_transitions:
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
