"""
orchestrator/marl_trainer.py

Minimal multi-agent trainer with shared communication bus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.agent_base import Agent, ExperienceBatch, Transition
from core.communication import Message, MessageBus, SharedMemoryPool
from core.rollout import RolloutConfig
from core.types import AgentRef, RunRef, VerseRef, VerseSpec, AgentSpec, make_step_event
from memory.event_log import EventLogConfig, EventLogger, make_on_step_writer
from memory.central_repository import CentralMemoryConfig, find_similar
from verses.registry import register_builtin as register_builtin_verses, create_verse
from agents.registry import register_builtin_agents, create_agent


def _hash_verse_spec(spec: VerseSpec) -> str:
    payload = {
        "spec_version": str(spec.spec_version),
        "verse_name": str(spec.verse_name),
        "verse_version": str(spec.verse_version),
        "seed": spec.seed,
        "tags": list(spec.tags or []),
        "params": dict(spec.params or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _obs_hash(obs: Any) -> str:
    try:
        raw = json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except Exception:
        raw = str(obs).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def _selector_routing_telemetry(
    *,
    action_info: Optional[Dict[str, Any]],
    obs: Any,
    verse_name: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(action_info, dict):
        return None
    experts_raw = action_info.get("experts")
    weights_raw = action_info.get("weights")
    if (not isinstance(experts_raw, list)) or (not isinstance(weights_raw, list)):
        selected = str(action_info.get("selected_expert", "")).strip()
        if not selected:
            return None
        try:
            conf = float(action_info.get("selector_confidence", 0.0))
        except Exception:
            conf = 0.0
        return {
            "timestamp_ms": int(event_time_ms()),
            "verse": str(verse_name),
            "obs_hash": _obs_hash(obs),
            "selected_expert": selected,
            "confidence": float(conf),
            "selector_active": bool(action_info.get("selector_active", False)),
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
    return {
        "timestamp_ms": int(event_time_ms()),
        "verse": str(verse_name),
        "obs_hash": _obs_hash(obs),
        "selected_expert": str(pairs[0]["expert"]),
        "confidence": float(pairs[0]["weight"]),
        "selector_active": bool(action_info.get("selector_active", False)),
        "top_experts": pairs[:5],
    }


def event_time_ms() -> int:
    from core.types import now_ms

    return int(now_ms())


@dataclass
class MARLConfig:
    episodes: int
    max_steps: int
    train: bool = False
    collect_transitions: bool = False
    shared_memory_enabled: bool = True
    shared_memory_top_k: int = 4
    negotiation_interval: int = 1
    lexicon_min_support: int = 2
    on_demand_memory_enabled: bool = False
    on_demand_memory_root: str = "central_memory"
    on_demand_query_budget: int = 6
    on_demand_min_interval: int = 2


def _transition_to_dict(tr: Transition) -> Dict[str, Any]:
    return {
        "obs": tr.obs,
        "action": tr.action,
        "reward": float(tr.reward),
        "next_obs": tr.next_obs,
        "done": bool(tr.done),
        "truncated": bool(tr.truncated),
        "info": dict(tr.info or {}),
    }


class MultiAgentTrainer:
    """
    Run multiple agents in parallel, each in its own Verse instance,
    sharing a communication bus for real-time message passing.
    """

    def __init__(self, *, run_root: str = "runs", schema_version: str = "v1", auto_register_builtin: bool = True):
        self.run_root = run_root
        self.schema_version = schema_version
        if auto_register_builtin:
            register_builtin_verses()
            register_builtin_agents()

    def run(
        self,
        *,
        verse_specs: List[VerseSpec],
        agent_specs: List[AgentSpec],
        config: MARLConfig,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if len(verse_specs) != len(agent_specs):
            raise ValueError("verse_specs and agent_specs must have the same length")

        run = RunRef.create()
        bus = MessageBus()
        shared_pool = SharedMemoryPool() if bool(config.shared_memory_enabled) else None
        central_mem_cfg = (
            CentralMemoryConfig(root_dir=str(config.on_demand_memory_root))
            if bool(config.on_demand_memory_enabled)
            else None
        )

        verses = []
        agents = []
        verse_refs = []
        agent_refs = []

        for i, (vs, ag) in enumerate(zip(verse_specs, agent_specs)):
            spec_hash = _hash_verse_spec(vs)
            verse_ref = VerseRef.create(vs.verse_name, vs.verse_version, spec_hash)
            verse = create_verse(vs)

            agent_ref = AgentRef.create(policy_id=ag.policy_id, policy_version=ag.policy_version)
            agent = create_agent(spec=ag, observation_space=verse.observation_space, action_space=verse.action_space)

            # Optional: agents can subscribe to message bus
            if hasattr(agent, "on_message"):
                bus.subscribe(getattr(agent, "on_message"))

            verses.append(verse)
            agents.append(agent)
            verse_refs.append(verse_ref)
            agent_refs.append(agent_ref)

        log_cfg = EventLogConfig(root_dir=self.run_root, run_id=run.run_id)
        rollout_cfg = RolloutConfig(
            schema_version=self.schema_version,
            max_steps=config.max_steps,
            train=config.train,
            collect_transitions=config.collect_transitions,
        )

        with EventLogger(log_cfg) as logger:
            on_step = make_on_step_writer(logger)

            total_steps = 0
            total_return = 0.0

            for ep in range(config.episodes):
                obs_list = []
                done_list = []
                step_idx = [0 for _ in agents]
                episode_ids = []
                transitions: List[List[Transition]] = [[] for _ in agents]
                return_sums = [0.0 for _ in agents]
                negative_rewards = [0 for _ in agents]
                step_counts = [0 for _ in agents]
                memory_queries_used = [0 for _ in agents]
                last_memory_query_step = [-(10**9) for _ in agents]

                for idx, verse in enumerate(verses):
                    ep_seed = None if seed is None else seed + ep + idx
                    verse.seed(ep_seed)
                    agents[idx].seed(ep_seed)

                    reset = verse.reset()
                    obs_list.append(reset.obs)
                    done_list.append(False)
                    episode_ids.append(f"{run.run_id}_a{idx}_ep{ep}")
                    if shared_pool is not None and hasattr(agents[idx], "on_social_contract"):
                        try:
                            contract = shared_pool.safety_contract(verse_name=verse_specs[idx].verse_name)
                            agents[idx].on_social_contract(contract)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                for t in range(config.max_steps):
                    all_done = True
                    for idx, (verse, agent) in enumerate(zip(verses, agents)):
                        if done_list[idx]:
                            continue
                        all_done = False

                        query_budget = max(0, int(config.on_demand_query_budget))
                        memory_query_state: Dict[str, Any] = {
                            "enabled": bool(config.on_demand_memory_enabled),
                            "used": int(memory_queries_used[idx]),
                            "budget": int(query_budget),
                            "remaining": int(max(0, query_budget - int(memory_queries_used[idx]))),
                            "can_query": False,
                            "query_requested": False,
                            "query_executed": False,
                            "block_reason": "",
                            "last_query_step_idx": int(last_memory_query_step[idx]),
                        }
                        hint: Optional[Dict[str, Any]] = None
                        if (
                            central_mem_cfg is not None
                            and hasattr(agent, "memory_query_request")
                        ):
                            can_budget = bool(int(memory_queries_used[idx]) < int(query_budget))
                            can_cooldown = bool(
                                (int(step_idx[idx]) - int(last_memory_query_step[idx]))
                                >= max(1, int(config.on_demand_min_interval))
                            )
                            can_query = bool(can_budget and can_cooldown)
                            memory_query_state["can_query"] = bool(can_query)
                            if not can_budget:
                                memory_query_state["block_reason"] = "budget_exhausted"
                            elif not can_cooldown:
                                memory_query_state["block_reason"] = "cooldown"
                            else:
                                memory_query_state["block_reason"] = "ready"
                        else:
                            can_query = False
                            if bool(config.on_demand_memory_enabled) and not hasattr(agent, "memory_query_request"):
                                memory_query_state["block_reason"] = "agent_no_query_api"

                        if can_query:
                            req = None
                            try:
                                req = agent.memory_query_request(obs=obs_list[idx], step_idx=step_idx[idx])  # type: ignore[attr-defined]
                            except TypeError:
                                try:
                                    req = agent.memory_query_request(obs_list[idx])  # type: ignore[attr-defined]
                                except Exception:
                                    req = None
                            except Exception:
                                req = None
                            if isinstance(req, dict):
                                memory_query_state["query_requested"] = True
                                memory_query_state["query_reason"] = str(req.get("reason", "agent_request"))
                                try:
                                    raw_types = req.get("memory_types")
                                    memory_types = None
                                    if isinstance(raw_types, (list, set, tuple)):
                                        memory_types = set(
                                            str(x).strip().lower() for x in raw_types if str(x).strip()
                                        )
                                        if not memory_types:
                                            memory_types = None
                                    raw_families = req.get("memory_families")
                                    memory_families = None
                                    if isinstance(raw_families, (list, set, tuple)):
                                        memory_families = set(
                                            str(x).strip().lower() for x in raw_families if str(x).strip()
                                        )
                                        if not memory_families:
                                            memory_families = None
                                    query_obs = req.get("query_obs", obs_list[idx])
                                    matches = find_similar(
                                        obs=query_obs,
                                        cfg=central_mem_cfg,
                                        top_k=max(1, int(req.get("top_k", 3))),
                                        verse_name=(
                                            None
                                            if req.get("verse_name") in (None, "")
                                            else str(req.get("verse_name")).strip().lower()
                                        ),
                                        min_score=float(req.get("min_score", -1.0)),
                                        memory_families=memory_families,
                                        memory_types=memory_types,
                                    )
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
                                                "pointer_path": (
                                                    f"runs/{str(getattr(m, 'run_id', ''))}/events.jsonl"
                                                    f"#episode_id={str(getattr(m, 'episode_id', ''))};"
                                                    f"step_idx={int(getattr(m, 'step_idx', 0))}"
                                                ),
                                            }
                                        )
                                    hint = {
                                        "memory_recall": {
                                            "mode": "on_demand",
                                            "reason": str(req.get("reason", "agent_request")),
                                            "query_step_idx": int(step_idx[idx]),
                                            "query": {
                                                "memory_families": sorted(list(memory_families or set())),
                                                "memory_types": sorted(list(memory_types or set())),
                                            },
                                            "matches": rows,
                                            "match_count": int(len(rows)),
                                        }
                                    }
                                    memory_queries_used[idx] += 1
                                    last_memory_query_step[idx] = int(step_idx[idx])
                                    memory_query_state["query_executed"] = True
                                    memory_query_state["match_count"] = int(len(rows))
                                    memory_query_state["used"] = int(memory_queries_used[idx])
                                    memory_query_state["remaining"] = int(
                                        max(0, query_budget - int(memory_queries_used[idx]))
                                    )
                                    memory_query_state["last_query_step_idx"] = int(last_memory_query_step[idx])
                                    memory_query_state["block_reason"] = "executed"
                                    if hasattr(agent, "on_memory_response"):
                                        try:
                                            agent.on_memory_response(hint["memory_recall"])  # type: ignore[attr-defined]
                                        except Exception:
                                            pass
                                except Exception:
                                    memory_query_state["block_reason"] = "lookup_error"
                            else:
                                memory_query_state["block_reason"] = "agent_declined"

                        if hasattr(agent, "act_with_hint"):
                            action_result = agent.act_with_hint(obs_list[idx], hint)
                        else:
                            action_result = agent.act(obs_list[idx])
                        step = verse.step(action_result.action)
                        event_info = dict(step.info or {})
                        if isinstance(action_result.info, dict) and action_result.info:
                            action_info = dict(action_result.info)
                            selector_route = _selector_routing_telemetry(
                                action_info=action_info,
                                obs=obs_list[idx],
                                verse_name=verse_refs[idx].verse_name,
                            )
                            if selector_route is not None:
                                action_info["selector_routing"] = dict(selector_route)
                                event_info["selector_routing"] = dict(selector_route)
                            event_info["action_info"] = action_info
                        memory_query_state["used"] = int(memory_queries_used[idx])
                        memory_query_state["remaining"] = int(
                            max(0, query_budget - int(memory_queries_used[idx]))
                        )
                        event_info["memory_query"] = memory_query_state
                        if shared_pool is not None:
                            event_info["social_contract"] = shared_pool.safety_contract(
                                verse_name=verse_specs[idx].verse_name
                            )

                        event = make_step_event(
                            schema_version=rollout_cfg.schema_version,
                            run=run,
                            episode_id=episode_ids[idx],
                            step_idx=step_idx[idx],
                            agent=agent_refs[idx],
                            verse=verse_refs[idx],
                            obs=obs_list[idx],
                            action=action_result.action,
                            reward=step.reward,
                            done=step.done,
                            truncated=step.truncated,
                            seed=seed,
                            info=event_info,
                        )
                        on_step(event)

                        if rollout_cfg.collect_transitions:
                            transitions[idx].append(
                                Transition(
                                    obs=obs_list[idx],
                                    action=action_result.action,
                                    reward=step.reward,
                                    next_obs=step.obs,
                                    done=step.done,
                                    truncated=step.truncated,
                                    info={
                                        "env_info": step.info,
                                        "action_info": action_result.info,
                                    },
                                )
                            )

                        msg = Message(
                            sender_id=agent_refs[idx].agent_id,
                            topic="step",
                            payload={
                                "episode_id": episode_ids[idx],
                                "step_idx": step_idx[idx],
                                "reward": step.reward,
                                "done": bool(step.done or step.truncated),
                            },
                        )
                        bus.publish(msg)

                        if shared_pool is not None:
                            step_counts[idx] += 1
                            if float(step.reward) < -0.05:
                                negative_rewards[idx] += 1
                            action_info = action_result.info if isinstance(action_result.info, dict) else {}
                            concept = str(
                                action_info.get("social_concept", "risk" if float(step.reward) < 0.0 else "success")
                            )
                            token = str(
                                action_info.get(
                                    "social_token",
                                    f"{concept}_{int(action_result.action)}",
                                )
                            )
                            confidence = min(1.0, max(0.05, abs(float(step.reward))))
                            shared_pool.record_token(
                                agent_id=agent_refs[idx].agent_id,
                                concept=concept,
                                token=token,
                                confidence=confidence,
                            )

                        obs_list[idx] = step.obs
                        done_list[idx] = bool(step.done or step.truncated)
                        step_idx[idx] += 1
                        return_sums[idx] += float(step.reward)
                        total_steps += 1
                        total_return += float(step.reward)

                    if all_done:
                        break

                if rollout_cfg.train and rollout_cfg.collect_transitions:
                    for idx, agent in enumerate(agents):
                        if not transitions[idx]:
                            continue
                        if shared_pool is not None and hasattr(agent, "learn_from_shared"):
                            try:
                                offers = shared_pool.sample_trajectories(
                                    consumer_agent_id=agent_refs[idx].agent_id,
                                    verse_name=verse_specs[idx].verse_name,
                                    top_k=max(1, int(config.shared_memory_top_k)),
                                )
                                if offers:
                                    agent.learn_from_shared(offers)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        batch = ExperienceBatch(
                            transitions=transitions[idx],
                            meta={
                                "episode_id": episode_ids[idx],
                                "steps": step_idx[idx],
                                "return_sum": return_sums[idx],
                            },
                        )
                        try:
                            agent.learn(batch)
                        except NotImplementedError:
                            pass

                if shared_pool is not None:
                    for idx in range(len(agents)):
                        if not transitions[idx]:
                            continue
                        shared_pool.publish_trajectory(
                            provider_agent_id=agent_refs[idx].agent_id,
                            verse_name=verse_specs[idx].verse_name,
                            transitions=[_transition_to_dict(tr) for tr in transitions[idx]],
                            return_sum=float(return_sums[idx]),
                            success=bool(return_sums[idx] > 0.0),
                            episode_id=str(episode_ids[idx]),
                        )
                        if (ep + 1) % max(1, int(config.negotiation_interval)) == 0:
                            risk_budget = float(negative_rewards[idx] / float(max(1, step_counts[idx])))
                            confidence = 0.9 if float(return_sums[idx]) > 0.0 else 0.6
                            shared_pool.propose_safety_boundary(
                                agent_id=agent_refs[idx].agent_id,
                                verse_name=verse_specs[idx].verse_name,
                                risk_budget=risk_budget,
                                veto_bias=float(max(0.0, min(1.0, 1.0 - risk_budget))),
                                confidence=float(confidence),
                            )

            for verse in verses:
                verse.close()
            for agent in agents:
                agent.close()

        print("MARL run complete")
        print(f"run_id       : {run.run_id}")
        print(f"agents       : {len(agents)}")
        print(f"episodes     : {config.episodes}")
        print(f"total_steps  : {total_steps}")
        print(f"total_return : {total_return:.3f}")
        print(f"log_dir      : {self.run_root}/{run.run_id}")

        return {
            "run_id": run.run_id,
            "total_return": total_return,
            "total_steps": total_steps,
            "shared_memory": (
                {}
                if shared_pool is None
                else shared_pool.snapshot(min_support=max(1, int(config.lexicon_min_support)))
            ),
        }
