"""
agents/aware_agent.py

Awareness-augmented tabular Q-learning with Universal Strategic DNA.
Supports:
- Atomic cross-process memory and performance syncing
- Per-verse contribution and overlap tracking
- Global Z-score normalization for cross-verse comparison
- DNA Benefit tracking (reward delta from memory hits)
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from core.agent_base import ActionResult, ExperienceBatch
from core.taxonomy import memory_family_for_type, memory_type_for_verse
from core.types import AgentSpec, JSONValue, SpaceSpec

from agents.q_agent import QLearningAgent, obs_key
from memory.vector_store import InMemoryVectorStore, VectorRecord
from memory.embeddings import obs_to_vector


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clip01(value: object, default: float) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _aware_defaults_for_verse(verse_name: str) -> Dict[str, Any]:
    """Per-verse tuned defaults for AwareAgent."""
    verse = str(verse_name or "").strip().lower()
    if verse == "cliff_world":
        return {
            "lr": 0.08,
            "gamma": 0.98,
            "epsilon_decay": 0.997,
            "performance_floor": -20.0,
            "learn_success_bonus": 2.0,
            "learn_hazard_penalty": 2.0,
            "awareness_window": 30,
        }
    if verse == "labyrinth_world":
        return {
            "lr": 0.08,
            "gamma": 0.98,
            "epsilon_decay": 0.997,
            "performance_floor": -15.0,
            "learn_success_bonus": 1.5,
            "learn_hazard_penalty": 1.5,
            "awareness_window": 40,
        }
    if verse == "warehouse_world":
        return {
            "lr": 0.08,
            "gamma": 0.98,
            "epsilon_decay": 0.997,
            "performance_floor": -15.0,
            "learn_success_bonus": 1.5,
            "learn_hazard_penalty": 1.5,
            "awareness_window": 30,
        }
    return {}


class VersePerformanceMonitor:
    """Tracks running stats per verse to compute Z-scores."""
    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = {}

    def update(self, verse_name: str, return_sum: float):
        if verse_name not in self.stats:
            self.stats[verse_name] = {"mean": return_sum, "sq_sum": 0.0, "count": 1}
        else:
            s = self.stats[verse_name]
            s["count"] += 1
            old_mean = s["mean"]
            s["mean"] += (return_sum - old_mean) / s["count"]
            s["sq_sum"] += (return_sum - old_mean) * (return_sum - s["mean"])

    def get_z_score(self, verse_name: str, return_sum: float) -> float:
        s = self.stats.get(verse_name)
        if not s or s["count"] < 2: return 0.0
        std = math.sqrt(s["sq_sum"] / (s["count"] - 1))
        return (return_sum - s["mean"]) / (std + 1e-8)

    def to_dict(self) -> Dict[str, Any]: return self.stats
    def from_dict(self, data: Dict[str, Any]): self.stats = data


class AwareAgent(QLearningAgent):
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        cfg_in = dict(spec.config) if isinstance(spec.config, dict) else {}
        verse_name = str(cfg_in.get("verse_name", "unknown"))

        # Apply per-verse tuned defaults before super().__init__ so Q-agent
        # picks up the tuned lr/gamma/epsilon_decay/shaping values.
        tuned = _aware_defaults_for_verse(verse_name)
        if tuned:
            merged = dict(cfg_in)
            for k, v in tuned.items():
                merged.setdefault(k, v)
            spec = spec.evolved(config=merged)

        super().__init__(spec=spec, observation_space=observation_space, action_space=action_space)

        cfg = spec.config if isinstance(spec.config, dict) else {}
        self._verse_name = str(cfg.get("verse_name", "unknown"))
        self._awareness_window = max(1, _safe_int(cfg.get("awareness_window", 20), 20))
        self._novelty_explore_bonus = _safe_float(cfg.get("novelty_explore_bonus", 0.25), 0.25)
        self._performance_explore_boost = _safe_float(cfg.get("performance_explore_boost", 0.20), 0.20)
        self._performance_floor = _safe_float(cfg.get("performance_floor", 0.0), 0.0)
        self._min_aware_epsilon = _safe_float(cfg.get("min_aware_epsilon", self.epsilon_min), self.epsilon_min)
        
        self._memory_sync_path = cfg.get("memory_sync_path")
        self._sync_interval_steps = _safe_int(cfg.get("sync_interval_steps", 100), 100)
        self._steps_since_sync = 0

        self._use_vector_memory = bool(cfg.get("use_vector_memory", True))
        self._vector_memory_top_k = _safe_int(cfg.get("vector_memory_top_k", 5), 5)
        self._vector_memory_weight = _safe_float(cfg.get("vector_memory_weight", 0.4), 0.4)
        self._vector_memory_mode = str(cfg.get("vector_memory_mode", "declarative")).strip().lower()
        if self._vector_memory_mode not in {"all", "procedural", "declarative"}:
            self._vector_memory_mode = "declarative"
        self._declarative_fact_weight = max(
            0.0,
            min(2.0, _safe_float(cfg.get("declarative_fact_weight", 0.35), 0.35)),
        )
        self._vector_store = InMemoryVectorStore()

        self._shared_dna: Dict[str, Dict[str, int]] = {}
        self._recent_returns: deque[float] = deque(maxlen=self._awareness_window)
        self._perf_monitor = VersePerformanceMonitor()
        self._social_contract: Dict[str, Any] = {
            "verse_name": str(self._verse_name).strip().lower(),
            "has_contract": False,
            "risk_budget": 0.25,
            "veto_bias": 0.50,
            "confidence": 0.0,
            "support": 0,
        }
        self._social_eps_cap = 1.0
        self._shared_hint_weight = max(0.0, min(3.0, _safe_float(cfg.get("shared_hint_weight", 0.40), 0.40)))
        self._shared_hint_decay = max(0.0, min(1.0, _safe_float(cfg.get("shared_hint_decay", 0.98), 0.98)))
        self._shared_action_hints: Dict[str, np.ndarray] = {}
        self._shared_offer_updates = 0
        self._sync_failures = 0
        self._last_sync_error = ""
        
        self._memory_hits = 0
        self._memory_queries = 0
        self._dna_benefit_sum = 0.0
        self._dna_hits_this_episode = 0

        self._sync_memory()

    def _coarse_context_key(self, obs: Dict[str, Any]) -> str:
        if all(k in obs for k in ("score_delta", "pressure", "risk")):
            score = _safe_int(obs.get("score_delta"), 0)
            pressure = _safe_int(obs.get("pressure"), 0)
            risk = _safe_int(obs.get("risk"), 0)
            return json.dumps({"d": "strat", "s": max(-4, min(4, score//2)), "p": max(0, min(8, pressure//2)), "r": max(0, min(8, risk//2))}, sort_keys=True)
        if all(k in obs for k in ("x", "y", "goal_x", "goal_y")):
            dx = _safe_int(obs.get("goal_x", 0), 0) - _safe_int(obs.get("x", 0), 0)
            dy = _safe_int(obs.get("goal_y", 0), 0) - _safe_int(obs.get("y", 0), 0)
            return json.dumps({"d": "nav", "dx": int(math.copysign(1, dx)) if dx != 0 else 0, "dy": int(math.copysign(1, dy)) if dy != 0 else 0}, sort_keys=True)
        return "unknown"

    def _extract_declarative_facts(self, obs: JSONValue, info: Optional[Dict[str, Any]] = None) -> List[str]:
        facts: List[str] = []
        if isinstance(obs, dict):
            for k in sorted(obs.keys()):
                v = obs.get(k)
                if isinstance(v, bool):
                    facts.append(f"{k}={1 if v else 0}")
                elif isinstance(v, int):
                    facts.append(f"{k}={v}")
                elif isinstance(v, float):
                    facts.append(f"{k}={round(float(v), 3)}")
                elif isinstance(v, str):
                    facts.append(f"{k}={v}")
                elif isinstance(v, list) and len(v) <= 4 and all(isinstance(x, (int, float, bool)) for x in v):
                    vals = ",".join(str(int(x) if isinstance(x, bool) else round(float(x), 3)) for x in v)
                    facts.append(f"{k}=[{vals}]")
        if isinstance(info, dict):
            for key in ("rule_flipped", "goal_hint_visible", "battery_depleted", "hit_laser", "fell_pit", "hit_wall"):
                if key in info:
                    facts.append(f"info:{key}={1 if bool(info.get(key)) else 0}")
        return facts[:32]

    def _fact_overlap(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        sa = set(str(x) for x in a if str(x).strip())
        sb = set(str(x) for x in b if str(x).strip())
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        denom = len(sa | sb)
        if denom <= 0:
            return 0.0
        return float(inter / float(denom))

    def act(self, obs: JSONValue) -> ActionResult:
        k = obs_key(obs)
        ck = self._coarse_context_key(obs) if isinstance(obs, dict) else "unknown"
        qvals = self._get_q(k)
        
        dna_entry = self._shared_dna.get(ck, {})
        is_shared = len(dna_entry) > 1
        
        memory_prior = self._query_vector_memory(obs) if self._use_vector_memory else None
        q_eval = np.asarray(qvals, dtype=np.float32)
        if memory_prior is not None:
            q_eval += self._vector_memory_weight * memory_prior
            self._dna_hits_this_episode += 1

        global_visits = sum(dna_entry.values())
        novelty_bonus = self._novelty_explore_bonus / math.sqrt(1.0 + float(global_visits))
        recent_mean = sum(self._recent_returns) / len(self._recent_returns) if self._recent_returns else 0.0
        performance_boost = self._performance_explore_boost * min(1.0, max(0.0, self._performance_floor - recent_mean))

        aware_epsilon = min(1.0, max(self._min_aware_epsilon, float(self.stats.epsilon) + novelty_bonus + performance_boost))
        # Respect explicit deterministic configs: if caller pins epsilon to zero,
        # disable awareness-driven exploration boosts.
        if float(self.stats.epsilon) <= 0.0 and float(self._min_aware_epsilon) <= 0.0:
            aware_epsilon = 0.0
        if bool(self._social_contract.get("has_contract", False)):
            aware_epsilon = min(float(aware_epsilon), float(self._social_eps_cap))
        
        if ck not in self._shared_dna: self._shared_dna[ck] = {}
        self._shared_dna[ck][self._verse_name] = self._shared_dna[ck].get(self._verse_name, 0) + 1

        social_hint = self._shared_action_hints.get(ck)
        if social_hint is not None:
            hint_max = float(np.max(social_hint)) if social_hint.size else 0.0
            if hint_max > 0.0:
                q_eval += float(self._shared_hint_weight) * (social_hint / hint_max)

        if self._rng.random() < aware_epsilon:
            a = int(self._rng.integers(0, self.n_actions))
            mode = "aware_explore"
        else:
            a = int(np.argmax(q_eval))
            mode = "aware_exploit"

        return ActionResult(
            action=a,
            info={
                "mode": mode,
                "aware_epsilon": float(aware_epsilon),
                "dna_shared": is_shared,
                "dna_hit": memory_prior is not None,
                "social_contract_active": bool(self._social_contract.get("has_contract", False)),
                "social_eps_cap": float(self._social_eps_cap),
                "shared_hint_used": bool(social_hint is not None and float(np.max(social_hint)) > 0.0),
            },
        )

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        metrics = super().learn(batch)
        ret = batch.meta.get("return_sum", 0.0)
        self._recent_returns.append(float(ret))
        self._perf_monitor.update(self._verse_name, float(ret))

        # Populate vector memory from observed transitions so _query_vector_memory
        # can retrieve relevant past experience for unseen-but-similar states.
        if self._use_vector_memory and batch.transitions:
            new_records = []
            verse_mt = memory_type_for_verse(self._verse_name)
            verse_family = memory_family_for_type(verse_mt)
            for idx, tr in enumerate(batch.transitions):
                vec = obs_to_vector(tr.obs)
                if vec:
                    rec_id = f"{id(batch)}_{idx}"
                    tr_info = tr.info if isinstance(tr.info, dict) else {}
                    new_records.append(VectorRecord(
                        vector_id=rec_id,
                        vector=vec,
                        metadata={
                            "action": int(tr.action),
                            "reward": float(tr.reward),
                            "done": bool(tr.done),
                            "verse_name": str(self._verse_name),
                            "memory_type": str(verse_mt),
                            "memory_family": str(verse_family),
                            "facts": self._extract_declarative_facts(tr.obs, tr_info),
                        },
                    ))
            if new_records:
                self._vector_store.add(new_records)

        # Track DNA benefit: if we had many hits, did we get a better return?
        if self._dna_hits_this_episode > 0:
            self._dna_benefit_sum += float(ret) / self._dna_hits_this_episode

        self._steps_since_sync += len(batch.transitions)
        if self._steps_since_sync >= self._sync_interval_steps:
            self._sync_memory()
            self._steps_since_sync = 0

        total_states = len(self._shared_dna)
        shared_states = sum(1 for d in self._shared_dna.values() if len(d) > 1)
        
        metrics.update({
            "dna_total_states": total_states,
            "dna_shared_states": shared_states,
            "dna_overlap_pct": (shared_states / total_states * 100) if total_states > 0 else 0,
            "dna_benefit_avg": self._dna_benefit_sum / max(1, self.stats.updates),
            "return_z_score": self._perf_monitor.get_z_score(self._verse_name, float(ret))
        })
        self._dna_hits_this_episode = 0
        return metrics

    def on_social_contract(self, contract: Dict[str, Any]) -> None:
        if not isinstance(contract, dict):
            return
        verse = str(contract.get("verse_name", "")).strip().lower()
        if verse and self._verse_name and verse != str(self._verse_name).strip().lower():
            return

        has_contract = bool(contract.get("has_contract", False))
        risk_budget = _clip01(contract.get("risk_budget", 0.25), 0.25)
        veto_bias = _clip01(contract.get("veto_bias", 0.50), 0.50)
        confidence = _clip01(contract.get("confidence", 0.0), 0.0)
        support = max(0, _safe_int(contract.get("support", 0), 0))
        support_factor = 1.0 if support <= 0 else min(1.0, float(support) / 3.0)
        caution = max(0.0, min(1.0, (float(risk_budget) + float(veto_bias)) * 0.5))

        self._social_contract = {
            "verse_name": (verse if verse else str(self._verse_name).strip().lower()),
            "has_contract": bool(has_contract),
            "risk_budget": float(risk_budget),
            "veto_bias": float(veto_bias),
            "confidence": float(confidence),
            "support": int(support),
        }
        if not bool(has_contract):
            self._social_eps_cap = 1.0
            return
        cap = 1.0 - (0.85 * float(caution) * float(confidence) * float(support_factor))
        self._social_eps_cap = max(0.02, min(1.0, float(cap)))

    def learn_from_shared(self, offers: List[Dict[str, Any]]) -> Dict[str, JSONValue]:
        if not isinstance(offers, list) or not offers:
            return {"shared_offers": 0, "shared_updates": 0, "shared_vector_records": 0}

        updates = 0
        vector_records: List[VectorRecord] = []
        decayed_contexts: set[str] = set()

        for offer_idx, offer in enumerate(offers):
            if not isinstance(offer, dict):
                continue
            provider = str(offer.get("provider_agent_id", "")).strip().lower() or "shared_provider"
            provider_trust = _clip01(offer.get("provider_trust", 0.5), 0.5)
            offer_verse = str(offer.get("verse_name", self._verse_name)).strip().lower() or str(self._verse_name).strip().lower()
            memory_type = memory_type_for_verse(offer_verse)
            memory_family = memory_family_for_type(memory_type)
            transitions = offer.get("transitions")
            if not isinstance(transitions, list):
                continue

            for tr_idx, tr in enumerate(transitions):
                if not isinstance(tr, dict):
                    continue
                obs = tr.get("obs")
                if not isinstance(obs, dict):
                    continue
                action = _safe_int(tr.get("action", -1), -1)
                if action < 0 or action >= int(self.n_actions):
                    continue
                reward = _safe_float(tr.get("reward", 0.0), 0.0)
                ck = self._coarse_context_key(obs)
                hint = self._shared_action_hints.get(ck)
                if hint is None:
                    hint = np.zeros((self.n_actions,), dtype=np.float32)
                    self._shared_action_hints[ck] = hint
                if ck not in decayed_contexts:
                    hint *= float(self._shared_hint_decay)
                    decayed_contexts.add(ck)

                reward_term = 0.25 + max(0.0, float(reward))
                weight = max(0.05, float(provider_trust)) * float(reward_term)
                hint[int(action)] += float(weight)

                if ck not in self._shared_dna:
                    self._shared_dna[ck] = {}
                self._shared_dna[ck][provider] = self._shared_dna[ck].get(provider, 0) + 1

                if self._use_vector_memory:
                    vec = obs_to_vector(obs)
                    if vec:
                        vector_records.append(
                            VectorRecord(
                                vector_id=f"shared_{provider}_{self._shared_offer_updates}_{offer_idx}_{tr_idx}",
                                vector=vec,
                                metadata={
                                    "action": int(action),
                                    "reward": float(reward),
                                    "done": bool(tr.get("done", False)),
                                    "verse_name": str(offer_verse),
                                    "memory_type": str(memory_type),
                                    "memory_family": str(memory_family),
                                    "facts": self._extract_declarative_facts(obs, tr.get("info") if isinstance(tr.get("info"), dict) else None),
                                    "source": "shared_offer",
                                    "provider_agent_id": provider,
                                    "provider_trust": float(provider_trust),
                                },
                            )
                        )
                updates += 1

        if vector_records:
            self._vector_store.add(vector_records)
        self._shared_offer_updates += int(updates)
        return {
            "shared_offers": int(len(offers)),
            "shared_updates": int(updates),
            "shared_vector_records": int(len(vector_records)),
        }

    def _record_sync_error(self, msg: str) -> None:
        self._sync_failures += 1
        self._last_sync_error = str(msg)

    def _sync_memory(self):
        if not self._memory_sync_path:
            return
        temp_path = self._memory_sync_path + ".tmp"
        shared_payload: Dict[str, Any] = {}
        if os.path.exists(self._memory_sync_path):
            try:
                with open(self._memory_sync_path, "r") as f:
                    shared_payload = json.load(f)
                if not isinstance(shared_payload, dict):
                    shared_payload = {}
            except json.JSONDecodeError as exc:
                self._record_sync_error(f"decode_error:{exc}")
                shared_payload = {}
            except OSError as exc:
                self._record_sync_error(f"read_error:{exc}")
                return

        shared_dna = shared_payload.get("dna", {})
        shared_perf = shared_payload.get("perf", {})
        if not isinstance(shared_dna, dict):
            shared_dna = {}
        if not isinstance(shared_perf, dict):
            shared_perf = {}

        # Merge DNA
        for k, local_counts in self._shared_dna.items():
            if k not in shared_dna:
                shared_dna[k] = local_counts
            else:
                cur = shared_dna[k] if isinstance(shared_dna.get(k), dict) else {}
                for v_name, count in local_counts.items():
                    cur[v_name] = max(count, cur.get(v_name, 0))
                shared_dna[k] = cur

        # Merge Perf
        self._perf_monitor.from_dict(shared_perf)  # Simple overwrite for now

        self._shared_dna = shared_dna
        try:
            with open(temp_path, "w") as f:
                json.dump({"dna": shared_dna, "perf": self._perf_monitor.to_dict()}, f)
            os.replace(temp_path, self._memory_sync_path)
        except OSError as exc:
            self._record_sync_error(f"write_error:{exc}")

    def save(self, path: str) -> None:
        super().save(path)
        aware_state = {
            "shared_dna": self._shared_dna,
            "perf_monitor": self._perf_monitor.to_dict(),
            "recent_returns": list(self._recent_returns),
            "memory_hits": int(self._memory_hits),
            "memory_queries": int(self._memory_queries),
            "dna_benefit_sum": float(self._dna_benefit_sum),
            "vector_memory_mode": str(self._vector_memory_mode),
            "declarative_fact_weight": float(self._declarative_fact_weight),
            "social_contract": dict(self._social_contract),
            "social_eps_cap": float(self._social_eps_cap),
            "shared_hint_weight": float(self._shared_hint_weight),
            "shared_hint_decay": float(self._shared_hint_decay),
            "shared_offer_updates": int(self._shared_offer_updates),
            "sync_failures": int(self._sync_failures),
            "last_sync_error": str(self._last_sync_error),
            "shared_action_hints": {k: v.tolist() for k, v in self._shared_action_hints.items()},
            "vector_records": [
                {"vector_id": r.vector_id, "vector": r.vector, "metadata": r.metadata}
                for r in self._vector_store._records
            ],
        }
        fp = os.path.join(path, "aware_state.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(aware_state, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        super().load(path)
        fp = os.path.join(path, "aware_state.json")
        if not os.path.isfile(fp):
            return
        with open(fp, "r", encoding="utf-8") as f:
            state = json.load(f)
        if isinstance(state.get("shared_dna"), dict):
            self._shared_dna = state["shared_dna"]
        if isinstance(state.get("perf_monitor"), dict):
            self._perf_monitor.from_dict(state["perf_monitor"])
        if isinstance(state.get("recent_returns"), list):
            self._recent_returns = deque(state["recent_returns"], maxlen=self._awareness_window)
        self._memory_hits = _safe_int(state.get("memory_hits", 0), 0)
        self._memory_queries = _safe_int(state.get("memory_queries", 0), 0)
        self._dna_benefit_sum = _safe_float(state.get("dna_benefit_sum", 0.0), 0.0)
        mode = str(state.get("vector_memory_mode", self._vector_memory_mode)).strip().lower()
        if mode in {"all", "procedural", "declarative"}:
            self._vector_memory_mode = mode
        self._declarative_fact_weight = max(
            0.0,
            min(2.0, _safe_float(state.get("declarative_fact_weight", self._declarative_fact_weight), self._declarative_fact_weight)),
        )
        if isinstance(state.get("social_contract"), dict):
            self._social_contract = dict(state["social_contract"])
        self._social_eps_cap = max(0.02, min(1.0, _safe_float(state.get("social_eps_cap", self._social_eps_cap), self._social_eps_cap)))
        self._shared_hint_weight = max(0.0, min(3.0, _safe_float(state.get("shared_hint_weight", self._shared_hint_weight), self._shared_hint_weight)))
        self._shared_hint_decay = max(0.0, min(1.0, _safe_float(state.get("shared_hint_decay", self._shared_hint_decay), self._shared_hint_decay)))
        self._shared_offer_updates = max(0, _safe_int(state.get("shared_offer_updates", self._shared_offer_updates), self._shared_offer_updates))
        self._sync_failures = max(0, _safe_int(state.get("sync_failures", self._sync_failures), self._sync_failures))
        self._last_sync_error = str(state.get("last_sync_error", self._last_sync_error))
        hints = state.get("shared_action_hints", {})
        if isinstance(hints, dict):
            restored: Dict[str, np.ndarray] = {}
            for k, arr in hints.items():
                if not isinstance(k, str) or not isinstance(arr, list):
                    continue
                try:
                    vals = np.asarray(arr, dtype=np.float32).flatten()
                except Exception:
                    continue
                if vals.size != int(self.n_actions):
                    continue
                restored[k] = vals
            if restored:
                self._shared_action_hints = restored
        # Restore vector memory records.
        recs = state.get("vector_records", [])
        if isinstance(recs, list) and recs:
            self._vector_store._records = [
                VectorRecord(
                    vector_id=str(r.get("vector_id", "")),
                    vector=list(r.get("vector", [])),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in recs if isinstance(r, dict)
            ]

    def _query_vector_memory(self, obs: JSONValue) -> Optional[np.ndarray]:
        self._memory_queries += 1
        qvec = obs_to_vector(obs)
        if not qvec:
            return None
        matches = self._vector_store.query(qvec, top_k=self._vector_memory_top_k)
        if not matches:
            return None
        query_facts = self._extract_declarative_facts(obs, None)
        prior = np.zeros(self.n_actions, dtype=np.float32)
        used = 0
        for m in matches:
            md = m.metadata if isinstance(m.metadata, dict) else {}
            fam = str(md.get("memory_family", "unknown")).strip().lower()
            if self._vector_memory_mode == "procedural" and fam not in {"procedural", "hybrid", "unknown"}:
                continue
            if self._vector_memory_mode == "declarative" and fam not in {"declarative", "hybrid", "unknown"}:
                continue
            a = _safe_int(md.get("action", -1), -1)
            if not (0 <= a < self.n_actions):
                continue
            support = max(0.0, float(m.score))
            if self._vector_memory_mode == "declarative":
                mem_facts = md.get("facts", [])
                mem_facts = [str(x) for x in mem_facts] if isinstance(mem_facts, list) else []
                overlap = self._fact_overlap(query_facts, mem_facts)
                support *= 1.0 + float(self._declarative_fact_weight) * float(overlap)
            prior[a] += float(support)
            used += 1
        if used <= 0:
            return None
        self._memory_hits += 1
        total = np.sum(prior)
        return prior / total if total > 0 else None
