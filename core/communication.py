"""
core/communication.py

Minimal communication primitives for multi-agent training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from core.types import JSONValue


@dataclass
class Message:
    sender_id: str
    topic: str
    payload: Dict[str, JSONValue]


class MessageBus:
    """
    Simple in-process publish/subscribe bus.
    """

    def __init__(self) -> None:
        self._subs: List[Callable[[Message], None]] = []

    def subscribe(self, fn: Callable[[Message], None]) -> None:
        self._subs.append(fn)

    def publish(self, msg: Message) -> None:
        for fn in list(self._subs):
            fn(msg)


@dataclass
class SharedTrajectory:
    provider_agent_id: str
    verse_name: str
    return_sum: float
    success: bool
    steps: int
    created_at_ms: int
    transitions: List[Dict[str, JSONValue]]
    episode_id: str = ""


class SharedMemoryPool:
    """
    Shared pool used by multi-agent runs:
    - trade successful trajectories,
    - negotiate safety boundaries,
    - track emergent communication tokens.
    """

    def __init__(
        self,
        *,
        max_total_trajectories: int = 500,
        max_per_verse: int = 120,
        max_provider_share: float = 0.35,
        trajectory_half_life_hours: float = 24.0,
        lexicon_consensus_floor: float = 0.20,
    ):
        self.max_total_trajectories = max(10, int(max_total_trajectories))
        self.max_per_verse = max(5, int(max_per_verse))
        self.max_provider_share = max(0.05, min(0.95, float(max_provider_share)))
        self.trajectory_half_life_hours = max(0.5, float(trajectory_half_life_hours))
        self.lexicon_consensus_floor = max(0.0, min(1.0, float(lexicon_consensus_floor)))
        self._trajectories: Dict[str, List[SharedTrajectory]] = {}
        self._safety_proposals: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._lexicon: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._provider_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    def publish_trajectory(
        self,
        *,
        provider_agent_id: str,
        verse_name: str,
        transitions: List[Dict[str, JSONValue]],
        return_sum: float,
        success: bool,
        episode_id: str = "",
    ) -> None:
        verse = str(verse_name).strip().lower()
        rec = SharedTrajectory(
            provider_agent_id=str(provider_agent_id),
            verse_name=verse,
            return_sum=float(return_sum),
            success=bool(success),
            steps=int(len(transitions)),
            created_at_ms=int(time.time() * 1000),
            transitions=list(transitions),
            episode_id=str(episode_id),
        )
        bucket = self._trajectories.setdefault(verse, [])
        bucket.append(rec)
        self._update_provider_stats(verse=verse, rec=rec)
        if len(bucket) > int(self.max_per_verse):
            self._trim_verse(verse=verse)
        self._enforce_provider_quota(verse=verse)
        self._trim_global()

    def sample_trajectories(
        self,
        *,
        consumer_agent_id: str,
        verse_name: Optional[str] = None,
        top_k: int = 5,
        min_return: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        wanted_verse = str(verse_name).strip().lower() if verse_name else ""
        rows: List[SharedTrajectory] = []
        for verse, bucket in self._trajectories.items():
            if wanted_verse and verse != wanted_verse:
                continue
            rows.extend(bucket)
        if min_return is not None:
            rows = [r for r in rows if float(r.return_sum) >= float(min_return)]
        rows = [r for r in rows if str(r.provider_agent_id) != str(consumer_agent_id)]
        now_ms = int(time.time() * 1000)
        rows.sort(key=lambda x: self._trajectory_rank_score(x, now_ms=now_ms), reverse=True)
        out: List[Dict[str, Any]] = []
        for r in rows[: max(1, int(top_k))]:
            trust = self._provider_trust(verse_name=r.verse_name, provider_agent_id=r.provider_agent_id)
            out.append(
                {
                    "provider_agent_id": r.provider_agent_id,
                    "verse_name": r.verse_name,
                    "return_sum": float(r.return_sum),
                    "success": bool(r.success),
                    "steps": int(r.steps),
                    "created_at_ms": int(r.created_at_ms),
                    "episode_id": r.episode_id,
                    "transitions": list(r.transitions),
                    "provider_trust": float(trust),
                }
            )
        return out

    def propose_safety_boundary(
        self,
        *,
        agent_id: str,
        verse_name: str,
        risk_budget: float,
        veto_bias: float,
        confidence: float,
    ) -> None:
        verse = str(verse_name).strip().lower()
        by_agent = self._safety_proposals.setdefault(verse, {})
        by_agent[str(agent_id)] = {
            "risk_budget": max(0.0, min(1.0, float(risk_budget))),
            "veto_bias": max(0.0, min(1.0, float(veto_bias))),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "t_ms": int(time.time() * 1000),
        }

    def safety_contract(self, *, verse_name: str) -> Dict[str, Any]:
        verse = str(verse_name).strip().lower()
        proposals = list((self._safety_proposals.get(verse) or {}).values())
        if not proposals:
            return {
                "verse_name": verse,
                "has_contract": False,
                "risk_budget": 0.25,
                "veto_bias": 0.50,
                "support": 0,
            }
        risk_budget = sum(float(p.get("risk_budget", 0.25)) for p in proposals) / float(len(proposals))
        veto_bias = sum(float(p.get("veto_bias", 0.50)) for p in proposals) / float(len(proposals))
        confidence = sum(float(p.get("confidence", 0.0)) for p in proposals) / float(len(proposals))
        return {
            "verse_name": verse,
            "has_contract": True,
            "risk_budget": float(risk_budget),
            "veto_bias": float(veto_bias),
            "confidence": float(confidence),
            "support": int(len(proposals)),
        }

    def record_token(
        self,
        *,
        agent_id: str,
        concept: str,
        token: str,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        c = str(concept).strip().lower()
        t = str(token).strip().lower()
        if not c or not t:
            return {"concept": c, "token": t, "accepted": False}
        tok_bucket = self._lexicon.setdefault(c, {}).setdefault(t, {"votes": 0.0, "agents": {}})
        tok_bucket["votes"] = float(tok_bucket.get("votes", 0.0)) + max(0.05, min(1.0, float(confidence)))
        agents = tok_bucket.get("agents")
        if not isinstance(agents, dict):
            agents = {}
            tok_bucket["agents"] = agents
        agents[str(agent_id)] = float(max(0.05, min(1.0, float(confidence))))
        return {"concept": c, "token": t, "accepted": True}

    def lexicon(self, *, min_support: int = 2) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        support_min = max(1, int(min_support))
        for concept, tok_map in self._lexicon.items():
            best_token = ""
            best_votes = -1.0
            best_support = 0
            best_consensus = 0.0
            concept_agents: set[str] = set()
            for rec in tok_map.values():
                agents = rec.get("agents") if isinstance(rec.get("agents"), dict) else {}
                for aid in agents.keys():
                    concept_agents.add(str(aid))
            concept_agent_count = max(1, len(concept_agents))
            for token, rec in tok_map.items():
                agents = rec.get("agents") if isinstance(rec.get("agents"), dict) else {}
                support = len(agents)
                votes = float(rec.get("votes", 0.0))
                consensus_ratio = float(support / float(concept_agent_count))
                if consensus_ratio < float(self.lexicon_consensus_floor):
                    continue
                if (
                    consensus_ratio > best_consensus
                    or (consensus_ratio == best_consensus and support > best_support)
                    or (consensus_ratio == best_consensus and support == best_support and votes > best_votes)
                ):
                    best_token = str(token)
                    best_votes = float(votes)
                    best_support = int(support)
                    best_consensus = float(consensus_ratio)
            if best_support >= support_min and best_token:
                out[concept] = {
                    "token": best_token,
                    "votes": float(best_votes),
                    "support": int(best_support),
                    "consensus_ratio": float(best_consensus),
                }
        return out

    def snapshot(self, *, min_support: int = 2) -> Dict[str, Any]:
        trajectory_count = 0
        by_verse: Dict[str, int] = {}
        for verse, bucket in self._trajectories.items():
            by_verse[str(verse)] = int(len(bucket))
            trajectory_count += int(len(bucket))
        contracts = {
            verse: self.safety_contract(verse_name=verse)
            for verse in sorted(self._safety_proposals.keys())
        }
        provider_share: Dict[str, float] = {}
        for verse, bucket in self._trajectories.items():
            total = max(1, len(bucket))
            counts: Dict[str, int] = {}
            for rec in bucket:
                pid = str(rec.provider_agent_id)
                counts[pid] = int(counts.get(pid, 0)) + 1
            top_share = 0.0
            for cnt in counts.values():
                top_share = max(top_share, float(cnt / float(total)))
            provider_share[str(verse)] = float(top_share)
        return {
            "trajectory_count": int(trajectory_count),
            "trajectories_by_verse": by_verse,
            "safety_contracts": contracts,
            "emergent_lexicon": self.lexicon(min_support=min_support),
            "max_provider_share_by_verse": provider_share,
        }

    def _trim_global(self) -> None:
        rows: List[tuple[str, SharedTrajectory]] = []
        for verse, bucket in self._trajectories.items():
            for rec in bucket:
                rows.append((verse, rec))
        if len(rows) <= int(self.max_total_trajectories):
            return
        now_ms = int(time.time() * 1000)
        rows.sort(key=lambda x: self._trajectory_rank_score(x[1], now_ms=now_ms), reverse=True)
        keep = rows[: int(self.max_total_trajectories)]
        allowed = {(verse, id(rec)) for verse, rec in keep}
        for verse in list(self._trajectories.keys()):
            bucket = self._trajectories[verse]
            self._trajectories[verse] = [r for r in bucket if (verse, id(r)) in allowed]
            self._enforce_provider_quota(verse=verse)

    def _trim_verse(self, *, verse: str) -> None:
        bucket = self._trajectories.get(verse) or []
        if not bucket:
            return
        now_ms = int(time.time() * 1000)
        bucket.sort(key=lambda x: self._trajectory_rank_score(x, now_ms=now_ms), reverse=True)
        del bucket[int(self.max_per_verse) :]

    def _provider_cap(self) -> int:
        return max(1, int(round(float(self.max_per_verse) * float(self.max_provider_share))))

    def _enforce_provider_quota(self, *, verse: str) -> None:
        bucket = self._trajectories.get(verse) or []
        if not bucket:
            return
        cap = self._provider_cap()
        now_ms = int(time.time() * 1000)
        ranked = sorted(bucket, key=lambda x: self._trajectory_rank_score(x, now_ms=now_ms), reverse=True)
        kept: List[SharedTrajectory] = []
        by_provider: Dict[str, int] = {}
        for rec in ranked:
            pid = str(rec.provider_agent_id)
            n = int(by_provider.get(pid, 0))
            if n >= cap:
                continue
            kept.append(rec)
            by_provider[pid] = n + 1
            if len(kept) >= int(self.max_per_verse):
                break
        self._trajectories[str(verse)] = kept

    def _update_provider_stats(self, *, verse: str, rec: SharedTrajectory) -> None:
        by_agent = self._provider_stats.setdefault(str(verse), {})
        provider = str(rec.provider_agent_id)
        row = by_agent.setdefault(
            provider,
            {
                "n": 0.0,
                "success_n": 0.0,
                "return_sum": 0.0,
                "return_ema": 0.0,
                "last_t_ms": 0.0,
            },
        )
        row["n"] = float(row.get("n", 0.0)) + 1.0
        row["success_n"] = float(row.get("success_n", 0.0)) + (1.0 if bool(rec.success) else 0.0)
        row["return_sum"] = float(row.get("return_sum", 0.0)) + float(rec.return_sum)
        old_ema = float(row.get("return_ema", 0.0))
        row["return_ema"] = (0.85 * old_ema) + (0.15 * float(rec.return_sum))
        row["last_t_ms"] = float(rec.created_at_ms)

    def _provider_trust(self, *, verse_name: str, provider_agent_id: str) -> float:
        by_agent = self._provider_stats.get(str(verse_name), {})
        row = by_agent.get(str(provider_agent_id), {})
        n = max(0.0, float(row.get("n", 0.0)))
        if n <= 0.0:
            return 0.5
        success_rate = float(row.get("success_n", 0.0) / max(1.0, n))
        mean_return = float(row.get("return_sum", 0.0) / max(1.0, n))
        reliability = min(1.0, n / 12.0)
        mean_return_norm = max(0.0, min(1.0, (mean_return + 2.0) / 4.0))
        return max(0.05, min(1.0, (0.55 * success_rate) + (0.30 * mean_return_norm) + (0.15 * reliability)))

    def _trajectory_rank_score(self, rec: SharedTrajectory, *, now_ms: Optional[int] = None) -> float:
        now = int(now_ms) if now_ms is not None else int(time.time() * 1000)
        age_hours = max(0.0, float(now - int(rec.created_at_ms)) / 3_600_000.0)
        freshness = math.exp(-age_hours / float(self.trajectory_half_life_hours))
        step_scale = max(1.0, float(rec.steps))
        return_norm = max(-1.0, min(1.0, float(rec.return_sum) / step_scale))
        success_term = 1.0 if bool(rec.success) else 0.0
        trust = self._provider_trust(verse_name=rec.verse_name, provider_agent_id=rec.provider_agent_id)
        base = (0.70 * success_term) + (0.30 * ((return_norm + 1.0) * 0.5))
        return float(base * freshness * (0.65 + 0.35 * trust))


class CommunicatingAgent(Protocol):
    """
    Optional agent interface for receiving messages.
    """

    def on_message(self, msg: Message) -> None:
        ...
