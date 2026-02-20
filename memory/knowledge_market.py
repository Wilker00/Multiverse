"""
memory/knowledge_market.py

Reputation-driven DNA recommendation market for cross-agent transfer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class KnowledgeMarketConfig:
    root_dir: str = "central_memory"
    memories_filename: str = "memories.jsonl"
    ledger_filename: str = "knowledge_market_ledger.json"
    max_offers: int = 10
    runs_roots: Optional[List[str]] = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm(x: Any) -> str:
    return str(x).strip().lower()


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = {_norm(x) for x in a if _norm(x)}
    sb = {_norm(x) for x in b if _norm(x)}
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return float(inter / float(max(1, uni)))


def _ledger_path(cfg: KnowledgeMarketConfig) -> str:
    return os.path.join(cfg.root_dir, cfg.ledger_filename)


def _mem_path(cfg: KnowledgeMarketConfig) -> str:
    return os.path.join(cfg.root_dir, cfg.memories_filename)


def _default_ledger() -> Dict[str, Any]:
    return {
        "version": "v1",
        "updated_at_ms": 0,
        "providers": {},
        "transactions": [],
    }


class KnowledgeMarket:
    def __init__(self, cfg: Optional[KnowledgeMarketConfig] = None):
        self.cfg = cfg or KnowledgeMarketConfig()
        os.makedirs(self.cfg.root_dir, exist_ok=True)

    def _load_ledger(self) -> Dict[str, Any]:
        path = _ledger_path(self.cfg)
        if not os.path.isfile(path):
            return _default_ledger()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_ledger()
        if not isinstance(data.get("providers"), dict):
            data["providers"] = {}
        if not isinstance(data.get("transactions"), list):
            data["transactions"] = []
        return data

    def _save_ledger(self, data: Dict[str, Any]) -> str:
        data = dict(data)
        data["updated_at_ms"] = int(time.time() * 1000)
        path = _ledger_path(self.cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _iter_memories(self):
        path = _mem_path(self.cfg)
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row

    def _summarize_providers(self) -> Dict[str, Dict[str, Any]]:
        by_run: Dict[str, Dict[str, Any]] = {}
        for row in self._iter_memories() or []:
            run_id = str(row.get("run_id", "")).strip()
            if not run_id:
                continue
            bucket = by_run.get(run_id)
            if not isinstance(bucket, dict):
                bucket = {
                    "run_id": run_id,
                    "verse_name": str(row.get("verse_name", "")),
                    "policy_id": str(row.get("policy_id", "")),
                    "tags": set(),
                    "mean_reward_sum": 0.0,
                    "count": 0,
                }
                by_run[run_id] = bucket
            tags = row.get("tags")
            if isinstance(tags, list):
                for t in tags:
                    bucket["tags"].add(_norm(t))
            bucket["mean_reward_sum"] += _safe_float(row.get("reward", 0.0))
            bucket["count"] += 1

        for run_id, b in by_run.items():
            c = max(1, int(b["count"]))
            b["mean_reward"] = float(b["mean_reward_sum"] / float(c))
            b["tags"] = sorted(list(b["tags"]))
            b["dna_paths"] = self._candidate_dna_paths(run_id)
        return by_run

    def _candidate_run_roots(self) -> List[str]:
        configured = self.cfg.runs_roots
        if isinstance(configured, list) and configured:
            out: List[str] = []
            for p in configured:
                s = str(p).strip()
                if s:
                    out.append(s)
            return out or ["runs"]

        out = ["runs"]
        try:
            for name in sorted(os.listdir(".")):
                if name == "runs":
                    continue
                if not str(name).startswith("runs"):
                    continue
                if os.path.isdir(name):
                    out.append(str(name))
        except Exception:
            pass
        return out

    def _candidate_dna_paths(self, run_id: str) -> List[str]:
        out = []
        for runs_root in self._candidate_run_roots():
            run_dir = os.path.join(str(runs_root), str(run_id))
            for name in ("golden_dna.jsonl", "dna_good.jsonl", "events.jsonl"):
                p = os.path.join(run_dir, name)
                if os.path.isfile(p):
                    out.append(p.replace("\\", "/"))
        return out

    def bid_for_dna(
        self,
        *,
        agent_id: str,
        task_tags: List[str],
        verse_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        ledger = self._load_ledger()
        providers = self._summarize_providers()
        top = int(top_k if top_k is not None else self.cfg.max_offers)
        out: List[Dict[str, Any]] = []

        for provider_id, p in providers.items():
            rep_bucket = ledger.get("providers", {}).get(provider_id, {})
            reputation = _safe_float(rep_bucket.get("reputation", 1.0), 1.0)
            mean_reward = _safe_float(p.get("mean_reward", 0.0), 0.0)
            sim = _jaccard(task_tags, list(p.get("tags", [])))
            verse_bonus = 0.0
            if verse_name and _norm(verse_name) == _norm(p.get("verse_name", "")):
                verse_bonus = 0.15
            transfer = (0.55 * sim) + (0.25 * reputation) + (0.20 * max(0.0, mean_reward)) + verse_bonus
            out.append(
                {
                    "provider_id": provider_id,
                    "verse_name": p.get("verse_name"),
                    "policy_id": p.get("policy_id"),
                    "tags": p.get("tags", []),
                    "mean_reward": mean_reward,
                    "reputation": reputation,
                    "transfer_potential": float(transfer),
                    "dna_paths": p.get("dna_paths", []),
                }
            )
        out.sort(key=lambda x: float(x.get("transfer_potential", 0.0)), reverse=True)
        return out[: max(1, top)]

    def update_reputation(
        self,
        *,
        provider_id: str,
        delta: float,
        reason: str,
        consumer_agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        ledger = self._load_ledger()
        providers = ledger.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            ledger["providers"] = providers
        p = providers.get(provider_id)
        if not isinstance(p, dict):
            p = {"reputation": 1.0, "updates": 0}
            providers[provider_id] = p

        old = _safe_float(p.get("reputation", 1.0), 1.0)
        new = max(0.0, old + float(delta))
        p["reputation"] = float(new)
        p["updates"] = int(p.get("updates", 0) or 0) + 1

        txs = ledger.setdefault("transactions", [])
        if not isinstance(txs, list):
            txs = []
            ledger["transactions"] = txs
        txs.append(
            {
                "t_ms": int(time.time() * 1000),
                "provider_id": provider_id,
                "consumer_agent_id": consumer_agent_id,
                "delta": float(delta),
                "reason": str(reason),
                "old_reputation": float(old),
                "new_reputation": float(new),
            }
        )
        self._save_ledger(ledger)
        return {
            "provider_id": provider_id,
            "old_reputation": float(old),
            "new_reputation": float(new),
            "reason": str(reason),
        }

    def summary(self) -> Dict[str, Any]:
        ledger = self._load_ledger()
        providers = ledger.get("providers", {})
        txs = ledger.get("transactions", [])
        if not isinstance(providers, dict):
            providers = {}
        if not isinstance(txs, list):
            txs = []
        ranked: List[Tuple[str, float]] = []
        for pid, rec in providers.items():
            if isinstance(rec, dict):
                ranked.append((pid, _safe_float(rec.get("reputation", 1.0), 1.0)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return {
            "providers": len(providers),
            "transactions": len(txs),
            "top_reputation": ranked[:10],
        }
