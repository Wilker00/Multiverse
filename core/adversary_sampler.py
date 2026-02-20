"""
core/adversary_sampler.py

Adversarial self-play utilities built from failure archives.

This module is local-first and file-backed:
- Reads failures from central memory or run events.
- Produces a compact behavior bundle (obs -> adversary action frequencies).
- Provides an action-mixing wrapper that can be applied to any base agent.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.agent_base import ActionResult
from core.types import JSONValue


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _obs_key(obs: JSONValue) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(obs)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
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


def _is_failure_row(row: Dict[str, Any]) -> bool:
    info = row.get("info")
    info = info if isinstance(info, dict) else {}
    if bool(row.get("done", False)) and not bool(info.get("reached_goal", False)):
        return True
    if _safe_float(row.get("reward", 0.0), 0.0) <= -10.0:
        return True
    se = info.get("safe_executor")
    if isinstance(se, dict):
        mode = str(se.get("failure_mode", "")).strip().lower()
        if mode and mode not in ("success", "unknown"):
            return True
    return False


def _is_near_miss_row(row: Dict[str, Any]) -> bool:
    info = row.get("info")
    info = info if isinstance(info, dict) else {}
    se = info.get("safe_executor")
    if not isinstance(se, dict):
        return False
    if bool(se.get("rewound", False)):
        return True
    if str(se.get("mode", "")).strip().lower() in ("shield_veto", "fallback", "planner_takeover"):
        return True
    conf = _safe_float(se.get("confidence", 1.0), 1.0)
    danger = _safe_float(se.get("danger", 0.0), 0.0)
    return bool(conf < 0.15 or danger > 0.85)


def _regret_score(row: Dict[str, Any]) -> float:
    info = row.get("info")
    info = info if isinstance(info, dict) else {}
    r = _safe_float(row.get("reward", 0.0), 0.0)
    base = max(0.0, -r)
    se = info.get("safe_executor")
    if isinstance(se, dict):
        conf = _safe_float(se.get("confidence", 1.0), 1.0)
        danger = _safe_float(se.get("danger", 0.0), 0.0)
        return float(base + max(0.0, danger - conf))
    return float(base)


@dataclass
class AdversaryBundle:
    verse_name: str
    source: str
    policy_id: str
    created_at_ms: int
    run_ids: List[str]
    obs_actions: Dict[str, Dict[str, float]]
    global_actions: Dict[str, float]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_name": self.verse_name,
            "source": self.source,
            "policy_id": self.policy_id,
            "created_at_ms": int(self.created_at_ms),
            "run_ids": list(self.run_ids),
            "obs_actions": dict(self.obs_actions),
            "global_actions": dict(self.global_actions),
            "score": float(self.score),
        }


class AdversarySampler:
    """
    Query-based failure sampler for adversarial self-play.

    Sources:
    - recent_failures: severe failures and terminal non-success transitions.
    - near_misses: safety rewinds/veto/fallback/high-danger events.
    - top_regret: highest regret events by negative reward and risk gap.
    """

    def __init__(self, memory_dir: str = "central_memory", runs_root: str = "runs"):
        self.memory_dir = str(memory_dir)
        self.runs_root = str(runs_root)

    def sample(
        self,
        *,
        verse_name: str,
        source: str = "recent_failures",
        top_k: int = 300,
    ) -> Optional[AdversaryBundle]:
        rows = self._collect_rows(verse_name=str(verse_name), source=str(source), top_k=max(10, int(top_k)))
        if not rows:
            return None
        return self._build_bundle(verse_name=str(verse_name), source=str(source), rows=rows)

    def save_bundle(self, bundle: AdversaryBundle, out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle.to_dict(), f, ensure_ascii=False, indent=2)
        return out_path

    def _collect_rows(self, *, verse_name: str, source: str, top_k: int) -> List[Dict[str, Any]]:
        source_n = str(source).strip().lower()
        mem_path = os.path.join(self.memory_dir, "memories.jsonl")
        rows: List[Dict[str, Any]] = []
        # 1) Prefer centralized memory.
        if os.path.isfile(mem_path):
            for row in _iter_jsonl(mem_path):
                if str(row.get("verse_name", "")).strip().lower() != str(verse_name).strip().lower():
                    continue
                info = row.get("info")
                info = info if isinstance(info, dict) else {}
                merged = {
                    "run_id": str(row.get("run_id", "")),
                    "policy_id": str(row.get("policy_id", "")),
                    "obs": row.get("obs"),
                    "action": row.get("action"),
                    "reward": row.get("reward"),
                    "done": row.get("done", False),
                    "info": info,
                    "t_ms": row.get("t_ms", 0),
                }
                rows.append(merged)
        # 2) Fallback to raw run events if memory is missing/empty.
        if not rows and os.path.isdir(self.runs_root):
            run_names = sorted(os.listdir(self.runs_root), reverse=True)
            for run_id in run_names[:50]:
                ep = os.path.join(self.runs_root, run_id, "events.jsonl")
                if not os.path.isfile(ep):
                    continue
                for row in _iter_jsonl(ep):
                    if str(row.get("verse_name", "")).strip().lower() != str(verse_name).strip().lower():
                        continue
                    rows.append(row)

        if source_n == "near_misses":
            filt = [r for r in rows if _is_near_miss_row(r)]
            filt.sort(key=lambda x: _safe_int(x.get("t_ms", 0), 0), reverse=True)
            return filt[:top_k]

        if source_n == "top_regret":
            ranked = sorted(rows, key=_regret_score, reverse=True)
            return ranked[:top_k]

        # default: recent_failures
        filt = [r for r in rows if _is_failure_row(r)]
        filt.sort(key=lambda x: _safe_int(x.get("t_ms", 0), 0), reverse=True)
        return filt[:top_k]

    def _build_bundle(self, *, verse_name: str, source: str, rows: List[Dict[str, Any]]) -> AdversaryBundle:
        per_obs: Dict[str, Dict[str, float]] = {}
        global_actions: Dict[str, float] = {}
        run_counts: Dict[str, int] = {}
        policy_counts: Dict[str, int] = {}
        score_sum = 0.0

        for r in rows:
            okey = _obs_key(r.get("obs"))
            action = str(_safe_int(r.get("action", 0), 0))
            reward = _safe_float(r.get("reward", 0.0), 0.0)
            sev = max(0.0, -reward)
            score_sum += sev

            bucket = per_obs.setdefault(okey, {})
            bucket[action] = float(bucket.get(action, 0.0) + max(1.0, sev))
            global_actions[action] = float(global_actions.get(action, 0.0) + max(1.0, sev))

            rid = str(r.get("run_id", "")).strip()
            if rid:
                run_counts[rid] = int(run_counts.get(rid, 0) + 1)
            pid = str(r.get("policy_id", "")).strip()
            if pid:
                policy_counts[pid] = int(policy_counts.get(pid, 0) + 1)

        policy_id = "adversary:unknown"
        if policy_counts:
            policy_id = max(policy_counts.items(), key=lambda kv: kv[1])[0]

        run_ids = sorted(run_counts.keys())
        avg_score = float(score_sum / float(max(1, len(rows))))
        return AdversaryBundle(
            verse_name=str(verse_name),
            source=str(source),
            policy_id=str(policy_id),
            created_at_ms=int(time.time() * 1000),
            run_ids=run_ids,
            obs_actions=per_obs,
            global_actions=global_actions,
            score=avg_score,
        )


class AdversarialMixWrapper:
    """
    Wrap a base agent and mix adversary actions with probability mix_ratio.
    """

    def __init__(
        self,
        base_agent: Any,
        *,
        mix_ratio: float,
        bundle: AdversaryBundle,
        seed: Optional[int] = None,
    ):
        self.base_agent = base_agent
        self.mix_ratio = max(0.0, min(1.0, float(mix_ratio)))
        self.bundle = bundle
        self._rng = random.Random(seed)

    def seed(self, seed: Optional[int]) -> None:
        self._rng = random.Random(seed)
        if hasattr(self.base_agent, "seed"):
            self.base_agent.seed(seed)

    def act(self, obs: JSONValue) -> ActionResult:
        base = self.base_agent.act(obs)
        use_adv = self._rng.random() < self.mix_ratio
        if not use_adv:
            info = dict(base.info or {})
            info["self_play"] = {"adversary_active": False, "mix_ratio": float(self.mix_ratio)}
            return ActionResult(action=base.action, info=info)

        obs_bucket = self.bundle.obs_actions.get(_obs_key(obs))
        action = self._sample_action(obs_bucket if isinstance(obs_bucket, dict) else self.bundle.global_actions)
        if action is None:
            info = dict(base.info or {})
            info["self_play"] = {"adversary_active": False, "fallback": "base_action"}
            return ActionResult(action=base.action, info=info)

        info = dict(base.info or {})
        info["self_play"] = {
            "adversary_active": True,
            "source": self.bundle.source,
            "policy_id": self.bundle.policy_id,
            "mix_ratio": float(self.mix_ratio),
            "adversary_action": int(action),
        }
        return ActionResult(action=int(action), info=info)

    def learn(self, batch: Any) -> Dict[str, JSONValue]:
        if hasattr(self.base_agent, "learn"):
            return self.base_agent.learn(batch)
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self.base_agent, "close"):
            self.base_agent.close()

    def save(self, path: str) -> None:
        if hasattr(self.base_agent, "save"):
            self.base_agent.save(path)

    def load(self, path: str) -> None:
        if hasattr(self.base_agent, "load"):
            self.base_agent.load(path)

    def _sample_action(self, weights_by_action: Dict[str, float]) -> Optional[int]:
        if not weights_by_action:
            return None
        total = 0.0
        pairs: List[Tuple[int, float]] = []
        for a_str, w in weights_by_action.items():
            a = _safe_int(a_str, -1)
            if a < 0:
                continue
            ww = max(0.0, _safe_float(w, 0.0))
            if ww <= 0.0:
                continue
            pairs.append((a, ww))
            total += ww
        if not pairs or total <= 0.0:
            return None
        r = self._rng.random() * total
        c = 0.0
        for a, w in pairs:
            c += w
            if r <= c:
                return int(a)
        return int(pairs[-1][0])

