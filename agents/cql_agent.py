"""
agents/cql_agent.py

Conservative Q-Learning style lookup agent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult
from memory.sample_weighter import ReplayWeightConfig, compute_sample_weight


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _obs_key(obs: JSONValue) -> str:
    return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class ActionStats:
    count: float = 0.0
    reward_sum: float = 0.0

    def mean_reward(self) -> float:
        return self.reward_sum / float(self.count) if self.count > 0 else 0.0


class CQLLookupAgent:
    """
    Conservative lookup policy using dataset action stats.

    Config keys in AgentSpec.config:
    - dataset_path: path to apprentice dataset (must include reward)
    - alpha: penalty for unknown actions (default 0.5)
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

        if action_space.type != "discrete" or not action_space.n:
            raise ValueError("CQLLookupAgent requires discrete action space with n")

        cfg = spec.config if isinstance(spec.config, dict) else {}
        self.alpha = float(cfg.get("alpha", 0.5))
        replay_cfg = cfg.get("replay.weighting")
        if not isinstance(replay_cfg, dict):
            replay_cfg = cfg.get("replay_weighting")
        self._replay_weight_cfg = ReplayWeightConfig.from_dict(replay_cfg)

        self._stats: Dict[str, Dict[int, ActionStats]] = {}
        dataset_path = cfg.get("dataset_path")
        if dataset_path:
            self.learn_from_dataset(str(dataset_path))

    def seed(self, seed: Optional[int]) -> None:
        return

    def act(self, obs: JSONValue) -> ActionResult:
        key = _obs_key(obs)
        actions = self._stats.get(key)
        if not actions:
            # Unknown state -> random fallback
            import random
            return ActionResult(action=int(random.randrange(int(self.action_space.n))), info={"mode": "random_fallback"})

        best_a = 0
        best_score = -1e9
        for a in range(int(self.action_space.n)):
            st = actions.get(a)
            mean = st.mean_reward() if st else 0.0
            penalty = self.alpha / float((st.count if st else 0) + 1) ** 0.5
            score = mean - penalty
            if score > best_score:
                best_score = score
                best_a = a

        return ActionResult(action=int(best_a), info={"mode": "cql", "score": best_score})

    def learn(self, batch) -> Dict[str, JSONValue]:
        raise NotImplementedError("Use learn_from_dataset() for this agent")

    def learn_from_dataset(self, dataset_path: str) -> None:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obs = row.get("obs")
                action = int(row.get("action"))
                reward = _safe_float(row.get("reward", 0.0))
                sample_w = compute_sample_weight(row, cfg=self._replay_weight_cfg)
                key = _obs_key(obs)
                bucket = self._stats.setdefault(key, {})
                st = bucket.setdefault(action, ActionStats())
                st.count += float(sample_w)
                st.reward_sum += float(reward) * float(sample_w)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "spec": self.spec.to_dict(),
            "alpha": self.alpha,
            "stats": {
                k: {str(a): {"count": v.count, "reward_sum": v.reward_sum} for a, v in bucket.items()}
                for k, bucket in self._stats.items()
            },
        }
        with open(os.path.join(path, "cql_lookup.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        file_path = os.path.join(path, "cql_lookup.json")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"model file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.alpha = float(payload.get("alpha", self.alpha))
        stats = payload.get("stats", {}) or {}
        self._stats = {}
        for k, bucket in stats.items():
            inner: Dict[int, ActionStats] = {}
            for a_str, v in bucket.items():
                inner[int(a_str)] = ActionStats(
                    count=float(v.get("count", 0.0)),
                    reward_sum=_safe_float(v.get("reward_sum", 0.0)),
                )
            self._stats[k] = inner

    def close(self) -> None:
        return
