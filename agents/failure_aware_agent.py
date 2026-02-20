"""
agents/failure_aware_agent.py

Failure-aware action masking from DNA_BAD trajectories.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

from agents.imitation_agent import obs_key
from core.agent_base import ActionResult
from core.types import AgentSpec, JSONValue, SpaceSpec


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class FailureAwareAgent:
    """
    Masks risky actions learned from bad trajectories.

    Config keys:
    - danger_temperature: float (default 1.0)
    - hard_block_threshold: float (default 0.98)
    - caution_penalty_scale: float (default 1.0)
    - danger_prior: float (default 1.0)
    - min_action_prob: float (default 1e-6)
    - bad_dna_path: optional path for bad dataset
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        if action_space.type != "discrete":
            raise ValueError("FailureAwareAgent currently supports discrete action spaces only.")
        if action_space.n is None or int(action_space.n) <= 0:
            raise ValueError("Discrete action space requires n > 0.")

        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space
        self._seed: Optional[int] = None
        self._rng = None
        import random

        self._random = random
        self._n = int(action_space.n)
        self._danger_temperature = max(1e-6, _safe_float((spec.config or {}).get("danger_temperature", 1.0), 1.0))
        self._hard_block_threshold = min(
            0.999999,
            max(0.0, _safe_float((spec.config or {}).get("hard_block_threshold", 0.98), 0.98)),
        )
        self._caution_penalty_scale = max(
            0.0,
            _safe_float((spec.config or {}).get("caution_penalty_scale", 1.0), 1.0),
        )
        self._danger_prior = max(0.0, _safe_float((spec.config or {}).get("danger_prior", 1.0), 1.0))
        self._min_action_prob = min(
            1e-2,
            max(0.0, _safe_float((spec.config or {}).get("min_action_prob", 1e-6), 1e-6)),
        )

        self._good_counts: Dict[str, Dict[int, int]] = {}
        self._bad_counts: Dict[str, Dict[int, int]] = {}

        cfg = spec.config if isinstance(spec.config, dict) else {}
        bad_path = cfg.get("bad_dna_path")
        if bad_path:
            self.learn_from_bad_dataset(str(bad_path))

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        self._rng = self._random.Random(seed)

    def act(self, obs: JSONValue) -> ActionResult:
        if self._rng is None:
            self.seed(self._seed)
        diag = self.action_diagnostics(obs)
        base = [float(x) for x in diag.get("base_probs", [1.0 / float(self._n) for _ in range(self._n)])]
        danger = [float(x) for x in diag.get("danger_scores", [0.0 for _ in range(self._n)])]
        legal = [float(x) for x in diag.get("adjusted_probs", [0.0 for _ in range(self._n)])]
        blocked = [bool(x) for x in diag.get("blocked_actions", [False for _ in range(self._n)])]

        if sum(legal) <= 0.0:
            # All actions are dangerous for this state. Choose least dangerous.
            best = min(range(self._n), key=lambda a: danger[a])
            caution_penalty = float(self._caution_penalty_scale * danger[int(best)])
            return ActionResult(
                action=int(best),
                info={
                    "mode": "failure_aware_least_danger",
                    "danger_scores": danger,
                    "base_probs": base,
                    "blocked_actions": blocked,
                    "caution_penalty": caution_penalty,
                },
            )

        action = self._weighted_sample(legal)
        caution_penalty = float(self._caution_penalty_scale * danger[int(action)])
        return ActionResult(
            action=int(action),
            info={
                "mode": "failure_aware",
                "danger_scores": danger,
                "base_probs": base,
                "blocked_actions": blocked,
                "caution_penalty": caution_penalty,
            },
        )

    def learn(self, batch) -> Dict[str, JSONValue]:
        raise NotImplementedError("Use learn_from_dataset() and learn_from_bad_dataset().")

    def learn_from_dataset(self, dataset_path: str) -> Dict[str, JSONValue]:
        added = self._ingest(dataset_path, target=self._good_counts)
        return {"good_rows_added": int(added)}

    def learn_from_bad_dataset(self, dataset_path: str) -> Dict[str, JSONValue]:
        added = self._ingest(dataset_path, target=self._bad_counts)
        return {"bad_rows_added": int(added)}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "spec": self.spec.to_dict(),
            "good_counts": self._good_counts,
            "bad_counts": self._bad_counts,
            "danger_temperature": float(self._danger_temperature),
            "hard_block_threshold": float(self._hard_block_threshold),
            "caution_penalty_scale": float(self._caution_penalty_scale),
            "danger_prior": float(self._danger_prior),
            "min_action_prob": float(self._min_action_prob),
        }
        with open(os.path.join(path, "failure_aware.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        fp = os.path.join(path, "failure_aware.json")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"model file not found: {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._good_counts = {
            str(k): {int(a): int(c) for a, c in (v or {}).items()}
            for k, v in (payload.get("good_counts") or {}).items()
        }
        self._bad_counts = {
            str(k): {int(a): int(c) for a, c in (v or {}).items()}
            for k, v in (payload.get("bad_counts") or {}).items()
        }
        self._danger_temperature = max(1e-6, _safe_float(payload.get("danger_temperature", 1.0), 1.0))
        self._hard_block_threshold = min(
            0.999999,
            max(0.0, _safe_float(payload.get("hard_block_threshold", self._hard_block_threshold), self._hard_block_threshold)),
        )
        self._caution_penalty_scale = max(
            0.0,
            _safe_float(payload.get("caution_penalty_scale", self._caution_penalty_scale), self._caution_penalty_scale),
        )
        self._danger_prior = max(0.0, _safe_float(payload.get("danger_prior", self._danger_prior), self._danger_prior))
        self._min_action_prob = min(
            1e-2,
            max(0.0, _safe_float(payload.get("min_action_prob", self._min_action_prob), self._min_action_prob)),
        )

    def close(self) -> None:
        return

    def _ingest(self, path: str, *, target: Dict[str, Dict[int, int]]) -> int:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"dataset not found: {path}")
        added = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                action = row.get("action")
                try:
                    a = int(action)
                except Exception:
                    continue
                if a < 0 or a >= self._n:
                    continue
                k = obs_key(row.get("obs"))
                bucket = target.setdefault(k, {})
                bucket[a] = int(bucket.get(a, 0)) + 1
                added += 1
        return added

    def _base_action_probs(self, obs_k: str) -> List[float]:
        bucket = self._good_counts.get(obs_k, {})
        if not bucket:
            return [1.0 / float(self._n) for _ in range(self._n)]
        total = float(sum(int(c) for c in bucket.values()))
        if total <= 0:
            return [1.0 / float(self._n) for _ in range(self._n)]
        out = [0.0 for _ in range(self._n)]
        for a in range(self._n):
            out[a] = float(int(bucket.get(a, 0)) / total)
        return out

    def _danger_scores(self, obs_k: str) -> List[float]:
        bad_bucket = self._bad_counts.get(obs_k, {})
        good_bucket = self._good_counts.get(obs_k, {})
        if not bad_bucket:
            return [0.0 for _ in range(self._n)]
        out = [0.0 for _ in range(self._n)]
        for a in range(self._n):
            bad_c = float(_safe_int(bad_bucket.get(a, 0), 0))
            good_c = float(_safe_int(good_bucket.get(a, 0), 0))
            if bad_c <= 0.0 and good_c <= 0.0:
                out[a] = 0.0
                continue
            denom = bad_c + good_c + float(self._danger_prior)
            if denom <= 0.0:
                out[a] = 0.0
            else:
                out[a] = min(1.0, max(0.0, bad_c / denom))
        return out

    def action_diagnostics(self, obs: JSONValue) -> Dict[str, JSONValue]:
        k = obs_key(obs)
        base = self._base_action_probs(k)
        danger = self._danger_scores(k)

        blocked = [bool(d >= self._hard_block_threshold) for d in danger]
        adjusted = [0.0 for _ in range(self._n)]
        caution = [0.0 for _ in range(self._n)]
        for a in range(self._n):
            d = min(1.0, max(0.0, float(danger[a])))
            caution[a] = float(self._caution_penalty_scale * d)
            if blocked[a]:
                adjusted[a] = 0.0
                continue
            retain = (1.0 - d) ** float(self._danger_temperature)
            retain *= math.exp(-float(self._caution_penalty_scale) * d)
            retain = max(float(self._min_action_prob), float(retain))
            adjusted[a] = max(0.0, float(base[a]) * retain)

        total = float(sum(adjusted))
        probs = [0.0 for _ in range(self._n)]
        if total > 0.0:
            probs = [float(x / total) for x in adjusted]

        return {
            "obs_key": k,
            "base_probs": [float(x) for x in base],
            "danger_scores": [float(x) for x in danger],
            "blocked_actions": blocked,
            "caution_penalties": [float(x) for x in caution],
            "adjusted_probs": [float(x) for x in adjusted],
            "sample_probs": [float(x) for x in probs],
            "config": {
                "danger_temperature": float(self._danger_temperature),
                "hard_block_threshold": float(self._hard_block_threshold),
                "caution_penalty_scale": float(self._caution_penalty_scale),
                "danger_prior": float(self._danger_prior),
                "min_action_prob": float(self._min_action_prob),
            },
        }

    def _weighted_sample(self, weights: List[float]) -> int:
        total = sum(float(w) for w in weights)
        if total <= 0.0:
            return int(self._rng.randrange(self._n))
        r = self._rng.random() * total
        c = 0.0
        for i, w in enumerate(weights):
            c += float(w)
            if r <= c:
                return int(i)
        return int(len(weights) - 1)
