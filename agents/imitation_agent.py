"""
agents/imitation_agent.py

Lookup-first imitation with generalization layers for unseen features:
1) exact observation lookup
2) optional learned MLP policy (interpolation)
3) optional nearest-neighbor retrieval (similarity fallback)
4) random fallback (last resort)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.agent_base import ActionResult
from core.taxonomy import memory_family_for_type, memory_type_for_verse
from core.types import AgentSpec, JSONValue, SpaceSpec
from memory.sample_weighter import ReplayWeightConfig, compute_sample_weight


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


def _is_list_of_numbers(x: Any) -> bool:
    if not isinstance(x, list):
        return False
    return all(isinstance(v, (int, float)) for v in x)


def obs_key(obs: JSONValue) -> str:
    return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _flatten_numeric(x: Any) -> List[float]:
    if isinstance(x, bool):
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, list):
        out: List[float] = []
        for v in x:
            out.extend(_flatten_numeric(v))
        return out
    if isinstance(x, dict):
        out: List[float] = []
        for k in sorted(x.keys()):
            out.extend(_flatten_numeric(x.get(k)))
        return out
    return []


def _pad(values: List[float], dim: int) -> List[float]:
    if dim <= 0:
        return []
    if len(values) >= dim:
        return values[:dim]
    return values + [0.0] * (dim - len(values))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return -1.0
    n = min(len(a), len(b))
    if n <= 0:
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    denom = math.sqrt(max(1e-9, na)) * math.sqrt(max(1e-9, nb))
    if denom <= 0.0:
        return -1.0
    return dot / denom


def _row_memory_family(row: Dict[str, Any]) -> str:
    mt = str(row.get("memory_type", "")).strip().lower()
    if mt:
        fam = memory_family_for_type(mt)
        if fam in {"procedural", "declarative", "hybrid"}:
            return fam
    fam = str(row.get("memory_family", "")).strip().lower()
    if fam in {"procedural", "declarative", "hybrid"}:
        return fam
    verse = str(row.get("verse_name", row.get("target_verse_name", ""))).strip().lower()
    if verse:
        inferred = memory_family_for_type(memory_type_for_verse(verse))
        if inferred in {"procedural", "declarative", "hybrid"}:
            return inferred
    return "unknown"


@dataclass
class ImitationStats:
    total_rows: int = 0
    unique_obs: int = 0
    matched_rate_estimate: float = 0.0
    mode: str = "all"
    mode_skipped_rows: int = 0


class ImitationLookupAgent:
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

        self._seed: Optional[int] = None
        self._rng = None
        import random

        self._random = random

        self._policy_discrete: Dict[str, Dict[str, float]] = {}
        self._policy_continuous_sum: Dict[str, List[float]] = {}
        self._policy_continuous_count: Dict[str, float] = {}

        self._is_discrete = action_space.type == "discrete"
        self._is_continuous = action_space.type == "continuous"
        if not (self._is_discrete or self._is_continuous):
            raise ValueError("ImitationLookupAgent supports 'discrete' or 'continuous' only")

        cfg = spec.config if isinstance(spec.config, dict) else {}
        replay_cfg = cfg.get("replay.weighting")
        if not isinstance(replay_cfg, dict):
            replay_cfg = cfg.get("replay_weighting")
        self._replay_weight_cfg = ReplayWeightConfig.from_dict(replay_cfg)
        raw_memory_mode = cfg.get("memory_mode", cfg.get("imitation_memory_mode", "procedural"))
        self._memory_mode = str(raw_memory_mode).strip().lower()
        if self._memory_mode not in {"all", "procedural", "declarative"}:
            self._memory_mode = "all"
        self._cross_memory_weight = max(0.0, min(1.0, _safe_float(cfg.get("cross_memory_weight", 0.25), 0.25)))
        self._unknown_memory_weight = max(0.0, min(1.0, _safe_float(cfg.get("unknown_memory_weight", 0.50), 0.50)))

        # kNN fallback settings.
        self._nn_enabled = bool(cfg.get("enable_nn_fallback", True))
        self._nn_k = max(1, _safe_int(cfg.get("nn_fallback_k", 5), 5))
        self._nn_min_similarity = _safe_float(cfg.get("nn_fallback_min_similarity", -1.0), -1.0)
        self._nn_max_samples = max(100, _safe_int(cfg.get("nn_max_samples", 5000), 5000))
        self._nn_vectors: List[List[float]] = []
        self._nn_actions_discrete: List[int] = []
        self._nn_actions_continuous: List[List[float]] = []
        self._nn_weights: List[float] = []
        self._vector_dim = 0

        # Learned generalizer settings.
        self._mlp_enabled = bool(cfg.get("enable_mlp_generalizer", False))
        self._mlp_hidden_dim = max(8, _safe_int(cfg.get("mlp_hidden_dim", 64), 64))
        self._mlp_epochs = max(1, _safe_int(cfg.get("mlp_epochs", 12), 12))
        self._mlp_lr = max(1e-6, _safe_float(cfg.get("mlp_lr", 1e-3), 1e-3))
        self._mlp_min_rows = max(50, _safe_int(cfg.get("mlp_min_rows", 200), 200))
        self._mlp_batch_size = max(8, _safe_int(cfg.get("mlp_batch_size", 128), 128))
        self._mlp_model = None
        self._mlp_mean: List[float] = []
        self._mlp_std: List[float] = []
        self._torch = None

    def _valid_discrete_action(self, action: int) -> Optional[int]:
        if not self._is_discrete:
            return None
        n = int(self.action_space.n or 0)
        a = int(action)
        if n > 0 and (a < 0 or a >= n):
            return None
        return a

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        self._rng = self._random.Random(seed)

    def act(self, obs: JSONValue) -> ActionResult:
        if self._rng is None:
            self.seed(self._seed)
        k = obs_key(obs)

        if self._is_discrete:
            action, mode = self._act_discrete(k, obs)
            return ActionResult(action=action, info={"mode": mode})
        if self._is_continuous:
            action, mode = self._act_continuous(k, obs)
            return ActionResult(action=action, info={"mode": mode})
        return ActionResult(action=self._sample_random_action(), info={"mode": "random_fallback"})

    def learn(self, batch) -> Dict[str, JSONValue]:
        raise NotImplementedError("Use learn_from_dataset() for this agent")

    def learn_from_dataset(self, dataset_path: str, limit_rows: Optional[int] = None) -> ImitationStats:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"dataset not found: {dataset_path}")

        total = 0
        train_vectors: List[List[float]] = []
        train_actions_discrete: List[int] = []
        train_actions_continuous: List[List[float]] = []
        train_weights: List[float] = []
        mode_skipped = 0

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                fam = _row_memory_family(row)
                if self._memory_mode == "procedural":
                    if fam == "declarative" and self._cross_memory_weight <= 0.0:
                        mode_skipped += 1
                        continue
                    if fam == "unknown" and self._unknown_memory_weight <= 0.0:
                        mode_skipped += 1
                        continue
                elif self._memory_mode == "declarative":
                    if fam == "procedural" and self._cross_memory_weight <= 0.0:
                        mode_skipped += 1
                        continue
                    if fam == "unknown" and self._unknown_memory_weight <= 0.0:
                        mode_skipped += 1
                        continue
                obs = row.get("obs")
                action = row.get("action")
                k = obs_key(obs)
                sample_w = compute_sample_weight(row, cfg=self._replay_weight_cfg)
                if self._memory_mode == "procedural":
                    if fam == "declarative":
                        sample_w *= float(self._cross_memory_weight)
                    elif fam == "unknown":
                        sample_w *= float(self._unknown_memory_weight)
                elif self._memory_mode == "declarative":
                    if fam == "procedural":
                        sample_w *= float(self._cross_memory_weight)
                    elif fam == "unknown":
                        sample_w *= float(self._unknown_memory_weight)

                if self._is_discrete:
                    a_raw = _safe_int(action, 0)
                    a_norm = self._valid_discrete_action(a_raw)
                    if a_norm is None:
                        continue
                    a = int(a_norm)
                    bucket = self._policy_discrete.setdefault(k, {})
                    key_a = str(a)
                    bucket[key_a] = float(bucket.get(key_a, 0.0)) + float(sample_w)
                else:
                    if not _is_list_of_numbers(action):
                        continue
                    vec = [float(v) for v in action]
                    if k not in self._policy_continuous_sum:
                        self._policy_continuous_sum[k] = [0.0 for _ in vec]
                        self._policy_continuous_count[k] = 0.0
                    if len(vec) != len(self._policy_continuous_sum[k]):
                        continue
                    for i, v in enumerate(vec):
                        self._policy_continuous_sum[k][i] += float(v) * float(sample_w)
                    self._policy_continuous_count[k] += float(sample_w)

                vec_obs = _flatten_numeric(obs)
                if vec_obs:
                    train_vectors.append(vec_obs)
                    train_weights.append(float(sample_w))
                    if self._is_discrete:
                        a_raw = _safe_int(action, 0)
                        a_norm = self._valid_discrete_action(a_raw)
                        if a_norm is None:
                            # Keep vector/action arrays aligned; drop this sample.
                            train_vectors.pop()
                            train_weights.pop()
                            continue
                        train_actions_discrete.append(int(a_norm))
                    else:
                        train_actions_continuous.append([float(v) for v in action] if _is_list_of_numbers(action) else [])

                total += 1
                if limit_rows is not None and total >= int(limit_rows):
                    break

        self._fit_vector_store(
            vectors=train_vectors,
            actions_discrete=train_actions_discrete,
            actions_continuous=train_actions_continuous,
            weights=train_weights,
        )
        self._fit_mlp_if_enabled()

        unique = len(self._policy_discrete) if self._is_discrete else len(self._policy_continuous_sum)
        return ImitationStats(
            total_rows=total,
            unique_obs=unique,
            matched_rate_estimate=0.0,
            mode=str(self._memory_mode),
            mode_skipped_rows=int(mode_skipped),
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload: Dict[str, Any] = {
            "spec": self.spec.to_dict(),
            "action_space": {
                "type": self.action_space.type,
                "n": self.action_space.n,
                "low": self.action_space.low,
                "high": self.action_space.high,
                "shape": list(self.action_space.shape) if self.action_space.shape else None,
            },
            "policy_discrete": self._policy_discrete,
            "policy_continuous_sum": self._policy_continuous_sum,
            "policy_continuous_count": self._policy_continuous_count,
            "vector_dim": self._vector_dim,
        }
        with open(os.path.join(path, "imitation_lookup.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        file_path = os.path.join(path, "imitation_lookup.json")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"model file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._policy_discrete = payload.get("policy_discrete", {}) or {}
        self._policy_continuous_sum = payload.get("policy_continuous_sum", {}) or {}
        self._policy_continuous_count = payload.get("policy_continuous_count", {}) or {}
        self._vector_dim = _safe_int(payload.get("vector_dim", self._vector_dim), self._vector_dim)

    def close(self) -> None:
        return

    def estimate_confidence(self, obs: JSONValue) -> float:
        k = obs_key(obs)
        if self._is_discrete:
            bucket = self._policy_discrete.get(k)
            if bucket:
                total = sum(float(v) for v in bucket.values())
                if total <= 0.0:
                    return 1.0
                best = max(float(v) for v in bucket.values())
                return max(0.0, min(1.0, best / total))
        else:
            if k in self._policy_continuous_sum and float(self._policy_continuous_count.get(k, 0.0)) > 0.0:
                return 1.0

        vec = _pad(_flatten_numeric(obs), self._vector_dim)
        if vec and self._nn_vectors:
            best = max((_cosine(vec, v) for v in self._nn_vectors), default=-1.0)
            return max(0.0, min(1.0, float(best)))
        if self._mlp_model is not None:
            return 0.5
        return 0.0

    def _fit_vector_store(
        self,
        *,
        vectors: List[List[float]],
        actions_discrete: List[int],
        actions_continuous: List[List[float]],
        weights: List[float],
    ) -> None:
        if not vectors:
            return
        self._vector_dim = max(self._vector_dim, max(len(v) for v in vectors))
        limit = min(len(vectors), self._nn_max_samples)

        self._nn_vectors = []
        self._nn_weights = []
        self._nn_actions_discrete = []
        self._nn_actions_continuous = []
        for i in range(limit):
            self._nn_vectors.append(_pad(vectors[i], self._vector_dim))
            self._nn_weights.append(float(weights[i] if i < len(weights) else 1.0))
            if self._is_discrete:
                self._nn_actions_discrete.append(int(actions_discrete[i] if i < len(actions_discrete) else 0))
            else:
                av = actions_continuous[i] if i < len(actions_continuous) else []
                self._nn_actions_continuous.append([float(x) for x in av])

    def _fit_mlp_if_enabled(self) -> None:
        if not self._mlp_enabled or self._vector_dim <= 0:
            return
        n = len(self._nn_vectors)
        if n < self._mlp_min_rows:
            return
        try:
            import torch
            import torch.nn as nn
        except Exception:
            return

        x = torch.tensor(self._nn_vectors, dtype=torch.float32)
        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp_min(1e-6)
        xn = (x - mean) / std

        if self._is_discrete:
            y = torch.tensor(self._nn_actions_discrete, dtype=torch.long)
            max_label = max((int(v) for v in self._nn_actions_discrete), default=0)
            num_classes = max(1, int((self.action_space.n or 0)), int(max_label + 1))
            model = nn.Sequential(
                nn.Linear(self._vector_dim, self._mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._mlp_hidden_dim, self._mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._mlp_hidden_dim, num_classes),
            )
            loss_fn = nn.CrossEntropyLoss()
        else:
            y_dim = min((len(v) for v in self._nn_actions_continuous if v), default=0)
            if y_dim <= 0:
                return
            yc = torch.tensor([_pad(v, y_dim) for v in self._nn_actions_continuous], dtype=torch.float32)
            model = nn.Sequential(
                nn.Linear(self._vector_dim, self._mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._mlp_hidden_dim, self._mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._mlp_hidden_dim, y_dim),
            )
            y = yc
            loss_fn = nn.MSELoss()

        opt = torch.optim.Adam(model.parameters(), lr=self._mlp_lr)
        model.train()
        bs = min(self._mlp_batch_size, n)
        for _ in range(self._mlp_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                xb = xn[idx]
                yb = y[idx]
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        model.eval()
        self._torch = torch
        self._mlp_model = model
        self._mlp_mean = [float(v) for v in mean.tolist()]
        self._mlp_std = [float(v) for v in std.tolist()]

    def _act_discrete(self, k: str, obs: JSONValue) -> Tuple[int, str]:
        bucket = self._policy_discrete.get(k)
        if bucket:
            valid_votes: Dict[int, float] = {}
            for ka, w in bucket.items():
                ai = self._valid_discrete_action(_safe_int(ka, -1))
                if ai is None:
                    continue
                valid_votes[int(ai)] = valid_votes.get(int(ai), 0.0) + float(w)
            if valid_votes:
                best_a = max(valid_votes.items(), key=lambda kv: float(kv[1]))[0]
                return int(best_a), "imitation_lookup"

        vec = _pad(_flatten_numeric(obs), self._vector_dim)
        if self._mlp_model is not None and vec:
            pred = self._mlp_predict_discrete(vec)
            if pred is not None and self._valid_discrete_action(int(pred)) is not None:
                return int(pred), "mlp_generalizer"
        if self._nn_enabled and vec:
            pred = self._nn_predict_discrete(vec)
            if pred is not None and self._valid_discrete_action(int(pred)) is not None:
                return int(pred), "nn_fallback"
        return int(self._sample_random_action()), "random_fallback"

    def _act_continuous(self, k: str, obs: JSONValue) -> Tuple[JSONValue, str]:
        if k in self._policy_continuous_sum:
            s = self._policy_continuous_sum[k]
            c = float(self._policy_continuous_count.get(k, 0.0))
            if c > 0.0:
                return [float(v) / c for v in s], "imitation_lookup"

        vec = _pad(_flatten_numeric(obs), self._vector_dim)
        if self._mlp_model is not None and vec:
            pred = self._mlp_predict_continuous(vec)
            if pred is not None:
                return pred, "mlp_generalizer"
        if self._nn_enabled and vec:
            pred = self._nn_predict_continuous(vec)
            if pred is not None:
                return pred, "nn_fallback"
        return self._sample_random_action(), "random_fallback"

    def _mlp_predict_discrete(self, vec: List[float]) -> Optional[int]:
        if self._mlp_model is None or self._torch is None:
            return None
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor([vec], dtype=torch.float32)
            if self._mlp_mean and self._mlp_std:
                m = torch.tensor([self._mlp_mean], dtype=torch.float32)
                s = torch.tensor([self._mlp_std], dtype=torch.float32)
                x = (x - m) / s
            logits = self._mlp_model(x)
            return int(torch.argmax(logits, dim=1).item())

    def _mlp_predict_continuous(self, vec: List[float]) -> Optional[List[float]]:
        if self._mlp_model is None or self._torch is None:
            return None
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor([vec], dtype=torch.float32)
            if self._mlp_mean and self._mlp_std:
                m = torch.tensor([self._mlp_mean], dtype=torch.float32)
                s = torch.tensor([self._mlp_std], dtype=torch.float32)
                x = (x - m) / s
            y = self._mlp_model(x).squeeze(0).tolist()
        return [float(v) for v in y]

    def _nn_predict_discrete(self, vec: List[float]) -> Optional[int]:
        if not self._nn_vectors or not self._nn_actions_discrete:
            return None
        scored: List[Tuple[float, int, float]] = []
        for i, v in enumerate(self._nn_vectors):
            sim = _cosine(vec, v)
            scored.append((sim, self._nn_actions_discrete[i], self._nn_weights[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._nn_k]
        if not top or float(top[0][0]) < self._nn_min_similarity:
            return None
        votes: Dict[int, float] = {}
        for sim, a, w in top:
            ww = max(0.0, float(sim)) * max(0.0, float(w))
            a_norm = self._valid_discrete_action(int(a))
            if a_norm is None:
                continue
            votes[int(a_norm)] = votes.get(int(a_norm), 0.0) + ww
        if not votes:
            return None
        return int(max(votes.items(), key=lambda kv: kv[1])[0])

    def _nn_predict_continuous(self, vec: List[float]) -> Optional[List[float]]:
        if not self._nn_vectors or not self._nn_actions_continuous:
            return None
        scored: List[Tuple[float, List[float], float]] = []
        for i, v in enumerate(self._nn_vectors):
            sim = _cosine(vec, v)
            a = self._nn_actions_continuous[i]
            scored.append((sim, a, self._nn_weights[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._nn_k]
        if not top or float(top[0][0]) < self._nn_min_similarity:
            return None
        dim = min((len(a) for _, a, _ in top if a), default=0)
        if dim <= 0:
            return None
        out = [0.0] * dim
        total = 0.0
        for sim, a, w in top:
            ww = max(0.0, float(sim)) * max(0.0, float(w))
            if ww <= 0.0:
                continue
            total += ww
            for i in range(dim):
                out[i] += ww * float(a[i])
        if total <= 0.0:
            return None
        return [v / total for v in out]

    def _sample_random_action(self) -> JSONValue:
        if self._rng is None:
            self.seed(self._seed)
        space = self.action_space
        if space.type == "discrete":
            n = int(space.n or 0)
            if n <= 0:
                raise ValueError("discrete SpaceSpec requires n > 0")
            return int(self._rng.randrange(n))
        if space.type == "continuous":
            low = space.low
            high = space.high
            if low is None or high is None or len(low) != len(high):
                raise ValueError("continuous SpaceSpec requires equal low/high lengths")
            return [float(self._rng.uniform(float(low[i]), float(high[i]))) for i in range(len(low))]
        raise ValueError(f"Unsupported action space type: {space.type}")
