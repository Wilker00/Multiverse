"""
agents/distilled_agent.py

Behavior cloning / policy distillation for discrete actions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult, ExperienceBatch


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _obs_to_vector(obs: JSONValue, keys: Optional[List[str]] = None) -> np.ndarray:
    if isinstance(obs, dict):
        if keys is None:
            keys = sorted(obs.keys())
        vec: List[float] = []
        for k in keys:
            v = obs.get(k)
            if _is_number(v):
                vec.append(float(v))
            elif isinstance(v, list) and all(_is_number(x) for x in v):
                vec.extend([float(x) for x in v])
            else:
                raise TypeError(f"obs dict value for key '{k}' must be numeric or list of numerics, got {type(v)}")
        return np.asarray(vec, dtype=np.float32)

    if isinstance(obs, list):
        if not all(_is_number(v) for v in obs):
            raise TypeError("obs list must contain only numbers")
        return np.asarray([float(v) for v in obs], dtype=np.float32)

    if _is_number(obs):
        return np.asarray([float(obs)], dtype=np.float32)

    raise TypeError(f"Unsupported obs type for vectorization: {type(obs)}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    exps = np.exp(logits - m)
    return exps / np.sum(exps)


class DistilledAgent:
    """
    Distilled policy via behavior cloning for discrete action spaces.

    Config keys in AgentSpec.config:
    - lr: learning rate (default 0.01)
    - epochs: number of training epochs (default 10)
    - obs_keys: list[str] fixed dict key order (optional)
    - init_scale: weight init scale (default 0.01)
    - model_path: load an existing model on init (optional)
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

        if action_space.type != "discrete" or not action_space.n:
            raise ValueError("DistilledAgent requires discrete action space with n")

        cfg = spec.config if isinstance(spec.config, dict) else {}
        self.lr = float(cfg.get("lr", 0.01))
        self.epochs = int(cfg.get("epochs", 10))
        self.init_scale = float(cfg.get("init_scale", 0.01))

        self.obs_keys = cfg.get("obs_keys")
        if self.obs_keys is not None and not isinstance(self.obs_keys, list):
            raise TypeError("config.obs_keys must be a list of strings or omitted")

        self.n_actions = int(action_space.n)
        self._rng = np.random.default_rng()
        self._seed: Optional[int] = None

        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._obs_dim: Optional[int] = None

        model_path = cfg.get("model_path") if isinstance(cfg, dict) else None
        if model_path:
            self.load(str(model_path))

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        if seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(int(seed))

    def act(self, obs: JSONValue) -> ActionResult:
        x = _obs_to_vector(obs, keys=self.obs_keys)
        self._ensure_params(x.shape[0])

        logits = (self._W @ x) + self._b
        probs = _softmax(logits)
        a = int(self._rng.choice(self.n_actions, p=probs))
        return ActionResult(action=a, info={"probs": probs.tolist()})

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        raise NotImplementedError("Use train_from_dataset() for this agent")

    def train_from_dataset(self, dataset_path: str) -> Dict[str, float]:
        X: List[np.ndarray] = []
        A: List[int] = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obs = row.get("obs")
                action = row.get("action")
                x = _obs_to_vector(obs, keys=self.obs_keys)
                X.append(x)
                A.append(int(action))

        if not X:
            return {"samples": 0, "loss": 0.0}

        self._ensure_params(X[0].shape[0])

        Xn = np.stack(X).astype(np.float32)
        A = np.asarray(A, dtype=np.int64)

        losses = []
        for _ in range(self.epochs):
            logits = (self._W @ Xn.T).T + self._b
            probs = np.apply_along_axis(_softmax, 1, logits)

            # Cross entropy loss
            p_a = probs[np.arange(len(A)), A]
            loss = -np.mean(np.log(p_a + 1e-8))
            losses.append(float(loss))

            # Gradients for softmax cross-entropy
            grad = probs.copy()
            grad[np.arange(len(A)), A] -= 1.0
            grad /= float(len(A))

            grad_W = grad.T @ Xn
            grad_b = np.sum(grad, axis=0)

            self._W -= self.lr * grad_W
            self._b -= self.lr * grad_b

        return {"samples": float(len(A)), "loss": float(sum(losses) / max(1, len(losses)))}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "spec": self.spec.to_dict(),
            "obs_dim": self._obs_dim,
            "n_actions": self.n_actions,
            "W": self._W.tolist() if self._W is not None else None,
            "b": self._b.tolist() if self._b is not None else None,
            "obs_keys": self.obs_keys,
        }
        with open(os.path.join(path, "distilled_policy.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        file_path = os.path.join(path, "distilled_policy.json")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self._obs_dim = payload.get("obs_dim")
        self.n_actions = int(payload.get("n_actions", self.n_actions))
        self.obs_keys = payload.get("obs_keys", self.obs_keys)

        W = payload.get("W")
        b = payload.get("b")
        self._W = np.asarray(W, dtype=np.float32) if W is not None else None
        self._b = np.asarray(b, dtype=np.float32) if b is not None else None

    def close(self) -> None:
        return

    def _ensure_params(self, obs_dim: int) -> None:
        if self._W is not None and self._b is not None:
            return
        self._obs_dim = int(obs_dim)
        self._W = (self._rng.standard_normal((self.n_actions, self._obs_dim)).astype(np.float32)) * self.init_scale
        self._b = np.zeros((self.n_actions,), dtype=np.float32)
