"""
agents/simple_pg_agent.py

A tiny, self contained policy gradient agent for discrete actions.

Why this exists:
- No external RL libraries required
- Fits the u.ai Agent interface
- Good enough to replace RandomAgent for "exploration with learning"
- Teaches you the mechanics of policy gradients without magic

Algorithm:
- REINFORCE with a baseline (moving average of returns)
- Linear softmax policy: pi(a|x) = softmax(Wx + b)
- Updates W and b using advantage weighted log probability gradients

Constraints:
- Discrete action spaces only
- Observations must be featurizable into a fixed length vector
  - Supports dict of scalar ints/floats
  - Supports list of numbers
  - Supports single int/float
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import os
import json

import numpy as np

from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult, ExperienceBatch, Transition


@dataclass
class PGStats:
    steps_seen: int = 0
    baseline_return: float = 0.0


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _obs_to_vector(obs: JSONValue, keys: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert JSONValue observation into a 1D float vector.

    Supported:
    - dict with scalar values
    - list of numbers
    - scalar number

    If dict:
    - if keys provided, uses that fixed order
    - else sorts keys for deterministic ordering

    Returns float32 vector.
    """
    if isinstance(obs, dict):
        if keys is None:
            keys = sorted(obs.keys())
        vec: List[float] = []
        for k in keys:
            v = obs.get(k)
            if _is_number(v):
                vec.append(float(v))
            else:
                raise TypeError(f"Observation dict value for key '{k}' must be a number, got {type(v)}")
        return np.asarray(vec, dtype=np.float32)

    if isinstance(obs, list):
        if not all(_is_number(v) for v in obs):
            raise TypeError("Observation list must contain only numbers")
        return np.asarray([float(v) for v in obs], dtype=np.float32)

    if _is_number(obs):
        return np.asarray([float(obs)], dtype=np.float32)

    raise TypeError(f"Unsupported observation type for featurization: {type(obs)}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    exps = np.exp(logits - m)
    return exps / np.sum(exps)


class SimplePolicyGradientAgent:
    """
    Minimal policy gradient agent.

    Config keys in AgentSpec.config:
    - lr: float learning rate (default 0.01)
    - gamma: float discount (default 0.99)
    - entropy_coef: float entropy bonus (default 0.0)
    - baseline_alpha: float moving average smoothing (default 0.05)
    - obs_keys: list[str] fixed ordering for dict obs (optional)
    - init_scale: float weight init scale (default 0.01)
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

        if action_space.type != "discrete" or not action_space.n:
            raise ValueError("SimplePolicyGradientAgent requires discrete action space with n")

        cfg = spec.config if isinstance(spec.config, dict) else {}

        self.lr = float(cfg.get("lr", 0.01))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.0))
        self.baseline_alpha = float(cfg.get("baseline_alpha", 0.05))
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

        self.stats = PGStats()

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
        info = {
            "probs": probs.tolist(),
        }
        return ActionResult(action=a, info=info)

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        if not batch.transitions:
            return {}

        # Build episode returns from the batch transitions in order
        # Assumption: transitions are contiguous from one episode
        obs_list: List[np.ndarray] = []
        act_list: List[int] = []
        rew_list: List[float] = []

        for tr in batch.transitions:
            x = _obs_to_vector(tr.obs, keys=self.obs_keys)
            obs_list.append(x)
            act_list.append(int(tr.action))
            rew_list.append(float(tr.reward))

        self._ensure_params(obs_list[0].shape[0])

        # Compute discounted returns G_t
        returns = np.zeros(len(rew_list), dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(rew_list))):
            G = rew_list[t] + self.gamma * G
            returns[t] = G

        # Baseline: moving average of returns
        batch_return_mean = float(np.mean(returns))
        if self.stats.steps_seen == 0:
            self.stats.baseline_return = batch_return_mean
        else:
            self.stats.baseline_return = (
                (1.0 - self.baseline_alpha) * self.stats.baseline_return
                + self.baseline_alpha * batch_return_mean
            )

        advantages = returns - float(self.stats.baseline_return)

        # Gradient ascent on expected return
        # For softmax linear policy:
        # grad_W = (one_hot(a) - probs) outer x
        # We multiply by advantage
        grad_W = np.zeros_like(self._W, dtype=np.float32)
        grad_b = np.zeros_like(self._b, dtype=np.float32)

        entropy_sum = 0.0

        for x, a, adv in zip(obs_list, act_list, advantages):
            logits = (self._W @ x) + self._b
            probs = _softmax(logits)

            one = np.zeros(self.n_actions, dtype=np.float32)
            one[a] = 1.0

            # Policy gradient term
            diff = one - probs  # shape (A,)
            grad_W += (adv * diff[:, None]) * x[None, :]
            grad_b += (adv * diff)

            # Optional entropy bonus gradient approximation
            # We only report entropy here, not apply its gradient, to keep MVP simple and stable.
            entropy_sum += float(-np.sum(probs * np.log(probs + 1e-8)))

        # Normalize by batch size for stable updates
        n = float(len(obs_list))
        grad_W /= n
        grad_b /= n

        # Update parameters
        self._W += self.lr * grad_W
        self._b += self.lr * grad_b

        self.stats.steps_seen += len(obs_list)

        return {
            "batch_size": len(obs_list),
            "return_mean": float(np.mean(returns)),
            "adv_mean": float(np.mean(advantages)),
            "baseline": float(self.stats.baseline_return),
            "entropy_mean": float(entropy_sum / max(1.0, n)),
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "spec": self.spec.to_dict(),
            "obs_dim": self._obs_dim,
            "n_actions": self.n_actions,
            "W": self._W.tolist() if self._W is not None else None,
            "b": self._b.tolist() if self._b is not None else None,
            "obs_keys": self.obs_keys,
            "stats": {
                "steps_seen": self.stats.steps_seen,
                "baseline_return": self.stats.baseline_return,
            },
        }
        with open(os.path.join(path, "simple_pg.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        file_path = os.path.join(path, "simple_pg.json")
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

        st = payload.get("stats", {}) or {}
        self.stats.steps_seen = int(st.get("steps_seen", 0))
        self.stats.baseline_return = float(st.get("baseline_return", 0.0))

    def close(self) -> None:
        return

    def _ensure_params(self, obs_dim: int) -> None:
        if self._W is not None and self._b is not None:
            return

        self._obs_dim = int(obs_dim)
        self._W = (self._rng.standard_normal((self.n_actions, self._obs_dim)).astype(np.float32)) * self.init_scale
        self._b = np.zeros((self.n_actions,), dtype=np.float32)
