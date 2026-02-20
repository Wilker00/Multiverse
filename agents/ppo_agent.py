"""
agents/ppo_agent.py

Advanced PPO agent for Multiverse with MLP Backpropagation.
Includes:
- Multi-Layer Perceptron (MLP) with ReLU activation
- Pure NumPy Backpropagation and PPO Update logic
- Observation Normalization (Running Mean/Std)
- Gradient Clipping & Logit Clipping
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult, ExperienceBatch, RunningMeanStd

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def _d_relu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def _softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(logits - m)
    return exps / np.sum(exps, axis=-1, keepdims=True)




def _obs_to_vector(obs: JSONValue, keys: Optional[List[str]] = None) -> np.ndarray:
    if isinstance(obs, dict):
        if keys:
            vals = []
            for k in keys:
                v = obs.get(k, 0)
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(0.0)
            return np.array(vals, dtype=np.float32)
        # Fallback: sort keys
        vals = []
        for k in sorted(obs.keys()):
            v = obs[k]
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(0.0)
        return np.array(vals, dtype=np.float32)
    try:
        return np.array(obs, dtype=np.float32).flatten()
    except Exception:
        return np.zeros((1,), dtype=np.float32)


class PPOAgent:
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.n_actions = int(action_space.n)
        cfg = spec.config if isinstance(spec.config, dict) else {}

        self.lr = float(cfg.get("lr", 0.0003))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(cfg.get("clip_eps", 0.2))
        self.epochs = int(cfg.get("epochs", 10))
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.reward_clip = float(cfg.get("reward_clip", 10.0))
        self.value_clip = float(cfg.get("value_clip", 100.0))
        self.ratio_clip = float(cfg.get("ratio_clip", 10.0))
        self.obs_keys = cfg.get("obs_keys")

        self._rng = np.random.default_rng(spec.seed)
        self._rms = None
        self._params = {}

    def seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _init_mlp(self, dim_in: int):
        d_h, d_out = self.hidden_dim, self.n_actions
        self._params = {
            "W1": self._rng.standard_normal((d_h, dim_in), dtype=np.float32) * np.sqrt(2/dim_in),
            "b1": np.zeros((d_h, 1), dtype=np.float32),
            "W_pi": self._rng.standard_normal((d_out, d_h), dtype=np.float32) * np.sqrt(2/d_h),
            "b_pi": np.zeros((d_out, 1), dtype=np.float32),
            "W_v": self._rng.standard_normal((1, d_h), dtype=np.float32) * np.sqrt(2/d_h),
            "b_v": np.zeros((1, 1), dtype=np.float32),
        }
        self._rms = RunningMeanStd(shape=(dim_in,))

    def _forward(self, x: np.ndarray):
        z1 = (self._params["W1"] @ x.T) + self._params["b1"]
        h = _relu(z1)
        logits = (self._params["W_pi"] @ h) + self._params["b_pi"]
        v = (self._params["W_v"] @ h) + self._params["b_v"]
        return logits.T, v.T, h.T, z1.T

    @staticmethod
    def _clip_grad(name_to_grad: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
        if max_norm <= 0.0:
            return name_to_grad
        total_sq = 0.0
        for g in name_to_grad.values():
            total_sq += float(np.sum(np.square(g)))
        total_norm = float(np.sqrt(max(0.0, total_sq)))
        if not np.isfinite(total_norm) or total_norm <= max_norm:
            return name_to_grad
        scale = float(max_norm / (total_norm + 1e-8))
        return {k: (v * scale).astype(np.float32) for k, v in name_to_grad.items()}

    def _params_finite(self) -> bool:
        for p in self._params.values():
            if not np.all(np.isfinite(p)):
                return False
        return True

    def act(self, obs: JSONValue) -> ActionResult:
        from agents.ppo_agent import _obs_to_vector
        x_raw = _obs_to_vector(obs, keys=self.obs_keys)
        if not self._params: self._init_mlp(x_raw.shape[0])
        x = self._rms.normalize(x_raw)
        logits, v, _, _ = self._forward(x.reshape(1, -1))
        # Clip logits to avoid overflow
        logits = np.clip(logits, -20.0, 20.0)
        probs = _softmax(logits[0])
        
        # Ensure numerical stability
        probs = np.nan_to_num(probs, nan=1.0/self.n_actions)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones_like(probs) / len(probs)
            
        a = int(self._rng.choice(self.n_actions, p=probs))
        return ActionResult(action=a, info={"logp": float(np.log(probs[a] + 1e-8)), "value": float(v[0,0])})

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        if not batch.transitions: return {}
        from agents.ppo_agent import _obs_to_vector
        from core.agent_base import apply_her
        
        transitions = list(batch.transitions)
        if self.spec.config.get("her_enabled"):
            # Generate synthetic hindsight transitions
            her_trans = apply_her(transitions, strategy="final")
            transitions.extend(her_trans)

        X_raw = np.stack([_obs_to_vector(tr.obs, keys=self.obs_keys) for tr in transitions])

        self._rms.update(X_raw)
        X = self._rms.normalize(X_raw)
        A = np.array([tr.action for tr in transitions])
        R = np.array([tr.reward for tr in transitions], dtype=np.float32)
        R = np.clip(R, -self.reward_clip, self.reward_clip)
        D = np.array([tr.done or tr.truncated for tr in transitions])
        
        # Pull logp/value from tr.info - HER synthetic transitions carry these from the parent
        Logp_old = []
        V_old = []
        for tr in transitions:
            a_info = tr.info.get("action_info") or {}
            Logp_old.append(float(a_info.get("logp", 0.0)))
            V_old.append(float(a_info.get("value", 0.0)))
            
        Logp_old = np.array(Logp_old, dtype=np.float32)
        V_old = np.array(V_old, dtype=np.float32)
        Logp_old = np.nan_to_num(Logp_old, nan=0.0, posinf=0.0, neginf=0.0)
        V_old = np.nan_to_num(V_old, nan=0.0, posinf=0.0, neginf=0.0)


        # GAE Advantages
        T = len(R)
        adv = np.zeros(T); last_gae = 0; next_v = 0
        for t in reversed(range(T)):
            if D[t]: next_v = 0; last_gae = 0
            delta = R[t] + self.gamma * next_v - V_old[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            adv[t] = last_gae
            next_v = V_old[t]
        returns = adv + V_old
        returns = np.clip(returns, -self.value_clip, self.value_clip)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)

        for _ in range(self.epochs):
            logits, v_pred, h, z1 = self._forward(X)
            logits = np.clip(logits, -20.0, 20.0)
            probs = _softmax(logits)
            probs = np.nan_to_num(probs, nan=1.0 / max(1, self.n_actions), posinf=1.0, neginf=0.0)
            probs = np.clip(probs, 1e-8, 1.0)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            p_a = probs[np.arange(T), A]
            ratio = np.exp(np.clip(np.log(p_a + 1e-8) - Logp_old, -self.ratio_clip, self.ratio_clip))

            # PPO Clip Loss
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv
            pi_loss = -np.minimum(surr1, surr2).mean()
            v_loss = 0.5 * np.square(returns - v_pred.flatten()).mean()
            pi_loss = float(np.nan_to_num(pi_loss, nan=0.0, posinf=0.0, neginf=0.0))
            v_loss = float(np.nan_to_num(v_loss, nan=0.0, posinf=0.0, neginf=0.0))

            # Backprop (Simplified)
            d_logits = probs.copy()
            d_logits[np.arange(T), A] -= 1.0
            surr_coeff = -np.where(surr1 < surr2, ratio, np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)) * adv
            surr_coeff = np.nan_to_num(surr_coeff, nan=0.0, posinf=0.0, neginf=0.0)
            d_logits *= surr_coeff[:, None] / T

            d_v = (v_pred.flatten() - returns)[:, None] / T
            d_v = np.clip(np.nan_to_num(d_v, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

            # Update Policy & Value Heads
            grad_w_pi = d_logits.T @ h
            grad_b_pi = np.sum(d_logits.T, axis=1, keepdims=True)
            grad_w_v = d_v.T @ h
            grad_b_v = np.sum(d_v.T, axis=1, keepdims=True)

            # Update Hidden Layer (use current weights for local linearization)
            d_h = (d_logits @ self._params["W_pi"]) + (d_v @ self._params["W_v"])
            d_z1 = d_h * _d_relu(z1)
            grad_w1 = d_z1.T @ X
            grad_b1 = np.sum(d_z1.T, axis=1, keepdims=True)

            grads = {
                "W1": np.nan_to_num(grad_w1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                "b1": np.nan_to_num(grad_b1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                "W_pi": np.nan_to_num(grad_w_pi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                "b_pi": np.nan_to_num(grad_b_pi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                "W_v": np.nan_to_num(grad_w_v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                "b_v": np.nan_to_num(grad_b_v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
            }
            grads = self._clip_grad(grads, self.max_grad_norm)

            for k, g in grads.items():
                self._params[k] = (self._params[k] - self.lr * g).astype(np.float32)

            if not self._params_finite():
                self._init_mlp(X.shape[1])
                return {"pi_loss": float(pi_loss), "v_loss": float(v_loss), "reset_on_nan": True}

        return {"pi_loss": float(pi_loss), "v_loss": float(v_loss)}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for k, v in self._params.items(): np.save(os.path.join(path, f"{k}.npy"), v)

    def load(self, path: str) -> None:
        for k in ["W1", "b1", "W_pi", "b_pi", "W_v", "b_v"]: self._params[k] = np.load(os.path.join(path, f"{k}.npy"))

    def close(self) -> None:
        pass
