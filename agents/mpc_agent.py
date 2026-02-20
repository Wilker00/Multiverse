"""
agents/mpc_agent.py

Model Predictive Control (MPC) agent using a learned World Model.
Uses Random Shooting (or CEM-lite) for planning.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple

from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult, ExperienceBatch
from models.world_model import WorldModel


def _obs_to_vector(obs: JSONValue, keys: Optional[List[str]] = None) -> np.ndarray:
    if isinstance(obs, dict):
        vals = []
        # Use provided keys or sort
        ks = keys if keys else sorted(obs.keys())
        for k in ks:
            v = obs.get(k, 0)
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(0.0)
        return np.array(vals, dtype=np.float32)
    try:
        return np.array(obs, dtype=np.float32).flatten()
    except Exception:
        return np.zeros((1,), dtype=np.float32)


class MPCAgent:
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.n_actions = int(action_space.n) if action_space.n else 4
        cfg = spec.config if isinstance(spec.config, dict) else {}

        self.lr = float(cfg.get("lr", 1e-3))
        self.horizon = int(cfg.get("horizon", 5))
        self.num_samples = int(cfg.get("num_samples", 64))  # Number of random trajectories
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.obs_keys = cfg.get("obs_keys")
        
        # Lazy init for input dim
        self.input_dim = 0
        self.world_model: Optional[WorldModel] = None

        if not self.obs_keys and observation_space.type == "dict" and observation_space.keys:
            self.obs_keys = sorted(observation_space.keys)
            
        # Try to determine dim eagerly
        if observation_space.type == "vector" and observation_space.shape:
            self.input_dim = int(np.prod(observation_space.shape))
        elif observation_space.type == "dict" and self.obs_keys and observation_space.subspaces:
            # Try to pre-calculate dim from subspaces
            dim = 0
            valid = True
            for k in self.obs_keys:
                sub = observation_space.subspaces.get(k)
                if sub and sub.shape:
                    dim += int(np.prod(sub.shape))
                elif sub:
                    dim += 1
                else:
                    valid = False
            if valid:
                self.input_dim = dim
                
        self._rng = np.random.default_rng(spec.seed)
        
        # If eager init possible
        if self.input_dim > 0:
            self.world_model = WorldModel(self.input_dim, self.n_actions, self.hidden_dim, self.lr)

    def seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _init_model(self, dim: int):
        self.input_dim = dim
        self.world_model = WorldModel(self.input_dim, self.n_actions, self.hidden_dim, self.lr)

    def act(self, obs: JSONValue) -> ActionResult:
        x = _obs_to_vector(obs, self.obs_keys)
        
        if self.world_model is None:
            self._init_model(x.shape[0])
            
        # MPC Planning: Random Shooting
        best_action = 0
        best_return = -float("inf")
        
        # We model-predict for 'num_samples' trajectories of length 'horizon'
        # For efficiency, we ideally batch this. But the simple WorldModel is mainly single-step.
        # Let's use simple loop for clarity, or batch manually if WorldModel supported it.
        # Our WorldModel is PyTorch, so we can manual batch.
        
        # Generate random action sequences: (num_samples, horizon)
        action_seqs = self._rng.integers(0, self.n_actions, size=(self.num_samples, self.horizon))
        
        # Initial state repeated
        # We need to access the internal network for batch prediction or loop.
        # Let's loop for simplicity/correctness first to avoid shape mismatch hell.
        # Optimization: We only care about the first action really.
        
        returns = np.zeros(self.num_samples)
        
        # Optimization: Group by first action? No, just evaluate.
        
        # To make it fast, we really should use torch batching.
        # obs vector -> tensor (1, dim) -> repeat (num_samples, dim)
        state_t = torch.tensor(x, dtype=torch.float32, device=self.world_model.device).unsqueeze(0).repeat(self.num_samples, 1)
        cumulative_rewards = torch.zeros(self.num_samples, device=self.world_model.device)
        dones = torch.zeros(self.num_samples, dtype=torch.bool, device=self.world_model.device)
        
        with torch.no_grad():
            for t in range(self.horizon):
                # Actions at step t: (num_samples,)
                actions_t = torch.tensor(action_seqs[:, t], dtype=torch.long, device=self.world_model.device)
                
                # One hot
                actions_oh = torch.zeros(self.num_samples, self.n_actions, device=self.world_model.device)
                actions_oh.scatter_(1, actions_t.unsqueeze(1), 1.0)
                
                # Predict
                # Reuse the network directly
                next_s, rew, done_logit = self.world_model.network(state_t, actions_oh)
                
                # Update
                # If already done, reward is 0 (or masked).
                # But our model predicts raw reward.
                # Let's assume reward is valid unless done was previously true.
                
                # Sigmoid for done
                step_dones = torch.sigmoid(done_logit).squeeze(1) > 0.5
                
                # Mask reward if previously done
                # current 'dones' vector is from t-1
                mask = ~dones
                cumulative_rewards += rew.squeeze(1) * mask.float()
                
                # Update state and done status
                state_t = next_s
                dones = dones | step_dones
            
        # CPU
        plan_returns = cumulative_rewards.cpu().numpy()
        
        # Pick best trajectory
        best_idx = np.argmax(plan_returns)
        best_action = action_seqs[best_idx, 0]
        
        # Epsilon greedy for exploration?
        if self._rng.random() < 0.1:
            best_action = self._rng.integers(0, self.n_actions)

        return ActionResult(
            action=int(best_action), 
            info={"plan_return": float(plan_returns[best_idx]), "mpc_horizon": self.horizon}
        )

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        if not batch.transitions:
            return {}
            
        if self.world_model is None:
            # Need to figure out dimension from first obs
            x = _obs_to_vector(batch.transitions[0].obs, self.obs_keys)
            self._init_model(x.shape[0])

        # Prepare vectors
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for tr in batch.transitions:
            states.append(_obs_to_vector(tr.obs, self.obs_keys))
            actions.append(tr.action)
            rewards.append(tr.reward)
            next_states.append(_obs_to_vector(tr.next_obs, self.obs_keys))
            dones.append(float(tr.done)) # Truncated doesn't mean terminal for physics, usually. But for done flag prediction it does.
            
        s = np.array(states, dtype=np.float32)
        a = np.array(actions, dtype=np.int64)
        r = np.array(rewards, dtype=np.float32)
        ns = np.array(next_states, dtype=np.float32)
        d = np.array(dones, dtype=np.float32)
        
        loss = self.world_model.update(s, a, ns, r, d)
        
        return {"world_model_loss": float(loss)}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if self.world_model:
            self.world_model.save(os.path.join(path, "world_model.pt"))

    def load(self, path: str) -> None:
        p = os.path.join(path, "world_model.pt")
        if self.world_model and os.path.isfile(p):
            self.world_model.load(p)

    def close(self) -> None:
        pass
