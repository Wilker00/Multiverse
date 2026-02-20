"""
agents/curious_agent.py

A PPO agent augmented with an Intrinsic Curiosity Module (ICM).
Adds intrinsic rewards for novel/unpredictable states to encourage exploration.
"""

from __future__ import annotations

import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult, ExperienceBatch
from models.curiosity_module import ICM


def _obs_to_tensor(obs: JSONValue, keys: Optional[List[str]] = None) -> torch.Tensor:
    if isinstance(obs, dict):
        vals = []
        ks = keys if keys else sorted(obs.keys())
        for k in ks:
            v = obs.get(k, 0)
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(0.0)
        return torch.tensor(vals, dtype=torch.float32)
    try:
        # If valid list/array
        return torch.tensor(obs, dtype=torch.float32).flatten()
    except Exception:
        return torch.zeros(1, dtype=torch.float32)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)


class CuriousAgent:
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        cfg = spec.config if isinstance(spec.config, dict) else {}

        # Hyperparameters
        self.lr = float(cfg.get("lr", 3e-4))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(cfg.get("clip_eps", 0.2))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.epochs = int(cfg.get("epochs", 4))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        
        # ICM Hyperparameters
        self.icm_lr = float(cfg.get("icm_lr", 1e-3))
        self.icm_beta = float(cfg.get("icm_beta", 0.2))  # Weight of forward model loss vs inverse
        self.intrinsic_scale = float(cfg.get("intrinsic_scale", 0.1)) # Scaling factor for intrinsic reward

        self.n_actions = int(action_space.n) if action_space.n else 4
        self.obs_keys = cfg.get("obs_keys")
        if not self.obs_keys and observation_space.type == "dict" and observation_space.keys:
            self.obs_keys = sorted(observation_space.keys)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # determine input dim 
        dummy_dim = 4
        if observation_space.type == "vector" and observation_space.shape:
            dummy_dim = int(np.prod(observation_space.shape))
        elif observation_space.type == "dict" and self.obs_keys and observation_space.subspaces:
            d = 0
            for k in self.obs_keys:
                sub = observation_space.subspaces.get(k)
                if sub and sub.shape:
                    d += int(np.prod(sub.shape))
                else: 
                     d+=1
            dummy_dim = d
            
        self.input_dim = dummy_dim
        
        # PPO Network
        self.policy = ActorCritic(self.input_dim, self.n_actions, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # ICM Network
        self.icm = ICM(self.input_dim, self.n_actions, self.hidden_dim).to(self.device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=self.icm_lr)

        self._rng = np.random.default_rng(spec.seed)

    def seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            self._rng = np.random.default_rng(seed)

    def act(self, obs: JSONValue) -> ActionResult:
        # PPO Act logic
        x = _obs_to_tensor(obs, self.obs_keys).to(self.device)
        
        # Check dim on first run? 
        # For simplicity, assume fixed dim. If mismatch, torch will error eventually.
        # But we can relu on lazy init if needed. Let's trust constructor logic for now.
        
        with torch.no_grad():
            logits, val = self.policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return ActionResult(
            action=int(action.item()),
            info={"val": float(val.item()), "log_prob": float(log_prob.item())}
        )

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        if not batch.transitions:
            return {}

        # 1. Prepare data
        states = []
        actions = []
        next_states = []
        ext_rewards = []
        dones = []
        old_log_probs = []
        
        for t in batch.transitions:
            states.append(_obs_to_tensor(t.obs, self.obs_keys))
            next_states.append(_obs_to_tensor(t.next_obs, self.obs_keys))
            actions.append(t.action)
            ext_rewards.append(t.reward)
            dones.append(t.done)
            old_log_probs.append(t.info.get("log_prob", 0.0) if t.info else 0.0)

        # To Tensor
        states_t = torch.stack(states).to(self.device)
        next_states_t = torch.stack(next_states).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        ext_rewards_t = torch.tensor(ext_rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)

        # 2. Calculate Intrinsic Rewards & ICM Loss
        # We process the whole batch through ICM
        # Note: In pure online PPO, we might want to do this step-by-step or minibatch, 
        # but doing it on full collected batch is fine.
        
        phi_next, pred_phi_next, pred_action_logits = self.icm(states_t, next_states_t, actions_t)
        
        # Inverse Loss: Cross Entropy between predicted action and actual action
        inv_loss = F.cross_entropy(pred_action_logits, actions_t)
        
        # Forward Loss: MSE between predicted next feature and actual next feature
        # (normalized by 0.5 usually)
        fwd_loss = 0.5 * F.mse_loss(pred_phi_next, phi_next.detach())
        
        # Intrinsic Reward = Forward Error (per sample)
        # We need per-sample error for rewards.
        # F.mse_loss with reduction='none'
        per_sample_mse = 0.5 * torch.sum((pred_phi_next - phi_next.detach())**2, dim=1)
        intrinsic_rewards = self.intrinsic_scale * per_sample_mse.detach()
        
        total_rewards = ext_rewards_t + intrinsic_rewards
        
        # Update ICM
        icm_loss = (self.icm_beta * fwd_loss) + ((1.0 - self.icm_beta) * inv_loss)
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()
        
        # 3. PPO Update (using Total Rewards)
        # Recalculate advantages/returns using new total_rewards
        with torch.no_grad():
            _, values = self.policy(states_t)
            values = values.squeeze()
            # GAE
            advantages = torch.zeros_like(total_rewards)
            last_gae = 0.0
            # We treat the batch as one contiguous trajectory or use dones. Use dones.
            # Assuming batch is ordered chronologically.
            
            # Simple assumption: full batch is a sequence. 
            # Ideally we handle multiple episodes correctly.
            # Here we just iterate backwards. 
            
            for t in reversed(range(len(total_rewards))):
                if t == len(total_rewards) - 1:
                    next_val = 0.0 # simplified
                else:
                    next_val = values[t + 1]
                    
                delta = total_rewards[t] + self.gamma * next_val * (1 - dones_t[t]) - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1 - dones_t[t]) * last_gae
                advantages[t] = last_gae
                
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Epochs
        idxs = np.arange(len(states))
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]
                
                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]
                b_old_lp = old_log_probs_t[batch_idx]
                
                logits, v = self.policy(b_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(v.squeeze(), b_ret)
                
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "icm_loss": float(icm_loss.item()),
            "mean_int_reward": float(intrinsic_rewards.mean().item()),
            "mean_ext_reward": float(ext_rewards_t.mean().item())
        }

    def save(self, path: str) -> None:
        pass # Optional

    def load(self, path: str) -> None:
        pass # Optional

    def close(self) -> None:
        pass
