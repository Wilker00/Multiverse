"""
agents/recurrent_ppo_agent.py

Recurrent PPO Agent using PyTorch (LSTM-based).
Suitable for partially observable environments (POMDPs) like escape_world and trade_world.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Any, Dict, List, Optional, Tuple

from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult, ExperienceBatch, RunningMeanStd


def _obs_to_tensor(obs: JSONValue, keys: Optional[List[str]] = None) -> torch.Tensor:
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
        return torch.tensor(vals, dtype=torch.float32)
    try:
        arr = np.array(obs, dtype=np.float32).flatten()
        return torch.from_numpy(arr)
    except Exception:
        return torch.tensor([0.0], dtype=torch.float32)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # x shape: (batch_size, seq_len, input_dim)
        # hidden: (1, batch_size, hidden_dim) tuple
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        z = self.relu(self.fc1(x))
        if hidden is None:
            # Initialize hidden state if not provided
            device = x.device
            h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
            c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
            hidden = (h0, c0)
            
        out, (h_n, c_n) = self.lstm(z, hidden)
        
        # out shape: (batch_size, seq_len, hidden_dim)
        logits = self.actor(out)
        values = self.critic(out)
        
        return logits, values, (h_n, c_n)


class RecurrentPPOAgent:
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.n_actions = int(action_space.n) if action_space.n else 2
        cfg = spec.config if isinstance(spec.config, dict) else {}

        self.lr = float(cfg.get("lr", 0.0003))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(cfg.get("clip_eps", 0.2))
        self.epochs = int(cfg.get("epochs", 4))
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.obs_keys = cfg.get("obs_keys")
        if not self.obs_keys and observation_space.type == "dict" and observation_space.keys:
            self.obs_keys = sorted(observation_space.keys)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine input dimension
        self.input_dim = 0
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
        
        # If input_dim is still 0, we will lazy init in act()
        if self.input_dim > 0:
            self.model = ActorCritic(self.input_dim, self.n_actions, self.hidden_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.model = None
            self.optimizer = None

        self._rng = np.random.default_rng(spec.seed)
        self._rms = None
        
        # Runtime hidden state (h, c)
        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None 

    def seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            self._rng = np.random.default_rng(seed)

    def _reset_hidden(self):
        self._hidden_state = None
        
    def _init_model(self, dim: int):
        self.input_dim = dim
        self.model = ActorCritic(self.input_dim, self.n_actions, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self._rms = RunningMeanStd(shape=(dim,))

    def act(self, obs: JSONValue) -> ActionResult:
        if isinstance(obs, dict) and obs.get("t", -1) == 0:
            self._reset_hidden()

        x_params = _obs_to_tensor(obs, self.obs_keys)
        if self.model is None:
            self._init_model(x_params.shape[0])
            
        if self._rms:
            x_np = x_params.numpy()
            x_norm = (x_np - self._rms.mean) / (np.sqrt(self._rms.var) + 1e-8)
            x_params = torch.from_numpy(x_norm).float()

        x = x_params.unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, dim)
        
        with torch.no_grad():
            logits, value, next_hidden = self.model(x, self._hidden_state)
            self._hidden_state = next_hidden
            
            # logits: (1, 1, n_actions)
            p = torch.softmax(logits.squeeze(), dim=-1)
            dist = Categorical(probs=p)
            action = dist.sample()
            
            # Map logp and value to the format used in learn()
            return ActionResult(
                action=int(action.item()),
                info={
                    "action_info": {
                        "logp": float(dist.log_prob(action).item()),
                        "value": float(value.squeeze().item())
                    }
                }
            )

    def get_state(self) -> Dict[str, Any]:
        if self._hidden_state is None:
            return {"hidden": None}
        h, c = self._hidden_state
        return {"hidden": (h.detach().cpu().clone(), c.detach().cpu().clone())}

    def set_state(self, state: Dict[str, Any]) -> None:
        hidden = state.get("hidden")
        if hidden is None:
            self._hidden_state = None
        else:
            h, c = hidden
            self._hidden_state = (h.to(self.device), c.to(self.device))


    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        if not batch.transitions:
            return {}

        # 1. Organize data into sequences (Episodes)
        # But ExperienceBatch is a flat list of transitions from rollout.
        # Assuming rollout provides contiguous steps for an episode (or chunks).
        # We will attempt to reconstruct sequences or just treat as one long sequence
        # if from single env. If parallel, this is harder.
        # For simplicity in this codebase, we assume 1 env, contiguous.
        
        transitions = batch.transitions
        
        # Prepare Tensors
        obs_list = []
        act_list = []
        rew_list = []
        val_list = []
        logp_list = []
        done_list = []
        
        for tr in transitions:
            obs_list.append(_obs_to_tensor(tr.obs, self.obs_keys))
            act_list.append(tr.action)
            rew_list.append(tr.reward)
            val_list.append(tr.info["action_info"]["value"])
            logp_list.append(tr.info["action_info"]["logp"])
            done_list.append(float(tr.done or tr.truncated))

        # (Seq_len, Dim)
        obs_t = torch.stack(obs_list)
        if self._rms:
            self._rms.update(obs_t.numpy())
            obs_np = obs_t.numpy()
            obs_norm = (obs_np - self._rms.mean) / (np.sqrt(self._rms.var) + 1e-8)
            obs_t = torch.from_numpy(obs_norm).float()

        obs_t = obs_t.to(self.device)
        act_t = torch.tensor(act_list, dtype=torch.long).to(self.device)
        rew_t = torch.tensor(rew_list, dtype=torch.float32).to(self.device)
        val_t = torch.tensor(val_list, dtype=torch.float32).to(self.device)
        logp_t = torch.tensor(logp_list, dtype=torch.float32).to(self.device)
        done_t = torch.tensor(done_list, dtype=torch.float32).to(self.device)


        # GAE Calculation
        T = len(rew_list)
        adv = torch.zeros(T, device=self.device)
        last_gae = 0.0
        next_val = 0.0
        
        # We need next value for the last step.
        # If done, next_val = 0. If not done, we treat as 0 for simplicity (or bootstrap if we had next_obs)
        # Given we don't have next_obs value easily here without running model, we assume 0 at truncated end 
        # or use the bootstrap value if provided in batch.meta (rare).
        
        for t in reversed(range(T)):
            if done_t[t] > 0.5:
                next_val = 0.0
                last_gae = 0.0
            
            delta = rew_t[t] + self.gamma * next_val - val_t[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            adv[t] = last_gae
            next_val = val_t[t] # Next value for t-1 is current value at t (approx) NO!
            # Wait, next_val should be V(s_{t+1}).
            # In loop: at step t, we need V(s_{t+1}).
            # val_t[t] is V(s_t).
            # So next_val variable holds V(s_{t+1}).
            # At t=T-1 (last step), next_val is 0 (or bootstrap).
            # At t=T-2, next_val is val_t[T-1]. Correct.
            
        returns = adv + val_t
        
        # Normalize Advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO Update
        # For LSTM, we usually process the whole sequence.
        # We will feed the entire collected rollout as one batch with seq_len = T.
        # Hidden state initialization: 0. 
        # Ideally, we should ignore gradients across episode boundaries if multiple episodes.
        # But here we assume one rollout = one chunk.
        
        # Add batch dim: (1, T, Dim)
        b_obs = obs_t.unsqueeze(0)
        b_act = act_t.unsqueeze(0)
        b_logp_old = logp_t.unsqueeze(0)
        b_adv = adv.unsqueeze(0)
        b_ret = returns.unsqueeze(0)
        
        pi_loss_epoch = 0.0
        v_loss_epoch = 0.0
        
        for _ in range(self.epochs):
            # Forward pass full sequence
            # Hidden state starts at 0 each epoch for the batch? 
            # Yes, mimicking the rollout state if we treat rollout as 1 seq.
            logits, v_pred, _ = self.model(b_obs) # (1, T, A), (1, T, 1)
            
            # Remove batch dim for calculation
            logits = logits.squeeze(0) # (T, A)
            v_pred = v_pred.squeeze(0).squeeze(-1) # (T)
            
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_logp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_logp - logp_t)
            
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            
            pi_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            v_loss = 0.5 * ((v_pred - returns) ** 2).mean()
            
            loss = pi_loss + 0.5 * v_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            pi_loss_epoch += pi_loss.item()
            v_loss_epoch += v_loss.item()
            
        return {
            "pi_loss": pi_loss_epoch / self.epochs,
            "v_loss": v_loss_epoch / self.epochs,
            "mean_reward": float(rew_t.mean().item()),
            "advantage_mean": float(adv.mean().item())
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

    def load(self, path: str) -> None:
        p = os.path.join(path, "model.pt")
        if os.path.isfile(p):
            self.model.load_state_dict(
                torch.load(p, map_location=self.device, weights_only=True)
            )
            self.model.eval()

    def close(self) -> None:
        pass
