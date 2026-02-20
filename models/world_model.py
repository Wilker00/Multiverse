"""
models/world_model.py

A learned dynamics model for Model Predictive Control (MPC).
Predicts (Next State, Reward, Done) given (State, Action).
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from core.types import JSONValue


class DynamicsModel(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Input: state + action
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Outputs
        self.next_state_head = nn.Linear(hidden_dim, input_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        z = self.relu(self.fc1(x))
        z = self.relu(self.fc2(z))
        
        next_state = self.next_state_head(z)
        reward = self.reward_head(z)
        done_logit = self.done_head(z)
        
        return next_state, reward, done_logit


class WorldModel:
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128, lr: float = 1e-3, device: str = "cpu"):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        self.network = DynamicsModel(input_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
        self.network.eval()
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # One-hot encoding for discrete action
            a_t = torch.zeros(1, self.action_dim, device=self.device)
            a_t[0, action] = 1.0
            
            next_s, rew, done_logit = self.network(s_t, a_t)
            
            next_state = next_s.cpu().numpy()[0]
            reward = float(rew.item())
            done = bool(torch.sigmoid(done_logit).item() > 0.5)
            
            return next_state, reward, done

    def update(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray, dones: np.ndarray) -> float:
        self.network.train()
        
        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        
        # One-hot actions
        a = torch.zeros(len(actions), self.action_dim, device=self.device)
        a[np.arange(len(actions)), actions] = 1.0
        
        ns_target = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        r_target = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        d_target = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        ns_pred, r_pred, d_logit = self.network(s, a)
        
        loss_ns = nn.MSELoss()(ns_pred, ns_target)
        loss_r = nn.MSELoss()(r_pred, r_target)
        loss_d = nn.BCEWithLogitsLoss()(d_logit, d_target)
        
        total_loss = loss_ns + loss_r + loss_d
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return float(total_loss.item())

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        if os.path.isfile(path):
            self.network.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
