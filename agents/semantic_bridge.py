"""
Semantic Bridge: Strategy Value Network

This network learns the Value V(s) of a state based ONLY on abstract strategy features.
It ignores the board and specific mechanics.

Input:  [score_delta, pressure, risk, tempo, control, resource, t]
Output: Expected Return (scalar)

Hypothesis: High 'risk' is bad in ANY game. Investing in 'control' is good in ANY game.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

class StrategyValueNetwork(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Value estimate
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def predict(self, features: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            return self.net(x).item()

    def train_step(self, features: np.ndarray, target_value: float):
        self.optimizer.zero_grad()
        x = torch.FloatTensor(features).unsqueeze(0)
        y = torch.FloatTensor([target_value]).unsqueeze(0)
        pred = self.net(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

def extract_strategy_features(obs: Dict[str, Any]) -> np.ndarray:
    """Extract the 7 universal strategy features from any V2 verse."""
    # Normalize roughly to [-1, 1] range for stability
    return np.array([
        obs.get("score_delta", 0) / 10.0,
        obs.get("pressure", 0) / 10.0,
        obs.get("risk", 0) / 10.0,
        obs.get("tempo", 0) / 10.0,
        obs.get("control", 0) / 10.0,
        obs.get("resource", 0) / 10.0,
        obs.get("t", 0) / 200.0,
    ], dtype=np.float32)
