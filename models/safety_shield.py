"""
models/safety_shield.py

A lightweight binary classifier trained to predict P(Failure | State, Action).
Acts as a fast-path reflex for SafeExecutor.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class ShieldNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SafetyShield:
    def __init__(self, input_dim: int, checkpoint_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ShieldNet(input_dim).to(self.device)
        self.input_dim = input_dim
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()

    def predict_danger(self, obs_vec: np.ndarray, action: int) -> float:
        # Simple one-hot action concat
        # Note: assumes discrete actions for now
        # We can expand to multi-discrete or continuous if needed.
        
        # obs_vec should be flat
        x = np.concatenate([obs_vec, [float(action)]])
        if len(x) != self.input_dim:
            # Handle mismatch or lazy init
            return 0.0
            
        t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model(t).item()
        return float(prob)

    def train_on_dna(self, transitions: List[Dict[str, Any]], epochs: int = 5):
        # transitions: list of {obs_vec, action, failure_label}
        if not transitions:
            return
            
        x_data = []
        y_data = []
        for tr in transitions:
            vec = tr["obs_vec"]
            act = tr["action"]
            label = tr["label"] # 1.0 for failure, 0.0 for success
            x_data.append(np.concatenate([vec, [float(act)]]))
            y_data.append([float(label)])
            
        X = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        Y = torch.tensor(y_data, dtype=torch.float32).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
        self.model.eval()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

