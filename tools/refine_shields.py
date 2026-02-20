"""
tools/refine_shields.py

Automated safety post-mortem.
Scans recent runs for failures, extracts dangerous (obs, action) pairs, 
and updates the SafetyShield model.
"""

import os
import sys
import json
import torch
import numpy as np
from typing import List, Tuple

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from models.safety_shield import SafetyShield, ShieldNet
from core.safe_executor import _is_safety_violation
from agents.ppo_agent import _obs_to_vector

def collect_failure_data(runs_root: str) -> List[Tuple[np.ndarray, int]]:
    print(f"Scanning {runs_root} for failures...")
    data = []
    
    # Walk through run directories
    for run_id in os.listdir(runs_root):
        run_path = os.path.join(runs_root, run_id)
        events_path = os.path.join(run_path, "events.jsonl")
        
        if not os.path.isfile(events_path):
            continue
            
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    info = ev.get("info", {})
                    # If this step was a safety violation
                    if _is_safety_violation(info):
                        obs = ev.get("obs")
                        action = ev.get("action")
                        if obs is not None and action is not None:
                            vec = _obs_to_vector(obs)
                            data.append((vec, int(action)))
                except Exception:
                    continue
                    
    print(f"Found {len(data)} failure points.")
    return data

def train_shield(data: List[Tuple[np.ndarray, int]], model_path: str, epochs: int = 20):
    if not data:
        return
        
    # Prepare X, Y
    # X = concat(obs, action_one_hot) or just concat(obs, [action])
    X_list = []
    Y_list = []
    
    for obs_vec, action in data:
        # For now, simple concat
        x = np.concatenate([obs_vec, [float(action)]])
        X_list.append(x)
        Y_list.append(1.0) # These are all failures
        
    # We also need negative samples (safe actions)
    # For a robust shield, we'd pull these from the same events.jsonl
    # but for this MVP, we focus on learning the known failures.
    
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32).unsqueeze(1)
    
    input_dim = X.shape[1]
    shield = SafetyShield(input_dim=input_dim)
    
    optimizer = torch.optim.Adam(shield.model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    print(f"Training shield on {input_dim} dims...")
    shield.model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = shield.model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch} loss: {loss.item():.4f}")
            
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    shield.save(model_path)
    print(f"Shield refined and saved to {model_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    data = collect_failure_data(args.runs_root)
    if not data:
        print("No failure data found. Run some episodes first!")
        return
        
    # Group by dimension
    by_dim = {}
    for vec, action in data:
        d = vec.shape[0]
        if d not in by_dim:
            by_dim[d] = []
        by_dim[d].append((vec, action))
        
    for dim, d_list in by_dim.items():
        out_path = f"models/refined_shield_dim{dim}.pth"
        print(f"\nRefining shield for dim={dim} ({len(d_list)} points)...")
        train_shield(d_list, out_path, args.epochs)

if __name__ == "__main__":
    main()

