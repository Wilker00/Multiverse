"""
models/curiosity_module.py

Intrinsic Curiosity Module (ICM) implementation.
Reference: "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ICM(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 128, feature_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        
        # 1. Feature Encoder (Phi)
        # Maps raw observation to a latent feature space relevant for dynamics
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        # 2. Forward Dynamics Model
        # Predicts next feature state given current feature state and action
        # phi(st), at -> phi(st+1)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        # 3. Inverse Dynamics Model
        # Predicts action taken given current feature state and next feature state
        # phi(st), phi(st+1) -> at
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(
        self, 
        state: torch.Tensor, 
        next_state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            phi_next_state, pred_phi_next_state, pred_action_logits
        """
        # Encode states
        phi_state = self.encoder(state)
        phi_next_state = self.encoder(next_state)
        
        # --- Inverse Model ---
        # Concatenate phi(st), phi(st+1)
        inv_input = torch.cat([phi_state, phi_next_state], dim=-1)
        pred_action_logits = self.inverse_model(inv_input)
        
        # --- Forward Model ---
        
        # Handle action tensor shape (batch,) -> (batch, n_actions)
        if action.dim() == 1:
            action_indices = action.long()
        elif action.dim() == 2 and action.shape[1] == 1:
            action_indices = action.long().squeeze(1)
        else:
            # Assume already processed or different format, but standard PPO passes (batch, 1) or (batch,)
            action_indices = action.long().squeeze()
            
        # Safety for index out of bounds if action space mismatch
        action_indices = torch.clamp(action_indices, 0, self.n_actions - 1)
        
        action_one_hot = F.one_hot(action_indices, num_classes=self.n_actions).float()
            
        fwd_input = torch.cat([phi_state, action_one_hot], dim=-1)
        pred_phi_next_state = self.forward_model(fwd_input)
        
        return phi_next_state, pred_phi_next_state, pred_action_logits
