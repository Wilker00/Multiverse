"""
models/meta_transformer.py

Shared latent "meta" policy model across verses.
Updated with GeneralizedInputEncoder for universal observation processing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from models.generalized_input import GeneralizedInputEncoder
except ImportError:
    GeneralizedInputEncoder = None


class MetaTransformer(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        n_embd: int = 256,
        context_layers: int = 1,
        dropout: float = 0.0,
        use_generalized_input: bool = False,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_embd = int(n_embd)
        self.context_input_dim = int(n_embd + 1 + 1) if use_generalized_input else int(state_dim + 1 + 1)
        self.use_generalized_input = bool(use_generalized_input)

        if self.use_generalized_input:
            if GeneralizedInputEncoder is None:
                raise ImportError("GeneralizedInputEncoder not found. Cannot use generalized input.")
            self.input_encoder = GeneralizedInputEncoder(n_embd=self.n_embd)
            # Legacy encoder is not used if generalized input is active, 
            # but we keep it conditionally if state_dim > 0 for hybrid support? 
            # For strictness, if specialized, we skip common_encoder.
            self.common_encoder = None
        else:
            if state_dim <= 0:
                raise ValueError("state_dim must be > 0")
            self.input_encoder = None
            self.common_encoder = nn.Sequential(
                nn.Linear(self.state_dim, self.n_embd),
                nn.GELU(),
                nn.Dropout(float(max(0.0, dropout))),
                nn.Linear(self.n_embd, self.n_embd),
            )

        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if n_embd <= 0:
            raise ValueError("n_embd must be > 0")
        if context_layers <= 0:
            raise ValueError("context_layers must be > 0")

        self.context_encoder = nn.GRU(
            input_size=self.context_input_dim,
            hidden_size=self.n_embd,
            num_layers=int(context_layers),
            batch_first=True,
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.n_embd * 2),
            nn.Linear(self.n_embd * 2, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.n_embd * 2),
            nn.Linear(self.n_embd * 2, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, 1),
            nn.Tanh(),
        )

    def _encode(
        self,
        state: Optional[torch.Tensor] = None,
        recent_history: Optional[torch.Tensor] = None,
        raw_obs: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        # 1. Encode State (h)
        h: torch.Tensor
        batch_size = 0
        device = None

        if self.use_generalized_input:
            if raw_obs is None:
                # Fallback: if state is provided (maybe already embedded?), handle it?
                # For now enforce raw_obs
                raise ValueError("MetaTransformer configured for Generalized Input but 'raw_obs' (List[Dict]) was not provided.")
            
            # Generalized Encoder
            # This handles [B, n_embd]
            # We assume the first tensor in history determines device if available, 
            # otherwise we rely on module device (handled inside encoder)
            h = self.input_encoder(raw_obs)
            batch_size = h.shape[0]
            device = h.device
        else:
            if state is None:
                raise ValueError("MetaTransformer require 'state' tensor.")
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if state.shape[-1] != self.state_dim:
                raise ValueError(f"Expected state_dim={self.state_dim}, got {state.shape[-1]}")
            h = self.common_encoder(state)
            batch_size = state.shape[0]
            device = state.device

        # 2. Encode History (z)
        if recent_history is None:
            recent_history = torch.zeros(
                (batch_size, 1, self.context_input_dim),
                dtype=torch.float32,
                device=device,
            )
        elif recent_history.dim() == 2:
            recent_history = recent_history.unsqueeze(0)
        if recent_history.dim() != 3:
            raise ValueError(
                f"recent_history must be rank-3 [B,T,D], got shape={tuple(recent_history.shape)}"
            )
        if recent_history.shape[-1] != self.context_input_dim:
            raise ValueError(
                f"recent_history feature dim mismatch: expected {self.context_input_dim}, got {recent_history.shape[-1]}"
            )
        if recent_history.shape[0] != batch_size:
            raise ValueError(
                f"recent_history batch mismatch: expected {batch_size}, got {recent_history.shape[0]}"
            )
        if recent_history.device != device:
            recent_history = recent_history.to(device)

        _, z = self.context_encoder(recent_history)
        z_last = z[-1]  # [B, n_embd]

        # 3. Fuse
        return torch.cat([h, z_last], dim=-1)

    def forward(
        self,
        state: Optional[torch.Tensor] = None,
        recent_history: Optional[torch.Tensor] = None,
        raw_obs: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        features = self._encode(state, recent_history, raw_obs=raw_obs)
        logits = self.policy_head(features)
        return logits

    def forward_value(
        self,
        state: Optional[torch.Tensor] = None,
        recent_history: Optional[torch.Tensor] = None,
        raw_obs: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        features = self._encode(state, recent_history, raw_obs=raw_obs)
        return self.value_head(features).squeeze(-1)

    def forward_policy_value(
        self,
        state: Optional[torch.Tensor] = None,
        recent_history: Optional[torch.Tensor] = None,
        raw_obs: Optional[List[Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self._encode(state, recent_history, raw_obs=raw_obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return {"logits": logits, "value": value}

    def predict(
        self,
        *,
        state: Optional[torch.Tensor] = None,
        recent_history: Optional[torch.Tensor] = None,
        raw_obs: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            out = self.forward_policy_value(state, recent_history, raw_obs=raw_obs)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            conf, action = torch.max(probs, dim=-1)
            value = out["value"]
        return {"action": action, "confidence": conf, "probs": probs, "value": value}

    def get_config(self) -> Dict[str, Any]:
        return {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "n_embd": int(self.n_embd),
            "context_input_dim": int(self.context_input_dim),
            "has_value_head": 1,
            "use_generalized_input": int(self.use_generalized_input),
        }
