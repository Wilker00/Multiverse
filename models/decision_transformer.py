"""
models/decision_transformer.py

Minimal Decision Transformer for discrete action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def _causal_mask(seq_len: int, *, device: torch.device) -> torch.Tensor:
    # True entries are masked for nn.TransformerEncoder.
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)


@dataclass
class DecisionTransformerConfig:
    state_dim: int
    action_dim: int
    context_len: int = 20
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    max_timestep: int = 4096
    bos_token_id: Optional[int] = None


class DecisionTransformer(nn.Module):
    """
    Causal sequence model for action prediction.

    Inputs are aligned by timestep:
    - state_t
    - return_to_go_t
    - prev_action_t (BOS for first token)
    - timestep_t
    """

    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        self.config = DecisionTransformerConfig(**dict(config.__dict__))

        if int(self.config.state_dim) <= 0:
            raise ValueError("state_dim must be > 0")
        if int(self.config.action_dim) <= 0:
            raise ValueError("action_dim must be > 0")
        if int(self.config.context_len) <= 0:
            raise ValueError("context_len must be > 0")
        if int(self.config.d_model) <= 0:
            raise ValueError("d_model must be > 0")
        if int(self.config.n_head) <= 0:
            raise ValueError("n_head must be > 0")
        if int(self.config.n_layer) <= 0:
            raise ValueError("n_layer must be > 0")
        if int(self.config.max_timestep) <= 0:
            raise ValueError("max_timestep must be > 0")

        bos = self.config.bos_token_id
        if bos is None:
            bos = int(self.config.action_dim)
        self.bos_token_id = int(bos)
        if self.bos_token_id < 0 or self.bos_token_id > int(self.config.action_dim):
            raise ValueError("bos_token_id must be in [0, action_dim]")
        self.config.bos_token_id = int(self.bos_token_id)

        d_model = int(self.config.d_model)
        self.state_embed = nn.Linear(int(self.config.state_dim), d_model)
        self.return_embed = nn.Linear(1, d_model)
        self.action_embed = nn.Embedding(int(self.config.action_dim) + 1, d_model)
        self.time_embed = nn.Embedding(int(self.config.max_timestep), d_model)
        self.embed_ln = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(float(self.config.dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(self.config.n_head),
            dim_feedforward=4 * d_model,
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=int(self.config.n_layer))
        self.final_ln = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, int(self.config.action_dim))

    def forward(
        self,
        *,
        states: torch.Tensor,
        returns_to_go: torch.Tensor,
        prev_actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns action logits with shape [B, T, action_dim].
        """
        if states.dim() != 3:
            raise ValueError(f"states must be rank-3 [B,T,D], got {tuple(states.shape)}")
        if returns_to_go.dim() != 2:
            raise ValueError(f"returns_to_go must be rank-2 [B,T], got {tuple(returns_to_go.shape)}")
        if prev_actions.dim() != 2:
            raise ValueError(f"prev_actions must be rank-2 [B,T], got {tuple(prev_actions.shape)}")
        if timesteps.dim() != 2:
            raise ValueError(f"timesteps must be rank-2 [B,T], got {tuple(timesteps.shape)}")

        batch_size, seq_len, state_dim = states.shape
        if int(state_dim) != int(self.config.state_dim):
            raise ValueError(f"states last dim mismatch: expected {self.config.state_dim}, got {state_dim}")
        if returns_to_go.shape != (batch_size, seq_len):
            raise ValueError("returns_to_go shape mismatch against states")
        if prev_actions.shape != (batch_size, seq_len):
            raise ValueError("prev_actions shape mismatch against states")
        if timesteps.shape != (batch_size, seq_len):
            raise ValueError("timesteps shape mismatch against states")

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float32, device=states.device)
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask shape mismatch against states")

        prev_actions = prev_actions.long().clamp(0, int(self.config.action_dim))
        timesteps = timesteps.long().clamp(0, int(self.config.max_timestep) - 1)

        x_state = self.state_embed(states)
        x_rtg = self.return_embed(returns_to_go.unsqueeze(-1))
        x_prev = self.action_embed(prev_actions)
        x_time = self.time_embed(timesteps)
        x = self.embed_ln(x_state + x_rtg + x_prev + x_time)
        x = self.embed_drop(x)

        key_padding_mask = attention_mask <= 0.0
        x = self.backbone(
            x,
            mask=_causal_mask(seq_len, device=states.device),
            src_key_padding_mask=key_padding_mask,
        )
        x = self.final_ln(x)
        return self.action_head(x)

    @torch.no_grad()
    def predict_next_action(
        self,
        *,
        states: torch.Tensor,
        returns_to_go: torch.Tensor,
        prev_actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict action for the last valid token in each batch item.
        Returns (actions, confidence, probs_last).
        """
        self.eval()
        logits = self.forward(
            states=states,
            returns_to_go=returns_to_go,
            prev_actions=prev_actions,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        batch_size, seq_len, _ = logits.shape
        if attention_mask is None:
            last_idx = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=logits.device)
        else:
            valid_counts = attention_mask.long().sum(dim=1).clamp(min=1)
            last_idx = valid_counts - 1

        gather_index = last_idx.view(batch_size, 1, 1).expand(batch_size, 1, logits.shape[-1])
        last_logits = logits.gather(dim=1, index=gather_index).squeeze(1)

        temp = max(1e-6, float(temperature))
        last_logits = last_logits / temp
        if int(top_k) > 0 and int(top_k) < int(self.config.action_dim):
            kth_vals = torch.topk(last_logits, k=int(top_k), dim=-1).values[:, -1].unsqueeze(-1)
            last_logits = torch.where(last_logits < kth_vals, torch.full_like(last_logits, -1e9), last_logits)

        probs = torch.softmax(last_logits, dim=-1)
        if bool(sample):
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            actions = torch.argmax(probs, dim=-1)
        conf = probs.gather(1, actions.view(-1, 1)).squeeze(1)
        return actions, conf, probs

    def get_config(self) -> Dict[str, Any]:
        return dict(self.config.__dict__)


def load_decision_transformer_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[DecisionTransformer, Dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    cfg = payload.get("model_config")
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint missing model_config")
    model = DecisionTransformer(DecisionTransformerConfig(**cfg))
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint missing model_state_dict")
    model.load_state_dict(state_dict)
    return model, payload
