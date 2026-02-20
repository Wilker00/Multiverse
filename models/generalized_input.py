"""
models/generalized_input.py

Realizing the 'Universal Input' concept.
DeepMind-style 'Perceiver' / 'Set Transformer' input encoder to handle arbitrary Verse observations
without fragile flattening or fixed input dimensions.

Mechanism:
1. Decompose JSON observation into a set of (Key, Value) pairs.
2. Embed Keys via hashing/learned dictionary.
3. Embed Values (scalars) via Fourier features or projection.
4. Process the set {Token_i} via Self-Attention or Cross-Attention (Perceiver IO).
5. Aggregation into a latent 'Command Vector' z.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _hash_key(key: str, num_buckets: int) -> int:
    # Deterministic hash for key embedding
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h, 16) % num_buckets


def _extract_numeric_items(
    obs: Any,
    *,
    max_keys: int,
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []

    def _walk(v: Any, key_path: str) -> None:
        if len(out) >= max_keys:
            return
        if isinstance(v, dict):
            # Sort for deterministic traversal independent of insertion order.
            for k in sorted(v.keys(), key=lambda x: str(x)):
                child_key = f"{key_path}.{k}" if key_path else str(k)
                _walk(v[k], child_key)
            return
        if isinstance(v, (list, tuple)):
            for i, child in enumerate(v):
                child_key = f"{key_path}[{i}]" if key_path else f"[{i}]"
                _walk(child, child_key)
            return
        try:
            vf = float(v)
        except (ValueError, TypeError):
            return
        if not math.isfinite(vf):
            return
        leaf_key = key_path if key_path else "$"
        out.append((leaf_key, vf))

    _walk(obs, "")
    return out


class GeneralizedInputEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_embd: int = 256,
        num_key_buckets: int = 1024,
        max_keys: int = 128,
        num_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.num_key_buckets = num_key_buckets
        self.max_keys = max_keys

        # Embeddings for Keys (concept identity)
        self.key_embed = nn.Embedding(num_key_buckets, n_embd)
        
        # Embeddings for Values (concept magnitude)
        # Using a simple projection for now, could act as a gate
        self.val_proj = nn.Linear(1, n_embd)

        # Perceiver-like Encoder: Process the set of (Key+Value) embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, 
            nhead=num_heads, 
            dim_feedforward=n_embd * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output query (Latent Bottleneck)
        # We learn a single "Summary Token" (or a few) that queries the set.
        self.latents = nn.Parameter(torch.randn(1, 1, n_embd))
        self.cross_attn = nn.MultiheadAttention(n_embd, num_heads, batch_first=True)
        
        # Final LayerNorm
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, observations: List[Any]) -> torch.Tensor:
        """
        Args:
            observations: Batch of JSON-like observations. Each sample can vary in structure.
        Returns:
            Tensor [B, n_embd] representing the state.
        """
        device = self.key_embed.weight.device
        batch_size = len(observations)
        if batch_size <= 0:
            return torch.zeros((0, self.n_embd), dtype=torch.float32, device=device)
        
        # 1. Batched Tensor Construction (Pad to max_keys)
        # This part is CPU-heavy; in production DeepMind code this would be optimized 
        # via custom collators or sparse tensors.
        
        # keys_idx: [B, max_keys]
        # vals: [B, max_keys, 1]
        # mask: [B, max_keys] (True = padding)

        keys_list = []
        vals_list = []
        masks_list = []

        for obs in observations:
            items = _extract_numeric_items(obs, max_keys=self.max_keys)
            cur_limit = len(items)

            k_tensor = torch.tensor(
                [_hash_key(k, self.num_key_buckets) for k, _ in items], 
                dtype=torch.long, device=device
            )
            if cur_limit > 0:
                v_tensor = torch.tensor(
                    [[v] for _, v in items],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                v_tensor = torch.zeros((0, 1), dtype=torch.float32, device=device)
            
            # Padding
            pad_len = self.max_keys - cur_limit
            if pad_len > 0:
                k_pad = torch.zeros(pad_len, dtype=torch.long, device=device)
                v_pad = torch.zeros((pad_len, 1), dtype=torch.float32, device=device)
                mask_entry = torch.cat([
                    torch.zeros(cur_limit, dtype=torch.bool, device=device),
                    torch.ones(pad_len, dtype=torch.bool, device=device),
                ])
                keys_list.append(torch.cat([k_tensor, k_pad]))
                vals_list.append(torch.cat([v_tensor, v_pad]))
                masks_list.append(mask_entry)
            else:
                keys_list.append(k_tensor)
                vals_list.append(v_tensor)
                masks_list.append(torch.zeros(self.max_keys, dtype=torch.bool, device=device))

        keys_enc = torch.stack(keys_list) # [B, S]
        vals_enc = torch.stack(vals_list) # [B, S, 1]
        mask_enc = torch.stack(masks_list) # [B, S]

        # 2. Embedding
        # E = KeyEmb(k) + ValProj(v)
        # Note: In more advanced versions, we modulate KeyEmb by Value magnitude
        k_emb = self.key_embed(keys_enc) # [B, S, E]
        v_emb = self.val_proj(vals_enc)  # [B, S, E]
        
        x = k_emb + v_emb # [B, S, E]

        # 3. Processing (Self-Attention within the Set)
        # Mask out padding
        x = self.transformer(x, src_key_padding_mask=mask_enc)

        # 4. Latent Query (Cross-Attention)
        # We query the processed set x with our learned Latent Vector
        # latents: [1, 1, E] -> [B, 1, E]
        latents = self.latents.repeat(batch_size, 1, 1)
        
        # attn_output: [B, 1, E]
        z, _ = self.cross_attn(
            query=latents, 
            key=x, 
            value=x, 
            key_padding_mask=mask_enc
        )
        
        z = self.ln(z.squeeze(1)) # [B, E]
        return z
