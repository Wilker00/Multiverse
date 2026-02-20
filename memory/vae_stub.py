"""
memory/vae_stub.py

Minimal encoder/decoder stub for DNA compression.
This is not a trained VAE; it provides a deterministic latent space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int
    seed: int = 123


class VAEStub:
    def __init__(self, cfg: VAEConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self._W_enc = rng.standard_normal((cfg.latent_dim, cfg.input_dim)).astype(np.float32) * 0.1
        self._W_dec = rng.standard_normal((cfg.input_dim, cfg.latent_dim)).astype(np.float32) * 0.1

    def encode(self, x: List[float]) -> List[float]:
        v = np.asarray(x, dtype=np.float32)
        z = self._W_enc @ v
        return z.tolist()

    def decode(self, z: List[float]) -> List[float]:
        v = np.asarray(z, dtype=np.float32)
        x = self._W_dec @ v
        return x.tolist()
