"""
models/contrastive_bridge.py

Contrastive cross-verse bridge model built on top of GeneralizedInputEncoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generalized_input import GeneralizedInputEncoder


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: bool = False) -> bool:
    try:
        if x is None:
            return bool(default)
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return bool(x)
    except Exception:
        return default


@dataclass
class ContrastiveBridgeConfig:
    n_embd: int = 256
    proj_dim: int = 256
    temperature_init: float = 0.07
    temperature_min: float = 0.01
    temperature_max: float = 1.0
    max_keys: int = 128
    num_key_buckets: int = 1024
    num_heads: int = 4
    n_layers: int = 2
    strict_config_validation: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "ContrastiveBridgeConfig":
        raw = dict(cfg) if isinstance(cfg, dict) else {}
        if not raw:
            return cls()
        strict = _safe_bool(raw.get("strict_config_validation"), default=True)
        known = {
            "n_embd",
            "proj_dim",
            "temperature_init",
            "temperature_min",
            "temperature_max",
            "max_keys",
            "num_key_buckets",
            "num_heads",
            "n_layers",
            "strict_config_validation",
        }
        unknown = sorted(k for k in raw.keys() if k not in known)
        if strict and unknown:
            raise ValueError(f"Unknown ContrastiveBridgeConfig key(s): {', '.join(unknown)}")
        return cls(
            n_embd=max(8, _safe_int(raw.get("n_embd", 256), 256)),
            proj_dim=max(8, _safe_int(raw.get("proj_dim", raw.get("n_embd", 256)), 256)),
            temperature_init=max(1e-4, _safe_float(raw.get("temperature_init", 0.07), 0.07)),
            temperature_min=max(1e-4, _safe_float(raw.get("temperature_min", 0.01), 0.01)),
            temperature_max=max(1e-4, _safe_float(raw.get("temperature_max", 1.0), 1.0)),
            max_keys=max(1, _safe_int(raw.get("max_keys", 128), 128)),
            num_key_buckets=max(8, _safe_int(raw.get("num_key_buckets", 1024), 1024)),
            num_heads=max(1, _safe_int(raw.get("num_heads", 4), 4)),
            n_layers=max(1, _safe_int(raw.get("n_layers", 2), 2)),
            strict_config_validation=bool(strict),
        )


class ContrastiveBridge(nn.Module):
    """
    Dual-observation contrastive model with a shared universal input encoder.
    """

    def __init__(
        self,
        *,
        config: Optional[ContrastiveBridgeConfig] = None,
        n_embd: int = 256,
        proj_dim: Optional[int] = None,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        if config is None:
            cfg = ContrastiveBridgeConfig(
                n_embd=int(n_embd),
                proj_dim=int(proj_dim if proj_dim is not None else n_embd),
                temperature_init=float(temperature_init),
            )
        else:
            cfg = config
        self.config = cfg

        if cfg.n_embd <= 0:
            raise ValueError("n_embd must be > 0")
        if cfg.proj_dim <= 0:
            raise ValueError("proj_dim must be > 0")
        if cfg.temperature_min <= 0 or cfg.temperature_max <= 0:
            raise ValueError("temperature bounds must be > 0")
        if cfg.temperature_min > cfg.temperature_max:
            raise ValueError("temperature_min must be <= temperature_max")

        self.encoder = GeneralizedInputEncoder(
            n_embd=cfg.n_embd,
            num_key_buckets=cfg.num_key_buckets,
            max_keys=cfg.max_keys,
            num_heads=cfg.num_heads,
            n_layers=cfg.n_layers,
        )
        self.projector = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.proj_dim),
        )

        # Parameterized in log-space for stable optimization.
        init_scale = 1.0 / float(max(1e-8, cfg.temperature_init))
        self.logit_scale_log = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_scale)).item())))

    def _clamped_logit_scale(self) -> torch.Tensor:
        min_scale = 1.0 / float(self.config.temperature_max)
        max_scale = 1.0 / float(self.config.temperature_min)
        return torch.clamp(torch.exp(self.logit_scale_log), min=min_scale, max=max_scale)

    def encode(self, observations: List[Any]) -> torch.Tensor:
        if not isinstance(observations, list):
            raise ValueError("observations must be a list")
        x = self.encoder(observations)
        z = self.projector(x)
        return F.normalize(z, dim=-1)

    # Compatibility alias.
    def embed(self, observations: List[Any]) -> torch.Tensor:
        return self.encode(observations)

    def similarity(self, obs_a: List[Any], obs_b: List[Any]) -> torch.Tensor:
        za = self.encode(obs_a)
        zb = self.encode(obs_b)
        if za.shape[0] == 0 or zb.shape[0] == 0:
            return za.new_zeros((int(za.shape[0]), int(zb.shape[0])))
        return torch.matmul(za, zb.transpose(0, 1)) * self._clamped_logit_scale()

    def contrastive_loss(self, logits_ab: torch.Tensor) -> torch.Tensor:
        if logits_ab.dim() != 2:
            raise ValueError("logits_ab must be rank-2 [B,B]")
        if logits_ab.shape[0] != logits_ab.shape[1]:
            raise ValueError("contrastive loss expects square logits [B,B]")
        labels = torch.arange(int(logits_ab.shape[0]), device=logits_ab.device, dtype=torch.long)
        loss_i = F.cross_entropy(logits_ab, labels)
        loss_t = F.cross_entropy(logits_ab.transpose(0, 1), labels)
        return 0.5 * (loss_i + loss_t)

    def forward(
        self,
        obs_a: List[Any],
        obs_b: List[Any],
        *,
        return_loss: bool = False,
    ) -> Any:
        logits = self.similarity(obs_a, obs_b)
        if return_loss:
            return {"logits": logits, "loss": self.contrastive_loss(logits)}
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {
            "n_embd": int(self.config.n_embd),
            "proj_dim": int(self.config.proj_dim),
            "temperature_init": float(self.config.temperature_init),
            "temperature_min": float(self.config.temperature_min),
            "temperature_max": float(self.config.temperature_max),
            "max_keys": int(self.config.max_keys),
            "num_key_buckets": int(self.config.num_key_buckets),
            "num_heads": int(self.config.num_heads),
            "n_layers": int(self.config.n_layers),
            "strict_config_validation": bool(self.config.strict_config_validation),
        }

    def save(self, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "model_config": self.get_config(),
        }
        if isinstance(extra, dict):
            payload["extra"] = dict(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, *, map_location: str = "cpu") -> "ContrastiveBridge":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid checkpoint format: {path}")
        cfg = ContrastiveBridgeConfig.from_dict(ckpt.get("model_config"))
        model = cls(config=cfg)
        model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
        model.eval()
        return model
