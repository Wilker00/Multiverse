"""
models/confidence_monitor.py

Lightweight neural danger estimator for SafeExecutor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class ConfidenceMonitorConfig:
    input_dim: int
    hidden_dim: int = 128
    hidden_layers: int = 2
    dropout: float = 0.10

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "ConfidenceMonitorConfig":
        d = data if isinstance(data, dict) else {}
        dropout_raw = d["dropout"] if isinstance(d, dict) and "dropout" in d else 0.10
        return ConfidenceMonitorConfig(
            input_dim=max(2, int(d.get("input_dim", 66) or 66)),
            hidden_dim=max(16, int(d.get("hidden_dim", 128) or 128)),
            hidden_layers=max(1, int(d.get("hidden_layers", 2) or 2)),
            dropout=max(0.0, min(0.7, float(dropout_raw))),
        )


class ConfidenceMonitor(nn.Module):
    def __init__(self, cfg: ConfidenceMonitorConfig):
        if nn is None:
            raise RuntimeError("torch is required for ConfidenceMonitor")
        super().__init__()
        layers = []
        in_dim = int(cfg.input_dim)
        for _ in range(int(cfg.hidden_layers)):
            layers.append(nn.Linear(in_dim, int(cfg.hidden_dim)))
            layers.append(nn.GELU())
            if float(cfg.dropout) > 0.0:
                layers.append(nn.Dropout(float(cfg.dropout)))
            in_dim = int(cfg.hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.cfg = cfg

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_danger_prob(self, x):
        if torch is None:
            raise RuntimeError("torch is required for ConfidenceMonitor")
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def save_confidence_monitor(
    *,
    model: ConfidenceMonitor,
    path: str,
    train_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    if torch is None:
        raise RuntimeError("torch is required for ConfidenceMonitor")
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": int(model.cfg.input_dim),
            "hidden_dim": int(model.cfg.hidden_dim),
            "hidden_layers": int(model.cfg.hidden_layers),
            "dropout": float(model.cfg.dropout),
        },
        "train_config": dict(train_config or {}),
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, path)


def load_confidence_monitor(path: str, *, map_location: str = "cpu") -> ConfidenceMonitor:
    if torch is None:
        raise RuntimeError("torch is required for ConfidenceMonitor")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ConfidenceMonitorConfig.from_dict(ckpt.get("model_config"))
    model = ConfidenceMonitor(cfg)
    model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    model.eval()
    return model
