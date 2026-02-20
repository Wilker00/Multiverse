"""
orchestrator/ray_adapter.py

Optional Ray/RLlib integration stub.
"""

from __future__ import annotations

from typing import Any, Dict


def ensure_ray() -> None:
    try:
        import ray  # type: ignore
    except Exception as exc:
        raise RuntimeError("Ray is not installed. Install ray[rllib] to enable scaling.") from exc


def run_with_ray(config: Dict[str, Any]) -> None:
    """
    Placeholder for Ray/RLlib execution.
    This module is intentionally minimal to avoid hard dependencies.
    """
    ensure_ray()
    # This is a stub. Plug in RLlib Trainer configs here.
    raise NotImplementedError("Ray/RLlib integration is not yet wired to core trainers.")
