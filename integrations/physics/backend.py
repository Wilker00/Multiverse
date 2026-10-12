"""
integrations/physics/backend.py

Physics backend protocol for embodied verses.

Reinforcement learning still calls ``Verse.reset`` and ``Verse.step``.
A backend only integrates bodies. NVIDIA Warp, Isaac, or another GPU
simulator should implement this protocol in its own module. Do not fork
``core/rollout.py`` to add a simulator.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol


class PhysicsBackend(Protocol):
    """Minimal integrator used by embodied verses."""

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, float]:
        """Return a fresh body state."""

    def step(self, state: Dict[str, float], action: int) -> Dict[str, float]:
        """Advance one control step and return the next state."""

    def state(self) -> Dict[str, float]:
        """Return the last simulated state."""
