"""
integrations/physics/toy_backend.py

CPU point-mass in a box.

This is the always-on embodied prototype. It is not a GPU simulator and it
does not replace the native grid verses. Swap in ``WarpPhysicsBackend`` when
NVIDIA Warp is installed.
"""

from __future__ import annotations

from typing import Dict, Optional


DT = 0.1
THRUST = 1.5
DAMPING = 0.9
BOUND = 1.0


def integrate_point_mass(state: Dict[str, float], action: int) -> Dict[str, float]:
    """Shared integrator so the Warp stub can round-trip the same equations."""
    x = float(state.get("x", 0.0))
    y = float(state.get("y", 0.0))
    vx = float(state.get("vx", 0.0))
    vy = float(state.get("vy", 0.0))
    ax = 0.0
    ay = 0.0
    act = int(action)
    if act == 1:
        ax = THRUST
    elif act == 2:
        ax = -THRUST
    elif act == 3:
        ay = THRUST
    elif act == 4:
        ay = -THRUST
    vx = (vx + ax * DT) * DAMPING
    vy = (vy + ay * DT) * DAMPING
    x = max(-BOUND, min(BOUND, x + vx * DT))
    y = max(-BOUND, min(BOUND, y + vy * DT))
    return {"x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy)}


class ToyPhysicsBackend:
    """Local CPU backend. Implements ``PhysicsBackend`` without optional deps."""

    def __init__(self) -> None:
        self._state: Dict[str, float] = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, float]:
        del seed
        self._state = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}
        return dict(self._state)

    def step(self, state: Dict[str, float], action: int) -> Dict[str, float]:
        self._state = integrate_point_mass(state, action)
        return dict(self._state)

    def state(self) -> Dict[str, float]:
        return dict(self._state)
