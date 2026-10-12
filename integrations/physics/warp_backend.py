"""
integrations/physics/warp_backend.py

GPU simulation plug point for NVIDIA Warp.

Importing this module does not import Warp. Constructing ``WarpPhysicsBackend``
raises ``WarpUnavailableError`` with an install hint when the package is
missing. The RL trainer keeps using ``ToyPhysicsBackend`` unless a verse asks
for this backend explicitly.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, Optional

from integrations.physics.toy_backend import integrate_point_mass


class WarpUnavailableError(RuntimeError):
    """NVIDIA Warp is not installed. The GPU backend is optional."""


def warp_backend_available() -> bool:
    try:
        return importlib.util.find_spec("warp") is not None
    except Exception:
        return False


class WarpPhysicsBackend:
    """
    Thin Warp-backed integrator.

    When Warp is present, each step round-trips the state through a Warp array
    and then uses the same point-mass equations as the toy backend. Replace
    ``integrate_point_mass`` here with a Warp kernel for a real GPU simulator.
    """

    def __init__(self) -> None:
        if not warp_backend_available():
            raise WarpUnavailableError(
                "NVIDIA Warp is not installed. This GPU simulation backend is a "
                "plug point, not a required dependency. Install warp-lang to enable "
                "it, or keep using integrations.physics.toy_backend.ToyPhysicsBackend."
            )
        import warp as wp

        self._wp = wp
        try:
            wp.init()
        except Exception:
            pass
        self._state: Dict[str, float] = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}

    def _round_trip(self, state: Dict[str, float]) -> Dict[str, float]:
        wp = self._wp
        try:
            values = [float(state.get(key, 0.0)) for key in ("x", "y", "vx", "vy")]
            device_values = wp.array(values, dtype=float)
            copied = [float(v) for v in device_values.numpy().tolist()]
            return {"x": copied[0], "y": copied[1], "vx": copied[2], "vy": copied[3]}
        except Exception:
            return {
                "x": float(state.get("x", 0.0)),
                "y": float(state.get("y", 0.0)),
                "vx": float(state.get("vx", 0.0)),
                "vy": float(state.get("vy", 0.0)),
            }

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, float]:
        del seed
        self._state = self._round_trip({"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0})
        return dict(self._state)

    def step(self, state: Dict[str, float], action: int) -> Dict[str, float]:
        current = self._round_trip(state)
        self._state = integrate_point_mass(current, action)
        return dict(self._state)

    def state(self) -> Dict[str, float]:
        return dict(self._state)
