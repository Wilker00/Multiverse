"""
Optional Isaac Lab / Isaac Sim integration surface.

This package is intentionally lazy-imported so Multiverse core can run
without NVIDIA simulator dependencies installed.
"""

from .factory import (
    IsaacLabUnavailableError,
    build_isaaclab_install_hint,
    isaaclab_available,
    register_isaaclab_verses,
)

__all__ = [
    "IsaacLabUnavailableError",
    "build_isaaclab_install_hint",
    "isaaclab_available",
    "register_isaaclab_verses",
]
