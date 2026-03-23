"""
Optional PettingZoo multi-agent integration surface.

This package is lazy-imported so Multiverse core can run
without PettingZoo installed.
"""

from .adapter import (
    PettingZooUnavailableError,
    PettingZooVerseAdapter,
    PettingZooVerseFactory,
    build_pettingzoo_install_hint,
    pettingzoo_available,
    require_pettingzoo,
)

__all__ = [
    "PettingZooUnavailableError",
    "PettingZooVerseAdapter",
    "PettingZooVerseFactory",
    "build_pettingzoo_install_hint",
    "pettingzoo_available",
    "require_pettingzoo",
]
