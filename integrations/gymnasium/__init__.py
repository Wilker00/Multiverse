"""
Optional Gymnasium integration surface.

This package is lazy-imported so Multiverse core can run
without Gymnasium installed.
"""

from .adapter import (
    GymnasiumUnavailableError,
    GymnasiumVerseAdapter,
    GymnasiumVerseFactory,
    build_gymnasium_install_hint,
    gymnasium_available,
    require_gymnasium,
)

__all__ = [
    "GymnasiumUnavailableError",
    "GymnasiumVerseAdapter",
    "GymnasiumVerseFactory",
    "build_gymnasium_install_hint",
    "gymnasium_available",
    "require_gymnasium",
]
