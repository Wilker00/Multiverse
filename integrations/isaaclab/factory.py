"""
Lazy import / registration helpers for Isaac Lab.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from integrations.isaaclab.config import DEFAULT_ISAACLAB_VERSE_BINDINGS, get_isaaclab_binding


class IsaacLabUnavailableError(RuntimeError):
    pass


def _try_import(module_name: str) -> bool:
    try:
        importlib.import_module(str(module_name))
        return True
    except Exception:
        return False


def isaaclab_available() -> bool:
    return bool(_try_import("isaaclab") or _try_import("omni.isaac.lab"))


def build_isaaclab_install_hint() -> str:
    return (
        "Isaac Lab integration is optional. Install Isaac Sim first, then Isaac Lab, "
        "in a separate environment, and keep Multiverse core free of those dependencies."
    )


def require_isaaclab() -> None:
    if isaaclab_available():
        return
    raise IsaacLabUnavailableError(build_isaaclab_install_hint())


class IsaacLabWarehouseNavFactory:
    @property
    def tags(self) -> List[str]:
        binding = get_isaaclab_binding("isaac_warehouse_nav")
        return list(binding.tags)

    def create(self, spec: Any) -> Any:
        from integrations.isaaclab.adapter import IsaacLabVerseAdapter

        binding = get_isaaclab_binding(str(getattr(spec, "verse_name", "isaac_warehouse_nav")))
        return IsaacLabVerseAdapter(spec=spec, binding=binding)


def register_isaaclab_verses() -> Dict[str, Any]:
    """
    Register concrete Isaac-backed verse names when Isaac is available.
    """
    require_isaaclab()
    from verses.registry import list_verses, register_verse

    registered: List[str] = []
    factories = {
        "isaac_warehouse_nav": IsaacLabWarehouseNavFactory(),
    }
    existing = set(list_verses().keys())
    for name, factory in factories.items():
        if name in existing:
            continue
        register_verse(name, factory)
        registered.append(name)
    return {
        "registered": registered,
        "available_bindings": sorted(DEFAULT_ISAACLAB_VERSE_BINDINGS.keys()),
        "note": "Isaac Lab verses are registered lazily and only when the dependency is available.",
    }
