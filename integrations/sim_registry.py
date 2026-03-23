"""
integrations/sim_registry.py

Registry for simulator providers that Multiverse can orchestrate.

This is a control-plane abstraction, not a replacement simulator.
The local provider is always available; external providers remain optional.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class SimProviderSpec:
    provider_id: str
    label: str
    kind: str
    external: bool
    description: str
    modules: Tuple[str, ...] = ()
    render_modes: Tuple[str, ...] = ()
    capabilities: Tuple[str, ...] = ()
    supported_verses: Tuple[str, ...] = ()
    install_hint: str = ""
    implemented: bool = False


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(str(name)) is not None
    except Exception:
        return False


_ALL_BUILTIN_VERSES: Tuple[str, ...] = (
    "line_world",
    "grid_world",
    "cliff_world",
    "labyrinth_world",
    "park_world",
    "pursuit_world",
    "warehouse_world",
    "chess_world",
    "chess_world_v2",
    "go_world",
    "go_world_v2",
    "uno_world",
    "uno_world_v2",
    "harvest_world",
    "bridge_world",
    "swamp_world",
    "escape_world",
    "factory_world",
    "trade_world",
    "memory_vault_world",
    "rule_flip_world",
    "risk_tutorial_world",
    "wind_master_world",
    "maze_world",
    "world_3d",
)

_PROVIDERS: Tuple[SimProviderSpec, ...] = (
    SimProviderSpec(
        provider_id="multiverse_local",
        label="Multiverse Local Visual",
        kind="local",
        external=False,
        description="Built-in lightweight visual/local verse backend using the native Verse interface.",
        render_modes=("ansi", "rgb_array", "human"),
        capabilities=("single_agent", "deterministic_seed", "lightweight_visual", "native_verse"),
        supported_verses=_ALL_BUILTIN_VERSES,
        install_hint="Included with the repo.",
        implemented=True,
    ),
    SimProviderSpec(
        provider_id="gymnasium",
        label="Gymnasium Adapter",
        kind="adapter",
        external=True,
        description="Adapter lane for Gymnasium environments behind the Multiverse Verse contract. "
                    "Pass env_id in verse params to select any registered Gymnasium environment.",
        modules=("gymnasium",),
        render_modes=("ansi", "human", "rgb_array"),
        capabilities=("single_agent", "adapter", "benchmark_friendly"),
        supported_verses=(),
        install_hint="Install `gymnasium` in the active environment: pip install gymnasium",
        implemented=True,
    ),
    SimProviderSpec(
        provider_id="pettingzoo",
        label="PettingZoo Adapter",
        kind="adapter",
        external=True,
        description="Adapter lane for multi-agent PettingZoo AEC environments. "
                    "Exposes a focal-agent single-agent Verse view; other agents are auto-played.",
        modules=("pettingzoo",),
        render_modes=("ansi", "human", "rgb_array"),
        capabilities=("multi_agent", "adapter", "focal_agent"),
        supported_verses=(),
        install_hint="Install `pettingzoo` in the active environment: pip install pettingzoo",
        implemented=True,
    ),
    SimProviderSpec(
        provider_id="isaaclab",
        label="Isaac Lab / Isaac Sim",
        kind="high_fidelity",
        external=True,
        description="High-fidelity robotics simulator lane for warehouse/logistics integration.",
        modules=("isaaclab", "omni.isaac.lab"),
        render_modes=("rgb_array", "human"),
        capabilities=("robotics", "high_fidelity", "sim2real"),
        supported_verses=("isaac_warehouse_nav",),
        install_hint="Install Isaac Sim first, then Isaac Lab, in a separate optional environment.",
        implemented=True,
    ),
)


def list_sim_providers() -> List[SimProviderSpec]:
    return list(_PROVIDERS)


def get_sim_provider(provider_id: str) -> SimProviderSpec:
    key = str(provider_id or "").strip().lower()
    for provider in _PROVIDERS:
        if provider.provider_id == key:
            return provider
    known = ", ".join(sorted(p.provider_id for p in _PROVIDERS))
    raise KeyError(f"Unknown sim provider '{provider_id}'. Known: {known}")


def provider_available(provider: SimProviderSpec) -> bool:
    if not provider.external:
        return True
    if not provider.modules:
        return False
    return any(_module_available(name) for name in provider.modules)


def provider_status(provider_id: str) -> Dict[str, Any]:
    provider = get_sim_provider(provider_id)
    return {
        "provider_id": provider.provider_id,
        "label": provider.label,
        "kind": provider.kind,
        "external": bool(provider.external),
        "available": bool(provider_available(provider)),
        "implemented": bool(provider.implemented),
        "description": provider.description,
        "render_modes": list(provider.render_modes),
        "capabilities": list(provider.capabilities),
        "supported_verses": list(provider.supported_verses),
        "modules": list(provider.modules),
        "install_hint": provider.install_hint,
    }


def list_sim_provider_status() -> List[Dict[str, Any]]:
    return [provider_status(provider.provider_id) for provider in _PROVIDERS]
