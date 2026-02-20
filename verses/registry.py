"""
verses/registry.py

Central registry for Verse factories.

Why this exists:
- Orchestrator should NOT import specific environments directly.
- You want plug-and-play verses (including open source ones) without changing trainer code.

Pattern:
- Each verse package registers a factory under a name.
- Orchestrator calls create_verse(VerseSpec) and gets a Verse.

Keep this simple now. Add entrypoints/plugins later.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Dict

from core.taxonomy import cognitive_tags_for_verse
from core.types import VerseSpec
from core.verse_base import Verse, VerseFactory
from orchestrator.curriculum_controller import apply_curriculum_params, load_curriculum_adjustments

# Global registry: verse_name -> factory
_FACTORIES: Dict[str, VerseFactory] = {}


def _is_numeric(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _default_adr_keys(verse_name: str) -> list[str]:
    v = str(verse_name).strip().lower()
    if v == "line_world":
        return ["goal_pos", "step_penalty", "friction", "gravity"]
    if v == "grid_world":
        return ["step_penalty"]
    if v == "cliff_world":
        return ["step_penalty", "cliff_penalty", "width", "height"]
    if v == "park_world":
        return ["goal_pos", "step_penalty", "park_reward", "wrong_park_penalty"]
    if v == "pursuit_world":
        return ["step_penalty", "catch_reward"]

    if v == "warehouse_world":
        return ["step_penalty", "obstacle_penalty", "battery_drain", "charge_rate", "width", "height"]
    if v == "labyrinth_world":
        return [
            "step_penalty",
            "wall_penalty",
            "pit_penalty",
            "laser_penalty",
            "battery_drain",
            "action_noise",
        ]
    if v == "chess_world":
        return ["step_penalty", "convert_bonus", "random_swing", "win_material"]
    if v == "go_world":
        return ["step_penalty", "convert_bonus", "random_swing", "target_territory"]
    if v == "uno_world":
        return ["step_penalty", "convert_bonus", "random_swing", "start_hand", "opp_start_hand"]
    if v == "harvest_world":
        return ["step_penalty", "fruit_reward", "deposit_reward", "obstacle_penalty", "spoil_probability"]
    if v == "bridge_world":
        return ["step_penalty", "wind_probability", "collapse_prob_weak", "collapse_prob_strong", "repair_penalty"]
    if v == "swamp_world":
        return ["step_penalty", "flood_penalty", "mud_slip_prob", "flood_rate"]
    if v == "escape_world":
        return ["step_penalty", "spotted_penalty", "exit_reward", "guard_vision", "hiding_cooldown"]
    if v == "factory_world":
        return ["step_penalty", "completion_reward", "breakdown_prob", "arrival_rate", "overflow_penalty"]
    if v == "trade_world":
        return ["step_penalty", "transaction_cost", "price_amplitude", "shock_probability", "shock_magnitude"]
    if v == "memory_vault_world":
        return ["step_penalty", "wall_penalty", "goal_reward", "wall_density", "width", "height"]
    if v == "rule_flip_world":
        return ["step_penalty", "target_reward", "wrong_target_penalty", "flip_step", "track_len"]
    if v == "risk_tutorial_world":
        return ["step_penalty", "risk_floor_start", "all_in_threshold", "target_control"]
    if v == "wind_master_world":
        return ["step_penalty", "gust_probability", "edge_penalty", "margin_reward_scale", "target_margin"]
    return []


def _apply_adr(spec: VerseSpec) -> VerseSpec:
    """
    Automatic Domain Randomization wrapper.
    Mutates selected numeric verse params by +/- jitter (default 10%).
    """
    params = dict(spec.params or {})
    enabled_raw = params.get("adr_enabled", True)
    enabled = bool(enabled_raw) if isinstance(enabled_raw, bool) else str(enabled_raw).lower() not in ("0", "false", "no")
    if not enabled:
        return spec

    jitter = float(params.get("adr_jitter", 0.10))
    jitter = max(0.0, min(0.9, jitter))
    if jitter <= 0.0:
        return spec

    keys = params.get("adr_keys")
    if isinstance(keys, list) and keys:
        adr_keys = [str(k) for k in keys]
    else:
        adr_keys = _default_adr_keys(spec.verse_name)
    if not adr_keys:
        return spec

    seed_value = spec.seed if isinstance(spec.seed, int) and not isinstance(spec.seed, bool) else None
    rng = random.Random(seed_value) if seed_value is not None else random.Random()
    new_params = dict(params)
    for k in adr_keys:
        if k not in params:
            continue
        v = params.get(k)
        if not _is_numeric(v):
            continue
        delta = rng.uniform(-jitter, jitter)
        new_v = float(v) * (1.0 + delta)
        if isinstance(v, int):
            # Keep positive integer fields valid.
            if k in ("goal_pos", "max_steps", "lane_len", "width", "height"):
                new_params[k] = max(1, int(round(new_v)))
            else:
                new_params[k] = int(round(new_v))
        else:
            new_params[k] = float(new_v)

    return dataclasses.replace(spec, params=new_params)


def register_verse(name: str, factory: VerseFactory) -> None:
    """
    Register a VerseFactory under a canonical verse name.

    Example:
        register_verse("line_world", LineWorldFactory())
    """
    key = name.strip()
    if not key:
        raise ValueError("Verse name cannot be empty")
    if key in _FACTORIES:
        raise ValueError(f"Verse '{key}' is already registered")
    _FACTORIES[key] = factory


def get_factory(name: str) -> VerseFactory:
    key = name.strip()
    if key not in _FACTORIES:
        known = ", ".join(sorted(_FACTORIES.keys())) or "(none)"
        raise KeyError(f"Unknown verse '{key}'. Registered: {known}")
    return _FACTORIES[key]


def create_verse(spec: VerseSpec) -> Verse:
    """
    Create a Verse from its spec using the registered factory.
    """
    params = dict(spec.params or {})
    try:
        adj = load_curriculum_adjustments()
        params = apply_curriculum_params(
            verse_name=str(spec.verse_name),
            params=params,
            adjustments=adj,
        )
    except Exception:
        pass
    spec_curr = dataclasses.replace(spec, params=params)
    spec_eff = _apply_adr(spec_curr)
    mem_tags = cognitive_tags_for_verse(spec_eff.verse_name)
    if mem_tags:
        merged_tags = list(spec_eff.tags)
        for t in mem_tags:
            if t not in merged_tags:
                merged_tags.append(t)
        spec_eff = dataclasses.replace(spec_eff, tags=merged_tags)
    factory = get_factory(spec_eff.verse_name)
    return factory.create(spec_eff)


def list_verses() -> Dict[str, VerseFactory]:
    """
    Returns the current registry mapping (read-only view).
    """
    return dict(_FACTORIES)


# -------------------------------------------------------
# Optional: built-in registrations
# -------------------------------------------------------

def register_builtin() -> None:
    """
    Call this once at app startup to register built-in verses.
    """
    # Import inside the function to avoid hard imports at module load time.
    from verses.line_world import LineWorldFactory
    from verses.grid_world import GridWorldFactory
    from verses.cliff_world import CliffWorldFactory
    from verses.labyrinth_world import LabyrinthWorldFactory
    from verses.park_world import ParkWorldFactory
    from verses.pursuit_world import PursuitWorldFactory
    from verses.warehouse_world import WarehouseWorldFactory
    from verses.chess_world import ChessWorldFactory
    from verses.go_world import GoWorldFactory
    from verses.uno_world import UnoWorldFactory
    from verses.harvest_world import HarvestWorldFactory
    from verses.bridge_world import BridgeWorldFactory
    from verses.swamp_world import SwampWorldFactory
    from verses.escape_world import EscapeWorldFactory
    from verses.factory_world import FactoryWorldFactory
    from verses.trade_world import TradeWorldFactory
    from verses.memory_vault_world import MemoryVaultWorldFactory
    from verses.rule_flip_world import RuleFlipWorldFactory
    from verses.risk_tutorial_world import RiskTutorialWorldFactory
    from verses.wind_master_world import WindMasterWorldFactory
    from verses.chess_world_v2 import ChessWorldV2Factory
    from verses.go_world_v2 import GoWorldV2Factory
    from verses.uno_world_v2 import UnoWorldV2Factory

    builtins = {
        "line_world": LineWorldFactory(),
        "grid_world": GridWorldFactory(),
        "cliff_world": CliffWorldFactory(),
        "labyrinth_world": LabyrinthWorldFactory(),
        "park_world": ParkWorldFactory(),
        "pursuit_world": PursuitWorldFactory(),
        "warehouse_world": WarehouseWorldFactory(),
        "chess_world": ChessWorldFactory(),
        "go_world": GoWorldFactory(),
        "uno_world": UnoWorldFactory(),
        "harvest_world": HarvestWorldFactory(),
        "bridge_world": BridgeWorldFactory(),
        "swamp_world": SwampWorldFactory(),
        "escape_world": EscapeWorldFactory(),
        "factory_world": FactoryWorldFactory(),
        "trade_world": TradeWorldFactory(),
        "memory_vault_world": MemoryVaultWorldFactory(),
        "rule_flip_world": RuleFlipWorldFactory(),
        "risk_tutorial_world": RiskTutorialWorldFactory(),
        "wind_master_world": WindMasterWorldFactory(),
        "chess_world_v2": ChessWorldV2Factory(),
        "go_world_v2": GoWorldV2Factory(),
        "uno_world_v2": UnoWorldV2Factory(),
    }
    for name, factory in builtins.items():
        if name not in _FACTORIES:
            register_verse(name, factory)
