"""
core/taxonomy.py

Hierarchical taxonomy for semantic memory bridging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set


TAXONOMY: Dict[str, List[str]] = {
    "navigation": [
        "line_world",
        "grid_world",
        "cliff_world",
        "labyrinth_world",
        "park_world",
        "pursuit_world",
        "warehouse_world",
        "harvest_world",
        "swamp_world",
        "escape_world",
        "memory_vault_world",
        "maze_world",
        "wind_master_world",
    ],
    "strategy_games": ["chess_world", "go_world", "uno_world", "chess_world_v2", "go_world_v2", "uno_world_v2"],
    "board_control": ["chess_world", "go_world", "chess_world_v2", "go_world_v2"],
    "card_strategy": ["uno_world", "uno_world_v2"],
    "complex_navigation": ["labyrinth_world", "swamp_world", "escape_world"],
    "physics": ["pursuit_world", "labyrinth_world"],
    "interaction": ["park_world"],
    "resource_management": ["labyrinth_world", "warehouse_world", "harvest_world", "trade_world"],
    "obstacle_navigation": ["grid_world", "labyrinth_world", "warehouse_world", "harvest_world", "escape_world"],
    "risk_sensitive": ["cliff_world", "labyrinth_world", "warehouse_world", "bridge_world", "swamp_world", "escape_world", "trade_world"],
    "dynamic_hazards": ["labyrinth_world", "swamp_world"],
    "transferable_logic": ["chess_world", "go_world", "uno_world"],
    "planning": ["bridge_world", "factory_world"],
    "survival": ["swamp_world", "cliff_world"],
    "resource_collection": ["harvest_world"],
    "stealth": ["escape_world"],
    "scheduling": ["factory_world"],
    "economics": ["trade_world"],
    "memory_diagnostics": ["memory_vault_world", "rule_flip_world"],
    "tutorial": ["risk_tutorial_world"],
}


VERSE_TAGS: Dict[str, List[str]] = {
    "line_world": ["navigation", "1d", "continuous_goal"],
    "grid_world": ["navigation", "2d", "discrete_grid"],
    "cliff_world": ["navigation", "2d", "discrete_grid", "risk_sensitive", "cliff", "survival"],
    "labyrinth_world": [
        "navigation",
        "complex_navigation",
        "2d",
        "discrete_grid",
        "maze",
        "dynamic_hazards",
        "partial_observable",
        "resource_management",
        "risk_sensitive",
    ],
    "park_world": ["navigation", "interaction", "parking"],
    "pursuit_world": ["navigation", "physics", "dynamic_target"],
    "warehouse_world": ["navigation", "2d", "resource_management", "obstacle_navigation", "battery", "risk_sensitive"],
    "chess_world": [
        "strategy_games",
        "board_control",
        "chess_like",
        "transferable_logic",
        "turn_based",
    ],
    "go_world": [
        "strategy_games",
        "board_control",
        "go_like",
        "transferable_logic",
        "turn_based",
    ],
    "uno_world": [
        "strategy_games",
        "card_strategy",
        "uno_like",
        "transferable_logic",
        "turn_based",
    ],
    "harvest_world": [
        "navigation",
        "2d",
        "discrete_grid",
        "resource_collection",
        "resource_management",
        "obstacle_navigation",
        "carry_capacity",
    ],
    "bridge_world": [
        "planning",
        "construction",
        "risk_sensitive",
        "sequential_decision",
    ],
    "swamp_world": [
        "navigation",
        "complex_navigation",
        "2d",
        "discrete_grid",
        "dynamic_hazards",
        "dynamic_terrain",
        "survival",
        "risk_sensitive",
        "time_pressure",
    ],
    "escape_world": [
        "navigation",
        "complex_navigation",
        "2d",
        "discrete_grid",
        "stealth",
        "evasion",
        "opponent_modeling",
        "risk_sensitive",
        "obstacle_navigation",
    ],
    "factory_world": [
        "scheduling",
        "production",
        "sequential_decision",
        "fault_tolerance",
        "planning",
    ],
    "trade_world": [
        "economics",
        "trading",
        "temporal_pattern",
        "risk_management",
        "inventory",
        "resource_management",
        "risk_sensitive",
    ],
    "memory_vault_world": [
        "navigation",
        "2d",
        "discrete_grid",
        "maze",
        "partial_observable",
        "memory_diagnostics",
        "sequential_memory",
        "risk_sensitive",
    ],
    "maze_world": [
        "navigation",
        "2d",
        "maze",
        "exploration",
        "discrete_grid",
        "risk_sensitive",
    ],
    "rule_flip_world": [
        "navigation",
        "1d",
        "discrete_grid",
        "memory_diagnostics",
        "rule_shift",
        "adaptive_control",
        "sequential_decision",
    ],
    "risk_tutorial_world": [
        "tutorial",
        "risk_sensitive",
        "sequential_decision",
        "adaptive_control",
    ],
    "wind_master_world": [
        "navigation",
        "2d",
        "discrete_grid",
        "risk_sensitive",
        "dynamic_hazards",
        "time_pressure",
    ],
    "chess_world_v2": [
        "strategy_games",
        "board_control",
        "chess_like",
        "transferable_logic",
        "turn_based",
    ],
    "go_world_v2": [
        "strategy_games",
        "board_control",
        "go_like",
        "transferable_logic",
        "turn_based",
    ],
    "uno_world_v2": [
        "strategy_games",
        "card_strategy",
        "uno_like",
        "transferable_logic",
        "turn_based",
    ],
}

# Cognitive memory taxonomy used by memory routing.
# Required field: memory_type for each verse.
VERSE_MEMORY_TYPES: Dict[str, str] = {
    "line_world": "spatial_procedural",
    "grid_world": "spatial_procedural",
    "cliff_world": "spatial_procedural",
    "labyrinth_world": "spatial_procedural",
    "park_world": "spatial_procedural",
    "pursuit_world": "spatial_procedural",
    "warehouse_world": "spatial_procedural",
    "harvest_world": "spatial_procedural",
    "bridge_world": "spatial_procedural",
    "swamp_world": "spatial_procedural",
    "escape_world": "spatial_procedural",
    "factory_world": "procedural_strategic",
    "trade_world": "declarative_strategic",
    "chess_world": "declarative_strategic",
    "go_world": "declarative_strategic",
    "uno_world": "declarative_strategic",
    "chess_world_v2": "declarative_strategic",
    "go_world_v2": "declarative_strategic",
    "uno_world_v2": "declarative_strategic",
    "memory_vault_world": "spatial_procedural",
    "rule_flip_world": "declarative_adaptive",
    "maze_world": "spatial_procedural",
    "wind_master_world": "spatial_procedural",
    "risk_tutorial_world": "declarative_adaptive",
}


# Universe system: similarity partitions for verses where positive competence
# transfer is expected to be more likely than negative transfer.
UNIVERSE_RUBRIC: Dict[str, str] = {
    "action_semantics": "Core action meanings are the same or can be mapped without ambiguity.",
    "state_structure": "Observations expose comparable task-relevant structure (position, goal, hazards, resources).",
    "success_condition": "Success/termination semantics are similar enough for competence transfer metrics to align.",
    "failure_modes": "Primary hazards/failure patterns overlap (e.g., collisions, cliffs, resource depletion).",
    "control_strategy": "A shared policy prior or dynamics prior should improve solving more often than harm it.",
}

VERSE_UNIVERSES: Dict[str, str] = {
    # Navigation core (clean spatial goal-seeking)
    "line_world": "navigation_core",
    "grid_world": "navigation_core",
    "maze_world": "navigation_core",
    # Navigation with explicit risk/stochastic hazards
    "cliff_world": "navigation_risk",
    "bridge_world": "navigation_risk",
    "swamp_world": "navigation_risk",
    "risk_tutorial_world": "navigation_risk",
    "wind_master_world": "navigation_risk",
    # Navigation with moving entities/opponents / dynamic interaction
    "park_world": "navigation_dynamic",
    "pursuit_world": "navigation_dynamic",
    "escape_world": "navigation_dynamic",
    # Logistics / operations-like environments (routing + constraints / process structure)
    "warehouse_world": "logistics_ops",
    "labyrinth_world": "logistics_ops",
    "factory_world": "logistics_ops",
    # Strategy games (turn-based symbolic planning)
    "chess_world": "strategy_games",
    "go_world": "strategy_games",
    "uno_world": "strategy_games",
    "chess_world_v2": "strategy_games",
    "go_world_v2": "strategy_games",
    "uno_world_v2": "strategy_games",
    # Resource / economy style tasks
    "harvest_world": "resource_economy",
    "trade_world": "resource_economy",
    # Memory / rule-shift tasks
    "memory_vault_world": "memory_rule_tasks",
    "rule_flip_world": "memory_rule_tasks",
}

# Optional adjacency lets tooling distinguish near cross-universe transfer from far transfer.
UNIVERSE_ADJACENCY: Dict[str, List[str]] = {
    "navigation_core": ["navigation_risk", "navigation_dynamic", "logistics_ops", "memory_rule_tasks"],
    "navigation_risk": ["navigation_core", "navigation_dynamic", "logistics_ops"],
    "navigation_dynamic": ["navigation_core", "navigation_risk", "logistics_ops"],
    "logistics_ops": ["navigation_core", "navigation_risk", "navigation_dynamic", "resource_economy"],
    "strategy_games": ["resource_economy", "memory_rule_tasks"],
    "resource_economy": ["logistics_ops", "strategy_games", "memory_rule_tasks"],
    "memory_rule_tasks": ["navigation_core", "strategy_games", "resource_economy"],
}


def _norm(x: str) -> str:
    return str(x).strip().lower()


def universe_for_verse(verse_name: str) -> str:
    return str(VERSE_UNIVERSES.get(_norm(verse_name), "unknown"))


def all_universes() -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for u in VERSE_UNIVERSES.values():
        uu = _norm(u)
        if not uu or uu in seen:
            continue
        seen.add(uu)
        out.append(uu)
    return out


def universe_members(universe_name: str) -> List[str]:
    u = _norm(universe_name)
    out = [v for v, uu in VERSE_UNIVERSES.items() if _norm(uu) == u]
    return sorted(out)


def same_universe(source_verse: str, target_verse: str) -> bool:
    su = universe_for_verse(source_verse)
    tu = universe_for_verse(target_verse)
    return bool(su != "unknown" and su == tu)


def universe_relation(source_verse: str, target_verse: str) -> str:
    su = universe_for_verse(source_verse)
    tu = universe_for_verse(target_verse)
    if su == "unknown" or tu == "unknown":
        return "unknown"
    if su == tu:
        return f"same_universe:{su}"
    near = set(_norm(x) for x in UNIVERSE_ADJACENCY.get(su, []))
    if tu in near:
        return f"adjacent_universe:{su}->{tu}"
    return f"cross_universe:{su}->{tu}"


def parent_domains(verse_name: str) -> Set[str]:
    v = _norm(verse_name)
    out: Set[str] = set()
    for parent, children in TAXONOMY.items():
        if v in {_norm(c) for c in children}:
            out.add(_norm(parent))
    return out


def tags_for_verse(verse_name: str) -> List[str]:
    verse = _norm(verse_name)
    base = list(VERSE_TAGS.get(verse, ["unknown"]))
    uni = universe_for_verse(verse)
    if uni != "unknown":
        base.append(f"universe:{uni}")
    cog = cognitive_tags_for_verse(verse)
    merged: List[str] = []
    seen: Set[str] = set()
    for t in base + cog:
        tt = _norm(t)
        if not tt or tt in seen:
            continue
        seen.add(tt)
        merged.append(tt)
    return merged


def memory_type_for_verse(verse_name: str) -> str:
    return str(VERSE_MEMORY_TYPES.get(_norm(verse_name), "unknown"))


def memory_family_for_type(memory_type: str) -> str:
    mt = _norm(memory_type)
    if "declarative" in mt:
        return "declarative"
    if "procedural" in mt or "spatial" in mt:
        return "procedural"
    if mt == "unknown":
        return "unknown"
    return "hybrid"


def memory_family_for_verse(verse_name: str) -> str:
    return memory_family_for_type(memory_type_for_verse(verse_name))


def cognitive_tags_for_verse(verse_name: str) -> List[str]:
    mt = memory_type_for_verse(verse_name)
    fam = memory_family_for_type(mt)
    out = [f"memory_type:{mt}"]
    if fam != "unknown":
        out.append(f"memory_family:{fam}")
    return out


def shared_parent(source_verse: str, target_verse: str) -> Optional[str]:
    src = parent_domains(source_verse)
    dst = parent_domains(target_verse)
    inter = sorted(src & dst)
    if not inter:
        return None
    # Prefer navigation when available.
    if "navigation" in inter:
        return "navigation"
    return inter[0]


def can_bridge(source_verse: str, target_verse: str) -> bool:
    src = _norm(source_verse)
    dst = _norm(target_verse)
    if src == dst:
        return True
    if same_universe(src, dst):
        return True
    if shared_parent(src, dst) is not None:
        return True

    
    src_tags = set(tags_for_verse(src))
    dst_tags = set(tags_for_verse(dst))
    # Allow cross-domain transfer from strategic logic into complex logistics/navigation verses.
    if "transferable_logic" in src_tags and (
        "resource_management" in dst_tags
        or "obstacle_navigation" in dst_tags
        or "complex_navigation" in dst_tags
        or "planning" in dst_tags
        or "sequential_decision" in dst_tags
    ):
        return True

    # Functional analogies (explicit pairs)
    # Bridge construction <-> Line traversal (1D progression)
    if {src, dst} == {"bridge_world", "line_world"}:
        return True
    # Factory production <-> Harvest collection (Resource streams)
    if {src, dst} == {"factory_world", "harvest_world"}:
        return True
    # Trade inventory <-> Harvest collection (Accumulation)
    if {src, dst} == {"trade_world", "harvest_world"}:
        return True
    # Trade inventory <-> Factory buffers (Stock management)
    if {src, dst} == {"trade_world", "factory_world"}:
        return True

    return False


def bridge_reason(source_verse: str, target_verse: str) -> str:
    src = _norm(source_verse)
    dst = _norm(target_verse)
    if src == dst:
        return "same_verse"
    if same_universe(src, dst):
        return universe_relation(src, dst)
    parent = shared_parent(src, dst)
    if parent is None:
        src_tags = set(tags_for_verse(src))
        dst_tags = set(tags_for_verse(dst))
        if "transferable_logic" in src_tags and (
            "resource_management" in dst_tags or "obstacle_navigation" in dst_tags or "complex_navigation" in dst_tags
        ):
            return "transferable_logic_projection"
            
        # Explicit analogies
        pairs = {src, dst}
        if pairs == {"bridge_world", "line_world"}:
            return "1d_progression_analogy"
        if pairs == {"factory_world", "harvest_world"}:
            return "resource_stream_analogy"
        if pairs == {"trade_world", "harvest_world"}:
            return "accumulation_analogy"
        if pairs == {"trade_world", "factory_world"}:
            return "inventory_management_analogy"

        return "no_shared_parent"
    return f"shared_parent:{parent}"
