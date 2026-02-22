"""
core/universe_registry.py

Universe-level grouping for transfer planning.

This provides a lightweight, config-like registry that lets the transfer
pipeline distinguish:
- near transfer (same universe / related mechanics)
- far transfer (cross-universe / abstract priors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _norm(x: str) -> str:
    return str(x or "").strip().lower()


def _uniq(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        v = _norm(raw)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


@dataclass(frozen=True)
class UniverseSpec:
    universe_id: str
    description: str
    members: Tuple[str, ...]
    preferred_sources_by_target: Dict[str, Tuple[str, ...]]
    far_fallback_sources: Tuple[str, ...] = ()


_UNIVERSES: Dict[str, UniverseSpec] = {
    "logistics_universe": UniverseSpec(
        universe_id="logistics_universe",
        description=(
            "Related logistics / navigation / resource-flow verses where transfer should focus "
            "on shared mechanics (routing, hazards, throughput, resource constraints)."
        ),
        # 10 verses to anchor universe-local transfer around warehouse-like mechanics.
        members=(
            "warehouse_world",
            "factory_world",
            "harvest_world",
            "trade_world",
            "bridge_world",
            "grid_world",
            "park_world",
            "pursuit_world",
            "labyrinth_world",
            "escape_world",
        ),
        preferred_sources_by_target={
            "warehouse_world": (
                "factory_world",
                "harvest_world",
                "grid_world",
                "bridge_world",
                "trade_world",
                "park_world",
                "pursuit_world",
                "labyrinth_world",
                "escape_world",
            ),
            "factory_world": (
                "warehouse_world",
                "harvest_world",
                "trade_world",
                "bridge_world",
                "grid_world",
            ),
            "trade_world": (
                "factory_world",
                "harvest_world",
                "warehouse_world",
                "bridge_world",
                "grid_world",
            ),
        },
        # Strategy games remain useful as far-transfer priors for safety/tempo planning.
        far_fallback_sources=("chess_world", "go_world", "trade_world", "uno_world"),
    ),
    "strategy_universe": UniverseSpec(
        universe_id="strategy_universe",
        description="Turn-based strategy/card games with abstract transferable logic.",
        members=("chess_world", "go_world", "uno_world"),
        preferred_sources_by_target={
            "chess_world": ("go_world", "uno_world"),
            "go_world": ("chess_world", "uno_world"),
            "uno_world": ("chess_world", "go_world"),
        },
        far_fallback_sources=("warehouse_world", "factory_world", "bridge_world"),
    ),
}


def list_universes() -> Dict[str, UniverseSpec]:
    return dict(_UNIVERSES)


def get_universe(universe_id: str) -> UniverseSpec:
    key = _norm(universe_id)
    if key not in _UNIVERSES:
        known = ", ".join(sorted(_UNIVERSES.keys())) or "(none)"
        raise KeyError(f"Unknown universe '{universe_id}'. Known: {known}")
    return _UNIVERSES[key]


def universes_for_verse(verse_name: str) -> List[str]:
    v = _norm(verse_name)
    out: List[str] = []
    for uid, spec in _UNIVERSES.items():
        if v in {_norm(x) for x in spec.members}:
            out.append(uid)
    return out


def primary_universe_for_verse(verse_name: str) -> Optional[str]:
    ids = universes_for_verse(verse_name)
    return ids[0] if ids else None


def same_universe_verses(verse_name: str, *, include_self: bool = False) -> List[str]:
    v = _norm(verse_name)
    uid = primary_universe_for_verse(v)
    if uid is None:
        return []
    spec = get_universe(uid)
    out = _uniq(list(spec.preferred_sources_by_target.get(v, ())) + list(spec.members))
    if not include_self:
        out = [x for x in out if x != v]
    return out


def source_transfer_lane(source_verse: str, target_verse: str) -> str:
    src = _norm(source_verse)
    tgt = _norm(target_verse)
    if not src or not tgt:
        return "unknown"
    if src == tgt:
        return "same_verse"
    src_u = primary_universe_for_verse(src)
    tgt_u = primary_universe_for_verse(tgt)
    if src_u and tgt_u and src_u == tgt_u:
        return "near_universe"
    return "far_universe"


def build_transfer_source_plan(target_verse: str) -> Dict[str, object]:
    tgt = _norm(target_verse)
    target_universe = primary_universe_for_verse(tgt)
    near_sources = same_universe_verses(tgt, include_self=False)
    far_sources: List[str] = []
    if target_universe:
        spec = get_universe(target_universe)
        far_sources.extend([_norm(v) for v in spec.far_fallback_sources])
    # Global fallback for targets outside configured universes.
    far_sources.extend(["chess_world", "go_world", "trade_world", "uno_world"])
    # Avoid duplicates and avoid reusing target as a source.
    far_sources = [v for v in _uniq(far_sources) if v and v != tgt and v not in set(near_sources)]
    ordered_sources = _uniq(list(near_sources) + list(far_sources))
    return {
        "target_verse": tgt,
        "target_universe": target_universe,
        "near_sources": list(near_sources),
        "far_sources": list(far_sources),
        "ordered_sources": list(ordered_sources),
    }

