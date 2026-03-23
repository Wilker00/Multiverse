"""
core/sim2real.py

Curated sim-to-real evaluation profiles for supported verses.

The goal is bounded and pragmatic:
- keep training semantics unchanged
- expose deterministic stress profiles for sim-to-real assessment
- fall back to ADR-based perturbation when no curated profile exists
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


_PROFILE_SEVERITY: Dict[str, float] = {
    "mild": 0.08,
    "moderate": 0.16,
    "severe": 0.28,
}


_BASE_DYNAMIC_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "warehouse_world": {
        "obstacle_count": 14,
        "battery_drain": 1,
        "charge_rate": 5,
        "conveyor_count": 3,
        "lidar_range": 4,
        "patrol_robot": True,
    },
    "labyrinth_world": {
        "battery_drain": 1,
        "action_noise": 0.08,
        "vision_radius": 3,
    },
    "trade_world": {
        "noise_stddev": 2.0,
        "shock_probability": 0.06,
        "shock_magnitude": 12.0,
    },
    "factory_world": {
        "breakdown_prob": 0.04,
        "arrival_rate": 0.25,
    },
    "bridge_world": {
        "wind_probability": 0.15,
        "collapse_prob_weak": 0.05,
        "collapse_prob_strong": 0.02,
    },
    "swamp_world": {
        "mud_slip_prob": 0.08,
        "flood_rate": 5,
    },
    "harvest_world": {
        "spoil_probability": 0.01,
    },
    "escape_world": {
        "guard_vision": 4,
        "hiding_cooldown": 2,
    },
    "wind_master_world": {
        "gust_probability": 0.12,
    },
    "world_3d": {
        "roughness": 0.4,
        "pit_density": 0.05,
        "max_climb": 1,
    },
}


@dataclass(frozen=True)
class Sim2RealProfile:
    name: str
    verse_name: str
    severity: float
    support_level: str
    description: str
    params: Dict[str, Any]
    param_changes: Dict[str, Dict[str, Any]]
    notes: Tuple[str, ...]


def _norm_profile(profile: str) -> str:
    name = str(profile or "").strip().lower()
    if name not in _PROFILE_SEVERITY:
        known = ", ".join(sorted(_PROFILE_SEVERITY))
        raise ValueError(f"Unknown sim-to-real profile '{profile}'. Known: {known}")
    return name


def _norm_verse(verse_name: str) -> str:
    return str(verse_name or "").strip().lower()


def _clip_probability(value: float) -> float:
    return max(0.0, min(0.95, float(value)))


def _increase_probability(value: float, severity: float) -> float:
    base = max(0.0, float(value))
    return _clip_probability(base + severity)


def _increase_int(value: int, *, severity: float, min_delta: int = 1) -> int:
    delta = max(int(min_delta), int(round(max(1.0, float(value)) * float(severity))))
    return int(value) + int(delta)


def _decrease_int(value: int, *, severity: float, floor: int = 1, min_delta: int = 1) -> int:
    delta = max(int(min_delta), int(round(max(1.0, float(value)) * float(severity))))
    return max(int(floor), int(value) - int(delta))


def _increase_float(value: float, *, severity: float, min_delta: float = 0.0) -> float:
    delta = max(float(min_delta), float(value) * float(severity))
    return float(value) + float(delta)


def _decrease_float(value: float, *, severity: float, floor: float = 0.0, min_delta: float = 0.0) -> float:
    delta = max(float(min_delta), float(value) * float(severity))
    return max(float(floor), float(value) - float(delta))


def list_sim2real_profiles() -> List[str]:
    return sorted(_PROFILE_SEVERITY.keys())


def _seed_base_params(verse_name: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(_BASE_DYNAMIC_DEFAULTS.get(_norm_verse(verse_name), {}))
    params.update(dict(base_params or {}))
    params.setdefault("adr_enabled", False)
    return params


def _record_change(changes: Dict[str, Dict[str, Any]], params: Dict[str, Any], key: str, new_value: Any) -> None:
    old_value = params.get(key)
    if old_value == new_value:
        return
    changes[str(key)] = {"from": old_value, "to": new_value}
    params[str(key)] = new_value


def _build_curated_profile(verse_name: str, params: Dict[str, Any], *, severity: float) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[str]]:
    verse = _norm_verse(verse_name)
    out = dict(params)
    changes: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []

    if verse == "warehouse_world":
        _record_change(changes, out, "obstacle_count", _increase_int(int(out.get("obstacle_count", 14)), severity=severity, min_delta=1))
        _record_change(changes, out, "battery_drain", _increase_int(int(out.get("battery_drain", 1)), severity=severity, min_delta=1))
        _record_change(changes, out, "charge_rate", _decrease_int(int(out.get("charge_rate", 5)), severity=severity, floor=1, min_delta=1))
        _record_change(changes, out, "conveyor_count", _increase_int(int(out.get("conveyor_count", 3)), severity=severity, min_delta=1))
        _record_change(changes, out, "lidar_range", _decrease_int(int(out.get("lidar_range", 4)), severity=severity, floor=1, min_delta=1))
        notes.append("Adds layout density, battery pressure, and shorter-range sensing.")
        return out, changes, notes

    if verse == "labyrinth_world":
        _record_change(changes, out, "action_noise", _increase_probability(float(out.get("action_noise", 0.08)), severity=severity))
        _record_change(changes, out, "battery_drain", _increase_int(int(out.get("battery_drain", 1)), severity=severity, min_delta=1))
        _record_change(changes, out, "vision_radius", _decrease_int(int(out.get("vision_radius", 3)), severity=severity, floor=1, min_delta=1))
        notes.append("Injects actuation noise, shorter visibility, and faster resource depletion.")
        return out, changes, notes

    if verse == "trade_world":
        _record_change(changes, out, "noise_stddev", _increase_float(float(out.get("noise_stddev", 2.0)), severity=severity, min_delta=0.5))
        _record_change(changes, out, "shock_probability", _increase_probability(float(out.get("shock_probability", 0.06)), severity=severity))
        _record_change(changes, out, "shock_magnitude", _increase_float(float(out.get("shock_magnitude", 12.0)), severity=severity, min_delta=1.0))
        notes.append("Increases market noise and rare regime-shift severity.")
        return out, changes, notes

    if verse == "factory_world":
        _record_change(changes, out, "breakdown_prob", _increase_probability(float(out.get("breakdown_prob", 0.04)), severity=severity))
        _record_change(changes, out, "arrival_rate", _increase_float(float(out.get("arrival_rate", 0.25)), severity=severity, min_delta=0.05))
        notes.append("Raises machine failures and inbound pressure.")
        return out, changes, notes

    if verse == "bridge_world":
        _record_change(changes, out, "wind_probability", _increase_probability(float(out.get("wind_probability", 0.15)), severity=severity))
        _record_change(changes, out, "collapse_prob_weak", _increase_probability(float(out.get("collapse_prob_weak", 0.05)), severity=severity))
        _record_change(changes, out, "collapse_prob_strong", _increase_probability(float(out.get("collapse_prob_strong", 0.02)), severity=severity))
        notes.append("Applies stronger wind and higher structural instability.")
        return out, changes, notes

    if verse == "swamp_world":
        _record_change(changes, out, "mud_slip_prob", _increase_probability(float(out.get("mud_slip_prob", 0.08)), severity=severity))
        _record_change(changes, out, "flood_rate", _increase_int(int(out.get("flood_rate", 5)), severity=severity, min_delta=1))
        notes.append("Makes terrain slip and flooding more frequent.")
        return out, changes, notes

    if verse == "harvest_world":
        _record_change(changes, out, "spoil_probability", _increase_probability(float(out.get("spoil_probability", 0.01)), severity=severity))
        notes.append("Reduces inventory stability through faster spoilage.")
        return out, changes, notes

    if verse == "escape_world":
        _record_change(changes, out, "guard_vision", _increase_int(int(out.get("guard_vision", 4)), severity=severity, min_delta=1))
        _record_change(changes, out, "hiding_cooldown", _increase_int(int(out.get("hiding_cooldown", 2)), severity=severity, min_delta=1))
        notes.append("Improves adversary sensing and slows hide-reset recovery.")
        return out, changes, notes

    if verse == "wind_master_world":
        _record_change(changes, out, "gust_probability", _increase_probability(float(out.get("gust_probability", 0.12)), severity=severity))
        notes.append("Raises crosswind variability.")
        return out, changes, notes

    if verse == "world_3d":
        _record_change(changes, out, "roughness", _increase_float(float(out.get("roughness", 0.4)), severity=severity, min_delta=0.05))
        _record_change(changes, out, "pit_density", _increase_probability(float(out.get("pit_density", 0.05)), severity=severity))
        _record_change(changes, out, "max_climb", _decrease_int(int(out.get("max_climb", 1)), severity=severity, floor=1, min_delta=0))
        notes.append("Raises terrain roughness and pit density; optionally tightens climb constraint.")
        return out, changes, notes

    return out, changes, notes


def build_sim2real_profile(
    *,
    verse_name: str,
    base_params: Dict[str, Any] | None = None,
    profile: str,
) -> Sim2RealProfile:
    profile_name = _norm_profile(profile)
    severity = float(_PROFILE_SEVERITY[profile_name])
    seeded_params = _seed_base_params(str(verse_name), dict(base_params or {}))
    params, changes, notes = _build_curated_profile(str(verse_name), seeded_params, severity=severity)

    if changes:
        description = f"{profile_name} curated sim-to-real stress profile"
        return Sim2RealProfile(
            name=profile_name,
            verse_name=_norm_verse(verse_name),
            severity=severity,
            support_level="curated",
            description=description,
            params=params,
            param_changes=changes,
            notes=tuple(notes),
        )

    fallback = dict(seeded_params)
    fallback["adr_enabled"] = True
    fallback["adr_jitter"] = round(float(severity), 4)
    fallback["adr_keys"] = list(fallback.get("adr_keys") or [])
    fallback_changes: Dict[str, Dict[str, Any]] = {
        "adr_enabled": {"from": seeded_params.get("adr_enabled"), "to": True},
        "adr_jitter": {"from": seeded_params.get("adr_jitter"), "to": fallback["adr_jitter"]},
    }
    if seeded_params.get("adr_keys") != fallback["adr_keys"]:
        fallback_changes["adr_keys"] = {"from": seeded_params.get("adr_keys"), "to": list(fallback["adr_keys"])}
    return Sim2RealProfile(
        name=profile_name,
        verse_name=_norm_verse(verse_name),
        severity=severity,
        support_level="adr_fallback",
        description=f"{profile_name} ADR-backed sim-to-real stress profile",
        params=fallback,
        param_changes=fallback_changes,
        notes=(
            "No curated sim-to-real profile exists for this verse yet.",
            "Assessment falls back to ADR jitter, so confidence is lower than curated profiles.",
        ),
    )


def build_sim2real_profiles(
    *,
    verse_name: str,
    base_params: Dict[str, Any] | None = None,
    profiles: Iterable[str],
) -> List[Sim2RealProfile]:
    return [
        build_sim2real_profile(
            verse_name=str(verse_name),
            base_params=dict(base_params or {}),
            profile=str(name),
        )
        for name in profiles
    ]
