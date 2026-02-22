"""
memory/universe_adapters.py

Universe-level feature adapters for transfer diagnostics and lane-aware filtering.

These adapters do not replace per-verse observations. They provide a shared
mechanics-oriented feature view (starting with logistics) that can be used to:
- inspect whether translated rows carry usable structure
- compare near/far transfer lanes
- later drive lane-specific weighting/gating
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.universe_registry import primary_universe_for_verse


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _logistics_features(obs: Dict[str, Any], next_obs: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    n = next_obs if isinstance(next_obs, dict) else {}

    x = _safe_int(obs.get("x", 0), 0)
    y = _safe_int(obs.get("y", 0), 0)
    gx = _safe_int(obs.get("goal_x", x), x)
    gy = _safe_int(obs.get("goal_y", y), y)
    old_dist = abs(x - gx) + abs(y - gy)

    nx = _safe_int(n.get("x", x), x)
    ny = _safe_int(n.get("y", y), y)
    ng_x = _safe_int(n.get("goal_x", gx), gx)
    ng_y = _safe_int(n.get("goal_y", gy), gy)
    new_dist = abs(nx - ng_x) + abs(ny - ng_y)

    battery = _safe_float(obs.get("battery", 0.0), 0.0)
    nearby_obstacles = _safe_float(obs.get("nearby_obstacles", 0.0), 0.0)
    patrol_dist = _safe_float(obs.get("patrol_dist", 99.0), 99.0)
    t = _safe_float(obs.get("t", 0.0), 0.0)
    nearest_charger = _safe_float(obs.get("nearest_charger_dist", -1.0), -1.0)
    on_conveyor = _safe_float(obs.get("on_conveyor", 0.0), 0.0)
    output_buf = _safe_float(obs.get("output_buf", 0.0), 0.0)
    completed = _safe_float(obs.get("completed", 0.0), 0.0)
    next_completed = _safe_float(n.get("completed", completed), completed)
    total_arrived = max(0.0, _safe_float(obs.get("total_arrived", 0.0), 0.0))
    inventory = max(0.0, _safe_float(obs.get("inventory", 0.0), 0.0))
    cash = max(0.0, _safe_float(obs.get("cash", 0.0), 0.0))
    portfolio_value = max(cash, _safe_float(obs.get("portfolio_value", cash), cash))
    price = max(0.0, _safe_float(obs.get("price", 0.0), 0.0))
    segments_placed = max(0.0, _safe_float(obs.get("segments_placed", 0.0), 0.0))
    broken_cnt = sum(_safe_float(obs.get(f"broken_{i}", 0.0), 0.0) for i in range(3))
    buf_total = sum(max(0.0, _safe_float(obs.get(f"buf_{i}", 0.0), 0.0)) for i in range(3))

    # Goal progress proxy (higher is better)
    if old_dist <= 0:
        goal_progress = 1.0
    else:
        goal_progress = 1.0 - (float(old_dist) / float(max(1.0, old_dist + 1.0)))
    # Throughput proxy blends task progress delta with production/trading progress where available.
    move_progress_delta = float(old_dist - new_dist)
    completed_delta = float(next_completed - completed)
    throughput = _clip01(0.5 + 0.15 * move_progress_delta + 0.05 * completed_delta)

    # Hazard proximity proxy
    obstacle_risk = _clip01(nearby_obstacles / 4.0)
    patrol_risk = _clip01(1.0 - min(max(patrol_dist, 0.0), 10.0) / 10.0)
    breakdown_risk = _clip01(broken_cnt / 3.0)
    hazard_proximity = _clip01(max(obstacle_risk, patrol_risk, breakdown_risk))

    # Resource level proxy
    battery_norm = _clip01(battery / 100.0) if battery > 0 else 0.0
    charger_urgency = _clip01(1.0 - min(max(nearest_charger, 0.0), 20.0) / 20.0) if nearest_charger >= 0 else 0.0
    factory_flow = _clip01((total_arrived - completed) / 50.0) if total_arrived > 0 else _clip01(buf_total / 12.0)
    trading_liquidity = _clip01((cash + 0.1 * portfolio_value) / 500.0) if (cash > 0 or portfolio_value > 0) else 0.0
    resource_level = _clip01(max(battery_norm, 1.0 - charger_urgency, trading_liquidity, 1.0 - factory_flow))

    # Queue pressure / congestion proxies
    queue_pressure = _clip01(max(buf_total / 12.0, output_buf / 10.0, inventory / 10.0))
    congestion = _clip01(max(obstacle_risk, queue_pressure, on_conveyor * 0.5))

    # Time pressure proxy (monotonic-ish; coarse, bounded)
    time_pressure = _clip01(t / 200.0)

    return {
        "goal_progress": float(goal_progress),
        "hazard_proximity": float(hazard_proximity),
        "resource_level": float(resource_level),
        "queue_pressure": float(queue_pressure),
        "throughput": float(throughput),
        "congestion": float(congestion),
        "time_pressure": float(time_pressure),
        "resource_signal_battery": float(battery_norm),
        "resource_signal_factory_flow": float(factory_flow),
        "resource_signal_trading_liquidity": float(trading_liquidity),
        "mechanics_progress_delta": float(move_progress_delta),
        "mechanics_completed_delta": float(completed_delta),
        "mechanics_price": float(price),
        "mechanics_segments_placed": float(segments_placed),
    }


def universe_features_for_obs(
    *,
    verse_name: str,
    obs: Any,
    next_obs: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(obs, dict):
        return None
    universe = primary_universe_for_verse(str(verse_name))
    if not universe:
        return None

    if universe == "logistics_universe":
        feats = _logistics_features(obs, next_obs if isinstance(next_obs, dict) else None)
        return {
            "universe_id": "logistics_universe",
            "feature_space": "logistics_shared_v1",
            "features": feats,
        }

    return {
        "universe_id": str(universe),
        "feature_space": "unknown",
        "features": {},
    }


def transfer_row_universe_metadata(
    *,
    source_verse: str,
    target_verse: str,
    translated_obs: Any,
    translated_next_obs: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    target_universe = primary_universe_for_verse(str(target_verse))
    source_universe = primary_universe_for_verse(str(source_verse))
    if not target_universe:
        return None
    target_feats = universe_features_for_obs(
        verse_name=str(target_verse),
        obs=translated_obs,
        next_obs=translated_next_obs,
    )
    if target_feats is None:
        return None
    return {
        "source_universe": source_universe or "",
        "target_universe": target_universe or "",
        "same_universe": bool(source_universe and target_universe and source_universe == target_universe),
        "adapter": target_feats,
    }

