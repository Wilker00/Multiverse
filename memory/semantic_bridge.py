"""
memory/semantic_bridge.py

Semantic bridge for cross-verse trajectory transfer.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.taxonomy import bridge_reason, can_bridge
from core.types import JSONValue
from memory.embeddings import cosine_similarity, obs_to_universal_vector, project_vector

_LEARNED_BRIDGE_CACHE: Dict[str, Any] = {}
_LEARNED_BRIDGE_MISSING: set[str] = set()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: bool = False) -> bool:
    try:
        if x is None:
            return bool(default)
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return bool(x)
    except Exception:
        return default


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_learned_bridge_model_path(model_path: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if isinstance(model_path, str) and model_path.strip():
        candidates.append(model_path.strip())
    env_path = str(os.environ.get("MULTIVERSE_BRIDGE_MODEL_PATH", "")).strip()
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(_project_root(), "models", "contrastive_bridge.pt"))
    for raw in candidates:
        p = os.path.abspath(raw)
        if os.path.isfile(p):
            return p
    return None


def _load_learned_bridge_model(model_path: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    resolved = _resolve_learned_bridge_model_path(model_path)
    if not resolved:
        return None, None
    if resolved in _LEARNED_BRIDGE_MISSING:
        return None, resolved
    cached = _LEARNED_BRIDGE_CACHE.get(resolved)
    if cached is not None:
        return cached, resolved

    try:
        from models.contrastive_bridge import ContrastiveBridge
    except Exception:
        _LEARNED_BRIDGE_MISSING.add(resolved)
        return None, resolved

    try:
        model = ContrastiveBridge.load(resolved, map_location="cpu")
        _LEARNED_BRIDGE_CACHE[resolved] = model
        return model, resolved
    except Exception:
        _LEARNED_BRIDGE_MISSING.add(resolved)
        return None, resolved


def _learned_bridge_metrics(
    *,
    source_obs: JSONValue,
    translated_obs: JSONValue,
    model_path: Optional[str],
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    model, resolved = _load_learned_bridge_model(model_path)
    if model is None:
        return None, None, resolved

    try:
        import torch
    except Exception:
        return None, None, resolved

    try:
        with torch.no_grad():
            emb = model.embed([source_obs, translated_obs])
        if emb.shape[0] < 2:
            return None, None, resolved
        similarity = float((emb[0] * emb[1]).sum().item())
        similarity = max(-1.0, min(1.0, similarity))
        confidence = max(0.0, min(1.0, 0.5 * (similarity + 1.0)))
        return similarity, confidence, resolved
    except Exception:
        return None, None, resolved


def _norm_verse_name(name: Optional[str]) -> str:
    return str(name or "").strip().lower()


def task_embedding_weights(
    *,
    target_verse_name: str,
    profile: str = "balanced",
) -> Dict[str, float]:
    """
    Return a lightweight semantic task embedding usable as SF reward weights.

    Keys are intentionally generic so downstream agents can map them onto their
    local feature basis:
      - bias
      - obstacle
      - goal
      - battery
      - patrol
      - conveyor
    """
    v = _norm_verse_name(target_verse_name)
    p = str(profile or "balanced").strip().lower()

    out: Dict[str, float] = {
        "bias": -0.03,
        "obstacle": -1.00,
        "goal": 2.00,
        "battery": 0.10,
        "patrol": -0.60,
        "conveyor": -0.10,
    }

    if p in {"safety", "safety_first"}:
        out["obstacle"] = -1.40
        out["goal"] = 1.60
        out["patrol"] = -0.90
    elif p in {"goal", "goal_first", "profit"}:
        out["obstacle"] = -0.70
        out["goal"] = 2.40
        out["patrol"] = -0.40

    # Verse-specific priors.
    if v == "warehouse_world":
        out["battery"] = max(out["battery"], 0.20)
        out["conveyor"] = -0.20
    elif v == "grid_world":
        out["battery"] = 0.0
        out["patrol"] = 0.0
        out["conveyor"] = 0.0

    return out


def _is_strategy_verse(name: str) -> bool:
    return _norm_verse_name(name) in {"chess_world", "go_world", "uno_world"}


def _strategy_signature(obs: JSONValue, verse_name: str) -> Optional[Dict[str, int]]:
    if not isinstance(obs, dict):
        return None
    v = _norm_verse_name(verse_name)

    if all(k in obs for k in ("score_delta", "pressure", "risk", "tempo", "control", "resource")):
        return {
            "score_delta": _safe_int(obs.get("score_delta", 0)),
            "pressure": _safe_int(obs.get("pressure", 0)),
            "risk": _safe_int(obs.get("risk", 0)),
            "tempo": _safe_int(obs.get("tempo", 0)),
            "control": _safe_int(obs.get("control", 0)),
            "resource": _safe_int(obs.get("resource", 0)),
        }

    if v == "chess_world":
        material = _safe_int(obs.get("material_delta", 0))
        dev = _safe_int(obs.get("development", 0))
        king = _safe_int(obs.get("king_safety", 5))
        center = _safe_int(obs.get("center_control", 0))
        return {
            "score_delta": material + center // 2,
            "pressure": dev + center,
            "risk": max(0, 6 - king),
            "tempo": _safe_int(obs.get("tempo", 2)),
            "control": center,
            "resource": dev,
        }

    if v == "go_world":
        territory = _safe_int(obs.get("territory_delta", 0))
        liberties = _safe_int(obs.get("liberties_delta", 0))
        influence = _safe_int(obs.get("influence", 0))
        threat = _safe_int(obs.get("capture_threat", 0))
        ko_risk = _safe_int(obs.get("ko_risk", 0))
        return {
            "score_delta": territory,
            "pressure": threat + influence,
            "risk": max(0, ko_risk - liberties),
            "tempo": _safe_int(obs.get("tempo", 2)),
            "control": influence,
            "resource": liberties,
        }

    if v == "uno_world":
        my_cards = _safe_int(obs.get("my_cards", 7))
        opp_cards = _safe_int(obs.get("opp_cards", 7))
        control = _safe_int(obs.get("color_control", 0))
        charge = _safe_int(obs.get("action_charge", 0))
        draw_pressure = _safe_int(obs.get("draw_pressure", 0))
        return {
            "score_delta": opp_cards - my_cards,
            "pressure": draw_pressure,
            "risk": max(0, draw_pressure - control),
            "tempo": _safe_int(obs.get("tempo", 2)),
            "control": control,
            "resource": charge,
        }
    return None


def _strategy_obs_from_signature(
    *,
    signature: Dict[str, int],
    target_verse_name: str,
    t_value: int,
) -> Dict[str, JSONValue]:
    v = _norm_verse_name(target_verse_name)
    score_delta = _safe_int(signature.get("score_delta", 0))
    pressure = _safe_int(signature.get("pressure", 0))
    risk = max(0, _safe_int(signature.get("risk", 0)))
    tempo = max(0, _safe_int(signature.get("tempo", 0)))
    control = _safe_int(signature.get("control", 0))
    resource = _safe_int(signature.get("resource", 0))

    if v == "chess_world":
        material = score_delta
        center = control
        development = max(0, resource)
        king_safety = max(0, 6 - risk)
        return {
            "material_delta": material,
            "development": development,
            "king_safety": king_safety,
            "center_control": center,
            "score_delta": score_delta,
            "pressure": pressure,
            "risk": risk,
            "tempo": tempo,
            "control": control,
            "resource": resource,
            "phase": 1,
            "t": int(t_value),
        }

    if v == "go_world":
        territory = score_delta
        influence = control
        liberties = resource
        return {
            "territory_delta": territory,
            "liberties_delta": liberties,
            "influence": influence,
            "capture_threat": max(0, pressure - influence),
            "ko_risk": risk,
            "score_delta": score_delta,
            "pressure": pressure,
            "risk": risk,
            "tempo": tempo,
            "control": control,
            "resource": resource,
            "consecutive_passes": 0,
            "t": int(t_value),
        }

    if v == "uno_world":
        # Keep total hand mass stable around 14 for synthetic transfer.
        my_cards = max(0, int(round((14.0 - float(score_delta)) / 2.0)))
        opp_cards = max(0, my_cards + score_delta)
        return {
            "my_cards": my_cards,
            "opp_cards": opp_cards,
            "color_control": control,
            "action_charge": max(0, resource),
            "draw_pressure": max(0, pressure),
            "uno_ready": int(my_cards <= 1),
            "score_delta": score_delta,
            "pressure": pressure,
            "risk": risk,
            "tempo": tempo,
            "control": control,
            "resource": resource,
            "t": int(t_value),
        }

    return {
        "score_delta": score_delta,
        "pressure": pressure,
        "risk": risk,
        "tempo": tempo,
        "control": control,
        "resource": resource,
        "t": int(t_value),
    }


def _strategy_obs_to_warehouse(
    *,
    signature: Dict[str, int],
    t_value: int,
) -> Dict[str, JSONValue]:
    score_delta = _safe_int(signature.get("score_delta", 0))
    pressure = _safe_int(signature.get("pressure", 0))
    risk = max(0, _safe_int(signature.get("risk", 0)))
    tempo = max(0, _safe_int(signature.get("tempo", 0)))
    control = _safe_int(signature.get("control", 0))
    resource = _safe_int(signature.get("resource", 0))

    # Project strategic features into warehouse navigation state.
    # Wider coefficient spread to cover the full 8x8 grid instead of clustering
    # around a narrow center region. This is critical for transfer DNA diversity.
    width = 8
    height = 8
    gx = width - 1
    gy = height - 1

    # X position: strong score_delta + control drive rightward progress.
    # Use wider range (0.85 * score_delta) so low/high scores reach grid edges.
    progress = int(
        round(
            3.5
            + 0.85 * float(score_delta)
            + 0.40 * float(control)
            + 0.20 * float(tempo)
            - 0.35 * float(risk)
        )
    )
    # Y position: pressure/control differential drives vertical spread.
    # Include tempo for more diversity; wider coefficients.
    lateral = int(
        round(
            3.5
            + 0.45 * float(pressure)
            - 0.30 * float(control)
            + 0.25 * float(score_delta)
            + 0.15 * float(tempo)
            - 0.20 * float(risk)
        )
    )
    x = max(0, min(gx, progress))
    y = max(0, min(gy, lateral))

    # Battery: resource-rich states get high battery; risky states get low battery.
    # Use battery_capacity=20 from default warehouse params.
    dist_to_goal = abs(x - gx) + abs(y - gy)
    battery_base = 10.0 + 1.8 * float(resource) + 0.5 * float(tempo) - 1.5 * float(risk)
    # Scale battery relative to remaining distance so agent learns resource pacing.
    battery_need = max(1.0, float(dist_to_goal) * 1.1)
    battery = max(1, min(20, int(round(max(battery_base, battery_need)))))

    # Nearby obstacles: high-risk states have more obstacle proximity.
    nearby_obstacles = max(0, min(4, int(round(0.8 * float(risk) + 0.15 * float(pressure)))))

    # Nearest charger: resource-poor + high-risk = closer charger urgency.
    nearest_charger = max(0, min(10, int(round(
        3.0 - 0.5 * float(resource) + 0.6 * float(risk) - 0.2 * float(control)
    ))))
    return {
        "x": int(x),
        "y": int(y),
        "goal_x": int(gx),
        "goal_y": int(gy),
        "battery": int(battery),
        "nearby_obstacles": int(nearby_obstacles),
        "nearest_charger_dist": int(nearest_charger),
        "t": int(t_value),
        "flat": [
            float(x),
            float(y),
            float(gx),
            float(gy),
            float(battery),
            float(nearby_obstacles),
            float(t_value),
        ],
    }


def _strategy_obs_to_labyrinth(
    *,
    signature: Dict[str, int],
    t_value: int,
) -> Dict[str, JSONValue]:
    score_delta = _safe_int(signature.get("score_delta", 0))
    pressure = _safe_int(signature.get("pressure", 0))
    risk = max(0, _safe_int(signature.get("risk", 0)))
    control = _safe_int(signature.get("control", 0))
    resource = _safe_int(signature.get("resource", 0))

    x = max(1, min(13, int(round(4.0 + 0.5 * float(score_delta) + 0.3 * float(control)))))
    y = max(1, min(9, int(round(3.0 + 0.25 * float(pressure - control)))))
    battery = max(5, min(80, int(round(35.0 + 2.2 * float(resource) - 1.5 * float(risk)))))
    near_pits = max(0, min(9, int(round(0.9 * float(risk)))))
    near_lasers = max(0, min(9, int(round(0.5 * float(max(0, pressure - control))))))
    return {
        "x": int(x),
        "y": int(y),
        "t": int(t_value),
        "battery": int(battery),
        "goal_visible": 0,
        "goal_dx": 0,
        "goal_dy": 0,
        "hazard_up": int(near_pits > 0),
        "hazard_down": int(near_lasers > 0),
        "hazard_left": int(risk > 0),
        "hazard_right": int(max(0, risk - 1) > 0),
        "near_pits": int(near_pits),
        "near_lasers": int(near_lasers),
    }


def _strategy_obs_to_trade(
    *,
    signature: Dict[str, int],
    t_value: int,
) -> Dict[str, JSONValue]:
    score_delta = _safe_int(signature.get("score_delta", 0))
    risk = max(0, _safe_int(signature.get("risk", 0)))
    control = _safe_int(signature.get("control", 0))
    
    # Trade:
    # Score -> Portfolio/Cash
    # Control -> Inventory
    # Risk -> Price volatility
    
    cash = 100.0 + float(score_delta) * 5.0
    inventory = max(0, min(10, int(control // 2)))
    price = 20.0 + float(risk) * 2.0
    
    return {
        "price": price,
        "price_delta": 0.0,
        "cash": cash,
        "inventory": inventory,
        "avg_buy_price": 18.0 if inventory > 0 else 0.0,
        "portfolio_value": cash + inventory * price,
        "cycle_phase": 0.0,
        "t": int(t_value),
    }


def _strategy_obs_to_escape(
    *,
    signature: Dict[str, int],
    t_value: int,
) -> Dict[str, JSONValue]:
    score_delta = _safe_int(signature.get("score_delta", 0))
    risk = max(0, _safe_int(signature.get("risk", 0)))
    control = _safe_int(signature.get("control", 0))
    
    # Escape: 10x10 grid.
    # Progress (score) maps to being closer to exit. Exit at (9,0). Start (0,9).
    # Normalized progress 0..1
    prog = max(0.0, min(1.0, (float(score_delta) + 5.0) / 15.0))
    x = int(prog * 9.0)
    y = int((1.0 - prog) * 9.0)
    
    exit_dist = 18 - (x + (9 - y)) # Manhattan dist to (9,0)
    
    # Risk maps to guard proximity
    guards = max(0, min(3, int(round(float(risk) * 0.5))))
    nearest_guard = 1 if guards > 0 else 5
    
    return {
        "x": x, "y": y,
        "exit_dist": max(0, exit_dist),
        "nearest_guard_dist": nearest_guard,
        "hidden_steps_left": 0,
        "guards_in_vision": guards,
        "on_hiding_spot": 0,
        "t": int(t_value),
    }


def _strategy_obs_to_factory(
    *,
    signature: Dict[str, int],
    t_value: int,
) -> Dict[str, JSONValue]:
    score_delta = _safe_int(signature.get("score_delta", 0))
    pressure = _safe_int(signature.get("pressure", 0))
    risk = max(0, _safe_int(signature.get("risk", 0)))
    resource = _safe_int(signature.get("resource", 0))
    
    # Factory: score -> completed items
    # Resource -> buffer levels
    completed = max(0, int(score_delta * 2 + 10))
    buf_lvl = max(0, min(4, int(resource // 2)))
    
    # Risk -> machine breakdowns
    broken_0 = 1 if risk > 2 else 0
    broken_1 = 1 if risk > 4 else 0
    broken_2 = 1 if risk > 5 else 0
    
    obs: Dict[str, JSONValue] = {
        "t": int(t_value),
        "completed": completed,
        "total_arrived": completed + 10,
        "output_buf": 0,
        "buf_0": buf_lvl, "buf_1": buf_lvl, "buf_2": buf_lvl,
        "broken_0": broken_0, "broken_1": broken_1, "broken_2": broken_2,
        "repair_0": 0, "repair_1": 0, "repair_2": 0,
    }
    return obs


def _strategy_obs_to_bridge(
    *,
    signature: Dict[str, int],
    t_value: int,
) -> Dict[str, JSONValue]:
    score_delta = _safe_int(signature.get("score_delta", 0))
    risk = max(0, _safe_int(signature.get("risk", 0)))
    control = _safe_int(signature.get("control", 0))
    
    # Bridge: 8 segments
    # Development/Score -> Cursor progression
    cursor = max(0, min(8, int(4 + score_delta)))
    placed = cursor
    
    # Risk -> Wind
    wind = 1 if risk > 2 else 0
    
    # Control -> Strong segments vs Weak
    strong_ratio = max(0.0, min(1.0, float(control) / 10.0))
    strong_cnt = int(placed * strong_ratio)
    weak_cnt = placed - strong_cnt
    
    return {
        "cursor": cursor,
        "segments_placed": placed,
        "segments_intact": placed,
        "weak_count": weak_cnt,
        "strong_count": strong_cnt,
        "wind_active": wind,
        "bridge_complete": 1 if placed >= 8 else 0,
        "t": int(t_value),
    }


def infer_verse_from_obs(obs: JSONValue) -> Optional[str]:
    if not isinstance(obs, dict):
        return None
    keys = set(str(k) for k in obs.keys())
    if {"material_delta", "development", "king_safety", "phase"}.issubset(keys):
        return "chess_world"
    if {"territory_delta", "liberties_delta", "ko_risk"}.issubset(keys):
        return "go_world"
    if {"my_cards", "opp_cards", "uno_ready"}.issubset(keys):
        return "uno_world"
    if {"x", "y", "exit_dist", "nearest_guard_dist", "hidden_steps_left"}.issubset(keys):
        return "escape_world"
    if {"t", "completed", "total_arrived", "buf_0"}.issubset(keys):
        return "factory_world"
    if {"cursor", "segments_placed", "weak_count", "bridge_complete"}.issubset(keys):
        return "bridge_world"
    if {"price", "cash", "inventory", "portfolio_value"}.issubset(keys):
        return "trade_world"
    if {"pos", "goal", "t"}.issubset(keys):
        return "line_world"
    if {"carrying", "deposited", "fruit_remaining"}.issubset(keys):
        return "harvest_world"
    if {"flood_level", "on_mud", "immunity_left"}.issubset(keys):
        return "swamp_world"
    if {"x", "y", "goal_x", "goal_y", "battery", "t"}.issubset(keys):
        return "warehouse_world"
    if {"x", "y", "goal_x", "goal_y", "t"}.issubset(keys):
        return "grid_world"
    if {"x", "y", "t"}.issubset(keys):
        return "cliff_world"
    return None


def translate_observation(
    *,
    obs: JSONValue,
    source_verse_name: str,
    target_verse_name: str,
) -> Optional[JSONValue]:
    src = _norm_verse_name(source_verse_name)
    dst = _norm_verse_name(target_verse_name)

    if src == dst:
        return obs
    if not can_bridge(src, dst):
        return None

    if not isinstance(obs, dict):
        return None

    if _is_strategy_verse(src) and _is_strategy_verse(dst):
        sig = _strategy_signature(obs, src)
        if sig is None:
            return None
        t_value = _safe_int(obs.get("t", 0))
        return _strategy_obs_from_signature(signature=sig, target_verse_name=dst, t_value=t_value)
    if _is_strategy_verse(src) and dst == "warehouse_world":
        sig = _strategy_signature(obs, src)
        if sig is None:
            return None
        t_value = _safe_int(obs.get("t", 0))
        return _strategy_obs_to_warehouse(signature=sig, t_value=t_value)
    if _is_strategy_verse(src) and dst == "labyrinth_world":
        sig = _strategy_signature(obs, src)
        if sig is None:
            return None
        t_value = _safe_int(obs.get("t", 0))
        return _strategy_obs_to_labyrinth(signature=sig, t_value=t_value)
    if _is_strategy_verse(src) and dst == "escape_world":
        sig = _strategy_signature(obs, src)
        if sig is None: return None
        return _strategy_obs_to_escape(signature=sig, t_value=_safe_int(obs.get("t", 0)))
    if _is_strategy_verse(src) and dst == "factory_world":
        sig = _strategy_signature(obs, src)
        if sig is None: return None
        return _strategy_obs_to_factory(signature=sig, t_value=_safe_int(obs.get("t", 0)))

    if _is_strategy_verse(src) and dst == "bridge_world":
        sig = _strategy_signature(obs, src)
        if sig is None: return None
        return _strategy_obs_to_bridge(signature=sig, t_value=_safe_int(obs.get("t", 0)))
    if _is_strategy_verse(src) and dst == "trade_world":
        sig = _strategy_signature(obs, src)
        if sig is None: return None
        return _strategy_obs_to_trade(signature=sig, t_value=_safe_int(obs.get("t", 0)))

    # 1D <-> 1D projections.
    if src == "line_world" and dst in ("park_world",):
        return {
            "pos": _safe_int(obs.get("pos", 0)),
            "goal": _safe_int(obs.get("goal", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "park_world" and dst == "line_world":
        return {
            "pos": _safe_int(obs.get("pos", 0)),
            "goal": _safe_int(obs.get("goal", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "grid_world" and dst == "park_world":
        return {
            "pos": _safe_int(obs.get("x", 0)),
            "goal": _safe_int(obs.get("goal_x", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "park_world" and dst == "grid_world":
        return {
            "x": _safe_int(obs.get("pos", 0)),
            "y": 0,
            "goal_x": _safe_int(obs.get("goal", 0)),
            "goal_y": 0,
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "line_world" and dst == "pursuit_world":
        return {
            "agent": _safe_int(obs.get("pos", 0)),
            "target": _safe_int(obs.get("goal", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "pursuit_world" and dst == "line_world":
        return {
            "pos": _safe_int(obs.get("agent", 0)),
            "goal": _safe_int(obs.get("target", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "park_world" and dst == "pursuit_world":
        return {
            "agent": _safe_int(obs.get("pos", 0)),
            "target": _safe_int(obs.get("goal", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "pursuit_world" and dst == "park_world":
        return {
            "pos": _safe_int(obs.get("agent", 0)),
            "goal": _safe_int(obs.get("target", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }

    if src == "line_world" and dst == "grid_world":
        return {
            "x": _safe_int(obs.get("pos", 0)),
            "y": 0,
            "goal_x": _safe_int(obs.get("goal", 0)),
            "goal_y": 0,
            "t": _safe_int(obs.get("t", 0)),
        }

    if src == "grid_world" and dst == "line_world":
        return {
            "pos": _safe_int(obs.get("x", 0)),
            "goal": _safe_int(obs.get("goal_x", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }

    if src == "grid_world" and dst == "cliff_world":
        return {
            "x": _safe_int(obs.get("x", 0)),
            "y": _safe_int(obs.get("y", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "grid_world" and dst == "warehouse_world":
        x = _safe_int(obs.get("x", 0))
        y = _safe_int(obs.get("y", 0))
        gx = _safe_int(obs.get("goal_x", 0))
        gy = _safe_int(obs.get("goal_y", 0))
        t = _safe_int(obs.get("t", 0))
        nearby = 0
        nearest_charger = abs(x - gx) + abs(y - gy)
        return {
            "x": x,
            "y": y,
            "goal_x": gx,
            "goal_y": gy,
            "battery": 20,
            "nearby_obstacles": nearby,
            "nearest_charger_dist": nearest_charger,
            "t": t,
            "flat": [float(x), float(y), float(gx), float(gy), 20.0, float(nearby), float(t)],
        }
    if src == "warehouse_world" and dst == "grid_world":
        return {
            "x": _safe_int(obs.get("x", 0)),
            "y": _safe_int(obs.get("y", 0)),
            "goal_x": _safe_int(obs.get("goal_x", 0)),
            "goal_y": _safe_int(obs.get("goal_y", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "line_world" and dst == "warehouse_world":
        x = _safe_int(obs.get("pos", 0))
        gx = _safe_int(obs.get("goal", 0))
        t = _safe_int(obs.get("t", 0))
        return {
            "x": x,
            "y": 0,
            "goal_x": gx,
            "goal_y": 0,
            "battery": 20,
            "nearby_obstacles": 0,
            "nearest_charger_dist": abs(x - gx),
            "t": t,
            "flat": [float(x), 0.0, float(gx), 0.0, 20.0, 0.0, float(t)],
        }
    if src == "warehouse_world" and dst == "line_world":
        return {
            "pos": _safe_int(obs.get("x", 0)),
            "goal": _safe_int(obs.get("goal_x", 0)),
            "t": _safe_int(obs.get("t", 0)),
        }

    # ---- harvest_world bridges ----
    # harvest <-> grid: project 2D position
    if src == "grid_world" and dst == "harvest_world":
        x = _safe_int(obs.get("x", 0))
        y = _safe_int(obs.get("y", 0))
        t = _safe_int(obs.get("t", 0))
        return {
            "x": x, "y": y,
            "carrying": 0, "deposited": 0, "fruit_remaining": 3,
            "nearby_fruit": 0, "nearest_fruit_dist": 5, "nearest_base_dist": x + y,
            "t": t,
        }
    if src == "harvest_world" and dst == "grid_world":
        return {
            "x": _safe_int(obs.get("x", 0)),
            "y": _safe_int(obs.get("y", 0)),
            "goal_x": 0, "goal_y": 0,  # base is the effective goal
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "warehouse_world" and dst == "harvest_world":
        x = _safe_int(obs.get("x", 0))
        y = _safe_int(obs.get("y", 0))
        t = _safe_int(obs.get("t", 0))
        return {
            "x": x, "y": y,
            "carrying": 0, "deposited": 0, "fruit_remaining": 3,
            "nearby_fruit": _safe_int(obs.get("nearby_obstacles", 0)),
            "nearest_fruit_dist": 3, "nearest_base_dist": x + y,
            "t": t,
        }
    if src == "harvest_world" and dst == "warehouse_world":
        x = _safe_int(obs.get("x", 0))
        y = _safe_int(obs.get("y", 0))
        t = _safe_int(obs.get("t", 0))
        return {
            "x": x, "y": y,
            "goal_x": 7, "goal_y": 7,
            "battery": 20, "nearby_obstacles": 0,
            "nearest_charger_dist": 3, "t": t,
            "on_conveyor": 0, "patrol_dist": -1,
            "flat": [float(x), float(y), 7.0, 7.0, 20.0, 0.0, float(t), 0.0, -1.0],
        }

    # ---- swamp_world bridges ----
    if src == "grid_world" and dst == "swamp_world":
        x = _safe_int(obs.get("x", 0))
        y = _safe_int(obs.get("y", 0))
        gx = _safe_int(obs.get("goal_x", 0))
        gy = _safe_int(obs.get("goal_y", 0))
        t = _safe_int(obs.get("t", 0))
        return {
            "x": x, "y": y,
            "flood_level": 0, "on_mud": 0, "immunity_left": 0,
            "nearest_haven_dist": 5, "goal_dist": abs(x - gx) + abs(y - gy),
            "t": t,
        }
    if src == "swamp_world" and dst == "grid_world":
        return {
            "x": _safe_int(obs.get("x", 0)),
            "y": _safe_int(obs.get("y", 0)),
            "goal_x": 0, "goal_y": 0,
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "cliff_world" and dst == "swamp_world":
        x = _safe_int(obs.get("x", 0))
        y = _safe_int(obs.get("y", 0))
        t = _safe_int(obs.get("t", 0))
        return {
            "x": x, "y": y,
            "flood_level": 0, "on_mud": 0, "immunity_left": 0,
            "nearest_haven_dist": 3, "goal_dist": 10 - x,
            "t": t,
        }
    if src == "swamp_world" and dst == "cliff_world":
        return {
            "x": _safe_int(obs.get("x", 0)),
            "y": _safe_int(obs.get("y", 0)),
            "t": _safe_int(obs.get("t", 0)),
            "cliff_adjacent": 1 if _safe_int(obs.get("flood_level", 0)) > 0 else 0,
            "wind_active": 0,
            "crumbled_count": 0,
        }

    # ---- escape_world bridges ----
    # escape -> grid: direct map
    if src == "escape_world" and dst == "grid_world":
        return {
            "x": _safe_int(obs.get("x", 0)),
            "y": _safe_int(obs.get("y", 0)),
            "goal_x": 9, "goal_y": 0,  # Escape exit is usually at (width-1, 0)
            "t": _safe_int(obs.get("t", 0)),
        }
    # grid -> escape:
    if src == "grid_world" and dst == "escape_world":
        x = _safe_int(obs.get("x", 0))
        y = _safe_int(obs.get("y", 0))
        gx = _safe_int(obs.get("goal_x", 0))
        gy = _safe_int(obs.get("goal_y", 0))
        return {
            "x": x, "y": y,
            "exit_dist": abs(x - gx) + abs(y - gy),
            "nearest_guard_dist": 5, "hidden_steps_left": 0,
            "guards_in_vision": 0, "on_hiding_spot": 0,
            "t": _safe_int(obs.get("t", 0)),
        }

    # ---- bridge_world bridges ----
    # bridge -> line: cursor is pos
    if src == "bridge_world" and dst == "line_world":
        return {
            "pos": _safe_int(obs.get("cursor", 0)),
            "goal": 8,  # approx bridge length
            "t": _safe_int(obs.get("t", 0)),
        }
    # line -> bridge
    if src == "line_world" and dst == "bridge_world":
        pos = _safe_int(obs.get("pos", 0))
        return {
            "cursor": pos,
            "segments_placed": pos, "segments_intact": pos,
            "weak_count": 0, "strong_count": pos,
            "wind_active": 0, "bridge_complete": 0,
            "t": _safe_int(obs.get("t", 0)),
        }

    # ---- factory_world bridges ----
    # factory -> harvest:
    # completed -> deposited
    # buf_0 (input) -> nearby_fruit (approx)
    if src == "factory_world" and dst == "harvest_world":
        comp = _safe_int(obs.get("completed", 0))
        return {
            "x": 0, "y": 0,
            "carrying": 0, "deposited": comp,
            "fruit_remaining": 100 - comp,
            "nearby_fruit": _safe_int(obs.get("buf_0", 0)),
            "nearest_fruit_dist": 1, "nearest_base_dist": 1,
            "t": _safe_int(obs.get("t", 0)),
        }
    # harvest -> factory
    if src == "harvest_world" and dst == "factory_world":
        dep = _safe_int(obs.get("deposited", 0))
        return {
            "t": _safe_int(obs.get("t", 0)),
            "completed": dep, "total_arrived": dep + 5,
            "buf_0": _safe_int(obs.get("nearby_fruit", 0)),
            "buf_1": 0, "buf_2": 0,
            "broken_0": 0, "broken_1": 0, "broken_2": 0,
            "output_buf": 0,
        }

    # ---- trade_world bridges ----
    # trade <-> harvest (resource cycle analogy)
    if src == "trade_world" and dst == "harvest_world":
        inv = _safe_int(obs.get("inventory", 0))
        return {
            "x": 0, "y": 0,
            "carrying": inv, 
            "deposited": int(_safe_float(obs.get("total_profit", 0.0)) / 2.0),
            "fruit_remaining": 50,
            "nearby_fruit": 1 if _safe_float(obs.get("price", 100)) < 20 else 0, # signal low price as fruit avail
            "nearest_fruit_dist": 1, "nearest_base_dist": 1,
            "t": _safe_int(obs.get("t", 0)),
        }
    if src == "harvest_world" and dst == "trade_world":
        carrying = _safe_int(obs.get("carrying", 0))
        return {
            "price": 20.0, "price_delta": 0.0,
            "cash": 100.0, "inventory": carrying,
            "avg_buy_price": 0.0,
            "portfolio_value": 100.0 + carrying * 20.0,
            "cycle_phase": 0.0,
            "t": _safe_int(obs.get("t", 0)),
        }

    return None


def translate_action(
    *,
    action: JSONValue,
    source_verse_name: str,
    target_verse_name: str,
) -> Optional[JSONValue]:
    src = _norm_verse_name(source_verse_name)
    dst = _norm_verse_name(target_verse_name)
    if src == dst:
        return action
    if not can_bridge(src, dst):
        return None

    try:
        a = int(action)  # type: ignore[arg-type]
    except Exception:
        return None

    if _is_strategy_verse(src) and _is_strategy_verse(dst):
        if 0 <= a <= 5:
            return int(a)
        return None
    if _is_strategy_verse(src) and dst in ("warehouse_world", "labyrinth_world"):
        # Map strategy primitives to navigation controls (balanced across all 5 actions):
        # build(0)->right(3)  move toward goal via progress
        # pressure(1)->down(1) lateral advance
        # capture(2)->up(0)   aggressive repositioning
        # defend(3)->wait/charge(4) consolidate / recharge
        # tempo(4)->left(2)   tactical retreat / reposition
        # convert(5)->right(3) push toward goal completion
        mapping = {
            0: 3,  # build -> right
            1: 1,  # pressure -> down
            2: 0,  # capture -> up
            3: 4,  # defend -> wait/charge
            4: 2,  # tempo -> left
            5: 3,  # convert -> right
        }
        mapping = {
            0: 3,  # build -> right
            1: 1,  # pressure -> down
            2: 0,  # capture -> up
            3: 4,  # defend -> wait/charge
            4: 2,  # tempo -> left
            5: 3,  # convert -> right
        }
        return mapping.get(int(a))
    if _is_strategy_verse(src) and dst == "escape_world":
        # 0=up,1=down,2=left,3=right,4=hide
        mapping = {
            0: 3, # build -> right (progress)
            1: 1, # pressure -> down
            2: 4, # capture -> hide (ambush)
            3: 4, # defend -> hide
            4: 2, # tempo -> left
            5: 0, # convert -> up
        }
        return mapping.get(int(a))
    if _is_strategy_verse(src) and dst == "bridge_world":
        # 0=weak, 1=strong, 2=wait, 3=cross
        mapping = {
            0: 0, # build -> weak
            1: 1, # pressure -> strong
            2: 3, # capture -> cross (aggression)
            3: 2, # defend -> wait
            4: 0, # tempo -> weak (fast)
            5: 3, # convert -> cross
        }
        return mapping.get(int(a))
    if _is_strategy_verse(src) and dst == "factory_world":
        # 0,1,2=run  3,4,5=repair  6=idle
        mapping = {
            0: 0, # build -> run 0
            1: 1, # pressure -> run 1
            2: 2, # capture -> run 2
            3: 3, # defend -> repair 0
            4: 6, # tempo -> idle
            5: 2, # convert -> run 2
        }
        return mapping.get(int(a))
    if _is_strategy_verse(src) and dst == "trade_world":
        # 0=buy, 1=sell, 2=hold
        mapping = {
            0: 0, # build -> buy (accumulate)
            1: 2, # pressure -> hold
            2: 0, # capture -> buy
            3: 2, # defend -> hold
            4: 2, # tempo -> hold
            5: 1, # convert -> sell (realize)
        }
        return mapping.get(int(a))

    # line: 0=left,1=right  -> grid: 2=left,3=right
    if src == "line_world" and dst == "grid_world":
        if a == 0:
            return 2
        if a == 1:
            return 3
        return None

    # grid: 2=left,3=right -> line: 0=left,1=right
    if src == "grid_world" and dst == "line_world":
        if a == 2:
            return 0
        if a == 3:
            return 1
        return None

    # grid <-> cliff share action semantics.
    if src == "grid_world" and dst == "cliff_world":
        if a in (0, 1, 2, 3):
            return a
        return None
    if src == "cliff_world" and dst == "grid_world":
        if a in (0, 1, 2, 3):
            return a
        return None
    if src == "grid_world" and dst == "warehouse_world":
        if a in (0, 1, 2, 3):
            return a
        return None
    if src == "warehouse_world" and dst == "grid_world":
        if a in (0, 1, 2, 3):
            return a
        if a == 4:
            return 0
        return None
    if src == "line_world" and dst == "warehouse_world":
        if a == 0:
            return 2
        if a == 1:
            return 3
        return None
    if src == "warehouse_world" and dst == "line_world":
        if a == 2:
            return 0
        if a == 3:
            return 1
        return None

    # line <-> park (ignore park action).
    if src == "line_world" and dst == "park_world":
        if a in (0, 1):
            return a
        return None
    if src == "park_world" and dst == "line_world":
        if a == 0:
            return 0
        if a == 1:
            return 1
        return None
    if src == "grid_world" and dst == "park_world":
        if a == 2:
            return 0
        if a == 3:
            return 1
        if a in (0, 1):
            return 2
        return None
    if src == "park_world" and dst == "grid_world":
        if a == 0:
            return 2
        if a == 1:
            return 3
        if a == 2:
            return 3
        return None

    # line/park <-> pursuit, preserve lateral motion.
    if src in ("line_world", "park_world") and dst == "pursuit_world":
        if a == 0:
            return 0
        if a == 1:
            return 1
        return None
    if src == "pursuit_world" and dst in ("line_world", "park_world"):
        if a == 0:
            return 0
        if a == 1:
            return 1
        return None

    # ---- harvest_world: 0=up,1=down,2=left,3=right,4=deposit ----
    nav_4d = {"grid_world", "cliff_world", "swamp_world"}
    nav_5d = {"warehouse_world", "harvest_world"}
    # 4-action nav <-> harvest
    if src in nav_4d and dst == "harvest_world":
        if a in (0, 1, 2, 3):
            return a
        return None
    if src == "harvest_world" and dst in nav_4d:
        if a in (0, 1, 2, 3):
            return a
        return None
    # 5-action nav <-> harvest
    if src in nav_5d and dst == "harvest_world":
        if a in (0, 1, 2, 3, 4):
            return a
        return None
    if src == "harvest_world" and dst in nav_5d:
        if a in (0, 1, 2, 3, 4):
            return a
        return None

    # ---- swamp_world: 0=up,1=down,2=left,3=right ----
    if src in nav_4d and dst == "swamp_world":
        if a in (0, 1, 2, 3):
            return a
        return None
    if src == "swamp_world" and dst in nav_4d:
        if a in (0, 1, 2, 3):
            return a
        return None
    if src in nav_5d and dst == "swamp_world":
        if a in (0, 1, 2, 3):
            return a
        return None
    if src == "swamp_world" and dst in nav_5d:
        if a in (0, 1, 2, 3):
            return a
        return None

    # ---- escape_world: 0..3=move, 4=hide ----
    nav_escape = {"grid_world", "cliff_world", "swamp_world", "warehouse_world", "labyrinth_world"}
    if src in nav_escape and dst == "escape_world":
        if a in (0, 1, 2, 3):
            return a
        if a == 4 and _norm_verse_name(src) == "warehouse_world":
             # warehouse wait/charge -> hide
            return 4
        return None
    if src == "escape_world" and dst in nav_escape:
        if a in (0, 1, 2, 3):
            return a
        if a == 4 and _norm_verse_name(dst) == "warehouse_world":
            return 4 # hide -> charge
        return None

    # ---- bridge_world: 0=weak, 1=strong, 2=wait, 3=cross ----
    # bridge <-> line (0=left, 1=right)
    if src == "bridge_world" and dst == "line_world":
        if a in (0, 1): # place -> right (progress)
            return 1
        if a == 2: # wait -> left (delay/regress)
            return 0
        if a == 3: # cross -> right
            return 1
        return None
    if src == "line_world" and dst == "bridge_world":
        if a == 1: # right -> place strong
            return 1
        return 2 # left -> wait

    # ---- factory_world: 0..2=run, 3..5=repair, 6=idle (default 3 machines) ----
    # factory <-> harvest (0..3 move, 4 deposit)
    if src == "factory_world" and dst == "harvest_world":
        if a < 3: # run -> deposit
            return 4
        return 0 # repair/idle -> move 0
    if src == "harvest_world" and dst == "factory_world":
        if a == 4: # deposit -> run machine 2 (finish)
            return 2
        return 6 # move -> idle

    return None


def translate_transition(
    *,
    obs: JSONValue,
    action: JSONValue,
    source_verse_name: str,
    target_verse_name: str,
    next_obs: Optional[JSONValue] = None,
    learned_bridge_enabled: bool = False,
    learned_bridge_model_path: Optional[str] = None,
) -> Optional[Dict[str, JSONValue]]:
    translated_obs = translate_observation(
        obs=obs,
        source_verse_name=source_verse_name,
        target_verse_name=target_verse_name,
    )
    if translated_obs is None:
        return None

    translated_action = translate_action(
        action=action,
        source_verse_name=source_verse_name,
        target_verse_name=target_verse_name,
    )
    if translated_action is None:
        return None

    translated_next_obs: Optional[JSONValue] = None
    if next_obs is not None:
        translated_next_obs = translate_observation(
            obs=next_obs,
            source_verse_name=source_verse_name,
            target_verse_name=target_verse_name,
        )

    out: Dict[str, JSONValue] = {
        "obs": translated_obs,
        "action": translated_action,
        "next_obs": translated_next_obs,
    }
    if _safe_bool(learned_bridge_enabled, False):
        sim, conf, resolved = _learned_bridge_metrics(
            source_obs=obs,
            translated_obs=translated_obs,
            model_path=learned_bridge_model_path,
        )
        out["learned_bridge_enabled"] = True
        out["learned_bridge_similarity"] = sim
        out["learned_bridge_confidence"] = conf
        out["learned_bridge_model_path"] = resolved
    return out


def _synthesize_warehouse_labels(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    c = cfg if isinstance(cfg, dict) else {}
    o = obs if isinstance(obs, dict) else {}
    n = next_obs if isinstance(next_obs, dict) else {}

    x = _safe_int(o.get("x", 0), 0)
    y = _safe_int(o.get("y", 0), 0)
    gx = _safe_int(o.get("goal_x", 7), 7)
    gy = _safe_int(o.get("goal_y", 7), 7)
    battery = max(0, _safe_int(o.get("battery", 20), 20))
    nearby = max(0, _safe_int(o.get("nearby_obstacles", 0), 0))
    charger_dist = _safe_int(o.get("nearest_charger_dist", -1), -1)
    t = _safe_int(o.get("t", 0), 0)

    nx = _safe_int(n.get("x", x), x)
    ny = _safe_int(n.get("y", y), y)
    nb = max(0, _safe_int(n.get("battery", battery), battery))

    a = _safe_int(action, 4)
    step_penalty = max(0.0, _safe_float(c.get("step_penalty", 0.08), 0.08))
    wall_penalty = max(0.0, _safe_float(c.get("wall_penalty", 0.40), 0.40))
    obstacle_penalty = max(0.0, _safe_float(c.get("obstacle_penalty", 0.70), 0.70))
    charge_reward = max(0.0, _safe_float(c.get("charge_reward", 0.60), 0.60))
    progress_bonus = max(0.0, _safe_float(c.get("progress_bonus", 0.35), 0.35))
    regress_penalty = max(0.0, _safe_float(c.get("regress_penalty", 0.08), 0.08))
    goal_reward = max(0.0, _safe_float(c.get("goal_reward", 10.0), 10.0))
    battery_fail_penalty = max(0.0, _safe_float(c.get("battery_fail_penalty", 10.0), 10.0))
    # Stronger dense shaping near the goal to reduce circling.
    proximity_bonus_scale = max(0.0, _safe_float(c.get("proximity_bonus_scale", 0.08), 0.08))
    stagnation_penalty = max(0.0, _safe_float(c.get("stagnation_penalty", 0.05), 0.05))
    # New: battery urgency penalty for wasting time when battery is low.
    battery_urgency_scale = max(0.0, _safe_float(c.get("battery_urgency_scale", 0.03), 0.03))

    reward = -float(step_penalty)
    done = False
    info: Dict[str, Any] = {"reached_goal": False}

    if a in (0, 1, 2, 3):
        tx, ty = x, y
        if a == 0:
            ty -= 1
        elif a == 1:
            ty += 1
        elif a == 2:
            tx -= 1
        elif a == 3:
            tx += 1

        out_of_bounds = bool(tx < 0 or ty < 0 or tx > 7 or ty > 7)
        stalled = bool(nx == x and ny == y)
        if out_of_bounds:
            info["hit_wall"] = True
            reward -= float(wall_penalty)
        elif stalled and nearby > 0:
            info["hit_obstacle"] = True
            reward -= float(obstacle_penalty)
        else:
            if next_obs is None:
                nx, ny = tx, ty
            if next_obs is None:
                nb = max(0, battery - 1)
    elif a == 4:
        if charger_dist == 0 and nb > battery:
            info["charged"] = True
            reward += float(charge_reward)
        elif charger_dist == 0 and next_obs is None:
            nb = min(100, battery + 5)
            info["charged"] = True
            reward += float(charge_reward)
        else:
            # Penalize waiting when not on a charger - waste of battery/steps.
            reward -= float(step_penalty) * 0.5

    old_dist = abs(x - gx) + abs(y - gy)
    new_dist = abs(nx - gx) + abs(ny - gy)
    dist_delta = int(old_dist - new_dist)
    if dist_delta > 0:
        # Reward decisive progress; farther states get a slightly larger push.
        far_factor = 1.0 + min(0.8, float(old_dist) / 12.0)
        reward += float(progress_bonus) * float(dist_delta) * float(far_factor)
    elif dist_delta < 0:
        reward -= float(regress_penalty) * float(abs(dist_delta))
    else:
        # Explicit anti-circling pressure for zero-progress moves.
        reward -= float(stagnation_penalty)

    # Additional proximity shaping grows as distance shrinks.
    if float(proximity_bonus_scale) > 0.0:
        reward += float(proximity_bonus_scale) * (1.0 / float(1 + max(0, new_dist)))

    # Battery urgency: penalize being far from goal with low battery.
    if float(battery_urgency_scale) > 0.0 and battery > 0 and new_dist > 0:
        steps_needed = float(new_dist)
        battery_margin = float(battery) - steps_needed
        if battery_margin < 3.0:
            urgency = max(0.0, min(1.0, (3.0 - battery_margin) / 3.0))
            reward -= float(battery_urgency_scale) * urgency

    reached_goal = bool(nx == gx and ny == gy)
    if reached_goal:
        info["reached_goal"] = True
        reward += float(goal_reward)
        done = True

    if nb <= 0 and not reached_goal:
        info["battery_death"] = True
        info["battery_depleted"] = True
        reward -= float(battery_fail_penalty)
        done = True

    info["t"] = int(max(t, _safe_int(n.get("t", t), t)))
    info["battery"] = int(nb)
    info["x"] = int(nx)
    info["y"] = int(ny)
    info["goal_x"] = int(gx)
    info["goal_y"] = int(gy)
    return float(reward), bool(done), info


def _synthesize_labyrinth_labels(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    c = cfg if isinstance(cfg, dict) else {}
    o = obs if isinstance(obs, dict) else {}
    n = next_obs if isinstance(next_obs, dict) else {}

    a = _safe_int(action, 0)
    step_penalty = max(0.0, _safe_float(c.get("step_penalty", 0.01), 0.01))
    hazard_penalty = max(0.0, _safe_float(c.get("hazard_penalty", 0.35), 0.35))
    progress_bonus = max(0.0, _safe_float(c.get("progress_bonus", 0.05), 0.05))
    regress_penalty = max(0.0, _safe_float(c.get("regress_penalty", 0.02), 0.02))
    goal_reward = max(0.0, _safe_float(c.get("goal_reward", 10.0), 10.0))
    battery_fail_penalty = max(0.0, _safe_float(c.get("battery_fail_penalty", 1.0), 1.0))

    reward = -float(step_penalty)
    done = False
    info: Dict[str, Any] = {"reached_goal": False}

    hazard_key = ""
    if a == 0:
        hazard_key = "hazard_up"
    elif a == 1:
        hazard_key = "hazard_down"
    elif a == 2:
        hazard_key = "hazard_left"
    elif a == 3:
        hazard_key = "hazard_right"

    if hazard_key and bool(o.get(hazard_key, 0)):
        if a in (0, 2):
            info["fell_pit"] = True
        else:
            info["hit_laser"] = True
        reward -= float(hazard_penalty)

    old_goal_dist = abs(_safe_int(o.get("goal_dx", 0), 0)) + abs(_safe_int(o.get("goal_dy", 0), 0))
    new_goal_dist = abs(_safe_int(n.get("goal_dx", _safe_int(o.get("goal_dx", 0), 0)), 0)) + abs(
        _safe_int(n.get("goal_dy", _safe_int(o.get("goal_dy", 0), 0)), 0)
    )
    if new_goal_dist < old_goal_dist:
        reward += float(progress_bonus)
    elif new_goal_dist > old_goal_dist:
        reward -= float(regress_penalty)

    reached_goal = bool(_safe_int(n.get("goal_visible", o.get("goal_visible", 0)), 0) == 1 and new_goal_dist == 0)
    if reached_goal:
        info["reached_goal"] = True
        reward += float(goal_reward)
        done = True

    nb = _safe_int(n.get("battery", _safe_int(o.get("battery", 50), 50)), _safe_int(o.get("battery", 50), 50))
    if nb <= 0 and not reached_goal:
        info["battery_depleted"] = True
        reward -= float(battery_fail_penalty)
        done = True

    return float(reward), bool(done), info


def _synthesize_target_labels(
    *,
    target_verse_name: str,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    t = _norm_verse_name(target_verse_name)
    if t == "warehouse_world":
        return _synthesize_warehouse_labels(obs=obs, next_obs=next_obs, action=action, cfg=cfg)
    if t == "labyrinth_world":
        return _synthesize_labyrinth_labels(obs=obs, next_obs=next_obs, action=action, cfg=cfg)
    if t == "harvest_world":
        return _synthesize_harvest_labels(obs=obs, next_obs=next_obs, action=action, cfg=cfg)
    if t == "swamp_world":
        return _synthesize_swamp_labels(obs=obs, next_obs=next_obs, action=action, cfg=cfg)
    if t == "escape_world":
        return _synthesize_escape_labels(obs=obs, next_obs=next_obs, action=action, cfg=cfg)
    if t == "factory_world":
        return _synthesize_factory_labels(obs=obs, next_obs=next_obs, action=action, cfg=cfg)
    if t == "bridge_world":
        return _synthesize_bridge_labels(obs=obs, next_obs=next_obs, action=action, cfg=cfg)
    if t == "trade_world":
        return _synthesize_trade_labels(obs=obs, next_obs=next_obs, action=action, cfg=cfg)
    return 0.0, False, {}


def _synthesize_harvest_labels(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Reward shaping for harvest_world transfer data."""
    c = cfg if isinstance(cfg, dict) else {}
    o = obs if isinstance(obs, dict) else {}
    n = next_obs if isinstance(next_obs, dict) else {}

    step_penalty = max(0.0, _safe_float(c.get("step_penalty", 0.05), 0.05))
    fruit_bonus = max(0.0, _safe_float(c.get("fruit_bonus", 0.50), 0.50))
    deposit_bonus = max(0.0, _safe_float(c.get("deposit_bonus", 3.0), 3.0))
    progress_bonus = max(0.0, _safe_float(c.get("progress_bonus", 0.10), 0.10))

    reward = -float(step_penalty)
    done = False
    info: Dict[str, Any] = {"reached_goal": False}

    carrying = _safe_int(o.get("carrying", 0), 0)
    n_carrying = _safe_int(n.get("carrying", carrying), carrying)
    deposited = _safe_int(o.get("deposited", 0), 0)
    n_deposited = _safe_int(n.get("deposited", deposited), deposited)
    fruit_rem = _safe_int(o.get("fruit_remaining", 3), 3)
    n_fruit_rem = _safe_int(n.get("fruit_remaining", fruit_rem), fruit_rem)

    # Picked fruit?
    if n_carrying > carrying:
        reward += float(fruit_bonus)
        info["picked_fruit"] = True
    # Deposited?
    if n_deposited > deposited:
        depo_count = n_deposited - deposited
        reward += float(deposit_bonus) * depo_count
        info["deposited_fruit"] = depo_count

    # Progress toward nearest fruit
    old_fruit_dist = _safe_int(o.get("nearest_fruit_dist", 5), 5)
    new_fruit_dist = _safe_int(n.get("nearest_fruit_dist", old_fruit_dist), old_fruit_dist)
    if new_fruit_dist < old_fruit_dist and n_carrying < _safe_int(o.get("carry_capacity", 3), 3):
        reward += float(progress_bonus)

    # Progress toward base when carrying
    if carrying > 0:
        old_base = _safe_int(o.get("nearest_base_dist", 5), 5)
        new_base = _safe_int(n.get("nearest_base_dist", old_base), old_base)
        if new_base < old_base:
            reward += float(progress_bonus) * 0.5

    # All deposited
    if n_fruit_rem == 0 and n_carrying == 0 and n_deposited > 0:
        info["reached_goal"] = True
        done = True
        reward += 5.0

    return float(reward), bool(done), info


def _synthesize_swamp_labels(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Reward shaping for swamp_world transfer data."""
    c = cfg if isinstance(cfg, dict) else {}
    o = obs if isinstance(obs, dict) else {}
    n = next_obs if isinstance(next_obs, dict) else {}

    step_penalty = max(0.0, _safe_float(c.get("step_penalty", 0.05), 0.05))
    flood_penalty = max(0.0, _safe_float(c.get("flood_penalty", 8.0), 8.0))
    goal_reward = max(0.0, _safe_float(c.get("goal_reward", 10.0), 10.0))
    progress_bonus = max(0.0, _safe_float(c.get("progress_bonus", 0.15), 0.15))

    reward = -float(step_penalty)
    done = False
    info: Dict[str, Any] = {"reached_goal": False}

    old_goal_dist = _safe_int(o.get("goal_dist", 10), 10)
    new_goal_dist = _safe_int(n.get("goal_dist", old_goal_dist), old_goal_dist)
    if new_goal_dist < old_goal_dist:
        reward += float(progress_bonus)
    elif new_goal_dist > old_goal_dist:
        reward -= float(progress_bonus) * 0.5

    # Check flood danger
    flood_level = _safe_int(n.get("flood_level", 0), 0)
    immunity = _safe_int(n.get("immunity_left", 0), 0)
    if flood_level > 0 and immunity <= 0:
        # Penalize being in flood zone
        reward -= float(step_penalty) * 0.5

    # Goal reached
    if new_goal_dist == 0:
        info["reached_goal"] = True
        done = True
        reward += float(goal_reward)

    return float(reward), bool(done), info


@dataclass
class BridgeStats:
    source_path: str
    output_path: str
    source_verse_name: str
    target_verse_name: str
    input_rows: int
    translated_rows: int
    dropped_rows: int
    learned_bridge_enabled: bool = False
    learned_bridge_model_path: Optional[str] = None
    learned_scored_rows: int = 0
    behavioral_bridge_enabled: bool = False
    behavioral_scored_rows: int = 0
    behavioral_prototype_rows: int = 0


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row




def _normalize_vector(vec: List[float]) -> Optional[List[float]]:
    if not vec:
        return None
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return [float(float(x) / norm) for x in vec]


def _behavior_transition_embedding(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    reward: float,
    done: bool,
) -> Optional[List[float]]:
    try:
        obs_u = obs_to_universal_vector(obs, dim=16)
        nxt_u = obs_to_universal_vector(next_obs if next_obs is not None else obs, dim=16)
    except Exception:
        return None
    if len(obs_u) != len(nxt_u) or len(obs_u) <= 0:
        return None
    delta = [float(nxt_u[i] - obs_u[i]) for i in range(len(obs_u))]
    a_bucket = float(max(0, min(32, _safe_int(action, 0)))) / 32.0
    r_clip = max(-5.0, min(5.0, float(reward))) / 5.0
    payload: List[float] = []
    payload.extend(float(x) for x in obs_u)
    payload.extend(float(x) for x in delta)
    payload.extend([float(a_bucket), float(r_clip), 1.0 if bool(done) else 0.0])
    emb = project_vector(payload, dim=32)
    return _normalize_vector([float(x) for x in emb]) if emb else None


def _build_behavioral_prototype(
    *,
    source_dna_path: str,
    max_rows: int,
) -> Tuple[Optional[List[float]], int]:
    if not os.path.isfile(source_dna_path):
        return None, 0
    limit = max(1, int(max_rows))
    acc = [0.0 for _ in range(32)]
    used = 0
    for row in _iter_jsonl(source_dna_path):
        emb = _behavior_transition_embedding(
            obs=row.get("obs"),
            next_obs=row.get("next_obs"),
            action=row.get("action"),
            reward=_safe_float(row.get("reward", 0.0), 0.0),
            done=bool(row.get("done", False) or row.get("truncated", False)),
        )
        if emb is None:
            continue
        adv = _safe_float(row.get("advantage", 0.0), 0.0)
        rew = _safe_float(row.get("reward", 0.0), 0.0)
        w = 1.0 + max(-0.50, min(2.0, adv))
        if rew > 0.0:
            w += 0.10
        if bool(row.get("done", False)) and rew > 0.0:
            w += 0.10
        w = max(0.05, float(w))
        for i, v in enumerate(emb):
            acc[i] += float(w) * float(v)
        used += 1
        if used >= limit:
            break
    return _normalize_vector(acc), int(used)


def _behavioral_bridge_metrics(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    reward: float,
    done: bool,
    prototype: Optional[List[float]],
) -> Tuple[Optional[float], Optional[float]]:
    if prototype is None:
        return None, None
    emb = _behavior_transition_embedding(
        obs=obs,
        next_obs=next_obs,
        action=action,
        reward=float(reward),
        done=bool(done),
    )
    if emb is None:
        return None, None
    try:
        sim = float(cosine_similarity(emb, prototype))
    except Exception:
        return None, None
    sim = max(-1.0, min(1.0, sim))
    conf = max(0.0, min(1.0, 0.5 * (sim + 1.0)))
    return float(sim), float(conf)

def _compute_transfer_confidence(
    *,
    source_verse: str,
    target_verse: str,
    raw_reward: float,
    syn_reward: float,
    translated_obs: Optional[Dict[str, Any]],
    translated_next_obs: Optional[Dict[str, Any]],
    translated_action: Optional[int],
    syn_info: Dict[str, Any],
) -> float:
    """Compute a [0,1] confidence score for a translated transition.

    Higher scores indicate the translated transition is more likely to be
    useful for the target agent's learning. Transitions with low scores
    are noisy projections that would poison the Q-table.

    Components:
    - reward_consistency: Are the source and synthetic rewards directionally aligned?
    - action_coherence: Does the translated action make progress given the observation?
    - observation_validity: Is the translated observation physically reasonable?
    """
    src = _norm_verse_name(source_verse)
    tgt = _norm_verse_name(target_verse)

    # Same-family transfers get full confidence — the mapping is semantic.
    if _is_strategy_verse(src) and _is_strategy_verse(tgt):
        return 1.0

    score = 0.0
    n_components = 0

    # --- Component 1: Reward consistency (0-1) ---
    # Do source and synthetic rewards agree on direction?
    n_components += 1
    if abs(raw_reward) < 0.01 and abs(syn_reward) < 0.01:
        score += 0.5  # Both near zero — neutral
    elif (raw_reward > 0 and syn_reward > 0) or (raw_reward < 0 and syn_reward < 0):
        score += 1.0  # Same sign — consistent
    elif abs(raw_reward) < 0.5 or abs(syn_reward) < 0.5:
        score += 0.3  # Mild disagreement
    else:
        score += 0.0  # Strong disagreement — probably noise

    # --- Component 2: Action-reward coherence (0-1) ---
    # Does the action make sense given the reward signal?
    n_components += 1
    if tgt in ("warehouse_world", "labyrinth_world", "escape_world"):
        obs = translated_obs if isinstance(translated_obs, dict) else {}
        nobs = translated_next_obs if isinstance(translated_next_obs, dict) else {}
        a = _safe_int(translated_action, -1)

        # Check if action makes progress toward goal
        x = _safe_int(obs.get("x", 0))
        y = _safe_int(obs.get("y", 0))
        gx = _safe_int(obs.get("goal_x", 7))
        gy = _safe_int(obs.get("goal_y", 7))
        nx = _safe_int(nobs.get("x", x))
        ny = _safe_int(nobs.get("y", y))

        old_dist = abs(x - gx) + abs(y - gy)
        new_dist = abs(nx - gx) + abs(ny - gy)

        if syn_reward > 0.1 and new_dist < old_dist:
            score += 1.0  # Positive reward + progress — coherent
        elif syn_reward < -0.3 and new_dist >= old_dist:
            score += 0.8  # Negative reward + no progress — coherent penalty
        elif syn_reward > 0.1 and new_dist >= old_dist:
            score += 0.2  # Positive reward but no progress — suspicious
        elif syn_reward < -0.3 and new_dist < old_dist:
            score += 0.2  # Negative reward but progress — contradictory
        else:
            score += 0.5  # Neutral
    else:
        score += 0.5  # Can't evaluate coherence for this target type

    # --- Component 3: Observation validity (0-1) ---
    # Is the translated observation within reasonable bounds?
    n_components += 1
    if tgt == "warehouse_world":
        obs = translated_obs if isinstance(translated_obs, dict) else {}
        x = _safe_int(obs.get("x", -1))
        y = _safe_int(obs.get("y", -1))
        battery = _safe_int(obs.get("battery", 0))
        # Position in valid range, battery positive
        pos_valid = (0 <= x <= 7 and 0 <= y <= 7)
        battery_valid = (1 <= battery <= 100)
        if pos_valid and battery_valid:
            score += 1.0
        elif pos_valid:
            score += 0.6
        else:
            score += 0.1
    elif tgt == "labyrinth_world":
        obs = translated_obs if isinstance(translated_obs, dict) else {}
        battery = _safe_int(obs.get("battery", 0))
        score += 0.8 if battery > 0 else 0.2
    else:
        score += 0.6  # Default — can't validate

    # --- Component 4: Hazard penalty (reduces confidence) ---
    # Reached a hazard state — this is valid negative data but only if coherent
    is_hazard = any(bool(syn_info.get(k, False)) for k in (
        "hit_wall", "hit_obstacle", "battery_death", "battery_depleted",
        "fell_pit", "hit_laser",
    ))
    if is_hazard:
        # Hazard transitions are useful IF the reward is negative (teaching avoidance).
        # But if blended reward is positive (from source success), it's contradictory.
        if syn_reward < -0.2:
            pass  # Good — teaches avoidance
        else:
            n_components += 1
            score += 0.0  # Hazard with positive reward — very bad signal

    confidence = float(score) / float(max(1, n_components))
    return max(0.0, min(1.0, confidence))


def translate_dna(
    *,
    source_dna_path: str,
    target_verse_name: str,
    output_path: Optional[str] = None,
    source_verse_name: Optional[str] = None,
    synthetic_reward_blend: float = 0.75,
    synthetic_done_union: bool = True,
    target_label_cfg: Optional[Dict[str, Any]] = None,
    learned_bridge_enabled: bool = False,
    learned_bridge_model_path: Optional[str] = None,
    learned_bridge_score_weight: float = 0.35,
    behavioral_bridge_enabled: bool = False,
    behavioral_bridge_score_weight: float = 0.35,
    behavioral_max_prototype_rows: int = 4096,
    confidence_threshold: float = 0.35,
) -> BridgeStats:
    if not os.path.isfile(source_dna_path):
        raise FileNotFoundError(f"DNA file not found: {source_dna_path}")

    target = _norm_verse_name(target_verse_name)
    if not target:
        raise ValueError("target_verse_name is required")

    if output_path is None:
        base_dir = os.path.dirname(source_dna_path) or "."
        output_path = os.path.join(base_dir, f"synthetic_expert_{target}.jsonl")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    inferred_source = _norm_verse_name(source_verse_name)
    blend = max(0.0, min(1.0, _safe_float(synthetic_reward_blend, 0.35)))
    learned_enabled = _safe_bool(learned_bridge_enabled, False)
    learned_weight = max(0.0, min(1.0, _safe_float(learned_bridge_score_weight, 0.35)))
    behavioral_enabled = _safe_bool(behavioral_bridge_enabled, False)
    behavioral_weight = max(0.0, min(1.0, _safe_float(behavioral_bridge_score_weight, 0.35)))
    behavioral_proto_max = max(1, _safe_int(behavioral_max_prototype_rows, 4096))
    conf_thresh = max(0.0, min(1.0, float(confidence_threshold)))

    input_rows = 0
    kept_rows = 0
    learned_scored_rows = 0
    behavioral_scored_rows = 0
    learned_model_resolved: Optional[str] = None
    behavioral_prototype, behavioral_prototype_rows = (
        _build_behavioral_prototype(source_dna_path=source_dna_path, max_rows=behavioral_proto_max)
        if behavioral_enabled
        else (None, 0)
    )

    with open(output_path, "w", encoding="utf-8") as out:
        for row in _iter_jsonl(source_dna_path):
            input_rows += 1
            obs = row.get("obs")
            action = row.get("action")
            if inferred_source == "":
                inferred_source = _norm_verse_name(row.get("verse_name"))
            if inferred_source == "":
                guessed = infer_verse_from_obs(obs)
                inferred_source = _norm_verse_name(guessed)
            if inferred_source == "":
                continue

            translated = translate_transition(
                obs=obs,
                action=action,
                source_verse_name=inferred_source,
                target_verse_name=target,
                next_obs=row.get("next_obs"),
                learned_bridge_enabled=learned_enabled,
                learned_bridge_model_path=learned_bridge_model_path,
            )
            if translated is None:
                continue

            translated_obs = translated.get("obs")
            translated_next_obs = translated.get("next_obs")
            translated_action = translated.get("action")
            learned_conf = translated.get("learned_bridge_confidence")
            learned_sim = translated.get("learned_bridge_similarity")
            learned_model_resolved = (
                str(translated.get("learned_bridge_model_path"))
                if translated.get("learned_bridge_model_path") is not None
                else learned_model_resolved
            )
            raw_reward = _safe_float(row.get("reward", 0.0), 0.0)
            syn_reward, syn_done, syn_info = _synthesize_target_labels(
                target_verse_name=target,
                obs=translated_obs,
                next_obs=translated_next_obs,
                action=translated_action,
                cfg=target_label_cfg,
            )

            # Confidence Gating
            transfer_confidence = _compute_transfer_confidence(
                source_verse=inferred_source,
                target_verse=target,
                raw_reward=raw_reward,
                syn_reward=syn_reward,
                translated_obs=translated_obs,
                translated_next_obs=translated_next_obs,
                translated_action=_safe_int(translated_action, -1) if translated_action is not None else None,
                syn_info=syn_info,
            )
            
            if transfer_confidence < conf_thresh:
                # Drop noisy transition
                continue

            reward = float(raw_reward)
            done = bool(row.get("done", False))
            if target in ("warehouse_world", "labyrinth_world", "harvest_world", "swamp_world"):
                reward = float((1.0 - blend) * raw_reward + blend * syn_reward)
                done = bool((done or syn_done) if bool(synthetic_done_union) else done)
                # Note: Artificial goal injection removed in previous fix.

            source_advantage = _safe_float(row.get("advantage", 0.0))
            base_transfer_score = max(0.0, 1.0 + source_advantage)
            transfer_score = float(base_transfer_score)
            learned_conf_f = _safe_float(learned_conf, -1.0)
            if learned_enabled and 0.0 <= learned_conf_f <= 1.0:
                learned_scored_rows += 1
                learned_component = max(0.0, min(2.0, 2.0 * learned_conf_f))
                transfer_score = float(
                    (1.0 - learned_weight) * float(base_transfer_score)
                    + learned_weight * float(learned_component)
                )
            behavioral_sim, behavioral_conf = _behavioral_bridge_metrics(
                obs=translated_obs,
                next_obs=translated_next_obs,
                action=translated_action,
                reward=float(reward),
                done=bool(done),
                prototype=behavioral_prototype,
            )
            behavioral_component: Optional[float] = None
            if behavioral_enabled and behavioral_conf is not None:
                behavioral_scored_rows += 1
                behavioral_component = max(0.0, min(2.0, 2.0 * float(behavioral_conf)))
                transfer_score = float(
                    (1.0 - behavioral_weight) * float(transfer_score)
                    + behavioral_weight * float(behavioral_component)
                )
            bridge_name = "semantic_projection_v2" if learned_enabled else "semantic_projection_v1"
            if behavioral_enabled:
                bridge_name = "semantic_behavioral_hybrid_v1"

            out_row: Dict[str, Any] = {
                "episode_id": row.get("episode_id"),
                "step_idx": _safe_int(row.get("step_idx", kept_rows)),
                "obs": translated_obs,
                "action": translated_action,
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(row.get("truncated", False)),
                "next_obs": translated_next_obs,
                "info": syn_info,
                "source_verse_name": inferred_source,
                "target_verse_name": target,
                "synthetic": True,
                "bridge": bridge_name,
                "bridge_reason": bridge_reason(inferred_source, target),
                "source_advantage": source_advantage,
                "transfer_score": transfer_score,
                "transfer_confidence": float(transfer_confidence),
            }
            if learned_enabled:
                out_row["learned_bridge_enabled"] = True
                out_row["learned_bridge_confidence"] = (
                    float(learned_conf_f) if 0.0 <= learned_conf_f <= 1.0 else None
                )
                out_row["learned_bridge_similarity"] = (
                    float(_safe_float(learned_sim)) if learned_sim is not None else None
                )
                out_row["learned_bridge_score_weight"] = float(learned_weight)
                out_row["base_transfer_score"] = float(base_transfer_score)
                if learned_model_resolved:
                    out_row["learned_bridge_model_path"] = learned_model_resolved
            if behavioral_enabled:
                out_row["behavioral_bridge_enabled"] = True
                out_row["behavioral_bridge_confidence"] = (
                    float(behavioral_conf) if behavioral_conf is not None else None
                )
                out_row["behavioral_bridge_similarity"] = (
                    float(behavioral_sim) if behavioral_sim is not None else None
                )
                out_row["behavioral_bridge_score_weight"] = float(behavioral_weight)
                out_row["behavioral_prototype_rows"] = int(behavioral_prototype_rows)
                if behavioral_component is not None:
                    out_row["behavioral_transfer_component"] = float(behavioral_component)
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept_rows += 1

    return BridgeStats(
        source_path=source_dna_path,
        output_path=output_path,
        source_verse_name=inferred_source or "",
        target_verse_name=target,
        input_rows=input_rows,
        translated_rows=kept_rows,
        dropped_rows=max(0, input_rows - kept_rows),
        learned_bridge_enabled=bool(learned_enabled),
        learned_bridge_model_path=learned_model_resolved,
        learned_scored_rows=int(learned_scored_rows),
        behavioral_bridge_enabled=bool(behavioral_enabled),
        behavioral_scored_rows=int(behavioral_scored_rows),
        behavioral_prototype_rows=int(behavioral_prototype_rows),
    )


def _synthesize_escape_labels(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Reward shaping for escape_world transfer data."""
    c = cfg if isinstance(cfg, dict) else {}
    o = obs if isinstance(obs, dict) else {}
    n = next_obs if isinstance(next_obs, dict) else {}

    step_penalty = max(0.0, _safe_float(c.get("step_penalty", 0.05), 0.05))
    exit_reward = max(0.0, _safe_float(c.get("exit_reward", 10.0), 10.0))
    progress_bonus = max(0.0, _safe_float(c.get("progress_bonus", 0.15), 0.15))
    spotted_penalty = max(0.0, _safe_float(c.get("spotted_penalty", 5.0), 5.0))

    reward = -float(step_penalty)
    done = False
    info: Dict[str, Any] = {"reached_goal": False}

    old_dist = _safe_int(o.get("exit_dist", 20), 20)
    new_dist = _safe_int(n.get("exit_dist", old_dist), old_dist)
    
    # Progress reward
    if new_dist < old_dist:
        reward += float(progress_bonus)
    elif new_dist > old_dist:
        reward -= float(progress_bonus) * 0.5

    # Check for spotted / reset (distance jumps back to start ~20ish)
    # Heuristic: if distance jumped UP significantly, we likely died/reset.
    if new_dist > old_dist + 5:
        reward -= float(spotted_penalty)
        info["spotted"] = True
    
    guards = _safe_int(n.get("guards_in_vision", 0), 0)
    if guards > 0:
        reward -= 0.1  # Anxiety penalty

    if new_dist == 0:
        info["reached_goal"] = True
        done = True
        reward += float(exit_reward)

    return float(reward), bool(done), info


def _synthesize_factory_labels(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Reward shaping for factory_world transfer data."""
    c = cfg if isinstance(cfg, dict) else {}
    o = obs if isinstance(obs, dict) else {}
    n = next_obs if isinstance(next_obs, dict) else {}

    step_penalty = max(0.0, _safe_float(c.get("step_penalty", 0.02), 0.02))
    completion_reward = max(0.0, _safe_float(c.get("completion_reward", 2.0), 2.0))
    
    reward = -float(step_penalty)
    done = False
    info: Dict[str, Any] = {}

    comp = _safe_int(o.get("completed", 0), 0)
    n_comp = _safe_int(n.get("completed", comp), comp)
    
    delta = n_comp - comp
    if delta > 0:
        reward += float(completion_reward) * delta
        info["items_completed"] = delta

    # If we mapped a strategy game here, 'completed' might be proxied by score/material
    # so high delta is good.
    
    if n_comp >= 50: # Arbitrary "job done" for synthetic episodes
        done = True

    return float(reward), bool(done), info


def _synthesize_bridge_labels(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Reward shaping for bridge_world transfer data."""
    c = cfg if isinstance(cfg, dict) else {}
    o = obs if isinstance(obs, dict) else {}
    n = next_obs if isinstance(next_obs, dict) else {}

    step_penalty = max(0.0, _safe_float(c.get("step_penalty", 0.1), 0.1))
    cross_reward = max(0.0, _safe_float(c.get("cross_reward", 10.0), 10.0))
    build_reward = max(0.0, _safe_float(c.get("build_reward", 0.3), 0.3))

    reward = -float(step_penalty)
    done = False
    info: Dict[str, Any] = {}

    placed = _safe_int(o.get("segments_placed", 0), 0)
    n_placed = _safe_int(n.get("segments_placed", placed), placed)
    
    if n_placed > placed:
        reward += float(build_reward) * (n_placed - placed)
        info["built_segment"] = True

    # Bridge complete + cross action (approximate by completion transition)
    complete = _safe_int(n.get("bridge_complete", 0), 0)
    if complete == 1 and _safe_int(o.get("bridge_complete", 0), 0) == 0:
         # Just finished building, big step
         reward += 1.0

    # If we magically crossed (e.g. from line world goal)
    # synthetic done signal
    if bool(n.get("crossed", False)) or (complete == 1 and n_placed >= 8):
         # If crossed flag exists or we really filled the bridge
         pass

    # For synthetic data, we often just want to reward "more bridge"
    # and map line-world goal to "crossed"
    
    return float(reward), bool(done), info


def _synthesize_trade_labels(
    *,
    obs: JSONValue,
    next_obs: Optional[JSONValue],
    action: JSONValue,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Reward shaping for trade_world transfer data."""
    c = cfg if isinstance(cfg, dict) else {}
    o = obs if isinstance(obs, dict) else {}
    n = next_obs if isinstance(next_obs, dict) else {}

    step_penalty = max(0.0, _safe_float(c.get("step_penalty", 0.01), 0.01))
    profit_scale = max(0.0, _safe_float(c.get("profit_scale", 1.0), 1.0))

    reward = -float(step_penalty)
    done = False
    info: Dict[str, Any] = {}

    cash = _safe_float(o.get("cash", 0.0))
    n_cash = _safe_float(n.get("cash", cash))
    
    # Reward cash increase (realized profit)
    delta = n_cash - cash
    if delta > 0:
        reward += delta * float(profit_scale)
        info["profit"] = delta
    
    # Reward portfolio growth slightly to encourage holding good assets?
    # Maybe better to stick to cash for robustness.
    
    if n_cash >= 200.0:  # Arbitrary wealth goal for synthetic data
        done = True

    return float(reward), bool(done), info
