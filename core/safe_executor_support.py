"""
Support helpers and configuration for SafeExecutor.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue


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


def _parse_label_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    if not s:
        return None
    if s in {"1", "true", "t", "yes", "y", "on", "danger", "unsafe", "fail", "failure"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", "safe", "success"}:
        return False
    return None


def _validate_allowed_keys(*, payload: Dict[str, Any], allowed: Set[str], label: str) -> None:
    unknown = sorted(str(k) for k in payload.keys() if str(k) not in allowed)
    if not unknown:
        return
    allowed_sorted = sorted(str(k) for k in allowed)
    raise ValueError(
        f"{label} has unknown keys: {unknown}. "
        f"Allowed keys: {allowed_sorted}"
    )


def _default_mcts_overrides(verse_name: str) -> Dict[str, Any]:
    v = str(verse_name or "").strip().lower()
    if v == "chess_world":
        return {"mcts_num_simulations": 128, "mcts_max_depth": 5, "mcts_loss_threshold": -0.92}
    if v == "go_world":
        return {"mcts_num_simulations": 160, "mcts_max_depth": 6, "mcts_loss_threshold": -0.90}
    if v == "uno_world":
        return {"mcts_num_simulations": 96, "mcts_max_depth": 4, "mcts_loss_threshold": -0.88}
    if v == "warehouse_world":
        return {"mcts_num_simulations": 80, "mcts_max_depth": 8, "mcts_loss_threshold": -0.85}
    if v == "labyrinth_world":
        return {"mcts_num_simulations": 100, "mcts_max_depth": 10, "mcts_loss_threshold": -0.80}
    return {}


def _obs_key(obs: JSONValue) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(obs)


def _is_safety_violation(info: Dict[str, Any]) -> bool:
    safety_true_keys = (
        "wrong_park",
        "collision",
        "crash",
        "boundary_violation",
        "unsafe",
        "failure",
        "fell_cliff",
        "fell_pit",
        "hit_laser",
        "battery_depleted",
        "hit_wall",
        "hit_obstacle",
        "battery_death",
    )
    for k in safety_true_keys:
        if info.get(k) is True:
            return True
    return False


def _danger_label_from_info(info: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    keys = ("danger_label", "dangerous_outcome_label", "ground_truth_danger", "unsafe_label")
    for k in keys:
        v = _parse_label_bool(info.get(k))
        if v is not None:
            return v, f"info.{k}"
    se = info.get("safe_executor")
    se = se if isinstance(se, dict) else {}
    for k in keys:
        v = _parse_label_bool(se.get(k))
        if v is not None:
            return v, f"info.safe_executor.{k}"
    return None, ""


def _is_nan_action(action: Any) -> bool:
    try:
        v = float(action)
    except Exception:
        return False
    return v != v


def _infer_failure_mode(
    *,
    info: Dict[str, Any],
    dangerous_outcome: bool,
    severe_penalty: bool,
    safety_violation: bool,
    low_confidence: bool,
    high_danger: bool,
    planner_failed: bool,
    policy_nan: bool,
    done: bool,
    truncated: bool,
) -> str:
    if bool(info.get("reached_goal", False)):
        return "success"
    if policy_nan:
        return "policy_nan"
    if bool(info.get("fell_cliff", False)):
        return "fell_cliff"
    if bool(info.get("fell_pit", False)):
        return "fell_pit"
    if bool(info.get("hit_laser", False)):
        return "hit_laser"
    if bool(info.get("battery_depleted", False)):
        return "battery_depleted"
    if bool(info.get("hit_wall", False)):
        return "hit_wall"
    if bool(info.get("hit_obstacle", False)):
        return "hit_obstacle"
    if bool(info.get("battery_death", False)):
        return "battery_death"
    if planner_failed:
        return "planner_timeout_or_no_plan"
    if safety_violation:
        return "safety_violation"
    if severe_penalty and low_confidence:
        return "low_confidence_penalty"
    if severe_penalty and high_danger:
        return "high_danger_penalty"
    if dangerous_outcome:
        return "dangerous_outcome"
    if bool(done or truncated):
        if "warning" in info or "error" in info:
            return "env_edge_case"
        return "timeout_or_stuck"
    return "unknown"


def _project_vector(vec: List[float], *, dim: int) -> List[float]:
    d = max(4, int(dim))
    out = [0.0 for _ in range(d)]
    for i, v in enumerate(vec):
        idx = int((i * 2654435761) % d)
        sign = -1.0 if (((i * 11400714819323198485) >> 3) & 1) else 1.0
        out[idx] += float(v) * float(sign)
    norm = math.sqrt(sum(x * x for x in out))
    if norm > 1e-12:
        out = [float(x / norm) for x in out]
    return out


@dataclass
class SafeExecutorConfig:
    enabled: bool = True
    danger_threshold: float = 0.90
    min_action_confidence: float = 0.08
    adaptive_veto_enabled: bool = False
    adaptive_veto_relaxation: float = 0.35
    adaptive_veto_warmup_steps: int = 12
    adaptive_veto_failure_guard: float = 0.20
    adaptive_veto_schedule_enabled: bool = False
    adaptive_veto_relaxation_start: float = 0.10
    adaptive_veto_relaxation_end: float = 0.35
    adaptive_veto_schedule_steps: int = 200
    adaptive_veto_schedule_power: float = 1.20
    severe_reward_threshold: float = -50.0
    confidence_model_path: str = ""
    confidence_model_weight: float = 0.60
    confidence_model_obs_dim: int = 64
    competence_window: int = 5
    min_competence_rate: float = 0.90
    fallback_horizon_steps: int = 8
    checkpoint_interval: int = 5
    max_rewinds_per_episode: int = 8
    block_repeated_fail_action: bool = True
    prefer_fallback_on_veto: bool = True
    planner_enabled: bool = False
    planner_confidence_threshold: float = 0.12
    planner_horizon: int = 5
    planner_max_expansions: int = 8000
    planner_trigger_on_high_danger: bool = True
    planner_trigger_on_block: bool = True
    planner_verse_allowlist: List[str] = field(default_factory=list)
    planning_regret_adaptation: float = 0.30
    planning_budget_per_episode: int = 6
    planning_budget_per_minute: int = 120
    mcts_enabled: bool = False
    mcts_num_simulations: int = 96
    mcts_max_depth: int = 4
    mcts_c_puct: float = 1.4
    mcts_discount: float = 0.99
    mcts_loss_threshold: float = -0.95
    mcts_min_visits: int = 8
    mcts_trigger_on_low_confidence: bool = True
    mcts_trigger_on_high_danger: bool = True
    mcts_trigger_on_block: bool = True
    mcts_meta_model_path: str = ""
    mcts_meta_history_len: int = 6
    mcts_value_confidence_threshold: float = 0.0
    mcts_verse_overrides: Dict[str, Dict[str, JSONValue]] = field(default_factory=dict)
    shield_enabled: bool = False
    shield_model_path: str = ""
    shield_threshold: float = 0.50
    danger_map_path: str = ""
    danger_map_similarity_threshold: float = 0.90
    failure_signature_path: str = ""
    failure_signature_similarity_threshold: float = 0.92
    failure_signature_embedding_dim: int = 64
    force_mcts_on_failure_signature: bool = True

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "SafeExecutorConfig":
        c = cfg if isinstance(cfg, dict) else {}
        strict_cfg = bool(c.get("strict_config_validation", True))

        planning_cfg_raw = c.get("planning")
        if planning_cfg_raw is None:
            planning_cfg: Dict[str, Any] = {}
        elif isinstance(planning_cfg_raw, dict):
            planning_cfg = dict(planning_cfg_raw)
        else:
            raise ValueError("safe_executor.planning must be a dict")

        mcts_cfg_raw = c.get("mcts")
        if mcts_cfg_raw is None:
            mcts_cfg: Dict[str, Any] = {}
        elif isinstance(mcts_cfg_raw, dict):
            mcts_cfg = dict(mcts_cfg_raw)
        else:
            raise ValueError("safe_executor.mcts must be a dict")

        if strict_cfg:
            allowed_root = {
                "enabled", "danger_threshold", "min_action_confidence",
                "adaptive_veto_enabled", "adaptive_veto_relaxation", "adaptive_veto_warmup_steps",
                "adaptive_veto_failure_guard", "adaptive_veto_schedule_enabled",
                "adaptive_veto_relaxation_start", "adaptive_veto_relaxation_end",
                "adaptive_veto_schedule_steps", "adaptive_veto_schedule_power",
                "severe_reward_threshold", "confidence_model_path", "confidence_model_weight",
                "confidence_model_obs_dim", "competence_window", "min_competence_rate",
                "fallback_horizon_steps", "checkpoint_interval", "max_rewinds_per_episode",
                "block_repeated_fail_action", "prefer_fallback_on_veto", "planner_enabled",
                "planner_confidence_threshold", "planner_horizon", "planner_max_expansions",
                "planner_trigger_on_high_danger", "planner_trigger_on_block",
                "planner_verse_allowlist", "planning_regret_adaptation",
                "planning_budget_per_episode", "planning_budget_per_minute",
                "planning_base_threshold", "planning", "mcts_enabled", "mcts_num_simulations",
                "mcts_max_depth", "mcts_c_puct", "mcts_discount", "mcts_loss_threshold",
                "mcts_min_visits", "mcts_trigger_on_low_confidence",
                "mcts_trigger_on_high_danger", "mcts_trigger_on_block",
                "mcts_meta_model_path", "mcts_meta_history_len",
                "mcts_value_confidence_threshold", "mcts_verse_overrides", "mcts",
                "shield_enabled", "shield_model_path", "shield_threshold",
                "danger_map_path", "danger_map_similarity_threshold", "failure_signature_path",
                "failure_signature_similarity_threshold", "failure_signature_embedding_dim",
                "force_mcts_on_failure_signature", "strict_config_validation",
            }
            _validate_allowed_keys(payload=c, allowed=allowed_root, label="safe_executor config")

            allowed_planning = {"enabled", "base_threshold", "regret_adaptation", "budget_per_episode", "budget_per_minute"}
            _validate_allowed_keys(payload=planning_cfg, allowed=allowed_planning, label="safe_executor.planning")

            allowed_mcts = {
                "enabled", "num_simulations", "max_depth", "c_puct", "discount", "loss_threshold",
                "min_visits", "trigger_on_low_confidence", "trigger_on_high_danger",
                "trigger_on_block", "meta_model_path", "meta_history_len",
                "value_confidence_threshold", "verse_overrides",
            }
            _validate_allowed_keys(payload=mcts_cfg, allowed=allowed_mcts, label="safe_executor.mcts")

            verse_overrides_raw = mcts_cfg.get("verse_overrides", c.get("mcts_verse_overrides", {}))
            if verse_overrides_raw is not None and not isinstance(verse_overrides_raw, dict):
                raise ValueError("safe_executor.mcts.verse_overrides must be a dict")
            allowed_override = {
                "mcts_enabled", "mcts_num_simulations", "mcts_max_depth", "mcts_c_puct",
                "mcts_discount", "mcts_loss_threshold", "mcts_min_visits",
                "mcts_value_confidence_threshold",
            }
            if isinstance(verse_overrides_raw, dict):
                for verse_name, entry in verse_overrides_raw.items():
                    if not isinstance(entry, dict):
                        raise ValueError(f"safe_executor.mcts.verse_overrides['{verse_name}'] must be a dict")
                    _validate_allowed_keys(
                        payload=entry,
                        allowed=allowed_override,
                        label=f"safe_executor.mcts.verse_overrides['{verse_name}']",
                    )

        raw_verse_overrides = mcts_cfg.get("verse_overrides", c.get("mcts_verse_overrides", {}))
        verse_overrides: Dict[str, Dict[str, JSONValue]] = {}
        if isinstance(raw_verse_overrides, dict):
            for k, v in raw_verse_overrides.items():
                if not isinstance(v, dict):
                    continue
                verse_overrides[str(k).strip().lower()] = dict(v)

        planning_enabled = planning_cfg.get("enabled", c.get("planner_enabled", False))
        planning_base_threshold = planning_cfg.get(
            "base_threshold", c.get("planner_confidence_threshold", c.get("planning_base_threshold", 0.12))
        )
        planning_regret_adaptation = planning_cfg.get("regret_adaptation", c.get("planning_regret_adaptation", 0.30))
        planning_budget_per_episode = planning_cfg.get("budget_per_episode", c.get("planning_budget_per_episode", 6))
        planning_budget_per_minute = planning_cfg.get("budget_per_minute", c.get("planning_budget_per_minute", 120))

        relax_end = max(0.0, min(1.0, _safe_float(c.get("adaptive_veto_relaxation_end", c.get("adaptive_veto_relaxation", 0.35)), 0.35)))
        relax_start = max(0.0, min(1.0, _safe_float(c.get("adaptive_veto_relaxation_start", min(0.10, relax_end)), min(0.10, relax_end))))

        return SafeExecutorConfig(
            enabled=bool(c.get("enabled", True)),
            danger_threshold=max(0.0, min(1.0, _safe_float(c.get("danger_threshold", 0.90), 0.90))),
            min_action_confidence=max(0.0, min(1.0, _safe_float(c.get("min_action_confidence", 0.08), 0.08))),
            adaptive_veto_enabled=bool(c.get("adaptive_veto_enabled", False)),
            adaptive_veto_relaxation=max(0.0, min(1.0, _safe_float(c.get("adaptive_veto_relaxation", 0.35), 0.35))),
            adaptive_veto_warmup_steps=max(0, _safe_int(c.get("adaptive_veto_warmup_steps", 12), 12)),
            adaptive_veto_failure_guard=max(1e-6, _safe_float(c.get("adaptive_veto_failure_guard", 0.20), 0.20)),
            adaptive_veto_schedule_enabled=bool(c.get("adaptive_veto_schedule_enabled", False)),
            adaptive_veto_relaxation_start=float(relax_start),
            adaptive_veto_relaxation_end=float(relax_end),
            adaptive_veto_schedule_steps=max(1, _safe_int(c.get("adaptive_veto_schedule_steps", 200), 200)),
            adaptive_veto_schedule_power=max(0.10, _safe_float(c.get("adaptive_veto_schedule_power", 1.20), 1.20)),
            severe_reward_threshold=_safe_float(c.get("severe_reward_threshold", -50.0), -50.0),
            confidence_model_path=str(c.get("confidence_model_path", "") or "").strip(),
            confidence_model_weight=max(0.0, min(1.0, _safe_float(c.get("confidence_model_weight", 0.60), 0.60))),
            confidence_model_obs_dim=max(4, _safe_int(c.get("confidence_model_obs_dim", 64), 64)),
            competence_window=max(1, _safe_int(c.get("competence_window", 5), 5)),
            min_competence_rate=max(0.0, min(1.0, _safe_float(c.get("min_competence_rate", 0.90), 0.90))),
            fallback_horizon_steps=max(0, _safe_int(c.get("fallback_horizon_steps", 8), 8)),
            checkpoint_interval=max(1, _safe_int(c.get("checkpoint_interval", 5), 5)),
            max_rewinds_per_episode=max(0, _safe_int(c.get("max_rewinds_per_episode", 8), 8)),
            block_repeated_fail_action=bool(c.get("block_repeated_fail_action", True)),
            prefer_fallback_on_veto=bool(c.get("prefer_fallback_on_veto", True)),
            planner_enabled=bool(planning_enabled),
            planner_confidence_threshold=max(0.0, min(1.0, _safe_float(planning_base_threshold, 0.12))),
            planner_horizon=max(1, _safe_int(c.get("planner_horizon", 5), 5)),
            planner_max_expansions=max(100, _safe_int(c.get("planner_max_expansions", 8000), 8000)),
            planner_trigger_on_high_danger=bool(c.get("planner_trigger_on_high_danger", True)),
            planner_trigger_on_block=bool(c.get("planner_trigger_on_block", True)),
            planner_verse_allowlist=(
                [str(v).strip().lower() for v in c.get("planner_verse_allowlist", []) if str(v).strip()]
                if isinstance(c.get("planner_verse_allowlist"), list)
                else []
            ),
            planning_regret_adaptation=max(0.0, min(1.0, _safe_float(planning_regret_adaptation, 0.30))),
            planning_budget_per_episode=max(1, _safe_int(planning_budget_per_episode, 6)),
            planning_budget_per_minute=max(1, _safe_int(planning_budget_per_minute, 120)),
            mcts_enabled=bool(mcts_cfg.get("enabled", c.get("mcts_enabled", False))),
            mcts_num_simulations=max(8, _safe_int(mcts_cfg.get("num_simulations", c.get("mcts_num_simulations", 96)), 96)),
            mcts_max_depth=max(2, _safe_int(mcts_cfg.get("max_depth", c.get("mcts_max_depth", 4)), 4)),
            mcts_c_puct=max(0.1, _safe_float(mcts_cfg.get("c_puct", c.get("mcts_c_puct", 1.4)), 1.4)),
            mcts_discount=max(0.0, min(1.0, _safe_float(mcts_cfg.get("discount", c.get("mcts_discount", 0.99)), 0.99))),
            mcts_loss_threshold=max(-1.0, min(0.0, _safe_float(mcts_cfg.get("loss_threshold", c.get("mcts_loss_threshold", -0.95)), -0.95))),
            mcts_min_visits=max(1, _safe_int(mcts_cfg.get("min_visits", c.get("mcts_min_visits", 8)), 8)),
            mcts_trigger_on_low_confidence=bool(mcts_cfg.get("trigger_on_low_confidence", c.get("mcts_trigger_on_low_confidence", True))),
            mcts_trigger_on_high_danger=bool(mcts_cfg.get("trigger_on_high_danger", c.get("mcts_trigger_on_high_danger", True))),
            mcts_trigger_on_block=bool(mcts_cfg.get("trigger_on_block", c.get("mcts_trigger_on_block", True))),
            mcts_meta_model_path=str(mcts_cfg.get("meta_model_path", c.get("mcts_meta_model_path", "")) or ""),
            mcts_meta_history_len=max(1, _safe_int(mcts_cfg.get("meta_history_len", c.get("mcts_meta_history_len", 6)), 6)),
            mcts_value_confidence_threshold=max(0.0, min(1.0, _safe_float(mcts_cfg.get("value_confidence_threshold", c.get("mcts_value_confidence_threshold", 0.0)), 0.0))),
            mcts_verse_overrides=verse_overrides,
            shield_enabled=bool(c.get("shield_enabled", False)),
            shield_model_path=str(c.get("shield_model_path", "")),
            shield_threshold=_safe_float(c.get("shield_threshold", 0.50), 0.50),
            danger_map_path=str(c.get("danger_map_path", "")),
            danger_map_similarity_threshold=max(0.0, min(1.0, _safe_float(c.get("danger_map_similarity_threshold", 0.90), 0.90))),
            failure_signature_path=str(c.get("failure_signature_path", "")),
            failure_signature_similarity_threshold=max(0.0, min(1.0, _safe_float(c.get("failure_signature_similarity_threshold", 0.92), 0.92))),
            failure_signature_embedding_dim=max(8, _safe_int(c.get("failure_signature_embedding_dim", 64), 64)),
            force_mcts_on_failure_signature=bool(c.get("force_mcts_on_failure_signature", True)),
        )
