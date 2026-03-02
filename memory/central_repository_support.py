"""
Support types and helpers for the central memory repository.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from core.types import JSONValue
from memory.task_taxonomy import memory_family_for_verse, memory_type_for_verse, tags_for_verse


@dataclass
class CentralMemoryConfig:
    root_dir: str = "central_memory"
    memories_filename: str = "memories.jsonl"
    ltm_memories_filename: str = "ltm_memories.jsonl"
    stm_memories_filename: str = "stm_memories.jsonl"
    dedupe_index_filename: str = "dedupe_index.json"
    dedupe_db_filename: str = "dedupe_index.sqlite"
    tier_policy_filename: str = "tier_policy.json"
    stm_decay_lambda: float = 1e-8


@dataclass
class IngestStats:
    run_dir: str
    input_events: int
    selected_events: int
    added_events: int
    skipped_duplicates: int


@dataclass
class ScenarioMatch:
    score: float
    run_id: str
    episode_id: str
    step_idx: int
    t_ms: int
    verse_name: str
    action: JSONValue
    reward: float
    obs: JSONValue
    source_greedy_action: Optional[int] = None
    source_action_matches_greedy: Optional[bool] = None
    recency_weight: float = 1.0
    trajectory: Optional[List[Dict[str, Any]]] = None


@dataclass
class SanitizeStats:
    input_lines: int
    kept_lines: int
    dropped_lines: int


@dataclass
class BackfillStats:
    rows_scanned: int
    rows_written: int
    malformed_rows: int
    backfilled_memory_tier: int
    recomputed_memory_tier: int
    support_guard_demotions: int
    backfilled_memory_family: int
    backfilled_memory_type: int
    ltm_rows: int
    stm_rows: int
    backfilled_obs_vector_u: int


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


def _json_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(value)


def _dedupe_key(event: Dict[str, Any]) -> str:
    reward = round(_safe_float(event.get("reward", 0.0)), 6)
    payload = (
        str(event.get("verse_name", "")),
        _json_key(event.get("obs")),
        _json_key(event.get("action")),
        _json_key(event.get("next_obs")),
        str(bool(event.get("done", False))),
        str(bool(event.get("truncated", False))),
        f"{reward:.6f}",
    )
    raw = "||".join(payload).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _memories_path(cfg: CentralMemoryConfig) -> str:
    return os.path.join(cfg.root_dir, cfg.memories_filename)


def _ltm_memories_path(cfg: CentralMemoryConfig) -> str:
    return os.path.join(cfg.root_dir, cfg.ltm_memories_filename)


def _stm_memories_path(cfg: CentralMemoryConfig) -> str:
    return os.path.join(cfg.root_dir, cfg.stm_memories_filename)


def _dedupe_path(cfg: CentralMemoryConfig) -> str:
    return os.path.join(cfg.root_dir, cfg.dedupe_index_filename)


def _dedupe_db_path(cfg: CentralMemoryConfig) -> str:
    name = str(getattr(cfg, "dedupe_db_filename", "dedupe_index.sqlite") or "").strip()
    if not name:
        name = "dedupe_index.sqlite"
    if os.path.isabs(name):
        return name
    return os.path.join(cfg.root_dir, name)


def _repo_lock_path(cfg: CentralMemoryConfig) -> str:
    return os.path.join(cfg.root_dir, ".repo.lock")


def _tier_policy_path(cfg: CentralMemoryConfig) -> str:
    name = str(getattr(cfg, "tier_policy_filename", "tier_policy.json") or "").strip()
    if not name:
        return ""
    if os.path.isabs(name):
        return name
    return os.path.join(cfg.root_dir, name)


def _as_set(value: Optional[Set[str]]) -> Optional[Set[str]]:
    if value is None:
        return None
    out: Set[str] = set()
    for item in value:
        s = str(item).strip().lower()
        if s:
            out.add(s)
    return out if out else None


def _row_verse_name(row: Dict[str, Any]) -> str:
    return str(row.get("verse_name") or row.get("verse") or "").strip().lower()


def _normalize_memory_tier(value: Any) -> str:
    tier = str(value).strip().lower()
    if tier in ("ltm", "stm"):
        return tier
    return ""


def _policy_float(policy: Dict[str, Any], key: str, default: float) -> float:
    return _safe_float(policy.get(key, default), default)


def _policy_bool(policy: Dict[str, Any], key: str, default: bool) -> bool:
    v = policy.get(key, default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _policy_int(policy: Dict[str, Any], key: str, default: int) -> int:
    return _safe_int(policy.get(key, default), default)


def _ltm_priority_tuple(row: Dict[str, Any]) -> tuple[int, int, float, int, int]:
    info = row.get("info")
    info = info if isinstance(info, dict) else {}
    done = 1 if bool(row.get("done", False) or row.get("truncated", False)) else 0
    goal = 1 if bool(info.get("reached_goal", False) or info.get("episode_success", False)) else 0
    reward = _safe_float(row.get("reward", 0.0), 0.0)
    t_step = _safe_int(info.get("t", row.get("step_idx", 0)), 0)
    t_ms = _safe_int(row.get("t_ms", 0), 0)
    return (done, goal, float(reward), int(t_step), int(t_ms))


def _default_tier_policy() -> Dict[str, Any]:
    return {
        "ltm_reward_threshold": 2.0,
        "ltm_done_reward_threshold": 0.5,
        "promotion_score_threshold": 1.0,
        "done_min_t": 0,
        "reward_scale": 1.5,
        "positive_reward_weight": 1.0,
        "max_reward_bonus": 1.0,
        "done_bonus": 0.15,
        "late_stage_min_t": 10_000_000,
        "late_stage_reward_floor": 0.0,
        "late_stage_bonus": 0.0,
        "risk_sensitive_bonus": 0.15,
        "declarative_bonus": 0.20,
        "high_adventure_bonus": 0.30,
        "goal_bonus": 0.45,
        "success_bonus": 0.35,
        "sovereign_bonus": 0.60,
        "promote_sovereign": True,
        "promote_high_adventure": True,
        "promote_goal": True,
        "promote_episode_success": True,
        "support_guard_enabled": True,
        "support_guard_min_rows": 50,
        "support_guard_max_ltm_ratio": 0.05,
        "support_guard_min_ltm": 1,
    }


def _load_tier_policy(cfg: CentralMemoryConfig) -> Dict[str, Any]:
    policy = {"default": _default_tier_policy(), "by_verse": {}}
    path = _tier_policy_path(cfg)
    if not path or not os.path.isfile(path):
        return policy
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return policy
    if not isinstance(obj, dict):
        return policy

    default_block = obj.get("default")
    if isinstance(default_block, dict):
        for k, v in default_block.items():
            kk = str(k).strip()
            if kk:
                policy["default"][kk] = v

    by_verse = obj.get("by_verse")
    if isinstance(by_verse, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for verse, block in by_verse.items():
            vname = str(verse).strip().lower()
            if not vname or not isinstance(block, dict):
                continue
            out[vname] = {str(k).strip(): val for k, val in block.items() if str(k).strip()}
        policy["by_verse"] = out
    return policy


def _tier_policy_for_verse(verse_name: str, tier_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = _default_tier_policy()
    policy = tier_policy if isinstance(tier_policy, dict) else {}
    default_block = policy.get("default")
    if isinstance(default_block, dict):
        for k, v in default_block.items():
            kk = str(k).strip()
            if kk:
                out[kk] = v

    by_verse = policy.get("by_verse")
    if isinstance(by_verse, dict):
        verse_block = by_verse.get(str(verse_name).strip().lower())
        if isinstance(verse_block, dict):
            for k, v in verse_block.items():
                kk = str(k).strip()
                if kk:
                    out[kk] = v
    return out


def _memory_tier_for_event(event: Dict[str, Any], *, tier_policy: Optional[Dict[str, Any]] = None) -> str:
    info = event.get("info")
    info = info if isinstance(info, dict) else {}
    verse_name = _row_verse_name(event)
    memory_family = memory_family_for_verse(verse_name)
    verse_tags = set(tags_for_verse(verse_name))
    policy = _tier_policy_for_verse(verse_name, tier_policy)

    reward = _safe_float(event.get("reward", 0.0), 0.0)
    done = bool(event.get("done", False) or event.get("truncated", False))
    t_step = _safe_int(info.get("t", event.get("step_idx", 0)), 0)
    done_min_t = max(0, _policy_int(policy, "done_min_t", 0))
    done_eligible = bool(done and t_step >= done_min_t)
    if _policy_bool(policy, "promote_sovereign", True) and bool(info.get("sovereign_skill", False)):
        return "ltm"
    if _policy_bool(policy, "promote_high_adventure", True) and bool(info.get("high_adventure", False)):
        return "ltm"
    if _policy_bool(policy, "promote_goal", True) and bool(info.get("reached_goal", False)):
        return "ltm"
    if _policy_bool(policy, "promote_episode_success", True) and bool(info.get("episode_success", False)):
        return "ltm"

    done_reward_threshold = _policy_float(policy, "ltm_done_reward_threshold", 0.5)
    reward_threshold = _policy_float(policy, "ltm_reward_threshold", 2.0)
    if done_eligible and reward > float(done_reward_threshold):
        return "ltm"
    if reward >= float(reward_threshold):
        return "ltm"

    score = 0.0
    if done_eligible:
        score += _policy_float(policy, "done_bonus", 0.15)
    if reward > 0.0:
        reward_scale = max(1e-9, abs(_policy_float(policy, "reward_scale", 1.5)))
        reward_weight = _policy_float(policy, "positive_reward_weight", 1.0)
        max_reward_bonus = max(0.0, _policy_float(policy, "max_reward_bonus", 1.0))
        score += min(max_reward_bonus, (reward / reward_scale) * reward_weight)
    if "risk_sensitive" in verse_tags and reward > 0.0:
        score += _policy_float(policy, "risk_sensitive_bonus", 0.15)
    if memory_family == "declarative":
        score += _policy_float(policy, "declarative_bonus", 0.20)
    if _policy_bool(policy, "promote_high_adventure", True) and bool(info.get("high_adventure", False)):
        score += _policy_float(policy, "high_adventure_bonus", 0.30)
    if _policy_bool(policy, "promote_goal", True) and bool(info.get("reached_goal", False)):
        score += _policy_float(policy, "goal_bonus", 0.45)
    if _policy_bool(policy, "promote_episode_success", True) and bool(info.get("episode_success", False)):
        score += _policy_float(policy, "success_bonus", 0.35)
    if bool(info.get("sovereign_skill", False)):
        score += _policy_float(policy, "sovereign_bonus", 0.60)
    late_stage_min_t = max(0, _policy_int(policy, "late_stage_min_t", 10_000_000))
    late_stage_reward_floor = _policy_float(policy, "late_stage_reward_floor", 0.0)
    if t_step >= late_stage_min_t and reward >= late_stage_reward_floor:
        score += _policy_float(policy, "late_stage_bonus", 0.0)
    if score >= _policy_float(policy, "promotion_score_threshold", 1.0):
        return "ltm"
    return "stm"


def _enrich_memory_row(
    row: Dict[str, Any],
    *,
    tier_policy: Optional[Dict[str, Any]] = None,
    recompute_tier: bool = False,
) -> Dict[str, Any]:
    out = dict(row)
    verse_name = _row_verse_name(out)
    if verse_name:
        out["verse_name"] = verse_name

    row_tags = out.get("tags")
    if not isinstance(row_tags, list) or not row_tags:
        out["tags"] = tags_for_verse(verse_name)

    memory_type = str(out.get("memory_type", "")).strip().lower()
    if not memory_type:
        memory_type = memory_type_for_verse(verse_name)
        out["memory_type"] = memory_type

    memory_family = str(out.get("memory_family", "")).strip().lower()
    if not memory_family:
        memory_family = memory_family_for_verse(verse_name)
        out["memory_family"] = memory_family

    tier = _normalize_memory_tier(out.get("memory_tier"))
    if not tier or bool(recompute_tier):
        tier = _memory_tier_for_event(out, tier_policy=tier_policy)
        out["memory_tier"] = tier

    if "sovereign_skill" not in out:
        out["sovereign_skill"] = bool(tier == "ltm")
    if "procedural_dna" not in out:
        out["procedural_dna"] = bool(memory_family == "procedural")
    if "declarative_dna" not in out:
        out["declarative_dna"] = bool(memory_family == "declarative")
    return out
