"""
memory/central_repository.py

Central memory repository for cross-run knowledge sharing.

This module supports:
- ingesting run events into a shared memory bank
- deduplicating overlapping experiences
- segmented LTM/STM storage for sovereign skill vs short-lived context memory
- scenario matching via cosine similarity over observation vectors
"""

from __future__ import annotations

import heapq
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.types import JSONValue
from memory.decay_manager import apply_temporal_decay
from memory.embeddings import cosine_similarity, obs_to_universal_vector, obs_to_vector, project_vector
from memory.selection import SelectionConfig, select_events
from memory.task_taxonomy import memory_family_for_verse, memory_type_for_verse, tags_for_verse

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None
try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover
    NearestNeighbors = None

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None


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
    recency_weight: float = 1.0


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


@dataclass
class _PreparedMemoryRow:
    run_id: str
    episode_id: str
    step_idx: int
    t_ms: int
    verse_name: str
    row_type: str
    row_family: str
    row_tier: str
    action: JSONValue
    reward: float
    obs: JSONValue
    obs_vector: List[float]
    obs_vector_u: List[float]


@dataclass
class _SimilarityCacheEntry:
    signature: Tuple[int, int]
    rows: List[_PreparedMemoryRow]
    by_verse: Dict[str, List[int]]
    vectors_by_dim: Dict[int, Any]
    row_indices_by_dim: Dict[int, List[int]]
    ann_by_dim: Dict[int, Any]
    universal_vectors: Any
    universal_row_indices: List[int]
    built_at_ms: int


_SIM_CACHE_LOCK = threading.Lock()
_SIM_CACHE: "OrderedDict[str, _SimilarityCacheEntry]" = OrderedDict()
_DEDUPE_READY_LOCK = threading.Lock()
_DEDUPE_READY: Set[str] = set()
_ANN_TUNE_LOCK = threading.Lock()
_ANN_DYNAMIC_FACTOR: Optional[int] = None
_ANN_QUERY_COUNT = 0
_ANN_DRIFT_CHECKS = 0
_ANN_LAST_DRIFT = 0.0
_ANN_MAX_DRIFT = 0.0


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


@contextmanager
def _repo_lock(cfg: CentralMemoryConfig):
    os.makedirs(cfg.root_dir, exist_ok=True)
    lock_path = _repo_lock_path(cfg)
    with open(lock_path, "a+b") as fh:
        locked = False
        try:
            if msvcrt is not None:
                fh.seek(0, os.SEEK_END)
                if fh.tell() <= 0:
                    fh.write(b"0")
                    fh.flush()
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                locked = True
            elif fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                locked = True
            yield
        finally:
            if locked:
                try:
                    fh.seek(0)
                    if msvcrt is not None:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    elif fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


def _ensure_repo(cfg: CentralMemoryConfig) -> None:
    os.makedirs(cfg.root_dir, exist_ok=True)
    for mem_path in (_memories_path(cfg), _ltm_memories_path(cfg), _stm_memories_path(cfg)):
        if not os.path.isfile(mem_path):
            with open(mem_path, "w", encoding="utf-8"):
                pass
    idx_path = _dedupe_path(cfg)
    if not os.path.isfile(idx_path):
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    db_path = os.path.abspath(_dedupe_db_path(cfg))
    with _DEDUPE_READY_LOCK:
        ready = db_path in _DEDUPE_READY
    if not ready:
        conn = _open_dedupe_db(cfg)
        try:
            _migrate_legacy_dedupe_json_to_db(cfg, conn)
        finally:
            conn.close()
        with _DEDUPE_READY_LOCK:
            _DEDUPE_READY.add(db_path)


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


def _open_dedupe_db(cfg: CentralMemoryConfig) -> sqlite3.Connection:
    db_path = _dedupe_db_path(cfg)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dedupe_keys(
            dedupe_key TEXT PRIMARY KEY
        )
        """
    )
    conn.commit()
    return conn


def _migrate_legacy_dedupe_json_to_db(cfg: CentralMemoryConfig, conn: sqlite3.Connection) -> None:
    try:
        n = int(conn.execute("SELECT COUNT(*) FROM dedupe_keys").fetchone()[0] or 0)
    except Exception:
        n = 0
    if n > 0:
        return

    idx_path = _dedupe_path(cfg)
    if not os.path.isfile(idx_path):
        return
    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, list) or not data:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO dedupe_keys(dedupe_key) VALUES (?)",
        [(str(x),) for x in data if str(x)],
    )
    conn.commit()


def _dedupe_try_reserve(conn: sqlite3.Connection, key: str) -> bool:
    cur = conn.execute("INSERT OR IGNORE INTO dedupe_keys(dedupe_key) VALUES (?)", (str(key),))
    try:
        return int(cur.rowcount or 0) > 0
    except Exception:
        return False


def _sim_cache_limit() -> int:
    raw = os.environ.get("MULTIVERSE_SIM_CACHE_MAX_FILES", "2")
    try:
        return max(1, int(raw))
    except Exception:
        return 2


def _ann_enabled() -> bool:
    raw = str(os.environ.get("MULTIVERSE_SIM_USE_ANN", "1")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _ann_candidate_count(top_n: int, n_rows: int) -> int:
    factor_raw = os.environ.get("MULTIVERSE_SIM_ANN_FACTOR", "64")
    try:
        factor = max(1, int(factor_raw))
    except Exception:
        factor = 64
    with _ANN_TUNE_LOCK:
        dyn = _ANN_DYNAMIC_FACTOR
    if dyn is not None:
        factor = max(factor, int(dyn))
    return min(max(256, int(top_n) * factor), int(n_rows))


def _ann_drift_check_every() -> int:
    raw = os.environ.get("MULTIVERSE_SIM_ANN_DRIFT_CHECK_EVERY", "250")
    try:
        return max(1, int(raw))
    except Exception:
        return 250


def _ann_max_allowed_drift() -> float:
    raw = os.environ.get("MULTIVERSE_SIM_ANN_MAX_DRIFT", "0.03")
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.03


def _ann_should_check_drift() -> bool:
    global _ANN_QUERY_COUNT
    every = _ann_drift_check_every()
    with _ANN_TUNE_LOCK:
        _ANN_QUERY_COUNT += 1
        return bool((_ANN_QUERY_COUNT % every) == 0)


def _ann_record_drift(*, drift: float, n_rows: int, top_n: int) -> None:
    global _ANN_DRIFT_CHECKS, _ANN_LAST_DRIFT, _ANN_MAX_DRIFT, _ANN_DYNAMIC_FACTOR
    d = max(0.0, float(drift))
    max_allowed = _ann_max_allowed_drift()
    with _ANN_TUNE_LOCK:
        _ANN_DRIFT_CHECKS += 1
        _ANN_LAST_DRIFT = float(d)
        _ANN_MAX_DRIFT = max(float(_ANN_MAX_DRIFT), float(d))
        if d <= max_allowed:
            return
        current = int(_ANN_DYNAMIC_FACTOR or 0)
        if current <= 0:
            current = 64
        max_factor = max(64, int(max(1, n_rows) / max(1, top_n)))
        _ANN_DYNAMIC_FACTOR = min(max_factor, max(current + 8, int(current * 1.5)))


def get_similarity_runtime_metrics() -> Dict[str, Any]:
    with _ANN_TUNE_LOCK:
        return {
            "ann_dynamic_factor": (None if _ANN_DYNAMIC_FACTOR is None else int(_ANN_DYNAMIC_FACTOR)),
            "ann_query_count": int(_ANN_QUERY_COUNT),
            "ann_drift_checks": int(_ANN_DRIFT_CHECKS),
            "ann_last_drift": float(_ANN_LAST_DRIFT),
            "ann_max_drift": float(_ANN_MAX_DRIFT),
            "ann_drift_check_every": int(_ann_drift_check_every()),
            "ann_max_allowed_drift": float(_ann_max_allowed_drift()),
        }


def reset_similarity_runtime_metrics() -> None:
    global _ANN_DYNAMIC_FACTOR, _ANN_QUERY_COUNT, _ANN_DRIFT_CHECKS, _ANN_LAST_DRIFT, _ANN_MAX_DRIFT
    with _ANN_TUNE_LOCK:
        _ANN_DYNAMIC_FACTOR = None
        _ANN_QUERY_COUNT = 0
        _ANN_DRIFT_CHECKS = 0
        _ANN_LAST_DRIFT = 0.0
        _ANN_MAX_DRIFT = 0.0


def _file_signature(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    mtime_ns = int(getattr(st, "st_mtime_ns", int(float(st.st_mtime) * 1_000_000_000)))
    return int(st.st_size), int(mtime_ns)


def _simcache_path(mem_path: str) -> str:
    return str(mem_path) + ".simcache.json"


def _legacy_simcache_path(mem_path: str) -> str:
    return str(mem_path) + ".simcache.pkl"


def _universal_obs_dim() -> int:
    raw = os.environ.get("MULTIVERSE_UNIVERSAL_OBS_DIM", "64")
    try:
        return max(4, int(raw))
    except Exception:
        return 64


def _extract_or_build_universal_vector(
    *,
    row: Dict[str, Any],
    obs_vector: Optional[List[float]] = None,
) -> List[float]:
    raw = row.get("obs_vector_u")
    if isinstance(raw, list):
        try:
            vec = [float(v) for v in raw]
            if vec:
                return vec
        except Exception:
            pass

    if obs_vector:
        try:
            # Re-projecting an existing vector keeps migration cheap and deterministic.
            return project_vector(obs_vector, dim=_universal_obs_dim())
        except Exception:
            pass
    try:
        return obs_to_universal_vector(row.get("obs"), dim=_universal_obs_dim())
    except Exception:
        return []


def _vectorize_rows_by_dim(rows: List[_PreparedMemoryRow]) -> tuple[Dict[int, Any], Dict[int, List[int]]]:
    vectors_by_dim: Dict[int, Any] = {}
    row_indices_by_dim: Dict[int, List[int]] = {}
    for idx, row in enumerate(rows):
        dim = int(len(row.obs_vector))
        if dim <= 0:
            continue
        row_indices_by_dim.setdefault(dim, []).append(int(idx))
    if np is None:
        return vectors_by_dim, row_indices_by_dim
    for dim, indices in row_indices_by_dim.items():
        mat = np.asarray([rows[i].obs_vector for i in indices], dtype=np.float32)
        if mat.size <= 0:
            continue
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        mat = mat / norms
        vectors_by_dim[dim] = mat
    return vectors_by_dim, row_indices_by_dim


def _vectorize_universal_rows(rows: List[_PreparedMemoryRow]) -> tuple[Any, List[int]]:
    indices: List[int] = []
    if np is None:
        return None, indices
    mat_rows: List[List[float]] = []
    dim = _universal_obs_dim()
    for idx, row in enumerate(rows):
        vec = list(row.obs_vector_u or [])
        if len(vec) != dim:
            continue
        indices.append(int(idx))
        mat_rows.append(vec)
    if not mat_rows:
        return None, []
    mat = np.asarray(mat_rows, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    mat = mat / norms
    return mat, indices


def _prepared_row_from_any(raw: Any) -> Optional[_PreparedMemoryRow]:
    if isinstance(raw, _PreparedMemoryRow):
        return raw
    if not isinstance(raw, dict):
        return None
    try:
        base_vec = [float(x) for x in list(raw.get("obs_vector", []) or [])]
    except Exception:
        return None
    if not base_vec:
        return None
    try:
        uni_vec = [float(x) for x in list(raw.get("obs_vector_u", []) or [])]
    except Exception:
        uni_vec = []
    if not uni_vec:
        uni_vec = _extract_or_build_universal_vector(row=raw, obs_vector=base_vec)
    return _PreparedMemoryRow(
        run_id=str(raw.get("run_id", "")),
        episode_id=str(raw.get("episode_id", "")),
        step_idx=_safe_int(raw.get("step_idx", 0)),
        t_ms=_safe_int(raw.get("t_ms", 0)),
        verse_name=str(raw.get("verse_name", "")),
        row_type=str(raw.get("row_type", "")),
        row_family=str(raw.get("row_family", "")),
        row_tier=str(raw.get("row_tier", "")),
        action=raw.get("action"),
        reward=_safe_float(raw.get("reward", 0.0)),
        obs=raw.get("obs"),
        obs_vector=base_vec,
        obs_vector_u=uni_vec,
    )


def _prepared_row_to_dict(row: _PreparedMemoryRow) -> Dict[str, Any]:
    return {
        "run_id": str(row.run_id),
        "episode_id": str(row.episode_id),
        "step_idx": int(row.step_idx),
        "t_ms": int(row.t_ms),
        "verse_name": str(row.verse_name),
        "row_type": str(row.row_type),
        "row_family": str(row.row_family),
        "row_tier": str(row.row_tier),
        "action": row.action,
        "reward": float(row.reward),
        "obs": row.obs,
        "obs_vector": [float(v) for v in list(row.obs_vector or [])],
        "obs_vector_u": [float(v) for v in list(row.obs_vector_u or [])],
    }


def _index_rows_by_verse(rows: List[_PreparedMemoryRow]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        verse = str(row.verse_name or "")
        out.setdefault(verse, []).append(int(idx))
    return out


def _get_ann_index(cache: _SimilarityCacheEntry, dim: int) -> Optional[Any]:
    if NearestNeighbors is None or np is None:
        return None
    if dim in cache.ann_by_dim:
        return cache.ann_by_dim.get(dim)
    mat = cache.vectors_by_dim.get(dim)
    if mat is None:
        return None
    try:
        model = NearestNeighbors(metric="euclidean", algorithm="auto")
        model.fit(mat)
    except Exception:
        return None
    cache.ann_by_dim[dim] = model
    return model


def _invalidate_similarity_cache(paths: Iterable[str]) -> None:
    keys = set(os.path.abspath(str(p)) for p in paths if str(p))
    if not keys:
        return
    with _SIM_CACHE_LOCK:
        for k in list(_SIM_CACHE.keys()):
            if k in keys:
                _SIM_CACHE.pop(k, None)
    for k in keys:
        for p in (_simcache_path(k), _legacy_simcache_path(k)):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass


def _build_similarity_cache_for_path(
    *,
    mem_path: str,
    tier_policy: Optional[Dict[str, Any]],
) -> _SimilarityCacheEntry:
    apath = os.path.abspath(mem_path)
    sig = _file_signature(apath)
    sidecar = _simcache_path(apath)
    legacy_sidecar = _legacy_simcache_path(apath)
    if os.path.isfile(legacy_sidecar):
        try:
            os.remove(legacy_sidecar)
        except Exception:
            pass
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if (
                isinstance(payload, dict)
                and str(payload.get("format", "")).strip() == "simcache_json_v1"
                and isinstance(payload.get("signature"), list)
                and tuple(payload.get("signature", ())) == tuple(sig)
                and isinstance(payload.get("rows"), list)
            ):
                rows: List[_PreparedMemoryRow] = []
                for raw_row in payload.get("rows", []):
                    rec = _prepared_row_from_any(raw_row)
                    if rec is not None:
                        rows.append(rec)
                if not rows:
                    raise ValueError("invalid simcache rows")
                by_verse = _index_rows_by_verse(rows)
                vectors_by_dim, row_indices_by_dim = _vectorize_rows_by_dim(rows)
                universal_vectors, universal_row_indices = _vectorize_universal_rows(rows)
                return _SimilarityCacheEntry(
                    signature=sig,
                    rows=rows,
                    by_verse=by_verse,
                    vectors_by_dim=vectors_by_dim,
                    row_indices_by_dim=row_indices_by_dim,
                    ann_by_dim={},
                    universal_vectors=universal_vectors,
                    universal_row_indices=[int(i) for i in universal_row_indices],
                    built_at_ms=int(time.time() * 1000),
                )
        except Exception:
            pass

    rows: List[_PreparedMemoryRow] = []
    by_verse: Dict[str, List[int]] = {}
    with open(apath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            row_vec = row.get("obs_vector")
            if not isinstance(row_vec, list):
                continue
            try:
                row_vec_f = [float(v) for v in row_vec]
            except Exception:
                continue
            row_verse = _row_verse_name(row)
            row_type = str(row.get("memory_type", memory_type_for_verse(row_verse))).strip().lower()
            row_family = str(row.get("memory_family", memory_family_for_verse(row_verse))).strip().lower()
            row_tier = _normalize_memory_tier(row.get("memory_tier"))
            if not row_tier:
                row_tier = _memory_tier_for_event(row, tier_policy=tier_policy)

            rec = _PreparedMemoryRow(
                run_id=str(row.get("run_id", "")),
                episode_id=str(row.get("episode_id", "")),
                step_idx=_safe_int(row.get("step_idx", 0)),
                t_ms=_safe_int(row.get("t_ms", 0)),
                verse_name=row_verse,
                row_type=row_type,
                row_family=row_family,
                row_tier=row_tier,
                action=row.get("action"),
                reward=_safe_float(row.get("reward", 0.0)),
                obs=row.get("obs"),
                obs_vector=row_vec_f,
                obs_vector_u=_extract_or_build_universal_vector(row=row, obs_vector=row_vec_f),
            )
            idx = len(rows)
            rows.append(rec)
            by_verse.setdefault(row_verse, []).append(idx)
    vectors_by_dim, row_indices_by_dim = _vectorize_rows_by_dim(rows)
    universal_vectors, universal_row_indices = _vectorize_universal_rows(rows)
    entry = _SimilarityCacheEntry(
        signature=sig,
        rows=rows,
        by_verse=by_verse,
        vectors_by_dim=vectors_by_dim,
        row_indices_by_dim=row_indices_by_dim,
        ann_by_dim={},
        universal_vectors=universal_vectors,
        universal_row_indices=[int(i) for i in universal_row_indices],
        built_at_ms=int(time.time() * 1000),
    )
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "simcache_json_v1",
                    "signature": [int(sig[0]), int(sig[1])],
                    "rows": [_prepared_row_to_dict(r) for r in rows],
                    "by_verse": {str(k): [int(i) for i in list(v)] for k, v in by_verse.items()},
                },
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )
    except Exception:
        pass
    return entry


def _get_similarity_cache_for_path(
    *,
    mem_path: str,
    tier_policy: Optional[Dict[str, Any]],
) -> _SimilarityCacheEntry:
    apath = os.path.abspath(mem_path)
    sig = _file_signature(apath)
    with _SIM_CACHE_LOCK:
        hit = _SIM_CACHE.get(apath)
        if hit is not None and hit.signature == sig:
            _SIM_CACHE.move_to_end(apath)
            return hit

    fresh = _build_similarity_cache_for_path(mem_path=apath, tier_policy=tier_policy)
    with _SIM_CACHE_LOCK:
        _SIM_CACHE[apath] = fresh
        _SIM_CACHE.move_to_end(apath)
        while len(_SIM_CACHE) > _sim_cache_limit():
            _SIM_CACHE.popitem(last=False)
    return fresh


def _load_dedupe_index(cfg: CentralMemoryConfig) -> Set[str]:
    _ensure_repo(cfg)
    conn = _open_dedupe_db(cfg)
    try:
        _migrate_legacy_dedupe_json_to_db(cfg, conn)
        rows = conn.execute("SELECT dedupe_key FROM dedupe_keys").fetchall()
        return set(str(r[0]) for r in rows if r and r[0])
    finally:
        conn.close()


def _save_dedupe_index(cfg: CentralMemoryConfig, keys: Iterable[str]) -> None:
    _ensure_repo(cfg)
    conn = _open_dedupe_db(cfg)
    try:
        conn.execute("DELETE FROM dedupe_keys")
        conn.executemany(
            "INSERT OR IGNORE INTO dedupe_keys(dedupe_key) VALUES (?)",
            [(str(k),) for k in set(str(k) for k in keys) if str(k)],
        )
        conn.commit()
    finally:
        conn.close()


def _iter_events(events_path: str):
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _load_events(events_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in _iter_events(events_path):
        rows.append(row)
    return rows


def sanitize_memory_file(cfg: CentralMemoryConfig) -> SanitizeStats:
    """
    Rewrites memory file keeping only valid JSON rows.
    Useful if previous interrupted parallel writes created malformed lines.
    """
    _ensure_repo(cfg)
    mem_path = _memories_path(cfg)
    tmp_path = mem_path + ".tmp"
    total = 0
    kept = 0
    with _repo_lock(cfg):
        with open(mem_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                total += 1
                s = line.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                except Exception:
                    continue
                dst.write(s + "\n")
                kept += 1
        os.replace(tmp_path, mem_path)
        _invalidate_similarity_cache((mem_path, _ltm_memories_path(cfg), _stm_memories_path(cfg)))
    return SanitizeStats(input_lines=total, kept_lines=kept, dropped_lines=max(0, total - kept))


def backfill_memory_metadata(
    *,
    cfg: CentralMemoryConfig,
    rebuild_tier_files: bool = True,
    recompute_tier: bool = False,
    apply_support_guards: bool = True,
) -> BackfillStats:
    """
    Enrich existing memory rows with missing tier/family/type metadata and
    optionally rebuild dedicated LTM/STM files.
    """
    _ensure_repo(cfg)
    mem_path = _memories_path(cfg)
    ltm_path = _ltm_memories_path(cfg)
    stm_path = _stm_memories_path(cfg)

    tmp_mem_stage_path = mem_path + ".backfill.stage.tmp"
    tmp_mem_final_path = mem_path + ".backfill.final.tmp"
    tmp_ltm_path = ltm_path + ".backfill.tmp"
    tmp_stm_path = stm_path + ".backfill.tmp"

    rows_scanned = 0
    rows_written = 0
    malformed_rows = 0
    backfilled_tier = 0
    recomputed_tier = 0
    support_guard_demotions = 0
    backfilled_family = 0
    backfilled_type = 0
    backfilled_obs_vector_u = 0
    ltm_rows = 0
    stm_rows = 0
    tier_policy = _load_tier_policy(cfg)
    verse_rows: Dict[str, int] = {}
    verse_ltm_candidates: Dict[str, List[tuple[tuple[int, int, float, int, int], int]]] = {}

    with _repo_lock(cfg):
        with open(mem_path, "r", encoding="utf-8") as src, open(tmp_mem_stage_path, "w", encoding="utf-8") as stage:
            row_ordinal = 0
            for line in src:
                s = line.strip()
                if not s:
                    continue
                rows_scanned += 1
                try:
                    row = json.loads(s)
                except Exception:
                    malformed_rows += 1
                    continue
                if not isinstance(row, dict):
                    malformed_rows += 1
                    continue

                had_type = bool(str(row.get("memory_type", "")).strip())
                had_family = bool(str(row.get("memory_family", "")).strip())
                old_tier = _normalize_memory_tier(row.get("memory_tier"))
                had_tier = bool(old_tier)
                enriched = _enrich_memory_row(
                    row,
                    tier_policy=tier_policy,
                    recompute_tier=bool(recompute_tier),
                )
                had_vector_u = isinstance(row.get("obs_vector_u"), list) and bool(row.get("obs_vector_u"))
                obs_vec_raw = row.get("obs_vector")
                obs_vec_list: Optional[List[float]] = None
                if isinstance(obs_vec_raw, list):
                    try:
                        obs_vec_list = [float(v) for v in obs_vec_raw]
                    except Exception:
                        obs_vec_list = None
                enriched["obs_vector_u"] = _extract_or_build_universal_vector(
                    row=enriched,
                    obs_vector=obs_vec_list,
                )
                if (not had_vector_u) and isinstance(enriched.get("obs_vector_u"), list) and enriched.get("obs_vector_u"):
                    backfilled_obs_vector_u += 1
                new_tier = _normalize_memory_tier(enriched.get("memory_tier"))

                if not had_type and bool(str(enriched.get("memory_type", "")).strip()):
                    backfilled_type += 1
                if not had_family and bool(str(enriched.get("memory_family", "")).strip()):
                    backfilled_family += 1
                if not had_tier and bool(new_tier):
                    backfilled_tier += 1
                if bool(recompute_tier) and had_tier and old_tier != new_tier:
                    recomputed_tier += 1

                verse_name = _row_verse_name(enriched)
                verse_rows[verse_name] = int(verse_rows.get(verse_name, 0)) + 1
                if new_tier == "ltm":
                    bucket = verse_ltm_candidates.setdefault(verse_name, [])
                    bucket.append((_ltm_priority_tuple(enriched), int(row_ordinal)))

                stage.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                row_ordinal += 1

        demote_ordinals: Set[int] = set()
        if bool(apply_support_guards):
            for verse_name, count in verse_rows.items():
                policy = _tier_policy_for_verse(verse_name, tier_policy)
                if not _policy_bool(policy, "support_guard_enabled", True):
                    continue
                min_rows = max(0, _policy_int(policy, "support_guard_min_rows", 50))
                if int(count) >= int(min_rows):
                    continue

                ratio = _policy_float(policy, "support_guard_max_ltm_ratio", 0.05)
                ratio = max(0.0, min(1.0, float(ratio)))
                min_ltm = max(0, _policy_int(policy, "support_guard_min_ltm", 1))
                max_ltm = max(int(min_ltm), int(float(count) * float(ratio)))

                cands = list(verse_ltm_candidates.get(verse_name, []))
                if len(cands) <= max_ltm:
                    continue
                cands.sort(key=lambda x: x[0], reverse=True)
                for _, ordinal in cands[max_ltm:]:
                    demote_ordinals.add(int(ordinal))

        ltm_out = open(tmp_ltm_path, "w", encoding="utf-8") if rebuild_tier_files else None
        stm_out = open(tmp_stm_path, "w", encoding="utf-8") if rebuild_tier_files else None
        try:
            with open(tmp_mem_stage_path, "r", encoding="utf-8") as stage, open(
                tmp_mem_final_path, "w", encoding="utf-8"
            ) as dst:
                row_ordinal = 0
                for line in stage:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        row = json.loads(s)
                    except Exception:
                        malformed_rows += 1
                        row_ordinal += 1
                        continue
                    if not isinstance(row, dict):
                        malformed_rows += 1
                        row_ordinal += 1
                        continue

                    tier = _normalize_memory_tier(row.get("memory_tier"))
                    if int(row_ordinal) in demote_ordinals and tier == "ltm":
                        row["memory_tier"] = "stm"
                        row["sovereign_skill"] = False
                        tier = "stm"
                        support_guard_demotions += 1
                    if tier not in ("ltm", "stm"):
                        tier = "stm"
                        row["memory_tier"] = "stm"
                        row["sovereign_skill"] = False

                    row_json = json.dumps(row, ensure_ascii=False) + "\n"
                    dst.write(row_json)
                    rows_written += 1
                    if tier == "ltm":
                        ltm_rows += 1
                        if ltm_out is not None:
                            ltm_out.write(row_json)
                    else:
                        stm_rows += 1
                        if stm_out is not None:
                            stm_out.write(row_json)
                    row_ordinal += 1
        finally:
            if ltm_out is not None:
                ltm_out.close()
            if stm_out is not None:
                stm_out.close()

        os.replace(tmp_mem_final_path, mem_path)
        if rebuild_tier_files:
            os.replace(tmp_ltm_path, ltm_path)
            os.replace(tmp_stm_path, stm_path)
        else:
            for p in (tmp_ltm_path, tmp_stm_path):
                if os.path.isfile(p):
                    os.remove(p)
        if os.path.isfile(tmp_mem_stage_path):
            os.remove(tmp_mem_stage_path)
        _invalidate_similarity_cache((mem_path, ltm_path, stm_path))

    return BackfillStats(
        rows_scanned=rows_scanned,
        rows_written=rows_written,
        malformed_rows=malformed_rows,
        backfilled_memory_tier=backfilled_tier,
        recomputed_memory_tier=recomputed_tier,
        support_guard_demotions=support_guard_demotions,
        backfilled_memory_family=backfilled_family,
        backfilled_memory_type=backfilled_type,
        ltm_rows=ltm_rows,
        stm_rows=stm_rows,
        backfilled_obs_vector_u=backfilled_obs_vector_u,
    )


def ingest_run(
    *,
    run_dir: str,
    cfg: CentralMemoryConfig,
    selection: Optional[SelectionConfig] = None,
    max_events: Optional[int] = None,
) -> IngestStats:
    """
    Ingest a run's events into the central repository.
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"events file not found: {events_path}")

    input_events = 0
    selected_events_count = 0
    if selection is not None:
        all_events = _load_events(events_path)
        input_events = len(all_events)
        selected_events = select_events(all_events, selection)
        if max_events is not None and max_events > 0:
            selected_events = selected_events[: int(max_events)]
        selected_events_count = len(selected_events)
        event_iter = iter(selected_events)
    else:
        event_iter = _iter_events(events_path)

    added = 0
    skipped = 0
    mem_path = _memories_path(cfg)
    ltm_path = _ltm_memories_path(cfg)
    stm_path = _stm_memories_path(cfg)
    run_id = os.path.basename(os.path.normpath(run_dir))
    tier_policy = _load_tier_policy(cfg)

    _ensure_repo(cfg)
    with _repo_lock(cfg):
        dedupe_conn = _open_dedupe_db(cfg)
        try:
            _migrate_legacy_dedupe_json_to_db(cfg, dedupe_conn)
            with (
                open(mem_path, "a", encoding="utf-8") as out,
                open(ltm_path, "a", encoding="utf-8") as out_ltm,
                open(stm_path, "a", encoding="utf-8") as out_stm,
            ):
                for ev in event_iter:
                    if selection is None:
                        input_events += 1
                        if max_events is not None and max_events > 0 and selected_events_count >= int(max_events):
                            continue
                        selected_events_count += 1

                    try:
                        obs_vec = obs_to_vector(ev.get("obs"))
                    except Exception:
                        # Non-vectorizable observations are ignored for scenario matching.
                        skipped += 1
                        continue

                    key = _dedupe_key(ev)
                    if not _dedupe_try_reserve(dedupe_conn, key):
                        skipped += 1
                        continue

                    verse_name = str(ev.get("verse_name", ""))
                    memory_type = memory_type_for_verse(verse_name)
                    memory_family = memory_family_for_verse(verse_name)
                    memory_tier = _memory_tier_for_event(ev, tier_policy=tier_policy)
                    row = {
                        "dedupe_key": key,
                        "run_id": run_id,
                        "episode_id": str(ev.get("episode_id", "")),
                        "step_idx": _safe_int(ev.get("step_idx", 0)),
                        "verse_name": verse_name,
                        "tags": tags_for_verse(verse_name),
                        "memory_type": memory_type,
                        "memory_family": memory_family,
                        "memory_tier": memory_tier,
                        "sovereign_skill": bool(memory_tier == "ltm"),
                        "procedural_dna": bool(memory_family == "procedural"),
                        "declarative_dna": bool(memory_family == "declarative"),
                        "policy_id": str(ev.get("policy_id", "")),
                        "t_ms": _safe_int(ev.get("t_ms", 0)),
                        "obs": ev.get("obs"),
                        "obs_vector": obs_vec,
                        "obs_vector_u": project_vector(obs_vec, dim=_universal_obs_dim()),
                        "action": ev.get("action"),
                        "reward": _safe_float(ev.get("reward", 0.0)),
                        "done": bool(ev.get("done", False)),
                        "info": ev.get("info", {}),
                    }
                    row = _enrich_memory_row(row, tier_policy=tier_policy)
                    row_tier = _normalize_memory_tier(row.get("memory_tier"))
                    row_json = json.dumps(row, ensure_ascii=False) + "\n"
                    out.write(row_json)
                    if row_tier == "ltm":
                        out_ltm.write(row_json)
                    else:
                        out_stm.write(row_json)
                    added += 1
            dedupe_conn.commit()
            _invalidate_similarity_cache((mem_path, ltm_path, stm_path))
        finally:
            dedupe_conn.close()

    return IngestStats(
        run_dir=run_dir,
        input_events=input_events,
        selected_events=selected_events_count,
        added_events=added,
        skipped_duplicates=skipped,
    )


def find_similar(
    *,
    obs: JSONValue,
    cfg: CentralMemoryConfig,
    top_k: int = 5,
    verse_name: Optional[str] = None,
    min_score: float = -1.0,
    exclude_run_ids: Optional[Set[str]] = None,
    decay_lambda: float = 0.0,
    current_time_ms: Optional[int] = None,
    memory_tiers: Optional[Set[str]] = None,
    memory_families: Optional[Set[str]] = None,
    memory_types: Optional[Set[str]] = None,
    stm_decay_lambda: Optional[float] = None,
) -> List[ScenarioMatch]:
    """
    Query central memory for similar observations.
    """
    _ensure_repo(cfg)
    query_vec = obs_to_vector(obs)
    query_vec_u = obs_to_universal_vector(obs, dim=_universal_obs_dim())
    q_u_dim = len(query_vec_u)
    q_u_norm: Optional[Any] = None
    if np is not None and q_u_dim > 0:
        try:
            q_arr_u = np.asarray(query_vec_u, dtype=np.float32)
            qn_u = float(np.linalg.norm(q_arr_u))
            if qn_u > 1e-12:
                q_u_norm = (q_arr_u / qn_u).astype(np.float32)
        except Exception:
            q_u_norm = None
    top_n = max(1, int(top_k))
    tier_filter = _as_set(memory_tiers)
    family_filter = _as_set(memory_families)
    type_filter = _as_set(memory_types)

    mem_paths: List[str]
    if tier_filter == {"ltm"}:
        mem_paths = [_ltm_memories_path(cfg)]
    elif tier_filter == {"stm"}:
        mem_paths = [_stm_memories_path(cfg)]
    else:
        mem_paths = [_memories_path(cfg)]
    tier_policy = _load_tier_policy(cfg)
    target_verse = str(verse_name).strip().lower() if verse_name else ""

    heap: List[tuple[float, int, ScenarioMatch]] = []
    ordinal = 0
    for mem_path in mem_paths:
        if not os.path.isfile(mem_path):
            continue
        cache = _get_similarity_cache_for_path(mem_path=mem_path, tier_policy=tier_policy)
        q_dim = len(query_vec)
        q_norm: Optional[Any] = None
        sims: Optional[Any] = None
        dim_row_idxs = list(cache.row_indices_by_dim.get(q_dim, []))
        raw_dim_available = bool(q_dim > 0 and dim_row_idxs)
        use_vec = bool(np is not None and raw_dim_available and q_dim in cache.vectors_by_dim)
        if use_vec:
            try:
                q_arr = np.asarray(query_vec, dtype=np.float32)
                qn = float(np.linalg.norm(q_arr))
                if qn > 1e-12:
                    q_norm = (q_arr / qn).astype(np.float32)
                    sims = cache.vectors_by_dim[q_dim] @ q_norm
            except Exception:
                use_vec = False
                sims = None

        if raw_dim_available and use_vec and sims is not None:
            local_best_score = float("-inf")

            def _score_position(pos: int) -> Optional[tuple[float, _PreparedMemoryRow, float]]:
                row_idx = dim_row_idxs[int(pos)]
                row = cache.rows[int(row_idx)]
                if target_verse and row.verse_name != target_verse:
                    return None
                if exclude_run_ids and str(row.run_id) in exclude_run_ids:
                    return None
                if tier_filter is not None and row.row_tier not in tier_filter:
                    return None
                if family_filter is not None and row.row_family not in family_filter:
                    return None
                if type_filter is not None and row.row_type not in type_filter:
                    return None

                raw_score = float(sims[pos])
                t_ms = int(row.t_ms)
                if row.row_tier == "ltm":
                    row_decay_lambda = 0.0
                elif stm_decay_lambda is None:
                    row_decay_lambda = max(float(decay_lambda), float(cfg.stm_decay_lambda))
                else:
                    row_decay_lambda = max(0.0, float(stm_decay_lambda))

                score, recency_weight = apply_temporal_decay(
                    score=float(raw_score),
                    t_ms=t_ms,
                    decay_lambda=float(row_decay_lambda),
                    current_time_ms=current_time_ms,
                )
                if score < float(min_score):
                    return None
                return float(score), row, float(recency_weight)

            def _scan_position(pos: int) -> None:
                nonlocal ordinal
                nonlocal local_best_score
                scored = _score_position(pos)
                if scored is None:
                    return
                score, row, recency_weight = scored
                local_best_score = max(float(local_best_score), float(score))
                t_ms = int(row.t_ms)

                match = ScenarioMatch(
                    score=float(score),
                    run_id=str(row.run_id),
                    episode_id=str(row.episode_id),
                    step_idx=int(row.step_idx),
                    t_ms=t_ms,
                    verse_name=str(row.verse_name),
                    action=row.action,
                    reward=float(row.reward),
                    obs=row.obs,
                    recency_weight=float(recency_weight),
                )
                if len(heap) < top_n:
                    heapq.heappush(heap, (float(score), ordinal, match))
                elif float(score) > heap[0][0]:
                    heapq.heapreplace(heap, (float(score), ordinal, match))
                ordinal += 1

            candidate_positions: Optional[List[int]] = None
            if _ann_enabled() and q_norm is not None and len(dim_row_idxs) > top_n:
                ann_index = _get_ann_index(cache, q_dim)
                if ann_index is not None:
                    try:
                        n_candidates = _ann_candidate_count(top_n, len(dim_row_idxs))
                        if 0 < n_candidates < len(dim_row_idxs):
                            idxs = ann_index.kneighbors(
                                np.asarray([q_norm], dtype=np.float32),
                                n_neighbors=int(n_candidates),
                                return_distance=False,
                            )
                            if idxs is not None and len(idxs):
                                candidate_positions = []
                                seen: Set[int] = set()
                                for raw_pos in list(idxs[0]):
                                    pos = int(raw_pos)
                                    if pos < 0 or pos >= len(dim_row_idxs):
                                        continue
                                    if pos in seen:
                                        continue
                                    seen.add(pos)
                                    candidate_positions.append(pos)
                    except Exception:
                        candidate_positions = None

            if candidate_positions is None:
                for pos in range(len(dim_row_idxs)):
                    _scan_position(pos)
            else:
                visited = set(candidate_positions)
                did_full_scan = False
                for pos in candidate_positions:
                    _scan_position(pos)
                if len(heap) < top_n:
                    did_full_scan = True
                    for pos in range(len(dim_row_idxs)):
                        if pos in visited:
                            continue
                        _scan_position(pos)
                if not did_full_scan and _ann_should_check_drift():
                    exact_best = float("-inf")
                    for pos in range(len(dim_row_idxs)):
                        scored = _score_position(pos)
                        if scored is None:
                            continue
                        exact_best = max(float(exact_best), float(scored[0]))
                    if exact_best > float("-inf"):
                        if local_best_score > float("-inf"):
                            drift = max(0.0, float(exact_best - local_best_score))
                        else:
                            drift = max(0.0, float(exact_best - float(min_score)))
                        _ann_record_drift(drift=drift, n_rows=len(dim_row_idxs), top_n=top_n)
        elif raw_dim_available:
            if target_verse:
                row_iter: Iterable[_PreparedMemoryRow] = (
                    cache.rows[i] for i in cache.by_verse.get(target_verse, [])
                )
            else:
                row_iter = cache.rows

            for row in row_iter:
                if exclude_run_ids and str(row.run_id) in exclude_run_ids:
                    continue
                if tier_filter is not None and row.row_tier not in tier_filter:
                    continue
                if family_filter is not None and row.row_family not in family_filter:
                    continue
                if type_filter is not None and row.row_type not in type_filter:
                    continue
                if len(row.obs_vector) != q_dim:
                    continue

                raw_score = cosine_similarity(query_vec, row.obs_vector)
                t_ms = int(row.t_ms)

                if row.row_tier == "ltm":
                    row_decay_lambda = 0.0
                elif stm_decay_lambda is None:
                    row_decay_lambda = max(float(decay_lambda), float(cfg.stm_decay_lambda))
                else:
                    row_decay_lambda = max(0.0, float(stm_decay_lambda))

                score, recency_weight = apply_temporal_decay(
                    score=float(raw_score),
                    t_ms=t_ms,
                    decay_lambda=float(row_decay_lambda),
                    current_time_ms=current_time_ms,
                )
                if score < float(min_score):
                    continue

                match = ScenarioMatch(
                    score=float(score),
                    run_id=str(row.run_id),
                    episode_id=str(row.episode_id),
                    step_idx=int(row.step_idx),
                    t_ms=t_ms,
                    verse_name=str(row.verse_name),
                    action=row.action,
                    reward=float(row.reward),
                    obs=row.obs,
                    recency_weight=float(recency_weight),
                )
                if len(heap) < top_n:
                    heapq.heappush(heap, (float(score), ordinal, match))
                elif float(score) > heap[0][0]:
                    heapq.heapreplace(heap, (float(score), ordinal, match))
                ordinal += 1

        # Universal encoder fallback for cross-verse / cross-schema retrieval.
        # Only used when we still need candidates, or when raw-dimension match is unavailable.
        need_universal = bool((len(heap) < top_n) or (not raw_dim_available))
        if not need_universal:
            continue
        if q_u_dim <= 0:
            continue
        row_indices = list(cache.universal_row_indices or [])
        if not row_indices:
            continue
        u_mat = cache.universal_vectors
        for pos, row_idx in enumerate(row_indices):
            row = cache.rows[int(row_idx)]
            if raw_dim_available and len(row.obs_vector) == q_dim:
                # Skip raw-compatible rows; those already participated in native matching.
                continue
            if target_verse and row.verse_name != target_verse:
                continue
            if exclude_run_ids and str(row.run_id) in exclude_run_ids:
                continue
            if tier_filter is not None and row.row_tier not in tier_filter:
                continue
            if family_filter is not None and row.row_family not in family_filter:
                continue
            if type_filter is not None and row.row_type not in type_filter:
                continue
            if len(row.obs_vector_u) != q_u_dim:
                continue
            if np is not None and u_mat is not None:
                try:
                    if q_u_norm is not None:
                        raw_score = float(u_mat[pos] @ q_u_norm)
                    else:
                        raw_score = 0.0
                except Exception:
                    raw_score = cosine_similarity(query_vec_u, row.obs_vector_u)
            else:
                raw_score = cosine_similarity(query_vec_u, row.obs_vector_u)

            t_ms = int(row.t_ms)
            if row.row_tier == "ltm":
                row_decay_lambda = 0.0
            elif stm_decay_lambda is None:
                row_decay_lambda = max(float(decay_lambda), float(cfg.stm_decay_lambda))
            else:
                row_decay_lambda = max(0.0, float(stm_decay_lambda))
            score, recency_weight = apply_temporal_decay(
                score=float(raw_score),
                t_ms=t_ms,
                decay_lambda=float(row_decay_lambda),
                current_time_ms=current_time_ms,
            )
            if score < float(min_score):
                continue
            match = ScenarioMatch(
                score=float(score),
                run_id=str(row.run_id),
                episode_id=str(row.episode_id),
                step_idx=int(row.step_idx),
                t_ms=t_ms,
                verse_name=str(row.verse_name),
                action=row.action,
                reward=float(row.reward),
                obs=row.obs,
                recency_weight=float(recency_weight),
            )
            if len(heap) < top_n:
                heapq.heappush(heap, (float(score), ordinal, match))
            elif float(score) > heap[0][0]:
                heapq.heapreplace(heap, (float(score), ordinal, match))
            ordinal += 1

    heap.sort(key=lambda x: x[0], reverse=True)
    return [m for _, _, m in heap]


def _similarity_canary_path(cfg: CentralMemoryConfig) -> str:
    return os.path.join(cfg.root_dir, "similarity_canaries.json")


def save_similarity_canary(
    *,
    cfg: CentralMemoryConfig,
    canary_id: str,
    obs: JSONValue,
    expected_run_id: str,
    top_k: int = 1,
    verse_name: Optional[str] = None,
    memory_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    _ensure_repo(cfg)
    path = _similarity_canary_path(cfg)
    payload: Dict[str, Any] = {"version": "v1", "updated_at_ms": 0, "canaries": []}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                payload = obj
        except Exception:
            payload = {"version": "v1", "updated_at_ms": 0, "canaries": []}
    rows = payload.get("canaries")
    rows = rows if isinstance(rows, list) else []
    cid = str(canary_id).strip()
    if not cid:
        raise ValueError("canary_id cannot be empty")
    row = {
        "canary_id": cid,
        "obs": obs,
        "expected_run_id": str(expected_run_id),
        "top_k": max(1, int(top_k)),
        "verse_name": (str(verse_name).strip().lower() if verse_name else ""),
        "memory_types": sorted(str(x).strip().lower() for x in (memory_types or set()) if str(x).strip()),
        "updated_at_ms": int(time.time() * 1000),
    }
    replaced = False
    for i, old in enumerate(rows):
        if not isinstance(old, dict):
            continue
        if str(old.get("canary_id", "")).strip() == cid:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)
    payload["version"] = "v1"
    payload["updated_at_ms"] = int(time.time() * 1000)
    payload["canaries"] = rows
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return row


def run_similarity_canaries(
    *,
    cfg: CentralMemoryConfig,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    _ensure_repo(cfg)
    path = _similarity_canary_path(cfg)
    if not os.path.isfile(path):
        return {
            "path": path,
            "total": 0,
            "ann_hits": 0,
            "exact_hits": 0,
            "agreement_rate": 1.0,
            "pass_rate_ann": 1.0,
            "cases": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {
            "path": path,
            "total": 0,
            "ann_hits": 0,
            "exact_hits": 0,
            "agreement_rate": 0.0,
            "pass_rate_ann": 0.0,
            "cases": [],
            "error": "invalid_canary_file",
        }
    rows = payload.get("canaries") if isinstance(payload, dict) else None
    rows = rows if isinstance(rows, list) else []
    if limit is not None and int(limit) > 0:
        rows = rows[: int(limit)]

    total = 0
    ann_hits = 0
    exact_hits = 0
    agreements = 0
    cases: List[Dict[str, Any]] = []
    old_ann = os.environ.get("MULTIVERSE_SIM_USE_ANN")
    try:
        for row in rows:
            if not isinstance(row, dict):
                continue
            obs = row.get("obs")
            expected = str(row.get("expected_run_id", "")).strip()
            if not expected:
                continue
            total += 1
            top_k = max(1, _safe_int(row.get("top_k", 1), 1))
            verse_name = str(row.get("verse_name", "")).strip().lower() or None
            raw_types = row.get("memory_types")
            mt: Optional[Set[str]] = None
            if isinstance(raw_types, list):
                mt = set(str(x).strip().lower() for x in raw_types if str(x).strip())
                if not mt:
                    mt = None

            os.environ["MULTIVERSE_SIM_USE_ANN"] = "1"
            ann_rows = find_similar(
                obs=obs,
                cfg=cfg,
                top_k=top_k,
                verse_name=verse_name,
                min_score=-1.0,
                memory_types=mt,
            )
            ann_ids = [str(m.run_id) for m in ann_rows]
            ann_hit = expected in ann_ids
            if ann_hit:
                ann_hits += 1

            os.environ["MULTIVERSE_SIM_USE_ANN"] = "0"
            exact_rows = find_similar(
                obs=obs,
                cfg=cfg,
                top_k=top_k,
                verse_name=verse_name,
                min_score=-1.0,
                memory_types=mt,
            )
            exact_ids = [str(m.run_id) for m in exact_rows]
            exact_hit = expected in exact_ids
            if exact_hit:
                exact_hits += 1
            if ann_ids == exact_ids:
                agreements += 1

            cases.append(
                {
                    "canary_id": str(row.get("canary_id", "")),
                    "expected_run_id": expected,
                    "ann_hit": bool(ann_hit),
                    "exact_hit": bool(exact_hit),
                    "ann_top_run_id": (ann_ids[0] if ann_ids else ""),
                    "exact_top_run_id": (exact_ids[0] if exact_ids else ""),
                }
            )
    finally:
        if old_ann is None:
            os.environ.pop("MULTIVERSE_SIM_USE_ANN", None)
        else:
            os.environ["MULTIVERSE_SIM_USE_ANN"] = old_ann

    pass_rate_ann = float(ann_hits / float(max(1, total)))
    agreement_rate = float(agreements / float(max(1, total)))
    return {
        "path": path,
        "total": int(total),
        "ann_hits": int(ann_hits),
        "exact_hits": int(exact_hits),
        "pass_rate_ann": float(pass_rate_ann),
        "agreement_rate": float(agreement_rate),
        "cases": cases,
    }
