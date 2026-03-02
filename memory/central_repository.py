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
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from multiprocessing import Manager
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.types import JSONValue
from memory.central_repository_cache_support import (
    _PreparedMemoryRow,
    _SimilarityCacheDelta,
    _SimilarityCacheEntry,
    _build_cache_from_rows,
    _build_similarity_cache_for_path,
    _extract_or_build_universal_vector,
    _file_signature,
    _get_ann_index,
    _index_rows_by_verse,
    _legacy_simcache_path,
    _prepared_row_from_any,
    _prepared_row_to_dict,
    _simcache_path,
    _universal_obs_dim,
    _vectorize_rows_by_dim,
    _vectorize_universal_rows,
)
from memory.central_repository_support import (
    BackfillStats,
    CentralMemoryConfig,
    IngestStats,
    SanitizeStats,
    ScenarioMatch,
    _as_set,
    _dedupe_db_path,
    _dedupe_key,
    _dedupe_path,
    _enrich_memory_row,
    _json_key,
    _load_tier_policy,
    _ltm_memories_path,
    _ltm_priority_tuple,
    _memories_path,
    _memory_tier_for_event,
    _normalize_memory_tier,
    _policy_bool,
    _policy_float,
    _policy_int,
    _repo_lock_path,
    _row_verse_name,
    _safe_float,
    _safe_int,
    _stm_memories_path,
    _tier_policy_for_verse,
    _tier_policy_path,
)
from memory.central_repository_query_support import (
    _now_ms,
    find_similar_cached_support,
    run_similarity_canaries_support,
    save_similarity_canary_support,
)
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

# Multiprocess-safe locks and caches for ProcessPoolExecutor compatibility
# Use lazy initialization to avoid Manager() at module level (Windows compatibility)
_mp_manager: Optional[Any] = None
_SIM_CACHE_LOCK: Optional[Any] = None
_DEDUPE_READY_LOCK: Optional[Any] = None
_ANN_TUNE_LOCK: Optional[Any] = None
_REPO_LOCK: Optional[Any] = None


def _get_mp_locks():
    """Lazy initialization of multiprocess manager and locks (Windows-safe)."""
    global _mp_manager, _SIM_CACHE_LOCK, _DEDUPE_READY_LOCK, _ANN_TUNE_LOCK, _REPO_LOCK
    if _mp_manager is None:
        _mp_manager = Manager()
        _SIM_CACHE_LOCK = _mp_manager.Lock()
        _DEDUPE_READY_LOCK = _mp_manager.Lock()
        _ANN_TUNE_LOCK = _mp_manager.Lock()
        _REPO_LOCK = _mp_manager.Lock()
    return _SIM_CACHE_LOCK, _DEDUPE_READY_LOCK, _ANN_TUNE_LOCK, _REPO_LOCK


# Note: Keep _SIM_CACHE as OrderedDict (process-local) for move_to_end() LRU support
# The shared lock coordinates cache invalidation across processes
# Each process maintains its own cache copy, which is acceptable for memory retrieval
_SIM_CACHE: "OrderedDict[str, _SimilarityCacheEntry]" = OrderedDict()
# Phase 2.5: Incremental cache delta tracking
_CACHE_DELTAS: Dict[str, List[_SimilarityCacheDelta]] = {}
_DELTA_MERGE_THRESHOLD = int(os.environ.get("MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD", "1000"))
# Note: Keep _DEDUPE_READY as Set (process-local) for standard set operations
# The shared lock coordinates access across processes
_DEDUPE_READY: Set[str] = set()
# Note: These counters remain module-level (per-process) as they're for metrics only
_ANN_DYNAMIC_FACTOR: Optional[int] = None
_ANN_QUERY_COUNT = 0
_ANN_DRIFT_CHECKS = 0
_ANN_LAST_DRIFT = 0.0
_ANN_MAX_DRIFT = 0.0
# Lock timeout in seconds (configurable via environment variable)
_LOCK_TIMEOUT = int(os.environ.get("MULTIVERSE_MEMORY_LOCK_TIMEOUT", "30"))



def _atomic_write(file_path: str, content: str, max_retries: int = 3) -> None:
    """
    Atomic file write to prevent partial JSON corruption from concurrent writes.
    Uses temp file + rename for atomicity, with exponential backoff retry logic.

    Args:
        file_path: Target file path
        content: Content to write
        max_retries: Maximum number of retry attempts (default: 3)

    Raises:
        IOError: If all retry attempts fail
    """
    import tempfile

    for attempt in range(max_retries):
        try:
            # Write to temp file in same directory as target (ensures same filesystem)
            dir_name = os.path.dirname(file_path) or "."
            os.makedirs(dir_name, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=dir_name,
                delete=False,
                suffix='.tmp'
            ) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            # Atomic rename (overwrites target if exists)
            os.replace(tmp_path, file_path)
            return

        except Exception as exc:
            if attempt < max_retries - 1:
                # Exponential backoff: 100ms, 200ms, 400ms
                delay_ms = 100 * (2 ** attempt)
                time.sleep(delay_ms / 1000.0)
            else:
                raise IOError(f"Failed to atomic write after {max_retries} attempts: {exc}") from exc


@contextmanager
def _repo_lock(cfg: CentralMemoryConfig):
    """
    Multiprocess-safe repository lock with timeout.

    Uses both multiprocess.Manager lock (primary) and file lock (secondary defense).
    The multiprocess lock coordinates between ProcessPoolExecutor workers,
    while the file lock provides additional protection against external processes.

    Raises:
        TimeoutError: If lock cannot be acquired within _LOCK_TIMEOUT seconds
    """
    # Initialize locks if needed (lazy, Windows-safe)
    _, _, _, repo_lock = _get_mp_locks()

    os.makedirs(cfg.root_dir, exist_ok=True)
    lock_path = _repo_lock_path(cfg)

    # Primary: multiprocess lock with timeout
    mp_locked = repo_lock.acquire(timeout=_LOCK_TIMEOUT)
    if not mp_locked:
        raise TimeoutError(f"Failed to acquire multiprocess repo lock within {_LOCK_TIMEOUT}s")

    try:
        # Secondary: file lock for external process protection
        with open(lock_path, "a+b") as fh:
            file_locked = False
            try:
                if msvcrt is not None:
                    fh.seek(0, os.SEEK_END)
                    if fh.tell() <= 0:
                        fh.write(b"0")
                        fh.flush()
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    file_locked = True
                elif fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                    file_locked = True
                yield
            finally:
                if file_locked:
                    try:
                        fh.seek(0)
                        if msvcrt is not None:
                            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                        elif fcntl is not None:
                            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
    finally:
        repo_lock.release()


def _ensure_repo(cfg: CentralMemoryConfig) -> None:
    # Initialize locks if needed (lazy, Windows-safe)
    _, dedupe_ready_lock, _, _ = _get_mp_locks()

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
    with dedupe_ready_lock:
        ready = db_path in _DEDUPE_READY
    if not ready:
        conn = _open_dedupe_db(cfg)
        try:
            _migrate_legacy_dedupe_json_to_db(cfg, conn)
        finally:
            conn.close()
        with dedupe_ready_lock:
            _DEDUPE_READY.add(db_path)

def _open_dedupe_db(cfg: CentralMemoryConfig) -> sqlite3.Connection:
    """
    Open dedupe database connection with multiprocess-safe configuration.

    Returns:
        sqlite3.Connection with WAL mode and busy_timeout configured
    """
    db_path = _dedupe_db_path(cfg)
    conn = sqlite3.connect(db_path)

    # Configure for multiprocess access
    try:
        # WAL mode allows concurrent reads + 1 writer
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass

    try:
        # Faster syncing (still crash-safe with WAL)
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass

    try:
        # Critical: Wait up to 30s for lock instead of immediate failure
        # Prevents "database locked" errors with ProcessPoolExecutor
        busy_timeout_ms = int(os.environ.get("MULTIVERSE_MEMORY_LOCK_TIMEOUT", "30")) * 1000
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
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
    # Initialize locks if needed (lazy, Windows-safe)
    _, _, ann_tune_lock, _ = _get_mp_locks()

    factor_raw = os.environ.get("MULTIVERSE_SIM_ANN_FACTOR", "64")
    try:
        factor = max(1, int(factor_raw))
    except Exception:
        factor = 64
    with ann_tune_lock:
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
    # Initialize locks if needed (lazy, Windows-safe)
    _, _, ann_tune_lock, _ = _get_mp_locks()

    every = _ann_drift_check_every()
    with ann_tune_lock:
        _ANN_QUERY_COUNT += 1
        return bool((_ANN_QUERY_COUNT % every) == 0)


def _ann_record_drift(*, drift: float, n_rows: int, top_n: int) -> None:
    global _ANN_DRIFT_CHECKS, _ANN_LAST_DRIFT, _ANN_MAX_DRIFT, _ANN_DYNAMIC_FACTOR
    # Initialize locks if needed (lazy, Windows-safe)
    _, _, ann_tune_lock, _ = _get_mp_locks()

    d = max(0.0, float(drift))
    max_allowed = _ann_max_allowed_drift()
    with ann_tune_lock:
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
    # Initialize locks if needed (lazy, Windows-safe)
    _, _, ann_tune_lock, _ = _get_mp_locks()

    with ann_tune_lock:
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
    # Initialize locks if needed (lazy, Windows-safe)
    _, _, ann_tune_lock, _ = _get_mp_locks()

    with ann_tune_lock:
        _ANN_DYNAMIC_FACTOR = None
        _ANN_QUERY_COUNT = 0
        _ANN_DRIFT_CHECKS = 0
        _ANN_LAST_DRIFT = 0.0
        _ANN_MAX_DRIFT = 0.0


def _invalidate_similarity_cache(paths: Iterable[str]) -> None:
    # Initialize locks if needed (lazy, Windows-safe)
    sim_cache_lock, _, _, _ = _get_mp_locks()

    keys = set(os.path.abspath(str(p)) for p in paths if str(p))
    if not keys:
        return
    with sim_cache_lock:
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


def _append_cache_delta(apath: str, new_rows: List[_PreparedMemoryRow]) -> None:
    """
    Append new rows to delta list instead of full cache rebuild.

    This reduces cache rebuild overhead from O(n) to O(Δ) where Δ << n.
    Deltas are automatically merged when threshold is reached.
    """
    sim_cache_lock, _, _, _ = _get_mp_locks()

    if not new_rows:
        return

    # Create delta entry
    base_sig = (0, 0)
    with sim_cache_lock:
        if apath in _SIM_CACHE:
            base_sig = _SIM_CACHE[apath].signature

    delta = _SimilarityCacheDelta(
        base_signature=base_sig,
        delta_rows=new_rows,
        added_at_ms=int(time.time() * 1000),
    )

    # Append to delta list
    with sim_cache_lock:
        if apath not in _CACHE_DELTAS:
            _CACHE_DELTAS[apath] = []
        _CACHE_DELTAS[apath].append(delta)

        # Check if we should merge
        total_delta_rows = sum(len(d.delta_rows) for d in _CACHE_DELTAS[apath])
        should_merge = total_delta_rows >= _DELTA_MERGE_THRESHOLD

    # Merge if threshold reached (outside lock to avoid blocking)
    if should_merge:
        _merge_cache_deltas(apath)


def _merge_cache_deltas(apath: str) -> None:
    """
    Merge accumulated deltas into base cache snapshot.

    This is called automatically when delta threshold is reached,
    or can be called manually to force a merge.
    """
    sim_cache_lock, _, _, _ = _get_mp_locks()

    with sim_cache_lock:
        if apath not in _CACHE_DELTAS or not _CACHE_DELTAS[apath]:
            return

        # Get base cache (if exists)
        base_cache = _SIM_CACHE.get(apath)
        if base_cache is None:
            # No base cache - trigger full rebuild
            _CACHE_DELTAS[apath] = []
            return

        # Collect all delta rows
        all_delta_rows = []
        for delta in _CACHE_DELTAS[apath]:
            all_delta_rows.extend(delta.delta_rows)

        if not all_delta_rows:
            _CACHE_DELTAS[apath] = []
            return

        # Merge: base rows + delta rows
        merged_rows = base_cache.rows + all_delta_rows

    merged_cache = _build_cache_from_rows(merged_rows, base_cache.signature)

    # Atomic update
    with sim_cache_lock:
        _SIM_CACHE[apath] = merged_cache
        _CACHE_DELTAS[apath] = []

def _get_similarity_cache_for_path(
    *,
    mem_path: str,
    tier_policy: Optional[Dict[str, Any]],
) -> _SimilarityCacheEntry:
    # Initialize locks if needed (lazy, Windows-safe)
    sim_cache_lock, _, _, _ = _get_mp_locks()

    apath = os.path.abspath(mem_path)
    sig = _file_signature(apath)
    with sim_cache_lock:
        hit = _SIM_CACHE.get(apath)
        if hit is not None and hit.signature == sig:
            _SIM_CACHE.move_to_end(apath)
            return hit

    fresh = _build_similarity_cache_for_path(mem_path=apath, tier_policy=tier_policy)
    with sim_cache_lock:
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


# LRU cache with configurable size and TTL
_QUERY_RESULT_CACHE_SIZE = int(os.environ.get("MULTIVERSE_MEMORY_QUERY_CACHE_SIZE", "10000"))
_QUERY_RESULT_CACHE: Dict[str, Tuple[List[ScenarioMatch], int]] = {}  # key -> (results, timestamp_ms)
_CACHE_TTL_MS = int(os.environ.get("MULTIVERSE_MEMORY_QUERY_CACHE_TTL_MS", "60000"))  # 1 minute default


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
    trajectory_window: int = 0,
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
        
        def _extract_trajectory(row_input: Any) -> Optional[List[Dict[str, Any]]]:
            if int(trajectory_window) <= 0:
                return None
            ep_id = str(row_input.episode_id)
            root_step = int(row_input.step_idx)
            # Find the row in cache.rows if we just have the row reference.
            # O(N) scan but only on matched rows. Usually matches are small top_n.
            idx = -1
            for i, r in enumerate(cache.rows):
                if r is row_input:
                    idx = i
                    break
            if idx < 0:
                return []
            traj = []
            curr = idx
            while curr >= 0 and len(traj) < int(trajectory_window):
                r = cache.rows[curr]
                if str(r.episode_id) != ep_id or int(r.step_idx) > root_step:
                    if str(r.episode_id) != ep_id:
                        break
                    curr -= 1
                    continue
                traj.append({
                    "step_idx": int(r.step_idx),
                    "obs": r.obs,
                    "action": r.action,
                    "reward": float(r.reward)
                })
                curr -= 1
            traj.reverse()
            return traj

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
                    source_greedy_action=row.source_greedy_action,
                    source_action_matches_greedy=row.source_action_matches_greedy,
                    recency_weight=float(recency_weight),
                    trajectory=_extract_trajectory(row),
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
                    source_greedy_action=row.source_greedy_action,
                    source_action_matches_greedy=row.source_action_matches_greedy,
                    recency_weight=float(recency_weight),
                    trajectory=_extract_trajectory(row),
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
                source_greedy_action=row.source_greedy_action,
                source_action_matches_greedy=row.source_action_matches_greedy,
                recency_weight=float(recency_weight),
            )
            if len(heap) < top_n:
                heapq.heappush(heap, (float(score), ordinal, match))
            elif float(score) > heap[0][0]:
                heapq.heapreplace(heap, (float(score), ordinal, match))
            ordinal += 1

    heap.sort(key=lambda x: x[0], reverse=True)
    return [m for _, _, m in heap]


def find_similar_cached(
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
    trajectory_window: int = 0,
) -> List[ScenarioMatch]:
    """
    Cached version of find_similar() with LRU eviction and TTL.

    Cache key includes: obs, verse_name, top_k, min_score, memory filters, exclude_run_ids
    Does NOT cache: decay_lambda, current_time_ms (time-sensitive params)

    Cache behavior:
    - Hit: Return cached results if age < TTL (default 60s)
    - Miss: Compute via find_similar(), store in cache
    - Eviction: Remove oldest 10% when size exceeds limit (default 10K entries)

    Performance: ~1000× faster for cache hits (0.01ms vs 10ms)
    """
    return find_similar_cached_support(
        find_similar_fn=find_similar,
        query_cache=_QUERY_RESULT_CACHE,
        cache_ttl_ms=_CACHE_TTL_MS,
        cache_size=_QUERY_RESULT_CACHE_SIZE,
        obs=obs,
        cfg=cfg,
        top_k=top_k,
        verse_name=verse_name,
        min_score=min_score,
        exclude_run_ids=exclude_run_ids,
        decay_lambda=decay_lambda,
        current_time_ms=current_time_ms,
        memory_tiers=memory_tiers,
        memory_families=memory_families,
        memory_types=memory_types,
        stm_decay_lambda=stm_decay_lambda,
        trajectory_window=trajectory_window,
    )


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
    return save_similarity_canary_support(
        ensure_repo_fn=_ensure_repo,
        cfg=cfg,
        canary_id=canary_id,
        obs=obs,
        expected_run_id=expected_run_id,
        top_k=top_k,
        verse_name=verse_name,
        memory_types=memory_types,
    )


def run_similarity_canaries(
    *,
    cfg: CentralMemoryConfig,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    return run_similarity_canaries_support(
        ensure_repo_fn=_ensure_repo,
        find_similar_fn=find_similar,
        cfg=cfg,
        limit=limit,
    )
