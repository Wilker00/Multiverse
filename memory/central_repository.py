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
    _index_rows_by_verse,
    _legacy_simcache_path,
    _prepared_row_from_any,
    _prepared_row_to_dict,
    _simcache_path,
    _universal_obs_dim,
    _vectorize_rows_by_dim,
    _vectorize_universal_rows,
)
from memory.central_repository_cache_runtime_support import (
    append_cache_delta_support,
    get_similarity_cache_for_path_support,
    invalidate_similarity_cache_support,
    merge_cache_deltas_support,
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
from memory.central_repository_similarity_support import (
    _ann_candidate_count_support,
    _ann_drift_check_every_support,
    _ann_enabled_support,
    _ann_max_allowed_drift_support,
    _ann_record_drift_support,
    _ann_should_check_drift_support,
    _sim_cache_limit_support,
    find_similar_support,
    get_similarity_runtime_metrics_support,
    reset_similarity_runtime_metrics_support,
)
from memory.embeddings import obs_to_vector, project_vector
from memory.selection import SelectionConfig, select_events
from memory.task_taxonomy import memory_family_for_verse, memory_type_for_verse, tags_for_verse

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
# Note: These counters remain process-local and are exposed through wrapper helpers.
_ANN_RUNTIME_STATE: Dict[str, Any] = {
    "dynamic_factor": None,
    "query_count": 0,
    "drift_checks": 0,
    "last_drift": 0.0,
    "max_drift": 0.0,
}
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
    return _sim_cache_limit_support()


def _ann_enabled() -> bool:
    return _ann_enabled_support()


def _ann_candidate_count(top_n: int, n_rows: int) -> int:
    return _ann_candidate_count_support(
        top_n=top_n,
        n_rows=n_rows,
        ann_dynamic_factor=_ANN_RUNTIME_STATE.get("dynamic_factor"),
        get_mp_locks_fn=_get_mp_locks,
    )


def _ann_drift_check_every() -> int:
    return _ann_drift_check_every_support()


def _ann_max_allowed_drift() -> float:
    return _ann_max_allowed_drift_support()


def _ann_should_check_drift() -> bool:
    return _ann_should_check_drift_support(
        runtime_state=_ANN_RUNTIME_STATE,
        get_mp_locks_fn=_get_mp_locks,
        ann_drift_check_every_fn=_ann_drift_check_every,
    )


def _ann_record_drift(*, drift: float, n_rows: int, top_n: int) -> None:
    _ann_record_drift_support(
        runtime_state=_ANN_RUNTIME_STATE,
        get_mp_locks_fn=_get_mp_locks,
        ann_max_allowed_drift_fn=_ann_max_allowed_drift,
        drift=drift,
        n_rows=n_rows,
        top_n=top_n,
    )


def get_similarity_runtime_metrics() -> Dict[str, Any]:
    return get_similarity_runtime_metrics_support(
        runtime_state=_ANN_RUNTIME_STATE,
        get_mp_locks_fn=_get_mp_locks,
        ann_drift_check_every_fn=_ann_drift_check_every,
        ann_max_allowed_drift_fn=_ann_max_allowed_drift,
    )


def reset_similarity_runtime_metrics() -> None:
    reset_similarity_runtime_metrics_support(
        runtime_state=_ANN_RUNTIME_STATE,
        get_mp_locks_fn=_get_mp_locks,
    )


def _invalidate_similarity_cache(paths: Iterable[str]) -> None:
    invalidate_similarity_cache_support(
        paths=paths,
        get_mp_locks_fn=_get_mp_locks,
        sim_cache=_SIM_CACHE,
        simcache_path_fn=_simcache_path,
        legacy_simcache_path_fn=_legacy_simcache_path,
    )


def _append_cache_delta(apath: str, new_rows: List[_PreparedMemoryRow]) -> None:
    """
    Append new rows to delta list instead of full cache rebuild.

    This reduces cache rebuild overhead from O(n) to O(delta) where delta << n.
    Deltas are automatically merged when threshold is reached.
    """
    append_cache_delta_support(
        apath=apath,
        new_rows=new_rows,
        get_mp_locks_fn=_get_mp_locks,
        sim_cache=_SIM_CACHE,
        cache_deltas=_CACHE_DELTAS,
        delta_merge_threshold=_DELTA_MERGE_THRESHOLD,
        merge_cache_deltas_fn=_merge_cache_deltas,
        similarity_cache_delta_cls=_SimilarityCacheDelta,
    )


def _merge_cache_deltas(apath: str) -> None:
    """
    Merge accumulated deltas into base cache snapshot.

    This is called automatically when delta threshold is reached,
    or can be called manually to force a merge.
    """
    merge_cache_deltas_support(
        apath=apath,
        get_mp_locks_fn=_get_mp_locks,
        sim_cache=_SIM_CACHE,
        cache_deltas=_CACHE_DELTAS,
        build_cache_from_rows_fn=_build_cache_from_rows,
    )

def _get_similarity_cache_for_path(
    *,
    mem_path: str,
    tier_policy: Optional[Dict[str, Any]],
) -> _SimilarityCacheEntry:
    return get_similarity_cache_for_path_support(
        mem_path=mem_path,
        tier_policy=tier_policy,
        get_mp_locks_fn=_get_mp_locks,
        sim_cache=_SIM_CACHE,
        file_signature_fn=_file_signature,
        build_similarity_cache_for_path_fn=_build_similarity_cache_for_path,
        sim_cache_limit_fn=_sim_cache_limit,
    )


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
    return find_similar_support(
        ensure_repo_fn=_ensure_repo,
        get_similarity_cache_for_path_fn=_get_similarity_cache_for_path,
        memories_path_fn=_memories_path,
        ltm_memories_path_fn=_ltm_memories_path,
        stm_memories_path_fn=_stm_memories_path,
        load_tier_policy_fn=_load_tier_policy,
        as_set_fn=_as_set,
        universal_obs_dim_fn=_universal_obs_dim,
        ann_enabled_fn=_ann_enabled,
        ann_candidate_count_fn=_ann_candidate_count,
        ann_should_check_drift_fn=_ann_should_check_drift,
        ann_record_drift_fn=_ann_record_drift,
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
