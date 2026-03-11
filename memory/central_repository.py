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

import os
import sqlite3
from collections import OrderedDict
from multiprocessing import Manager
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.types import JSONValue
from memory.central_repository_cache_support import (
    _PreparedMemoryRow,
    _SimilarityCacheDelta,
    _SimilarityCacheEntry,
    _build_cache_from_rows,
    _build_similarity_cache_for_path,
    _file_signature,
    _legacy_simcache_path,
    _simcache_path,
    _universal_obs_dim,
)
from memory.central_repository_cache_runtime_support import (
    append_cache_delta_support,
    get_similarity_cache_for_path_support,
    invalidate_similarity_cache_support,
    merge_cache_deltas_support,
)
from memory.central_repository_ingest_support import (
    backfill_memory_metadata_support,
    dedupe_try_reserve_support,
    ingest_run_support,
    iter_events_support,
    load_dedupe_index_support,
    load_events_support,
    migrate_legacy_dedupe_json_to_db_support,
    open_dedupe_db_support,
    sanitize_memory_file_support,
    save_dedupe_index_support,
)
from memory.central_repository_support import (
    BackfillStats,
    CentralMemoryConfig,
    IngestStats,
    SanitizeStats,
    ScenarioMatch,
    _as_set,
    _dedupe_db_path,
    _dedupe_path,
    _load_tier_policy,
    _ltm_memories_path,
    _memories_path,
    _repo_lock_path,
    _stm_memories_path,
)
from memory.central_repository_query_support import (
    find_similar_cached_support,
    run_similarity_canaries_support,
    save_similarity_canary_support,
)
from memory.central_repository_runtime_support import (
    atomic_write_support,
    ensure_repo_support,
    repo_lock_support,
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
from memory.selection import SelectionConfig

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
    atomic_write_support(file_path=file_path, content=content, max_retries=max_retries)


def _repo_lock(cfg: CentralMemoryConfig):
    return repo_lock_support(
        cfg=cfg,
        get_mp_locks_fn=_get_mp_locks,
        repo_lock_path_fn=_repo_lock_path,
        lock_timeout=_LOCK_TIMEOUT,
        msvcrt_module=msvcrt,
        fcntl_module=fcntl,
    )


def _ensure_repo(cfg: CentralMemoryConfig) -> None:
    ensure_repo_support(
        cfg=cfg,
        get_mp_locks_fn=_get_mp_locks,
        memories_path_fn=_memories_path,
        ltm_memories_path_fn=_ltm_memories_path,
        stm_memories_path_fn=_stm_memories_path,
        dedupe_path_fn=_dedupe_path,
        dedupe_db_path_fn=_dedupe_db_path,
        dedupe_ready=_DEDUPE_READY,
        atomic_write_fn=_atomic_write,
        open_dedupe_db_fn=_open_dedupe_db,
        migrate_legacy_dedupe_json_to_db_fn=_migrate_legacy_dedupe_json_to_db,
    )


def _open_dedupe_db(cfg: CentralMemoryConfig) -> sqlite3.Connection:
    return open_dedupe_db_support(cfg)


def _migrate_legacy_dedupe_json_to_db(cfg: CentralMemoryConfig, conn: sqlite3.Connection) -> None:
    migrate_legacy_dedupe_json_to_db_support(cfg=cfg, conn=conn)


def _dedupe_try_reserve(conn: sqlite3.Connection, key: str) -> bool:
    return dedupe_try_reserve_support(conn=conn, key=key)


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
    return load_dedupe_index_support(
        cfg=cfg,
        ensure_repo_fn=_ensure_repo,
        open_dedupe_db_fn=_open_dedupe_db,
        migrate_legacy_dedupe_json_to_db_fn=_migrate_legacy_dedupe_json_to_db,
    )


def _save_dedupe_index(cfg: CentralMemoryConfig, keys: Iterable[str]) -> None:
    save_dedupe_index_support(
        cfg=cfg,
        keys=keys,
        ensure_repo_fn=_ensure_repo,
        open_dedupe_db_fn=_open_dedupe_db,
    )


def _iter_events(events_path: str):
    return iter_events_support(events_path)


def _load_events(events_path: str) -> List[Dict[str, Any]]:
    return load_events_support(events_path)


def sanitize_memory_file(cfg: CentralMemoryConfig) -> SanitizeStats:
    return sanitize_memory_file_support(
        cfg=cfg,
        ensure_repo_fn=_ensure_repo,
        repo_lock_fn=_repo_lock,
        invalidate_similarity_cache_fn=_invalidate_similarity_cache,
    )


def backfill_memory_metadata(
    *,
    cfg: CentralMemoryConfig,
    rebuild_tier_files: bool = True,
    recompute_tier: bool = False,
    apply_support_guards: bool = True,
) -> BackfillStats:
    return backfill_memory_metadata_support(
        cfg=cfg,
        rebuild_tier_files=rebuild_tier_files,
        recompute_tier=recompute_tier,
        apply_support_guards=apply_support_guards,
        ensure_repo_fn=_ensure_repo,
        repo_lock_fn=_repo_lock,
        invalidate_similarity_cache_fn=_invalidate_similarity_cache,
    )


def ingest_run(
    *,
    run_dir: str,
    cfg: CentralMemoryConfig,
    selection: Optional[SelectionConfig] = None,
    max_events: Optional[int] = None,
) -> IngestStats:
    return ingest_run_support(
        run_dir=run_dir,
        cfg=cfg,
        selection=selection,
        max_events=max_events,
        ensure_repo_fn=_ensure_repo,
        repo_lock_fn=_repo_lock,
        open_dedupe_db_fn=_open_dedupe_db,
        migrate_legacy_dedupe_json_to_db_fn=_migrate_legacy_dedupe_json_to_db,
        dedupe_try_reserve_fn=_dedupe_try_reserve,
        invalidate_similarity_cache_fn=_invalidate_similarity_cache,
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
