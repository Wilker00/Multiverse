"""
memory/central_repository_state.py

Process-local and multiprocess-shared state for the central memory repository.

Extracted from central_repository.py to separate mutable runtime state
from the public API surface.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from multiprocessing import Manager
from typing import Any, Dict, List, Optional, Set, Tuple

from memory.central_repository_cache_support import (
    _SimilarityCacheDelta,
    _SimilarityCacheEntry,
)
from memory.central_repository_support import ScenarioMatch

# ---------------------------------------------------------------------------
# Multiprocess manager and locks (lazy-initialized for Windows compatibility)
# ---------------------------------------------------------------------------
_mp_manager: Optional[Any] = None
_SIM_CACHE_LOCK: Optional[Any] = None
_DEDUPE_READY_LOCK: Optional[Any] = None
_ANN_TUNE_LOCK: Optional[Any] = None
_REPO_LOCK: Optional[Any] = None


def get_mp_locks():
    """Lazy initialization of multiprocess manager and locks (Windows-safe)."""
    global _mp_manager, _SIM_CACHE_LOCK, _DEDUPE_READY_LOCK, _ANN_TUNE_LOCK, _REPO_LOCK
    if _mp_manager is None:
        _mp_manager = Manager()
        _SIM_CACHE_LOCK = _mp_manager.Lock()
        _DEDUPE_READY_LOCK = _mp_manager.Lock()
        _ANN_TUNE_LOCK = _mp_manager.Lock()
        _REPO_LOCK = _mp_manager.Lock()
    return _SIM_CACHE_LOCK, _DEDUPE_READY_LOCK, _ANN_TUNE_LOCK, _REPO_LOCK


# ---------------------------------------------------------------------------
# Process-local caches
# ---------------------------------------------------------------------------

# Similarity cache: OrderedDict for move_to_end() LRU support.
# Each process maintains its own cache copy; the shared lock coordinates
# invalidation across processes.
SIM_CACHE: "OrderedDict[str, _SimilarityCacheEntry]" = OrderedDict()

# Incremental cache delta tracking (Phase 2.5)
CACHE_DELTAS: Dict[str, List[_SimilarityCacheDelta]] = {}
DELTA_MERGE_THRESHOLD = int(
    os.environ.get("MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD", "1000")
)

# Dedupe readiness tracker (process-local set, shared lock)
DEDUPE_READY: Set[str] = set()

# ANN runtime metrics (process-local counters exposed via wrapper helpers)
ANN_RUNTIME_STATE: Dict[str, Any] = {
    "dynamic_factor": None,
    "query_count": 0,
    "drift_checks": 0,
    "last_drift": 0.0,
    "max_drift": 0.0,
}

# Lock timeout in seconds
LOCK_TIMEOUT = int(os.environ.get("MULTIVERSE_MEMORY_LOCK_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Query result cache (LRU with TTL)
# ---------------------------------------------------------------------------
QUERY_RESULT_CACHE_SIZE = int(
    os.environ.get("MULTIVERSE_MEMORY_QUERY_CACHE_SIZE", "10000")
)
QUERY_RESULT_CACHE: Dict[str, Tuple[List[ScenarioMatch], int]] = {}
CACHE_TTL_MS = int(
    os.environ.get("MULTIVERSE_MEMORY_QUERY_CACHE_TTL_MS", "60000")
)

# ---------------------------------------------------------------------------
# Platform-specific file locking modules
# ---------------------------------------------------------------------------
try:
    import msvcrt as _msvcrt_module  # type: ignore
except Exception:  # pragma: no cover
    _msvcrt_module = None

try:
    import fcntl as _fcntl_module  # type: ignore
except Exception:  # pragma: no cover
    _fcntl_module = None

msvcrt_mod = _msvcrt_module
fcntl_mod = _fcntl_module
