"""
Runtime cache ownership and invalidation helpers for the central memory repository.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

from memory.central_repository_cache_support import _PreparedMemoryRow, _SimilarityCacheDelta


def invalidate_similarity_cache_support(
    *,
    paths: Iterable[str],
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
    sim_cache: Any,
    simcache_path_fn: Callable[[str], str],
    legacy_simcache_path_fn: Callable[[str], str],
) -> None:
    sim_cache_lock, _, _, _ = get_mp_locks_fn()

    keys = set(os.path.abspath(str(path)) for path in paths if str(path))
    if not keys:
        return
    with sim_cache_lock:
        for key in list(sim_cache.keys()):
            if key in keys:
                sim_cache.pop(key, None)
    for key in keys:
        for sidecar_path in (simcache_path_fn(key), legacy_simcache_path_fn(key)):
            try:
                if os.path.isfile(sidecar_path):
                    os.remove(sidecar_path)
            except Exception:
                pass


def merge_cache_deltas_support(
    *,
    apath: str,
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
    sim_cache: Dict[str, Any],
    cache_deltas: Dict[str, List[_SimilarityCacheDelta]],
    build_cache_from_rows_fn: Callable[[List[_PreparedMemoryRow], Tuple[int, int]], Any],
) -> None:
    sim_cache_lock, _, _, _ = get_mp_locks_fn()

    with sim_cache_lock:
        if apath not in cache_deltas or not cache_deltas[apath]:
            return

        base_cache = sim_cache.get(apath)
        if base_cache is None:
            cache_deltas[apath] = []
            return

        all_delta_rows: List[_PreparedMemoryRow] = []
        for delta in cache_deltas[apath]:
            all_delta_rows.extend(delta.delta_rows)

        if not all_delta_rows:
            cache_deltas[apath] = []
            return

        merged_rows = base_cache.rows + all_delta_rows
        base_signature = base_cache.signature

    merged_cache = build_cache_from_rows_fn(merged_rows, base_signature)

    with sim_cache_lock:
        sim_cache[apath] = merged_cache
        cache_deltas[apath] = []


def append_cache_delta_support(
    *,
    apath: str,
    new_rows: List[_PreparedMemoryRow],
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
    sim_cache: Dict[str, Any],
    cache_deltas: Dict[str, List[_SimilarityCacheDelta]],
    delta_merge_threshold: int,
    merge_cache_deltas_fn: Callable[[str], None],
    similarity_cache_delta_cls: Any,
) -> None:
    if not new_rows:
        return

    sim_cache_lock, _, _, _ = get_mp_locks_fn()
    base_sig = (0, 0)
    with sim_cache_lock:
        if apath in sim_cache:
            base_sig = sim_cache[apath].signature

    delta = similarity_cache_delta_cls(
        base_signature=base_sig,
        delta_rows=new_rows,
        added_at_ms=int(time.time() * 1000),
    )

    with sim_cache_lock:
        if apath not in cache_deltas:
            cache_deltas[apath] = []
        cache_deltas[apath].append(delta)
        total_delta_rows = sum(len(item.delta_rows) for item in cache_deltas[apath])
        should_merge = total_delta_rows >= delta_merge_threshold

    if should_merge:
        merge_cache_deltas_fn(apath)


def get_similarity_cache_for_path_support(
    *,
    mem_path: str,
    tier_policy: Dict[str, Any] | None,
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
    sim_cache: Any,
    file_signature_fn: Callable[[str], Tuple[int, int]],
    build_similarity_cache_for_path_fn: Callable[..., Any],
    sim_cache_limit_fn: Callable[[], int],
) -> Any:
    sim_cache_lock, _, _, _ = get_mp_locks_fn()

    apath = os.path.abspath(mem_path)
    signature = file_signature_fn(apath)
    with sim_cache_lock:
        hit = sim_cache.get(apath)
        if hit is not None and hit.signature == signature:
            sim_cache.move_to_end(apath)
            return hit

    fresh = build_similarity_cache_for_path_fn(mem_path=apath, tier_policy=tier_policy)
    with sim_cache_lock:
        sim_cache[apath] = fresh
        sim_cache.move_to_end(apath)
        while len(sim_cache) > sim_cache_limit_fn():
            sim_cache.popitem(last=False)
    return fresh
