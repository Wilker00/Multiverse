# Phase 2 Performance Optimization - Implementation Guide

**Status:** Phase 2.1-2.3 Complete ✅ | Phase 2.4-2.6 In Progress ⏳
**Branch:** `feature/phase2-performance-optimizations`
**Target:** 17× query speedup (85ms → 5ms P50 latency)

---

## ✅ Completed (Already Merged to Main)

### Phase 1: Thread-Safety Hardening
- ✅ Multiprocess-safe locks (Manager().Lock())
- ✅ Atomic file writes (temp + rename)
- ✅ Enhanced `_repo_lock()` with 30s timeout
- ✅ SQLite `busy_timeout=30000`
- ✅ **291 tests passing, 0 failures**

### Phase 2.1-2.3: Performance Infrastructure
- ✅ FAISSVectorStore with auto-selection (Flat <10K, IVF ≥10K)
- ✅ ResilientVectorStore with fallback chain
- ✅ Error handling for Pinecone/Milvus
- ✅ Logging and metrics tracking

---

## ⏳ Remaining Work (Phase 2.4-2.6)

### Phase 2.4: Batch Query API

**Goal:** Enable vectorized operations for 15-20× speedup vs sequential calls

**Location:** `memory/central_repository.py`

**Implementation:**

```python
def find_similar_batch(
    *,
    observations: List[JSONValue],
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
) -> List[List[ScenarioMatch]]:
    """
    Batch query for similar observations.

    Provides 15-20× speedup vs sequential find_similar() calls by:
    - Loading cache once instead of N times
    - Vectorizing all queries into single numpy matrix
    - Performing batch matrix multiplication (queries × memories)
    - Reusing normalization and filtering operations

    Args:
        observations: List of observations to query (batch size)
        ... (same parameters as find_similar)

    Returns:
        List of result lists, one per input observation
    """
    if not observations:
        return []

    _ensure_repo(cfg)

    # Vectorize all queries upfront
    query_vecs = [obs_to_vector(obs) for obs in observations]
    query_vecs_u = [obs_to_universal_vector(obs, dim=_universal_obs_dim()) for obs in observations]

    # Load cache once (shared across all queries)
    mem_paths = _determine_memory_paths(cfg, memory_tiers)
    tier_policy = _load_tier_policy(cfg)

    results = []
    for mem_path in mem_paths:
        if not os.path.isfile(mem_path):
            continue

        cache = _get_similarity_cache_for_path(mem_path=mem_path, tier_policy=tier_policy)

        # Batch process all queries at once
        if np is not None and len(query_vecs) > 0:
            # Stack queries into matrix: (n_queries × dim)
            Q = np.array(query_vecs, dtype=np.float32)

            # Normalize query matrix
            Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
            Q_normalized = Q / np.maximum(Q_norms, 1e-12)

            # Batch matrix multiplication: (n_queries × dim) @ (dim × n_memories)
            # Result shape: (n_queries × n_memories)
            similarities = Q_normalized @ cache.vectors_by_dim[q_dim].T

            # Extract top-k for each query
            for i, obs in enumerate(observations):
                top_indices = np.argsort(similarities[i])[-top_k:][::-1]
                matches = [_build_scenario_match(cache.rows[idx], ...) for idx in top_indices]
                results.append(matches)
        else:
            # Fallback: process sequentially
            for obs in observations:
                matches = find_similar(obs=obs, cfg=cfg, top_k=top_k, ...)
                results.append(matches)

    return results
```

**Testing:**

```python
# tests/test_memory_performance.py
def test_batch_query_speedup():
    """Verify batch query is >15× faster than sequential."""
    obs_list = [generate_test_obs() for _ in range(100)]

    # Sequential timing
    start = time.time()
    sequential_results = [find_similar(obs=obs, ...) for obs in obs_list]
    sequential_time = time.time() - start

    # Batch timing
    start = time.time()
    batch_results = find_similar_batch(observations=obs_list, ...)
    batch_time = time.time() - start

    speedup = sequential_time / batch_time
    assert speedup > 15, f"Batch speedup only {speedup:.1f}× (target: 15×)"
    assert batch_results == sequential_results  # Verify correctness
```

---

### Phase 2.5: Incremental Cache Updates

**Goal:** Reduce cache rebuild overhead from O(n) to O(Δ)

**Location:** `memory/central_repository.py` - Modify `_build_similarity_cache_for_path()`

**Current Problem:**
- Single memory write → entire 50MB+ cache rebuilt from scratch
- Takes ~4.2 seconds for 52K rows
- N workers × M invalidations = excessive I/O

**Solution: Delta Tracking**

```python
@dataclass
class _SimilarityCacheDelta:
    """Incremental cache update tracking."""
    base_snapshot_signature: str
    base_snapshot_row_count: int
    delta_rows: List[_PreparedMemoryRow]
    delta_added_at_ms: int

# Module-level delta storage
_CACHE_DELTAS: Dict[str, List[_SimilarityCacheDelta]] = {}
_DELTA_MERGE_THRESHOLD = int(os.environ.get("MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD", "1000"))

def _append_cache_delta(apath: str, new_rows: List[_PreparedMemoryRow]) -> None:
    """Append new rows to delta list instead of full rebuild."""
    if apath not in _CACHE_DELTAS:
        _CACHE_DELTAS[apath] = []

    delta = _SimilarityCacheDelta(
        base_snapshot_signature=_file_signature(apath),
        base_snapshot_row_count=len(_SIM_CACHE[apath].rows) if apath in _SIM_CACHE else 0,
        delta_rows=new_rows,
        delta_added_at_ms=now_ms()
    )
    _CACHE_DELTAS[apath].append(delta)

    # Merge deltas when threshold reached
    total_delta_rows = sum(len(d.delta_rows) for d in _CACHE_DELTAS[apath])
    if total_delta_rows >= _DELTA_MERGE_THRESHOLD:
        _merge_cache_deltas(apath)

def _merge_cache_deltas(apath: str) -> None:
    """Merge deltas into base snapshot (background thread safe)."""
    if apath not in _SIM_CACHE or apath not in _CACHE_DELTAS:
        return

    base = _SIM_CACHE[apath]
    deltas = _CACHE_DELTAS[apath]

    # Merge all delta rows into base
    all_delta_rows = []
    for d in deltas:
        all_delta_rows.extend(d.delta_rows)

    merged_rows = base.rows + all_delta_rows

    # Rebuild vectors with merged rows
    merged_cache = _build_cache_from_rows(merged_rows)

    # Atomic update
    with _SIM_CACHE_LOCK:
        _SIM_CACHE[apath] = merged_cache
        _CACHE_DELTAS[apath] = []

    logger.info(f"Merged {len(all_delta_rows)} delta rows into cache: {apath}")
```

**Background Merge Thread:**

```python
import threading
import time

def _start_delta_merge_thread():
    """Background thread that merges deltas every 60 seconds."""
    def merge_loop():
        while True:
            time.sleep(60)
            with _SIM_CACHE_LOCK:
                paths_to_merge = list(_CACHE_DELTAS.keys())

            for apath in paths_to_merge:
                try:
                    _merge_cache_deltas(apath)
                except Exception as exc:
                    logger.error(f"Delta merge failed for {apath}: {exc}")

    thread = threading.Thread(target=merge_loop, daemon=True)
    thread.start()

# Start on module load
_start_delta_merge_thread()
```

**Testing:**

```python
def test_incremental_cache_updates():
    """Verify delta tracking reduces rebuild overhead."""
    # Initial cache build
    initial_cache = _build_similarity_cache_for_path(...)

    # Add 500 new rows (below 1000 threshold)
    new_rows = [create_test_row() for _ in range(500)]
    _append_cache_delta(apath, new_rows)

    # Verify delta list exists
    assert len(_CACHE_DELTAS[apath]) == 1
    assert len(_CACHE_DELTAS[apath][0].delta_rows) == 500

    # Add 600 more rows (triggers merge at 1100 total)
    more_rows = [create_test_row() for _ in range(600)]
    _append_cache_delta(apath, more_rows)

    # Verify merge occurred
    assert len(_CACHE_DELTAS[apath]) == 0  # Deltas cleared after merge
    assert len(_SIM_CACHE[apath].rows) == initial_count + 1100
```

---

### Phase 2.6: LRU Query Result Cache

**Goal:** Cache query results to avoid redundant similarity computations

**Location:** `memory/central_repository.py` - Wrap `find_similar()`

**Implementation:**

```python
from functools import lru_cache
import hashlib

def _query_cache_key(
    obs: JSONValue,
    verse_name: Optional[str],
    top_k: int,
    memory_tiers: Optional[Set[str]],
    memory_families: Optional[Set[str]],
    memory_types: Optional[Set[str]]
) -> str:
    """Generate cache key from query parameters."""
    key_parts = [
        json.dumps(obs, sort_keys=True),
        str(verse_name or ""),
        str(top_k),
        str(sorted(memory_tiers or [])),
        str(sorted(memory_families or [])),
        str(sorted(memory_types or []))
    ]
    key_str = "||".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()

# LRU cache with configurable size
_QUERY_RESULT_CACHE_SIZE = int(os.environ.get("MULTIVERSE_MEMORY_QUERY_CACHE_SIZE", "10000"))
_QUERY_RESULT_CACHE: Dict[str, Tuple[List[ScenarioMatch], int]] = {}  # key -> (results, timestamp_ms)
_CACHE_TTL_MS = 60000  # 1 minute TTL

def find_similar_cached(
    *,
    obs: JSONValue,
    cfg: CentralMemoryConfig,
    top_k: int = 5,
    verse_name: Optional[str] = None,
    memory_tiers: Optional[Set[str]] = None,
    memory_families: Optional[Set[str]] = None,
    memory_types: Optional[Set[str]] = None,
    **kwargs
) -> List[ScenarioMatch]:
    """Cached version of find_similar() with LRU eviction."""
    # Generate cache key
    cache_key = _query_cache_key(obs, verse_name, top_k, memory_tiers, memory_families, memory_types)

    # Check cache
    if cache_key in _QUERY_RESULT_CACHE:
        results, cached_at_ms = _QUERY_RESULT_CACHE[cache_key]
        age_ms = now_ms() - cached_at_ms

        if age_ms < _CACHE_TTL_MS:
            logger.debug(f"Query cache hit (age: {age_ms}ms)")
            return results
        else:
            # TTL expired, remove from cache
            del _QUERY_RESULT_CACHE[cache_key]

    # Cache miss - compute results
    results = find_similar(
        obs=obs,
        cfg=cfg,
        top_k=top_k,
        verse_name=verse_name,
        memory_tiers=memory_tiers,
        memory_families=memory_families,
        memory_types=memory_types,
        **kwargs
    )

    # Store in cache with LRU eviction
    _QUERY_RESULT_CACHE[cache_key] = (results, now_ms())

    # Evict oldest entries if over limit
    if len(_QUERY_RESULT_CACHE) > _QUERY_RESULT_CACHE_SIZE:
        # Sort by timestamp and remove oldest 10%
        sorted_items = sorted(_QUERY_RESULT_CACHE.items(), key=lambda x: x[1][1])
        evict_count = len(sorted_items) // 10
        for key, _ in sorted_items[:evict_count]:
            del _QUERY_RESULT_CACHE[key]
        logger.debug(f"Evicted {evict_count} old query cache entries")

    return results
```

**Testing:**

```python
def test_query_result_cache_hit_rate():
    """Verify query cache provides speedup on repeated queries."""
    obs = generate_test_obs()

    # First query (cache miss)
    start = time.time()
    results1 = find_similar_cached(obs=obs, cfg=cfg, top_k=5)
    miss_time = time.time() - start

    # Second identical query (cache hit)
    start = time.time()
    results2 = find_similar_cached(obs=obs, cfg=cfg, top_k=5)
    hit_time = time.time() - start

    # Cache hit should be >100× faster
    speedup = miss_time / hit_time
    assert speedup > 100, f"Cache speedup only {speedup:.1f}×"
    assert results1 == results2  # Results must match
```

---

## Performance Benchmarks

**Create:** `tests/test_memory_performance.py`

```python
import pytest
import time
import numpy as np
from memory.central_repository import find_similar, find_similar_batch
from memory.vector_store import FAISSVectorStore, VectorRecord

class TestMemoryPerformance:
    """Performance benchmark suite for Phase 2."""

    @pytest.fixture
    def large_memory_bank(self):
        """Create 52K vector memory bank (production scale)."""
        vectors = np.random.rand(52000, 10).astype(np.float32)
        records = [
            VectorRecord(
                vector_id=f"mem_{i}",
                vector=vectors[i].tolist(),
                metadata={"episode_id": f"ep_{i//100}", "step": i%100}
            )
            for i in range(52000)
        ]
        return records

    def test_faiss_query_latency_p50_p99(self, large_memory_bank):
        """Verify FAISS query latency meets targets."""
        store = FAISSVectorStore(dimension=10, auto_select=True)
        store.add(large_memory_bank)

        # Run 1000 queries
        query_times = []
        for _ in range(1000):
            query_vec = np.random.rand(10).tolist()
            start = time.time()
            results = store.query(query_vec, top_k=5)
            query_times.append((time.time() - start) * 1000)  # ms

        p50 = np.percentile(query_times, 50)
        p99 = np.percentile(query_times, 99)

        print(f"FAISS Query Latency - P50: {p50:.2f}ms, P99: {p99:.2f}ms")

        assert p50 < 5.0, f"P50 latency {p50:.2f}ms exceeds 5ms target"
        assert p99 < 20.0, f"P99 latency {p99:.2f}ms exceeds 20ms target"

    def test_batch_vs_sequential_speedup(self, large_memory_bank):
        """Verify batch API achieves 17× speedup target."""
        # Setup
        observations = [{"x": float(i), "y": float(i+1)} for i in range(100)]

        # Sequential timing
        start = time.time()
        sequential_results = [find_similar(obs=obs, cfg=cfg, top_k=5) for obs in observations]
        sequential_time = time.time() - start

        # Batch timing
        start = time.time()
        batch_results = find_similar_batch(observations=observations, cfg=cfg, top_k=5)
        batch_time = time.time() - start

        speedup = sequential_time / batch_time
        print(f"Batch Speedup: {speedup:.1f}×")

        assert speedup >= 15.0, f"Batch speedup {speedup:.1f}× below 15× minimum"
        assert speedup <= 100.0, f"Batch speedup {speedup:.1f}× suspiciously high"

    def test_memory_footprint_100k_vectors(self):
        """Verify memory usage <500MB for 100K vectors."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Create 100K vectors
        store = FAISSVectorStore(dimension=10, auto_select=True)
        vectors = np.random.rand(100000, 10).astype(np.float32)
        records = [
            VectorRecord(vector_id=f"v{i}", vector=vectors[i].tolist(), metadata={})
            for i in range(100000)
        ]
        store.add(records)

        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_delta = mem_after - mem_before

        print(f"Memory footprint for 100K vectors: {mem_delta:.1f} MB")
        assert mem_delta < 500, f"Memory usage {mem_delta:.1f}MB exceeds 500MB target"

    def test_cache_rebuild_frequency(self):
        """Verify incremental cache keeps rebuilds <1/hour."""
        # Monitor cache rebuilds over 10 minutes
        rebuild_count = 0
        start_time = time.time()

        # Simulate writes every 5 seconds
        while time.time() - start_time < 600:  # 10 minutes
            # Add single row
            add_memory_row(...)

            # Check if cache was rebuilt (vs delta append)
            if cache_was_rebuilt():
                rebuild_count += 1

            time.sleep(5)

        # Extrapolate to hourly rate
        rebuilds_per_hour = rebuild_count * (3600 / 600)

        print(f"Cache rebuilds per hour: {rebuilds_per_hour:.1f}")
        assert rebuilds_per_hour < 1.0, f"Rebuild frequency {rebuilds_per_hour}/hr too high"
```

---

## Success Criteria

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Query latency P50 | <5ms | `test_faiss_query_latency_p50_p99` |
| Query latency P99 | <20ms | `test_faiss_query_latency_p50_p99` |
| Batch speedup | ≥15× | `test_batch_vs_sequential_speedup` |
| Memory usage (100K vectors) | <500MB | `test_memory_footprint_100k_vectors` |
| Cache rebuild frequency | <1/hour | `test_cache_rebuild_frequency` |
| Query cache hit rate | >30% | Monitor `_QUERY_RESULT_CACHE` metrics |

---

## Deployment Checklist

### Before Merge:
- [ ] All Phase 2 tests passing (7 new tests)
- [ ] Existing 291 tests still passing
- [ ] Performance benchmarks meet targets
- [ ] Memory usage within limits
- [ ] No regressions in accuracy (compare results)

### Configuration:
```bash
# Enable FAISS (auto-detected if installed)
export MULTIVERSE_MEMORY_USE_FAISS=1

# Configure cache sizes
export MULTIVERSE_MEMORY_QUERY_CACHE_SIZE=10000
export MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD=1000

# Tune lock timeouts if needed
export MULTIVERSE_MEMORY_LOCK_TIMEOUT=30
```

### Monitoring:
- Track query latency (P50, P95, P99)
- Monitor cache hit rates
- Watch memory usage trends
- Count cache rebuilds per hour
- Check fallback rates for external services

---

## Expected Performance Impact

### Before (Baseline):
- Single query: ~85ms (P50)
- 100 queries: ~8.5 seconds (sequential)
- Memory: ~330MB for 52K vectors
- Cache rebuilds: ~10/hour

### After (Phase 2 Complete):
- Single query: **<5ms** (P50) - **17× faster**
- 100 queries: **<500ms** (batch) - **17× faster**
- Memory: **<500MB** for 100K vectors - **1.5× capacity**
- Cache rebuilds: **<1/hour** - **10× reduction**

---

## Next Steps After Phase 2

1. **Phase 3: Validation & Resilience** (1 week)
   - Input validation for vectors/graphs/lambdas
   - Specific exception handling
   - Enhanced monitoring

2. **Phase 4: Production Verification** (1 week)
   - 24-hour soak tests
   - 1M query stress tests
   - Sustained 2000 QPS verification

---

## Quick Reference Commands

```bash
# Run Phase 2 performance tests
python -m pytest tests/test_memory_performance.py -v

# Run full test suite
python -m pytest tests/ -q

# Benchmark query performance manually
python tools/benchmark_memory_queries.py --queries 100 --vectors 52000

# Profile memory usage
python tools/profile_memory_usage.py --max-transitions 100000

# Check cache statistics
python -c "from memory.central_repository import _QUERY_RESULT_CACHE; print(len(_QUERY_RESULT_CACHE))"
```

---

**Document Version:** 1.0
**Last Updated:** 2026-02-28
**Author:** Implementation from Phase 1 + 2.1-2.3 complete
**Status:** Ready for Phase 2.4-2.6 implementation
