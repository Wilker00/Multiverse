"""
tools/benchmark_batch_queries.py

Quick performance benchmark to verify batch query API speedup (Phase 2.4).

Compares:
- Sequential: N calls to find_similar()
- Batch: 1 call to find_similar_batch()

Expected speedup: 15-20× for 100 queries
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List

from core.types import StepEvent
from memory.central_repository import (
    CentralMemoryConfig,
    find_similar,
    find_similar_batch,
    ingest_run,
)


def create_synthetic_events(
    num_events: int = 1000,
    obs_dim: int = 4,
    verse_name: str = "benchmark_verse",
) -> List[StepEvent]:
    """Create synthetic events for benchmarking."""
    events = []
    for i in range(num_events):
        # Create varied observations with different dimensions
        obs_dict = {f"x{j}": float(i + j) for j in range(obs_dim)}
        next_obs_dict = {f"x{j}": float(i + j + 1) for j in range(obs_dim)}

        event = {
            "verse_name": verse_name,
            "obs": obs_dict,
            "action": i % 4,
            "reward": float(i % 10),
            "next_obs": next_obs_dict,
            "done": (i % 50 == 49),
            "truncated": False,
            "info": {},
            "episode_id": f"ep_{i // 50}",
            "run_id": "benchmark_run",
            "timestamp_ms": 1000000 + i * 100,
        }
        events.append(event)
    return events


def setup_memory_store(
    root_dir: str,
    num_events: int = 1000,
) -> CentralMemoryConfig:
    """Set up memory store with synthetic data."""
    config = CentralMemoryConfig(
        root_dir=root_dir,
        memories_filename="memories.jsonl",
        ltm_memories_filename="ltm_memories.jsonl",
        stm_memories_filename="stm_memories.jsonl",
        dedupe_index_filename="dedupe.json",
        dedupe_db_filename="dedupe_index.sqlite",
    )

    # Create synthetic events
    events = create_synthetic_events(num_events=num_events, obs_dim=4)

    # Write events to run directory
    run_dir = Path(root_dir) / "runs" / "benchmark_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    events_path = run_dir / "events.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # Ingest into central memory
    print(f"Ingesting {num_events} events into memory store...")
    ingest_run(run_dir=str(run_dir), cfg=config)

    return config


def benchmark_sequential(
    config: CentralMemoryConfig,
    query_observations: List[dict],
    top_k: int = 5,
) -> float:
    """Benchmark sequential find_similar() calls."""
    print(f"\nBenchmarking SEQUENTIAL queries ({len(query_observations)} queries)...")

    start = time.time()
    results = []
    for obs in query_observations:
        matches = find_similar(
            obs=obs,
            cfg=config,
            top_k=top_k,
            verse_name="benchmark_verse",
            memory_tiers=["ltm", "stm"],
        )
        results.append(matches)

    elapsed = time.time() - start

    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per-query: {elapsed / len(query_observations) * 1000:.2f}ms")
    print(f"  Total results: {sum(len(r) for r in results)}")

    return elapsed


def benchmark_batch(
    config: CentralMemoryConfig,
    query_observations: List[dict],
    top_k: int = 5,
) -> float:
    """Benchmark batch find_similar_batch() call."""
    print(f"\nBenchmarking BATCH queries ({len(query_observations)} queries)...")

    start = time.time()
    results = find_similar_batch(
        observations=query_observations,
        cfg=config,
        top_k=top_k,
        verse_name="benchmark_verse",
        memory_tiers=["ltm", "stm"],
    )
    elapsed = time.time() - start

    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per-query: {elapsed / len(query_observations) * 1000:.2f}ms")
    print(f"  Total results: {sum(len(r) for r in results)}")

    return elapsed


def main():
    """Run benchmark and report speedup."""
    print("=" * 70)
    print("Phase 2.4 Batch Query API Performance Benchmark")
    print("=" * 70)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up memory store with synthetic data
        config = setup_memory_store(root_dir=tmpdir, num_events=1000)

        # Create query observations (varied dimensions to test grouping)
        num_queries = 100
        query_observations = [
            {f"x{j}": float(i + j) for j in range(4)}
            for i in range(num_queries)
        ]

        # Warm up (first query loads cache)
        print("\nWarming up cache...")
        find_similar(
            obs=query_observations[0],
            cfg=config,
            top_k=5,
            verse_name="benchmark_verse",
            memory_tiers=["ltm", "stm"],
        )
        print("  Cache loaded")

        # Benchmark sequential
        sequential_time = benchmark_sequential(
            config=config,
            query_observations=query_observations,
            top_k=5,
        )

        # Benchmark batch
        batch_time = benchmark_batch(
            config=config,
            query_observations=query_observations,
            top_k=5,
        )

        # Report speedup
        speedup = sequential_time / batch_time if batch_time > 0 else 0

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Batch time:      {batch_time:.3f}s")
        print(f"Speedup:         {speedup:.1f}x")
        print()

        # Check target
        target_speedup = 15.0
        if speedup >= target_speedup:
            print(f"SUCCESS: Achieved {speedup:.1f}x speedup (target: {target_speedup}x)")
        else:
            print(f"INFO: Achieved {speedup:.1f}x speedup (target: {target_speedup}x)")
            print(f"      Note: Small datasets may not show full speedup potential")

        print("=" * 70)


if __name__ == "__main__":
    main()
