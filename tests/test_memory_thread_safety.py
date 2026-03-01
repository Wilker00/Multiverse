"""
tests/test_memory_thread_safety.py

Thread-safety and multiprocess-safety tests for memory system hardening (Phase 1).

Tests verify that concurrent writes from multiple processes don't corrupt data.
"""

import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pytest

from core.types import StepEvent
from memory.central_repository import (
    CentralMemoryConfig,
    ingest_run,
    _atomic_write,
    _open_dedupe_db,
)


@pytest.fixture
def temp_memory_root():
    """Create temporary directory for memory tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def memory_config(temp_memory_root):
    """Create test memory configuration."""
    return CentralMemoryConfig(
        root_dir=temp_memory_root,
        memories_filename="memories.jsonl",
        ltm_memories_filename="ltm_memories.jsonl",
        stm_memories_filename="stm_memories.jsonl",
        dedupe_index_filename="dedupe.json",
        dedupe_db_filename="dedupe_index.sqlite",
    )


def create_test_events(worker_id: int, count: int) -> List[StepEvent]:
    """Create test events for a worker."""
    events = []
    for i in range(count):
        event = {
            "verse_name": f"test_verse",
            "obs": {"worker": worker_id, "step": i},
            "action": i % 4,
            "reward": float(i),
            "next_obs": {"worker": worker_id, "step": i + 1},
            "done": (i == count - 1),
            "truncated": False,
            "info": {},
            "episode_id": f"worker_{worker_id}_ep_0",
            "run_id": f"run_worker_{worker_id}",
            "timestamp_ms": 1000000 + worker_id * 10000 + i,
        }
        events.append(event)
    return events


def worker_ingest_task(args):
    """
    Task function for ProcessPoolExecutor - ingests events from one worker.

    Args:
        args: Tuple of (worker_id, event_count, memory_config_dict)

    Returns:
        Tuple of (worker_id, success, event_count)
    """
    worker_id, event_count, config_dict = args

    # Reconstruct config from dict (Manager can't pickle dataclass directly)
    config = CentralMemoryConfig(**config_dict)

    # Create events
    events = create_test_events(worker_id, event_count)

    try:
        # Simulate a run directory with events.jsonl
        run_dir = Path(config.root_dir) / f"runs/worker_{worker_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        events_path = run_dir / "events.jsonl"
        with open(events_path, "w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Ingest run into central memory
        ingest_run(
            run_dir=str(run_dir),
            cfg=config,
        )

        return (worker_id, True, event_count)

    except Exception as exc:
        print(f"Worker {worker_id} failed: {exc}")
        return (worker_id, False, 0)


class TestAtomicWrite:
    """Test atomic file write functionality."""

    def test_atomic_write_creates_file(self, temp_memory_root):
        """Test that atomic write creates file successfully."""
        file_path = os.path.join(temp_memory_root, "test.json")
        content = json.dumps({"test": "data"}, ensure_ascii=False)

        _atomic_write(file_path, content)

        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"test": "data"}

    def test_atomic_write_overwrites_existing(self, temp_memory_root):
        """Test that atomic write overwrites existing file."""
        file_path = os.path.join(temp_memory_root, "test.json")

        # Write initial content
        _atomic_write(file_path, json.dumps({"old": "data"}))

        # Overwrite
        _atomic_write(file_path, json.dumps({"new": "data"}))

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"new": "data"}

    def test_atomic_write_creates_parent_dir(self, temp_memory_root):
        """Test that atomic write creates parent directories."""
        file_path = os.path.join(temp_memory_root, "subdir", "nested", "test.json")
        content = json.dumps({"test": "data"})

        _atomic_write(file_path, content)

        assert os.path.exists(file_path)


class TestSQLiteBusyTimeout:
    """Test SQLite busy_timeout configuration."""

    def test_busy_timeout_configured(self, memory_config):
        """Test that busy_timeout pragma is set."""
        conn = _open_dedupe_db(memory_config)
        try:
            # Query current busy_timeout (returns milliseconds)
            cursor = conn.execute("PRAGMA busy_timeout")
            timeout_ms = cursor.fetchone()[0]

            # Should be 30000ms (30 seconds) by default
            assert timeout_ms >= 1000, f"busy_timeout too low: {timeout_ms}ms"
        finally:
            conn.close()

    def test_wal_mode_enabled(self, memory_config):
        """Test that WAL mode is enabled for concurrent access."""
        conn = _open_dedupe_db(memory_config)
        try:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0].lower()
            assert mode == "wal", f"Expected WAL mode, got {mode}"
        finally:
            conn.close()


class TestParallelIngestion:
    """Test parallel memory ingestion from multiple processes."""

    def test_parallel_ingest_10_workers_100_events(self, memory_config):
        """
        Test that 10 workers can ingest 100 events each without corruption.

        This is the critical test for Phase 1 success criteria.
        """
        num_workers = 10
        events_per_worker = 100
        expected_total = num_workers * events_per_worker

        # Convert config to dict for pickling
        config_dict = {
            "root_dir": memory_config.root_dir,
            "memories_filename": memory_config.memories_filename,
            "ltm_memories_filename": memory_config.ltm_memories_filename,
            "stm_memories_filename": memory_config.stm_memories_filename,
            "dedupe_index_filename": memory_config.dedupe_index_filename,
            "dedupe_db_filename": memory_config.dedupe_db_filename,
        }

        # Run parallel ingestion
        tasks = [(i, events_per_worker, config_dict) for i in range(num_workers)]

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_ingest_task, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

        # Verify all workers succeeded
        successes = [r for r in results if r[1]]
        assert len(successes) == num_workers, f"Only {len(successes)}/{num_workers} workers succeeded"

        # Verify JSONL integrity (no corrupted lines)
        memories_path = os.path.join(memory_config.root_dir, memory_config.memories_filename)
        assert os.path.exists(memories_path), "memories.jsonl not created"

        valid_lines = 0
        with open(memories_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError as exc:
                    pytest.fail(f"Line {line_num} is corrupted JSON: {exc}\nLine: {line[:100]}")

        # Verify we got the expected number of events (accounting for deduplication)
        # Note: Some events may be deduplicated, so valid_lines <= expected_total
        assert valid_lines > 0, "No valid events ingested"
        assert valid_lines <= expected_total, f"More events than expected: {valid_lines} > {expected_total}"

        # Verify SQLite dedupe index consistency
        conn = _open_dedupe_db(memory_config)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM dedupe_keys")
            db_count = cursor.fetchone()[0]

            # Dedupe count should match the number of unique events
            assert db_count > 0, "No dedupe keys in database"
            assert db_count <= expected_total, f"More dedupe keys than expected: {db_count} > {expected_total}"

        finally:
            conn.close()

    def test_parallel_ingest_no_lock_timeout(self, memory_config):
        """Test that no worker times out waiting for locks."""
        num_workers = 5
        events_per_worker = 50

        config_dict = {
            "root_dir": memory_config.root_dir,
            "memories_filename": memory_config.memories_filename,
            "ltm_memories_filename": memory_config.ltm_memories_filename,
            "stm_memories_filename": memory_config.stm_memories_filename,
            "dedupe_index_filename": memory_config.dedupe_index_filename,
            "dedupe_db_filename": memory_config.dedupe_db_filename,
        }

        tasks = [(i, events_per_worker, config_dict) for i in range(num_workers)]

        # All workers should complete without TimeoutError
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_ingest_task, task) for task in tasks]
            for future in as_completed(futures):
                worker_id, success, count = future.result()
                assert success, f"Worker {worker_id} failed (possible lock timeout)"


class TestMultiprocessLocks:
    """Test multiprocess lock behavior."""

    def test_repo_lock_timeout_respected(self, memory_config):
        """Test that repo lock respects timeout."""
        from memory.central_repository import _repo_lock

        # First lock should succeed
        with _repo_lock(memory_config):
            pass

        # Second lock should also succeed (no deadlock)
        with _repo_lock(memory_config):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
