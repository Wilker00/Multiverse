"""
Runtime ownership helpers for the central memory repository.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable

from memory.central_repository_support import CentralMemoryConfig


def atomic_write_support(file_path: str, content: str, max_retries: int = 3) -> None:
    """
    Atomically write file content using a temp file + replace workflow.
    """
    for attempt in range(max_retries):
        tmp_path = ""
        try:
            dir_name = os.path.dirname(file_path) or "."
            os.makedirs(dir_name, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=dir_name,
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            os.replace(tmp_path, file_path)
            return
        except Exception as exc:
            if tmp_path:
                try:
                    if os.path.isfile(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            if attempt < max_retries - 1:
                time.sleep((100 * (2**attempt)) / 1000.0)
                continue
            raise IOError(f"Failed to atomic write after {max_retries} attempts: {exc}") from exc


@contextmanager
def repo_lock_support(
    *,
    cfg: CentralMemoryConfig,
    get_mp_locks_fn: Callable[[], Any],
    repo_lock_path_fn: Callable[[CentralMemoryConfig], str],
    lock_timeout: int,
    msvcrt_module: Any,
    fcntl_module: Any,
):
    """
    Acquire the repository's process lock and optional platform file lock.
    """
    _, _, _, repo_lock = get_mp_locks_fn()
    os.makedirs(cfg.root_dir, exist_ok=True)
    lock_path = repo_lock_path_fn(cfg)

    mp_locked = repo_lock.acquire(timeout=lock_timeout)
    if not mp_locked:
        raise TimeoutError(f"Failed to acquire multiprocess repo lock within {lock_timeout}s")

    try:
        with open(lock_path, "a+b") as fh:
            file_locked = False
            try:
                if msvcrt_module is not None:
                    fh.seek(0, os.SEEK_END)
                    if fh.tell() <= 0:
                        fh.write(b"0")
                        fh.flush()
                    fh.seek(0)
                    msvcrt_module.locking(fh.fileno(), msvcrt_module.LK_LOCK, 1)
                    file_locked = True
                elif fcntl_module is not None:
                    fcntl_module.flock(fh.fileno(), fcntl_module.LOCK_EX)
                    file_locked = True
                yield
            finally:
                if file_locked:
                    try:
                        fh.seek(0)
                        if msvcrt_module is not None:
                            msvcrt_module.locking(fh.fileno(), msvcrt_module.LK_UNLCK, 1)
                        elif fcntl_module is not None:
                            fcntl_module.flock(fh.fileno(), fcntl_module.LOCK_UN)
                    except Exception:
                        pass
    finally:
        repo_lock.release()


def ensure_repo_support(
    *,
    cfg: CentralMemoryConfig,
    get_mp_locks_fn: Callable[[], Any],
    memories_path_fn: Callable[[CentralMemoryConfig], str],
    ltm_memories_path_fn: Callable[[CentralMemoryConfig], str],
    stm_memories_path_fn: Callable[[CentralMemoryConfig], str],
    dedupe_path_fn: Callable[[CentralMemoryConfig], str],
    dedupe_db_path_fn: Callable[[CentralMemoryConfig], str],
    dedupe_ready: set[str],
    atomic_write_fn: Callable[[str, str, int], None],
    open_dedupe_db_fn: Callable[[CentralMemoryConfig], Any],
    migrate_legacy_dedupe_json_to_db_fn: Callable[[CentralMemoryConfig, Any], None],
) -> None:
    """
    Ensure repository files and dedupe database exist and are bootstrapped.
    """
    _, dedupe_ready_lock, _, _ = get_mp_locks_fn()

    os.makedirs(cfg.root_dir, exist_ok=True)
    for mem_path in (
        memories_path_fn(cfg),
        ltm_memories_path_fn(cfg),
        stm_memories_path_fn(cfg),
    ):
        if not os.path.isfile(mem_path):
            with open(mem_path, "w", encoding="utf-8"):
                pass

    idx_path = dedupe_path_fn(cfg)
    if not os.path.isfile(idx_path):
        atomic_write_fn(idx_path, json.dumps([], ensure_ascii=False), 3)

    db_path = os.path.abspath(dedupe_db_path_fn(cfg))
    with dedupe_ready_lock:
        ready = db_path in dedupe_ready
    if ready:
        return

    conn = open_dedupe_db_fn(cfg)
    try:
        migrate_legacy_dedupe_json_to_db_fn(cfg, conn)
    finally:
        conn.close()

    with dedupe_ready_lock:
        dedupe_ready.add(db_path)
