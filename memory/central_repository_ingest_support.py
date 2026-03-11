"""
memory/central_repository_ingest_support.py

Support helpers for repository ingest/backfill/event file operations.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set

from memory.central_repository_cache_support import _extract_or_build_universal_vector, _universal_obs_dim
from memory.central_repository_support import (
    BackfillStats,
    CentralMemoryConfig,
    IngestStats,
    SanitizeStats,
    _dedupe_key,
    _dedupe_db_path,
    _dedupe_path,
    _enrich_memory_row,
    _load_tier_policy,
    _ltm_memories_path,
    _ltm_priority_tuple,
    _memories_path,
    _memory_tier_for_event,
    _normalize_memory_tier,
    _policy_bool,
    _policy_float,
    _policy_int,
    _row_verse_name,
    _safe_float,
    _safe_int,
    _stm_memories_path,
    _tier_policy_for_verse,
)
from memory.embeddings import obs_to_vector, project_vector
from memory.selection import SelectionConfig, select_events
from memory.task_taxonomy import memory_family_for_verse, memory_type_for_verse, tags_for_verse


def open_dedupe_db_support(cfg: CentralMemoryConfig) -> sqlite3.Connection:
    """
    Open dedupe database connection with multiprocess-safe configuration.

    Returns:
        sqlite3.Connection with WAL mode and busy_timeout configured
    """
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

    try:
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


def migrate_legacy_dedupe_json_to_db_support(
    cfg: CentralMemoryConfig,
    conn: sqlite3.Connection,
) -> None:
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


def dedupe_try_reserve_support(conn: sqlite3.Connection, key: str) -> bool:
    cur = conn.execute("INSERT OR IGNORE INTO dedupe_keys(dedupe_key) VALUES (?)", (str(key),))
    try:
        return int(cur.rowcount or 0) > 0
    except Exception:
        return False


def load_dedupe_index_support(
    *,
    cfg: CentralMemoryConfig,
    ensure_repo_fn: Callable[[CentralMemoryConfig], None],
    open_dedupe_db_fn: Callable[[CentralMemoryConfig], sqlite3.Connection],
    migrate_legacy_dedupe_json_to_db_fn: Callable[[CentralMemoryConfig, sqlite3.Connection], None],
) -> Set[str]:
    ensure_repo_fn(cfg)
    conn = open_dedupe_db_fn(cfg)
    try:
        migrate_legacy_dedupe_json_to_db_fn(cfg, conn)
        rows = conn.execute("SELECT dedupe_key FROM dedupe_keys").fetchall()
        return set(str(r[0]) for r in rows if r and r[0])
    finally:
        conn.close()


def save_dedupe_index_support(
    *,
    cfg: CentralMemoryConfig,
    keys: Iterable[str],
    ensure_repo_fn: Callable[[CentralMemoryConfig], None],
    open_dedupe_db_fn: Callable[[CentralMemoryConfig], sqlite3.Connection],
) -> None:
    ensure_repo_fn(cfg)
    conn = open_dedupe_db_fn(cfg)
    try:
        conn.execute("DELETE FROM dedupe_keys")
        conn.executemany(
            "INSERT OR IGNORE INTO dedupe_keys(dedupe_key) VALUES (?)",
            [(str(k),) for k in set(str(k) for k in keys) if str(k)],
        )
        conn.commit()
    finally:
        conn.close()


def iter_events_support(events_path: str) -> Iterator[Dict[str, Any]]:
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


def load_events_support(events_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in iter_events_support(events_path):
        rows.append(row)
    return rows


def sanitize_memory_file_support(
    *,
    cfg: CentralMemoryConfig,
    ensure_repo_fn: Callable[[CentralMemoryConfig], None],
    repo_lock_fn: Callable[[CentralMemoryConfig], Any],
    invalidate_similarity_cache_fn: Callable[[Iterable[str]], None],
) -> SanitizeStats:
    """
    Rewrites memory file keeping only valid JSON rows.
    Useful if previous interrupted parallel writes created malformed lines.
    """
    ensure_repo_fn(cfg)
    mem_path = _memories_path(cfg)
    tmp_path = mem_path + ".tmp"
    total = 0
    kept = 0
    with repo_lock_fn(cfg):
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
        invalidate_similarity_cache_fn((mem_path, _ltm_memories_path(cfg), _stm_memories_path(cfg)))
    return SanitizeStats(input_lines=total, kept_lines=kept, dropped_lines=max(0, total - kept))


def backfill_memory_metadata_support(
    *,
    cfg: CentralMemoryConfig,
    rebuild_tier_files: bool = True,
    recompute_tier: bool = False,
    apply_support_guards: bool = True,
    ensure_repo_fn: Callable[[CentralMemoryConfig], None],
    repo_lock_fn: Callable[[CentralMemoryConfig], Any],
    invalidate_similarity_cache_fn: Callable[[Iterable[str]], None],
) -> BackfillStats:
    """
    Enrich existing memory rows with missing tier/family/type metadata and
    optionally rebuild dedicated LTM/STM files.
    """
    ensure_repo_fn(cfg)
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

    with repo_lock_fn(cfg):
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
        invalidate_similarity_cache_fn((mem_path, ltm_path, stm_path))

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


def ingest_run_support(
    *,
    run_dir: str,
    cfg: CentralMemoryConfig,
    selection: Optional[SelectionConfig],
    max_events: Optional[int],
    ensure_repo_fn: Callable[[CentralMemoryConfig], None],
    repo_lock_fn: Callable[[CentralMemoryConfig], Any],
    open_dedupe_db_fn: Callable[[CentralMemoryConfig], sqlite3.Connection],
    migrate_legacy_dedupe_json_to_db_fn: Callable[[CentralMemoryConfig, sqlite3.Connection], None],
    dedupe_try_reserve_fn: Callable[[sqlite3.Connection, str], bool],
    invalidate_similarity_cache_fn: Callable[[Iterable[str]], None],
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
        all_events = load_events_support(events_path)
        input_events = len(all_events)
        selected_events = select_events(all_events, selection)
        if max_events is not None and max_events > 0:
            selected_events = selected_events[: int(max_events)]
        selected_events_count = len(selected_events)
        event_iter = iter(selected_events)
    else:
        event_iter = iter_events_support(events_path)

    added = 0
    skipped = 0
    mem_path = _memories_path(cfg)
    ltm_path = _ltm_memories_path(cfg)
    stm_path = _stm_memories_path(cfg)
    run_id = os.path.basename(os.path.normpath(run_dir))
    tier_policy = _load_tier_policy(cfg)

    ensure_repo_fn(cfg)
    with repo_lock_fn(cfg):
        dedupe_conn = open_dedupe_db_fn(cfg)
        try:
            migrate_legacy_dedupe_json_to_db_fn(cfg, dedupe_conn)
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
                        skipped += 1
                        continue

                    key = _dedupe_key(ev)
                    if not dedupe_try_reserve_fn(dedupe_conn, key):
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
            invalidate_similarity_cache_fn((mem_path, ltm_path, stm_path))
        finally:
            dedupe_conn.close()

    return IngestStats(
        run_dir=run_dir,
        input_events=input_events,
        selected_events=selected_events_count,
        added_events=added,
        skipped_duplicates=skipped,
    )
