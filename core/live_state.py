"""
core/live_state.py

File-backed live snapshot of the multiverse.

Training still writes per-run ``events.jsonl`` through the existing event
logger. This module only reads those artifacts (plus ``lifecycle.jsonl`` and
the central memory bank) so a separate dashboard process can render verses,
memory traffic, and transfers. It is not a second training loop and it does
not replace ``core.streaming_pipeline`` or ``core.communication``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ACTIVE_MTIME_SEC = 15.0
_TAIL_BYTES = 256_000
_TAIL_LINES = 40
_RECENT_RUNS = 12
_FLOW_LIMIT = 12
_MEMORY_COUNT_CAP = 5000


def write_lifecycle_event(run_dir: str, event: str, **fields: Any) -> None:
    """Append one ``run_started`` or ``run_finished`` sidecar line."""
    os.makedirs(str(run_dir), exist_ok=True)
    record: Dict[str, Any] = {"event": str(event), "t_ms": int(time.time() * 1000)}
    for key, value in fields.items():
        record[str(key)] = value
    path = os.path.join(str(run_dir), "lifecycle.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path, *, tail_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        if tail_lines is None:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            size = int(path.stat().st_size)
            with open(path, "rb") as handle:
                if size > _TAIL_BYTES:
                    handle.seek(max(0, size - _TAIL_BYTES))
                    handle.readline()
                text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if tail_lines is not None:
        lines = lines[-int(tail_lines) :]
    out: List[Dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _discover_run_dirs(runs_root: str) -> List[Path]:
    root = Path(runs_root)
    if not root.is_dir():
        return []
    found: List[tuple[float, Path]] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.is_dir():
            continue
        events = child / "events.jsonl"
        lifecycle = child / "lifecycle.jsonl"
        if not events.is_file() and not lifecycle.is_file():
            continue
        stamp = 0.0
        for candidate in (events, lifecycle, child):
            if candidate.exists():
                try:
                    stamp = max(stamp, float(candidate.stat().st_mtime))
                except OSError:
                    continue
        found.append((stamp, child))
    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _stamp, path in found]


def _run_status(run_dir: Path, *, now: float, active_window_sec: float) -> str:
    lifecycle = _read_jsonl(run_dir / "lifecycle.jsonl")
    finished = any(str(row.get("event")) == "run_finished" for row in lifecycle)
    started = any(str(row.get("event")) == "run_started" for row in lifecycle)
    if finished:
        return "idle"
    events = run_dir / "events.jsonl"
    fresh = False
    if events.is_file():
        try:
            fresh = (float(now) - float(events.stat().st_mtime)) <= float(active_window_sec)
        except OSError:
            fresh = False
    if started or fresh:
        return "active"
    return "idle"


def _verse_from_run(run_dir: Path, events: List[Dict[str, Any]]) -> str:
    for row in reversed(events):
        name = str(row.get("verse_name") or "").strip()
        if name:
            return name
    for row in reversed(_read_jsonl(run_dir / "lifecycle.jsonl")):
        name = str(row.get("verse_name") or "").strip()
        if name:
            return name
    manifest = run_dir / "run_manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            name = str(payload.get("verse_name") or "").strip()
            if name:
                return name
    return ""


def _flows_from_event(event: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    info = event.get("info") if isinstance(event.get("info"), dict) else {}
    action_info = info.get("action_info") if isinstance(info.get("action_info"), dict) else {}
    verse = str(event.get("verse_name") or "")
    run_id = str(event.get("run_id") or "")
    step_idx = event.get("step_idx")
    retrievals: List[Dict[str, Any]] = []
    memory_query = info.get("memory_query") if isinstance(info.get("memory_query"), dict) else {}
    match_count = action_info.get("memory_recall_match_count", memory_query.get("match_count", 0))
    try:
        match_count_i = int(match_count or 0)
    except (TypeError, ValueError):
        match_count_i = 0
    recalled = bool(action_info.get("memory_recall_used", False)) or bool(memory_query.get("query_executed", False))
    if match_count_i > 0 or recalled:
        retrievals.append(
            {
                "kind": "recall",
                "verse_name": verse,
                "run_id": run_id,
                "step_idx": step_idx,
                "match_count": match_count_i,
                "used": bool(action_info.get("memory_recall_used", False)),
            }
        )
    transfers: List[Dict[str, Any]] = []
    records = info.get("transfer_decision_records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("decision") or "") == "no_transfer_candidate":
                continue
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            target = record.get("target") if isinstance(record.get("target"), dict) else {}
            transfers.append(
                {
                    "kind": "transfer",
                    "source_verse": str(source.get("verse_name") or ""),
                    "target_verse": str(target.get("verse_name") or verse),
                    "decision": str(record.get("decision") or ""),
                    "confidence": source.get("transfer_confidence"),
                    "similarity": source.get("score"),
                    "run_id": run_id or str(record.get("run_id") or ""),
                    "step_idx": step_idx,
                }
            )
    safety: List[Dict[str, Any]] = []
    safe = action_info.get("safe_executor")
    if isinstance(safe, dict):
        mode = str(safe.get("mode") or "")
        if mode and mode != "primary":
            safety.append(
                {
                    "kind": "safety",
                    "verse_name": verse,
                    "mode": mode,
                    "run_id": run_id,
                    "step_idx": step_idx,
                }
            )
    return retrievals, transfers, safety


def _memory_bank(memory_root: str) -> Dict[str, Any]:
    root = Path(memory_root)
    path = root / "memories.jsonl"
    payload: Dict[str, Any] = {
        "root": str(memory_root),
        "exists": bool(path.is_file()),
        "bytes": 0,
        "row_count": 0,
        "row_count_capped": False,
        "last_ingest_mtime": None,
    }
    if not path.is_file():
        return payload
    try:
        stat = path.stat()
        payload["bytes"] = int(stat.st_size)
        payload["last_ingest_mtime"] = float(stat.st_mtime)
    except OSError:
        return payload
    count = 0
    capped = False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                if count >= _MEMORY_COUNT_CAP:
                    capped = True
                    break
    except OSError:
        count = 0
    payload["row_count"] = int(count)
    payload["row_count_capped"] = bool(capped)
    return payload


def _registered_verses() -> List[str]:
    try:
        from verses.registry import list_verses, register_builtin

        register_builtin()
        return sorted(str(name) for name in list_verses().keys())
    except Exception:
        return []


def _provider_health() -> List[Dict[str, Any]]:
    try:
        from integrations.sim_registry import list_sim_provider_status

        rows = list_sim_provider_status()
        out: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "provider_id": str(row.get("provider_id") or ""),
                    "available": bool(row.get("available", False)),
                    "implemented": bool(row.get("implemented", False)),
                    "kind": str(row.get("kind") or ""),
                    "capabilities": list(row.get("capabilities") or []),
                }
            )
        return out
    except Exception:
        return []


def _trim(rows: List[Dict[str, Any]], limit: int = _FLOW_LIMIT) -> List[Dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    return rows[-int(limit) :]


def build_live_snapshot(
    *,
    runs_root: str = "runs",
    memory_root: str = "central_memory",
    now: Optional[float] = None,
    active_window_sec: float = ACTIVE_MTIME_SEC,
    recent_runs: int = _RECENT_RUNS,
) -> Dict[str, Any]:
    """JSON-serializable snapshot for the terminal map and the local web view."""
    from core.concept_taxonomy import concept_tags_for_verse
    from core.taxonomy import universe_for_verse

    clock = float(time.time() if now is None else now)
    run_dirs = _discover_run_dirs(runs_root)
    runs: List[Dict[str, Any]] = []
    retrievals: List[Dict[str, Any]] = []
    transfers: List[Dict[str, Any]] = []
    safety: List[Dict[str, Any]] = []
    verse_stats: Dict[str, Dict[str, Any]] = {}

    def verse_bucket(name: str) -> Dict[str, Any]:
        key = str(name or "").strip() or "unknown"
        bucket = verse_stats.get(key)
        if bucket is None:
            bucket = {
                "run_count": 0,
                "active_runs": 0,
                "reward_sum": 0.0,
                "success_count": 0,
                "recall_count": 0,
                "transfer_count": 0,
                "safety_count": 0,
            }
            verse_stats[key] = bucket
        return bucket

    for run_dir in run_dirs[: max(1, int(recent_runs))]:
        events = _read_jsonl(run_dir / "events.jsonl", tail_lines=_TAIL_LINES)
        status = _run_status(run_dir, now=clock, active_window_sec=float(active_window_sec))
        verse_name = _verse_from_run(run_dir, events)
        reward_sum = 0.0
        for event in events:
            try:
                reward_sum += float(event.get("reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            event_verse = str(event.get("verse_name") or verse_name or "")
            bucket = verse_bucket(event_verse)
            try:
                bucket["reward_sum"] = float(bucket["reward_sum"]) + float(event.get("reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            info = event.get("info") if isinstance(event.get("info"), dict) else {}
            try:
                reward_value = float(event.get("reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                reward_value = 0.0
            if bool(event.get("done", False)) and (bool(info.get("reached_goal", False)) or reward_value > 0.0):
                bucket["success_count"] = int(bucket["success_count"]) + 1
            got_retrievals, got_transfers, got_safety = _flows_from_event(event)
            bucket["recall_count"] = int(bucket["recall_count"]) + len(got_retrievals)
            bucket["transfer_count"] = int(bucket["transfer_count"]) + len(got_transfers)
            bucket["safety_count"] = int(bucket["safety_count"]) + len(got_safety)
            retrievals.extend(got_retrievals)
            transfers.extend(got_transfers)
            safety.extend(got_safety)
        if verse_name:
            bucket = verse_bucket(verse_name)
            bucket["run_count"] = int(bucket["run_count"]) + 1
            if status == "active":
                bucket["active_runs"] = int(bucket["active_runs"]) + 1
        runs.append(
            {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "status": status,
                "verse_name": verse_name,
                "reward_sum": float(reward_sum),
                "event_sample": int(len(events)),
            }
        )

    catalog = _registered_verses()
    for name in catalog:
        verse_bucket(name)
    verses: List[Dict[str, Any]] = []
    for name in sorted(verse_stats.keys()):
        bucket = verse_stats[name]
        concepts = concept_tags_for_verse(name) if name != "unknown" else ["task"]
        verses.append(
            {
                "verse_name": name,
                "universe": universe_for_verse(name) if name != "unknown" else "unknown",
                "concept_tags": concepts,
                "status": "active" if int(bucket["active_runs"]) > 0 else "idle",
                "run_count": int(bucket["run_count"]),
                "reward_sum": float(bucket["reward_sum"]),
                "success_count": int(bucket["success_count"]),
                "recall_count": int(bucket["recall_count"]),
                "transfer_count": int(bucket["transfer_count"]),
                "safety_count": int(bucket["safety_count"]),
                "registered": bool(name in catalog),
            }
        )

    active_runs = sum(1 for row in runs if row["status"] == "active")
    memory = _memory_bank(memory_root)
    return {
        "schema_version": "live_snapshot.v1",
        "generated_at_ms": int(clock * 1000),
        "runs_root": str(runs_root),
        "memory_root": str(memory_root),
        "verses": verses,
        "runs": runs,
        "memory": memory,
        "flows": {
            "retrievals": _trim(retrievals),
            "transfers": _trim(transfers),
            "safety": _trim(safety),
        },
        "health": {"providers": _provider_health()},
        "counts": {
            "registered_verses": int(len(catalog)),
            "active_runs": int(active_runs),
            "idle_runs": int(len(runs) - active_runs),
            "memory_rows": int(memory.get("row_count", 0) or 0),
            "recent_transfers": int(len(transfers)),
            "recent_retrievals": int(len(retrievals)),
            "recent_safety": int(len(safety)),
        },
    }


def live_summary_line(snapshot: Dict[str, Any]) -> str:
    counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
    memory = snapshot.get("memory") if isinstance(snapshot.get("memory"), dict) else {}
    rows = memory.get("row_count", counts.get("memory_rows", 0))
    return (
        f"verses={int(counts.get('registered_verses', 0))} "
        f"active_runs={int(counts.get('active_runs', 0))} "
        f"memory_rows={int(rows or 0)} "
        f"transfers={int(counts.get('recent_transfers', 0))}"
    )


def map_verse_rows(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Verses with runs or live traffic. Falls back to a short idle catalog."""
    verses = snapshot.get("verses") if isinstance(snapshot.get("verses"), list) else []
    active = [
        row
        for row in verses
        if isinstance(row, dict)
        and (
            int(row.get("run_count", 0) or 0) > 0
            or int(row.get("recall_count", 0) or 0) > 0
            or int(row.get("transfer_count", 0) or 0) > 0
            or int(row.get("safety_count", 0) or 0) > 0
        )
    ]
    if active:
        return active
    return [row for row in verses if isinstance(row, dict)][:8]


def iter_flow_edges(snapshot: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    flows = snapshot.get("flows") if isinstance(snapshot.get("flows"), dict) else {}
    for kind in ("retrievals", "transfers", "safety"):
        rows = flows.get(kind)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row
