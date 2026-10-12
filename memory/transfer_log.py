"""
memory/transfer_log.py

Normalized cross-verse transfer events.

Rollout already records ``transfer_decision_records`` inside step info. This
log copies the useful fields into ``central_memory/transfer_events.jsonl`` so
the live map and transfer overview can read them without scanning every run.
It does not update Q-values or successor features.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional


def transfer_log_path(root_dir: str) -> str:
    return os.path.join(str(root_dir), "transfer_events.jsonl")


def _concept_for(record: Dict[str, Any], target_verse: str, source_verse: str) -> str:
    raw = record.get("concept")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    tags = record.get("concept_tags")
    if isinstance(tags, str) and tags.strip():
        return tags.split(",")[0].strip().lower()
    if isinstance(tags, list) and tags:
        return str(tags[0]).strip().lower()
    try:
        from core.concept_taxonomy import concept_tags_for_verse

        derived = concept_tags_for_verse(target_verse or source_verse)
    except Exception:
        derived = ["task"]
    return str(derived[0] if derived else "task")


def normalize_transfer_record(
    record: Dict[str, Any],
    *,
    episode_return: Optional[float] = None,
) -> Dict[str, Any]:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    target = record.get("target") if isinstance(record.get("target"), dict) else {}
    source_verse = str(record.get("source_verse") or source.get("verse_name") or "unknown").strip().lower()
    target_verse = str(record.get("target_verse") or target.get("verse_name") or "").strip().lower()
    similarity = record.get("similarity", source.get("score"))
    confidence = record.get("confidence", source.get("transfer_confidence"))
    if episode_return is None and record.get("episode_return") is not None:
        episode_return = record.get("episode_return")
    try:
        similarity_value = None if similarity is None else float(similarity)
    except (TypeError, ValueError):
        similarity_value = None
    try:
        confidence_value = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence_value = None
    try:
        return_value = None if episode_return is None else float(episode_return)
    except (TypeError, ValueError):
        return_value = None
    try:
        t_ms = int(record.get("timestamp_ms") or record.get("t_ms") or int(time.time() * 1000))
    except (TypeError, ValueError):
        t_ms = int(time.time() * 1000)
    return {
        "schema_version": "transfer_event.v1",
        "t_ms": t_ms,
        "run_id": str(record.get("run_id") or ""),
        "episode_id": str(record.get("episode_id") or ""),
        "step_idx": record.get("step_idx"),
        "source_verse": source_verse or "unknown",
        "target_verse": target_verse,
        "concept": _concept_for(record, target_verse, source_verse),
        "similarity": similarity_value,
        "confidence": confidence_value,
        "decision": str(record.get("decision") or ""),
        "episode_return": return_value,
    }


def append_transfer_records(
    records: Iterable[Dict[str, Any]],
    *,
    root_dir: str = "central_memory",
    episode_return: Optional[float] = None,
) -> int:
    """Append normalized rows. Returns the number of lines written."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            rows.append(normalize_transfer_record(record, episode_return=episode_return))
    if not rows:
        return 0
    os.makedirs(str(root_dir), exist_ok=True)
    path = transfer_log_path(root_dir)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
