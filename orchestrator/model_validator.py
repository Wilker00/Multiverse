"""
orchestrator/model_validator.py

Validation helpers for UniversalModel behavior on recorded runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from core.types import JSONValue
from models.universal_model import UniversalModel, UniversalModelConfig


@dataclass
class ValidationReport:
    run_id: str
    total_events: int
    matched_events: int
    action_match_events: int
    coverage: float
    action_accuracy: float
    by_verse: Dict[str, Dict[str, float]]


def _action_key(action: JSONValue) -> str:
    try:
        return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(action)


def _load_events(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"events file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_run_against_memory(
    *,
    run_dir: str,
    memory_dir: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> ValidationReport:
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    run_id = os.path.basename(os.path.normpath(run_dir))
    events = _load_events(run_dir)
    model = UniversalModel(
        config=UniversalModelConfig(
            memory_dir=memory_dir,
            default_top_k=top_k,
            default_min_score=min_score,
        )
    )

    matched = 0
    action_match = 0
    per_verse: Dict[str, Dict[str, int]] = {}

    for ev in events:
        verse = str(ev.get("verse_name", "unknown"))
        bucket = per_verse.setdefault(verse, {"total": 0, "matched": 0, "action_match": 0})
        bucket["total"] += 1

        advice = model.recommend(
            obs=ev.get("obs"),
            verse_name=verse,
            top_k=top_k,
            min_score=min_score,
            exclude_run_ids=[run_id],
        )
        if advice is None:
            continue

        matched += 1
        bucket["matched"] += 1

        if _action_key(advice.action) == _action_key(ev.get("action")):
            action_match += 1
            bucket["action_match"] += 1

    total = len(events)
    coverage = (matched / float(total)) if total > 0 else 0.0
    action_accuracy = (action_match / float(matched)) if matched > 0 else 0.0

    by_verse_out: Dict[str, Dict[str, float]] = {}
    for verse, b in per_verse.items():
        total_v = max(1, int(b["total"]))
        matched_v = int(b["matched"])
        action_match_v = int(b["action_match"])
        by_verse_out[verse] = {
            "total_events": float(b["total"]),
            "coverage": matched_v / float(total_v),
            "action_accuracy": (action_match_v / float(matched_v)) if matched_v > 0 else 0.0,
        }

    return ValidationReport(
        run_id=run_id,
        total_events=total,
        matched_events=matched,
        action_match_events=action_match,
        coverage=coverage,
        action_accuracy=action_accuracy,
        by_verse=by_verse_out,
    )


def print_validation_report(report: ValidationReport) -> None:
    print("Model validation report")
    print(f"run_id          : {report.run_id}")
    print(f"total_events    : {report.total_events}")
    print(f"matched_events  : {report.matched_events}")
    print(f"action_match    : {report.action_match_events}")
    print(f"coverage        : {report.coverage:.3f}")
    print(f"action_accuracy : {report.action_accuracy:.3f}")
    print("")
    print("By verse:")
    for verse in sorted(report.by_verse.keys()):
        row = report.by_verse[verse]
        print(
            f"  {verse:<12} total={int(row['total_events']):4d} "
            f"coverage={row['coverage']:.3f} action_accuracy={row['action_accuracy']:.3f}"
        )
