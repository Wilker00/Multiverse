"""
memory/transfer_metrics.py

Read-only summary of ``transfer_events.jsonl``.

Success means a later episode return was positive. This is a benchmark hook
for shared knowledge across verses. It does not retrain agents.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from memory.transfer_log import transfer_log_path


def _load(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def summarize_transfer_log(path: str) -> Dict[str, Any]:
    rows = _load(path)
    attempts = 0
    cross = 0
    confidences: List[float] = []
    successes = 0
    known_returns = 0
    pairs: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        attempts += 1
        source = str(row.get("source_verse") or "unknown")
        target = str(row.get("target_verse") or "")
        concept = str(row.get("concept") or "task")
        if source and target and source != target and source != "unknown":
            cross += 1
        confidence = row.get("confidence")
        if confidence is not None:
            try:
                confidences.append(float(confidence))
            except (TypeError, ValueError):
                pass
        episode_return = row.get("episode_return")
        positive = False
        if episode_return is not None:
            try:
                value = float(episode_return)
            except (TypeError, ValueError):
                value = None
            if value is not None:
                known_returns += 1
                positive = value > 0.0
                if positive:
                    successes += 1
        key = f"{source}->{target}:{concept}"
        bucket = pairs.get(key)
        if bucket is None:
            bucket = {
                "source_verse": source,
                "target_verse": target,
                "concept": concept,
                "attempts": 0,
                "successes": 0,
                "confidence_sum": 0.0,
                "confidence_count": 0,
            }
            pairs[key] = bucket
        bucket["attempts"] = int(bucket["attempts"]) + 1
        if positive:
            bucket["successes"] = int(bucket["successes"]) + 1
        if confidence is not None:
            try:
                bucket["confidence_sum"] = float(bucket["confidence_sum"]) + float(confidence)
                bucket["confidence_count"] = int(bucket["confidence_count"]) + 1
            except (TypeError, ValueError):
                pass
    pair_rows: List[Dict[str, Any]] = []
    for bucket in pairs.values():
        count = int(bucket["confidence_count"])
        pair_rows.append(
            {
                "source_verse": bucket["source_verse"],
                "target_verse": bucket["target_verse"],
                "concept": bucket["concept"],
                "attempts": int(bucket["attempts"]),
                "successes": int(bucket["successes"]),
                "mean_confidence": (None if count <= 0 else float(bucket["confidence_sum"]) / float(count)),
            }
        )
    pair_rows.sort(key=lambda item: int(item["attempts"]), reverse=True)
    return {
        "path": str(path),
        "attempts": int(attempts),
        "cross_verse_fraction": (0.0 if attempts <= 0 else float(cross) / float(attempts)),
        "mean_confidence": (None if not confidences else float(sum(confidences)) / float(len(confidences))),
        "success_rate": (None if known_returns <= 0 else float(successes) / float(known_returns)),
        "known_returns": int(known_returns),
        "pairs": pair_rows,
    }


def summarize_transfer_root(root_dir: str = "central_memory") -> Dict[str, Any]:
    return summarize_transfer_log(transfer_log_path(root_dir))


def format_transfer_overview(summary: Dict[str, Any]) -> str:
    lines = [
        "Transfer overview",
        "-----------------",
        f"Log             : {summary.get('path')}",
        f"Attempts        : {int(summary.get('attempts', 0) or 0)}",
        f"Cross-verse     : {float(summary.get('cross_verse_fraction', 0.0) or 0.0):.3f}",
        f"Mean confidence : {summary.get('mean_confidence')}",
        f"Success rate    : {summary.get('success_rate')}",
        "Pairs",
    ]
    pairs = summary.get("pairs") if isinstance(summary.get("pairs"), list) else []
    if not pairs:
        lines.append("  none")
    for row in pairs[:20]:
        lines.append(
            "  {source} -> {target}  concept={concept}  n={attempts}  "
            "successes={successes}  conf={confidence}".format(
                source=row.get("source_verse"),
                target=row.get("target_verse"),
                concept=row.get("concept"),
                attempts=row.get("attempts"),
                successes=row.get("successes"),
                confidence=row.get("mean_confidence"),
            )
        )
    return "\n".join(lines)
