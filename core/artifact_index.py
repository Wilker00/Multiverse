"""
Shared artifact index for operator-facing JSON outputs.

This keeps a lightweight registry of the latest health, readiness, benchmark,
and sentinel artifacts so CLI status surfaces do not need to guess from
directory scans alone.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ARTIFACT_INDEX_PATH = os.path.join("models", "ops", "artifact_index.json")
ARTIFACT_INDEX_FORMAT = "multiverse_artifact_index_v1"


def _iso_utc(epoch_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch_s)))


def _normalize_path(path: str) -> str:
    return str(path).replace("\\", "/")


def load_artifact_index(path: str = DEFAULT_ARTIFACT_INDEX_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {
            "format": ARTIFACT_INDEX_FORMAT,
            "updated_at_iso": None,
            "artifacts": {},
        }
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {p}")
    artifacts = obj.get("artifacts")
    if not isinstance(artifacts, dict):
        obj["artifacts"] = {}
    return obj


def save_artifact_index(index: Dict[str, Any], path: str = DEFAULT_ARTIFACT_INDEX_PATH) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(index)
    payload["format"] = ARTIFACT_INDEX_FORMAT
    payload["updated_at_iso"] = _iso_utc(time.time())
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(p).replace("\\", "/")


def register_artifact(
    *,
    artifact_type: str,
    artifact_path: str,
    status: str,
    created_at_iso: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    index_path: str = DEFAULT_ARTIFACT_INDEX_PATH,
    keep: int = 12,
) -> str:
    idx = load_artifact_index(index_path)
    artifacts = idx.setdefault("artifacts", {})
    rows = list(artifacts.get(str(artifact_type), [])) if isinstance(artifacts.get(str(artifact_type)), list) else []
    now_iso = _iso_utc(time.time())
    record = {
        "artifact_type": str(artifact_type),
        "path": _normalize_path(artifact_path),
        "status": str(status),
        "created_at_iso": str(created_at_iso or now_iso),
        "registered_at_iso": now_iso,
        "metadata": dict(metadata or {}),
    }
    rows = [r for r in rows if not (isinstance(r, dict) and str(r.get("path", "")) == record["path"])]
    rows.insert(0, record)
    artifacts[str(artifact_type)] = rows[: max(1, int(keep))]
    return save_artifact_index(idx, index_path)


def latest_artifact(index: Dict[str, Any], artifact_type: str) -> Optional[Dict[str, Any]]:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    rows = artifacts.get(str(artifact_type))
    if not isinstance(rows, list) or not rows:
        return None
    first = rows[0]
    return dict(first) if isinstance(first, dict) else None


def summarize_artifact_index(path: str = DEFAULT_ARTIFACT_INDEX_PATH) -> Dict[str, Any]:
    idx = load_artifact_index(path)
    artifacts = idx.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    latest: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    for art_type, rows in artifacts.items():
        if isinstance(rows, list):
            counts[str(art_type)] = int(len(rows))
        record = latest_artifact(idx, str(art_type))
        if record is not None:
            latest[str(art_type)] = record
    return {
        "path": _normalize_path(path),
        "exists": bool(Path(path).is_file()),
        "format": idx.get("format"),
        "updated_at_iso": idx.get("updated_at_iso"),
        "artifact_types": sorted(str(k) for k in artifacts.keys()),
        "artifact_counts": counts,
        "latest": latest,
    }
