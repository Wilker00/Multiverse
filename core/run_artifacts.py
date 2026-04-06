"""
Run artifact manifest helpers.

The goal is to provide a lightweight, versioned artifact contract for each run
directory so operator tooling can reason about run health without guessing from
ad hoc file presence checks.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_FORMAT = "multiverse_run_manifest_v1"


def _iso_utc(epoch_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch_s)))


def _safe_stat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except OSError:
        return None


def _iter_run_files(run_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not run_dir.is_dir():
        return out
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == RUN_MANIFEST_FILENAME:
            continue
        out.append(path)
    out.sort(key=lambda p: p.relative_to(run_dir).as_posix().lower())
    return out


def infer_run_kind(run_dir: str | Path) -> str:
    path = Path(run_dir)
    if (path / "distributed_meta.json").is_file():
        return "distributed_aggregate"
    if (path / "parallel_meta.json").is_file():
        return "parallel_aggregate"
    return "single"


def expected_artifacts_for_kind(run_kind: str) -> List[str]:
    kind = str(run_kind or "single").strip().lower()
    if kind == "distributed_aggregate":
        return ["events.jsonl", "distributed_meta.json"]
    if kind == "parallel_aggregate":
        return ["events.jsonl", "episodes.jsonl", "parallel_meta.json"]
    return ["events.jsonl", "metrics.jsonl"]


def build_run_artifact_manifest(
    run_dir: str | Path,
    *,
    verse_name: Optional[str] = None,
    policy_id: Optional[str] = None,
    algo: Optional[str] = None,
    seed: Optional[int] = None,
    episodes_requested: Optional[int] = None,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    total_return: Optional[float] = None,
    worker_runs: Optional[List[str]] = None,
    run_kind: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = Path(run_dir).resolve()
    now = time.time()
    detected_kind = str(run_kind or infer_run_kind(base)).strip().lower() or "single"
    expected = expected_artifacts_for_kind(detected_kind)
    files = _iter_run_files(base)

    artifact_rows: List[Dict[str, Any]] = []
    present_names = set()
    latest_artifact_mtime = 0.0
    total_size_bytes = 0
    for path in files:
        st = _safe_stat(path)
        if st is None:
            continue
        rel = path.relative_to(base).as_posix()
        present_names.add(rel)
        latest_artifact_mtime = max(latest_artifact_mtime, float(st.st_mtime))
        total_size_bytes += int(st.st_size)
        artifact_rows.append(
            {
                "path": rel,
                "size_bytes": int(st.st_size),
                "modified_epoch_s": float(st.st_mtime),
                "modified_iso": _iso_utc(float(st.st_mtime)),
                "required": bool(rel in expected),
            }
        )

    missing_required = [name for name in expected if name not in present_names]
    status = "ok" if not missing_required else "missing_required_artifacts"

    manifest: Dict[str, Any] = {
        "format": RUN_MANIFEST_FORMAT,
        "generated_at_epoch_s": float(now),
        "generated_at_iso": _iso_utc(now),
        "run_id": base.name,
        "run_dir": str(base).replace("\\", "/"),
        "run_kind": detected_kind,
        "status": status,
        "expected_artifacts": expected,
        "missing_required_artifacts": missing_required,
        "artifact_count": int(len(artifact_rows)),
        "total_size_bytes": int(total_size_bytes),
        "latest_artifact_modified_epoch_s": float(latest_artifact_mtime or 0.0),
        "latest_artifact_modified_iso": (_iso_utc(latest_artifact_mtime) if latest_artifact_mtime > 0 else None),
        "artifacts": artifact_rows,
        "summary": {
            "verse_name": (str(verse_name) if verse_name is not None else None),
            "policy_id": (str(policy_id) if policy_id is not None else None),
            "algo": (str(algo) if algo is not None else None),
            "seed": (None if seed is None else int(seed)),
            "episodes_requested": (None if episodes_requested is None else int(episodes_requested)),
            "max_steps": (None if max_steps is None else int(max_steps)),
            "total_steps": (None if total_steps is None else int(total_steps)),
            "total_return": (None if total_return is None else float(total_return)),
            "worker_runs": [str(x) for x in (worker_runs or []) if str(x).strip()],
        },
    }
    if isinstance(extra, dict) and extra:
        manifest["extra"] = dict(extra)
    return manifest


def write_run_artifact_manifest(run_dir: str | Path, **kwargs: Any) -> str:
    base = Path(run_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    manifest = build_run_artifact_manifest(base, **kwargs)
    out_path = base / RUN_MANIFEST_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return str(out_path)


def read_run_artifact_manifest(run_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(run_dir) / RUN_MANIFEST_FILENAME
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def get_run_artifact_summary(run_dir: str | Path) -> Dict[str, Any]:
    base = Path(run_dir).resolve()
    manifest_path = base / RUN_MANIFEST_FILENAME
    manifest = read_run_artifact_manifest(base)
    run_kind = infer_run_kind(base)
    expected = expected_artifacts_for_kind(run_kind)
    files = _iter_run_files(base)

    latest_artifact_mtime = 0.0
    total_size_bytes = 0
    present_names = set()
    for path in files:
        st = _safe_stat(path)
        if st is None:
            continue
        latest_artifact_mtime = max(latest_artifact_mtime, float(st.st_mtime))
        total_size_bytes += int(st.st_size)
        present_names.add(path.relative_to(base).as_posix())

    missing_required = [name for name in expected if name not in present_names]
    manifest_mtime = 0.0
    if manifest_path.is_file():
        st = _safe_stat(manifest_path)
        if st is not None:
            manifest_mtime = float(st.st_mtime)
    manifest_stale = bool(manifest_mtime > 0 and latest_artifact_mtime > manifest_mtime + 1e-6)
    manifest_status = str(manifest.get("status", "")).strip() if isinstance(manifest, dict) else ""

    return {
        "run_id": base.name,
        "run_dir": str(base).replace("\\", "/"),
        "run_kind": str((manifest or {}).get("run_kind", run_kind)).strip() or run_kind,
        "manifest_present": bool(manifest is not None),
        "manifest_path": (str(manifest_path).replace("\\", "/") if manifest_path.exists() else None),
        "manifest_format": ((manifest or {}).get("format") if isinstance(manifest, dict) else None),
        "manifest_generated_at_iso": ((manifest or {}).get("generated_at_iso") if isinstance(manifest, dict) else None),
        "manifest_status": (manifest_status or ("ok" if not missing_required else "missing_required_artifacts")),
        "manifest_stale": bool(manifest_stale),
        "expected_artifacts": expected,
        "missing_required_artifacts": missing_required,
        "artifact_count": int(len(files)),
        "total_size_bytes": int(total_size_bytes),
        "latest_artifact_modified_iso": (_iso_utc(latest_artifact_mtime) if latest_artifact_mtime > 0 else None),
        "summary": (dict((manifest or {}).get("summary", {})) if isinstance((manifest or {}).get("summary"), dict) else {}),
    }
