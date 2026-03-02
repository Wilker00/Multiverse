"""
Run discovery and inspection helpers for the Multiverse CLI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    modified_ts: float
    total_size_bytes: int


def _human_bytes(n: int) -> str:
    size = float(max(0, int(n)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(size)}{units[idx]}"
    return f"{size:.1f}{units[idx]}"


def _iso_local(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                total += int(p.stat().st_size)
            except OSError:
                continue
    return total


def discover_runs(runs_root: str) -> List[RunRecord]:
    root = Path(runs_root)
    if not root.is_dir():
        return []

    out: List[RunRecord] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        events = item / "events.jsonl"
        if not events.is_file():
            continue
        try:
            modified = float(max(item.stat().st_mtime, events.stat().st_mtime))
        except OSError:
            modified = 0.0
        out.append(
            RunRecord(
                run_id=item.name,
                run_dir=item,
                modified_ts=modified,
                total_size_bytes=_dir_size_bytes(item),
            )
        )
    out.sort(key=lambda r: r.modified_ts, reverse=True)
    return out


def resolve_run_dir(runs_root: str, run_id: str | None) -> Path:
    root = Path(runs_root)
    if run_id:
        candidate = root / str(run_id)
        if not candidate.is_dir():
            raise FileNotFoundError(f"Run not found: {candidate}")
        return candidate
    runs = discover_runs(runs_root)
    if not runs:
        raise FileNotFoundError(f"No runs with events.jsonl found under: {root}")
    return runs[0].run_dir


def _line_count(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def run_snapshot(runs_root: str) -> Dict[str, Any]:
    rows = discover_runs(runs_root)
    latest = rows[0] if rows else None
    return {
        "runs_root": str(runs_root),
        "run_count": int(len(rows)),
        "latest": {
            "run_id": latest.run_id,
            "path": str(latest.run_dir),
            "modified": _iso_local(latest.modified_ts),
            "size_bytes": latest.total_size_bytes,
            "size_human": _human_bytes(latest.total_size_bytes),
        }
        if latest
        else None,
    }


def cmd_runs_list(args: argparse.Namespace) -> int:
    rows = discover_runs(args.runs_root)
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    if not rows:
        print(f"No runs found under {args.runs_root}")
        return 0
    if bool(args.json):
        payload = [
            {
                "run_id": r.run_id,
                "path": str(r.run_dir),
                "modified": _iso_local(r.modified_ts),
                "size_bytes": r.total_size_bytes,
                "size_human": _human_bytes(r.total_size_bytes),
            }
            for r in rows
        ]
        print(json.dumps({"runs": payload}, indent=2))
        return 0
    print("RUN ID                           MODIFIED             SIZE")
    for row in rows:
        rid = row.run_id[:32].ljust(32)
        print(f"{rid} {_iso_local(row.modified_ts)}  {_human_bytes(row.total_size_bytes)}")
    return 0


def cmd_runs_latest(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.runs_root, run_id=None)
    if bool(args.json):
        st = run_dir.stat()
        payload = {
            "run_id": run_dir.name,
            "path": str(run_dir),
            "modified": _iso_local(st.st_mtime),
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.path_only:
        print(str(run_dir))
    else:
        print(f"Latest run: {run_dir.name}")
        print(str(run_dir))
    return 0


def cmd_runs_files(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.runs_root, run_id=args.run_id)
    iterator = run_dir.rglob("*") if bool(args.recursive) else run_dir.glob("*")
    files = [p for p in iterator if p.is_file()]
    files.sort(key=lambda p: str(p.relative_to(run_dir)).lower())
    if args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        print(f"No files found in {run_dir}")
        return 0
    if bool(args.json):
        payload = []
        for p in files:
            rel = p.relative_to(run_dir).as_posix()
            try:
                size = int(p.stat().st_size)
            except OSError:
                size = -1
            payload.append(
                {
                    "file": rel,
                    "size_bytes": size,
                    "size_human": _human_bytes(max(size, 0)) if size >= 0 else "?",
                }
            )
        print(json.dumps({"run_id": run_dir.name, "files": payload}, indent=2))
        return 0
    for p in files:
        rel = p.relative_to(run_dir).as_posix()
        try:
            size = _human_bytes(int(p.stat().st_size))
        except OSError:
            size = "?"
        print(f"{rel}\t{size}")
    return 0


def cmd_runs_tail(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.runs_root, run_id=args.run_id)
    file_path = run_dir / str(args.file)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    tail = deque(maxlen=max(1, int(args.lines)))
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tail.append(line.rstrip("\n"))
    for line in tail:
        print(line)
    return 0


def cmd_runs_inspect(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.runs_root, run_id=args.run_id)
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    events_path = run_dir / "events.jsonl"
    episode_path = run_dir / "episodes.jsonl"
    events_count = None
    if bool(args.count_events) and events_path.is_file():
        events_count = _line_count(events_path)
    size_bytes = _dir_size_bytes(run_dir)
    payload = {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "modified": _iso_local(run_dir.stat().st_mtime),
        "size_bytes": size_bytes,
        "size_human": _human_bytes(size_bytes),
        "file_count": len(files),
        "has_events_jsonl": bool(events_path.is_file()),
        "has_episodes_jsonl": bool(episode_path.is_file()),
        "events_line_count": events_count,
    }
    if bool(args.json):
        print(json.dumps(payload, indent=2))
        return 0
    print(f"Run ID          : {payload['run_id']}")
    print(f"Path            : {payload['path']}")
    print(f"Modified        : {payload['modified']}")
    print(f"Size            : {payload['size_human']} ({payload['size_bytes']} bytes)")
    print(f"File count      : {payload['file_count']}")
    print(f"events.jsonl    : {'yes' if payload['has_events_jsonl'] else 'no'}")
    print(f"episodes.jsonl  : {'yes' if payload['has_episodes_jsonl'] else 'no'}")
    if events_count is not None:
        print(f"events lines    : {events_count}")
    return 0
