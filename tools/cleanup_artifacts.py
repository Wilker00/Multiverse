"""
tools/cleanup_artifacts.py

Prune generated artifact directories (runs/memory snapshots) to keep repo
workspaces manageable at scale.

Default behavior is safe:
- dry-run (no deletion) unless --delete is provided
- only targets top-level directories matching --pattern
- keeps the newest --keep directories regardless of age
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import time
from dataclasses import dataclass
from typing import Iterable, List, Set


@dataclass
class Candidate:
    name: str
    path: str
    mtime: float
    size_bytes: int


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += int(os.path.getsize(fp))
            except OSError:
                continue
    return int(total)


def _collect_candidates(root: str, patterns: Iterable[str], protect: Set[str]) -> List[Candidate]:
    pats = [str(p).strip() for p in patterns if str(p).strip()]
    out: List[Candidate] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if name in protect:
            continue
        if not any(fnmatch.fnmatch(name, pat) for pat in pats):
            continue
        try:
            mtime = float(os.path.getmtime(path))
        except OSError:
            mtime = 0.0
        out.append(
            Candidate(
                name=name,
                path=path,
                mtime=mtime,
                size_bytes=_dir_size_bytes(path),
            )
        )
    out.sort(key=lambda c: c.mtime, reverse=True)
    return out


def _fmt_gb(size_bytes: int) -> str:
    return f"{(float(size_bytes) / float(1 << 30)):.3f} GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--pattern", action="append", default=None)
    ap.add_argument("--protect", action="append", default=["runs", "central_memory"])
    ap.add_argument("--keep", type=int, default=6, help="Always keep newest N matching directories.")
    ap.add_argument(
        "--min_age_days",
        type=float,
        default=7.0,
        help="Only prune directories older than this age (days).",
    )
    ap.add_argument("--delete", action="store_true", help="Actually delete. Without this flag, dry-run only.")
    args = ap.parse_args()

    root = os.path.abspath(str(args.root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root not found: {root}")

    keep_n = max(0, _safe_int(args.keep, 6))
    min_age_days = max(0.0, _safe_float(args.min_age_days, 7.0))
    protect = {str(x).strip() for x in (args.protect or []) if str(x).strip()}
    patterns = list(args.pattern or [])
    if not patterns:
        patterns = ["runs*", "central_memory*"]
    candidates = _collect_candidates(root=root, patterns=patterns, protect=protect)

    if not candidates:
        print("No matching artifact directories found.")
        return

    now = time.time()
    keep = set(c.path for c in candidates[:keep_n])
    prune: List[Candidate] = []
    skipped: List[Candidate] = []

    for c in candidates:
        if c.path in keep:
            skipped.append(c)
            continue
        age_days = max(0.0, (now - float(c.mtime)) / 86400.0)
        if age_days < min_age_days:
            skipped.append(c)
            continue
        prune.append(c)

    total_prune_bytes = sum(c.size_bytes for c in prune)
    print("Artifact cleanup plan")
    print(f"root            : {root}")
    print(f"patterns        : {list(patterns)}")
    print(f"protect         : {sorted(protect)}")
    print(f"keep_newest     : {keep_n}")
    print(f"min_age_days    : {min_age_days:.2f}")
    print(f"matches         : {len(candidates)}")
    print(f"to_prune        : {len(prune)} ({_fmt_gb(total_prune_bytes)})")
    print(f"mode            : {'DELETE' if bool(args.delete) else 'DRY_RUN'}")
    print("")

    for c in prune:
        age_days = max(0.0, (now - float(c.mtime)) / 86400.0)
        print(f"PRUNE  {c.name:<40} age={age_days:>7.2f}d size={_fmt_gb(c.size_bytes)}")
    for c in skipped[: min(10, len(skipped))]:
        age_days = max(0.0, (now - float(c.mtime)) / 86400.0)
        print(f"KEEP   {c.name:<40} age={age_days:>7.2f}d size={_fmt_gb(c.size_bytes)}")

    if not args.delete:
        return

    deleted = 0
    failed = 0
    for c in prune:
        try:
            shutil.rmtree(c.path)
            deleted += 1
        except OSError as e:
            failed += 1
            print(f"FAILED {c.path}: {e}")
    print("")
    print(f"Deleted: {deleted}, Failed: {failed}")


if __name__ == "__main__":
    main()
