"""
experiment/runs.py

Run directory helpers for u.ai.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RunInfo:
    run_id: str
    run_dir: str
    mtime: float


def list_runs(root_dir: str = "runs") -> List[RunInfo]:
    if not os.path.isdir(root_dir):
        return []

    infos: List[RunInfo] = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            infos.append(RunInfo(run_id=name, run_dir=path, mtime=os.path.getmtime(path)))

    infos.sort(key=lambda r: r.mtime, reverse=True)
    return infos


def latest_run(root_dir: str = "runs") -> Optional[RunInfo]:
    runs = list_runs(root_dir=root_dir)
    return runs[0] if runs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs")
    ap.add_argument("--latest", action="store_true", help="print only the latest run")
    args = ap.parse_args()

    if args.latest:
        info = latest_run(args.root)
        if not info:
            print("No runs found.")
            return
        print(info.run_dir)
        return

    for info in list_runs(args.root):
        print(f"{info.run_id}  {info.run_dir}")


if __name__ == "__main__":
    main()
