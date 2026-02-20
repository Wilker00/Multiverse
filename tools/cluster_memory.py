"""
tools/cluster_memory.py

Cluster interactions into task-like groups.
"""

from __future__ import annotations

import argparse

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.clustering import ClusterConfig, cluster_events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_iters", type=int, default=20)
    args = ap.parse_args()

    out = cluster_events(ClusterConfig(run_dir=args.run_dir, k=args.k, max_iters=args.max_iters))
    print(f"Wrote clusters: {out}")


if __name__ == "__main__":
    main()




