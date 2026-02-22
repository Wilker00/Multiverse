"""
tools/evolve_policy.py

Combine two datasets into a hybrid dataset.
"""

from __future__ import annotations

import argparse

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.evolution import merge_datasets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=str, required=True, help="dataset A")
    ap.add_argument("--b", type=str, required=True, help="dataset B")
    ap.add_argument("--out", type=str, required=True, help="output dataset")
    ap.add_argument("--ratio_a", type=float, default=0.5)
    args = ap.parse_args()

    out = merge_datasets(args.a, args.b, args.out, ratio_a=args.ratio_a)
    print(f"Hybrid dataset: {out}")


if __name__ == "__main__":
    main()




