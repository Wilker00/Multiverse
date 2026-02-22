"""
tools/value_baseline.py

Build a value baseline V(s) from event logs.
"""

from __future__ import annotations

import argparse

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.value_baseline import ValueBaselineConfig, build_value_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--output", type=str, default="value_baseline.jsonl")
    args = ap.parse_args()

    cfg = ValueBaselineConfig(run_dir=args.run_dir, gamma=args.gamma, output_filename=args.output)
    out = build_value_baseline(cfg)
    print(f"Wrote value baseline: {out}")


if __name__ == "__main__":
    main()




