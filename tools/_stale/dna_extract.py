"""
tools/dna_extract.py

Extract good/bad DNA from a run based on advantage.
"""

from __future__ import annotations

import argparse

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.advantage import AdvantageConfig, extract_dna


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--good_out", type=str, default="dna_good.jsonl")
    ap.add_argument("--bad_out", type=str, default="dna_bad.jsonl")
    ap.add_argument("--adv_thresh", type=float, default=0.0)
    ap.add_argument("--baseline_by", type=str, default="verse_name")
    ap.add_argument("--value_baseline", type=str, default=None, help="value_baseline.jsonl path")
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()

    cfg = AdvantageConfig(
        run_dir=args.run_dir,
        good_output=args.good_out,
        bad_output=args.bad_out,
        advantage_threshold=args.adv_thresh,
        baseline_by=args.baseline_by,
        value_baseline_path=args.value_baseline,
        gamma=args.gamma,
    )
    good, bad = extract_dna(cfg)
    print(f"Good DNA: {good}")
    print(f"Bad DNA: {bad}")


if __name__ == "__main__":
    main()




