"""
tools/active_forgetting.py

Central-memory curation:
- Run-level deduplication (default)
- Optional quality gating for low-value/high-risk runs
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.selection import ActiveForgettingConfig, active_forget_central_memory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_dir", type=str, default="central_memory")
    ap.add_argument("--similarity_threshold", type=float, default=0.95)
    ap.add_argument("--min_events_per_run", type=int, default=20)
    ap.add_argument("--quality_filter", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min_mean_reward", type=float, default=-1e9)
    ap.add_argument("--min_success_rate", type=float, default=0.0)
    ap.add_argument("--max_hazard_rate", type=float, default=1.0)
    ap.add_argument("--no_backup", action="store_true", help="Skip writing memories.jsonl.bak before pruning.")
    args = ap.parse_args()

    st = active_forget_central_memory(
        ActiveForgettingConfig(
            memory_dir=args.memory_dir,
            similarity_threshold=float(args.similarity_threshold),
            min_events_per_run=int(args.min_events_per_run),
            write_backup=not bool(args.no_backup),
            quality_filter_enabled=bool(args.quality_filter),
            min_mean_reward=float(args.min_mean_reward),
            min_success_rate=float(args.min_success_rate),
            max_hazard_rate=float(args.max_hazard_rate),
        )
    )
    print("Central memory curation summary")
    print(f"mode          : {'dedupe+quality' if bool(args.quality_filter) else 'dedupe-only'}")
    print(f"memory_dir    : {args.memory_dir}")
    print(f"input_rows    : {st.input_rows}")
    print(f"kept_rows     : {st.kept_rows}")
    print(f"dropped_rows  : {st.dropped_rows}")
    print(f"input_runs    : {st.input_runs}")
    print(f"kept_runs     : {st.kept_runs}")
    print(f"dropped_runs  : {st.dropped_runs}")
    if bool(args.quality_filter):
        print(f"quality_min_mean_reward : {float(args.min_mean_reward):.6g}")
        print(f"quality_min_success_rate: {float(args.min_success_rate):.6g}")
        print(f"quality_max_hazard_rate : {float(args.max_hazard_rate):.6g}")
        print(f"quality_dropped_runs    : {st.dropped_low_quality_runs}")
        print(f"quality_dropped_rows    : {st.dropped_low_quality_rows}")
    if st.dropped_run_ids:
        print(f"dropped_ids   : {', '.join(st.dropped_run_ids[:20])}")
        if len(st.dropped_run_ids) > 20:
            print(f"dropped_ids+  : {len(st.dropped_run_ids) - 20} more")


if __name__ == "__main__":
    main()
