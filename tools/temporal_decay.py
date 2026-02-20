"""
tools/temporal_decay.py

Applies temporal decay archiving for stale centralized memories.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.decay_manager import DecayConfig, archive_stale_memories


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_dir", type=str, default="central_memory")
    ap.add_argument("--decay_lambda", type=float, default=1e-10)
    ap.add_argument("--stale_threshold", type=float, default=0.1)
    ap.add_argument("--archive_filename", type=str, default="memories.archive.jsonl")
    args = ap.parse_args()

    st = archive_stale_memories(
        memory_dir=args.memory_dir,
        archive_filename=args.archive_filename,
        cfg=DecayConfig(
            decay_lambda=float(args.decay_lambda),
            stale_threshold=float(args.stale_threshold),
        ),
    )
    print("Temporal decay archive complete")
    print(f"memory_dir    : {args.memory_dir}")
    print(f"input_rows    : {st.input_rows}")
    print(f"active_rows   : {st.active_rows}")
    print(f"archived_rows : {st.archived_rows}")
    print(f"archive_path  : {st.archive_path}")


if __name__ == "__main__":
    main()
