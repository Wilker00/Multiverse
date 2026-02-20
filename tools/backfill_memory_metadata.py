"""
tools/backfill_memory_metadata.py

One-shot metadata enrichment for central memory rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.central_repository import CentralMemoryConfig, backfill_memory_metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--central_memory_dir", type=str, default="central_memory")
    ap.add_argument(
        "--tier_policy_filename",
        type=str,
        default="tier_policy.json",
        help="Absolute path or path relative to --central_memory_dir.",
    )
    ap.add_argument("--rebuild_tier_files", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--recompute_tier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute memory_tier even when already present.",
    )
    ap.add_argument(
        "--apply_support_guards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap LTM ratios for low-support verses using policy settings.",
    )
    args = ap.parse_args()

    stats = backfill_memory_metadata(
        cfg=CentralMemoryConfig(
            root_dir=str(args.central_memory_dir),
            tier_policy_filename=str(args.tier_policy_filename),
        ),
        rebuild_tier_files=bool(args.rebuild_tier_files),
        recompute_tier=bool(args.recompute_tier),
        apply_support_guards=bool(args.apply_support_guards),
    )
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
