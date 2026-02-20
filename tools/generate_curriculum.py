"""
tools/generate_curriculum.py

Generate ACED-style curriculum verse specs from trajectory data.
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

from orchestrator.curriculum_gen import generate_task_sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_dir", type=str, default="runs", help="runs root or run dir")
    ap.add_argument("--verse", type=str, default=None)
    ap.add_argument("--max_specs", type=int, default=8)
    ap.add_argument("--success_target", type=float, default=0.8)
    ap.add_argument("--advance_ratio", type=float, default=0.1)
    ap.add_argument("--out_path", type=str, default=os.path.join("experiment", "curriculum_specs.json"))
    args = ap.parse_args()

    specs = generate_task_sequence(
        args.memory_dir,
        verse_name=args.verse,
        max_specs=args.max_specs,
        success_target=args.success_target,
        advance_ratio=args.advance_ratio,
    )
    payload = [s.to_dict() for s in specs]
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Curriculum generation complete")
    print(f"memory_dir : {args.memory_dir}")
    print(f"verse      : {args.verse}")
    print(f"specs      : {len(payload)}")
    print(f"out_path   : {args.out_path}")


if __name__ == "__main__":
    main()
