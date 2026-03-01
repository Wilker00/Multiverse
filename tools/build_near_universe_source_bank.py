"""
tools/build_near_universe_source_bank.py

Build a near-universe source bank for task-solving transfer.

Outputs:
- Per-verse merged DNA files under out_dir
- Manifest JSON with source metadata and recommended CLI args
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.universe_registry import build_transfer_source_plan
from tools.run_transfer_challenge import (
    _discover_transfer_sources,
    _iter_jsonl,
    _merge_jsonl,
)


def _count_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    return sum(1 for _ in _iter_jsonl(path))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_verse", type=str, default="warehouse_world")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--out_dir", type=str, default=os.path.join("models", "expert_datasets", "near_source_bank"))
    ap.add_argument("--manifest_out", type=str, default="")
    ap.add_argument("--max_runs_per_verse", type=int, default=3)
    ap.add_argument("--min_source_success_rate", type=float, default=0.60)
    ap.add_argument("--min_rows_per_source", type=int, default=80)
    ap.add_argument("--max_source_scan", type=int, default=400)
    ap.add_argument("--max_rows_per_file", type=int, default=4000)
    args = ap.parse_args()

    plan = build_transfer_source_plan(str(args.target_verse))
    near_set = set(str(v).strip().lower() for v in (plan.get("near_sources") or []) if str(v).strip())
    all_sources = _discover_transfer_sources(
        target_verse=str(args.target_verse),
        runs_root=str(args.runs_root),
        max_runs_per_verse=max(1, int(args.max_runs_per_verse)),
        min_success_rate=float(args.min_source_success_rate),
        min_rows_per_source=max(1, int(args.min_rows_per_source)),
        max_source_scan=max(0, int(args.max_source_scan)),
    )
    near_sources = [s for s in all_sources if str(s.source_lane or "") == "near_universe" and str(s.verse_name or "") in near_set]

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    by_verse: Dict[str, List[str]] = defaultdict(list)
    manifest_sources: List[Dict[str, Any]] = []
    for s in near_sources:
        by_verse[str(s.verse_name)].append(str(s.path))
        manifest_sources.append(
            {
                "verse_name": str(s.verse_name),
                "path": str(s.path).replace("\\", "/"),
                "run_id": str(s.run_id),
                "source_kind": str(s.source_kind),
                "source_lane": str(s.source_lane),
                "source_universe": str(s.source_universe),
                "rows": int(_count_rows(str(s.path))),
            }
        )

    merged_files: List[Dict[str, Any]] = []
    source_dna_args: List[str] = []
    source_verse_args: List[str] = []
    total_rows = 0
    for verse in sorted(by_verse.keys()):
        merged_path = os.path.join(out_dir, f"{verse}.jsonl")
        rows = _merge_jsonl(
            paths=list(by_verse.get(verse, [])),
            out_path=merged_path,
            max_rows_per_file=max(0, int(args.max_rows_per_file)),
        )
        if int(rows) <= 0:
            continue
        total_rows += int(rows)
        merged_files.append(
            {
                "verse_name": str(verse),
                "path": str(merged_path).replace("\\", "/"),
                "rows": int(rows),
                "inputs": list(by_verse.get(verse, [])),
            }
        )
        source_dna_args.append(str(merged_path).replace("\\", "/"))
        source_verse_args.append(str(verse))

    payload: Dict[str, Any] = {
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_verse": str(args.target_verse),
        "target_universe": plan.get("target_universe"),
        "planned_near_sources": list(plan.get("near_sources") or []),
        "discovered_near_sources": manifest_sources,
        "merged_near_bank_files": merged_files,
        "near_source_count": int(len(manifest_sources)),
        "merged_file_count": int(len(merged_files)),
        "total_merged_rows": int(total_rows),
        "recommended_cli": {
            "source_dna": list(source_dna_args),
            "source_verse": list(source_verse_args),
            "challenge_args": (
                [tok for pair in zip(["--source_dna"] * len(source_dna_args), source_dna_args) for tok in pair]
                + [tok for pair in zip(["--source_verse"] * len(source_verse_args), source_verse_args) for tok in pair]
                + ["--far_lane_enabled", "false"]
            ),
        },
    }

    manifest_out = str(args.manifest_out).strip()
    if not manifest_out:
        manifest_out = os.path.join(
            out_dir,
            f"near_source_bank_{str(args.target_verse).strip().lower()}.json",
        )
    os.makedirs(os.path.dirname(manifest_out) or ".", exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "target_verse": str(args.target_verse),
                "near_source_count": int(len(manifest_sources)),
                "merged_file_count": int(len(merged_files)),
                "total_merged_rows": int(total_rows),
                "manifest_out": str(manifest_out).replace("\\", "/"),
            },
            ensure_ascii=False,
        )
    )
    if int(len(merged_files)) <= 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

