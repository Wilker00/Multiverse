#!/usr/bin/env python3
"""
Build warehouse expert dataset by merging:
1) translated synthetic data from existing verse experts
2) optional real warehouse dataset from a run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.imitation_agent import obs_key
from memory.semantic_bridge import translate_dna


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in rows:
        key = (obs_key(row.get("obs")), json.dumps(row.get("action"), ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_dir", type=str, default=os.path.join("models", "expert_datasets"))
    ap.add_argument("--grid_source", type=str, default=os.path.join("models", "expert_datasets", "grid_world.jsonl"))
    ap.add_argument("--line_source", type=str, default=os.path.join("models", "expert_datasets", "line_world.jsonl"))
    ap.add_argument("--warehouse_real", type=str, default=None, help="Optional real warehouse dataset path")
    ap.add_argument("--out", type=str, default=os.path.join("models", "expert_datasets", "warehouse_world.jsonl"))
    args = ap.parse_args()

    expert_dir = Path(args.expert_dir)
    expert_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    translated_stats: List[Tuple[str, int]] = []

    for src_path in [args.grid_source, args.line_source]:
        if not os.path.isfile(src_path):
            continue
        src_name = Path(src_path).stem
        syn_path = expert_dir / f"synthetic_transfer_{src_name}_to_warehouse_world.jsonl"
        stats = translate_dna(
            source_dna_path=src_path,
            target_verse_name="warehouse_world",
            output_path=str(syn_path),
        )
        rows = list(_iter_jsonl(str(syn_path)))
        translated_stats.append((src_name, len(rows)))
        all_rows.extend(rows)

    real_candidates = []
    if args.warehouse_real:
        real_candidates.append(args.warehouse_real)
    real_candidates.append(str(expert_dir / "warehouse_world_real.jsonl"))
    real_candidates.append(str(expert_dir / "warehouse_world.jsonl"))

    for rp in real_candidates:
        if os.path.isfile(rp):
            all_rows.extend(_iter_jsonl(rp))

    deduped = _dedupe_rows(all_rows)
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(deduped):
            row.setdefault("episode_id", f"ep_syn_{i:06d}")
            row.setdefault("step_idx", i)
            row.setdefault("target_verse_name", "warehouse_world")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("warehouse_expert_build")
    print(f"output={out_path}")
    print(f"rows_total={len(all_rows)} rows_deduped={len(deduped)}")
    for name, n in translated_stats:
        print(f"translated_{name}={n}")


if __name__ == "__main__":
    main()

