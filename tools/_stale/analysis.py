"""
tools/analysis.py

Run analysis helper.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from orchestrator.evaluator import evaluate_run, print_report
from memory.episode_index import EpisodeIndexConfig, build_episode_index
from memory.summarizer import SummaryConfig, summarize_run


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--make_index", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--top", type=int, default=0, help="show top-N returns if episodes.jsonl exists")
    args = ap.parse_args()

    stats = evaluate_run(args.run_dir)
    print_report(stats)

    if args.make_index:
        idx = build_episode_index(EpisodeIndexConfig(run_dir=args.run_dir))
        print("")
        print(f"Built episode index: {idx}")

    if args.summarize:
        out = summarize_run(SummaryConfig(run_dir=args.run_dir))
        print("")
        print(f"Wrote summaries: {out}")

    if args.top > 0:
        episodes_path = os.path.join(args.run_dir, "episodes.jsonl")
        if os.path.isfile(episodes_path):
            rows = _load_jsonl(episodes_path)
            rows.sort(key=lambda r: float(r.get("return_sum", 0.0)), reverse=True)
            print("")
            print(f"Top {args.top} episodes by return:")
            for row in rows[: args.top]:
                print(f"{row.get('episode_id')}  return={float(row.get('return_sum', 0.0)):.3f}")


if __name__ == "__main__":
    main()




