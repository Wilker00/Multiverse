"""
tools/confidence_auditor.py

CLI for the semantic-bridge confidence auditor.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.confidence_auditor import (
    ConfidenceAuditConfig,
    OutcomeSample,
    get_bridge_weight,
    load_state,
    record_outcomes_batch,
    summarize_state,
)


def cmd_summary(args: argparse.Namespace) -> None:
    cfg = ConfidenceAuditConfig(root_dir=args.memory_dir)
    state = load_state(cfg)
    rows = summarize_state(state)
    print("Confidence auditor summary")
    print(f"memory_dir : {args.memory_dir}")
    print(f"entries    : {len(rows)}")
    if not rows:
        return
    for r in rows[: args.limit]:
        print(
            f"- task={r['task']} bridge={r['bridge']} "
            f"weight={r['weight']:.3f} delta={r['delta_vs_direct']:.3f} "
            f"direct_n={r['direct_count']} bridge_n={r['bridge_count']}"
        )
    if len(rows) > args.limit:
        print(f"... {len(rows) - args.limit} more")


def cmd_record(args: argparse.Namespace) -> None:
    cfg = ConfidenceAuditConfig(root_dir=args.memory_dir)
    sample = OutcomeSample(
        strategy=str(args.strategy),
        target_verse=str(args.target_verse),
        reward=float(args.reward),
        source_verse=(None if args.source_verse is None else str(args.source_verse)),
        task_tag=(None if args.task_tag is None else str(args.task_tag)),
    )
    state = record_outcomes_batch([sample], cfg)
    rows = summarize_state(state)
    print("Recorded outcome")
    print(f"strategy   : {args.strategy}")
    print(f"target     : {args.target_verse}")
    print(f"source     : {args.source_verse}")
    print(f"reward     : {float(args.reward):.3f}")
    print(f"rows       : {len(rows)}")


def cmd_weight(args: argparse.Namespace) -> None:
    cfg = ConfidenceAuditConfig(root_dir=args.memory_dir)
    w = get_bridge_weight(
        source_verse=args.source_verse,
        target_verse=args.target_verse,
        task_tag=args.task_tag,
        cfg=cfg,
    )
    print("Bridge weight")
    print(f"source     : {args.source_verse}")
    print(f"target     : {args.target_verse}")
    print(f"task_tag   : {args.task_tag}")
    print(f"weight     : {w:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help="print confidence-auditor summary")
    s.add_argument("--memory_dir", type=str, default="central_memory")
    s.add_argument("--limit", type=int, default=30)
    s.set_defaults(func=cmd_summary)

    r = sub.add_parser("record", help="record one reward outcome sample")
    r.add_argument("--memory_dir", type=str, default="central_memory")
    r.add_argument("--strategy", type=str, required=True, choices=["direct", "semantic_bridge", "hybrid_low_confidence"])
    r.add_argument("--target_verse", type=str, required=True)
    r.add_argument("--reward", type=float, required=True)
    r.add_argument("--source_verse", type=str, default=None)
    r.add_argument("--task_tag", type=str, default=None)
    r.set_defaults(func=cmd_record)

    w = sub.add_parser("weight", help="query bridge trust weight")
    w.add_argument("--memory_dir", type=str, default="central_memory")
    w.add_argument("--source_verse", type=str, required=True)
    w.add_argument("--target_verse", type=str, required=True)
    w.add_argument("--task_tag", type=str, default=None)
    w.set_defaults(func=cmd_weight)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
