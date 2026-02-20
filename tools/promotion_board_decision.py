"""
tools/promotion_board_decision.py

Record human Bless/Veto decisions used by orchestrator/promotion_board.py.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from orchestrator.promotion_board import record_human_decision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default=os.path.join("models", "promotion_board_human_decisions.json"))
    ap.add_argument("--target_verse", type=str, required=True)
    ap.add_argument("--candidate_policy_id", type=str, required=True)
    ap.add_argument("--decision", type=str, required=True, choices=["bless", "veto"])
    ap.add_argument("--actor", type=str, default="operator")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    row = record_human_decision(
        path=str(args.path),
        target_verse=str(args.target_verse),
        candidate_policy_id=str(args.candidate_policy_id),
        decision=str(args.decision),
        actor=str(args.actor),
        note=str(args.note),
    )
    print(
        "recorded_decision "
        f"target_verse={row['target_verse']} "
        f"candidate_policy_id={row['candidate_policy_id']} "
        f"decision={row['decision']}"
    )


if __name__ == "__main__":
    main()

