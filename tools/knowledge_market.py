"""
tools/knowledge_market.py

CLI for reputation-driven DNA recommendation market.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.knowledge_market import KnowledgeMarket, KnowledgeMarketConfig


def cmd_bid(args: argparse.Namespace) -> None:
    market = KnowledgeMarket(KnowledgeMarketConfig(root_dir=args.memory_dir))
    tags = [t.strip() for t in args.task_tags.split(",") if t.strip()]
    offers = market.bid_for_dna(
        agent_id=args.agent_id,
        task_tags=tags,
        verse_name=args.verse,
        top_k=args.top_k,
    )
    print("Knowledge market offers")
    print(f"agent_id : {args.agent_id}")
    print(f"offers   : {len(offers)}")
    for o in offers:
        print(
            f"- provider={o['provider_id']} verse={o['verse_name']} "
            f"score={o['transfer_potential']:.3f} rep={o['reputation']:.3f} "
            f"paths={o['dna_paths']}"
        )


def cmd_reward(args: argparse.Namespace) -> None:
    market = KnowledgeMarket(KnowledgeMarketConfig(root_dir=args.memory_dir))
    out = market.update_reputation(
        provider_id=args.provider_id,
        delta=float(args.delta),
        reason=args.reason,
        consumer_agent_id=args.agent_id,
    )
    print("Reputation updated")
    print(f"provider : {out['provider_id']}")
    print(f"old      : {out['old_reputation']:.3f}")
    print(f"new      : {out['new_reputation']:.3f}")
    print(f"reason   : {out['reason']}")


def cmd_summary(args: argparse.Namespace) -> None:
    market = KnowledgeMarket(KnowledgeMarketConfig(root_dir=args.memory_dir))
    s = market.summary()
    print("Knowledge market summary")
    print(f"providers    : {s['providers']}")
    print(f"transactions : {s['transactions']}")
    for pid, rep in s["top_reputation"]:
        print(f"- {pid}: {rep:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bid", help="request top DNA offers for an agent")
    b.add_argument("--memory_dir", type=str, default="central_memory")
    b.add_argument("--agent_id", type=str, required=True)
    b.add_argument("--task_tags", type=str, required=True, help="comma-separated tags")
    b.add_argument("--verse", type=str, default=None)
    b.add_argument("--top_k", type=int, default=5)
    b.set_defaults(func=cmd_bid)

    r = sub.add_parser("reward", help="update provider reputation by observed transfer quality")
    r.add_argument("--memory_dir", type=str, default="central_memory")
    r.add_argument("--provider_id", type=str, required=True)
    r.add_argument("--delta", type=float, required=True)
    r.add_argument("--reason", type=str, required=True)
    r.add_argument("--agent_id", type=str, default=None)
    r.set_defaults(func=cmd_reward)

    s = sub.add_parser("summary", help="print market summary")
    s.add_argument("--memory_dir", type=str, default="central_memory")
    s.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
