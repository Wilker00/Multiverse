"""
tools/knowledge_graph.py

CLI for the lightweight relational knowledge graph.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.knowledge_graph import (
    KnowledgeGraphConfig,
    concept_closure_for_verse,
    ensure_graph,
    load_graph,
    verse_relatedness,
)


def cmd_init(args: argparse.Namespace) -> None:
    path = ensure_graph(KnowledgeGraphConfig(root_dir=args.memory_dir))
    print(f"knowledge graph ready: {path}")


def cmd_related(args: argparse.Namespace) -> None:
    cfg = KnowledgeGraphConfig(root_dir=args.memory_dir)
    graph = load_graph(cfg)
    score = verse_relatedness(args.source_verse, args.target_verse, graph)
    print("Knowledge graph relatedness")
    print(f"source     : {args.source_verse}")
    print(f"target     : {args.target_verse}")
    print(f"score      : {score:.3f}")


def cmd_closure(args: argparse.Namespace) -> None:
    cfg = KnowledgeGraphConfig(root_dir=args.memory_dir)
    graph = load_graph(cfg)
    closure = sorted(concept_closure_for_verse(args.verse, graph))
    print("Knowledge graph closure")
    print(f"verse      : {args.verse}")
    print(f"concepts   : {len(closure)}")
    for c in closure:
        print(f"- {c}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="create default graph file if missing")
    i.add_argument("--memory_dir", type=str, default="central_memory")
    i.set_defaults(func=cmd_init)

    r = sub.add_parser("related", help="compute graph relatedness between two verses")
    r.add_argument("--memory_dir", type=str, default="central_memory")
    r.add_argument("--source_verse", type=str, required=True)
    r.add_argument("--target_verse", type=str, required=True)
    r.set_defaults(func=cmd_related)

    c = sub.add_parser("closure", help="show closure/ancestors for a verse")
    c.add_argument("--memory_dir", type=str, default="central_memory")
    c.add_argument("--verse", type=str, required=True)
    c.set_defaults(func=cmd_closure)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
