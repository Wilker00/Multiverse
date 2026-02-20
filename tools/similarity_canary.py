"""
tools/similarity_canary.py

Manage and run similarity canary checks for ANN retrieval drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from memory.central_repository import (
    CentralMemoryConfig,
    run_similarity_canaries,
    save_similarity_canary,
)


def _parse_obs(raw: str) -> Any:
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"invalid JSON for --obs: {exc}") from exc
    if not isinstance(obj, (dict, list, int, float, str, bool)) and obj is not None:
        raise ValueError("--obs must be a JSON scalar/dict/list")
    return obj


def cmd_add(args: argparse.Namespace) -> None:
    cfg = CentralMemoryConfig(root_dir=str(args.memory_root))
    obs = _parse_obs(str(args.obs))
    mtypes = set(str(x).strip().lower() for x in str(args.memory_types).split(",") if str(x).strip())
    row = save_similarity_canary(
        cfg=cfg,
        canary_id=str(args.canary_id),
        obs=obs,
        expected_run_id=str(args.expected_run_id),
        top_k=max(1, int(args.top_k)),
        verse_name=(str(args.verse_name).strip().lower() if str(args.verse_name).strip() else None),
        memory_types=(mtypes or None),
    )
    print(json.dumps(row, ensure_ascii=False))


def cmd_run(args: argparse.Namespace) -> None:
    cfg = CentralMemoryConfig(root_dir=str(args.memory_root))
    out = run_similarity_canaries(cfg=cfg, limit=(None if int(args.limit) <= 0 else int(args.limit)))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if bool(args.fail_on_miss):
        if int(out.get("ann_hits", 0)) < int(out.get("total", 0)):
            raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add/update a similarity canary")
    p_add.add_argument("--memory_root", type=str, default="central_memory")
    p_add.add_argument("--canary_id", type=str, required=True)
    p_add.add_argument("--obs", type=str, required=True, help="JSON observation payload")
    p_add.add_argument("--expected_run_id", type=str, required=True)
    p_add.add_argument("--top_k", type=int, default=1)
    p_add.add_argument("--verse_name", type=str, default="")
    p_add.add_argument("--memory_types", type=str, default="")
    p_add.set_defaults(func=cmd_add)

    p_run = sub.add_parser("run", help="Execute similarity canary set")
    p_run.add_argument("--memory_root", type=str, default="central_memory")
    p_run.add_argument("--limit", type=int, default=0)
    p_run.add_argument("--fail_on_miss", action="store_true")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
