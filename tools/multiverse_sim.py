"""
tools/multiverse_sim.py

Multi-sim control-plane helper:
- list known simulator providers
- preview the built-in local visual backend
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from integrations.local_visual_sim import preview_local_visual_sim
from integrations.sim_registry import get_sim_provider, list_sim_provider_status
from tools.evaluate_sim2real import add_cli_arguments as add_sim2real_cli_arguments
from tools.evaluate_sim2real import run_cli as run_sim2real_cli


def _parse_scalar(raw: str) -> Any:
    text = str(raw).strip()
    lo = text.lower()
    if lo in {"true", "false"}:
        return lo == "true"
    if lo in {"none", "null"}:
        return None
    try:
        if "." not in text and "e" not in lo:
            return int(text)
        return float(text)
    except Exception:
        pass
    return text


def _parse_kv_pairs(values: Optional[Sequence[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in values or []:
        token = str(raw).strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected k=v token, got '{token}'")
        key, value = token.split("=", 1)
        out[str(key).strip()] = _parse_scalar(value)
    return out


def _cmd_list(*, as_json: bool) -> int:
    rows = list_sim_provider_status()
    if bool(as_json):
        print(json.dumps({"providers": rows}, ensure_ascii=False, indent=2))
        return 0
    print("Multiverse Sim Providers")
    print("------------------------")
    for row in rows:
        print(
            f"{row['provider_id']:<16} available={row['available']} "
            f"implemented={row['implemented']} kind={row['kind']}"
        )
        print(f"  {row['description']}")
        if row.get("supported_verses"):
            print(f"  verses: {', '.join(row['supported_verses'])}")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    provider = get_sim_provider(str(args.provider))
    if provider.provider_id != "multiverse_local":
        raise RuntimeError(
            f"Provider '{provider.provider_id}' is registered but not yet previewable in-tree. "
            f"Use 'multiverse_local' for the built-in lightweight visual backend."
        )
    result = preview_local_visual_sim(
        verse_name=str(args.verse),
        verse_params=_parse_kv_pairs(args.vparam),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
    )
    if bool(args.json):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("Local Sim Preview")
    print("-----------------")
    print(f"Provider        : {result['provider_id']}")
    print(f"Verse           : {result['verse_name']}")
    for ep in result.get("episodes", []):
        print("")
        print(f"Episode {ep['episode']} seed={ep['seed']} return={ep['return_sum']:.3f} steps={ep['steps']}")
        print(ep.get("initial_frame", ""))
        if bool(args.show_final_frame) and ep.get("final_frame") != ep.get("initial_frame"):
            print("")
            print("Final")
            print(ep.get("final_frame", ""))
    return 0


def _cmd_sim2real(args: argparse.Namespace) -> int:
    return int(run_sim2real_cli(args))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Multi-sim control-plane helpers for Multiverse.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", aliases=["ls"], help="List known simulator providers.")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=lambda args: _cmd_list(as_json=bool(args.json)))

    p_preview = sub.add_parser("preview", aliases=["show"], help="Preview the built-in lightweight local visual backend.")
    p_preview.add_argument("--provider", type=str, default="multiverse_local")
    p_preview.add_argument("--verse", type=str, default="line_world")
    p_preview.add_argument("--episodes", type=int, default=1)
    p_preview.add_argument("--max_steps", type=int, default=12)
    p_preview.add_argument("--seed", type=int, default=123)
    p_preview.add_argument("--show-final-frame", dest="show_final_frame", action="store_true")
    p_preview.add_argument("--vparam", action="append", default=None)
    p_preview.add_argument("--json", action="store_true")
    p_preview.set_defaults(func=_cmd_preview)

    p_s2r = sub.add_parser(
        "sim2real",
        aliases=["s2r"],
        help="Run bounded sim-to-real stress assessment for a verse and agent.",
    )
    add_sim2real_cli_arguments(p_s2r)
    p_s2r.set_defaults(func=_cmd_sim2real)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
