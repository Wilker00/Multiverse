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
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from integrations.local_visual_sim import preview_local_visual_sim
from integrations.sim_registry import get_sim_provider, list_sim_provider_status
from tools.evaluate_sim2real import assess_sim2real, list_sim2real_profiles


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


def _parse_profile_list(raw: str) -> List[str]:
    items = [str(x).strip().lower() for x in str(raw or "").replace(";", ",").split(",") if str(x).strip()]
    if not items or (len(items) == 1 and items[0] == "all"):
        return list_sim2real_profiles()
    return items


def _cmd_sim2real(args: argparse.Namespace) -> int:
    verse_params = _parse_kv_pairs(args.vparam)
    agent_config = _parse_kv_pairs(args.aconfig)
    if bool(args.train):
        agent_config.setdefault("train", True)
    if str(args.manifest_path or "").strip():
        agent_config.setdefault("manifest_path", str(args.manifest_path).strip())

    report = assess_sim2real(
        verse_name=str(args.verse),
        algo=str(args.algo),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        runs_root=str(args.runs_root),
        profiles=_parse_profile_list(str(args.profiles)),
        verse_params=verse_params,
        agent_config=agent_config,
        max_success_rate_drop=float(args.max_success_rate_drop),
        max_return_drop=float(args.max_return_drop),
    )

    if str(args.out_json or "").strip():
        out_path = str(args.out_json).strip()
        import os
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            import json as _json
            _json.dump(report, fh, ensure_ascii=False, indent=2)

    if bool(args.json):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    summary = dict(report.get("summary") or {})
    print("Sim-to-Real Assessment")
    print("----------------------")
    print(f"Verse           : {report.get('verse_name')}")
    print(f"Algo            : {report.get('algo')}")
    print(f"Episodes        : {report.get('episodes')}")
    print(f"Max steps       : {report.get('max_steps')}")
    print(f"Passed          : {summary.get('passed')}")
    base = dict((summary.get("baseline") or {}))
    base_metrics = dict(base.get("metrics") or {})
    print("")
    print("Baseline")
    print(f"  mean_return   : {base_metrics.get('mean_return')}")
    print(f"  success_rate  : {base_metrics.get('success_rate')}")
    print("")
    print("Profiles")
    for row in summary.get("comparisons") or []:
        print(
            f"  {row['name']:<10} pass={row['passed']} "
            f"return_drop={row['mean_return_drop']:.4f} "
            f"support={row['support_level']}"
        )
    if str(args.out_json or "").strip():
        print("")
        print(f"report: {str(args.out_json).strip()}")
    return 0


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
    p_s2r.add_argument("--verse", type=str, default="warehouse_world")
    p_s2r.add_argument("--algo", type=str, default="random")
    p_s2r.add_argument("--episodes", type=int, default=10)
    p_s2r.add_argument("--max_steps", type=int, default=60)
    p_s2r.add_argument("--seed", type=int, default=123)
    p_s2r.add_argument("--runs_root", type=str, default="runs")
    p_s2r.add_argument("--profiles", type=str, default="all",
                       help="Comma-separated list of profiles or 'all'. Choices: mild, moderate, severe.")
    p_s2r.add_argument("--vparam", action="append", default=None, help="Verse param override k=v")
    p_s2r.add_argument("--aconfig", action="append", default=None, help="Agent config override k=v")
    p_s2r.add_argument("--manifest_path", type=str, default="")
    p_s2r.add_argument("--train", action="store_true")
    p_s2r.add_argument("--max_success_rate_drop", type=float, default=0.15)
    p_s2r.add_argument("--max_return_drop", type=float, default=2.0)
    p_s2r.add_argument("--out_json", type=str, default="")
    p_s2r.add_argument("--json", action="store_true")
    p_s2r.set_defaults(func=_cmd_sim2real)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
