"""
tools/evaluate_sim2real.py

Bounded sim-to-real assessment:
- run a nominal evaluation
- run one or more perturbed sim-to-real profiles
- summarize robustness deltas against the nominal baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.sim2real import build_sim2real_profiles, list_sim2real_profiles
from core.types import AgentSpec, VerseSpec
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


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
    try:
        decoded = json.loads(text)
        if isinstance(decoded, (dict, list)):
            return decoded
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
        name = str(key).strip()
        if not name:
            raise ValueError(f"Expected non-empty key in '{token}'")
        out[name] = _parse_scalar(value)
    return out


def _parse_profile_list(raw: str) -> List[str]:
    items = [str(x).strip().lower() for x in str(raw or "").replace(";", ",").split(",") if str(x).strip()]
    if not items:
        return list_sim2real_profiles()
    if len(items) == 1 and items[0] == "all":
        return list_sim2real_profiles()
    return items


def _run_once(
    *,
    verse_name: str,
    verse_params: Dict[str, Any],
    algo: str,
    agent_config: Dict[str, Any],
    episodes: int,
    max_steps: int,
    seed: int,
    runs_root: str,
    tag: str,
) -> Dict[str, Any]:
    trainer = Trainer(run_root=str(runs_root), schema_version="v1", auto_register_builtin=True)
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=str(verse_name),
        verse_version="0.1",
        seed=int(seed),
        tags=["sim2real", str(tag)],
        params=dict(verse_params),
    )
    cfg = dict(agent_config)
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=f"{algo}_{tag}",
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
        tags=["sim2real", str(tag)],
        config=(cfg if cfg else None),
    )
    result = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
        verbose=False,
    )
    run_id = str(result.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError(f"Trainer did not return run_id: {result}")
    stats = evaluate_run(os.path.join(str(runs_root), run_id))
    return {
        "run_id": run_id,
        "metrics": {
            "mean_return": float(stats.mean_return),
            "success_rate": (None if stats.success_rate is None else float(stats.success_rate)),
            "mean_steps": float(stats.mean_steps),
            "episodes": int(stats.episodes),
        },
    }


def summarize_sim2real_results(
    results: Sequence[Dict[str, Any]],
    *,
    max_success_rate_drop: float,
    max_return_drop: float,
) -> Dict[str, Any]:
    baseline = next((row for row in results if str(row.get("name")) == "nominal"), None)
    if baseline is None:
        raise ValueError("Sim-to-real results require a nominal baseline row")

    base_metrics = dict(baseline.get("metrics") or {})
    base_success = base_metrics.get("success_rate")
    base_return = _safe_float(base_metrics.get("mean_return"), 0.0)

    comparisons: List[Dict[str, Any]] = []
    worst_success_drop = 0.0
    worst_return_drop = 0.0
    for row in results:
        if str(row.get("name")) == "nominal":
            continue
        metrics = dict(row.get("metrics") or {})
        success = metrics.get("success_rate")
        ret = _safe_float(metrics.get("mean_return"), 0.0)
        success_drop = None
        if isinstance(base_success, (int, float)) and isinstance(success, (int, float)):
            success_drop = max(0.0, float(base_success) - float(success))
            worst_success_drop = max(worst_success_drop, float(success_drop))
        return_drop = max(0.0, float(base_return) - float(ret))
        worst_return_drop = max(worst_return_drop, float(return_drop))
        passed = (success_drop is None or float(success_drop) <= float(max_success_rate_drop)) and float(return_drop) <= float(max_return_drop)
        comparisons.append(
            {
                "name": str(row.get("name")),
                "support_level": str(row.get("support_level", "")),
                "success_rate_drop": success_drop,
                "mean_return_drop": float(return_drop),
                "passed": bool(passed),
            }
        )

    overall_passed = all(bool(row.get("passed", False)) for row in comparisons) if comparisons else True
    return {
        "baseline": {
            "name": "nominal",
            "metrics": base_metrics,
        },
        "thresholds": {
            "max_success_rate_drop": float(max_success_rate_drop),
            "max_return_drop": float(max_return_drop),
        },
        "comparisons": comparisons,
        "worst_case": {
            "success_rate_drop": float(worst_success_drop),
            "mean_return_drop": float(worst_return_drop),
        },
        "passed": bool(overall_passed),
    }


def assess_sim2real(
    *,
    verse_name: str,
    algo: str,
    episodes: int,
    max_steps: int,
    seed: int,
    runs_root: str,
    profiles: Sequence[str],
    verse_params: Optional[Dict[str, Any]] = None,
    agent_config: Optional[Dict[str, Any]] = None,
    max_success_rate_drop: float = 0.15,
    max_return_drop: float = 2.0,
) -> Dict[str, Any]:
    base_params = dict(verse_params or {})
    base_params.setdefault("max_steps", int(max_steps))
    base_params.setdefault("adr_enabled", False)

    run_rows: List[Dict[str, Any]] = []
    nominal = _run_once(
        verse_name=str(verse_name),
        verse_params=base_params,
        algo=str(algo),
        agent_config=dict(agent_config or {}),
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
        runs_root=str(runs_root),
        tag="nominal",
    )
    run_rows.append(
        {
            "name": "nominal",
            "support_level": "baseline",
            "description": "Nominal simulator baseline",
            "params": dict(base_params),
            "param_changes": {},
            "notes": [],
            **nominal,
        }
    )

    for idx, profile in enumerate(profiles, start=1):
        built = build_sim2real_profiles(
            verse_name=str(verse_name),
            base_params=base_params,
            profiles=[str(profile)],
        )[0]
        result = _run_once(
            verse_name=str(verse_name),
            verse_params=dict(built.params),
            algo=str(algo),
            agent_config=dict(agent_config or {}),
            episodes=int(episodes),
            max_steps=int(max_steps),
            seed=int(seed + idx),
            runs_root=str(runs_root),
            tag=str(built.name),
        )
        run_rows.append(
            {
                "name": str(built.name),
                "support_level": str(built.support_level),
                "description": str(built.description),
                "params": dict(built.params),
                "param_changes": dict(built.param_changes),
                "notes": list(built.notes),
                **result,
            }
        )

    summary = summarize_sim2real_results(
        run_rows,
        max_success_rate_drop=float(max_success_rate_drop),
        max_return_drop=float(max_return_drop),
    )
    return {
        "verse_name": str(verse_name),
        "algo": str(algo),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "profiles": run_rows,
        "summary": summary,
    }


def _print_human_report(report: Dict[str, Any]) -> None:
    summary = dict(report.get("summary") or {})
    print("Sim-to-Real Assessment")
    print("----------------------")
    print(f"Verse           : {report.get('verse_name')}")
    print(f"Algo            : {report.get('algo')}")
    print(f"Episodes        : {report.get('episodes')}")
    print(f"Max steps       : {report.get('max_steps')}")
    print(f"Passed          : {summary.get('passed')}")
    print("")
    base = dict(summary.get("baseline") or {})
    base_metrics = dict(base.get("metrics") or {})
    print("Baseline")
    print("--------")
    print(f"Mean return     : {base_metrics.get('mean_return')}")
    print(f"Success rate    : {base_metrics.get('success_rate')}")
    print(f"Mean steps      : {base_metrics.get('mean_steps')}")
    print("")
    print("Profiles")
    print("--------")
    for row in summary.get("comparisons") or []:
        print(
            f"{row['name']:<10} pass={row['passed']} "
            f"success_drop={row['success_rate_drop']} "
            f"return_drop={row['mean_return_drop']:.4f} "
            f"support={row['support_level']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an agent under bounded sim-to-real stress profiles.")
    ap.add_argument("--verse", type=str, default="warehouse_world")
    ap.add_argument("--algo", type=str, default="gateway")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--profiles", type=str, default="all", help="Comma-separated profile list or 'all'.")
    ap.add_argument("--vparam", action="append", default=None, help="Verse param override k=v")
    ap.add_argument("--aconfig", action="append", default=None, help="Agent config override k=v")
    ap.add_argument("--manifest_path", type=str, default="", help="Convenience gateway manifest path.")
    ap.add_argument("--train", action="store_true", help="Set agent config train=true for trainable agents.")
    ap.add_argument("--max_success_rate_drop", type=float, default=0.15)
    ap.add_argument("--max_return_drop", type=float, default=2.0)
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    verse_params = _parse_kv_pairs(args.vparam)
    agent_config = _parse_kv_pairs(args.aconfig)
    if bool(args.train):
        agent_config.setdefault("train", True)
    if str(args.manifest_path).strip():
        agent_config.setdefault("manifest_path", str(args.manifest_path).strip())

    report = assess_sim2real(
        verse_name=str(args.verse),
        algo=str(args.algo),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        runs_root=str(args.runs_root),
        profiles=_parse_profile_list(args.profiles),
        verse_params=verse_params,
        agent_config=agent_config,
        max_success_rate_drop=float(args.max_success_rate_drop),
        max_return_drop=float(args.max_return_drop),
    )

    if str(args.out_json).strip():
        out_path = str(args.out_json).strip()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if bool(args.json):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    _print_human_report(report)
    if str(args.out_json).strip():
        print("")
        print(f"report: {str(args.out_json).strip()}")


if __name__ == "__main__":
    main()
