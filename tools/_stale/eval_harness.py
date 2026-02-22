"""
tools/eval_harness.py

CLI for strict reproducible A/B evaluation gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec
from orchestrator.eval_harness import gate_to_dict, print_gate_report, run_ab_gate


def _parse_json_dict(value: str | None, name: str) -> Dict[str, Any]:
    if value is None or str(value).strip() == "":
        return {}
    try:
        parsed = json.loads(value)
    except Exception as e:
        raise ValueError(f"{name} must be valid JSON object string") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _make_spec(algo: str, cfg_json: str | None, policy_id: str, seed: int) -> AgentSpec:
    return AgentSpec(
        spec_version="v1",
        policy_id=policy_id,
        policy_version="0.0",
        algo=str(algo),
        seed=int(seed),
        config=_parse_json_dict(cfg_json, f"{policy_id}_config"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_algo", type=str, required=True)
    ap.add_argument("--baseline_config_json", type=str, default="{}")
    ap.add_argument("--candidate_algo", type=str, required=True)
    ap.add_argument("--candidate_config_json", type=str, default="{}")
    ap.add_argument("--suite", type=str, default="target", choices=["target", "quick", "full"])
    ap.add_argument("--verse", type=str, default=None, help="Required when --suite=target")
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--fail_on_gate", action="store_true")
    args = ap.parse_args()

    baseline = _make_spec(args.baseline_algo, args.baseline_config_json, "baseline", args.seed)
    candidate = _make_spec(args.candidate_algo, args.candidate_config_json, "candidate", args.seed)
    result = run_ab_gate(
        baseline_spec=baseline,
        candidate_spec=candidate,
        suite_mode=args.suite,
        target_verse=args.verse,
        bootstrap_samples=args.bootstrap_samples,
        alpha=args.alpha,
    )
    print_gate_report(result, label="cli_ab")

    if args.out_json:
        payload = gate_to_dict(result)
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"saved_json: {args.out_json}")

    if args.fail_on_gate and not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
