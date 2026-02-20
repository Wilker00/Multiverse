"""
tools/production_readiness_gate.py

Production-readiness quality gate for model manifests and benchmark outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_json_obj(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def _parse_iso_utc(value: str) -> dt.datetime:
    v = str(value).strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    return dt.datetime.fromisoformat(v).astimezone(dt.timezone.utc)


def _manifest_entries(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    dep = manifest.get("deployment_ready_defaults")
    if isinstance(dep, dict) and dep:
        return {str(k): v for k, v in dep.items() if isinstance(v, dict)}
    robust = manifest.get("winners_robust")
    if isinstance(robust, dict) and robust:
        return {
            str(k): {"picked_run": v}
            for k, v in robust.items()
            if isinstance(v, dict)
        }
    return {}


def _validate_manifest(
    manifest: Dict[str, Any],
    *,
    min_verses: int,
    min_episodes: int,
    min_success_rate: float,
    require_run_dirs: bool,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    entries = _manifest_entries(manifest)
    if len(entries) < int(min_verses):
        errors.append(f"manifest has {len(entries)} verse entries, requires >= {int(min_verses)}")

    per_verse: Dict[str, Any] = {}
    for verse, entry in sorted(entries.items()):
        picked = entry.get("picked_run")
        if not isinstance(picked, dict):
            errors.append(f"{verse}: missing picked_run")
            continue
        run_id = str(picked.get("run_id", "")).strip()
        run_dir = str(picked.get("run_dir", "")).strip()
        policy = str(picked.get("policy", "")).strip()
        episodes = _safe_int(picked.get("episodes"), 0)
        success_rate = _safe_float(picked.get("success_rate"), -1.0)
        mean_return = _safe_float(picked.get("mean_return"), 0.0)

        if not run_id:
            errors.append(f"{verse}: missing run_id")
        if not policy:
            errors.append(f"{verse}: missing policy")
        if episodes < int(min_episodes):
            errors.append(f"{verse}: episodes {episodes} < {int(min_episodes)}")
        if success_rate < float(min_success_rate):
            errors.append(f"{verse}: success_rate {success_rate:.3f} < {float(min_success_rate):.3f}")
        if require_run_dirs and (not run_dir or not os.path.isdir(run_dir)):
            errors.append(f"{verse}: run_dir missing on disk: {run_dir}")

        per_verse[verse] = {
            "run_id": run_id,
            "policy": policy,
            "episodes": episodes,
            "success_rate": success_rate,
            "mean_return": mean_return,
            "run_dir_exists": bool(run_dir and os.path.isdir(run_dir)),
        }

    return (len(errors) == 0), errors, {"verses": per_verse, "count": len(entries)}


def _validate_benchmark(
    bench: Dict[str, Any],
    *,
    max_age_hours: float,
    min_success_rate: float,
    max_safety_violation_rate: float,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    details: Dict[str, Any] = {}

    if not bool(bench.get("overall_pass", False)):
        errors.append("benchmark overall_pass is false")

    created_iso = str(bench.get("created_at_iso", "")).strip()
    if created_iso and float(max_age_hours) > 0:
        try:
            created = _parse_iso_utc(created_iso)
            age_hours = (_utc_now() - created).total_seconds() / 3600.0
            details["age_hours"] = age_hours
            if age_hours > float(max_age_hours):
                errors.append(f"benchmark too old: {age_hours:.2f}h > {float(max_age_hours):.2f}h")
        except Exception:
            errors.append("invalid benchmark created_at_iso")

    by_verse = bench.get("by_verse")
    verse_out: Dict[str, Any] = {}
    if isinstance(by_verse, dict):
        for verse, row in sorted(by_verse.items()):
            if not isinstance(row, dict):
                continue
            cand = row.get("candidate")
            if not isinstance(cand, dict):
                errors.append(f"{verse}: missing candidate row in benchmark")
                continue
            success_rate = _safe_float(cand.get("success_rate"), 0.0)
            safety_rate = _safe_float(cand.get("safety_violation_rate"), 0.0)
            if success_rate < float(min_success_rate):
                errors.append(f"{verse}: benchmark success_rate {success_rate:.3f} < {float(min_success_rate):.3f}")
            if safety_rate > float(max_safety_violation_rate):
                errors.append(
                    f"{verse}: benchmark safety_violation_rate {safety_rate:.3f} > {float(max_safety_violation_rate):.3f}"
                )
            verse_out[verse] = {
                "success_rate": success_rate,
                "safety_violation_rate": safety_rate,
                "passed": bool(row.get("passed", False)),
            }
    else:
        errors.append("benchmark missing by_verse object")
    details["verses"] = verse_out
    return (len(errors) == 0), errors, details


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--bench_json", type=str, default=os.path.join("models", "benchmarks", "latest.json"))
    ap.add_argument("--require_benchmark", action="store_true")
    ap.add_argument("--min_verses", type=int, default=1)
    ap.add_argument("--min_episodes", type=int, default=50)
    ap.add_argument("--min_success_rate", type=float, default=0.6)
    ap.add_argument("--max_bench_age_hours", type=float, default=72.0)
    ap.add_argument("--max_safety_violation_rate", type=float, default=0.2)
    ap.add_argument("--require_run_dirs", action="store_true")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    report: Dict[str, Any] = {"passed": False, "checks": {}, "errors": []}
    all_errors: List[str] = []

    manifest = _load_json_obj(args.manifest_path)
    ok_manifest, manifest_errors, manifest_details = _validate_manifest(
        manifest,
        min_verses=int(args.min_verses),
        min_episodes=int(args.min_episodes),
        min_success_rate=float(args.min_success_rate),
        require_run_dirs=bool(args.require_run_dirs),
    )
    all_errors.extend(manifest_errors)
    report["checks"]["manifest"] = {"passed": ok_manifest, "details": manifest_details, "errors": manifest_errors}

    if args.require_benchmark:
        bench = _load_json_obj(args.bench_json)
        ok_bench, bench_errors, bench_details = _validate_benchmark(
            bench,
            max_age_hours=float(args.max_bench_age_hours),
            min_success_rate=float(args.min_success_rate),
            max_safety_violation_rate=float(args.max_safety_violation_rate),
        )
        all_errors.extend(bench_errors)
        report["checks"]["benchmark"] = {"passed": ok_bench, "details": bench_details, "errors": bench_errors}
    else:
        report["checks"]["benchmark"] = {"passed": True, "details": "skipped", "errors": []}

    report["errors"] = all_errors
    report["passed"] = len(all_errors) == 0

    if args.out_json:
        out_path = str(args.out_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("production readiness report")
    print(f"passed: {report['passed']}")
    print(f"errors: {len(all_errors)}")
    for e in all_errors:
        print(f"- {e}")

    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
