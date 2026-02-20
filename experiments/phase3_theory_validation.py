"""
experiments/phase3_theory_validation.py

Runs Phase 3 theory checks:
1) Safety certificate (Hoeffding bound)
2) Transfer bound calibration + empirical mismatch
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from theory.safety_bounds import (
    derive_safety_certificate,
    extract_episode_violation_flags_from_events,
)
from theory.transfer_bounds import (
    analyze_transfer_report,
    calibrate_lambda_from_analyses,
    load_transfer_reports,
    summarize_prediction_error,
)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def run_phase3_theory_validation(
    *,
    safety_events_jsonl: str,
    safety_episodes: int,
    safety_violations: int,
    safety_confidence: float,
    transfer_reports: str,
    transfer_lambda: str,
    transfer_tolerance: float,
) -> Dict[str, Any]:
    safety: Dict[str, Any]
    if str(safety_events_jsonl).strip() and os.path.isfile(str(safety_events_jsonl)):
        extracted = extract_episode_violation_flags_from_events(events_jsonl_path=str(safety_events_jsonl))
        certificate = derive_safety_certificate(
            violation_flags=extracted["violation_flags"],
            confidence=float(safety_confidence),
        )
        safety = {
            "mode": "events_jsonl",
            "input": extracted,
            "certificate": certificate,
        }
    else:
        certificate = derive_safety_certificate(
            observed_violations=int(safety_violations),
            total_episodes=int(safety_episodes),
            confidence=float(safety_confidence),
        )
        safety = {
            "mode": "counts",
            "input": {"episodes": int(safety_episodes), "violations": int(safety_violations)},
            "certificate": certificate,
        }

    reports = load_transfer_reports(str(transfer_reports))
    probe = [
        analyze_transfer_report(r, lambda_term=0.10, report_path=str(r.get("_report_path", "")))
        for r in reports
    ]

    lambda_raw = str(transfer_lambda).strip().lower()
    if lambda_raw == "auto":
        lambda_used = calibrate_lambda_from_analyses(probe)
    else:
        lambda_used = max(0.0, min(1.0, _safe_float(transfer_lambda, 0.10)))

    analyses = [
        analyze_transfer_report(r, lambda_term=float(lambda_used), report_path=str(r.get("_report_path", "")))
        for r in reports
    ]
    transfer_summary = summarize_prediction_error(analyses, tolerance=float(transfer_tolerance))
    transfer = {
        "lambda_term_used": float(lambda_used),
        "summary": transfer_summary,
        "analyses": analyses,
    }

    return {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": {
            "safety_events_jsonl": str(safety_events_jsonl),
            "safety_episodes": int(safety_episodes),
            "safety_violations": int(safety_violations),
            "safety_confidence": float(safety_confidence),
            "transfer_reports": str(transfer_reports),
            "transfer_lambda": str(transfer_lambda),
            "transfer_tolerance": float(transfer_tolerance),
        },
        "safety": safety,
        "transfer": transfer,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 3 safety + transfer theory validation.")
    ap.add_argument(
        "--safety_events_jsonl",
        type=str,
        default=os.path.join("runs", "run_016326eed33c48ddaeee72ca8ff21544", "events.jsonl"),
    )
    ap.add_argument("--safety_episodes", type=int, default=200)
    ap.add_argument("--safety_violations", type=int, default=0)
    ap.add_argument("--safety_confidence", type=float, default=0.95)

    ap.add_argument(
        "--transfer_reports",
        type=str,
        default=os.path.join("models", "benchmarks", "fixed_seed", "transfer_seed_*.json"),
    )
    ap.add_argument("--transfer_lambda", type=str, default="auto")
    ap.add_argument("--transfer_tolerance", type=float, default=0.10)
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "validation", "phase3_theory_validation.json"),
    )
    args = ap.parse_args()

    report = run_phase3_theory_validation(
        safety_events_jsonl=str(args.safety_events_jsonl),
        safety_episodes=int(args.safety_episodes),
        safety_violations=int(args.safety_violations),
        safety_confidence=float(args.safety_confidence),
        transfer_reports=str(args.transfer_reports),
        transfer_lambda=str(args.transfer_lambda),
        transfer_tolerance=float(args.transfer_tolerance),
    )

    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    cert = ((report.get("safety") or {}).get("certificate") or {})
    tr = ((report.get("transfer") or {}).get("summary") or {})
    print("Phase 3 theory validation summary")
    print(
        f"- safety episodes={cert.get('episodes')} violations={cert.get('observed_violations')} "
        f"upper_bound@{float(cert.get('confidence', 0.95)):.0%}={_safe_float(cert.get('upper_bound', 1.0), 1.0):.4f}"
    )
    print(
        f"- transfer lambda={_safe_float((report.get('transfer') or {}).get('lambda_term_used', 0.0), 0.0):.4f} "
        f"max_abs_error={tr.get('max_abs_error')} tolerance={tr.get('tolerance')} "
        f"within_tolerance={bool(tr.get('within_tolerance', False))}"
    )
    print(f"report: {args.out_json}")


if __name__ == "__main__":
    main()
