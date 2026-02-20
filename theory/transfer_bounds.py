"""
theory/transfer_bounds.py

Transfer-learning bounds utilities based on
err_target <= err_source + divergence(Ds, Dt) + lambda.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def estimate_domain_divergence_from_bridge_stats(bridge_stats: Sequence[Dict[str, Any]]) -> float:
    """
    Proxy for domain divergence from translation coverage.

    Sources with zero translated rows are excluded from the calculation
    because they represent completely incompatible bridges (e.g. cliff_world
    -> warehouse_world) that the semantic bridge correctly rejects. Including
    them would inflate divergence to ~1.0 regardless of the quality of the
    remaining bridges.
    """
    total_rows = 0.0
    weighted_gap = 0.0
    for row in bridge_stats:
        inp = _safe_float(row.get("input_rows"), 0.0)
        if inp <= 0.0:
            continue
        tr = max(0.0, _safe_float(row.get("translated_rows"), 0.0))
        # Skip sources with zero coverage — they were correctly rejected
        # by the bridge and should not poison the divergence estimate.
        if tr <= 0.0:
            continue
        coverage = max(0.0, min(1.0, tr / inp))
        total_rows += inp
        weighted_gap += inp * (1.0 - coverage)
    if total_rows <= 1e-12:
        return 1.0
    return _clip01(weighted_gap / total_rows)


def estimate_semantic_agreement(
    source_values: Sequence[float],
    target_values: Sequence[float],
) -> float:
    """
    Compute semantic agreement as the correlation between source and target value estimates
    on the same set of abstract feature vectors.
    Returns: -1.0 to 1.0 (Pearson correlation).
    """
    if len(source_values) != len(target_values) or len(source_values) < 2:
        return 0.0
    import math
    n = len(source_values)
    mean_s = sum(source_values) / n
    mean_t = sum(target_values) / n
    num = sum((s - mean_s) * (t - mean_t) for s, t in zip(source_values, target_values))
    den_s = math.sqrt(sum((s - mean_s)**2 for s in source_values))
    den_t = math.sqrt(sum((t - mean_t)**2 for t in target_values))
    if den_s < 1e-9 or den_t < 1e-9:
        return 0.0
    return num / (den_s * den_t)


def estimate_source_error_from_bridge_stats(
    bridge_stats: Sequence[Dict[str, Any]],
    *,
    success_events_error: float = 0.02,
    dna_good_error: float = 0.10,
    default_error: float = 0.12,
) -> float:
    """
    Source-domain error proxy weighted by source row counts.
    """
    total_rows = 0.0
    weighted_err = 0.0
    for row in bridge_stats:
        inp = _safe_float(row.get("input_rows"), 0.0)
        if inp <= 0.0:
            continue
        kind = str(row.get("source_kind", "")).strip().lower()
        if "success" in kind:
            e = float(success_events_error)
        elif "dna_good" in kind or kind == "good":
            e = float(dna_good_error)
        else:
            e = float(default_error)
        total_rows += inp
        weighted_err += inp * e
    if total_rows <= 1e-12:
        return _clip01(float(default_error))
    return _clip01(weighted_err / total_rows)


def compute_transfer_bound(
    *,
    source_error: float, #lol
    domain_divergence: float,
    lambda_term: float = 0.10,
) -> Dict[str, Any]:
    src = _clip01(source_error)
    div = _clip01(domain_divergence)
    lam = _clip01(lambda_term)
    predicted_target_error = _clip01(src + div + lam)
    baseline = max(src, 1e-8)
    transfer_efficiency = 1.0 - (predicted_target_error / baseline)
    return {
        "source_error": float(src),
        "domain_divergence": float(div),
        "lambda_term": float(lam),
        "predicted_target_error": float(predicted_target_error),
        "predicted_target_success": float(1.0 - predicted_target_error),
        "transfer_efficiency": float(transfer_efficiency),
        "bound_family": "ben_david_style",
    }


def analyze_transfer_report(
    report: Dict[str, Any],
    *,
    lambda_term: float = 0.10,
    report_path: str = "",
) -> Dict[str, Any]:
    # Support both flat and nested schemas
    transfer_agent = report.get("transfer_agent")
    if isinstance(transfer_agent, dict):
        transfer_eval = transfer_agent.get("eval") or {}
    else:
        transfer_eval = report.get("target") or {}

    transfer_success = _clip01(_safe_float(transfer_eval.get("success_rate"), 0.0))
    empirical_target_error = float(1.0 - transfer_success)

    transfer_dataset = report.get("transfer_dataset") or {}
    bridge_stats = transfer_dataset.get("bridge_stats") or []
    
    # If missing bridge stats, we might be in an ablation report or raw run
    if not bridge_stats:
        # Check if we have source info to estimate from
        source_eval = report.get("source") or {}
        if source_eval:
            # Fake a bridge stat entry to allow the estimation logic to run
            # We'll assume a "default" divergence if we can't find stats
            bridge_stats = [{
                "source_verse": report.get("source_verse", "unknown"),
                "input_rows": 1000,
                "translated_rows": 0, # Assume high divergence if stats are missing
                "source_kind": "dna_good"
            }]

    source_error = estimate_source_error_from_bridge_stats(bridge_stats)
    divergence = estimate_domain_divergence_from_bridge_stats(bridge_stats)
    bound = compute_transfer_bound(
        source_error=float(source_error),
        domain_divergence=float(divergence),
        lambda_term=float(lambda_term),
    )
    predicted_error = float(bound["predicted_target_error"])
    abs_error = abs(predicted_error - empirical_target_error)

    return {
        "report_path": str(report_path),
        "seed": report.get("seed"),
        "target_verse": report.get("target_verse"),
        "empirical_target_success": float(transfer_success),
        "empirical_target_error": float(empirical_target_error),
        "analysis_inputs": {
            "source_error_estimate": float(source_error),
            "domain_divergence_estimate": float(divergence),
        },
        "bound": bound,
        "prediction_abs_error": float(abs_error),
    }


def calibrate_lambda_from_analyses(analyses: Sequence[Dict[str, Any]]) -> float:
    residuals: List[float] = []
    for row in analyses:
        emp = _safe_float(row.get("empirical_target_error"), 1.0)
        src = _safe_float(((row.get("analysis_inputs") or {}).get("source_error_estimate")), 0.1)
        div = _safe_float(((row.get("analysis_inputs") or {}).get("domain_divergence_estimate")), 1.0)
        residuals.append(emp - src - div)
    if not residuals:
        return 0.10
    return _clip01(sum(residuals) / float(len(residuals)))


def summarize_prediction_error(analyses: Sequence[Dict[str, Any]], *, tolerance: float = 0.10) -> Dict[str, Any]:
    errs = [abs(_safe_float(a.get("prediction_abs_error"), 0.0)) for a in analyses]
    if not errs:
        return {
            "n_reports": 0,
            "mean_abs_error": None,
            "max_abs_error": None,
            "within_tolerance": False,
            "tolerance": float(tolerance),
        }
    mean_err = sum(errs) / float(len(errs))
    max_err = max(errs)
    return {
        "n_reports": int(len(errs)),
        "mean_abs_error": float(mean_err),
        "max_abs_error": float(max_err),
        "within_tolerance": bool(max_err <= float(tolerance)),
        "tolerance": float(tolerance),
    }


def _iter_report_paths(pattern_or_csv: str) -> Iterable[str]:
    raw = str(pattern_or_csv).strip()
    if not raw:
        return []
    chunks = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    paths: List[str] = []
    for chunk in chunks:
        if any(ch in chunk for ch in ("*", "?", "[")):
            paths.extend(glob.glob(chunk))
        else:
            paths.append(chunk)
    out: List[str] = []
    seen = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen or (not os.path.isfile(ap)):
            continue
        seen.add(ap)
        out.append(ap)
    return out


def load_transfer_reports(pattern_or_csv: str) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for p in _iter_report_paths(pattern_or_csv):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        
        # Candidate 1: Standard Transfer Challenge Report
        if "transfer_agent" in d and "transfer_dataset" in d:
            d["_report_path"] = p
            reports.append(d)
            continue
            
        # Candidate 2: Ablation Report (contains "results" dict)
        if "results" in d and isinstance(d["results"], dict):
            for name, item in d["results"].items():
                if isinstance(item, dict) and "target" in item:
                    item["_report_path"] = f"{p}#{name}"
                    # Inherit top-level metadata if available
                    item.setdefault("source_verse", d.get("source_verse"))
                    item.setdefault("target_verse", d.get("target_verse"))
                    reports.append(item)
            continue

        # Candidate 3: Generic run directory dump or partial report
        if "target" in d and ("success_rate" in d["target"] or "mean_return" in d["target"]):
            d["_report_path"] = p
            reports.append(d)
            continue

    return reports


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute transfer-theory bounds from transfer report JSON files.")
    ap.add_argument("--reports", type=str, required=True, help="Glob/csv of transfer report JSON files.")
    ap.add_argument("--lambda_term", type=str, default="auto", help="Float or 'auto'.")
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    reports = load_transfer_reports(str(args.reports))
    if not reports:
        raise RuntimeError("no transfer reports loaded")

    probe = [
        analyze_transfer_report(r, lambda_term=0.10, report_path=str(r.get("_report_path", "")))
        for r in reports
    ]

    lam_raw = str(args.lambda_term).strip().lower()
    if lam_raw == "auto":
        lambda_term = calibrate_lambda_from_analyses(probe)
    else:
        lambda_term = _clip01(_safe_float(args.lambda_term, 0.10))

    analyses = [
        analyze_transfer_report(r, lambda_term=float(lambda_term), report_path=str(r.get("_report_path", "")))
        for r in reports
    ]
    summary = summarize_prediction_error(analyses, tolerance=float(args.tolerance))
    out = {
        "lambda_term_used": float(lambda_term),
        "summary": summary,
        "analyses": analyses,
    }

    if str(args.out_json).strip():
        os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
        with open(str(args.out_json), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
