"""
tools/promote_candidate.py

Benchmark-gated promotion tool:
- Loads benchmark_suite.yaml
- Evaluates baseline vs candidate across all suite worlds
- Enforces absolute thresholds + retention guard
- Logs history to SQLite for trend analysis
- Optionally writes manifest promotion on pass
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec
from orchestrator.eval_harness import BenchmarkCase, VerseEvalSummary, evaluate_agent_case


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


def _parse_kv(items: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not items:
        return out
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid k=v pair: {raw}")
        k, v = raw.split("=", 1)
        k = str(k).strip()
        v = str(v).strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue
        try:
            out[k] = int(v)
            continue
        except Exception:
            pass
        try:
            out[k] = float(v)
            continue
        except Exception:
            pass
        out[k] = v
    return out


def _load_suite(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"suite not found: {path}")
    text = ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("benchmark suite must parse to a JSON object")
    return data


def _build_cases(suite: Dict[str, Any]) -> List[BenchmarkCase]:
    rows = suite.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("suite.cases must be a non-empty list")

    cases: List[BenchmarkCase] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cases.append(
            BenchmarkCase(
                verse_name=str(row.get("verse_name", "")).strip().lower(),
                verse_version=str(row.get("verse_version", "0.1")),
                params=dict(row.get("params") or {}),
                seeds=[int(s) for s in (row.get("seeds") or [101, 203])],
                episodes_per_seed=max(1, int(row.get("episodes_per_seed", 4))),
                max_steps=max(1, int(row.get("max_steps", 60))),
            )
        )
    return [c for c in cases if c.verse_name]


def _suite_case_row_map(suite: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    rows = suite.get("cases")
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("verse_name", "")).strip().lower()
        if not name:
            continue
        out[name] = row
    return out


def _parse_case_cfg(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        return _parse_kv([str(x) for x in raw])
    s = str(raw).strip()
    if not s:
        return {}
    return _parse_kv([s])


def _check_absolute(candidate: VerseEvalSummary, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    min_success = _safe_float(thresholds.get("min_success_rate", 0.0), 0.0)
    min_return = _safe_float(thresholds.get("min_mean_return", -1e18), -1e18)
    max_safety = _safe_float(thresholds.get("max_safety_violation_rate", 1.0), 1.0)

    ok = (
        float(candidate.success_rate) >= float(min_success)
        and float(candidate.mean_return) >= float(min_return)
        and float(candidate.safety_violation_rate) <= float(max_safety)
    )
    return {
        "passed": bool(ok),
        "min_success_rate": float(min_success),
        "min_mean_return": float(min_return),
        "max_safety_violation_rate": float(max_safety),
    }


def _check_retention(candidate: VerseEvalSummary, baseline: VerseEvalSummary, max_drop_pct: float) -> Dict[str, Any]:
    p = max(0.0, min(0.95, float(max_drop_pct)))
    min_success_allowed = float(baseline.success_rate) * (1.0 - p)
    min_return_allowed = float(baseline.mean_return) - abs(float(baseline.mean_return)) * p

    ok = (
        float(candidate.success_rate) >= float(min_success_allowed)
        and float(candidate.mean_return) >= float(min_return_allowed)
    )
    return {
        "passed": bool(ok),
        "retention_max_drop_pct": float(p),
        "min_success_allowed": float(min_success_allowed),
        "min_return_allowed": float(min_return_allowed),
    }


def _summary_dict(s: VerseEvalSummary) -> Dict[str, Any]:
    return {
        "episodes": int(s.episodes),
        "mean_return": float(s.mean_return),
        "success_rate": float(s.success_rate),
        "failure_rate": float(s.failure_rate),
        "safety_violation_rate": float(s.safety_violation_rate),
        "return_variance": float(s.return_variance),
        "mean_steps": float(s.mean_steps),
    }


def _init_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bench_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            suite_path TEXT NOT NULL,
            candidate_algo TEXT NOT NULL,
            baseline_algo TEXT NOT NULL,
            overall_pass INTEGER NOT NULL,
            retention_max_drop_pct REAL NOT NULL,
            report_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bench_case_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bench_id INTEGER NOT NULL,
            verse_name TEXT NOT NULL,
            candidate_mean_return REAL NOT NULL,
            candidate_success_rate REAL NOT NULL,
            baseline_mean_return REAL NOT NULL,
            baseline_success_rate REAL NOT NULL,
            pass_absolute INTEGER NOT NULL,
            pass_retention INTEGER NOT NULL,
            passed INTEGER NOT NULL,
            FOREIGN KEY(bench_id) REFERENCES bench_history(id)
        )
        """
    )
    conn.commit()
    return conn


def _log_history(
    conn: sqlite3.Connection,
    *,
    suite_path: str,
    candidate_algo: str,
    baseline_algo: str,
    overall_pass: bool,
    retention_max_drop_pct: float,
    report: Dict[str, Any],
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO bench_history (
            created_at, suite_path, candidate_algo, baseline_algo, overall_pass, retention_max_drop_pct, report_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dt.datetime.now(dt.timezone.utc).isoformat(),
            str(suite_path),
            str(candidate_algo),
            str(baseline_algo),
            1 if overall_pass else 0,
            float(retention_max_drop_pct),
            json.dumps(report, ensure_ascii=False),
        ),
    )
    bench_id = int(cur.lastrowid)

    by_verse = report.get("by_verse") or {}
    if isinstance(by_verse, dict):
        for verse_name, row in by_verse.items():
            if not isinstance(row, dict):
                continue
            c = row.get("candidate") or {}
            b = row.get("baseline") or {}
            abs_chk = row.get("absolute_check") or {}
            ret_chk = row.get("retention_check") or {}
            cur.execute(
                """
                INSERT INTO bench_case_history (
                    bench_id, verse_name,
                    candidate_mean_return, candidate_success_rate,
                    baseline_mean_return, baseline_success_rate,
                    pass_absolute, pass_retention, passed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(bench_id),
                    str(verse_name),
                    float(c.get("mean_return", 0.0)),
                    float(c.get("success_rate", 0.0)),
                    float(b.get("mean_return", 0.0)),
                    float(b.get("success_rate", 0.0)),
                    1 if bool(abs_chk.get("passed", False)) else 0,
                    1 if bool(ret_chk.get("passed", False)) else 0,
                    1 if bool(row.get("passed", False)) else 0,
                ),
            )

    conn.commit()
    return bench_id


def _promote_manifest(
    *,
    manifest_path: str,
    verse_name: str,
    candidate_algo: str,
    candidate_config: Dict[str, Any],
    benchmark_report: Dict[str, Any],
) -> None:
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")

    dep = manifest.get("deployment_ready_defaults")
    if not isinstance(dep, dict):
        dep = {}

    verse_report = (benchmark_report.get("by_verse") or {}).get(verse_name) or {}
    cand = (verse_report.get("candidate") or {}) if isinstance(verse_report, dict) else {}

    cfg_args = " ".join([f"--aconfig {k}={v}" for k, v in sorted(candidate_config.items())])
    dep[str(verse_name)] = {
        "picked_run": {
            "run_id": "benchmark_promoted",
            "run_dir": "",
            "verse": str(verse_name),
            "policy": str(candidate_algo),
            "mean_return": float(cand.get("mean_return", 0.0)),
            "success_rate": float(cand.get("success_rate", 0.0)),
            "mean_steps": float(cand.get("mean_steps", 0.0)),
            "episodes": int(cand.get("episodes", 0)),
        },
        "command": f"python tools/train_agent.py --algo {candidate_algo} --verse {verse_name} {cfg_args}".strip(),
        "promotion_source": "tools/promote_candidate.py",
        "bench_passed": True,
    }
    manifest["deployment_ready_defaults"] = dep
    manifest["last_benchmark_report"] = benchmark_report

    backup_path = manifest_path + ".bak"
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            old = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(old)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default=os.path.join("experiment", "benchmark_suite.yaml"))
    ap.add_argument("--history_db", type=str, default=os.path.join("models", "bench_history.sqlite"))

    ap.add_argument("--candidate_algo", type=str, required=True)
    ap.add_argument("--candidate_policy_id", type=str, default="candidate")
    ap.add_argument("--candidate_config", action="append", default=None)

    ap.add_argument("--baseline_algo", type=str, default="gateway")
    ap.add_argument("--baseline_policy_id", type=str, default="baseline")
    ap.add_argument("--baseline_config", action="append", default=None)

    ap.add_argument("--manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--retention_max_drop_pct", type=float, default=None)

    ap.add_argument("--apply_manifest", action="store_true", help="On pass, write promoted default entry to manifest.")
    ap.add_argument("--promote_verse", type=str, default="", help="Required with --apply_manifest.")
    ap.add_argument("--out", type=str, default=os.path.join("models", "last_benchmark_report.json"))
    args = ap.parse_args()

    suite = _load_suite(args.suite)
    cases = _build_cases(suite)
    if not cases:
        raise RuntimeError("No benchmark cases found in suite")
    case_row_map = _suite_case_row_map(suite)

    cand_cfg = _parse_kv(args.candidate_config)
    base_cfg = _parse_kv(args.baseline_config)
    if args.baseline_algo == "gateway":
        base_cfg.setdefault("manifest_path", args.manifest_path)

    candidate_spec_base = AgentSpec(
        spec_version="v1",
        policy_id=str(args.candidate_policy_id),
        policy_version="0.1",
        algo=str(args.candidate_algo),
        seed=123,
        tags=["bench_candidate"],
        config=(cand_cfg if cand_cfg else None),
    )
    baseline_spec_base = AgentSpec(
        spec_version="v1",
        policy_id=str(args.baseline_policy_id),
        policy_version="0.1",
        algo=str(args.baseline_algo),
        seed=123,
        tags=["bench_baseline"],
        config=(base_cfg if base_cfg else None),
    )

    retention_max_drop_pct = (
        float(args.retention_max_drop_pct)
        if args.retention_max_drop_pct is not None
        else _safe_float(suite.get("retention_max_drop_pct", 0.05), 0.05)
    )

    by_verse: Dict[str, Any] = {}
    overall_pass = True
    for case in cases:
        row_cfg = case_row_map.get(case.verse_name, {})
        case_cand_cfg = _parse_case_cfg(row_cfg.get("candidate_config"))
        case_base_cfg = _parse_case_cfg(row_cfg.get("baseline_config"))
        cand_cfg_eff = dict(cand_cfg)
        base_cfg_eff = dict(base_cfg)
        cand_cfg_eff.update(case_cand_cfg)
        base_cfg_eff.update(case_base_cfg)
        if args.baseline_algo == "gateway":
            base_cfg_eff.setdefault("manifest_path", args.manifest_path)

        candidate_spec = dataclasses.replace(
            candidate_spec_base,
            config=(cand_cfg_eff if cand_cfg_eff else None),
        )
        baseline_spec = dataclasses.replace(
            baseline_spec_base,
            config=(base_cfg_eff if base_cfg_eff else None),
        )

        baseline = evaluate_agent_case(agent_spec=baseline_spec, case=case)
        candidate = evaluate_agent_case(agent_spec=candidate_spec, case=case)
        thresholds = dict((row_cfg.get("thresholds") or {})) if isinstance(row_cfg, dict) else {}

        abs_check = _check_absolute(candidate, thresholds)
        ret_check = _check_retention(candidate, baseline, retention_max_drop_pct)
        passed = bool(abs_check["passed"] and ret_check["passed"])
        overall_pass = bool(overall_pass and passed)

        by_verse[case.verse_name] = {
            "passed": bool(passed),
            "absolute_check": abs_check,
            "retention_check": ret_check,
            "candidate": _summary_dict(candidate),
            "baseline": _summary_dict(baseline),
            "candidate_config_used": cand_cfg_eff,
            "baseline_config_used": base_cfg_eff,
        }

        print(
            f"[{case.verse_name}] pass={passed} "
            f"cand_return={candidate.mean_return:.3f} base_return={baseline.mean_return:.3f} "
            f"cand_success={candidate.success_rate:.3f} base_success={baseline.success_rate:.3f}"
        )

    report = {
        "suite": str(args.suite),
        "suite_name": str(suite.get("suite_name", "")),
        "candidate_algo": str(args.candidate_algo),
        "baseline_algo": str(args.baseline_algo),
        "retention_max_drop_pct": float(retention_max_drop_pct),
        "overall_pass": bool(overall_pass),
        "by_verse": by_verse,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    conn = _init_db(args.history_db)
    bench_id = _log_history(
        conn,
        suite_path=args.suite,
        candidate_algo=args.candidate_algo,
        baseline_algo=args.baseline_algo,
        overall_pass=bool(overall_pass),
        retention_max_drop_pct=float(retention_max_drop_pct),
        report=report,
    )
    conn.close()

    print(f"Report written: {args.out}")
    print(f"Bench history id: {bench_id} ({args.history_db})")

    if bool(overall_pass) and args.apply_manifest:
        if not str(args.promote_verse).strip():
            raise ValueError("--promote_verse is required when --apply_manifest is used")
        promote_verse = str(args.promote_verse).strip().lower()
        row_cfg = case_row_map.get(promote_verse, {})
        promote_cand_cfg = dict(cand_cfg)
        promote_cand_cfg.update(_parse_case_cfg(row_cfg.get("candidate_config")))
        _promote_manifest(
            manifest_path=args.manifest_path,
            verse_name=promote_verse,
            candidate_algo=str(args.candidate_algo),
            candidate_config=promote_cand_cfg,
            benchmark_report=report,
        )
        print(f"Manifest promoted for verse '{args.promote_verse}': {args.manifest_path}")

    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
