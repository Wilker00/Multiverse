"""
tools/run_paper_readiness_pack.py

Runs a canonical, reproducible evidence pack for paper-readiness assessment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return bool(default)
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _read_json_obj(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _try_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        s = str(out).strip()
        return s if s else None
    except Exception:
        return None


def _normalize_pack(raw: Dict[str, Any]) -> Dict[str, Any]:
    schema = str(raw.get("schema_version", "")).strip().lower()
    if schema != "v1":
        raise ValueError("pack schema_version must be 'v1'")

    bench_raw = raw.get("benchmark_gate")
    bench_raw = bench_raw if isinstance(bench_raw, dict) else {}
    fixed_raw = raw.get("fixed_seed_transfer")
    fixed_raw = fixed_raw if isinstance(fixed_raw, dict) else {}
    theory_raw = raw.get("theory_validation")
    theory_raw = theory_raw if isinstance(theory_raw, dict) else {}

    challenge_args = fixed_raw.get("challenge_args")
    if not isinstance(challenge_args, list):
        challenge_args = []
    challenge_args = [str(x).strip() for x in challenge_args if str(x).strip()]
    bench_candidate_cfg = bench_raw.get("candidate_config")
    if not isinstance(bench_candidate_cfg, list):
        bench_candidate_cfg = []
    bench_candidate_cfg = [str(x).strip() for x in bench_candidate_cfg if str(x).strip()]
    bench_baseline_cfg = bench_raw.get("baseline_config")
    if not isinstance(bench_baseline_cfg, list):
        bench_baseline_cfg = []
    bench_baseline_cfg = [str(x).strip() for x in bench_baseline_cfg if str(x).strip()]

    return {
        "schema_version": "v1",
        "name": str(raw.get("name", "paper_readiness_pack_v1")).strip() or "paper_readiness_pack_v1",
        "benchmark_gate": {
            "enabled": _safe_bool(bench_raw.get("enabled", True), True),
            "suite_path": str(bench_raw.get("suite_path", "benchmark_suite.yaml")),
            "retention_max_drop_pct": float(_safe_float(bench_raw.get("retention_max_drop_pct", 0.05), 0.05)),
            "candidate_config": bench_candidate_cfg,
            "baseline_config": bench_baseline_cfg,
        },
        "fixed_seed_transfer": {
            "enabled": _safe_bool(fixed_raw.get("enabled", True), True),
            "target_verse": str(fixed_raw.get("target_verse", "warehouse_world")),
            "episodes": int(max(1, _safe_int(fixed_raw.get("episodes", 60), 60))),
            "max_steps": int(max(1, _safe_int(fixed_raw.get("max_steps", 100), 100))),
            "seeds": str(fixed_raw.get("seeds", "123,223,337")),
            "challenge_args": challenge_args,
        },
        "theory_validation": {
            "enabled": _safe_bool(theory_raw.get("enabled", True), True),
            "safety_events_jsonl": str(theory_raw.get("safety_events_jsonl", "")),
            "safety_episodes": int(max(1, _safe_int(theory_raw.get("safety_episodes", 200), 200))),
            "safety_violations": int(max(0, _safe_int(theory_raw.get("safety_violations", 0), 0))),
            "safety_confidence": float(_safe_float(theory_raw.get("safety_confidence", 0.95), 0.95)),
            "transfer_reports_glob": str(theory_raw.get("transfer_reports_glob", "")),
            "transfer_lambda": str(theory_raw.get("transfer_lambda", "auto")),
            "transfer_tolerance": float(_safe_float(theory_raw.get("transfer_tolerance", 0.10), 0.10)),
        },
    }


def _append_kv_flags(cmd: List[str], flag: str, items: List[str]) -> None:
    for item in items:
        s = str(item).strip()
        if s:
            cmd.extend([str(flag), s])


def _build_promote_cmd(
    *,
    py: str,
    suite_path: str,
    out_json: str,
    history_db: str,
    candidate_algo: str,
    baseline_algo: str,
    candidate_config: List[str],
    baseline_config: List[str],
    retention_max_drop_pct: float,
    manifest_path: str,
) -> List[str]:
    cmd = [
        py,
        os.path.join("tools", "promote_candidate.py"),
        "--suite",
        str(suite_path),
        "--candidate_algo",
        str(candidate_algo),
        "--baseline_algo",
        str(baseline_algo),
        "--retention_max_drop_pct",
        str(float(retention_max_drop_pct)),
        "--manifest_path",
        str(manifest_path),
        "--out",
        str(out_json),
        "--history_db",
        str(history_db),
    ]
    _append_kv_flags(cmd, "--candidate_config", list(candidate_config))
    _append_kv_flags(cmd, "--baseline_config", list(baseline_config))
    return cmd


def _build_fixed_seed_cmd(
    *,
    py: str,
    report_dir: str,
    out_json: str,
    target_verse: str,
    episodes: int,
    max_steps: int,
    seeds_csv: str,
    transfer_algo: str,
    baseline_algo: str,
    challenge_args: List[str],
) -> List[str]:
    cmd = [
        py,
        os.path.join("tools", "run_fixed_seed_benchmark.py"),
        "--report_dir",
        str(report_dir),
        "--out_json",
        str(out_json),
        "--target_verse",
        str(target_verse),
        "--episodes",
        str(int(episodes)),
        "--max_steps",
        str(int(max_steps)),
        "--seeds",
        str(seeds_csv),
        "--transfer_algo",
        str(transfer_algo),
        "--baseline_algo",
        str(baseline_algo),
    ]
    for tok in challenge_args:
        s = str(tok).strip()
        if s:
            cmd.append(f"--challenge_arg={s}")
    return cmd


def _build_theory_cmd(
    *,
    py: str,
    out_json: str,
    transfer_reports_glob: str,
    safety_events_jsonl: str,
    safety_episodes: int,
    safety_violations: int,
    safety_confidence: float,
    transfer_lambda: str,
    transfer_tolerance: float,
) -> List[str]:
    cmd = [
        py,
        os.path.join("experiments", "phase3_theory_validation.py"),
        "--transfer_reports",
        str(transfer_reports_glob),
        "--transfer_lambda",
        str(transfer_lambda),
        "--transfer_tolerance",
        str(float(transfer_tolerance)),
        "--safety_episodes",
        str(int(safety_episodes)),
        "--safety_violations",
        str(int(safety_violations)),
        "--safety_confidence",
        str(float(safety_confidence)),
        "--out_json",
        str(out_json),
    ]
    ses = str(safety_events_jsonl).strip()
    if ses:
        cmd.extend(["--safety_events_jsonl", ses])
    return cmd


def _run_step(name: str, cmd: List[str], *, dry_run: bool = False) -> Dict[str, Any]:
    if bool(dry_run):
        return {
            "name": str(name),
            "cmd": list(cmd),
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "ok": True,
            "dry_run": True,
        }
    started = time.time()
    proc = subprocess.run(cmd, cwd=os.getcwd())
    elapsed = time.time() - started
    return {
        "name": str(name),
        "cmd": list(cmd),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(round(elapsed, 3)),
        "ok": bool(proc.returncode == 0),
        "dry_run": False,
    }


def _prepare_paths(out_dir: str) -> Dict[str, str]:
    root = os.path.normpath(str(out_dir))
    return {
        "root": root,
        "benchmark_gate_json": os.path.join(root, "benchmark_gate.json"),
        "bench_history_db": os.path.join(root, "bench_history.sqlite"),
        "fixed_seed_dir": os.path.join(root, "fixed_seed"),
        "fixed_seed_summary_json": os.path.join(root, "fixed_seed_summary.json"),
        "phase3_theory_json": os.path.join(root, "phase3_theory_validation.json"),
        "pack_summary_json": os.path.join(root, "pack_summary.json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run canonical paper-readiness evidence pack.")
    ap.add_argument("--pack", type=str, default=os.path.join("experiment", "paper_readiness_pack_v1.json"))
    ap.add_argument("--out_dir", type=str, default=os.path.join("models", "paper", "paper_readiness", "latest"))
    ap.add_argument("--candidate_algo", type=str, required=True)
    ap.add_argument("--baseline_algo", type=str, default="q")
    ap.add_argument("--candidate_config", action="append", default=None)
    ap.add_argument("--baseline_config", action="append", default=None)
    ap.add_argument("--manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    pack_raw = _read_json_obj(str(args.pack))
    pack = _normalize_pack(pack_raw)
    py = sys.executable

    paths = _prepare_paths(str(args.out_dir))
    os.makedirs(paths["root"], exist_ok=True)
    os.makedirs(paths["fixed_seed_dir"], exist_ok=True)

    steps: List[Dict[str, Any]] = []

    bench_cfg = pack["benchmark_gate"]
    if _safe_bool(bench_cfg.get("enabled", True), True):
        candidate_config = [str(x).strip() for x in (args.candidate_config or []) if str(x).strip()]
        if not candidate_config:
            candidate_config = [str(x).strip() for x in (bench_cfg.get("candidate_config") or []) if str(x).strip()]
        baseline_config = [str(x).strip() for x in (args.baseline_config or []) if str(x).strip()]
        if not baseline_config:
            baseline_config = [str(x).strip() for x in (bench_cfg.get("baseline_config") or []) if str(x).strip()]
        cmd = _build_promote_cmd(
            py=py,
            suite_path=str(bench_cfg.get("suite_path", "benchmark_suite.yaml")),
            out_json=paths["benchmark_gate_json"],
            history_db=paths["bench_history_db"],
            candidate_algo=str(args.candidate_algo),
            baseline_algo=str(args.baseline_algo),
            candidate_config=candidate_config,
            baseline_config=baseline_config,
            retention_max_drop_pct=float(_safe_float(bench_cfg.get("retention_max_drop_pct", 0.05), 0.05)),
            manifest_path=str(args.manifest_path),
        )
        result = _run_step("benchmark_gate", cmd, dry_run=bool(args.dry_run))
        steps.append(result)

    fixed_cfg = pack["fixed_seed_transfer"]
    if _safe_bool(fixed_cfg.get("enabled", True), True):
        cmd = _build_fixed_seed_cmd(
            py=py,
            report_dir=paths["fixed_seed_dir"],
            out_json=paths["fixed_seed_summary_json"],
            target_verse=str(fixed_cfg.get("target_verse", "warehouse_world")),
            episodes=int(max(1, _safe_int(fixed_cfg.get("episodes", 60), 60))),
            max_steps=int(max(1, _safe_int(fixed_cfg.get("max_steps", 100), 100))),
            seeds_csv=str(fixed_cfg.get("seeds", "123,223,337")),
            transfer_algo=str(args.candidate_algo),
            baseline_algo=str(args.baseline_algo),
            challenge_args=[str(x).strip() for x in (fixed_cfg.get("challenge_args") or []) if str(x).strip()],
        )
        result = _run_step("fixed_seed_transfer", cmd, dry_run=bool(args.dry_run))
        steps.append(result)

    theory_cfg = pack["theory_validation"]
    if _safe_bool(theory_cfg.get("enabled", True), True):
        transfer_glob = str(theory_cfg.get("transfer_reports_glob", "")).strip()
        if not transfer_glob:
            transfer_glob = os.path.join(paths["fixed_seed_dir"], "transfer_seed_*.json")
        cmd = _build_theory_cmd(
            py=py,
            out_json=paths["phase3_theory_json"],
            transfer_reports_glob=transfer_glob,
            safety_events_jsonl=str(theory_cfg.get("safety_events_jsonl", "")),
            safety_episodes=int(max(1, _safe_int(theory_cfg.get("safety_episodes", 200), 200))),
            safety_violations=int(max(0, _safe_int(theory_cfg.get("safety_violations", 0), 0))),
            safety_confidence=float(_safe_float(theory_cfg.get("safety_confidence", 0.95), 0.95)),
            transfer_lambda=str(theory_cfg.get("transfer_lambda", "auto")),
            transfer_tolerance=float(_safe_float(theory_cfg.get("transfer_tolerance", 0.10), 0.10)),
        )
        result = _run_step("phase3_theory_validation", cmd, dry_run=bool(args.dry_run))
        steps.append(result)

    all_ok = all(bool(s.get("ok", False)) for s in steps) if steps else False
    summary = {
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pack_path": os.path.normpath(str(args.pack)),
        "pack_name": str(pack.get("name", "")),
        "pack_schema_version": str(pack.get("schema_version", "")),
        "git_sha": _try_git_sha(),
        "python": sys.version,
        "candidate_algo": str(args.candidate_algo),
        "baseline_algo": str(args.baseline_algo),
        "dry_run": bool(args.dry_run),
        "out_dir": paths["root"].replace("\\", "/"),
        "artifacts": {
            "benchmark_gate_json": paths["benchmark_gate_json"].replace("\\", "/"),
            "bench_history_db": paths["bench_history_db"].replace("\\", "/"),
            "fixed_seed_summary_json": paths["fixed_seed_summary_json"].replace("\\", "/"),
            "fixed_seed_dir": paths["fixed_seed_dir"].replace("\\", "/"),
            "phase3_theory_json": paths["phase3_theory_json"].replace("\\", "/"),
        },
        "steps": steps,
        "overall_ok": bool(all_ok),
    }

    with open(paths["pack_summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"pack_summary_json={paths['pack_summary_json']}")
    print(f"overall_ok={bool(all_ok)}")
    if bool(args.strict) and not bool(all_ok):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
