"""
tools/promotion_sentinel.py

Bounded promotion sentinel that composes:
1) `tools/agent_health_monitor.py`
2) `tools/production_readiness_gate.py`
3) optional `tools/deploy_agent.py`

The sentinel is intended to be safe-by-default:
- it does not deploy unless checks pass
- it emits machine-readable cycle summaries
- it can run once or on a fixed interval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


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


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _summary_path_from_args(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "summary_json", "") or "").strip()
    if explicit:
        return explicit
    return os.path.join(str(args.out_dir), "promotion_sentinel_summary.json")


def build_health_cmd(*, py: str, args: argparse.Namespace, out_json: str) -> List[str]:
    cmd = [
        py,
        os.path.join("tools", "agent_health_monitor.py"),
        "--runs_root",
        str(args.runs_root),
        "--manifest_path",
        str(args.manifest_path),
        "--trace_root",
        str(args.health_trace_root),
        "--central_memory_dir",
        str(args.central_memory_dir),
        "--latest_runs",
        str(max(1, int(args.health_latest_runs))),
        "--limit",
        str(max(1, int(args.health_limit))),
        "--format",
        "json",
        "--out_json",
        str(out_json),
    ]
    if bool(args.auto_heal):
        cmd.append("--auto_heal")
        cmd.extend(["--auto_heal_max_agents", str(max(1, int(args.auto_heal_max_agents)))])
    return cmd


def build_readiness_cmd(*, py: str, args: argparse.Namespace, out_json: str) -> List[str]:
    cmd = [
        py,
        os.path.join("tools", "production_readiness_gate.py"),
        "--manifest_path",
        str(args.manifest_path),
        "--min_verses",
        str(max(1, int(args.min_verses))),
        "--min_episodes",
        str(max(1, int(args.min_episodes))),
        "--min_success_rate",
        str(float(args.min_success_rate)),
        "--max_safety_violation_rate",
        str(float(args.max_safety_violation_rate)),
        "--max_bench_age_hours",
        str(float(args.max_bench_age_hours)),
        "--out_json",
        str(out_json),
    ]
    if bool(args.require_benchmark):
        cmd.append("--require_benchmark")
        cmd.extend(["--bench_json", str(args.bench_json)])
    if bool(args.require_run_dirs):
        cmd.append("--require_run_dirs")
    return cmd


def build_deploy_cmd(*, py: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        py,
        os.path.join("tools", "deploy_agent.py"),
        "--manifest_path",
        str(args.manifest_path),
        "--runs_root",
        str(args.runs_root),
        "--episodes",
        str(max(1, int(args.deploy_episodes))),
        "--seed",
        str(int(args.deploy_seed)),
    ]
    if str(args.deploy_verse).strip():
        cmd.extend(["--verse", str(args.deploy_verse).strip()])
    if bool(args.deploy_skip_eval_gate):
        cmd.append("--skip_eval_gate")
    if bool(args.deploy_skip_promotion_board):
        cmd.append("--skip_promotion_board")
    if bool(args.deploy_ingest_memory):
        cmd.append("--ingest_memory")
    return cmd


def _run_cmd(cmd: List[str], *, cwd: str, allow_failure: bool = False) -> int:
    proc = subprocess.run(cmd, cwd=cwd)
    rc = int(proc.returncode)
    if rc != 0 and not allow_failure:
        raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)}")
    return rc


def _health_summary(health: Dict[str, Any]) -> Dict[str, Any]:
    rows = health.get("rows", []) if isinstance(health.get("rows"), list) else []
    status_counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).strip().lower() or "unknown"
        status_counts[status] = int(status_counts.get(status, 0)) + 1
    critical = int(status_counts.get("critical", 0))
    unhealthy = int(status_counts.get("unhealthy", 0))
    healthy = int(status_counts.get("healthy", 0))
    return {
        "agents_scored": int(len(rows)),
        "healthy_count": healthy,
        "unhealthy_count": unhealthy,
        "critical_count": critical,
        "status_counts": status_counts,
    }


def _readiness_summary(readiness: Dict[str, Any]) -> Dict[str, Any]:
    errors = readiness.get("errors", []) if isinstance(readiness.get("errors"), list) else []
    checks = readiness.get("checks", {}) if isinstance(readiness.get("checks"), dict) else {}
    return {
        "passed": bool(readiness.get("passed", False)),
        "error_count": int(len(errors)),
        "manifest_passed": bool((checks.get("manifest") or {}).get("passed", False) if isinstance(checks.get("manifest"), dict) else False),
        "benchmark_passed": bool((checks.get("benchmark") or {}).get("passed", False) if isinstance(checks.get("benchmark"), dict) else False),
    }


def _decide_cycle(
    *,
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    max_critical_agents: int,
    max_unhealthy_agents: int,
) -> Dict[str, Any]:
    hs = _health_summary(health)
    rs = _readiness_summary(readiness)
    health_ok = bool(
        int(hs["critical_count"]) <= int(max_critical_agents)
        and int(hs["unhealthy_count"]) <= int(max_unhealthy_agents)
    )
    deploy_allowed = bool(health_ok and rs["passed"])
    reasons: List[str] = []
    if not rs["passed"]:
        reasons.append("readiness_failed")
    if int(hs["critical_count"]) > int(max_critical_agents):
        reasons.append("critical_agents_exceeded")
    if int(hs["unhealthy_count"]) > int(max_unhealthy_agents):
        reasons.append("unhealthy_agents_exceeded")
    return {
        "health_ok": bool(health_ok),
        "readiness_ok": bool(rs["passed"]),
        "deploy_allowed": bool(deploy_allowed),
        "block_reasons": reasons,
        "health": hs,
        "readiness": rs,
    }


def _render_status(summary: Dict[str, Any]) -> str:
    cycle_rows = summary.get("cycle_rows", []) if isinstance(summary.get("cycle_rows"), list) else []
    latest = cycle_rows[-1] if cycle_rows else {}
    latest_decision = latest.get("decision", {}) if isinstance(latest.get("decision"), dict) else {}
    latest_health = latest_decision.get("health", {}) if isinstance(latest_decision.get("health"), dict) else {}
    lines = [
        "Promotion Sentinel",
        "------------------",
        f"Created at     : {str(summary.get('created_at_iso', '')) or 'unknown'}",
        f"Cycles         : {int(_safe_int(summary.get('cycles', len(cycle_rows)), len(cycle_rows)))}",
    ]
    if latest:
        lines.extend(
            [
                f"Latest cycle   : {int(_safe_int(latest.get('cycle', 0), 0))}",
                f"Readiness ok   : {bool(latest_decision.get('readiness_ok', False))}",
                f"Health ok      : {bool(latest_decision.get('health_ok', False))}",
                f"Deploy allowed : {bool(latest_decision.get('deploy_allowed', False))}",
                f"Critical agents: {int(_safe_int(latest_health.get('critical_count', 0), 0))}",
                f"Unhealthy      : {int(_safe_int(latest_health.get('unhealthy_count', 0), 0))}",
            ]
        )
        reasons = latest_decision.get("block_reasons", [])
        if isinstance(reasons, list) and reasons:
            lines.append(f"Block reasons  : {', '.join(str(x) for x in reasons)}")
        deploy = latest.get("deploy", {}) if isinstance(latest.get("deploy"), dict) else {}
        if bool(deploy.get("attempted", False)):
            lines.append(f"Deploy rc      : {deploy.get('returncode')}")
    else:
        lines.append("Latest cycle   : none")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--summary_json", type=str, default="")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--sleep_seconds", type=float, default=0.0)
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--central_memory_dir", type=str, default="central_memory")
    ap.add_argument("--health_trace_root", type=str, default=os.path.join("models", "expert_datasets"))
    ap.add_argument("--health_latest_runs", type=int, default=12)
    ap.add_argument("--health_limit", type=int, default=20)
    ap.add_argument("--auto_heal", action="store_true")
    ap.add_argument("--auto_heal_max_agents", type=int, default=2)
    ap.add_argument("--bench_json", type=str, default=os.path.join("models", "benchmarks", "latest.json"))
    ap.add_argument("--require_benchmark", action="store_true")
    ap.add_argument("--require_run_dirs", action="store_true")
    ap.add_argument("--min_verses", type=int, default=1)
    ap.add_argument("--min_episodes", type=int, default=50)
    ap.add_argument("--min_success_rate", type=float, default=0.6)
    ap.add_argument("--max_bench_age_hours", type=float, default=72.0)
    ap.add_argument("--max_safety_violation_rate", type=float, default=0.2)
    ap.add_argument("--max_critical_agents", type=int, default=0)
    ap.add_argument("--max_unhealthy_agents", type=int, default=0)
    ap.add_argument("--deploy_on_pass", action="store_true")
    ap.add_argument("--deploy_verse", type=str, default="")
    ap.add_argument("--deploy_episodes", type=int, default=50)
    ap.add_argument("--deploy_seed", type=int, default=123)
    ap.add_argument("--deploy_skip_eval_gate", action="store_true")
    ap.add_argument("--deploy_skip_promotion_board", action="store_true")
    ap.add_argument("--deploy_ingest_memory", action="store_true")
    ap.add_argument("--out_dir", type=str, default=os.path.join("models", "tuning", "promotion_sentinel"))
    args = ap.parse_args()

    if bool(args.status):
        summary = _read_json(_summary_path_from_args(args))
        if bool(args.json):
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(_render_status(summary))
        return

    py = sys.executable
    os.makedirs(str(args.out_dir), exist_ok=True)
    cycle_rows: List[Dict[str, Any]] = []

    for i in range(max(1, int(args.cycles))):
        cycle_no = i + 1
        cycle_tag = f"cycle_{cycle_no:03d}"
        cycle_dir = os.path.join(str(args.out_dir), cycle_tag)
        os.makedirs(cycle_dir, exist_ok=True)

        health_json = os.path.join(cycle_dir, "agent_health_report.json")
        readiness_json = os.path.join(cycle_dir, "production_readiness_report.json")
        deploy_status: Dict[str, Any] = {"attempted": False, "returncode": None}

        health_cmd = build_health_cmd(py=py, args=args, out_json=health_json)
        print(f"[{cycle_tag}] running agent health monitor...")
        _run_cmd(health_cmd, cwd=os.getcwd(), allow_failure=False)
        health = _read_json(health_json)

        readiness_cmd = build_readiness_cmd(py=py, args=args, out_json=readiness_json)
        print(f"[{cycle_tag}] running production readiness gate...")
        readiness_rc = _run_cmd(readiness_cmd, cwd=os.getcwd(), allow_failure=True)
        readiness = _read_json(readiness_json)

        decision = _decide_cycle(
            health=health,
            readiness=readiness,
            max_critical_agents=int(args.max_critical_agents),
            max_unhealthy_agents=int(args.max_unhealthy_agents),
        )

        if bool(args.deploy_on_pass) and bool(decision["deploy_allowed"]):
            deploy_cmd = build_deploy_cmd(py=py, args=args)
            print(f"[{cycle_tag}] running deploy agent...")
            deploy_status["attempted"] = True
            deploy_status["returncode"] = int(_run_cmd(deploy_cmd, cwd=os.getcwd(), allow_failure=True))

        row = {
            "cycle": int(cycle_no),
            "cycle_dir": cycle_dir.replace("\\", "/"),
            "health_json": health_json.replace("\\", "/"),
            "readiness_json": readiness_json.replace("\\", "/"),
            "readiness_returncode": int(readiness_rc),
            "decision": decision,
            "deploy": deploy_status,
        }
        cycle_rows.append(row)
        print(
            f"[{cycle_tag}] readiness_ok={bool(decision['readiness_ok'])} "
            f"health_ok={bool(decision['health_ok'])} "
            f"deploy_allowed={bool(decision['deploy_allowed'])} "
            f"critical_agents={int(decision['health']['critical_count'])}"
        )

        if float(args.sleep_seconds) > 0.0 and cycle_no < int(args.cycles):
            time.sleep(float(args.sleep_seconds))

    summary = {
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycles": int(len(cycle_rows)),
        "config": {
            "manifest_path": str(args.manifest_path),
            "require_benchmark": bool(args.require_benchmark),
            "deploy_on_pass": bool(args.deploy_on_pass),
            "max_critical_agents": int(args.max_critical_agents),
            "max_unhealthy_agents": int(args.max_unhealthy_agents),
        },
        "cycle_rows": cycle_rows,
    }
    out_json = _summary_path_from_args(args)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"promotion_sentinel_summary={out_json}")


if __name__ == "__main__":
    main()
