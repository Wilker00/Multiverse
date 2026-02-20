"""
tools/run_autonomous_cycle.py

Autonomous lifecycle loop that chains existing tools:
1) Fixed-seed transfer benchmark (`tools/run_fixed_seed_benchmark.py`)
2) Agent health scorecard (`tools/agent_health_monitor.py`)

Optional: run health monitor in `--auto_heal` mode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return obj


def build_benchmark_cmd(*, py: str, args: argparse.Namespace, out_json: str) -> List[str]:
    cmd = [
        py,
        os.path.join("tools", "run_fixed_seed_benchmark.py"),
        "--runs_root",
        str(args.runs_root),
        "--target_verse",
        str(args.target_verse),
        "--episodes",
        str(max(1, int(args.episodes))),
        "--max_steps",
        str(max(1, int(args.max_steps))),
        "--transfer_algo",
        str(args.transfer_algo),
        "--baseline_algo",
        str(args.baseline_algo),
        "--seeds",
        str(args.seeds),
        "--health_trace_root",
        str(args.health_trace_root),
        "--report_dir",
        str(args.benchmark_report_dir),
        "--out_json",
        str(out_json),
    ]
    for tok in (args.challenge_arg or []):
        s = str(tok).strip()
        if s:
            cmd.extend(["--challenge_arg", s])
    return cmd


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


def _run_cmd(cmd: List[str], *, cwd: str) -> None:
    proc = subprocess.run(cmd, cwd=cwd)
    if int(proc.returncode) != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _cycle_summary(bench: Dict[str, Any], health: Dict[str, Any]) -> Dict[str, Any]:
    agg = bench.get("aggregate", {}) if isinstance(bench.get("aggregate"), dict) else {}
    rows = health.get("rows", []) if isinstance(health.get("rows"), list) else []
    transfer_healthy = 0
    critical = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        status = str(r.get("status", "")).strip().lower()
        if status == "healthy":
            transfer_healthy += 1
        if status == "critical":
            critical += 1
    return {
        "bench_win_rate": float(_safe_float(agg.get("win_rate", 0.0), 0.0)),
        "bench_hazard_gain_pct": float(_safe_float(agg.get("mean_hazard_improvement_pct", 0.0), 0.0)),
        "health_agents_scored": int(len(rows)),
        "health_healthy_count": int(transfer_healthy),
        "health_critical_count": int(critical),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--target_verse", type=str, default="warehouse_world", choices=["warehouse_world", "labyrinth_world"])
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--transfer_algo", type=str, default="q")
    ap.add_argument("--baseline_algo", type=str, default="q")
    ap.add_argument("--seeds", type=str, default="123,223,337")
    ap.add_argument("--health_trace_root", type=str, default=os.path.join("models", "expert_datasets"))
    ap.add_argument("--central_memory_dir", type=str, default="central_memory")
    ap.add_argument("--manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--health_latest_runs", type=int, default=12)
    ap.add_argument("--health_limit", type=int, default=20)
    ap.add_argument("--auto_heal", action="store_true")
    ap.add_argument("--auto_heal_max_agents", type=int, default=2)
    ap.add_argument("--benchmark_report_dir", type=str, default=os.path.join("models", "benchmarks", "fixed_seed"))
    ap.add_argument("--out_dir", type=str, default=os.path.join("models", "tuning", "autonomous_cycles"))
    ap.add_argument("--sleep_seconds", type=float, default=0.0)
    ap.add_argument("--challenge_arg", action="append", default=None, help="Pass-through arg token for run_transfer_challenge.py")
    args = ap.parse_args()

    py = sys.executable
    os.makedirs(str(args.out_dir), exist_ok=True)
    cycles_out: List[Dict[str, Any]] = []
    for i in range(max(1, int(args.cycles))):
        cycle_no = i + 1
        cycle_tag = f"cycle_{cycle_no:03d}"
        cycle_dir = os.path.join(str(args.out_dir), cycle_tag)
        os.makedirs(cycle_dir, exist_ok=True)

        bench_json = os.path.join(cycle_dir, "fixed_seed_summary.json")
        health_json = os.path.join(cycle_dir, "agent_health_report.json")

        bench_cmd = build_benchmark_cmd(py=py, args=args, out_json=bench_json)
        print(f"[{cycle_tag}] running fixed-seed benchmark...")
        _run_cmd(bench_cmd, cwd=os.getcwd())
        bench = _read_json(bench_json)

        health_cmd = build_health_cmd(py=py, args=args, out_json=health_json)
        print(f"[{cycle_tag}] running agent health monitor...")
        _run_cmd(health_cmd, cwd=os.getcwd())
        health = _read_json(health_json)

        csum = _cycle_summary(bench, health)
        row = {
            "cycle": int(cycle_no),
            "cycle_dir": cycle_dir.replace("\\", "/"),
            "benchmark_json": bench_json.replace("\\", "/"),
            "health_json": health_json.replace("\\", "/"),
            "summary": csum,
        }
        cycles_out.append(row)
        print(
            f"[{cycle_tag}] win_rate={float(_safe_float(csum.get('bench_win_rate', 0.0), 0.0)):.3f} "
            f"hazard_gain_pct={float(_safe_float(csum.get('bench_hazard_gain_pct', 0.0), 0.0)):.2f} "
            f"critical_agents={int(csum.get('health_critical_count', 0))}"
        )

        if float(args.sleep_seconds) > 0.0 and cycle_no < int(args.cycles):
            time.sleep(float(args.sleep_seconds))

    summary = {
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycles": int(len(cycles_out)),
        "config": {
            "target_verse": str(args.target_verse),
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "transfer_algo": str(args.transfer_algo),
            "baseline_algo": str(args.baseline_algo),
            "seeds": str(args.seeds),
            "auto_heal": bool(args.auto_heal),
        },
        "cycle_rows": cycles_out,
    }
    out_json = os.path.join(str(args.out_dir), "autonomous_cycle_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"autonomous_summary={out_json}")


if __name__ == "__main__":
    main()

