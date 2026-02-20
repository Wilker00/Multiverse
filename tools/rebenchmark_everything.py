"""
tools/rebenchmark_everything.py

Runs an end-to-end benchmark sweep after major model/safety/transfer upgrades.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
from typing import Any, Dict, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _run(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout),
        "stderr": str(proc.stderr),
    }


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _selector_online_diagnostics(runs_root: str) -> Dict[str, Any]:
    events_files = sorted(glob.glob(os.path.join(runs_root, "run_*", "events.jsonl")))
    pairs: List[List[float]] = []
    confs: List[float] = []
    rewards: List[float] = []
    total_routes = 0
    for path in events_files[-50:]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        for i, ev in enumerate(rows):
            info = ev.get("info")
            info = info if isinstance(info, dict) else {}
            route = info.get("selector_routing")
            route = route if isinstance(route, dict) else {}
            if not route:
                continue
            conf = float(route.get("confidence", 0.0) or 0.0)
            total_routes += 1
            confs.append(conf)
            if i + 1 < len(rows):
                nxt = rows[i + 1]
                if str(nxt.get("episode_id", "")) == str(ev.get("episode_id", "")):
                    rew = float(nxt.get("reward", 0.0) or 0.0)
                    rewards.append(rew)
                    pairs.append([conf, rew])
    if not pairs:
        return {
            "routing_events": int(total_routes),
            "pairs": 0,
            "mean_confidence": (None if not confs else float(statistics.mean(confs))),
            "mean_reward": (None if not rewards else float(statistics.mean(rewards))),
            "pearson_conf_next_reward": None,
        }
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    corr = 0.0 if vx <= 1e-12 or vy <= 1e-12 else float(cov / ((vx * vy) ** 0.5))
    return {
        "routing_events": int(total_routes),
        "pairs": int(len(pairs)),
        "mean_confidence": float(mx),
        "mean_reward": float(my),
        "pearson_conf_next_reward": float(corr),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--out_dir", type=str, default=os.path.join("models", "tuning", "rebenchmark"))
    ap.add_argument("--candidate_algo", type=str, default="special_moe")
    ap.add_argument("--baseline_algo", type=str, default="gateway")
    ap.add_argument("--benchmark_mode", type=str, default="hard", choices=["quick", "full", "hard"])
    ap.add_argument("--target_verse", type=str, default="warehouse_world", choices=["warehouse_world", "labyrinth_world"])
    ap.add_argument("--transfer_episodes", type=int, default=80)
    ap.add_argument("--transfer_max_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bench_dir = os.path.join(out_dir, "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    transfer_report = os.path.join(out_dir, "transfer_report.json")
    health_report = os.path.join(out_dir, "agent_health_report.json")

    bench_cmd = [
        sys.executable,
        os.path.join("tools", "run_benchmarks.py"),
        "--candidate_algo",
        str(args.candidate_algo),
        "--baseline_algo",
        str(args.baseline_algo),
        "--mode",
        str(args.benchmark_mode),
        "--out_dir",
        bench_dir,
    ]
    transfer_cmd = [
        sys.executable,
        os.path.join("tools", "run_transfer_challenge.py"),
        "--runs_root",
        str(args.runs_root),
        "--target_verse",
        str(args.target_verse),
        "--episodes",
        str(max(10, int(args.transfer_episodes))),
        "--max_steps",
        str(max(20, int(args.transfer_max_steps))),
        "--seed",
        str(int(args.seed)),
        "--report_out",
        transfer_report,
    ]
    health_cmd = [
        sys.executable,
        os.path.join("tools", "agent_health_monitor.py"),
        "--runs_root",
        str(args.runs_root),
        "--retrieval_audit",
        "--no-retrieval_audit_same_verse_only",
        "--out_json",
        health_report,
        "--format",
        "json",
    ]

    bench_res = _run(bench_cmd)
    transfer_res = _run(transfer_cmd)
    health_res = _run(health_cmd)

    bench_latest = _load_json(os.path.join(bench_dir, "latest.json"))
    transfer_json = _load_json(transfer_report)
    health_json = _load_json(health_report)
    selector_diag = _selector_online_diagnostics(str(args.runs_root))

    summary = {
        "commands": {
            "benchmarks": bench_res,
            "transfer_challenge": transfer_res,
            "agent_health_monitor": health_res,
        },
        "artifacts": {
            "benchmark_latest": os.path.join(bench_dir, "latest.json").replace("\\", "/"),
            "transfer_report": transfer_report.replace("\\", "/"),
            "health_report": health_report.replace("\\", "/"),
        },
        "kpis": {
            "benchmark_overall_pass": bool(bench_latest.get("overall_pass", False)) if bench_latest else None,
            "benchmark_mean_candidate_return": (
                bench_latest.get("summary", {}).get("mean_candidate_return")
                if isinstance(bench_latest.get("summary"), dict)
                else None
            ),
            "transfer_speedup_ratio": (
                transfer_json.get("comparison", {}).get("transfer_speedup_ratio")
                if isinstance(transfer_json.get("comparison"), dict)
                else None
            ),
            "transfer_hazard_improvement_pct": (
                transfer_json.get("comparison", {}).get("hazard_improvement_pct")
                if isinstance(transfer_json.get("comparison"), dict)
                else None
            ),
            "retrieval_hit_rate": (
                health_json.get("retrieval_audit_summary", {}).get("mean_hit_rate")
                if isinstance(health_json.get("retrieval_audit_summary"), dict)
                else None
            ),
            "selector_confidence_reward_corr": selector_diag.get("pearson_conf_next_reward"),
        },
        "selector_online_diagnostics": selector_diag,
    }
    out_json = os.path.join(out_dir, "rebenchmark_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"rebenchmark_summary={out_json.replace('\\', '/')}")


if __name__ == "__main__":
    main()
