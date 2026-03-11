#!/usr/bin/env python3
"""
tools/multiverse_cli.py

Unified convenience CLI for Multiverse:
- list universes (verses)
- run single/distributed training
- browse run artifacts quickly
- launch the terminal hub
- status snapshot and training profiles
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from verses.registry import list_verses, register_builtin
from tools.multiverse_cli_runs import (
    cmd_runs_files,
    cmd_runs_inspect,
    cmd_runs_latest,
    cmd_runs_list,
    cmd_runs_tail,
    discover_runs as _discover_runs,
    resolve_run_dir as _resolve_run_dir,
    run_snapshot as _run_snapshot,
)
from tools.multiverse_cli_shell import run_shell


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent

TRAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "verse": "line_world",
        "algo": "random",
        "episodes": 20,
        "max_steps": 40,
        "eval": False,
        "make_index": False,
    },
    "balanced": {
        "verse": "line_world",
        "algo": "q",
        "episodes": 80,
        "max_steps": 60,
        "eval": True,
        "make_index": True,
    },
    "research": {
        "verse": "warehouse_world",
        "algo": "q",
        "episodes": 200,
        "max_steps": 100,
        "eval": True,
        "make_index": True,
    },
}


DISTRIBUTED_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "mode": "sharded",
        "verse": "line_world",
        "algo": "q",
        "episodes": 100,
        "max_steps": 50,
        "workers": 2,
    },
    "balanced": {
        "mode": "sharded",
        "verse": "warehouse_world",
        "algo": "q",
        "episodes": 240,
        "max_steps": 100,
        "workers": 4,
    },
    "research": {
        "mode": "pbt",
        "verse": "warehouse_world",
        "algo": "q",
        "episodes": 400,
        "max_steps": 120,
        "workers": 6,
    },
}


def discover_runs(runs_root: str):
    """Compatibility re-export for run discovery helpers."""
    return _discover_runs(runs_root)


def resolve_run_dir(runs_root: str, run_id: str | None):
    """Compatibility re-export for run path resolution helpers."""
    return _resolve_run_dir(runs_root, run_id)


def _normalize_remainder(extra: Iterable[str] | None) -> List[str]:
    parts = [str(x) for x in (extra or [])]
    if parts and parts[0] == "--":
        return parts[1:]
    return parts


def _flag_present(raw_argv: Iterable[str], names: Iterable[str]) -> bool:
    candidates = [str(x).strip() for x in names if str(x).strip()]
    for token in [str(x) for x in raw_argv]:
        for name in candidates:
            if token == name or token.startswith(name + "="):
                return True
            if name.startswith("--"):
                neg = "--no-" + name[2:]
                if token == neg or token.startswith(neg + "="):
                    return True
    return False


def _maybe_set_attr(args: argparse.Namespace, raw_argv: Iterable[str], flag_names: Iterable[str], attr: str, value: Any) -> None:
    if _flag_present(raw_argv, flag_names):
        return
    setattr(args, attr, value)


def apply_train_profile(args: argparse.Namespace, raw_argv: Iterable[str]) -> None:
    profile = str(getattr(args, "profile", "custom") or "custom").strip().lower()
    if profile == "custom":
        return
    if profile not in TRAIN_PROFILES:
        raise ValueError(f"Unknown train profile '{profile}'.")
    cfg = TRAIN_PROFILES[profile]
    _maybe_set_attr(args, raw_argv, ["--universe", "--verse"], "verse", str(cfg["verse"]))
    _maybe_set_attr(args, raw_argv, ["--algo"], "algo", str(cfg["algo"]))
    _maybe_set_attr(args, raw_argv, ["--episodes"], "episodes", int(cfg["episodes"]))
    _maybe_set_attr(args, raw_argv, ["--max-steps", "--max_steps"], "max_steps", int(cfg["max_steps"]))
    _maybe_set_attr(args, raw_argv, ["--eval"], "eval", bool(cfg["eval"]))
    _maybe_set_attr(args, raw_argv, ["--make-index", "--make_index"], "make_index", bool(cfg["make_index"]))


def apply_distributed_profile(args: argparse.Namespace, raw_argv: Iterable[str]) -> None:
    profile = str(getattr(args, "profile", "custom") or "custom").strip().lower()
    if profile == "custom":
        return
    if profile not in DISTRIBUTED_PROFILES:
        raise ValueError(f"Unknown distributed profile '{profile}'.")
    cfg = DISTRIBUTED_PROFILES[profile]
    _maybe_set_attr(args, raw_argv, ["--mode"], "mode", str(cfg["mode"]))
    _maybe_set_attr(args, raw_argv, ["--universe", "--verse"], "verse", str(cfg["verse"]))
    _maybe_set_attr(args, raw_argv, ["--algo"], "algo", str(cfg["algo"]))
    _maybe_set_attr(args, raw_argv, ["--episodes"], "episodes", int(cfg["episodes"]))
    _maybe_set_attr(args, raw_argv, ["--max-steps", "--max_steps"], "max_steps", int(cfg["max_steps"]))
    _maybe_set_attr(args, raw_argv, ["--workers"], "workers", int(cfg["workers"]))

def _render_cmd(cmd: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(x) for x in cmd])
    return shlex.join([str(x) for x in cmd])


def run_process(cmd: List[str], *, capture_output: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=bool(capture_output),
        text=bool(capture_output),
    )
    if int(proc.returncode) != 0:
        detail = ""
        if capture_output:
            text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if text:
                detail = f"\n--- command output ---\n{text[-4000:]}"
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}{detail}")
    if capture_output:
        return str(proc.stdout or "")
    return ""

def build_train_agent_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_agent.py"),
        "--algo",
        str(args.algo),
        "--verse",
        str(args.verse),
        "--episodes",
        str(int(args.episodes)),
        "--max_steps",
        str(int(args.max_steps)),
        "--seed",
        str(int(args.seed)),
        "--runs_root",
        str(args.runs_root),
    ]
    if args.policy_id:
        cmd.extend(["--policy_id", str(args.policy_id)])
    if bool(args.train):
        cmd.append("--train")
    if bool(args.eval):
        cmd.append("--eval")
    if bool(args.make_index):
        cmd.append("--make_index")
    cmd.extend(_normalize_remainder(args.extra))
    return cmd


def build_train_distributed_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_distributed.py"),
        "--mode",
        str(args.mode),
        "--algo",
        str(args.algo),
        "--verse",
        str(args.verse),
        "--episodes",
        str(int(args.episodes)),
        "--max_steps",
        str(int(args.max_steps)),
        "--workers",
        str(int(args.workers)),
        "--seed",
        str(int(args.seed)),
        "--run_root",
        str(args.runs_root),
    ]
    if args.policy_id:
        cmd.extend(["--policy_id", str(args.policy_id)])
    if bool(args.train):
        cmd.append("--train")
    cmd.extend(_normalize_remainder(args.extra))
    return cmd


def cmd_universe_list(args: argparse.Namespace) -> int:
    register_builtin()
    names = sorted(list_verses().keys())
    if args.contains:
        q = str(args.contains).strip().lower()
        names = [n for n in names if q in str(n).lower()]
    if bool(args.json):
        print(json.dumps({"count": len(names), "universes": names}, indent=2))
        return 0
    for name in names:
        print(name)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cmd = build_train_agent_cmd(args)
    if bool(args.dry_run):
        print(_render_cmd(cmd))
        return 0
    capture = bool(getattr(args, "_capture_subprocess_output", False))
    out = run_process(cmd, capture_output=capture)
    if capture and out:
        print(out.rstrip("\n"))
    return 0


def cmd_distributed(args: argparse.Namespace) -> int:
    cmd = build_train_distributed_cmd(args)
    if bool(args.dry_run):
        print(_render_cmd(cmd))
        return 0
    capture = bool(getattr(args, "_capture_subprocess_output", False))
    out = run_process(cmd, capture_output=capture)
    if capture and out:
        print(out.rstrip("\n"))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    register_builtin()
    verses = sorted(list_verses().keys())
    snap = _run_snapshot(args.runs_root)
    payload = {
        "universes": {"count": len(verses), "sample": verses[:10]},
        "runs": snap,
    }
    if bool(args.json):
        print(json.dumps(payload, indent=2))
        return 0

    print("Multiverse Status")
    print("-----------------")
    print(f"Universes      : {len(verses)}")
    print(f"Runs root      : {snap['runs_root']}")
    print(f"Run count      : {snap['run_count']}")
    latest = snap.get("latest")
    if latest:
        print(f"Latest run     : {latest['run_id']}")
        print(f"Latest path    : {latest['path']}")
        print(f"Latest updated : {latest['modified']}")
        print(f"Latest size    : {latest['size_human']}")
    else:
        print("Latest run     : none")
    print("")
    print("Quick Start")
    print("-----------")
    print("multiverse universe list")
    print("multiverse train --profile quick")
    print("multiverse runs latest")
    return 0


def cmd_hub(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "universe_hub.py"),
        "--runs_root",
        str(args.runs_root),
        "--refresh_sec",
        str(float(args.refresh_sec)),
    ]
    if bool(args.once):
        cmd.append("--once")
    cmd.extend(_normalize_remainder(args.extra))
    if bool(args.dry_run):
        print(_render_cmd(cmd))
        return 0
    capture = bool(getattr(args, "_capture_subprocess_output", False))
    out = run_process(cmd, capture_output=capture)
    if capture and out:
        print(out.rstrip("\n"))
    return 0


def build_promotion_sentinel_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "promotion_sentinel.py"),
    ]
    if bool(args.status):
        cmd.append("--status")
        cmd.extend(["--out_dir", str(args.out_dir)])
        if str(args.summary_json).strip():
            cmd.extend(["--summary_json", str(args.summary_json).strip()])
        if bool(args.json):
            cmd.append("--json")
        return cmd
    cmd.extend(
        [
            "--cycles",
            str(max(1, int(args.cycles))),
            "--sleep_seconds",
            str(float(args.sleep_seconds)),
            "--runs_root",
            str(args.runs_root),
            "--manifest_path",
            str(args.manifest_path),
            "--central_memory_dir",
            str(args.central_memory_dir),
            "--health_trace_root",
            str(args.health_trace_root),
            "--health_latest_runs",
            str(max(1, int(args.health_latest_runs))),
            "--health_limit",
            str(max(1, int(args.health_limit))),
            "--bench_json",
            str(args.bench_json),
            "--min_verses",
            str(max(1, int(args.min_verses))),
            "--min_episodes",
            str(max(1, int(args.min_episodes))),
            "--min_success_rate",
            str(float(args.min_success_rate)),
            "--max_bench_age_hours",
            str(float(args.max_bench_age_hours)),
            "--max_safety_violation_rate",
            str(float(args.max_safety_violation_rate)),
            "--max_critical_agents",
            str(max(0, int(args.max_critical_agents))),
            "--max_unhealthy_agents",
            str(max(0, int(args.max_unhealthy_agents))),
            "--deploy_episodes",
            str(max(1, int(args.deploy_episodes))),
            "--deploy_seed",
            str(int(args.deploy_seed)),
            "--out_dir",
            str(args.out_dir),
        ]
    )
    if bool(args.auto_heal):
        cmd.append("--auto_heal")
        cmd.extend(["--auto_heal_max_agents", str(max(1, int(args.auto_heal_max_agents)))])
    if bool(args.require_benchmark):
        cmd.append("--require_benchmark")
    if bool(args.require_run_dirs):
        cmd.append("--require_run_dirs")
    if bool(args.deploy_on_pass):
        cmd.append("--deploy_on_pass")
    if str(args.deploy_verse).strip():
        cmd.extend(["--deploy_verse", str(args.deploy_verse).strip()])
    if bool(args.deploy_skip_eval_gate):
        cmd.append("--deploy_skip_eval_gate")
    if bool(args.deploy_skip_promotion_board):
        cmd.append("--deploy_skip_promotion_board")
    if bool(args.deploy_ingest_memory):
        cmd.append("--deploy_ingest_memory")
    return cmd


def cmd_sentinel(args: argparse.Namespace) -> int:
    cmd = build_promotion_sentinel_cmd(args)
    if bool(args.dry_run):
        print(_render_cmd(cmd))
        return 0
    capture = bool(getattr(args, "_capture_subprocess_output", False))
    out = run_process(cmd, capture_output=capture)
    if capture and out:
        print(out.rstrip("\n"))
    return 0


def cmd_shell(args: argparse.Namespace) -> int:
    return run_shell(
        runs_root=str(args.runs_root),
        build_parser_fn=build_parser,
        execute_argv_fn=execute_argv,
        run_snapshot_fn=_run_snapshot,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="multiverse",
        description="Multiverse convenience CLI for universes, training, and run artifacts.",
        epilog=(
            "Examples:\n"
            "  multiverse status\n"
            "  multiverse shell\n"
            "  multiverse universe list --contains line\n"
            "  multiverse train --profile quick\n"
            "  multiverse train --profile research --dry-run\n"
            "  multiverse sentinel --dry-run\n"
            "  multiverse runs inspect --count-events\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", aliases=["st"], help="Snapshot summary of universes and runs.")
    p_status.add_argument("--runs-root", type=str, default="runs")
    p_status.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_status.set_defaults(func=cmd_status)

    p_uni = sub.add_parser("universe", aliases=["u"], help="Universe (verse) commands.")
    sub_uni = p_uni.add_subparsers(dest="universe_command", required=True)
    p_uni_list = sub_uni.add_parser("list", aliases=["ls"], help="List registered universes/verses.")
    p_uni_list.add_argument("--contains", type=str, default=None, help="Filter names by substring.")
    p_uni_list.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_uni_list.set_defaults(func=cmd_universe_list)

    p_train = sub.add_parser("train", aliases=["t"], help="Run a single training command (wraps tools/train_agent.py).")
    p_train.add_argument("--universe", "--verse", dest="verse", type=str, default="line_world")
    p_train.add_argument("--algo", type=str, default="random")
    p_train.add_argument("--profile", type=str, default="custom", choices=["custom", *sorted(TRAIN_PROFILES.keys())])
    p_train.add_argument("--episodes", type=int, default=20)
    p_train.add_argument("--max-steps", dest="max_steps", type=int, default=40)
    p_train.add_argument("--seed", type=int, default=123)
    p_train.add_argument("--policy-id", type=str, default=None)
    p_train.add_argument("--runs-root", type=str, default="runs")
    p_train.add_argument("--eval", action=argparse.BooleanOptionalAction, default=False)
    p_train.add_argument("--make-index", dest="make_index", action=argparse.BooleanOptionalAction, default=False)
    p_train.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Print underlying command and exit.")
    p_train.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags for train_agent.py (use -- before extras).")
    p_train.set_defaults(func=cmd_train)

    p_dist = sub.add_parser(
        "distributed",
        aliases=["dist", "d"],
        help="Run distributed training (wraps tools/train_distributed.py).",
    )
    p_dist.add_argument("--mode", type=str, default="sharded", choices=["sharded", "pbt"])
    p_dist.add_argument("--universe", "--verse", dest="verse", type=str, default="line_world")
    p_dist.add_argument("--algo", type=str, default="q")
    p_dist.add_argument("--profile", type=str, default="custom", choices=["custom", *sorted(DISTRIBUTED_PROFILES.keys())])
    p_dist.add_argument("--episodes", type=int, default=100)
    p_dist.add_argument("--max-steps", dest="max_steps", type=int, default=50)
    p_dist.add_argument("--workers", type=int, default=2)
    p_dist.add_argument("--seed", type=int, default=123)
    p_dist.add_argument("--policy-id", type=str, default=None)
    p_dist.add_argument("--runs-root", type=str, default="runs")
    p_dist.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    p_dist.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Print underlying command and exit.")
    p_dist.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags for train_distributed.py (use -- before extras).")
    p_dist.set_defaults(func=cmd_distributed)

    p_runs = sub.add_parser("runs", aliases=["r"], help="Run artifact browsing commands.")
    sub_runs = p_runs.add_subparsers(dest="runs_command", required=True)

    p_runs_list = sub_runs.add_parser("list", aliases=["ls"], help="List runs under runs root.")
    p_runs_list.add_argument("--runs-root", type=str, default="runs")
    p_runs_list.add_argument("--limit", type=int, default=20)
    p_runs_list.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_runs_list.set_defaults(func=cmd_runs_list)

    p_runs_latest = sub_runs.add_parser("latest", aliases=["last"], help="Show latest run path.")
    p_runs_latest.add_argument("--runs-root", type=str, default="runs")
    p_runs_latest.add_argument("--path-only", action=argparse.BooleanOptionalAction, default=False)
    p_runs_latest.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_runs_latest.set_defaults(func=cmd_runs_latest)

    p_runs_files = sub_runs.add_parser("files", aliases=["f"], help="List files in a run directory.")
    p_runs_files.add_argument("--runs-root", type=str, default="runs")
    p_runs_files.add_argument("--run-id", type=str, default=None)
    p_runs_files.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    p_runs_files.add_argument("--limit", type=int, default=200)
    p_runs_files.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_runs_files.set_defaults(func=cmd_runs_files)

    p_runs_tail = sub_runs.add_parser("tail", aliases=["log"], help="Tail a file inside a run directory.")
    p_runs_tail.add_argument("--runs-root", type=str, default="runs")
    p_runs_tail.add_argument("--run-id", type=str, default=None)
    p_runs_tail.add_argument("--file", type=str, default="events.jsonl")
    p_runs_tail.add_argument("--lines", type=int, default=30)
    p_runs_tail.set_defaults(func=cmd_runs_tail)

    p_runs_inspect = sub_runs.add_parser("inspect", aliases=["i"], help="Inspect run metadata and key artifacts.")
    p_runs_inspect.add_argument("--runs-root", type=str, default="runs")
    p_runs_inspect.add_argument("--run-id", type=str, default=None)
    p_runs_inspect.add_argument("--count-events", action=argparse.BooleanOptionalAction, default=False)
    p_runs_inspect.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_runs_inspect.set_defaults(func=cmd_runs_inspect)

    p_shell = sub.add_parser(
        "shell",
        aliases=["live", "session"],
        help="Full-screen interactive shell with right-side next-action panel.",
    )
    p_shell.add_argument("--runs-root", type=str, default="runs")
    p_shell.set_defaults(func=cmd_shell)

    p_hub = sub.add_parser("hub", aliases=["h"], help="Launch terminal dashboard (wraps tools/universe_hub.py).")
    p_hub.add_argument("--runs-root", type=str, default="runs")
    p_hub.add_argument("--refresh-sec", type=float, default=2.0)
    p_hub.add_argument("--once", action=argparse.BooleanOptionalAction, default=False)
    p_hub.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Print underlying command and exit.")
    p_hub.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags for universe_hub.py (use -- before extras).")
    p_hub.set_defaults(func=cmd_hub)

    p_sentinel = sub.add_parser(
        "sentinel",
        aliases=["guard"],
        help="Run bounded promotion sentinel (health + readiness + optional deploy).",
    )
    p_sentinel.add_argument("--cycles", type=int, default=1)
    p_sentinel.add_argument("--status", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--summary-json", dest="summary_json", type=str, default="")
    p_sentinel.add_argument("--sleep-seconds", dest="sleep_seconds", type=float, default=0.0)
    p_sentinel.add_argument("--runs-root", type=str, default="runs")
    p_sentinel.add_argument("--manifest-path", dest="manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    p_sentinel.add_argument("--central-memory-dir", dest="central_memory_dir", type=str, default="central_memory")
    p_sentinel.add_argument("--health-trace-root", dest="health_trace_root", type=str, default=os.path.join("models", "expert_datasets"))
    p_sentinel.add_argument("--health-latest-runs", dest="health_latest_runs", type=int, default=12)
    p_sentinel.add_argument("--health-limit", dest="health_limit", type=int, default=20)
    p_sentinel.add_argument("--auto-heal", dest="auto_heal", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--auto-heal-max-agents", dest="auto_heal_max_agents", type=int, default=2)
    p_sentinel.add_argument("--bench-json", dest="bench_json", type=str, default=os.path.join("models", "benchmarks", "latest.json"))
    p_sentinel.add_argument("--require-benchmark", dest="require_benchmark", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--require-run-dirs", dest="require_run_dirs", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--min-verses", dest="min_verses", type=int, default=1)
    p_sentinel.add_argument("--min-episodes", dest="min_episodes", type=int, default=50)
    p_sentinel.add_argument("--min-success-rate", dest="min_success_rate", type=float, default=0.6)
    p_sentinel.add_argument("--max-bench-age-hours", dest="max_bench_age_hours", type=float, default=72.0)
    p_sentinel.add_argument("--max-safety-violation-rate", dest="max_safety_violation_rate", type=float, default=0.2)
    p_sentinel.add_argument("--max-critical-agents", dest="max_critical_agents", type=int, default=0)
    p_sentinel.add_argument("--max-unhealthy-agents", dest="max_unhealthy_agents", type=int, default=0)
    p_sentinel.add_argument("--deploy-on-pass", dest="deploy_on_pass", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--deploy-verse", dest="deploy_verse", type=str, default="")
    p_sentinel.add_argument("--deploy-episodes", dest="deploy_episodes", type=int, default=50)
    p_sentinel.add_argument("--deploy-seed", dest="deploy_seed", type=int, default=123)
    p_sentinel.add_argument("--deploy-skip-eval-gate", dest="deploy_skip_eval_gate", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--deploy-skip-promotion-board", dest="deploy_skip_promotion_board", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--deploy-ingest-memory", dest="deploy_ingest_memory", action=argparse.BooleanOptionalAction, default=False)
    p_sentinel.add_argument("--out-dir", dest="out_dir", type=str, default=os.path.join("models", "tuning", "promotion_sentinel"))
    p_sentinel.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Print underlying command and exit.")
    p_sentinel.set_defaults(func=cmd_sentinel)

    return ap


def execute_argv(
    raw_argv: List[str],
    *,
    allow_shell: bool = True,
    capture_subprocess: bool = False,
) -> int:
    ap = build_parser()
    if len(raw_argv) <= 0:
        cmd_status(argparse.Namespace(runs_root="runs", json=False))
        print("")
        ap.print_help()
        return 0
    args = ap.parse_args(raw_argv)
    if (not allow_shell) and str(getattr(args, "command", "")).lower() in ("shell", "live", "session"):
        raise ValueError("Cannot launch nested shell from shell mode.")
    if str(getattr(args, "command", "")) in ("train", "t"):
        apply_train_profile(args, raw_argv)
    if str(getattr(args, "command", "")) in ("distributed", "dist", "d"):
        apply_distributed_profile(args, raw_argv)
    if bool(capture_subprocess):
        setattr(args, "_capture_subprocess_output", True)
    func: Callable[[argparse.Namespace], int] = args.func
    return int(func(args))


def main() -> int:
    raw_argv = sys.argv[1:]
    if len(raw_argv) <= 0:
        return execute_argv([])
    try:
        return execute_argv(raw_argv)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
