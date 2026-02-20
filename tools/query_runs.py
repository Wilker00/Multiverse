"""
tools/query_runs.py

Query run directories by verse/policy/metrics for large-scale workflows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from orchestrator.evaluator import evaluate_run


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


@dataclass
class RunRow:
    run_id: str
    run_dir: str
    mtime: float
    verse_name: str
    policy_id: str
    episodes: int
    mean_return: float
    success_rate: Optional[float]
    mean_steps: float


def _iter_run_dirs(runs_root: str) -> Iterable[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[str] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        if os.path.isfile(os.path.join(run_dir, "events.jsonl")):
            out.append(run_dir)
    return out


def _first_event(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                return None
            obj = json.loads(line)
            return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _load_run_row(run_dir: str, *, with_eval: bool) -> Optional[RunRow]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return None
    first = _first_event(events_path) or {}
    verse_name = str(first.get("verse_name", "")).strip()
    policy_id = str(first.get("policy_id", "")).strip()
    run_id = os.path.basename(run_dir)
    mtime = _safe_float(os.path.getmtime(events_path), 0.0)

    episodes = 0
    mean_return = 0.0
    success_rate: Optional[float] = None
    mean_steps = 0.0
    if with_eval:
        stats = evaluate_run(run_dir)
        episodes = int(stats.episodes)
        mean_return = float(stats.mean_return)
        success_rate = (None if stats.success_rate is None else float(stats.success_rate))
        mean_steps = float(stats.mean_steps)
    return RunRow(
        run_id=run_id,
        run_dir=run_dir.replace("\\", "/"),
        mtime=mtime,
        verse_name=verse_name,
        policy_id=policy_id,
        episodes=episodes,
        mean_return=mean_return,
        success_rate=success_rate,
        mean_steps=mean_steps,
    )


def _passes_filters(row: RunRow, args: argparse.Namespace) -> bool:
    if args.verse and row.verse_name != str(args.verse).strip():
        return False
    if args.policy and row.policy_id != str(args.policy).strip():
        return False
    if args.run_id_contains and str(args.run_id_contains) not in row.run_id:
        return False
    if row.episodes < int(args.min_episodes):
        return False
    if row.mean_return < float(args.min_mean_return):
        return False
    if row.mean_return > float(args.max_mean_return):
        return False
    if row.success_rate is not None:
        if row.success_rate < float(args.min_success_rate):
            return False
        if row.success_rate > float(args.max_success_rate):
            return False
    return True


def _sort_rows(rows: List[RunRow], sort_by: str, order: str) -> List[RunRow]:
    reverse = str(order).strip().lower() != "asc"
    key_name = str(sort_by).strip().lower()

    if key_name == "mtime":
        key = lambda r: r.mtime
    elif key_name == "mean_return":
        key = lambda r: r.mean_return
    elif key_name == "success_rate":
        key = lambda r: (-1.0 if r.success_rate is None else float(r.success_rate))
    elif key_name == "mean_steps":
        key = lambda r: r.mean_steps
    elif key_name == "episodes":
        key = lambda r: r.episodes
    else:
        key = lambda r: r.mtime
    return sorted(rows, key=key, reverse=reverse)


def _print_table(rows: List[RunRow]) -> None:
    print(
        f"{'run_id':<36} {'verse':<12} {'policy':<18} {'episodes':>8} "
        f"{'mean_return':>12} {'success':>9} {'steps':>8} {'mtime':>20}"
    )
    for r in rows:
        mtime = datetime.fromtimestamp(float(r.mtime)).strftime("%Y-%m-%d %H:%M:%S")
        succ = "n/a" if r.success_rate is None else f"{100.0 * float(r.success_rate):.1f}%"
        print(
            f"{r.run_id:<36} {r.verse_name:<12} {r.policy_id:<18} {r.episodes:>8d} "
            f"{r.mean_return:>12.3f} {succ:>9} {r.mean_steps:>8.2f} {mtime:>20}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--verse", type=str, default="")
    ap.add_argument("--policy", type=str, default="")
    ap.add_argument("--run_id_contains", type=str, default="")
    ap.add_argument("--min_episodes", type=int, default=0)
    ap.add_argument("--min_mean_return", type=float, default=-1e18)
    ap.add_argument("--max_mean_return", type=float, default=1e18)
    ap.add_argument("--min_success_rate", type=float, default=0.0)
    ap.add_argument("--max_success_rate", type=float, default=1.0)
    ap.add_argument("--sort_by", type=str, default="mtime", choices=["mtime", "mean_return", "success_rate", "mean_steps", "episodes"])
    ap.add_argument("--order", type=str, default="desc", choices=["asc", "desc"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--latest", action="store_true", help="Shortcut for --sort_by mtime --order desc --limit 1")
    ap.add_argument("--format", type=str, default="table", choices=["table", "json", "ids", "ids_csv"])
    args = ap.parse_args()

    metric_filters_active = bool(
        int(args.min_episodes) > 0
        or float(args.min_mean_return) > -1e18
        or float(args.max_mean_return) < 1e18
        or float(args.min_success_rate) > 0.0
        or float(args.max_success_rate) < 1.0
    )
    if args.latest:
        args.sort_by = "mtime"
        args.order = "desc"
        args.limit = 1

    metric_sort = str(args.sort_by).strip().lower() in ("mean_return", "success_rate", "mean_steps", "episodes")
    metric_output = str(args.format).strip().lower() in ("table", "json")
    with_eval = bool(metric_filters_active or metric_sort or metric_output)

    rows: List[RunRow] = []
    for run_dir in _iter_run_dirs(args.runs_root):
        row = _load_run_row(run_dir, with_eval=with_eval)
        if row is None:
            continue
        if _passes_filters(row, args):
            rows.append(row)

    rows = _sort_rows(rows, args.sort_by, args.order)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    fmt = str(args.format).strip().lower()
    if fmt == "table":
        _print_table(rows)
    elif fmt == "json":
        payload: List[Dict[str, Any]] = []
        for r in rows:
            payload.append(
                {
                    "run_id": r.run_id,
                    "run_dir": r.run_dir,
                    "verse_name": r.verse_name,
                    "policy_id": r.policy_id,
                    "episodes": int(r.episodes),
                    "mean_return": float(r.mean_return),
                    "success_rate": r.success_rate,
                    "mean_steps": float(r.mean_steps),
                    "mtime": float(r.mtime),
                }
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif fmt == "ids":
        for r in rows:
            print(r.run_id)
    elif fmt == "ids_csv":
        print(",".join(r.run_id for r in rows))


if __name__ == "__main__":
    main()
