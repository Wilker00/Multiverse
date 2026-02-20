"""
tools/mcts_telemetry_dashboard.py

Summarize SafeExecutor MCTS telemetry from events.jsonl files.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


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


def _maybe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class MCTSTelemetryRow:
    run_id: str
    run_dir: str
    mtime: float
    verse_name: str
    policy_id: str
    steps: int
    episodes: int
    mcts_enabled_episodes: int
    query_events: int
    veto_events: int
    forced_loss_queries: int
    avg_root_value: float
    avg_leaf_value: float
    avg_simulations: float
    avg_pv_len: float

    @property
    def veto_rate(self) -> float:
        return float(self.veto_events / float(max(1, self.query_events)))


def _iter_run_dirs(runs_root: str) -> Iterable[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[str] = []
    for name in os.listdir(runs_root):
        p = os.path.join(runs_root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "events.jsonl")):
            out.append(p)
    return out


def analyze_events_file(events_path: str) -> Optional[MCTSTelemetryRow]:
    if not os.path.isfile(events_path):
        return None
    run_dir = os.path.dirname(os.path.abspath(events_path))
    run_id = os.path.basename(run_dir)
    mtime = _safe_float(os.path.getmtime(events_path), 0.0)

    verse_name = ""
    policy_id = ""

    steps = 0
    episodes: set[str] = set()
    enabled_eps: set[str] = set()
    ep_counters: Dict[str, Dict[str, int]] = {}

    query_events = 0
    veto_events = 0
    forced_loss_queries = 0

    root_sum = 0.0
    root_n = 0
    leaf_sum = 0.0
    leaf_n = 0
    sims_sum = 0.0
    sims_n = 0
    pv_sum = 0.0
    pv_n = 0

    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            steps += 1

            if not verse_name:
                verse_name = str(ev.get("verse_name", "")).strip()
            if not policy_id:
                policy_id = str(ev.get("policy_id", "")).strip()

            ep_id = str(ev.get("episode_id", "")).strip()
            if ep_id:
                episodes.add(ep_id)
            else:
                ep_id = f"_unknown_{steps}"

            info = ev.get("info")
            if not isinstance(info, dict):
                continue
            se = info.get("safe_executor")
            if not isinstance(se, dict):
                continue
            stats = se.get("mcts_stats")
            if not isinstance(stats, dict):
                continue

            if bool(stats.get("enabled", False)):
                enabled_eps.add(ep_id)

            q = max(0, _safe_int(stats.get("queries", 0), 0))
            v = max(0, _safe_int(stats.get("vetoes", 0), 0))
            prev = ep_counters.get(ep_id, {"queries": 0, "vetoes": 0})
            prev_q = _safe_int(prev.get("queries", 0), 0)
            prev_v = _safe_int(prev.get("vetoes", 0), 0)
            dq = q - prev_q if q >= prev_q else q
            dv = v - prev_v if v >= prev_v else v
            ep_counters[ep_id] = {"queries": q, "vetoes": v}

            if dq > 0:
                query_events += int(dq)
                last_query = stats.get("last_query")
                if isinstance(last_query, dict):
                    if bool(last_query.get("forced_loss_detected", False)):
                        forced_loss_queries += int(dq)
                    rv = _maybe_float(last_query.get("root_value"))
                    if rv is not None:
                        root_sum += float(rv) * float(dq)
                        root_n += int(dq)
                    lv = _maybe_float(last_query.get("avg_leaf_value"))
                    if lv is not None:
                        leaf_sum += float(lv) * float(dq)
                        leaf_n += int(dq)
                    sims = _maybe_float(last_query.get("simulations"))
                    if sims is not None:
                        sims_sum += float(sims) * float(dq)
                        sims_n += int(dq)
                    pv = last_query.get("principal_variation")
                    if isinstance(pv, list):
                        pv_sum += float(len(pv)) * float(dq)
                        pv_n += int(dq)

            if dv > 0:
                veto_events += int(dv)

    return MCTSTelemetryRow(
        run_id=run_id,
        run_dir=run_dir.replace("\\", "/"),
        mtime=mtime,
        verse_name=verse_name,
        policy_id=policy_id,
        steps=int(steps),
        episodes=len(episodes),
        mcts_enabled_episodes=len(enabled_eps),
        query_events=int(query_events),
        veto_events=int(veto_events),
        forced_loss_queries=int(forced_loss_queries),
        avg_root_value=(float(root_sum) / float(root_n)) if root_n > 0 else 0.0,
        avg_leaf_value=(float(leaf_sum) / float(leaf_n)) if leaf_n > 0 else 0.0,
        avg_simulations=(float(sims_sum) / float(sims_n)) if sims_n > 0 else 0.0,
        avg_pv_len=(float(pv_sum) / float(pv_n)) if pv_n > 0 else 0.0,
    )


def load_rows(*, runs_root: str, run_dir: str, events_path: str) -> List[MCTSTelemetryRow]:
    rows: List[MCTSTelemetryRow] = []
    if str(events_path).strip():
        row = analyze_events_file(str(events_path).strip())
        return [row] if row is not None else []
    if str(run_dir).strip():
        row = analyze_events_file(os.path.join(str(run_dir).strip(), "events.jsonl"))
        return [row] if row is not None else []
    for rd in _iter_run_dirs(runs_root):
        row = analyze_events_file(os.path.join(rd, "events.jsonl"))
        if row is not None:
            rows.append(row)
    return rows


def aggregate_by_verse(rows: List[MCTSTelemetryRow]) -> List[Dict[str, Any]]:
    bucket: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = str(r.verse_name or "unknown")
        if key not in bucket:
            bucket[key] = {
                "verse_name": key,
                "runs": 0,
                "episodes": 0,
                "mcts_enabled_episodes": 0,
                "query_events": 0,
                "veto_events": 0,
                "forced_loss_queries": 0,
                "root_sum": 0.0,
                "root_weight": 0,
                "leaf_sum": 0.0,
                "leaf_weight": 0,
                "sim_sum": 0.0,
                "sim_weight": 0,
            }
        b = bucket[key]
        b["runs"] = int(b["runs"]) + 1
        b["episodes"] = int(b["episodes"]) + int(r.episodes)
        b["mcts_enabled_episodes"] = int(b["mcts_enabled_episodes"]) + int(r.mcts_enabled_episodes)
        b["query_events"] = int(b["query_events"]) + int(r.query_events)
        b["veto_events"] = int(b["veto_events"]) + int(r.veto_events)
        b["forced_loss_queries"] = int(b["forced_loss_queries"]) + int(r.forced_loss_queries)
        b["root_sum"] = float(b["root_sum"]) + float(r.avg_root_value) * float(r.query_events)
        b["root_weight"] = int(b["root_weight"]) + int(r.query_events)
        b["leaf_sum"] = float(b["leaf_sum"]) + float(r.avg_leaf_value) * float(r.query_events)
        b["leaf_weight"] = int(b["leaf_weight"]) + int(r.query_events)
        b["sim_sum"] = float(b["sim_sum"]) + float(r.avg_simulations) * float(r.query_events)
        b["sim_weight"] = int(b["sim_weight"]) + int(r.query_events)
    out: List[Dict[str, Any]] = []
    for key, b in bucket.items():
        queries = int(b["query_events"])
        root_weight = int(b["root_weight"])
        leaf_weight = int(b["leaf_weight"])
        sim_weight = int(b["sim_weight"])
        out.append(
            {
                "verse_name": key,
                "runs": int(b["runs"]),
                "episodes": int(b["episodes"]),
                "mcts_enabled_episodes": int(b["mcts_enabled_episodes"]),
                "query_events": queries,
                "veto_events": int(b["veto_events"]),
                "veto_rate": float(int(b["veto_events"]) / float(max(1, queries))),
                "forced_loss_queries": int(b["forced_loss_queries"]),
                "avg_root_value": float(b["root_sum"] / float(max(1, root_weight))),
                "avg_leaf_value": float(b["leaf_sum"] / float(max(1, leaf_weight))),
                "avg_simulations": float(b["sim_sum"] / float(max(1, sim_weight))),
            }
        )
    out.sort(key=lambda x: _safe_int(x.get("query_events"), 0), reverse=True)
    return out


def _sort_rows(rows: List[MCTSTelemetryRow], sort_by: str, order: str) -> List[MCTSTelemetryRow]:
    reverse = str(order).strip().lower() != "asc"
    key_name = str(sort_by).strip().lower()
    if key_name == "queries":
        key = lambda r: r.query_events
    elif key_name == "veto_rate":
        key = lambda r: r.veto_rate
    elif key_name == "forced_loss":
        key = lambda r: r.forced_loss_queries
    else:
        key = lambda r: r.mtime
    return sorted(rows, key=key, reverse=reverse)


def _print_table(rows: List[MCTSTelemetryRow]) -> None:
    print(
        f"{'run_id':<36} {'verse':<12} {'policy':<16} {'eps':>5} {'en_eps':>6} "
        f"{'queries':>8} {'vetoes':>7} {'v_rate':>8} {'forced':>7} {'avg_root':>9} {'avg_leaf':>9} {'avg_sim':>8} {'mtime':>19}"
    )
    for r in rows:
        mtime = datetime.fromtimestamp(float(r.mtime)).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{r.run_id:<36} {r.verse_name:<12} {r.policy_id:<16} {r.episodes:>5d} {r.mcts_enabled_episodes:>6d} "
            f"{r.query_events:>8d} {r.veto_events:>7d} {100.0 * r.veto_rate:>7.1f}% {r.forced_loss_queries:>7d} "
            f"{r.avg_root_value:>9.3f} {r.avg_leaf_value:>9.3f} {r.avg_simulations:>8.1f} {mtime:>19}"
        )


def _print_verse_table(rows: List[Dict[str, Any]]) -> None:
    print(
        f"{'verse':<14} {'runs':>4} {'eps':>6} {'en_eps':>6} {'queries':>8} {'vetoes':>7} "
        f"{'v_rate':>8} {'forced':>7} {'avg_root':>9} {'avg_leaf':>9} {'avg_sim':>8}"
    )
    for r in rows:
        print(
            f"{str(r.get('verse_name', '')):<14} {int(r.get('runs', 0)):>4d} {int(r.get('episodes', 0)):>6d} "
            f"{int(r.get('mcts_enabled_episodes', 0)):>6d} {int(r.get('query_events', 0)):>8d} "
            f"{int(r.get('veto_events', 0)):>7d} {100.0 * _safe_float(r.get('veto_rate', 0.0)):>7.1f}% "
            f"{int(r.get('forced_loss_queries', 0)):>7d} {_safe_float(r.get('avg_root_value', 0.0)):>9.3f} "
            f"{_safe_float(r.get('avg_leaf_value', 0.0)):>9.3f} {_safe_float(r.get('avg_simulations', 0.0)):>8.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--run_dir", type=str, default="")
    ap.add_argument("--events_path", type=str, default="")
    ap.add_argument("--verse", type=str, default="")
    ap.add_argument("--policy", type=str, default="")
    ap.add_argument("--min_queries", type=int, default=0)
    ap.add_argument("--only_enabled", action="store_true", help="Only include runs with mcts_enabled_episodes > 0")
    ap.add_argument("--sort_by", type=str, default="mtime", choices=["mtime", "queries", "veto_rate", "forced_loss"])
    ap.add_argument("--order", type=str, default="desc", choices=["asc", "desc"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--by_verse", action="store_true")
    ap.add_argument("--format", type=str, default="table", choices=["table", "json"])
    args = ap.parse_args()

    rows = load_rows(
        runs_root=str(args.runs_root),
        run_dir=str(args.run_dir),
        events_path=str(args.events_path),
    )
    filtered: List[MCTSTelemetryRow] = []
    for row in rows:
        if args.verse and row.verse_name != str(args.verse).strip():
            continue
        if args.policy and row.policy_id != str(args.policy).strip():
            continue
        if row.query_events < int(args.min_queries):
            continue
        if bool(args.only_enabled) and int(row.mcts_enabled_episodes) <= 0:
            continue
        filtered.append(row)

    filtered = _sort_rows(filtered, str(args.sort_by), str(args.order))
    if int(args.limit) > 0:
        filtered = filtered[: int(args.limit)]

    if str(args.format).strip().lower() == "json":
        payload: Dict[str, Any] = {
            "runs": [
                {
                    "run_id": r.run_id,
                    "run_dir": r.run_dir,
                    "verse_name": r.verse_name,
                    "policy_id": r.policy_id,
                    "steps": int(r.steps),
                    "episodes": int(r.episodes),
                    "mcts_enabled_episodes": int(r.mcts_enabled_episodes),
                    "query_events": int(r.query_events),
                    "veto_events": int(r.veto_events),
                    "veto_rate": float(r.veto_rate),
                    "forced_loss_queries": int(r.forced_loss_queries),
                    "avg_root_value": float(r.avg_root_value),
                    "avg_leaf_value": float(r.avg_leaf_value),
                    "avg_simulations": float(r.avg_simulations),
                    "avg_pv_len": float(r.avg_pv_len),
                    "mtime": float(r.mtime),
                }
                for r in filtered
            ]
        }
        if bool(args.by_verse):
            payload["by_verse"] = aggregate_by_verse(filtered)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not filtered:
        print("No runs matched filters.")
        return
    _print_table(filtered)
    if bool(args.by_verse):
        print("")
        _print_verse_table(aggregate_by_verse(filtered))


if __name__ == "__main__":
    main()
