"""
tools/vector_store_refiner.py

Builds a cross-verse "Danger Map" by clustering failed states from run logs.
Uses memory/vector_store.py for similarity-backed incremental clustering.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.embeddings import obs_to_vector
from memory.vector_store import InMemoryVectorStore, VectorRecord


HAZARD_KEYS = {
    "hit_obstacle",
    "hit_wall",
    "battery_death",
    "battery_depleted",
    "fell_cliff",
    "fell_pit",
    "hit_laser",
    "collision",
    "crash",
    "unsafe",
}


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


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _list_run_dirs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[Tuple[float, str]] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        events_path = os.path.join(run_dir, "events.jsonl")
        if os.path.isdir(run_dir) and os.path.isfile(events_path):
            out.append((_safe_float(os.path.getmtime(events_path), 0.0), run_dir))
    out.sort(key=lambda t: float(t[0]), reverse=True)
    return [p for _, p in out]


def _peek_first_event(events_path: str) -> Dict[str, Any]:
    for row in _iter_jsonl(events_path):
        return row
    return {}


def _is_failed_event(row: Dict[str, Any], *, severe_reward_threshold: float) -> Tuple[bool, Dict[str, Any]]:
    info = row.get("info")
    info = info if isinstance(info, dict) else {}
    reward = _safe_float(row.get("reward", 0.0), 0.0)
    done = bool(row.get("done", False) or row.get("truncated", False))
    reached_goal = bool(info.get("reached_goal", False))
    hazard = any(bool(info.get(k, False)) for k in HAZARD_KEYS)
    severe = bool(reward <= float(severe_reward_threshold))
    terminal_fail = bool(done and not reached_goal)
    failed = bool(hazard or severe or terminal_fail)
    return failed, {
        "hazard": bool(hazard),
        "severe": bool(severe),
        "terminal_fail": bool(terminal_fail),
        "reward": float(reward),
    }


def _project_vector(vec: List[float], *, dim: int) -> List[float]:
    d = max(4, int(dim))
    out = [0.0 for _ in range(d)]
    for i, v in enumerate(vec):
        idx = int((i * 2654435761) % d)
        sign = -1.0 if (((i * 11400714819323198485) >> 3) & 1) else 1.0
        out[idx] += float(v) * float(sign)
    norm = math.sqrt(sum(x * x for x in out))
    if norm > 1e-12:
        out = [float(x / norm) for x in out]
    return out


@dataclass
class VectorStoreRefinerConfig:
    runs_root: str = "runs"
    out_json: str = os.path.join("models", "tuning", "global_danger_map.json")
    out_signatures_jsonl: str = os.path.join("models", "expert_datasets", "global_danger_signatures.jsonl")
    verse_filter: Optional[List[str]] = None
    policy_prefix: str = ""
    max_runs: int = 120
    max_events_per_run: int = 5000
    embedding_dim: int = 64
    cluster_similarity_threshold: float = 0.93
    match_top_k: int = 3
    min_cluster_size: int = 4
    severe_reward_threshold: float = -0.8
    samples_per_cluster: int = 4


def refine_danger_map(cfg: VectorStoreRefinerConfig) -> Dict[str, Any]:
    run_dirs = _list_run_dirs(str(cfg.runs_root))
    if int(cfg.max_runs) > 0:
        run_dirs = run_dirs[: int(cfg.max_runs)]
    verse_allow = set(str(v).strip().lower() for v in (cfg.verse_filter or []) if str(v).strip())
    prefix = str(cfg.policy_prefix or "").strip().lower()

    store = InMemoryVectorStore()
    cluster_records: Dict[int, VectorRecord] = {}
    clusters: Dict[int, Dict[str, Any]] = {}
    next_cluster_id = 0
    total_events = 0
    failed_events = 0

    for run_dir in run_dirs:
        events_path = os.path.join(run_dir, "events.jsonl")
        first = _peek_first_event(events_path)
        if not first:
            continue
        verse_name = str(first.get("verse_name", "")).strip().lower()
        if verse_allow and verse_name not in verse_allow:
            continue
        policy_id = str(first.get("policy_id", "")).strip().lower()
        if prefix and not policy_id.startswith(prefix):
            continue

        seen = 0
        run_id = os.path.basename(run_dir)
        for ev in _iter_jsonl(events_path):
            total_events += 1
            seen += 1
            if int(cfg.max_events_per_run) > 0 and seen > int(cfg.max_events_per_run):
                break
            failed, diag = _is_failed_event(ev, severe_reward_threshold=float(cfg.severe_reward_threshold))
            if not failed:
                continue
            obs = ev.get("obs")
            try:
                raw_vec = obs_to_vector(obs)
            except Exception:
                continue
            if not raw_vec:
                continue
            failed_events += 1
            vec = _project_vector(raw_vec, dim=max(4, int(cfg.embedding_dim)))

            cluster_id: Optional[int] = None
            matches = store.query(vec, top_k=max(1, int(cfg.match_top_k))) if cluster_records else []
            for m in matches:
                score = _safe_float(getattr(m, "score", 0.0), 0.0)
                md = getattr(m, "metadata", {})
                md = md if isinstance(md, dict) else {}
                cid = _safe_int(md.get("cluster_id", -1), -1)
                if cid < 0:
                    continue
                if score >= float(cfg.cluster_similarity_threshold):
                    cluster_id = int(cid)
                    break
            if cluster_id is None:
                cluster_id = int(next_cluster_id)
                next_cluster_id += 1
                clusters[cluster_id] = {
                    "cluster_id": int(cluster_id),
                    "size": 0,
                    "reward_sum": 0.0,
                    "hazard_count": 0,
                    "severe_count": 0,
                    "terminal_fail_count": 0,
                    "verse_counts": {},
                    "action_counts": {},
                    "centroid": [0.0 for _ in vec],
                    "samples": [],
                }
                rec = VectorRecord(
                    vector_id=f"cluster:{int(cluster_id)}",
                    vector=list(vec),
                    metadata={"cluster_id": int(cluster_id), "verse_name": verse_name},
                )
                store.add([rec])
                cluster_records[int(cluster_id)] = rec

            c = clusters[cluster_id]
            n = int(c["size"])
            centroid = list(c["centroid"])
            for i in range(len(centroid)):
                centroid[i] = float(centroid[i] + (vec[i] - centroid[i]) / float(n + 1))
            c["centroid"] = centroid
            c["size"] = int(n + 1)
            c["reward_sum"] = float(c["reward_sum"]) + float(diag["reward"])
            if bool(diag["hazard"]):
                c["hazard_count"] = int(c["hazard_count"]) + 1
            if bool(diag["severe"]):
                c["severe_count"] = int(c["severe_count"]) + 1
            if bool(diag["terminal_fail"]):
                c["terminal_fail_count"] = int(c["terminal_fail_count"]) + 1
            vc = c["verse_counts"]
            vc = vc if isinstance(vc, dict) else {}
            vc[verse_name] = int(_safe_int(vc.get(verse_name, 0), 0) + 1)
            c["verse_counts"] = vc
            try:
                action = int(ev.get("action"))
            except Exception:
                action = -1
            if action >= 0:
                ac = c["action_counts"]
                ac = ac if isinstance(ac, dict) else {}
                k = str(action)
                ac[k] = int(_safe_int(ac.get(k, 0), 0) + 1)
                c["action_counts"] = ac
            samples = c["samples"] if isinstance(c.get("samples"), list) else []
            if len(samples) < int(max(1, cfg.samples_per_cluster)):
                samples.append(
                    {
                        "run_id": str(run_id),
                        "episode_id": str(ev.get("episode_id", "")),
                        "step_idx": int(_safe_int(ev.get("step_idx", 0), 0)),
                        "verse_name": str(verse_name),
                        "obs": obs,
                        "action": int(action),
                        "reward": float(diag["reward"]),
                    }
                )
            c["samples"] = samples
            rec = cluster_records.get(int(cluster_id))
            if rec is not None:
                rec.vector = list(centroid)
                md = rec.metadata if isinstance(rec.metadata, dict) else {}
                md["verse_name"] = str(verse_name)
                rec.metadata = md

    rows: List[Dict[str, Any]] = []
    for cid in sorted(clusters.keys()):
        c = clusters[cid]
        size = max(1, int(c.get("size", 0)))
        if size < int(max(1, cfg.min_cluster_size)):
            continue
        hazard_rate = float(_safe_int(c.get("hazard_count", 0), 0) / float(size))
        severe_rate = float(_safe_int(c.get("severe_count", 0), 0) / float(size))
        terminal_fail_rate = float(_safe_int(c.get("terminal_fail_count", 0), 0) / float(size))
        mean_reward = float(_safe_float(c.get("reward_sum", 0.0), 0.0) / float(size))
        risk_score = float((0.5 * hazard_rate + 0.3 * severe_rate + 0.2 * terminal_fail_rate) * math.log1p(size))
        action_counts = c.get("action_counts", {})
        action_counts = action_counts if isinstance(action_counts, dict) else {}
        top_actions = sorted(
            [{"action": int(_safe_int(a, -1)), "count": int(_safe_int(n, 0))} for a, n in action_counts.items()],
            key=lambda r: int(r["count"]),
            reverse=True,
        )[:3]
        rows.append(
            {
                "cluster_id": int(cid),
                "size": int(size),
                "risk_score": float(risk_score),
                "mean_reward": float(mean_reward),
                "hazard_rate": float(hazard_rate),
                "severe_rate": float(severe_rate),
                "terminal_fail_rate": float(terminal_fail_rate),
                "verse_counts": dict(c.get("verse_counts", {})),
                "top_actions": top_actions,
                "centroid": list(c.get("centroid", [])),
                "samples": list(c.get("samples", [])),
            }
        )
    rows.sort(key=lambda r: float(r.get("risk_score", 0.0)), reverse=True)

    signatures_rows = 0
    os.makedirs(os.path.dirname(str(cfg.out_signatures_jsonl)) or ".", exist_ok=True)
    with open(str(cfg.out_signatures_jsonl), "w", encoding="utf-8") as out_sig:
        for row in rows:
            cid = int(_safe_int(row.get("cluster_id", -1), -1))
            risk = float(_safe_float(row.get("risk_score", 0.0), 0.0))
            actions = row.get("top_actions", [])
            actions = actions if isinstance(actions, list) else []
            avoid_action = None
            if actions:
                avoid_action = int(_safe_int((actions[0] or {}).get("action", -1), -1))
                if avoid_action < 0:
                    avoid_action = None
            for s in (row.get("samples", []) or []):
                if not isinstance(s, dict):
                    continue
                sig = {
                    "cluster_id": int(cid),
                    "risk_score": float(risk),
                    "source_verse": str(s.get("verse_name", "")),
                    "obs": s.get("obs"),
                    "avoid_action": avoid_action,
                    "source_reward": float(_safe_float(s.get("reward", 0.0), 0.0)),
                }
                out_sig.write(json.dumps(sig, ensure_ascii=False) + "\n")
                signatures_rows += 1

    report = {
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs_root": str(cfg.runs_root),
        "verse_filter": [str(v) for v in (cfg.verse_filter or [])],
        "policy_prefix": str(cfg.policy_prefix),
        "total_runs_scanned": int(len(run_dirs)),
        "total_events_scanned": int(total_events),
        "failed_events_vectorized": int(failed_events),
        "clusters_total": int(len(clusters)),
        "clusters_kept": int(len(rows)),
        "min_cluster_size": int(cfg.min_cluster_size),
        "embedding_dim": int(cfg.embedding_dim),
        "cluster_similarity_threshold": float(cfg.cluster_similarity_threshold),
        "out_signatures_jsonl": str(cfg.out_signatures_jsonl).replace("\\", "/"),
        "signatures_rows": int(signatures_rows),
        "clusters": rows,
    }
    os.makedirs(os.path.dirname(str(cfg.out_json)) or ".", exist_ok=True)
    with open(str(cfg.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--out_json", type=str, default=os.path.join("models", "tuning", "global_danger_map.json"))
    ap.add_argument(
        "--out_signatures_jsonl",
        type=str,
        default=os.path.join("models", "expert_datasets", "global_danger_signatures.jsonl"),
    )
    ap.add_argument("--verse_filter", action="append", default=None)
    ap.add_argument("--policy_prefix", type=str, default="")
    ap.add_argument("--max_runs", type=int, default=120)
    ap.add_argument("--max_events_per_run", type=int, default=5000)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--cluster_similarity_threshold", type=float, default=0.93)
    ap.add_argument("--match_top_k", type=int, default=3)
    ap.add_argument("--min_cluster_size", type=int, default=4)
    ap.add_argument("--severe_reward_threshold", type=float, default=-0.8)
    ap.add_argument("--samples_per_cluster", type=int, default=4)
    ap.add_argument("--print_top", type=int, default=10)
    args = ap.parse_args()

    cfg = VectorStoreRefinerConfig(
        runs_root=str(args.runs_root),
        out_json=str(args.out_json),
        out_signatures_jsonl=str(args.out_signatures_jsonl),
        verse_filter=args.verse_filter,
        policy_prefix=str(args.policy_prefix),
        max_runs=max(1, int(args.max_runs)),
        max_events_per_run=max(1, int(args.max_events_per_run)),
        embedding_dim=max(4, int(args.embedding_dim)),
        cluster_similarity_threshold=max(-1.0, min(1.0, float(args.cluster_similarity_threshold))),
        match_top_k=max(1, int(args.match_top_k)),
        min_cluster_size=max(1, int(args.min_cluster_size)),
        severe_reward_threshold=float(args.severe_reward_threshold),
        samples_per_cluster=max(1, int(args.samples_per_cluster)),
    )
    rep = refine_danger_map(cfg)
    print(
        f"danger_map_clusters={int(_safe_int(rep.get('clusters_kept', 0), 0))} "
        f"failed_events={int(_safe_int(rep.get('failed_events_vectorized', 0), 0))}"
    )
    top = max(0, int(args.print_top))
    for row in (rep.get("clusters", []) or [])[:top]:
        if not isinstance(row, dict):
            continue
        print(
            f"[cluster {int(_safe_int(row.get('cluster_id', -1), -1))}] "
            f"size={int(_safe_int(row.get('size', 0), 0))} "
            f"risk={float(_safe_float(row.get('risk_score', 0.0), 0.0)):.3f} "
            f"mean_reward={float(_safe_float(row.get('mean_reward', 0.0), 0.0)):.3f}"
        )
    print(f"danger_map_out={cfg.out_json}")
    print(f"danger_signatures_out={cfg.out_signatures_jsonl}")


if __name__ == "__main__":
    main()
