from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from core.universe_registry import build_transfer_source_plan, primary_universe_for_verse
from orchestrator.evaluator import evaluate_run


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


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
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


def peek_first_event(events_path: str) -> Dict[str, Any]:
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
    except Exception:
        return {}
    return {}


def list_run_dirs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[str] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        if os.path.isdir(run_dir) and os.path.isfile(os.path.join(run_dir, "events.jsonl")):
            out.append(run_dir)
    return out


def extract_success_dna_from_events(
    *,
    run_dir: str,
    out_path: str,
    max_rows: int = 5000,
) -> int:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return 0
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for ev in iter_jsonl(events_path):
        ep = str(ev.get("episode_id", "")).strip()
        if not ep:
            continue
        if ep not in by_ep:
            by_ep[ep] = []
            order.append(ep)
        by_ep[ep].append(ev)

    success_eps = set()
    for ep in order:
        rows = by_ep.get(ep, [])
        if any(bool((r.get("info") or {}).get("reached_goal", False)) for r in rows if isinstance(r, dict)):
            success_eps.add(ep)
    if not success_eps:
        return 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ep in order:
            if ep not in success_eps:
                continue
            rows = by_ep.get(ep, [])
            rows.sort(key=lambda r: _safe_int(r.get("step_idx", 0), 0))
            for i, ev in enumerate(rows):
                if written >= int(max_rows):
                    return written
                obs = ev.get("obs")
                action = ev.get("action")
                try:
                    a = int(action)
                except Exception:
                    continue
                next_obs = rows[i + 1].get("obs") if i + 1 < len(rows) else obs
                row = {
                    "episode_id": str(ev.get("episode_id", "")),
                    "step_idx": _safe_int(ev.get("step_idx", i), i),
                    "verse_name": str(ev.get("verse_name", "")),
                    "obs": obs,
                    "action": int(a),
                    "reward": _safe_float(ev.get("reward", 0.0), 0.0),
                    "done": bool(ev.get("done", False) or ev.get("truncated", False)),
                    "next_obs": next_obs,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    return written


@dataclass
class TransferSourceDNA:
    verse_name: str
    path: str
    run_id: str
    source_kind: str
    source_lane: str = "far_universe"
    source_universe: str = ""


def discover_transfer_sources(
    *,
    target_verse: str,
    runs_root: str,
    max_runs_per_verse: int,
    min_success_rate: float,
    min_rows_per_source: int,
    max_source_scan: int = 200,
) -> List[TransferSourceDNA]:
    plan = build_transfer_source_plan(str(target_verse))
    ordered_sources = [str(v).strip().lower() for v in (plan.get("ordered_sources") or []) if str(v).strip()]
    if not ordered_sources:
        ordered_sources = ["chess_world", "go_world", "trade_world", "uno_world"]
    near_set = set(str(v).strip().lower() for v in (plan.get("near_sources") or []) if str(v).strip())
    all_targets = tuple(ordered_sources)
    candidates_by_verse: Dict[str, List[Tuple[str, float, float]]] = {v: [] for v in all_targets}
    scanned = 0
    for run_dir in list_run_dirs(runs_root):
        events_path = os.path.join(run_dir, "events.jsonl")
        first = peek_first_event(events_path)
        verse_name = str(first.get("verse_name", "")).strip().lower()
        if verse_name not in all_targets:
            continue
        scanned += 1
        if int(max_source_scan) > 0 and scanned > int(max_source_scan):
            break
        try:
            st = evaluate_run(run_dir)
        except Exception:
            continue
        success_rate = float(st.success_rate or 0.0)
        if success_rate < float(min_success_rate):
            continue
        mtime = _safe_float(os.path.getmtime(events_path), 0.0)
        candidates_by_verse[verse_name].append((run_dir, success_rate, mtime))

    out: List[TransferSourceDNA] = []
    for verse_name in all_targets:
        cands = sorted(
            candidates_by_verse.get(verse_name, []),
            key=lambda t: (float(t[1]), float(t[2])),
            reverse=True,
        )[: max(1, int(max_runs_per_verse))]
        for run_dir, _, _ in cands:
            run_id = os.path.basename(run_dir)
            dna_good = os.path.join(run_dir, "dna_good.jsonl")
            if os.path.isfile(dna_good):
                rows = sum(1 for _ in iter_jsonl(dna_good))
                if rows >= int(min_rows_per_source):
                    out.append(
                        TransferSourceDNA(
                            verse_name=verse_name,
                            path=dna_good,
                            run_id=run_id,
                            source_kind="dna_good",
                            source_lane=("near_universe" if verse_name in near_set else "far_universe"),
                            source_universe=(primary_universe_for_verse(verse_name) or ""),
                        )
                    )
                    continue
            succ = os.path.join(run_dir, "dna_success_only.jsonl")
            rows = extract_success_dna_from_events(run_dir=run_dir, out_path=succ, max_rows=12000)
            if rows >= int(min_rows_per_source):
                out.append(
                    TransferSourceDNA(
                        verse_name=verse_name,
                        path=succ,
                        run_id=run_id,
                        source_kind="success_events",
                        source_lane=("near_universe" if verse_name in near_set else "far_universe"),
                        source_universe=(primary_universe_for_verse(verse_name) or ""),
                    )
                )

    seen_verses = set(str(s.verse_name).strip().lower() for s in out if str(s.verse_name).strip())
    for verse_name in all_targets:
        if verse_name in seen_verses:
            continue
        p = os.path.join("models", "expert_datasets", f"{verse_name}.jsonl")
        if os.path.isfile(p):
            out.append(
                TransferSourceDNA(
                    verse_name=verse_name,
                    path=p,
                    run_id="expert_dataset",
                    source_kind="fallback",
                    source_lane=("near_universe" if verse_name in near_set else "far_universe"),
                    source_universe=(primary_universe_for_verse(verse_name) or ""),
                )
            )
    return out


def merge_jsonl(paths: List[str], out_path: str, *, max_rows_per_file: int = 0) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            if not os.path.isfile(p):
                continue
            taken = 0
            for row in iter_jsonl(p):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                taken += 1
                if int(max_rows_per_file) > 0 and taken >= int(max_rows_per_file):
                    break
    return rows
