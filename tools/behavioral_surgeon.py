"""
tools/behavioral_surgeon.py

Behavioral Surgeon pipeline:
1) Bridge grid_world expert DNA into park_world synthetic expert data.
2) Score cliff_world trajectories and deduplicate cliff memory slice.
3) Extract hard death transitions for failure-aware action masking.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.central_repository import CentralMemoryConfig, ingest_run
from memory.embeddings import cosine_similarity, obs_to_vector
from memory.selection import ActiveForgettingConfig, SelectionConfig, active_forget_central_memory
from memory.semantic_bridge import translate_dna


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _obs_key(obs: Any) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(obs)


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


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _merge_datasets(paths: List[str], out_path: str) -> Tuple[int, int]:
    seen = set()
    merged: List[Dict[str, Any]] = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        for row in _iter_jsonl(p):
            key = (_obs_key(row.get("obs")), json.dumps(row.get("action"), ensure_ascii=False))
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    count = _write_jsonl(out_path, merged)
    return len(seen), count


def _discover_cliff_runs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    found: List[Tuple[float, str]] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        events_path = os.path.join(run_dir, "events.jsonl")
        if not (os.path.isdir(run_dir) and os.path.isfile(events_path)):
            continue
        first = None
        with open(events_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                try:
                    first = json.loads(first_line)
                except Exception:
                    first = None
        if not isinstance(first, dict):
            continue
        if str(first.get("verse_name", "")).strip().lower() != "cliff_world":
            continue
        try:
            mtime = os.path.getmtime(events_path)
        except Exception:
            mtime = 0.0
        found.append((float(mtime), run_dir))
    found.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in found]


@dataclass
class DeathExtractStats:
    input_rows: int
    death_rows: int
    pruned_rows: int


def _extract_death_transitions(
    *,
    run_dir: str,
    reward_threshold: float,
    similarity_threshold: float,
) -> Tuple[List[Dict[str, Any]], DeathExtractStats]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return [], DeathExtractStats(0, 0, 0)

    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    input_rows = 0
    for row in _iter_jsonl(events_path):
        input_rows += 1
        ep = str(row.get("episode_id", ""))
        by_ep.setdefault(ep, []).append(row)

    death_rows: List[Dict[str, Any]] = []
    for ep_events in by_ep.values():
        ep_events.sort(key=lambda x: int(x.get("step_idx", 0) or 0))
        for i, e in enumerate(ep_events):
            info = e.get("info") if isinstance(e.get("info"), dict) else {}
            fell = bool(info.get("fell_cliff") is True)
            reward = _safe_float(e.get("reward", 0.0))
            severe = reward <= float(reward_threshold)
            if not (fell or severe):
                continue
            row = {
                "episode_id": e.get("episode_id"),
                "step_idx": int(e.get("step_idx", i) or i),
                "obs": e.get("obs"),
                "action": e.get("action"),
                "reward": reward,
                "done": bool(e.get("done") or e.get("truncated")),
                "next_obs": ep_events[i + 1].get("obs") if i + 1 < len(ep_events) else None,
                "death_signal": "fell_cliff" if fell else "severe_penalty",
                "source_run_id": str(e.get("run_id", os.path.basename(run_dir))),
            }
            death_rows.append(row)

    # Active-forgetting style pruning for death transitions:
    # keep representative rows, prefer more negative rewards for near-duplicates.
    masters: List[Tuple[List[float], int, Dict[str, Any]]] = []
    for row in death_rows:
        obs = row.get("obs")
        action = int(row.get("action", 0) or 0)
        try:
            vec = obs_to_vector(obs)
        except Exception:
            vec = []
        if not vec:
            masters.append((vec, action, row))
            continue

        replaced = False
        keep_new = True
        for idx, (m_vec, m_act, m_row) in enumerate(masters):
            if not m_vec or len(m_vec) != len(vec):
                continue
            if m_act != action:
                continue
            sim = cosine_similarity(vec, m_vec)
            if sim >= float(similarity_threshold):
                # Keep the row with more severe penalty.
                if _safe_float(row.get("reward", 0.0)) < _safe_float(m_row.get("reward", 0.0)):
                    masters[idx] = (vec, action, row)
                    replaced = True
                keep_new = False
                break
        if keep_new:
            masters.append((vec, action, row))
        elif replaced:
            pass

    pruned = [row for _, _, row in masters]
    st = DeathExtractStats(input_rows=input_rows, death_rows=len(death_rows), pruned_rows=len(pruned))
    return pruned, st


def _run_behavioral_scorer(run_dir: str, top_percent: float) -> None:
    cmd = [
        sys.executable,
        os.path.join("tools", "behavioral_scorer.py"),
        "--run_dir",
        run_dir,
        "--top_percent",
        str(float(top_percent)),
        "--scores_out",
        "behavior_scores.cliff.jsonl",
        "--golden_out",
        "golden_dna.cliff.jsonl",
    ]
    subprocess.run(cmd, check=True)


def _train_failure_aware(
    *,
    verse: str,
    good_dataset: str,
    bad_dataset: str,
    episodes: int,
    max_steps: int,
    seed: int,
    danger_temperature: float,
) -> None:
    cmd = [
        sys.executable,
        os.path.join("tools", "train_agent.py"),
        "--algo",
        "failure_aware",
        "--verse",
        str(verse),
        "--episodes",
        str(int(episodes)),
        "--max_steps",
        str(int(max_steps)),
        "--seed",
        str(int(seed)),
        "--dataset",
        str(good_dataset),
        "--bad_dna",
        str(bad_dataset),
        "--aconfig",
        f"danger_temperature={float(danger_temperature)}",
        "--eval",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--grid_dataset", type=str, default=os.path.join("models", "expert_datasets", "grid_world.jsonl"))
    ap.add_argument("--park_dataset", type=str, default=os.path.join("models", "expert_datasets", "park_world.jsonl"))
    ap.add_argument(
        "--park_synthetic_out",
        type=str,
        default=os.path.join("models", "expert_datasets", "synthetic_expert_park_world.jsonl"),
    )
    ap.add_argument(
        "--park_augmented_out",
        type=str,
        default=os.path.join("models", "expert_datasets", "park_world_augmented.jsonl"),
    )

    ap.add_argument(
        "--cliff_run_ids",
        type=str,
        default="",
        help="Comma-separated run ids. Empty => auto-discover cliff_world runs.",
    )
    ap.add_argument("--cliff_lookback_runs", type=int, default=10)
    ap.add_argument("--cliff_reward_threshold", type=float, default=-50.0)
    ap.add_argument("--death_similarity_threshold", type=float, default=0.95)
    ap.add_argument(
        "--death_out",
        type=str,
        default=os.path.join("models", "expert_datasets", "cliff_death_transitions.jsonl"),
    )
    ap.add_argument("--behavior_top_percent", type=float, default=20.0)

    ap.add_argument(
        "--cliff_memory_dir",
        type=str,
        default="central_memory_cliff",
        help="Dedicated memory dir for cliff-only active forgetting.",
    )
    ap.add_argument("--active_forgetting_similarity", type=float, default=0.95)
    ap.add_argument("--active_forgetting_min_events", type=int, default=20)

    ap.add_argument("--train_failure_aware", action="store_true")
    ap.add_argument("--failure_good_dataset", type=str, default=os.path.join("models", "expert_datasets", "cliff_world.jsonl"))
    ap.add_argument("--failure_episodes", type=int, default=80)
    ap.add_argument("--failure_max_steps", type=int, default=100)
    ap.add_argument("--failure_seed", type=int, default=123)
    ap.add_argument("--failure_danger_temperature", type=float, default=1.8)
    args = ap.parse_args()

    # 1) Semantic bridge: grid -> park
    bridge = translate_dna(
        source_dna_path=args.grid_dataset,
        target_verse_name="park_world",
        output_path=args.park_synthetic_out,
        source_verse_name="grid_world",
    )
    merged_unique, merged_rows = _merge_datasets(
        [args.park_dataset, args.park_synthetic_out],
        args.park_augmented_out,
    )

    # 2) Cliff run selection
    if args.cliff_run_ids.strip():
        cliff_runs = [os.path.join(args.runs_root, rid.strip()) for rid in args.cliff_run_ids.split(",") if rid.strip()]
    else:
        cliff_runs = _discover_cliff_runs(args.runs_root)[: max(1, int(args.cliff_lookback_runs))]
    cliff_runs = [p for p in cliff_runs if os.path.isdir(p)]

    if not cliff_runs:
        raise RuntimeError("No cliff_world runs found for behavioral surgery.")

    # 3) Behavioral scoring + cliff-only memory ingest
    total_input = 0
    total_death = 0
    all_pruned_death_rows: List[Dict[str, Any]] = []
    os.makedirs(args.cliff_memory_dir, exist_ok=True)
    for run_dir in cliff_runs:
        _run_behavioral_scorer(run_dir, top_percent=float(args.behavior_top_percent))

        ingest_run(
            run_dir=run_dir,
            cfg=CentralMemoryConfig(root_dir=args.cliff_memory_dir),
            selection=SelectionConfig(
                keep_top_k_per_episode=40,
                keep_top_k_episodes=40,
                novelty_bonus=0.1,
            ),
        )

        pruned_rows, st = _extract_death_transitions(
            run_dir=run_dir,
            reward_threshold=float(args.cliff_reward_threshold),
            similarity_threshold=float(args.death_similarity_threshold),
        )
        total_input += int(st.input_rows)
        total_death += int(st.death_rows)
        all_pruned_death_rows.extend(pruned_rows)

    # 4) Global pruning over merged death transitions
    final_rows: List[Dict[str, Any]] = []
    seen_local: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in all_pruned_death_rows:
        k = _obs_key(row.get("obs"))
        a = int(row.get("action", 0) or 0)
        bucket = seen_local.setdefault(k, {})
        prev = bucket.get(a)
        if prev is None or _safe_float(row.get("reward", 0.0)) < _safe_float(prev.get("reward", 0.0)):
            bucket[a] = row
    for by_action in seen_local.values():
        final_rows.extend(by_action.values())
    death_written = _write_jsonl(args.death_out, final_rows)

    # 5) Active forgetting over cliff memory slice
    forget = active_forget_central_memory(
        ActiveForgettingConfig(
            memory_dir=args.cliff_memory_dir,
            similarity_threshold=float(args.active_forgetting_similarity),
            min_events_per_run=int(args.active_forgetting_min_events),
            write_backup=True,
        )
    )

    # 6) Optional failure-aware training run
    if args.train_failure_aware:
        _train_failure_aware(
            verse="cliff_world",
            good_dataset=args.failure_good_dataset,
            bad_dataset=args.death_out,
            episodes=int(args.failure_episodes),
            max_steps=int(args.failure_max_steps),
            seed=int(args.failure_seed),
            danger_temperature=float(args.failure_danger_temperature),
        )

    print("Behavioral Surgeon complete")
    print(f"park_bridge_input_rows         : {bridge.input_rows}")
    print(f"park_bridge_translated_rows    : {bridge.translated_rows}")
    print(f"park_augmented_unique_rows     : {merged_unique}")
    print(f"park_augmented_written_rows    : {merged_rows}")
    print(f"cliff_runs_processed           : {len(cliff_runs)}")
    print(f"cliff_input_rows               : {total_input}")
    print(f"cliff_death_rows_raw           : {total_death}")
    print(f"cliff_death_rows_written       : {death_written}")
    print(f"cliff_memory_input_rows        : {forget.input_rows}")
    print(f"cliff_memory_kept_rows         : {forget.kept_rows}")
    print(f"cliff_memory_dropped_rows      : {forget.dropped_rows}")
    print(f"cliff_memory_dropped_runs      : {forget.dropped_runs}")
    print(f"death_dataset_path             : {args.death_out}")
    print(f"park_augmented_dataset_path    : {args.park_augmented_out}")


if __name__ == "__main__":
    main()
