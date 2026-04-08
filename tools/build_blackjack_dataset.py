"""
Build logged blackjack transition datasets for offline DQN training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.blackjack_basic_agent import BlackjackBasicAgent
from tools.eval_blackjack import run_blackjack_case


def _load_events(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(str(run_dir), "events.jsonl")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _is_synthetic_action_row(row: Dict[str, Any]) -> bool:
    info = row.get("info") or {}
    if not isinstance(info, dict):
        return False
    action_info = info.get("action_info") or {}
    if not isinstance(action_info, dict):
        return False
    return bool(action_info.get("synthetic_action", False))


def _augment_row(
    row: Dict[str, Any],
    *,
    next_obs: Dict[str, Any],
    source_algo: str,
    source_run_id: str,
    seed: int,
) -> Dict[str, Any]:
    legal = (
        BlackjackBasicAgent._legal_actions_from_obs(next_obs)
        if isinstance(next_obs, dict)
        else [0, 1, 2, 3]
    )
    return {
        "episode_id": row.get("episode_id"),
        "step_idx": int(row.get("step_idx", 0) or 0),
        "obs": row.get("obs"),
        "action": int(row.get("action")),
        "reward": float(row.get("reward", 0.0) or 0.0),
        "done": bool(row.get("done", False) or row.get("truncated", False)),
        "truncated": bool(row.get("truncated", False)),
        "next_obs": next_obs,
        "legal_actions": [int(a) for a in legal],
        "source_policy": str(source_algo),
        "source_run_id": str(source_run_id),
        "seed": int(seed),
    }


def build_blackjack_dataset_from_run(
    *,
    run_dir: str,
    out_path: str,
    source_algo: str,
    source_run_id: str,
    seed: int,
) -> Dict[str, Any]:
    events = _load_events(run_dir)
    by_episode: Dict[str, List[Dict[str, Any]]] = {}
    for row in events:
        ep = str(row.get("episode_id", "") or "")
        if not ep:
            continue
        by_episode.setdefault(ep, []).append(row)

    rows = 0
    skipped_synthetic = 0
    with open(out_path, "w", encoding="utf-8") as dst:
        for _, ep_rows in by_episode.items():
            ep_rows.sort(key=lambda r: int(r.get("step_idx", 0) or 0))
            for idx, row in enumerate(ep_rows):
                if _is_synthetic_action_row(row):
                    skipped_synthetic += 1
                    continue
                action = row.get("action")
                if action is None:
                    continue
                try:
                    int(action)
                except Exception:
                    continue
                next_obs = row.get("obs")
                if idx + 1 < len(ep_rows):
                    next_obs = ep_rows[idx + 1].get("obs", next_obs)
                payload = _augment_row(
                    row,
                    next_obs=next_obs,
                    source_algo=source_algo,
                    source_run_id=source_run_id,
                    seed=seed,
                )
                dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
                rows += 1

    return {
        "dataset_path": str(out_path).replace("\\", "/"),
        "rows": int(rows),
        "skipped_synthetic_rows": int(skipped_synthetic),
        "source_algo": str(source_algo),
        "source_run_id": str(source_run_id),
        "source_run_dir": str(run_dir).replace("\\", "/"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_algo", type=str, default="blackjack_basic")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--runs_root", type=str, default="runs_smoke")
    ap.add_argument("--out_path", type=str, default="")
    args = ap.parse_args()

    run = run_blackjack_case(
        algo=str(args.source_algo),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        runs_root=str(args.runs_root),
    )
    run_dir = str(run["run_dir"])

    out_path = str(args.out_path or "").strip()
    if not out_path:
        out_dir = os.path.join(str(args.runs_root), "datasets")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"blackjack_{str(args.source_algo).strip().lower()}_{int(args.episodes)}_{int(args.seed)}.jsonl",
        )
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    payload = build_blackjack_dataset_from_run(
        run_dir=run_dir,
        out_path=out_path,
        source_algo=str(args.source_algo),
        source_run_id=str(run["run_id"]),
        seed=int(args.seed),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
