"""
memory/apprentice_dataset.py

Build supervised / offline RL datasets from stored experience.

Purpose:
- Convert retrieved episodes into (obs -> action) pairs
- Optionally include rewards, next_obs for offline RL
- Serve as the bridge between exploration data and native agents

This file does NOT train models.
It only prepares clean, explicit datasets.

Design rule:
Data in, data out. No learning logic here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class ApprenticeDatasetConfig:
    run_dir: str
    events_filename: str = "events.jsonl"
    output_filename: str = "apprentice_dataset.jsonl"

    include_next_obs: bool = True
    include_reward: bool = True
    include_done: bool = True


def _load_events(events_path: str) -> Iterator[Dict[str, Any]]:
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def build_apprentice_dataset(
    *,
    cfg: ApprenticeDatasetConfig,
    episode_ids: Optional[List[str]] = None,
) -> str:
    """
    Build a dataset from selected episodes.

    Each output row is one transition:
    {
        "episode_id",
        "step_idx",
        "obs",
        "action",
        "next_obs",
        "reward",
        "done"
    }

    Returns output file path.
    """
    if not os.path.isdir(cfg.run_dir):
        raise FileNotFoundError(f"run_dir not found: {cfg.run_dir}")

    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"events file not found: {events_path}")

    events = _load_events(events_path)

    # Group by episode
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ep = str(e.get("episode_id"))
        if episode_ids is not None and ep not in episode_ids:
            continue
        by_ep.setdefault(ep, []).append(e)

    out_path = os.path.join(cfg.run_dir, cfg.output_filename)
    count = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for ep_id, ep_events in by_ep.items():
            ep_events.sort(key=lambda x: _safe_int(x.get("step_idx", 0)))

            for i, e in enumerate(ep_events):
                row: Dict[str, Any] = {
                    "episode_id": ep_id,
                    "step_idx": _safe_int(e.get("step_idx", 0)),
                    "obs": e.get("obs"),
                    "action": e.get("action"),
                }

                if cfg.include_reward:
                    row["reward"] = e.get("reward")

                if cfg.include_done:
                    row["done"] = bool(e.get("done") or e.get("truncated"))

                if cfg.include_next_obs:
                    if i + 1 < len(ep_events):
                        row["next_obs"] = ep_events[i + 1].get("obs")
                    else:
                        row["next_obs"] = None

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    return out_path


if __name__ == "__main__":
    # Example:
    # python memory/apprentice_dataset.py runs/<run_id> ep1 ep2 ep3
    import sys

    if len(sys.argv) < 2:
        print("Usage: python memory/apprentice_dataset.py runs/<run_id> [episode_id ...]")
        raise SystemExit(2)

    run_dir = sys.argv[1]
    eps = sys.argv[2:] if len(sys.argv) > 2 else None

    cfg = ApprenticeDatasetConfig(run_dir=run_dir)
    path = build_apprentice_dataset(cfg=cfg, episode_ids=eps)

    print(f"Apprentice dataset written to: {path}")
