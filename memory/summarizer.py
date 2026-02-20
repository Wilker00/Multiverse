"""
memory/summarizer.py

Episode summarization utilities for u.ai.

Produces EpisodeSummary JSONL from raw StepEvent logs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.types import EpisodeSummary


@dataclass
class SummaryConfig:
    run_dir: str
    events_filename: str = "events.jsonl"
    output_filename: str = "summaries.jsonl"


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


def _load_events(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def summarize_episode_events(ep_events: List[Dict[str, Any]]) -> EpisodeSummary:
    ep_events.sort(key=lambda e: _safe_int(e.get("step_idx", 0)))
    first = ep_events[0]
    last = ep_events[-1]

    steps = len(ep_events)
    return_sum = sum(_safe_float(e.get("reward", 0.0)) for e in ep_events)
    t_start = _safe_int(first.get("t_ms", 0))
    t_end = _safe_int(last.get("t_ms", t_start))

    tags: List[str] = []
    tags.append(f"verse:{first.get('verse_name')}")
    tags.append(f"policy:{first.get('policy_id')}")

    reached_goal: Optional[bool] = None
    saw_key = False
    for e in ep_events:
        info = e.get("info") or {}
        if isinstance(info, dict) and "reached_goal" in info:
            saw_key = True
            if info.get("reached_goal") is True:
                reached_goal = True
                break
    if reached_goal is None and saw_key:
        reached_goal = False

    if reached_goal is True:
        tags.append("success:true")
    elif reached_goal is False:
        tags.append("success:false")

    return EpisodeSummary(
        schema_version="v1",
        run_id=str(first.get("run_id")),
        episode_id=str(first.get("episode_id")),
        agent_id=str(first.get("agent_id")),
        policy_id=str(first.get("policy_id")),
        policy_version=str(first.get("policy_version")),
        verse_name=str(first.get("verse_name")),
        verse_version=str(first.get("verse_version")),
        spec_hash=str(first.get("spec_hash")),
        steps=steps,
        return_sum=float(return_sum),
        start_ms=t_start,
        end_ms=t_end,
        tags=tags,
        notes=None,
    )


def summarize_run(cfg: SummaryConfig) -> str:
    if not os.path.isdir(cfg.run_dir):
        raise FileNotFoundError(f"run_dir not found: {cfg.run_dir}")

    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"events file not found: {events_path}")

    out_path = os.path.join(cfg.run_dir, cfg.output_filename)

    events = _load_events(events_path)
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ep = str(e.get("episode_id"))
        by_ep.setdefault(ep, []).append(e)

    with open(out_path, "w", encoding="utf-8") as out:
        for ep_id in sorted(by_ep.keys()):
            summary = summarize_episode_events(by_ep[ep_id])
            out.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")

    return out_path


if __name__ == "__main__":
    # Example:
    # python memory/summarizer.py runs/<run_id>
    import sys

    if len(sys.argv) < 2:
        print("Usage: python memory/summarizer.py runs/<run_id>")
        raise SystemExit(2)

    run_dir = sys.argv[1]
    out = summarize_run(SummaryConfig(run_dir=run_dir))
    print(f"Wrote summaries: {out}")
