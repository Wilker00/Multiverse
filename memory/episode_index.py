"""
memory/episode_index.py

Build an episode-level index from StepEvent JSONL logs.

Purpose:
- Convert raw step events into per-episode summaries
- Enable fast retrieval by tags/verse/policy/return/length
- Create a clean bridge to later: embeddings, vector search, knowledge graph

Output:
- episodes.jsonl (one episode summary per line)

This is MVP-grade: simple, deterministic, and readable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

def _get_strategic_signature(obs: Any, verse_name: str) -> Optional[Dict[str, int]]:
    try:
        from memory.semantic_bridge import _strategy_signature
        return _strategy_signature(obs, verse_name)
    except Exception:
        return None



@dataclass
class EpisodeIndexConfig:
    run_dir: str
    events_filename: str = "events.jsonl"
    output_filename: str = "episodes.jsonl"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def _load_events(events_path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _episode_summary(ep_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Sort by step index for stability
    ep_events.sort(key=lambda e: _safe_int(e.get("step_idx", 0)))

    first = ep_events[0]
    last = ep_events[-1]

    episode_id = str(first.get("episode_id"))
    run_id = str(first.get("run_id"))

    agent_id = str(first.get("agent_id"))
    policy_id = str(first.get("policy_id"))
    policy_version = str(first.get("policy_version"))

    verse_name = str(first.get("verse_name"))
    verse_version = str(first.get("verse_version"))
    spec_hash = str(first.get("spec_hash"))

    t_start = _safe_int(first.get("t_ms", 0))
    t_end = _safe_int(last.get("t_ms", t_start))

    steps = len(ep_events)
    return_sum = sum(_safe_float(e.get("reward", 0.0)) for e in ep_events)

    # Try to infer "success" if info contains reached_goal
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

    # Collect lightweight tags
    tags: List[str] = []
    # Example tag sources: verse_name, policy_id
    tags.append(f"verse:{verse_name}")
    tags.append(f"policy:{policy_id}")

    if reached_goal is True:
        tags.append("success:true")
    elif reached_goal is False:
        tags.append("success:false")

    summary: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "episode_id": episode_id,
        "agent_id": agent_id,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "verse_name": verse_name,
        "verse_version": verse_version,
        "spec_hash": spec_hash,
        "steps": steps,
        "return_sum": return_sum,
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "reached_goal": reached_goal,
        "tags": tags,
    }

    # Add Strategic Signature if available from last observation
    sig = _get_strategic_signature(last.get("obs"), verse_name)
    if sig:
        summary["strategic_signature"] = sig
        for k, v in sig.items():
            summary["tags"].append(f"strat:{k}:{v}")

    return summary



def build_episode_index(cfg: EpisodeIndexConfig) -> str:
    """
    Reads events.jsonl and writes episodes.jsonl in the same run directory.
    Returns output file path.
    """
    if not os.path.isdir(cfg.run_dir):
        raise FileNotFoundError(f"run_dir not found: {cfg.run_dir}")

    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"events file not found: {events_path}")

    out_path = os.path.join(cfg.run_dir, cfg.output_filename)

    events = _load_events(events_path)

    # Group by episode_id
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ep = str(e.get("episode_id"))
        by_ep.setdefault(ep, []).append(e)

    # Write summaries
    with open(out_path, "w", encoding="utf-8") as out:
        for ep_id in sorted(by_ep.keys()):
            summary = _episode_summary(by_ep[ep_id])
            out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return out_path


if __name__ == "__main__":
    # Example:
    # python memory/episode_index.py runs/<run_id>
    import sys

    if len(sys.argv) < 2:
        print("Usage: python memory/episode_index.py runs/<run_id>")
        raise SystemExit(2)

    run_dir = sys.argv[1]
    out_path = build_episode_index(EpisodeIndexConfig(run_dir=run_dir))
    print(f"Wrote episode index: {out_path}")
