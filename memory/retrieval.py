"""
memory/retrieval.py

Episode retrieval utilities for u.ai.

Reads:
- episodes.jsonl (episode-level summaries)
Optionally uses:
- events.jsonl (raw step events) when you want to fetch full trajectories

This is the MVP retrieval layer:
- filter episodes by verse, policy, tags, success, return, steps
- fetch episode summaries
- fetch full episode step events when needed

Later upgrades:
- embeddings + vector search
- semantic queries
- skill indexing
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class RetrievalConfig:
    run_dir: str
    episodes_filename: str = "episodes.jsonl"
    events_filename: str = "events.jsonl"


@dataclass
class EpisodeFilter:
    verse_name: Optional[str] = None
    policy_id: Optional[str] = None

    tag_includes: Optional[List[str]] = None
    tag_excludes: Optional[List[str]] = None

    reached_goal: Optional[bool] = None

    min_return: Optional[float] = None
    max_return: Optional[float] = None

    min_steps: Optional[int] = None
    max_steps: Optional[int] = None
    
    # Cross-verse strategic matching
    strategic_match: Optional[Dict[str, int]] = None

    limit: Optional[int] = None


class RetrievalClient:
    """
    Small OO adapter around module-level retrieval functions.
    Keeps compatibility with rollout code that expects retriever.filter_episodes(...).
    """

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg

    def filter_episodes(self, flt: EpisodeFilter) -> List[Dict[str, Any]]:
        return filter_episodes(self.cfg, flt)

    def get_episode_summaries(self, episode_ids: List[str]) -> List[Dict[str, Any]]:
        return get_episode_summaries(self.cfg, episode_ids)

    def load_episode_events(self, episode_id: str) -> List[Dict[str, Any]]:
        return load_episode_events(self.cfg, episode_id)

    def load_many_episode_events(self, episode_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        return load_many_episode_events(self.cfg, episode_ids)



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


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _matches_tags(tags: List[str], includes: Optional[List[str]], excludes: Optional[List[str]]) -> bool:
    if includes:
        for t in includes:
            if t not in tags:
                return False
    if excludes:
        for t in excludes:
            if t in tags:
                return False
    return True


def filter_episodes(cfg: RetrievalConfig, flt: EpisodeFilter) -> List[Dict[str, Any]]:
    """
    Returns episode summary dicts matching the filter.
    """
    episodes_path = os.path.join(cfg.run_dir, cfg.episodes_filename)
    if not os.path.isfile(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    rows = _load_jsonl(episodes_path)

    out: List[Dict[str, Any]] = []

    for row in rows:
        if flt.verse_name and str(row.get("verse_name")) != flt.verse_name:
            continue
        if flt.policy_id and str(row.get("policy_id")) != flt.policy_id:
            continue

        tags = row.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t) for t in tags]

        if not _matches_tags(tags, flt.tag_includes, flt.tag_excludes):
            continue

        if flt.reached_goal is not None:
            rg = row.get("reached_goal")
            if rg is None:
                continue
            if bool(rg) != bool(flt.reached_goal):
                continue

        ret = _safe_float(row.get("return_sum", 0.0))
        steps = _safe_int(row.get("steps", 0))

        if flt.min_return is not None and ret < float(flt.min_return):
            continue
        if flt.max_return is not None and ret > float(flt.max_return):
            continue
        if flt.min_steps is not None and steps < int(flt.min_steps):
            continue
        if flt.max_steps is not None and steps > int(flt.max_steps):
            continue

        if flt.strategic_match:
            row_sig = row.get("strategic_signature")
            if not isinstance(row_sig, dict):
                continue
            # Simple Manhattan distance check or just exact key match for MVP
            match = True
            for k, expected_v in flt.strategic_match.items():
                actual_v = _safe_int(row_sig.get(k, 0))
                # For MVP: treat as match if within +/- 1
                if abs(actual_v - expected_v) > 1:
                    match = False
                    break
            if not match:
                continue

        out.append(row)


        if flt.limit is not None and len(out) >= int(flt.limit):
            break

    return out


def get_episode_summaries(cfg: RetrievalConfig, episode_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch summaries by episode_id.
    """
    episodes_path = os.path.join(cfg.run_dir, cfg.episodes_filename)
    if not os.path.isfile(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    want = set(episode_ids)
    rows = _load_jsonl(episodes_path)
    return [r for r in rows if str(r.get("episode_id")) in want]


def load_episode_events(cfg: RetrievalConfig, episode_id: str) -> List[Dict[str, Any]]:
    """
    Load raw StepEvents for a given episode_id from events.jsonl.
    Returns list of event dicts sorted by step_idx.
    """
    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")

    events: List[Dict[str, Any]] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if str(e.get("episode_id")) == episode_id:
                events.append(e)

    events.sort(key=lambda x: _safe_int(x.get("step_idx", 0)))
    return events


def load_many_episode_events(cfg: RetrievalConfig, episode_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load raw StepEvents for multiple episode_ids.
    Returns mapping: episode_id -> list of event dicts
    """
    wanted = set(episode_ids)
    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")

    out: Dict[str, List[Dict[str, Any]]] = {ep: [] for ep in wanted}

    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            ep = str(e.get("episode_id"))
            if ep in wanted:
                out[ep].append(e)

    for ep in out:
        out[ep].sort(key=lambda x: _safe_int(x.get("step_idx", 0)))

    return out


if __name__ == "__main__":
    # Example:
    # python memory/retrieval.py runs/<run_id> --success
    import sys

    if len(sys.argv) < 2:
        print("Usage: python memory/retrieval.py runs/<run_id>")
        raise SystemExit(2)

    run_dir = sys.argv[1]
    cfg = RetrievalConfig(run_dir=run_dir)

    # Default example filter: successes first, limit 5
    flt = EpisodeFilter(
        tag_includes=None,
        reached_goal=True,
        limit=5,
    )

    eps = filter_episodes(cfg, flt)
    print(f"Found {len(eps)} episode(s)")
    for e in eps:
        print(
            f"{e['episode_id']}  return={_safe_float(e.get('return_sum')):.3f}  "
            f"steps={_safe_int(e.get('steps'))}  tags={e.get('tags')}"
        )
