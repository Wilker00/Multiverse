"""
memory/refiner.py

The "Cook" of the Multiverse. This module finds high-advantage success
snippets in raw event logs and compiles them into structured, human-readable
.txt "Lesson" files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RefinerConfig:
    """Configuration for the Text Refiner."""
    run_dir: str
    events_filename: str = "events.jsonl"
    lessons_dir: str = "lessons"
    advantage_threshold: float = 1.0  # Threshold for a sequence to be a "success snippet"
    gamma: float = 0.99
    baseline_by: str = "verse_name"

def _format_action_sequence(events: List[Dict[str, Any]]) -> str:
    """Creates a simplified, readable representation of an action sequence."""
    sequence_str = []
    for i, event in enumerate(events):
        action = event.get("action")
        # This can be expanded to be more descriptive based on the verse
        sequence_str.append(f"  {i+1}. DO_ACTION({json.dumps(action)})")
    return "\n".join(sequence_str)

def _write_lesson_file(lesson_data: Dict[str, Any], lessons_dir: str, run_id: str):
    """Writes a single .txt lesson file."""
    ep_id = lesson_data["source_episode_id"]
    verse_name = lesson_data["verse_name"]
    lesson_name = f"{verse_name}_{ep_id}".replace(":", "-")
    
    os.makedirs(lessons_dir, exist_ok=True)
    filepath = os.path.join(lessons_dir, f"{lesson_name}.txt")

    content = f"""TITLE: {lesson_data['title']}
CONTEXT: {verse_name}
SOURCE_RUN: {run_id}
REWARD: {lesson_data['reward']:.4f}
ADVANTAGE: {lesson_data['advantage']:.4f}
UTILITY_SCORE: 1.0

SEQUENCE:
{lesson_data['sequence']}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath

def _iter_events(path: str):
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

def refine_event_log(cfg: RefinerConfig) -> List[str]:
    """
    Scans an event log, finds high-advantage episodes, and writes them
    out as structured .txt lesson files.
    """
    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")

    ep_returns: Dict[str, float] = {}
    ep_baseline_key: Dict[str, str] = {}
    ep_verse: Dict[str, str] = {}
    rows = 0
    for e in _iter_events(events_path):
        rows += 1
        ep_id = str(e.get("episode_id"))
        ep_returns[ep_id] = float(ep_returns.get(ep_id, 0.0)) + float(e.get("reward", 0.0))
        if ep_id not in ep_baseline_key:
            ep_baseline_key[ep_id] = str(e.get(cfg.baseline_by))
            ep_verse[ep_id] = str(e.get("verse_name", "unknown_verse"))

    if rows <= 0 or not ep_returns:
        return []

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for ep_id, ret in ep_returns.items():
        key = ep_baseline_key.get(ep_id, "")
        sums[key] = float(sums.get(key, 0.0)) + float(ret)
        counts[key] = int(counts.get(key, 0)) + 1
    baselines: Dict[str, float] = {k: float(v / float(max(1, counts.get(k, 1)))) for k, v in sums.items()}

    ep_advantages: Dict[str, float] = {}
    selected_eps: set[str] = set()
    for ep_id, ret in ep_returns.items():
        key = ep_baseline_key.get(ep_id, "")
        adv = float(ret - float(baselines.get(key, 0.0)))
        ep_advantages[ep_id] = adv
        if adv >= float(cfg.advantage_threshold):
            selected_eps.add(ep_id)
    if not selected_eps:
        return []

    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in _iter_events(events_path):
        ep_id = str(e.get("episode_id"))
        if ep_id in selected_eps:
            by_ep.setdefault(ep_id, []).append(e)

    created_lessons = []
    run_id = os.path.basename(cfg.run_dir)

    for ep_id, ep_events in by_ep.items():
        ep_advantage = float(ep_advantages.get(ep_id, 0.0))
        ep_events.sort(key=lambda e: int(e.get("step_idx", 0)))
        ep_reward = float(ep_returns.get(ep_id, 0.0))
        verse_name = str(ep_verse.get(ep_id) or ep_events[0].get("verse_name", "unknown_verse"))

        lesson_data = {
            "title": f"Successful execution in {verse_name}",
            "verse_name": verse_name,
            "source_episode_id": ep_id,
            "reward": ep_reward,
            "advantage": ep_advantage,
            "sequence": _format_action_sequence(ep_events),
        }

        lesson_path = _write_lesson_file(lesson_data, cfg.lessons_dir, run_id)
        created_lessons.append(lesson_path)

    print(f"Refiner created {len(created_lessons)} new lessons from run '{run_id}'.")
    return created_lessons
