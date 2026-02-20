"""
orchestrator/scheduler.py

Contains logic for both static and curiosity-driven run scheduling.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from core.types import VerseSpec, AgentSpec
from orchestrator.trainer import Trainer
from policies.skill_path import SkillPath
from verses.registry import list_verses, register_builtin as register_builtin_verses

logger = logging.getLogger(__name__)


@dataclass
class ScheduledRun:
    verse_spec: VerseSpec
    agent_spec: AgentSpec
    episodes: int
    max_steps: int
    seed: Optional[int] = None

def get_existing_skill_tags(skill_library_dir: str) -> Counter:
    """Scans the skill library and returns a count of tags."""
    skill_tags = Counter()
    if not os.path.isdir(skill_library_dir):
        logger.info("Skill library not found at '%s'; assuming no skills exist.", skill_library_dir)
        return skill_tags

    for skill_file in os.listdir(skill_library_dir):
        if skill_file.endswith(".json"):
            try:
                skill_path = os.path.join(skill_library_dir, skill_file)
                skill = SkillPath.load(skill_path)
                skill_tags.update(skill.tags)
            except Exception as e:
                logger.warning("Could not load skill '%s': %s", skill_file, e)
    return skill_tags

def get_verse_registry_tags() -> Dict[str, List[str]]:
    """Returns a mapping from verse names to their advertised tags."""
    verse_factories = list_verses()
    if not verse_factories:
        register_builtin_verses()
        verse_factories = list_verses()
    verse_tags = {}
    for verse_name, factory in verse_factories.items():
        try:
            # Access the tags directly from the factory property
            if hasattr(factory, 'tags'):
                verse_tags[verse_name] = factory.tags
            else:
                verse_tags[verse_name] = []
        except Exception as e:
            logger.warning("Could not get tags for verse '%s': %s", verse_name, e)
            verse_tags[verse_name] = []
    return verse_tags

def _recent_run_dirs(runs_root: str, lookback_runs: int) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    candidates: List[tuple[float, str]] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        events_path = os.path.join(run_dir, "events.jsonl")
        if not os.path.isdir(run_dir) or not os.path.isfile(events_path):
            continue
        try:
            mtime = os.path.getmtime(events_path)
        except Exception:
            continue
        candidates.append((float(mtime), run_dir))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in candidates[: max(1, int(lookback_runs))]]


def _count_cliff_signals(runs_root: str, lookback_runs: int, cliff_reward_cutoff: float = -50.0) -> int:
    total = 0
    for run_dir in _recent_run_dirs(runs_root, lookback_runs):
        events_path = os.path.join(run_dir, "events.jsonl")
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        row = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    info = row.get("info")
                    fell = isinstance(info, dict) and info.get("fell_cliff") is True
                    reward = row.get("reward")
                    try:
                        penalized = float(reward) <= float(cliff_reward_cutoff)
                    except Exception:
                        penalized = False
                    if fell or penalized:
                        total += 1
        except Exception:
            continue
    return total


def select_next_verse(
    skill_library_dir: str,
    *,
    runs_root: Optional[str] = None,
    prefer_cliff_on_penalty: bool = True,
    cliff_penalty_threshold: int = 10,
    lookback_runs: int = 5,
) -> Optional[str]:
    """
    Selects the next verse to run based on skill gaps.
    """
    skill_tags = get_existing_skill_tags(skill_library_dir)
    verse_tags = get_verse_registry_tags()

    if not verse_tags:
        logger.error("No verses found in the registry.")
        return None

    all_possible_tags = set(tag for tags in verse_tags.values() for tag in tags)
    if not all_possible_tags:
        logger.warning("No verses with tags found.")
        return None

    if (
        prefer_cliff_on_penalty
        and runs_root
        and "cliff_world" in verse_tags
        and int(cliff_penalty_threshold) > 0
    ):
        cliff_signals = _count_cliff_signals(
            str(runs_root),
            lookback_runs=max(1, int(lookback_runs)),
            cliff_reward_cutoff=-50.0,
        )
        if cliff_signals >= int(cliff_penalty_threshold):
            logger.info(
                "Recent high-penalty cliff signals detected (%s >= %s); prioritizing cliff_world.",
                cliff_signals,
                cliff_penalty_threshold,
            )
            return "cliff_world"

    min_count = float('inf')
    gap_tag = None
    # Sort for deterministic selection
    for tag in sorted(list(all_possible_tags)):
        count = skill_tags.get(tag, 0)
        if count < min_count:
            min_count = count
            gap_tag = tag

    if gap_tag is None:
        logger.info("No skill gap identified. All tagged skills are equally represented.")
        return None

    logger.info("Identified skill gap '%s' (count=%s).", gap_tag, min_count)

    # Find a verse that can teach this skill (in a deterministic order)
    for verse_name in sorted(verse_tags.keys()):
        if gap_tag in verse_tags[verse_name]:
            logger.info("Selected verse '%s' to address gap '%s'.", verse_name, gap_tag)
            return verse_name
            
    logger.warning("No verse found that teaches skill '%s'.", gap_tag)
    return None

def run_schedule(trainer: Trainer, runs: Iterable[ScheduledRun]) -> List[Dict[str, Any]]:
    # ... (existing implementation)
    results: List[Dict[str, Any]] = []
    for item in runs:
        res = trainer.run(
            verse_spec=item.verse_spec,
            agent_spec=item.agent_spec,
            episodes=item.episodes,
            max_steps=item.max_steps,
            seed=item.seed,
        )
        results.append(res)
    return results

def load_schedule(path: str) -> List[ScheduledRun]:
    # ... (existing implementation)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    runs: List[ScheduledRun] = []
    for item in payload:
        runs.append(
            ScheduledRun(
                verse_spec=VerseSpec(**item["verse_spec"]),
                agent_spec=AgentSpec(**item["agent_spec"]),
                episodes=int(item.get("episodes", 50)),
                max_steps=int(item.get("max_steps", 40)),
                seed=item.get("seed"),
            )
        )
    return runs

def main():
    # ... (existing main for static schedules)
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=str, required=True, help="path to schedule.json")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    trainer = Trainer(run_root=args.runs_root, schema_version="v1", auto_register_builtin=True)
    runs = load_schedule(args.schedule)
    run_schedule(trainer, runs)

if __name__ == "__main__":
    main()
