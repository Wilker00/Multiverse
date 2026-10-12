"""
memory/skill_catalog.py

Groups successful central-memory rows by shared skill id.

Agents can list skills before training. This catalog does not replace
Q-learning, PPO, or successor-feature updates.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from memory.central_repository_support import CentralMemoryConfig


def _memories_path(root_dir: str) -> str:
    return os.path.join(str(root_dir), "memories.jsonl")


def list_skills(cfg: CentralMemoryConfig, *, limit: int = 50) -> List[Dict[str, Any]]:
    path = _memories_path(cfg.root_dir)
    grouped: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            try:
                reward = float(row.get("reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                reward = 0.0
            skill_ids = row.get("skill_ids")
            if not isinstance(skill_ids, list) or not skill_ids:
                try:
                    from core.concept_taxonomy import skill_ids_for_verse

                    skill_ids = skill_ids_for_verse(str(row.get("verse_name") or ""), successful=reward > 0.0)
                except Exception:
                    skill_ids = []
            concepts = row.get("concept_tags") if isinstance(row.get("concept_tags"), list) else []
            for skill_id in skill_ids:
                key = str(skill_id).strip()
                if not key:
                    continue
                bucket = grouped.get(key)
                if bucket is None:
                    bucket = {
                        "skill_id": key,
                        "verse_name": str(row.get("verse_name") or ""),
                        "concept_tags": list(concepts),
                        "count": 0,
                        "success_count": 0,
                        "reward_sum": 0.0,
                    }
                    grouped[key] = bucket
                bucket["count"] = int(bucket["count"]) + 1
                bucket["reward_sum"] = float(bucket["reward_sum"]) + reward
                if reward > 0.0:
                    bucket["success_count"] = int(bucket["success_count"]) + 1
    rows = list(grouped.values())
    rows.sort(key=lambda item: (int(item["success_count"]), int(item["count"])), reverse=True)
    cap = max(1, int(limit))
    return rows[:cap]
