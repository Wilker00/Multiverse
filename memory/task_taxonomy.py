"""
memory/task_taxonomy.py

Lightweight task taxonomy used to label verse memories.
"""

from __future__ import annotations

from typing import List

from core.taxonomy import (
    all_universes as _all_universes,
    memory_family_for_verse as _memory_family_for_verse,
    memory_type_for_verse as _memory_type_for_verse,
    same_universe as _same_universe,
    tags_for_verse as _core_tags_for_verse,
    universe_for_verse as _universe_for_verse,
    universe_members as _universe_members,
    universe_relation as _universe_relation,
)

def tags_for_verse(verse_name: str) -> List[str]:
    return list(_core_tags_for_verse(verse_name))


def memory_type_for_verse(verse_name: str) -> str:
    return str(_memory_type_for_verse(verse_name))


def memory_family_for_verse(verse_name: str) -> str:
    return str(_memory_family_for_verse(verse_name))


def universe_for_verse(verse_name: str) -> str:
    return str(_universe_for_verse(verse_name))


def same_universe(source_verse: str, target_verse: str) -> bool:
    return bool(_same_universe(source_verse, target_verse))


def universe_relation(source_verse: str, target_verse: str) -> str:
    return str(_universe_relation(source_verse, target_verse))


def universe_members(universe_name: str) -> List[str]:
    return list(_universe_members(universe_name))


def all_universes() -> List[str]:
    return list(_all_universes())


def primary_task_tag(verse_name: str) -> str:
    """
    Returns the first high-level task tag for a verse.
    """
    tags = tags_for_verse(verse_name)
    return str(tags[0]) if tags else "unknown"
