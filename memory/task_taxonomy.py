"""
memory/task_taxonomy.py

Lightweight task taxonomy used to label verse memories.
"""

from __future__ import annotations

from typing import List

from core.taxonomy import (
    memory_family_for_verse as _memory_family_for_verse,
    memory_type_for_verse as _memory_type_for_verse,
    tags_for_verse as _core_tags_for_verse,
)

def tags_for_verse(verse_name: str) -> List[str]:
    return list(_core_tags_for_verse(verse_name))


def memory_type_for_verse(verse_name: str) -> str:
    return str(_memory_type_for_verse(verse_name))


def memory_family_for_verse(verse_name: str) -> str:
    return str(_memory_family_for_verse(verse_name))


def primary_task_tag(verse_name: str) -> str:
    """
    Returns the first high-level task tag for a verse.
    """
    tags = tags_for_verse(verse_name)
    return str(tags[0]) if tags else "unknown"
