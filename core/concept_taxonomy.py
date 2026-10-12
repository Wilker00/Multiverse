"""
core/concept_taxonomy.py

Concept and skill labels for the shared intelligence layer.

This module does not train policies. Reinforcement learning still happens in
the trainer and in each agent. These labels sit on top of ``core.taxonomy`` so
central memory can be retrieved by skill family (navigation, risk, planning)
across verses. A skill id is a stable string, not a learned embedding.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.taxonomy import parent_domains, tags_for_verse


# Fixed concept set used by retrieval filters and the live map.
CONCEPTS: tuple[str, ...] = (
    "navigation",
    "planning",
    "risk",
    "spatial_reasoning",
    "resource",
    "strategy",
    "memory",
    "robotics",
    "safety",
)

# Substrings matched against verse tags and parent domains.
# ``memory_type:`` / ``memory_family:`` / ``universe:`` prefixes are ignored so
# every verse does not inherit a fake "memory" concept from the cognitive tag.
_CONCEPT_NEEDLES: Dict[str, tuple[str, ...]] = {
    "robotics": ("robot", "embodied", "isaac"),
    "navigation": ("navigation", "obstacle", "maze"),
    "planning": ("planning", "scheduling", "sequential_decision"),
    "risk": ("risk", "survival", "hazard", "cliff"),
    "spatial_reasoning": ("spatial", "2d", "3d", "terrain"),
    "resource": ("resource", "economics", "trading", "harvest", "warehouse", "inventory"),
    "strategy": ("strategy", "board", "card", "chess", "transferable_logic"),
    "memory": ("memory_diagnostics", "sequential_memory", "memory_rule", "memory_vault"),
    "safety": ("safety", "shield"),
}


def _norm(value: str) -> str:
    return str(value).strip().lower()


def _verse_label_blob(verse_name: str) -> List[str]:
    labels: List[str] = []
    seen = set()
    for raw in list(tags_for_verse(verse_name)) + sorted(parent_domains(verse_name)):
        text = _norm(raw)
        if not text or text in seen:
            continue
        if text.startswith("memory_type:") or text.startswith("memory_family:") or text.startswith("universe:"):
            continue
        seen.add(text)
        labels.append(text)
    return labels


def concept_tags_for_verse(verse_name: str) -> List[str]:
    """Map a verse onto the fixed concept set. Unknown verses get ``task``."""
    blob = _verse_label_blob(verse_name)
    out: List[str] = []
    for concept in CONCEPTS:
        needles = _CONCEPT_NEEDLES.get(concept, ())
        if any(any(needle in label for needle in needles) for label in blob):
            out.append(concept)
    if not out:
        return ["task"]
    return out


def skill_id_for(verse: str, concept: str, successful: bool) -> str:
    """Stable skill identifier shared by every agent type."""
    outcome = "success" if bool(successful) else "attempt"
    verse_key = _norm(verse) or "unknown"
    concept_key = _norm(concept) or "task"
    return f"skill:{concept_key}:{verse_key}:{outcome}"


def skill_ids_for_verse(
    verse_name: str,
    *,
    successful: bool,
    concepts: Optional[Sequence[str]] = None,
) -> List[str]:
    chosen = list(concepts) if concepts else concept_tags_for_verse(verse_name)
    return [skill_id_for(verse_name, concept, successful) for concept in chosen]


def memory_priority(
    *,
    reward: float,
    verified_transfer_score: Optional[float],
    memory_tier: str,
) -> float:
    """
    Priority in ``[0, 1]`` from reward sign, verified transfer score, and tier.

    Long-term memories and positive rewards rank above short-term failures.
    This value is a retrieval filter, not a policy gradient weight.
    """
    try:
        reward_value = float(reward)
    except (TypeError, ValueError):
        reward_value = 0.0
    if reward_value > 0.0:
        reward_term = 1.0
    elif reward_value < 0.0:
        reward_term = 0.0
    else:
        reward_term = 0.5
    if verified_transfer_score is None:
        verified_term = 0.0
    else:
        try:
            verified_term = max(0.0, min(1.0, float(verified_transfer_score)))
        except (TypeError, ValueError):
            verified_term = 0.0
    tier_term = 1.0 if _norm(memory_tier) == "ltm" else 0.35
    score = (0.5 * reward_term) + (0.3 * verified_term) + (0.2 * tier_term)
    return max(0.0, min(1.0, float(score)))


def _extra_concepts(info: Any) -> List[str]:
    if not isinstance(info, dict):
        return []
    raw = info.get("concept_tags")
    if isinstance(raw, str):
        raw_items: Iterable[Any] = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        raw_items = list(raw)
    else:
        return []
    allowed = set(CONCEPTS) | {"task"}
    out: List[str] = []
    for item in raw_items:
        text = _norm(item)
        if text in allowed and text not in out:
            out.append(text)
    return out


def annotate_memory_concepts(
    *,
    verse_name: str,
    reward: float,
    verified_transfer_score: Optional[float],
    memory_tier: str,
    info: Any = None,
) -> Dict[str, Any]:
    """Fields stored on a central-memory row. Missing on old rows is allowed."""
    concepts = concept_tags_for_verse(verse_name)
    for extra in _extra_concepts(info):
        if extra not in concepts:
            concepts.append(extra)
    successful = False
    try:
        successful = float(reward) > 0.0
    except (TypeError, ValueError):
        successful = False
    return {
        "concept_tags": concepts,
        "skill_ids": skill_ids_for_verse(verse_name, successful=successful, concepts=concepts),
        "memory_priority": memory_priority(
            reward=reward,
            verified_transfer_score=verified_transfer_score,
            memory_tier=memory_tier,
        ),
    }
