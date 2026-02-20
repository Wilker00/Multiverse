"""
orchestrator/scenario_matcher.py

Scenario matching engine:
- retrieves closest memories from the shared repository
- recommends an action using weighted voting over matches
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from core.types import JSONValue
from memory.central_repository import CentralMemoryConfig, ScenarioMatch, find_similar
from memory.confidence_auditor import (
    ConfidenceAuditConfig,
    get_bridge_weight,
    load_state as load_audit_state,
)
from memory.decay_manager import apply_temporal_decay
from memory.embeddings import cosine_similarity, obs_to_universal_vector, obs_to_vector
from memory.knowledge_graph import KnowledgeGraphConfig, load_graph, verse_relatedness
from memory.semantic_bridge import translate_transition
from memory.task_taxonomy import primary_task_tag, tags_for_verse


@dataclass
class ScenarioRequest:
    obs: JSONValue
    verse_name: Optional[str] = None
    top_k: int = 5
    min_score: float = 0.0
    exclude_run_ids: Optional[List[str]] = None
    semantic_fallback_threshold: float = 0.35
    enable_tag_fallback: bool = True
    enable_semantic_bridge: bool = True
    cross_verse_pool: int = 250
    enable_knowledge_graph: bool = True
    min_graph_relatedness: float = 0.05
    graph_bonus_scale: float = 0.15
    enable_confidence_auditor: bool = True
    decay_lambda: float = 0.0
    learned_bridge_enabled: bool = False
    learned_bridge_model_path: Optional[str] = None
    learned_bridge_score_weight: float = 0.35


@dataclass
class ScenarioAdvice:
    action: JSONValue
    confidence: float
    matches: List[ScenarioMatch]
    weights: Dict[str, float]
    strategy: str = "direct"
    direct_candidates: int = 0
    semantic_candidates: int = 0
    bridge_source_verse: Optional[str] = None


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _action_key(action: JSONValue) -> str:
    try:
        return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(action)


def _as_tag_set(value: object) -> Set[str]:
    if isinstance(value, list):
        out: Set[str] = set()
        for t in value:
            s = str(t).strip().lower()
            if s:
                out.add(s)
        return out
    return set()


def _iter_memory_rows(cfg: CentralMemoryConfig):
    path = os.path.join(cfg.root_dir, cfg.memories_filename)
    if not os.path.isfile(path):
        return
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


def _find_semantic_fallback_matches(
    *,
    request: ScenarioRequest,
    cfg: CentralMemoryConfig,
    exclude_run_ids: Set[str],
) -> List[ScenarioMatch]:
    target_verse = str(request.verse_name or "").strip()
    if not target_verse:
        return []

    query_vec = obs_to_vector(request.obs)
    query_vec_u = obs_to_universal_vector(request.obs)
    target_tags = set(tags_for_verse(target_verse))
    kg = None
    if request.enable_knowledge_graph:
        kg = load_graph(KnowledgeGraphConfig(root_dir=cfg.root_dir))

    audit_cfg = ConfidenceAuditConfig(root_dir=cfg.root_dir)
    audit_state = load_audit_state(audit_cfg) if request.enable_confidence_auditor else None
    task_tag = primary_task_tag(target_verse)
    matches: List[ScenarioMatch] = []

    for row in _iter_memory_rows(cfg):
        run_id = str(row.get("run_id", ""))
        if run_id in exclude_run_ids:
            continue

        source_verse = str(row.get("verse_name", "")).strip()
        if not source_verse or source_verse == target_verse:
            continue

        row_tags = _as_tag_set(row.get("tags"))
        if not row_tags:
            row_tags = set(tags_for_verse(source_verse))
        tag_related = bool(target_tags & row_tags) if target_tags else False
        graph_relatedness = 0.0
        if request.enable_knowledge_graph and kg is not None:
            graph_relatedness = float(verse_relatedness(source_verse, target_verse, kg))
        is_related = (
            (request.enable_tag_fallback and tag_related)
            or (
                request.enable_knowledge_graph
                and graph_relatedness >= float(request.min_graph_relatedness)
            )
        )
        if not is_related:
            continue

        translated = translate_transition(
            obs=row.get("obs"),
            action=row.get("action"),
            source_verse_name=source_verse,
            target_verse_name=target_verse,
            learned_bridge_enabled=bool(request.learned_bridge_enabled),
            learned_bridge_model_path=request.learned_bridge_model_path,
        )
        if translated is None:
            continue

        try:
            candidate_vec = obs_to_vector(translated["obs"])
        except Exception:
            continue
        if len(candidate_vec) == len(query_vec):
            score = cosine_similarity(query_vec, candidate_vec)
        else:
            try:
                candidate_vec_u = obs_to_universal_vector(translated["obs"])
            except Exception:
                continue
            if len(candidate_vec_u) != len(query_vec_u):
                continue
            score = cosine_similarity(query_vec_u, candidate_vec_u)
        if bool(request.learned_bridge_enabled):
            learned_conf = _safe_float(translated.get("learned_bridge_confidence"), -1.0)
            if 0.0 <= learned_conf <= 1.0:
                w = max(0.0, min(1.0, float(request.learned_bridge_score_weight)))
                learned_score = (2.0 * learned_conf) - 1.0
                score = ((1.0 - w) * float(score)) + (w * float(learned_score))
        if score < float(request.min_score):
            continue

        bridge_weight = 1.0
        if request.enable_confidence_auditor and audit_state is not None:
            bridge_weight = get_bridge_weight(
                source_verse=source_verse,
                target_verse=target_verse,
                task_tag=task_tag,
                cfg=audit_cfg,
                state=audit_state,
            )

        structure_bonus = 1.0
        if request.enable_knowledge_graph:
            structure_bonus += float(request.graph_bonus_scale) * float(graph_relatedness)
        score *= float(bridge_weight) * float(structure_bonus)
        t_ms = _safe_int(row.get("t_ms", 0))
        score, recency_weight = apply_temporal_decay(
            score=float(score),
            t_ms=t_ms,
            decay_lambda=float(request.decay_lambda),
            current_time_ms=None,
        )

        if score < float(request.min_score):
            continue

        matches.append(
            ScenarioMatch(
                score=float(score),
                run_id=run_id,
                episode_id=str(row.get("episode_id", "")),
                step_idx=_safe_int(row.get("step_idx", 0)),
                t_ms=t_ms,
                verse_name=source_verse,
                action=translated["action"],
                reward=_safe_float(row.get("reward", 0.0)),
                obs=translated["obs"],
                recency_weight=float(recency_weight),
            )
        )

    matches.sort(key=lambda m: m.score, reverse=True)
    pool_size = max(int(request.top_k), int(request.cross_verse_pool), 1)
    return matches[:pool_size]


def _best_source_for_action(matches: List[ScenarioMatch], best_action_key: str) -> Optional[str]:
    source_scores: Dict[str, float] = {}
    for m in matches:
        if _action_key(m.action) != best_action_key:
            continue
        src = str(m.verse_name or "").strip()
        if not src:
            continue
        source_scores[src] = source_scores.get(src, 0.0) + max(0.0, float(m.score))
    if not source_scores:
        return None
    return max(source_scores.items(), key=lambda kv: kv[1])[0]


def recommend_action(
    *,
    request: ScenarioRequest,
    cfg: CentralMemoryConfig,
) -> Optional[ScenarioAdvice]:
    """
    Recommend action for a scenario using similarity-weighted memory votes.
    """
    exclude_run_ids = set(str(x) for x in request.exclude_run_ids or [])

    direct_matches = find_similar(
        obs=request.obs,
        cfg=cfg,
        top_k=request.top_k,
        verse_name=request.verse_name,
        min_score=request.min_score,
        exclude_run_ids=exclude_run_ids,
        decay_lambda=float(request.decay_lambda),
        current_time_ms=None,
    )
    direct_best = direct_matches[0].score if direct_matches else -1.0

    semantic_matches: List[ScenarioMatch] = []
    if (
        request.enable_semantic_bridge
        and (request.enable_tag_fallback or request.enable_knowledge_graph)
        and request.verse_name
        and (not direct_matches or direct_best < float(request.semantic_fallback_threshold))
    ):
        semantic_matches = _find_semantic_fallback_matches(
            request=request,
            cfg=cfg,
            exclude_run_ids=exclude_run_ids,
        )

    strategy = "direct"
    matches: List[ScenarioMatch]
    if not direct_matches and semantic_matches:
        strategy = "semantic_bridge"
        matches = semantic_matches[: max(1, int(request.top_k))]
    elif direct_matches and semantic_matches and direct_best < float(request.semantic_fallback_threshold):
        strategy = "hybrid_low_confidence"
        combined = list(direct_matches) + list(semantic_matches)
        combined.sort(key=lambda m: m.score, reverse=True)
        matches = combined[: max(1, int(request.top_k))]
    else:
        matches = direct_matches[: max(1, int(request.top_k))]

    if not matches:
        return None

    scores: Dict[str, float] = {}
    values: Dict[str, JSONValue] = {}
    total = 0.0

    for m in matches:
        w = max(0.0, float(m.score))
        if w <= 0.0:
            continue
        k = _action_key(m.action)
        scores[k] = scores.get(k, 0.0) + w
        values[k] = m.action
        total += w

    # Fallback if all similarities are <= 0.
    if not scores:
        fallback = matches[0].action
        return ScenarioAdvice(
            action=fallback,
            confidence=0.0,
            matches=matches,
            weights={_action_key(fallback): 1.0},
            strategy=strategy,
            direct_candidates=len(direct_matches),
            semantic_candidates=len(semantic_matches),
            bridge_source_verse=None if strategy == "direct" else str(matches[0].verse_name or ""),
        )

    best_key = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = float(scores[best_key])
    confidence = (best_score / total) if total > 0 else 0.0

    norm = {}
    for k, v in scores.items():
        norm[k] = float(v / total) if total > 0 else 0.0

    best_source = _best_source_for_action(matches, best_key)
    bridge_source = None
    if strategy != "direct" and best_source:
        if request.verse_name is None or str(best_source) != str(request.verse_name):
            bridge_source = best_source

    return ScenarioAdvice(
        action=values[best_key],
        confidence=float(confidence),
        matches=matches,
        weights=norm,
        strategy=strategy,
        direct_candidates=len(direct_matches),
        semantic_candidates=len(semantic_matches),
        bridge_source_verse=bridge_source,
    )
