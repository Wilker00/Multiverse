"""
memory/knowledge_graph.py

Lightweight relational task graph for hierarchy-aware scenario matching.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from memory.task_taxonomy import tags_for_verse


@dataclass
class KnowledgeGraphConfig:
    root_dir: str = "central_memory"
    filename: str = "knowledge_graph.json"


def _graph_path(cfg: KnowledgeGraphConfig) -> str:
    return os.path.join(cfg.root_dir, cfg.filename)


def _default_graph() -> Dict[str, Any]:
    # child -> parent hierarchical edges.
    return {
        "version": "v1",
        "nodes": [
            "task",
            "environment",
            "navigation_task",
            "complex_navigation_task",
            "interaction_task",
            "risk_sensitive_task",
            "dynamic_hazards_task",
            "goal_conditioned_task",
            "continuous_goal_task",
            "discrete_grid_task",
            "dynamic_target_task",
            "parking_task",
            "maze_task",
            "partial_observable_task",
            "resource_management_task",
            "1d_environment",
            "2d_environment",
            "line_world",
            "grid_world",
            "cliff_world",
            "labyrinth_world",
            "park_world",
            "pursuit_world",
        ],
        "edges": [
            {"child": "navigation_task", "parent": "task"},
            {"child": "complex_navigation_task", "parent": "navigation_task"},
            {"child": "interaction_task", "parent": "task"},
            {"child": "risk_sensitive_task", "parent": "task"},
            {"child": "dynamic_hazards_task", "parent": "risk_sensitive_task"},
            {"child": "goal_conditioned_task", "parent": "task"},
            {"child": "continuous_goal_task", "parent": "goal_conditioned_task"},
            {"child": "discrete_grid_task", "parent": "goal_conditioned_task"},
            {"child": "dynamic_target_task", "parent": "goal_conditioned_task"},
            {"child": "parking_task", "parent": "interaction_task"},
            {"child": "maze_task", "parent": "navigation_task"},
            {"child": "partial_observable_task", "parent": "task"},
            {"child": "resource_management_task", "parent": "task"},
            {"child": "1d_environment", "parent": "environment"},
            {"child": "2d_environment", "parent": "environment"},
            {"child": "line_world", "parent": "navigation_task"},
            {"child": "line_world", "parent": "1d_environment"},
            {"child": "line_world", "parent": "continuous_goal_task"},
            {"child": "grid_world", "parent": "navigation_task"},
            {"child": "grid_world", "parent": "2d_environment"},
            {"child": "grid_world", "parent": "discrete_grid_task"},
            {"child": "cliff_world", "parent": "navigation_task"},
            {"child": "cliff_world", "parent": "risk_sensitive_task"},
            {"child": "cliff_world", "parent": "2d_environment"},
            {"child": "cliff_world", "parent": "discrete_grid_task"},
            {"child": "labyrinth_world", "parent": "navigation_task"},
            {"child": "labyrinth_world", "parent": "complex_navigation_task"},
            {"child": "labyrinth_world", "parent": "risk_sensitive_task"},
            {"child": "labyrinth_world", "parent": "dynamic_hazards_task"},
            {"child": "labyrinth_world", "parent": "maze_task"},
            {"child": "labyrinth_world", "parent": "partial_observable_task"},
            {"child": "labyrinth_world", "parent": "resource_management_task"},
            {"child": "labyrinth_world", "parent": "2d_environment"},
            {"child": "labyrinth_world", "parent": "discrete_grid_task"},
            {"child": "park_world", "parent": "navigation_task"},
            {"child": "park_world", "parent": "2d_environment"},
            {"child": "park_world", "parent": "parking_task"},
            {"child": "pursuit_world", "parent": "navigation_task"},
            {"child": "pursuit_world", "parent": "1d_environment"},
            {"child": "pursuit_world", "parent": "dynamic_target_task"},
        ],
        # Optional extra links between tags and graph concepts.
        "tag_aliases": {
            "navigation": ["navigation_task"],
            "complex_navigation": ["complex_navigation_task"],
            "interaction": ["interaction_task"],
            "continuous_goal": ["continuous_goal_task"],
            "discrete_grid": ["discrete_grid_task"],
            "dynamic_target": ["dynamic_target_task"],
            "parking": ["parking_task"],
            "risk_sensitive": ["risk_sensitive_task"],
            "cliff": ["discrete_grid_task"],
            "maze": ["maze_task"],
            "partial_observable": ["partial_observable_task"],
            "resource_management": ["resource_management_task"],
            "dynamic_hazard": ["dynamic_hazards_task"],
            "dynamic_hazards": ["dynamic_hazards_task"],
            "1d": ["1d_environment"],
            "2d": ["2d_environment"],
        },
    }


def ensure_graph(cfg: Optional[KnowledgeGraphConfig] = None) -> str:
    if cfg is None:
        cfg = KnowledgeGraphConfig()
    os.makedirs(cfg.root_dir, exist_ok=True)
    path = _graph_path(cfg)
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_default_graph(), f, ensure_ascii=False, indent=2)
    return path


def load_graph(cfg: Optional[KnowledgeGraphConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = KnowledgeGraphConfig()
    path = ensure_graph(cfg)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return _default_graph()
    if "edges" not in data or not isinstance(data.get("edges"), list):
        return _default_graph()
    return data


def _parent_map(graph: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        child = str(edge.get("child", "")).strip()
        parent = str(edge.get("parent", "")).strip()
        if not child or not parent:
            continue
        out.setdefault(child, set()).add(parent)
        out.setdefault(parent, set())
    return out


def _undirected_map(graph: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        child = str(edge.get("child", "")).strip()
        parent = str(edge.get("parent", "")).strip()
        if not child or not parent:
            continue
        out.setdefault(child, set()).add(parent)
        out.setdefault(parent, set()).add(child)
    return out


def ancestors(node: str, graph: Dict[str, Any]) -> Set[str]:
    node = str(node).strip()
    if not node:
        return set()
    pmap = _parent_map(graph)
    if node not in pmap:
        return {node}
    seen: Set[str] = {node}
    q = deque([node])
    while q:
        cur = q.popleft()
        for p in pmap.get(cur, set()):
            if p in seen:
                continue
            seen.add(p)
            q.append(p)
    return seen


def _tag_concepts(tags: List[str], graph: Dict[str, Any]) -> Set[str]:
    aliases = graph.get("tag_aliases", {})
    if not isinstance(aliases, dict):
        aliases = {}
    out: Set[str] = set()
    for t in tags:
        tag = str(t).strip().lower()
        if not tag:
            continue
        out.add(tag)
        mapped = aliases.get(tag, [])
        if isinstance(mapped, list):
            for m in mapped:
                out.add(str(m).strip().lower())
    return out


def concept_closure_for_verse(verse_name: str, graph: Dict[str, Any]) -> Set[str]:
    verse = str(verse_name).strip().lower()
    tags = [str(t).strip().lower() for t in tags_for_verse(verse) if str(t).strip()]
    nodes = {verse}
    nodes.update(_tag_concepts(tags, graph))

    closure: Set[str] = set()
    for n in nodes:
        closure.update(ancestors(n, graph))
    return closure


def shortest_graph_distance(
    source_nodes: Set[str],
    target_nodes: Set[str],
    graph: Dict[str, Any],
) -> Optional[int]:
    if not source_nodes or not target_nodes:
        return None
    if source_nodes & target_nodes:
        return 0
    umap = _undirected_map(graph)
    q = deque()
    seen: Set[str] = set()
    for s in source_nodes:
        q.append((s, 0))
        seen.add(s)
    target_set = set(target_nodes)

    while q:
        cur, dist = q.popleft()
        for nb in umap.get(cur, set()):
            if nb in seen:
                continue
            if nb in target_set:
                return dist + 1
            seen.add(nb)
            q.append((nb, dist + 1))
    return None


def verse_relatedness(
    source_verse: str,
    target_verse: str,
    graph: Optional[Dict[str, Any]] = None,
) -> float:
    if graph is None:
        graph = load_graph()

    src = concept_closure_for_verse(source_verse, graph)
    tgt = concept_closure_for_verse(target_verse, graph)
    if not src or not tgt:
        return 0.0
    inter = src & tgt
    union = src | tgt
    jaccard = float(len(inter) / float(len(union))) if union else 0.0

    dist = shortest_graph_distance(src, tgt, graph)
    hop = 0.0 if dist is None else (1.0 / float(1 + dist))

    return float(max(jaccard, hop))
