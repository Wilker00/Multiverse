"""
Query-cache and canary helpers for the central memory repository.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.types import JSONValue
from memory.central_repository_support import CentralMemoryConfig, ScenarioMatch, _safe_int


def _query_cache_key(
    obs: JSONValue,
    verse_name: Optional[str],
    top_k: int,
    min_score: float,
    memory_tiers: Optional[Set[str]],
    memory_families: Optional[Set[str]],
    memory_types: Optional[Set[str]],
    exclude_run_ids: Optional[Set[str]],
) -> str:
    import hashlib

    key_parts = [
        json.dumps(obs, sort_keys=True, ensure_ascii=False),
        str(verse_name or ""),
        str(top_k),
        str(min_score),
        str(sorted(memory_tiers) if memory_tiers else ""),
        str(sorted(memory_families) if memory_families else ""),
        str(sorted(memory_types) if memory_types else ""),
        str(sorted(exclude_run_ids) if exclude_run_ids else ""),
    ]
    key_str = "||".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _now_ms() -> int:
    return int(time.time() * 1000)


def find_similar_cached_support(
    *,
    find_similar_fn: Callable[..., List[ScenarioMatch]],
    query_cache: Dict[str, Tuple[List[ScenarioMatch], int]],
    cache_ttl_ms: int,
    cache_size: int,
    obs: JSONValue,
    cfg: CentralMemoryConfig,
    top_k: int = 5,
    verse_name: Optional[str] = None,
    min_score: float = -1.0,
    exclude_run_ids: Optional[Set[str]] = None,
    decay_lambda: float = 0.0,
    current_time_ms: Optional[int] = None,
    memory_tiers: Optional[Set[str]] = None,
    memory_families: Optional[Set[str]] = None,
    memory_types: Optional[Set[str]] = None,
    stm_decay_lambda: Optional[float] = None,
    trajectory_window: int = 0,
) -> List[ScenarioMatch]:
    if decay_lambda != 0.0 or stm_decay_lambda is not None or current_time_ms is not None:
        return find_similar_fn(
            obs=obs,
            cfg=cfg,
            top_k=top_k,
            verse_name=verse_name,
            min_score=min_score,
            exclude_run_ids=exclude_run_ids,
            decay_lambda=decay_lambda,
            current_time_ms=current_time_ms,
            memory_tiers=memory_tiers,
            memory_families=memory_families,
            memory_types=memory_types,
            stm_decay_lambda=stm_decay_lambda,
            trajectory_window=trajectory_window,
        )

    cache_key = _query_cache_key(
        obs=obs,
        verse_name=verse_name,
        top_k=top_k,
        min_score=min_score,
        memory_tiers=memory_tiers,
        memory_families=memory_families,
        memory_types=memory_types,
        exclude_run_ids=exclude_run_ids,
    )

    if cache_key in query_cache:
        results, cached_at_ms = query_cache[cache_key]
        age_ms = _now_ms() - cached_at_ms
        if age_ms < cache_ttl_ms:
            return results
        del query_cache[cache_key]

    results = find_similar_fn(
        obs=obs,
        cfg=cfg,
        top_k=top_k,
        verse_name=verse_name,
        min_score=min_score,
        exclude_run_ids=exclude_run_ids,
        decay_lambda=decay_lambda,
        current_time_ms=current_time_ms,
        memory_tiers=memory_tiers,
        memory_families=memory_families,
        memory_types=memory_types,
        stm_decay_lambda=stm_decay_lambda,
        trajectory_window=trajectory_window,
    )
    query_cache[cache_key] = (results, _now_ms())
    if len(query_cache) > cache_size:
        sorted_items = sorted(query_cache.items(), key=lambda x: x[1][1])
        evict_count = max(1, len(sorted_items) // 10)
        for key, _ in sorted_items[:evict_count]:
            del query_cache[key]
    return results


def _similarity_canary_path(cfg: CentralMemoryConfig) -> str:
    return os.path.join(cfg.root_dir, "similarity_canaries.json")


def save_similarity_canary_support(
    *,
    ensure_repo_fn: Callable[[CentralMemoryConfig], None],
    cfg: CentralMemoryConfig,
    canary_id: str,
    obs: JSONValue,
    expected_run_id: str,
    top_k: int = 1,
    verse_name: Optional[str] = None,
    memory_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    ensure_repo_fn(cfg)
    path = _similarity_canary_path(cfg)
    payload: Dict[str, Any] = {"version": "v1", "updated_at_ms": 0, "canaries": []}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                payload = obj
        except Exception:
            payload = {"version": "v1", "updated_at_ms": 0, "canaries": []}
    rows = payload.get("canaries")
    rows = rows if isinstance(rows, list) else []
    cid = str(canary_id).strip()
    if not cid:
        raise ValueError("canary_id cannot be empty")
    row = {
        "canary_id": cid,
        "obs": obs,
        "expected_run_id": str(expected_run_id),
        "top_k": max(1, int(top_k)),
        "verse_name": (str(verse_name).strip().lower() if verse_name else ""),
        "memory_types": sorted(str(x).strip().lower() for x in (memory_types or set()) if str(x).strip()),
        "updated_at_ms": int(time.time() * 1000),
    }
    replaced = False
    for i, old in enumerate(rows):
        if not isinstance(old, dict):
            continue
        if str(old.get("canary_id", "")).strip() == cid:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)
    payload["version"] = "v1"
    payload["updated_at_ms"] = int(time.time() * 1000)
    payload["canaries"] = rows
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return row


def run_similarity_canaries_support(
    *,
    ensure_repo_fn: Callable[[CentralMemoryConfig], None],
    find_similar_fn: Callable[..., List[ScenarioMatch]],
    cfg: CentralMemoryConfig,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    ensure_repo_fn(cfg)
    path = _similarity_canary_path(cfg)
    if not os.path.isfile(path):
        return {
            "path": path,
            "total": 0,
            "ann_hits": 0,
            "exact_hits": 0,
            "agreement_rate": 1.0,
            "pass_rate_ann": 1.0,
            "cases": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {
            "path": path,
            "total": 0,
            "ann_hits": 0,
            "exact_hits": 0,
            "agreement_rate": 0.0,
            "pass_rate_ann": 0.0,
            "cases": [],
            "error": "invalid_canary_file",
        }
    rows = payload.get("canaries") if isinstance(payload, dict) else None
    rows = rows if isinstance(rows, list) else []
    if limit is not None and int(limit) > 0:
        rows = rows[: int(limit)]

    total = 0
    ann_hits = 0
    exact_hits = 0
    agreements = 0
    cases: List[Dict[str, Any]] = []
    old_ann = os.environ.get("MULTIVERSE_SIM_USE_ANN")
    try:
        for row in rows:
            if not isinstance(row, dict):
                continue
            obs = row.get("obs")
            expected = str(row.get("expected_run_id", "")).strip()
            if not expected:
                continue
            total += 1
            top_k = max(1, _safe_int(row.get("top_k", 1), 1))
            verse_name = str(row.get("verse_name", "")).strip().lower() or None
            raw_types = row.get("memory_types")
            mt: Optional[Set[str]] = None
            if isinstance(raw_types, list):
                mt = set(str(x).strip().lower() for x in raw_types if str(x).strip())
                if not mt:
                    mt = None

            os.environ["MULTIVERSE_SIM_USE_ANN"] = "1"
            ann_rows = find_similar_fn(
                obs=obs,
                cfg=cfg,
                top_k=top_k,
                verse_name=verse_name,
                min_score=-1.0,
                memory_types=mt,
            )
            ann_ids = [str(m.run_id) for m in ann_rows]
            ann_hit = expected in ann_ids
            if ann_hit:
                ann_hits += 1

            os.environ["MULTIVERSE_SIM_USE_ANN"] = "0"
            exact_rows = find_similar_fn(
                obs=obs,
                cfg=cfg,
                top_k=top_k,
                verse_name=verse_name,
                min_score=-1.0,
                memory_types=mt,
            )
            exact_ids = [str(m.run_id) for m in exact_rows]
            exact_hit = expected in exact_ids
            if exact_hit:
                exact_hits += 1
            if ann_ids == exact_ids:
                agreements += 1

            cases.append(
                {
                    "canary_id": str(row.get("canary_id", "")),
                    "expected_run_id": expected,
                    "ann_hit": bool(ann_hit),
                    "exact_hit": bool(exact_hit),
                    "ann_top_run_id": (ann_ids[0] if ann_ids else ""),
                    "exact_top_run_id": (exact_ids[0] if exact_ids else ""),
                }
            )
    finally:
        if old_ann is None:
            os.environ.pop("MULTIVERSE_SIM_USE_ANN", None)
        else:
            os.environ["MULTIVERSE_SIM_USE_ANN"] = old_ann

    pass_rate_ann = float(ann_hits / float(max(1, total)))
    agreement_rate = float(agreements / float(max(1, total)))
    return {
        "path": path,
        "total": int(total),
        "ann_hits": int(ann_hits),
        "exact_hits": int(exact_hits),
        "pass_rate_ann": float(pass_rate_ann),
        "agreement_rate": float(agreement_rate),
        "cases": cases,
    }
