"""
Cache-building and vectorization helpers for the central memory repository.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.types import JSONValue
from memory.central_repository_support import (
    _memory_tier_for_event,
    _normalize_memory_tier,
    _row_verse_name,
    _safe_float,
    _safe_int,
)
from memory.embeddings import obs_to_universal_vector, project_vector
from memory.task_taxonomy import memory_family_for_verse, memory_type_for_verse

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None
try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover
    NearestNeighbors = None


@dataclass
class _PreparedMemoryRow:
    run_id: str
    episode_id: str
    step_idx: int
    t_ms: int
    verse_name: str
    row_type: str
    row_family: str
    row_tier: str
    action: JSONValue
    reward: float
    obs: JSONValue
    source_greedy_action: Optional[int]
    source_action_matches_greedy: Optional[bool]
    obs_vector: List[float]
    obs_vector_u: List[float]


@dataclass
class _SimilarityCacheEntry:
    signature: Tuple[int, int]
    rows: List[_PreparedMemoryRow]
    by_verse: Dict[str, List[int]]
    vectors_by_dim: Dict[int, Any]
    row_indices_by_dim: Dict[int, List[int]]
    ann_by_dim: Dict[int, Any]
    universal_vectors: Any
    universal_row_indices: List[int]
    built_at_ms: int


@dataclass
class _SimilarityCacheDelta:
    base_signature: Tuple[int, int]
    delta_rows: List[_PreparedMemoryRow]
    added_at_ms: int


def _file_signature(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    mtime_ns = int(getattr(st, "st_mtime_ns", int(float(st.st_mtime) * 1_000_000_000)))
    return int(st.st_size), int(mtime_ns)


def _simcache_path(mem_path: str) -> str:
    return str(mem_path) + ".simcache.json"


def _legacy_simcache_path(mem_path: str) -> str:
    return str(mem_path) + ".simcache.pkl"


def _universal_obs_dim() -> int:
    raw = os.environ.get("MULTIVERSE_UNIVERSAL_OBS_DIM", "64")
    try:
        return max(4, int(raw))
    except Exception:
        return 64


def _extract_or_build_universal_vector(
    *,
    row: Dict[str, Any],
    obs_vector: Optional[List[float]] = None,
) -> List[float]:
    raw = row.get("obs_vector_u")
    if isinstance(raw, list):
        try:
            vec = [float(v) for v in raw]
            if vec:
                return vec
        except Exception:
            pass

    if obs_vector:
        try:
            return project_vector(obs_vector, dim=_universal_obs_dim())
        except Exception:
            pass
    try:
        return obs_to_universal_vector(row.get("obs"), dim=_universal_obs_dim())
    except Exception:
        return []


def _vectorize_rows_by_dim(rows: List[_PreparedMemoryRow]) -> tuple[Dict[int, Any], Dict[int, List[int]]]:
    vectors_by_dim: Dict[int, Any] = {}
    row_indices_by_dim: Dict[int, List[int]] = {}
    for idx, row in enumerate(rows):
        dim = int(len(row.obs_vector))
        if dim <= 0:
            continue
        row_indices_by_dim.setdefault(dim, []).append(int(idx))
    if np is None:
        return vectors_by_dim, row_indices_by_dim
    for dim, indices in row_indices_by_dim.items():
        mat = np.asarray([rows[i].obs_vector for i in indices], dtype=np.float32)
        if mat.size <= 0:
            continue
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        mat = mat / norms
        vectors_by_dim[dim] = mat
    return vectors_by_dim, row_indices_by_dim


def _vectorize_universal_rows(rows: List[_PreparedMemoryRow]) -> tuple[Any, List[int]]:
    indices: List[int] = []
    if np is None:
        return None, indices
    mat_rows: List[List[float]] = []
    dim = _universal_obs_dim()
    for idx, row in enumerate(rows):
        vec = list(row.obs_vector_u or [])
        if len(vec) != dim:
            continue
        indices.append(int(idx))
        mat_rows.append(vec)
    if not mat_rows:
        return None, []
    mat = np.asarray(mat_rows, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    mat = mat / norms
    return mat, indices


def _prepared_row_from_any(raw: Any) -> Optional[_PreparedMemoryRow]:
    if isinstance(raw, _PreparedMemoryRow):
        return raw
    if not isinstance(raw, dict):
        return None
    try:
        base_vec = [float(x) for x in list(raw.get("obs_vector", []) or [])]
    except Exception:
        return None
    if not base_vec:
        return None
    try:
        uni_vec = [float(x) for x in list(raw.get("obs_vector_u", []) or [])]
    except Exception:
        uni_vec = []
    if not uni_vec:
        uni_vec = _extract_or_build_universal_vector(row=raw, obs_vector=base_vec)
    return _PreparedMemoryRow(
        run_id=str(raw.get("run_id", "")),
        episode_id=str(raw.get("episode_id", "")),
        step_idx=_safe_int(raw.get("step_idx", 0)),
        t_ms=_safe_int(raw.get("t_ms", 0)),
        verse_name=str(raw.get("verse_name", "")),
        row_type=str(raw.get("row_type", "")),
        row_family=str(raw.get("row_family", "")),
        row_tier=str(raw.get("row_tier", "")),
        action=raw.get("action"),
        reward=_safe_float(raw.get("reward", 0.0)),
        obs=raw.get("obs"),
        source_greedy_action=(
            None if raw.get("source_greedy_action") in (None, "") else _safe_int(raw.get("source_greedy_action"), -1)
        ),
        source_action_matches_greedy=(
            None
            if raw.get("source_action_matches_greedy") is None
            else bool(raw.get("source_action_matches_greedy"))
        ),
        obs_vector=base_vec,
        obs_vector_u=uni_vec,
    )


def _prepared_row_to_dict(row: _PreparedMemoryRow) -> Dict[str, Any]:
    return {
        "run_id": str(row.run_id),
        "episode_id": str(row.episode_id),
        "step_idx": int(row.step_idx),
        "t_ms": int(row.t_ms),
        "verse_name": str(row.verse_name),
        "row_type": str(row.row_type),
        "row_family": str(row.row_family),
        "row_tier": str(row.row_tier),
        "action": row.action,
        "reward": float(row.reward),
        "obs": row.obs,
        "source_greedy_action": (None if row.source_greedy_action is None else int(row.source_greedy_action)),
        "source_action_matches_greedy": (
            None if row.source_action_matches_greedy is None else bool(row.source_action_matches_greedy)
        ),
        "obs_vector": [float(v) for v in list(row.obs_vector or [])],
        "obs_vector_u": [float(v) for v in list(row.obs_vector_u or [])],
    }


def _index_rows_by_verse(rows: List[_PreparedMemoryRow]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        verse = str(row.verse_name or "")
        out.setdefault(verse, []).append(int(idx))
    return out


def _get_ann_index(cache: _SimilarityCacheEntry, dim: int) -> Optional[Any]:
    if NearestNeighbors is None or np is None:
        return None
    if dim in cache.ann_by_dim:
        return cache.ann_by_dim.get(dim)
    mat = cache.vectors_by_dim.get(dim)
    if mat is None:
        return None
    try:
        model = NearestNeighbors(metric="euclidean", algorithm="auto")
        model.fit(mat)
    except Exception:
        return None
    cache.ann_by_dim[dim] = model
    return model


def _build_cache_from_rows(
    rows: List[_PreparedMemoryRow],
    signature: Tuple[int, int],
) -> _SimilarityCacheEntry:
    by_verse = _index_rows_by_verse(rows)
    vectors_by_dim, row_indices_by_dim = _vectorize_rows_by_dim(rows)
    universal_vectors, universal_row_indices = _vectorize_universal_rows(rows)
    return _SimilarityCacheEntry(
        signature=signature,
        rows=rows,
        by_verse=by_verse,
        vectors_by_dim=vectors_by_dim,
        row_indices_by_dim=row_indices_by_dim,
        ann_by_dim={},
        universal_vectors=universal_vectors,
        universal_row_indices=[int(i) for i in universal_row_indices],
        built_at_ms=int(time.time() * 1000),
    )


def _build_similarity_cache_for_path(
    *,
    mem_path: str,
    tier_policy: Optional[Dict[str, Any]],
) -> _SimilarityCacheEntry:
    apath = os.path.abspath(mem_path)
    sig = _file_signature(apath)
    sidecar = _simcache_path(apath)
    legacy_sidecar = _legacy_simcache_path(apath)
    if os.path.isfile(legacy_sidecar):
        try:
            os.remove(legacy_sidecar)
        except Exception:
            pass
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if (
                isinstance(payload, dict)
                and str(payload.get("format", "")).strip() == "simcache_json_v1"
                and isinstance(payload.get("signature"), list)
                and tuple(payload.get("signature", ())) == tuple(sig)
                and isinstance(payload.get("rows"), list)
            ):
                rows: List[_PreparedMemoryRow] = []
                for raw_row in payload.get("rows", []):
                    rec = _prepared_row_from_any(raw_row)
                    if rec is not None:
                        rows.append(rec)
                if not rows:
                    raise ValueError("invalid simcache rows")
                return _build_cache_from_rows(rows, sig)
        except Exception:
            pass

    rows: List[_PreparedMemoryRow] = []
    with open(apath, "r", encoding="utf-8") as f:
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
            row_vec = row.get("obs_vector")
            if not isinstance(row_vec, list):
                continue
            try:
                row_vec_f = [float(v) for v in row_vec]
            except Exception:
                continue
            row_verse = _row_verse_name(row)
            row_type = str(row.get("memory_type", memory_type_for_verse(row_verse))).strip().lower()
            row_family = str(row.get("memory_family", memory_family_for_verse(row_verse))).strip().lower()
            row_tier = _normalize_memory_tier(row.get("memory_tier"))
            if not row_tier:
                row_tier = _memory_tier_for_event(row, tier_policy=tier_policy)

            rec = _PreparedMemoryRow(
                run_id=str(row.get("run_id", "")),
                episode_id=str(row.get("episode_id", "")),
                step_idx=_safe_int(row.get("step_idx", 0)),
                t_ms=_safe_int(row.get("t_ms", 0)),
                verse_name=row_verse,
                row_type=row_type,
                row_family=row_family,
                row_tier=row_tier,
                action=row.get("action"),
                reward=_safe_float(row.get("reward", 0.0)),
                obs=row.get("obs"),
                source_greedy_action=(
                    None
                    if not isinstance(row.get("info"), dict)
                    else (
                        None
                        if not isinstance((row.get("info") or {}).get("action_info"), dict)
                        else (
                            None
                            if ((row.get("info") or {}).get("action_info") or {}).get("greedy_action") in (None, "")
                            else _safe_int((((row.get("info") or {}).get("action_info") or {}).get("greedy_action")), -1)
                        )
                    )
                ),
                source_action_matches_greedy=(
                    None
                    if not isinstance(row.get("info"), dict)
                    else (
                        None
                        if not isinstance((row.get("info") or {}).get("action_info"), dict)
                        else bool((((row.get("info") or {}).get("action_info") or {}).get("action_matches_greedy")))
                    )
                ),
                obs_vector=row_vec_f,
                obs_vector_u=_extract_or_build_universal_vector(row=row, obs_vector=row_vec_f),
            )
            rows.append(rec)

    entry = _build_cache_from_rows(rows, sig)
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "simcache_json_v1",
                    "signature": [int(sig[0]), int(sig[1])],
                    "rows": [_prepared_row_to_dict(r) for r in rows],
                    "by_verse": {str(k): [int(i) for i in list(v)] for k, v in entry.by_verse.items()},
                },
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )
    except Exception:
        pass
    return entry
