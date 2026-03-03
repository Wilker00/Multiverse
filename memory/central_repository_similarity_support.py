"""
Similarity execution and ANN runtime helpers for the central memory repository.
"""

from __future__ import annotations

import heapq
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from core.types import JSONValue
from memory.central_repository_cache_support import _PreparedMemoryRow, _SimilarityCacheEntry
from memory.central_repository_support import CentralMemoryConfig, ScenarioMatch
from memory.decay_manager import apply_temporal_decay
from memory.embeddings import cosine_similarity, obs_to_universal_vector, obs_to_vector

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


def _sim_cache_limit_support() -> int:
    raw = os.environ.get("MULTIVERSE_SIM_CACHE_MAX_FILES", "2")
    try:
        return max(1, int(raw))
    except Exception:
        return 2


def _ann_enabled_support() -> bool:
    raw = str(os.environ.get("MULTIVERSE_SIM_USE_ANN", "1")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _ann_candidate_count_support(
    *,
    top_n: int,
    n_rows: int,
    ann_dynamic_factor: Optional[int],
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
) -> int:
    _, _, ann_tune_lock, _ = get_mp_locks_fn()

    factor_raw = os.environ.get("MULTIVERSE_SIM_ANN_FACTOR", "64")
    try:
        factor = max(1, int(factor_raw))
    except Exception:
        factor = 64
    with ann_tune_lock:
        dyn = ann_dynamic_factor
    if dyn is not None:
        factor = max(factor, int(dyn))
    return min(max(256, int(top_n) * factor), int(n_rows))


def _ann_drift_check_every_support() -> int:
    raw = os.environ.get("MULTIVERSE_SIM_ANN_DRIFT_CHECK_EVERY", "250")
    try:
        return max(1, int(raw))
    except Exception:
        return 250


def _ann_max_allowed_drift_support() -> float:
    raw = os.environ.get("MULTIVERSE_SIM_ANN_MAX_DRIFT", "0.03")
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.03


def _ann_should_check_drift_support(
    *,
    runtime_state: Dict[str, Any],
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
    ann_drift_check_every_fn: Callable[[], int],
) -> bool:
    _, _, ann_tune_lock, _ = get_mp_locks_fn()
    every = ann_drift_check_every_fn()
    with ann_tune_lock:
        runtime_state["query_count"] = int(runtime_state.get("query_count", 0)) + 1
        return bool((int(runtime_state["query_count"]) % every) == 0)


def _ann_record_drift_support(
    *,
    runtime_state: Dict[str, Any],
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
    ann_max_allowed_drift_fn: Callable[[], float],
    drift: float,
    n_rows: int,
    top_n: int,
) -> None:
    _, _, ann_tune_lock, _ = get_mp_locks_fn()
    d = max(0.0, float(drift))
    max_allowed = ann_max_allowed_drift_fn()
    with ann_tune_lock:
        runtime_state["drift_checks"] = int(runtime_state.get("drift_checks", 0)) + 1
        runtime_state["last_drift"] = float(d)
        runtime_state["max_drift"] = max(float(runtime_state.get("max_drift", 0.0)), float(d))
        if d <= max_allowed:
            return
        current = int(runtime_state.get("dynamic_factor") or 0)
        if current <= 0:
            current = 64
        max_factor = max(64, int(max(1, n_rows) / max(1, top_n)))
        runtime_state["dynamic_factor"] = min(max_factor, max(current + 8, int(current * 1.5)))


def get_similarity_runtime_metrics_support(
    *,
    runtime_state: Dict[str, Any],
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
    ann_drift_check_every_fn: Callable[[], int],
    ann_max_allowed_drift_fn: Callable[[], float],
) -> Dict[str, Any]:
    _, _, ann_tune_lock, _ = get_mp_locks_fn()
    with ann_tune_lock:
        return {
            "ann_dynamic_factor": (
                None if runtime_state.get("dynamic_factor") is None else int(runtime_state["dynamic_factor"])
            ),
            "ann_query_count": int(runtime_state.get("query_count", 0)),
            "ann_drift_checks": int(runtime_state.get("drift_checks", 0)),
            "ann_last_drift": float(runtime_state.get("last_drift", 0.0)),
            "ann_max_drift": float(runtime_state.get("max_drift", 0.0)),
            "ann_drift_check_every": int(ann_drift_check_every_fn()),
            "ann_max_allowed_drift": float(ann_max_allowed_drift_fn()),
        }


def reset_similarity_runtime_metrics_support(
    *,
    runtime_state: Dict[str, Any],
    get_mp_locks_fn: Callable[[], Tuple[Any, Any, Any, Any]],
) -> None:
    _, _, ann_tune_lock, _ = get_mp_locks_fn()
    with ann_tune_lock:
        runtime_state["dynamic_factor"] = None
        runtime_state["query_count"] = 0
        runtime_state["drift_checks"] = 0
        runtime_state["last_drift"] = 0.0
        runtime_state["max_drift"] = 0.0


def _normalize_query_vector(vec: List[float]) -> Optional[Any]:
    if np is None or not vec:
        return None
    try:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            return None
        return (arr / norm).astype(np.float32)
    except Exception:
        return None


def _row_matches_filters(
    *,
    row: _PreparedMemoryRow,
    target_verse: str,
    exclude_run_ids: Optional[Set[str]],
    tier_filter: Optional[Set[str]],
    family_filter: Optional[Set[str]],
    type_filter: Optional[Set[str]],
) -> bool:
    if target_verse and row.verse_name != target_verse:
        return False
    if exclude_run_ids and str(row.run_id) in exclude_run_ids:
        return False
    if tier_filter is not None and row.row_tier not in tier_filter:
        return False
    if family_filter is not None and row.row_family not in family_filter:
        return False
    if type_filter is not None and row.row_type not in type_filter:
        return False
    return True


def _decayed_score(
    *,
    raw_score: float,
    row: _PreparedMemoryRow,
    cfg: CentralMemoryConfig,
    decay_lambda: float,
    current_time_ms: Optional[int],
    min_score: float,
    stm_decay_lambda: Optional[float],
) -> Optional[Tuple[float, float]]:
    if row.row_tier == "ltm":
        row_decay_lambda = 0.0
    elif stm_decay_lambda is None:
        row_decay_lambda = max(float(decay_lambda), float(cfg.stm_decay_lambda))
    else:
        row_decay_lambda = max(0.0, float(stm_decay_lambda))

    score, recency_weight = apply_temporal_decay(
        score=float(raw_score),
        t_ms=int(row.t_ms),
        decay_lambda=float(row_decay_lambda),
        current_time_ms=current_time_ms,
    )
    if score < float(min_score):
        return None
    return float(score), float(recency_weight)


def _extract_trajectory(
    *,
    cache: _SimilarityCacheEntry,
    row_input: _PreparedMemoryRow,
    trajectory_window: int,
) -> Optional[List[Dict[str, Any]]]:
    if int(trajectory_window) <= 0:
        return None
    ep_id = str(row_input.episode_id)
    root_step = int(row_input.step_idx)
    idx = -1
    for pos, row in enumerate(cache.rows):
        if row is row_input:
            idx = pos
            break
    if idx < 0:
        return []
    traj = []
    curr = idx
    while curr >= 0 and len(traj) < int(trajectory_window):
        row = cache.rows[curr]
        if str(row.episode_id) != ep_id or int(row.step_idx) > root_step:
            if str(row.episode_id) != ep_id:
                break
            curr -= 1
            continue
        traj.append(
            {
                "step_idx": int(row.step_idx),
                "obs": row.obs,
                "action": row.action,
                "reward": float(row.reward),
            }
        )
        curr -= 1
    traj.reverse()
    return traj


def _build_match(
    *,
    row: _PreparedMemoryRow,
    score: float,
    recency_weight: float,
    trajectory: Optional[List[Dict[str, Any]]] = None,
) -> ScenarioMatch:
    return ScenarioMatch(
        score=float(score),
        run_id=str(row.run_id),
        episode_id=str(row.episode_id),
        step_idx=int(row.step_idx),
        t_ms=int(row.t_ms),
        verse_name=str(row.verse_name),
        action=row.action,
        reward=float(row.reward),
        obs=row.obs,
        source_greedy_action=row.source_greedy_action,
        source_action_matches_greedy=row.source_action_matches_greedy,
        recency_weight=float(recency_weight),
        trajectory=trajectory,
    )


def _push_match(
    *,
    heap: List[Tuple[float, int, ScenarioMatch]],
    top_n: int,
    ordinal: int,
    match: ScenarioMatch,
) -> int:
    score = float(match.score)
    if len(heap) < top_n:
        heapq.heappush(heap, (score, ordinal, match))
    elif score > heap[0][0]:
        heapq.heapreplace(heap, (score, ordinal, match))
    return ordinal + 1


def find_similar_support(
    *,
    ensure_repo_fn: Callable[[CentralMemoryConfig], None],
    get_similarity_cache_for_path_fn: Callable[..., _SimilarityCacheEntry],
    memories_path_fn: Callable[[CentralMemoryConfig], str],
    ltm_memories_path_fn: Callable[[CentralMemoryConfig], str],
    stm_memories_path_fn: Callable[[CentralMemoryConfig], str],
    load_tier_policy_fn: Callable[[CentralMemoryConfig], Optional[Dict[str, Any]]],
    as_set_fn: Callable[[Optional[Iterable[str]]], Optional[Set[str]]],
    universal_obs_dim_fn: Callable[[], int],
    ann_enabled_fn: Callable[[], bool],
    ann_candidate_count_fn: Callable[[int, int], int],
    ann_should_check_drift_fn: Callable[[], bool],
    ann_record_drift_fn: Callable[..., None],
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
    ensure_repo_fn(cfg)
    query_vec = obs_to_vector(obs)
    query_vec_u = obs_to_universal_vector(obs, dim=universal_obs_dim_fn())
    q_u_dim = len(query_vec_u)
    q_u_norm = _normalize_query_vector(query_vec_u)
    top_n = max(1, int(top_k))
    tier_filter = as_set_fn(memory_tiers)
    family_filter = as_set_fn(memory_families)
    type_filter = as_set_fn(memory_types)

    if tier_filter == {"ltm"}:
        mem_paths = [ltm_memories_path_fn(cfg)]
    elif tier_filter == {"stm"}:
        mem_paths = [stm_memories_path_fn(cfg)]
    else:
        mem_paths = [memories_path_fn(cfg)]

    tier_policy = load_tier_policy_fn(cfg)
    target_verse = str(verse_name).strip().lower() if verse_name else ""
    q_dim = len(query_vec)
    heap: List[Tuple[float, int, ScenarioMatch]] = []
    ordinal = 0

    for mem_path in mem_paths:
        if not os.path.isfile(mem_path):
            continue
        cache = get_similarity_cache_for_path_fn(mem_path=mem_path, tier_policy=tier_policy)
        dim_row_idxs = list(cache.row_indices_by_dim.get(q_dim, []))
        raw_dim_available = bool(q_dim > 0 and dim_row_idxs)

        q_norm: Optional[Any] = None
        sims: Optional[Any] = None
        use_vec = bool(np is not None and raw_dim_available and q_dim in cache.vectors_by_dim)
        if use_vec:
            q_norm = _normalize_query_vector(query_vec)
            if q_norm is None:
                use_vec = False
            else:
                try:
                    sims = cache.vectors_by_dim[q_dim] @ q_norm
                except Exception:
                    use_vec = False
                    sims = None

        def score_row(row: _PreparedMemoryRow, raw_score: float) -> Optional[Tuple[float, float]]:
            return _decayed_score(
                raw_score=raw_score,
                row=row,
                cfg=cfg,
                decay_lambda=decay_lambda,
                current_time_ms=current_time_ms,
                min_score=min_score,
                stm_decay_lambda=stm_decay_lambda,
            )

        def push_row(row: _PreparedMemoryRow, score: float, recency_weight: float, with_trajectory: bool) -> None:
            nonlocal ordinal
            match = _build_match(
                row=row,
                score=score,
                recency_weight=recency_weight,
                trajectory=(
                    _extract_trajectory(cache=cache, row_input=row, trajectory_window=trajectory_window)
                    if with_trajectory
                    else None
                ),
            )
            ordinal = _push_match(heap=heap, top_n=top_n, ordinal=ordinal, match=match)

        if raw_dim_available and use_vec and sims is not None:
            local_best_score = float("-inf")

            def score_position(pos: int) -> Optional[Tuple[float, _PreparedMemoryRow, float]]:
                row_idx = dim_row_idxs[int(pos)]
                row = cache.rows[int(row_idx)]
                if not _row_matches_filters(
                    row=row,
                    target_verse=target_verse,
                    exclude_run_ids=exclude_run_ids,
                    tier_filter=tier_filter,
                    family_filter=family_filter,
                    type_filter=type_filter,
                ):
                    return None
                scored = score_row(row, float(sims[pos]))
                if scored is None:
                    return None
                score, recency_weight = scored
                return float(score), row, float(recency_weight)

            candidate_positions: Optional[List[int]] = None
            if ann_enabled_fn() and q_norm is not None and len(dim_row_idxs) > top_n:
                ann_index = cache.ann_by_dim.get(q_dim)
                if ann_index is not None:
                    try:
                        n_candidates = ann_candidate_count_fn(top_n, len(dim_row_idxs))
                        if 0 < n_candidates < len(dim_row_idxs):
                            idxs = ann_index.kneighbors(
                                np.asarray([q_norm], dtype=np.float32),
                                n_neighbors=int(n_candidates),
                                return_distance=False,
                            )
                            if idxs is not None and len(idxs):
                                candidate_positions = []
                                seen: Set[int] = set()
                                for raw_pos in list(idxs[0]):
                                    pos = int(raw_pos)
                                    if pos < 0 or pos >= len(dim_row_idxs) or pos in seen:
                                        continue
                                    seen.add(pos)
                                    candidate_positions.append(pos)
                    except Exception:
                        candidate_positions = None

            def scan_positions(positions: Iterable[int]) -> None:
                nonlocal local_best_score
                for pos in positions:
                    scored = score_position(int(pos))
                    if scored is None:
                        continue
                    score, row, recency_weight = scored
                    local_best_score = max(float(local_best_score), float(score))
                    push_row(row, score, recency_weight, True)

            if candidate_positions is None:
                scan_positions(range(len(dim_row_idxs)))
            else:
                visited = set(candidate_positions)
                did_full_scan = False
                scan_positions(candidate_positions)
                if len(heap) < top_n:
                    did_full_scan = True
                    scan_positions(pos for pos in range(len(dim_row_idxs)) if pos not in visited)
                if not did_full_scan and ann_should_check_drift_fn():
                    exact_best = float("-inf")
                    for pos in range(len(dim_row_idxs)):
                        scored = score_position(pos)
                        if scored is None:
                            continue
                        exact_best = max(float(exact_best), float(scored[0]))
                    if exact_best > float("-inf"):
                        if local_best_score > float("-inf"):
                            drift = max(0.0, float(exact_best - local_best_score))
                        else:
                            drift = max(0.0, float(exact_best - float(min_score)))
                        ann_record_drift_fn(drift=drift, n_rows=len(dim_row_idxs), top_n=top_n)
        elif raw_dim_available:
            if target_verse:
                row_iter: Iterable[_PreparedMemoryRow] = (
                    cache.rows[i] for i in cache.by_verse.get(target_verse, [])
                )
            else:
                row_iter = cache.rows

            for row in row_iter:
                if len(row.obs_vector) != q_dim:
                    continue
                if not _row_matches_filters(
                    row=row,
                    target_verse=target_verse,
                    exclude_run_ids=exclude_run_ids,
                    tier_filter=tier_filter,
                    family_filter=family_filter,
                    type_filter=type_filter,
                ):
                    continue
                scored = score_row(row, cosine_similarity(query_vec, row.obs_vector))
                if scored is None:
                    continue
                score, recency_weight = scored
                push_row(row, score, recency_weight, True)

        need_universal = bool((len(heap) < top_n) or (not raw_dim_available))
        if not need_universal or q_u_dim <= 0:
            continue
        row_indices = list(cache.universal_row_indices or [])
        if not row_indices:
            continue
        u_mat = cache.universal_vectors
        for pos, row_idx in enumerate(row_indices):
            row = cache.rows[int(row_idx)]
            if raw_dim_available and len(row.obs_vector) == q_dim:
                continue
            if len(row.obs_vector_u) != q_u_dim:
                continue
            if not _row_matches_filters(
                row=row,
                target_verse=target_verse,
                exclude_run_ids=exclude_run_ids,
                tier_filter=tier_filter,
                family_filter=family_filter,
                type_filter=type_filter,
            ):
                continue
            if np is not None and u_mat is not None:
                try:
                    raw_score = float(u_mat[pos] @ q_u_norm) if q_u_norm is not None else 0.0
                except Exception:
                    raw_score = cosine_similarity(query_vec_u, row.obs_vector_u)
            else:
                raw_score = cosine_similarity(query_vec_u, row.obs_vector_u)

            scored = score_row(row, raw_score)
            if scored is None:
                continue
            score, recency_weight = scored
            push_row(row, score, recency_weight, False)

    heap.sort(key=lambda item: item[0], reverse=True)
    return [match for _, _, match in heap]
