"""
Shared runtime helpers for agent-issued memory requests.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from core.rollout_support import _as_set, _build_memory_bundle
from core.types import JSONValue

MemoryErrorSink = Callable[[str, str, Optional[int], Exception], None]


def request_memory_payload(
    *,
    agent: Any,
    method_name: str,
    request_obs: JSONValue,
    request_step_idx: int,
    on_error: MemoryErrorSink,
    error_code: str,
    component: str,
) -> Optional[Dict[str, Any]]:
    if not hasattr(agent, method_name):
        return None
    fn = getattr(agent, method_name)
    try:
        req = fn(obs=request_obs, step_idx=request_step_idx)
    except TypeError:
        try:
            req = fn(request_obs)
        except Exception as exc:
            on_error(str(error_code), str(component), int(request_step_idx), exc)
            return None
    except Exception as exc:
        on_error(str(error_code), str(component), int(request_step_idx), exc)
        return None
    return req if isinstance(req, dict) else None


def resolve_memory_request(
    *,
    agent: Any,
    req: Dict[str, Any],
    default_obs: JSONValue,
    request_step_idx: int,
    find_similar_fn: Optional[Callable[..., List[Any]]],
    memory_cfg: Optional[Any],
    on_error: MemoryErrorSink,
    lookup_error_code: str,
    lookup_component: str,
    response_error_code: str,
) -> Optional[Dict[str, Any]]:
    try:
        if find_similar_fn is None or memory_cfg is None:
            raise RuntimeError("on-demand memory lookup unavailable")
        query_obs = req.get("query_obs", default_obs)
        top_k = max(1, int(req.get("top_k", 3)))
        min_score = float(req.get("min_score", -1.0))
        trajectory_window = max(0, int(req.get("trajectory_window", 0)))
        verse_name = req.get("verse_name")
        verse_name = None if verse_name in (None, "") else str(verse_name).strip().lower()
        matches = find_similar_fn(
            obs=query_obs,
            cfg=memory_cfg,
            top_k=top_k,
            verse_name=verse_name,
            min_score=min_score,
            memory_families=_as_set(req.get("memory_families")),
            memory_types=_as_set(req.get("memory_types")),
            policy_ids=_as_set(req.get("policy_ids")),
            exclude_policy_ids=_as_set(req.get("exclude_policy_ids")),
            source_verse_names=_as_set(req.get("source_verse_names")),
            min_transfer_score=(
                None if req.get("min_transfer_score") is None else float(req.get("min_transfer_score"))
            ),
            min_transfer_confidence=(
                None if req.get("min_transfer_confidence") is None else float(req.get("min_transfer_confidence"))
            ),
            trajectory_window=trajectory_window,
        )
        bundle = _build_memory_bundle(req=req, matches=matches, step_idx=request_step_idx)
        if hasattr(agent, "on_memory_response"):
            try:
                agent.on_memory_response(bundle)  # type: ignore[attr-defined]
            except Exception as exc:
                on_error(str(response_error_code), "rollout.memory.on_memory_response", int(request_step_idx), exc)
        return {
            "bundle": bundle,
            "match_count": int(len(matches)),
        }
    except Exception as exc:
        on_error(str(lookup_error_code), str(lookup_component), int(request_step_idx), exc)
        return None

