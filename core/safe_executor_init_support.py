"""
core/safe_executor_init_support.py

Initialization helpers for SafeExecutor.

Extracted from safe_executor.py to reduce __init__ complexity.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.mcts_search import MCTSConfig, MCTSSearch, MetaTransformerValue


def init_mcts(
    *,
    config: Any,
    verse: Any,
    record_runtime_error: Callable,
) -> Tuple[Optional[MCTSSearch], Optional[Any], bool]:
    """
    Initialize MCTS search and optional value network.

    Returns (mcts, value_net, mcts_active).
    """
    mcts: Optional[MCTSSearch] = None
    mcts_value_net: Optional[Any] = None
    mcts_active = bool(config.mcts_enabled)

    if not mcts_active:
        return None, None, False

    try:
        mcts = MCTSSearch(
            verse=verse,
            config=MCTSConfig(
                num_simulations=int(config.mcts_num_simulations),
                max_depth=int(config.mcts_max_depth),
                c_puct=float(config.mcts_c_puct),
                discount=float(config.mcts_discount),
                forced_loss_threshold=float(config.mcts_loss_threshold),
                forced_loss_min_visits=int(config.mcts_min_visits),
                value_confidence_threshold=float(config.mcts_value_confidence_threshold),
            ),
        )
    except Exception as exc:
        mcts = None
        mcts_active = False
        record_runtime_error(
            code="mcts_init_error",
            exc=exc,
            context="safe_executor.mcts.init",
        )

    if mcts_active and str(config.mcts_meta_model_path).strip():
        try:
            mcts_value_net = MetaTransformerValue(
                checkpoint_path=str(config.mcts_meta_model_path).strip(),
                history_len=int(config.mcts_meta_history_len),
            )
        except Exception as exc:
            mcts_value_net = None
            record_runtime_error(
                code="mcts_value_model_load_error",
                exc=exc,
                context="safe_executor.mcts.value_model",
            )

    return mcts, mcts_value_net, mcts_active


def resolve_verse_name(verse: Any, record_runtime_error: Callable) -> str:
    """Extract verse name from a verse object, with error handling."""
    try:
        return str(getattr(getattr(verse, "spec", None), "verse_name", "")).strip().lower()
    except Exception as exc:
        record_runtime_error(
            code="verse_name_resolution_error",
            exc=exc,
            context="safe_executor.verse_name",
        )
        return ""


def init_danger_map(
    *,
    danger_map_path: str,
    record_runtime_error: Callable,
    load_danger_map_fn: Callable,
) -> Tuple[List[Dict[str, Any]], int]:
    """Load danger map if path exists, otherwise return empty defaults."""
    if danger_map_path and os.path.isfile(danger_map_path):
        return load_danger_map_fn(danger_map_path, record_runtime_error)
    return [], 0


def init_failure_signatures(
    *,
    failure_signature_path: str,
    failure_signature_embedding_dim: int,
    record_runtime_error: Callable,
    load_failure_signatures_fn: Callable,
) -> List[Dict[str, Any]]:
    """Load failure signatures if path exists, otherwise return empty list."""
    if failure_signature_path and os.path.isfile(failure_signature_path):
        return load_failure_signatures_fn(
            failure_signature_path,
            failure_signature_embedding_dim,
            record_runtime_error,
        )
    return []
