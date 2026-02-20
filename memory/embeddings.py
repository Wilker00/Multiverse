"""
memory/embeddings.py

Tiny embedding helpers for JSON observations.
"""

from __future__ import annotations

import math
from typing import Any, List

from core.types import JSONValue


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _flatten_numeric(value: Any, out: List[float], *, path: str) -> None:
    if _is_number(value):
        out.append(float(value))
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _flatten_numeric(item, out, path=f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k in sorted(value.keys()):
            _flatten_numeric(value.get(k), out, path=f"{path}.{k}" if path else str(k))
        return
    raise TypeError(f"obs value at '{path or '<root>'}' must be numeric/list/dict, got {type(value)}")


def obs_to_vector(obs: JSONValue, keys: List[str] | None = None) -> List[float]:
    """
    Convert JSON obs into a flat numeric vector.
    Supports nested dict/list/tuple structures containing numeric leaves.
    Dict keys are traversed in sorted order for deterministic vectors.
    """
    if isinstance(obs, dict):
        if keys is None:
            # Exclude 'flat' — it's a redundant copy of the same numeric
            # fields already present as top-level dict keys and would double
            # the embedding dimensionality, hurting cosine similarity quality.
            keys = sorted(k for k in obs.keys() if k != "flat")
        vec: List[float] = []
        for k in keys:
            _flatten_numeric(obs.get(k), vec, path=str(k))
        return vec

    if isinstance(obs, (list, tuple)):
        vec: List[float] = []
        _flatten_numeric(obs, vec, path="<root>")
        return vec

    if _is_number(obs):
        return [float(obs)]

    raise TypeError(f"Unsupported obs type for embedding: {type(obs)}")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("cosine similarity requires vectors of equal length")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def project_vector(vec: List[float], *, dim: int = 64) -> List[float]:
    """
    Project vectors of arbitrary dimension into a shared fixed-size space.
    Uses a deterministic signed hash projection (feature hashing).
    """
    d = max(4, int(dim))
    out = [0.0 for _ in range(d)]
    if not vec:
        return out
    for i, v in enumerate(vec):
        idx = int((i * 2654435761) % d)
        sign = -1.0 if (((i * 11400714819323198485) >> 3) & 1) else 1.0
        out[idx] += float(v) * float(sign)
    norm = math.sqrt(sum(x * x for x in out))
    if norm > 1e-12:
        out = [float(x / norm) for x in out]
    return out


def obs_to_universal_vector(obs: JSONValue, *, dim: int = 64) -> List[float]:
    """
    Encode obs into a fixed-size universal vector for cross-verse matching.
    """
    return project_vector(obs_to_vector(obs), dim=int(dim))


def cosine_similarity_projected(a: List[float], b: List[float], *, dim: int = 64) -> float:
    """
    Cosine similarity for vectors with different dimensions via shared projection.
    """
    pa = project_vector(a, dim=int(dim))
    pb = project_vector(b, dim=int(dim))
    return cosine_similarity(pa, pb)
