"""
memory/encoders.py

Encoder interfaces for turning observations into vectors.
"""

from __future__ import annotations

from typing import Any, List, Protocol

from core.types import JSONValue
from memory.embeddings import obs_to_vector

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class Encoder(Protocol):
    def encode(self, obs: JSONValue) -> List[float]:
        ...


class RawEncoder:
    def encode(self, obs: JSONValue) -> List[float]:
        return obs_to_vector(obs)


class ClipEncoder:
    """
    CLIP-style encoder for text observations.
    """

    def __init__(self, model_name: str | None = None) -> None:
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformer library not found. Please `pip install sentence-transformers`")
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)

    def encode(self, obs: JSONValue) -> List[float]:
        if not isinstance(obs, str):
            raise ValueError("CLIPEncoder expects a string observation.")
        embedding = self.model.encode(obs)
        return embedding.tolist()


def get_encoder(name: str, model_name: str | None = None) -> Encoder:
    if name == "raw":
        return RawEncoder()
    if name == "clip":
        return ClipEncoder(model_name=model_name)
    raise ValueError(f"Unknown encoder: {name}")
