"""
memory/vector_store.py

Vector store interfaces and a lightweight in-memory backend.
Optional adapters for Pinecone and Milvus are included but not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from memory.embeddings import cosine_similarity


@dataclass
class VectorRecord:
    vector_id: str
    vector: List[float]
    metadata: Dict[str, Any]


@dataclass
class VectorMatch:
    vector_id: str
    score: float
    metadata: Dict[str, Any]


class VectorStore(Protocol):
    def add(self, records: Iterable[VectorRecord]) -> None:
        ...

    def query(self, vector: List[float], top_k: int = 5) -> List[VectorMatch]:
        ...


class InMemoryVectorStore:
    """
    Simple in-memory vector store with cosine similarity.
    """

    def __init__(self) -> None:
        self._records: List[VectorRecord] = []

    def add(self, records: Iterable[VectorRecord]) -> None:
        self._records.extend(records)

    def query(self, vector: List[float], top_k: int = 5) -> List[VectorMatch]:
        scored: List[VectorMatch] = []
        for rec in self._records:
            if len(rec.vector) != len(vector):
                continue
            score = cosine_similarity(vector, rec.vector)
            scored.append(VectorMatch(vector_id=rec.vector_id, score=score, metadata=rec.metadata))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[: int(top_k)]


class PineconeVectorStore:
    """
    Optional Pinecone adapter.
    Requires `pinecone-client` to be installed.
    """

    def __init__(self, index_name: str, api_key: str, environment: str) -> None:
        try:
            import pinecone  # type: ignore
        except Exception as exc:
            raise RuntimeError("Pinecone client not installed") from exc

        pinecone.init(api_key=api_key, environment=environment)
        self._index = pinecone.Index(index_name)

    def add(self, records: Iterable[VectorRecord]) -> None:
        items = [(r.vector_id, r.vector, r.metadata) for r in records]
        if items:
            self._index.upsert(items)

    def query(self, vector: List[float], top_k: int = 5) -> List[VectorMatch]:
        res = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        out: List[VectorMatch] = []
        for match in res.get("matches", []):
            out.append(
                VectorMatch(
                    vector_id=str(match.get("id")),
                    score=float(match.get("score", 0.0)),
                    metadata=match.get("metadata", {}) or {},
                )
            )
        return out


class MilvusVectorStore:
    """
    Optional Milvus adapter.
    Requires `pymilvus` to be installed.
    """

    def __init__(self, collection_name: str, host: str = "localhost", port: str = "19530") -> None:
        try:
            from pymilvus import Collection, connections  # type: ignore
        except Exception as exc:
            raise RuntimeError("pymilvus not installed") from exc

        connections.connect(host=host, port=port)
        self._collection = Collection(collection_name)

    def add(self, records: Iterable[VectorRecord]) -> None:
        ids = []
        vectors = []
        metadata = []
        for r in records:
            ids.append(r.vector_id)
            vectors.append(r.vector)
            metadata.append(r.metadata)
        if ids:
            self._collection.insert([ids, vectors, metadata])

    def query(self, vector: List[float], top_k: int = 5) -> List[VectorMatch]:
        res = self._collection.search(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=int(top_k),
            output_fields=["metadata"],
        )
        out: List[VectorMatch] = []
        for hit in res[0]:
            out.append(
                VectorMatch(
                    vector_id=str(hit.id),
                    score=float(hit.score),
                    metadata=getattr(hit, "entity", {}).get("metadata", {}) if hasattr(hit, "entity") else {},
                )
            )
        return out
