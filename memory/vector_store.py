"""
memory/vector_store.py

Vector store interfaces and a lightweight in-memory backend.
Optional adapters for Pinecone and Milvus are included but not required.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from memory.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


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


class FAISSVectorStore:
    """
    High-performance ANN search using FAISS.
    Auto-selects Flat (<10K vectors, exact) or IVF (≥10K vectors, 95-99% recall).

    Requires faiss-cpu or faiss-gpu to be installed.
    """

    def __init__(self, dimension: int, auto_select: bool = True) -> None:
        """
        Initialize FAISS vector store.

        Args:
            dimension: Vector dimensionality
            auto_select: If True, auto-select index type based on vector count
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            ) from exc

        self._faiss = faiss
        self._dimension = dimension
        self._auto_select = auto_select
        self._records: List[VectorRecord] = []
        self._index: Optional[Any] = None
        self._use_ivf = False

    def add(self, records: Iterable[VectorRecord]) -> None:
        """Add vectors to the store."""
        new_records = list(records)
        self._records.extend(new_records)

        # Rebuild index with new vectors
        self._build_index()

    def _build_index(self) -> None:
        """Build or rebuild the FAISS index."""
        if not self._records:
            return

        import numpy as np

        # Extract vectors
        vectors = np.array([rec.vector for rec in self._records], dtype=np.float32)
        n_vectors = len(vectors)

        # Auto-select index type based on size
        if self._auto_select:
            # Use Flat for <10K vectors (exact search)
            # Use IVF for ≥10K vectors (approximate, 10-100× faster)
            self._use_ivf = n_vectors >= 10000

        if self._use_ivf:
            # IVF index with clustering
            nlist = min(100, int(np.sqrt(n_vectors)))  # Number of clusters
            quantizer = self._faiss.IndexFlatIP(self._dimension)  # Inner product (cosine after normalization)
            self._index = self._faiss.IndexIVFFlat(quantizer, self._dimension, nlist)

            # Train the index (required for IVF)
            if not self._index.is_trained:
                self._index.train(vectors)

            # Add vectors
            self._index.add(vectors)

            # Set search parameters (nprobe = clusters to search)
            self._index.nprobe = min(10, nlist)  # Search 10 clusters by default

            logger.info(f"Built IVF FAISS index: {n_vectors} vectors, {nlist} clusters, nprobe={self._index.nprobe}")
        else:
            # Flat index (exact search)
            self._index = self._faiss.IndexFlatIP(self._dimension)
            self._index.add(vectors)
            logger.info(f"Built Flat FAISS index: {n_vectors} vectors (exact search)")

    def query(self, vector: List[float], top_k: int = 5) -> List[VectorMatch]:
        """Query for similar vectors."""
        if not self._index or not self._records:
            return []

        import numpy as np

        # Prepare query vector
        query_vec = np.array([vector], dtype=np.float32)

        # Search
        scores, indices = self._index.search(query_vec, int(top_k))

        # Build results
        results: List[VectorMatch] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._records):
                continue
            rec = self._records[idx]
            results.append(VectorMatch(
                vector_id=rec.vector_id,
                score=float(score),
                metadata=rec.metadata
            ))

        return results


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
        try:
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
        except Exception as exc:
            logger.error(f"Pinecone query failed: {exc}")
            raise


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
        try:
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
        except Exception as exc:
            logger.error(f"Milvus query failed: {exc}")
            raise


class ResilientVectorStore:
    """
    Resilient vector store with automatic fallback chain.

    Tries: Primary → Secondary → InMemory fallback

    Logs all fallbacks and tracks fallback rates for monitoring.
    """

    def __init__(
        self,
        primary: VectorStore,
        secondary: Optional[VectorStore] = None,
        fallback_timeout: float = 10.0
    ) -> None:
        """
        Initialize resilient store with fallback chain.

        Args:
            primary: Primary vector store to use
            secondary: Optional secondary store (falls back to InMemory if None)
            fallback_timeout: Timeout in seconds for external service calls
        """
        self._primary = primary
        self._secondary = secondary or InMemoryVectorStore()
        self._fallback = InMemoryVectorStore()
        self._fallback_timeout = fallback_timeout

        # Metrics
        self._primary_failures = 0
        self._secondary_failures = 0
        self._total_queries = 0

    def add(self, records: Iterable[VectorRecord]) -> None:
        """Add records to all stores in fallback chain."""
        records_list = list(records)

        # Try to add to primary
        try:
            self._primary.add(records_list)
        except Exception as exc:
            logger.warning(f"Failed to add to primary store: {exc}")

        # Try to add to secondary
        try:
            self._secondary.add(records_list)
        except Exception as exc:
            logger.warning(f"Failed to add to secondary store: {exc}")

        # Always add to fallback
        self._fallback.add(records_list)

    def query(self, vector: List[float], top_k: int = 5) -> List[VectorMatch]:
        """Query with automatic fallback on failure."""
        self._total_queries += 1

        # Try primary
        try:
            import signal

            # Set timeout for external service calls
            def timeout_handler(signum, frame):
                raise TimeoutError("Query timeout")

            # Note: signal.alarm only works on Unix, skip timeout on Windows
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self._fallback_timeout))

            try:
                results = self._primary.query(vector, top_k)
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel alarm
                return results
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)

        except Exception as exc:
            self._primary_failures += 1
            logger.warning(f"Primary store failed, trying secondary: {exc}")

            # Try secondary
            try:
                results = self._secondary.query(vector, top_k)
                logger.info("Secondary store succeeded")
                return results
            except Exception as exc2:
                self._secondary_failures += 1
                logger.warning(f"Secondary store failed, using fallback: {exc2}")

                # Use fallback
                results = self._fallback.query(vector, top_k)
                logger.info("Fallback store succeeded")
                return results

    def get_fallback_rate(self) -> Dict[str, float]:
        """Get fallback statistics."""
        if self._total_queries == 0:
            return {"primary_failure_rate": 0.0, "secondary_failure_rate": 0.0}

        return {
            "primary_failure_rate": self._primary_failures / self._total_queries,
            "secondary_failure_rate": self._secondary_failures / self._total_queries,
            "total_queries": self._total_queries
        }
