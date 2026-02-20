"""
memory/vector_memory.py

Index and query experience as vectors for similarity search.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from core.types import JSONValue
from memory.encoders import get_encoder
from memory.vector_store import InMemoryVectorStore, VectorRecord, VectorMatch, VectorStore
from memory.selection import SelectionConfig, select_events


@dataclass
class VectorMemoryConfig:
    run_dir: str
    events_filename: str = "events.jsonl"
    obs_keys: Optional[List[str]] = None
    encoder: str = "raw"
    encoder_model: Optional[str] = None


def _iter_events(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def build_inmemory_index(cfg: VectorMemoryConfig, selection: SelectionConfig | None = None) -> InMemoryVectorStore:
    if not os.path.isdir(cfg.run_dir):
        raise FileNotFoundError(f"run_dir not found: {cfg.run_dir}")
    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"events file not found: {events_path}")

    events = list(_iter_events(events_path))
    if selection is not None:
        events = select_events(events, selection)
    store = InMemoryVectorStore()
    encoder = get_encoder(cfg.encoder, cfg.encoder_model)

    records: List[VectorRecord] = []
    for e in events:
        obs = e.get("obs")
        try:
            vec = encoder.encode(obs)
        except Exception:
            continue
        vector_id = f"{e.get('episode_id')}_{e.get('step_idx')}"
        records.append(
            VectorRecord(
                vector_id=str(vector_id),
                vector=vec,
                metadata={
                    "episode_id": e.get("episode_id"),
                    "step_idx": e.get("step_idx"),
                    "reward": e.get("reward"),
                    "verse_name": e.get("verse_name"),
                },
            )
        )

    store.add(records)
    return store


def query_memory(
    store: VectorStore,
    obs: JSONValue,
    top_k: int = 5,
    obs_keys: Optional[List[str]] = None,
    encoder: str = "raw",
    encoder_model: Optional[str] = None,
) -> List[VectorMatch]:
    enc = get_encoder(encoder, encoder_model)
    vec = enc.encode(obs)
    return store.query(vec, top_k=top_k)
