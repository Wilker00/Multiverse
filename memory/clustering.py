"""
memory/clustering.py

Simple clustering utilities for grouping interactions by task similarity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from memory.embeddings import obs_to_vector


@dataclass
class ClusterConfig:
    run_dir: str
    events_filename: str = "events.jsonl"
    output_filename: str = "clusters.jsonl"
    k: int = 5
    max_iters: int = 20


def _load_events(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def kmeans(X: np.ndarray, k: int, max_iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    idx = rng.choice(len(X), size=min(k, len(X)), replace=False)
    centroids = X[idx].copy()

    labels = np.zeros((len(X),), dtype=np.int64)
    for _ in range(max_iters):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update
        for i in range(len(centroids)):
            pts = X[labels == i]
            if len(pts) > 0:
                centroids[i] = np.mean(pts, axis=0)
    return labels, centroids


def cluster_events(cfg: ClusterConfig) -> str:
    if not os.path.isdir(cfg.run_dir):
        raise FileNotFoundError(f"run_dir not found: {cfg.run_dir}")
    events_path = os.path.join(cfg.run_dir, cfg.events_filename)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"events file not found: {events_path}")

    events = _load_events(events_path)
    vectors: List[List[float]] = []
    keep: List[Dict[str, Any]] = []

    for e in events:
        try:
            vec = obs_to_vector(e.get("obs"))
        except Exception:
            continue
        vectors.append(vec)
        keep.append(e)

    if not vectors:
        raise RuntimeError("No vectorizable observations found.")

    X = np.asarray(vectors, dtype=np.float32)
    labels, _ = kmeans(X, k=cfg.k, max_iters=cfg.max_iters)

    out_path = os.path.join(cfg.run_dir, cfg.output_filename)
    with open(out_path, "w", encoding="utf-8") as out:
        for e, label in zip(keep, labels):
            row = {
                "episode_id": e.get("episode_id"),
                "step_idx": e.get("step_idx"),
                "cluster_id": int(label),
                "verse_name": e.get("verse_name"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path
