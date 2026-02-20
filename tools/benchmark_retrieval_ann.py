"""
tools/benchmark_retrieval_ann.py

Benchmark approximate-nearest-neighbor (ANN) retrieval versus exact scan in the
central memory repository and emit a JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from memory.central_repository import CentralMemoryConfig, NearestNeighbors, find_similar


def _safe_mkdir_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _safe_mkdir_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _build_memory(rows: int, value_max: int, seed: int) -> str:
    rng = random.Random(int(seed))
    td = tempfile.mkdtemp(prefix="ann_bench_")
    mem_path = os.path.join(td, "memories.jsonl")
    with open(mem_path, "w", encoding="utf-8") as f:
        for i in range(max(1, int(rows))):
            x = rng.randint(0, max(1, int(value_max)))
            y = rng.randint(0, max(1, int(value_max)))
            row = {
                "run_id": f"run_{i}",
                "episode_id": f"ep_{i}",
                "step_idx": i,
                "t_ms": i + 1,
                "verse_name": "grid_world",
                "obs": {"x": x, "y": y},
                "obs_vector": [float(x), float(y)],
                "action": 0,
                "reward": 0.0,
                "memory_tier": "stm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return td


def _run_pass(cfg: CentralMemoryConfig, queries: int, value_max: int, seed: int, ann_enabled: bool, top_k: int) -> float:
    rng = random.Random(int(seed))
    q = [{"x": rng.randint(0, value_max), "y": rng.randint(0, value_max)} for _ in range(max(1, int(queries)))]
    old_ann = os.environ.get("MULTIVERSE_SIM_USE_ANN")
    try:
        os.environ["MULTIVERSE_SIM_USE_ANN"] = "1" if ann_enabled else "0"
        start = time.perf_counter()
        for obs in q:
            find_similar(obs=obs, cfg=cfg, top_k=max(1, int(top_k)), min_score=-1.0)
        return float(time.perf_counter() - start)
    finally:
        if old_ann is None:
            os.environ.pop("MULTIVERSE_SIM_USE_ANN", None)
        else:
            os.environ["MULTIVERSE_SIM_USE_ANN"] = old_ann


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark ANN retrieval speedup versus exact retrieval.")
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--value_max", type=int, default=1999)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_json", type=str, default=os.path.join("models", "validation", "retrieval_ann_benchmark_v1.json"))
    args = ap.parse_args()

    if NearestNeighbors is None:
        raise RuntimeError("scikit-learn is required for ANN benchmarking (NearestNeighbors unavailable).")

    td = _build_memory(rows=int(args.rows), value_max=int(args.value_max), seed=int(args.seed))
    cfg = CentralMemoryConfig(root_dir=td)

    # Warm cache once so both passes share the same loaded-memory path.
    find_similar(obs={"x": int(args.value_max) // 2, "y": int(args.value_max) // 2}, cfg=cfg, top_k=max(1, int(args.top_k)), min_score=-1.0)

    exact_s = _run_pass(
        cfg=cfg,
        queries=int(args.queries),
        value_max=int(args.value_max),
        seed=int(args.seed) + 17,
        ann_enabled=False,
        top_k=int(args.top_k),
    )
    ann_s = _run_pass(
        cfg=cfg,
        queries=int(args.queries),
        value_max=int(args.value_max),
        seed=int(args.seed) + 17,
        ann_enabled=True,
        top_k=int(args.top_k),
    )
    speedup = float(exact_s / max(ann_s, 1e-12))
    out = {
        "created_at_iso": _iso_now(),
        "config": {
            "rows": int(args.rows),
            "queries": int(args.queries),
            "value_max": int(args.value_max),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
        },
        "results": {
            "exact_seconds": float(exact_s),
            "ann_seconds": float(ann_s),
            "speedup_exact_over_ann": speedup,
        },
        "interpretation": {
            "ann_faster": bool(speedup > 1.0),
            "meets_66x": bool(speedup >= 66.0),
        },
    }
    _write_json(args.out_json, out)
    print(f"out_json={args.out_json.replace(chr(92), '/')}")
    print(f"speedup_exact_over_ann={speedup:.3f}x")


if __name__ == "__main__":
    main()
