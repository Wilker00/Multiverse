"""
memory/splits.py

Deterministic run-level split assignment for train/val/test workflows.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 123

    def normalize(self) -> "SplitConfig":
        total = float(self.train_ratio + self.val_ratio + self.test_ratio)
        if total <= 0.0:
            raise ValueError("split ratios must sum to > 0")
        return SplitConfig(
            train_ratio=self.train_ratio / total,
            val_ratio=self.val_ratio / total,
            test_ratio=self.test_ratio / total,
            seed=self.seed,
        )


def _hash01(key: str) -> float:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    # 64-bit chunk -> [0,1)
    n = int(h[:16], 16)
    return n / float(2**64)


def assign_split(run_id: str, cfg: SplitConfig) -> str:
    cfg = cfg.normalize()
    x = _hash01(f"{cfg.seed}:{run_id}")
    t = cfg.train_ratio
    v = cfg.train_ratio + cfg.val_ratio
    if x < t:
        return "train"
    if x < v:
        return "val"
    return "test"


def split_run_dirs(run_dirs: Iterable[str], cfg: SplitConfig) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for rd in run_dirs:
        run_id = os.path.basename(os.path.normpath(str(rd)))
        split = assign_split(run_id, cfg)
        out[split].append(str(rd))
    for k in out:
        out[k].sort()
    return out


def save_split_manifest(path: str, run_dirs: Iterable[str], cfg: SplitConfig) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rd in sorted(str(x) for x in run_dirs):
            run_id = os.path.basename(os.path.normpath(rd))
            row = {
                "run_id": run_id,
                "run_dir": rd,
                "split": assign_split(run_id, cfg),
                "seed": int(cfg.seed),
                "train_ratio": float(cfg.train_ratio),
                "val_ratio": float(cfg.val_ratio),
                "test_ratio": float(cfg.test_ratio),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
