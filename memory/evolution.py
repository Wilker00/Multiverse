"""
memory/evolution.py

Combine datasets to produce a hybrid policy dataset.
"""

from __future__ import annotations

import json
from typing import Iterable, List


def merge_datasets(
    dataset_a: str,
    dataset_b: str,
    out_path: str,
    ratio_a: float = 0.5,
) -> str:
    """
    Merge two JSONL datasets by interleaving rows based on ratio_a.
    """
    with open(dataset_a, "r", encoding="utf-8") as fa, open(dataset_b, "r", encoding="utf-8") as fb:
        rows_a = [line.strip() for line in fa if line.strip()]
        rows_b = [line.strip() for line in fb if line.strip()]

    out: List[str] = []
    ia = 0
    ib = 0
    toggle_a = ratio_a >= 0.5

    while ia < len(rows_a) or ib < len(rows_b):
        if toggle_a and ia < len(rows_a):
            out.append(rows_a[ia])
            ia += 1
        elif ib < len(rows_b):
            out.append(rows_b[ib])
            ib += 1
        toggle_a = not toggle_a

    with open(out_path, "w", encoding="utf-8") as f:
        for line in out:
            f.write(line + "\n")
    return out_path
