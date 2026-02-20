"""
memory/boundary.py

Build and query a boundary set of "bad" states.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Set


def obs_key(obs: Any) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(obs)


def load_bad_obs(path: str) -> Set[str]:
    bad: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bad.add(obs_key(row.get("obs")))
    return bad
