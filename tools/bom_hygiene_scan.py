"""
tools/bom_hygiene_scan.py

Scan the repository for UTF-8 BOM-prefixed files and emit a JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


SKIP_PARTS = (".venv", "node_modules", ".git", "__pycache__", ".history", "runs")


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scan(root_dir: str) -> List[str]:
    hits: List[str] = []
    root_abs = os.path.abspath(root_dir)
    for walk_root, _, files in os.walk(root_abs):
        if any(part in walk_root for part in SKIP_PARTS):
            continue
        for name in files:
            path = os.path.join(walk_root, name)
            try:
                with open(path, "rb") as f:
                    sig = f.read(3)
                if sig == b"\xef\xbb\xbf":
                    hits.append(path.replace("\\", "/"))
            except Exception:
                continue
    hits.sort()
    return hits


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan repository for UTF-8 BOM-prefixed files.")
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--out_json", type=str, default=os.path.join("models", "validation", "bom_hygiene_scan_v1.json"))
    args = ap.parse_args()

    hits = _scan(args.root)
    out = {
        "created_at_iso": _iso_now(),
        "root": os.path.abspath(args.root).replace("\\", "/"),
        "skip_contains": list(SKIP_PARTS),
        "bom_file_count": int(len(hits)),
        "bom_files": hits,
    }
    _write_json(args.out_json, out)
    print(f"out_json={args.out_json.replace(chr(92), '/')}")
    print(f"bom_file_count={len(hits)}")


if __name__ == "__main__":
    main()
