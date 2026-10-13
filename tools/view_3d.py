"""Launch Multiverse's optional custom OpenGL scene viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import VerseSpec
from engine3d.opengl_viewer import view_verse
from verses.registry import create_verse, register_builtin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verse", required=True, choices=(
        "world_3d", "embodied_point_world", "robot_reacher_world", "warehouse_3d_world"
    ))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    register_builtin()
    spec = VerseSpec(
        spec_version="v1", verse_name=args.verse, verse_version="1", seed=args.seed,
        params={"adr_enabled": False},
    )
    verse = create_verse(spec)
    try:
        verse.reset()
        view_verse(verse)
    finally:
        verse.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
