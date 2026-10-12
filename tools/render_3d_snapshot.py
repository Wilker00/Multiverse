"""Render warehouse_3d perspective and isometric frames to docs/images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine3d.rasterizer import Rasterizer
from verses.warehouse_3d_world import build_warehouse_scene


def _write_png(path: Path, frame: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError:
        _write_png_stdlib(path, frame)
        return
    Image.fromarray(frame, mode="RGB").save(path)


def _write_png_stdlib(path: Path, frame: np.ndarray) -> None:
    import struct
    import zlib

    height, width, _ = frame.shape
    raw = b"".join(b"\x00" + frame[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    out_dir = ROOT / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    raster = Rasterizer(720, 480)
    views = {
        "warehouse_3d_perspective.png": "follow",
        "warehouse_3d_isometric.png": "isometric",
    }
    for name, mode in views.items():
        scene = build_warehouse_scene(1.15, -1.8, 0.15, mode)
        frame = raster.render(scene)
        path = out_dir / name
        _write_png(path, frame)
        print(f"wrote {path} {frame.shape}")


if __name__ == "__main__":
    main()
