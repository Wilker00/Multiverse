"""Render the default pygame warehouse viewer to a reviewable PNG."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame

from core.types import VerseSpec
from tools import watch_pygame as ui
from verses.warehouse_3d_world import Warehouse3DWorldVerse


def main() -> None:
    pygame.init()
    verse = Warehouse3DWorldVerse(
        VerseSpec(
            spec_version="1",
            verse_name="warehouse_3d_world",
            verse_version="1",
            seed=42,
            params={
                "camera_mode": "isometric",
                "render_backend": "auto",
                "render_width": 640,
                "render_height": 426,
            },
        )
    )
    reset = verse.reset()
    info = dict(reset.info or {})
    total_reward = 0.0
    action = 0
    for action in (1, 1, 4):
        result = verse.step(action)
        info = dict(result.info or {})
        total_reward += float(result.reward)

    width, height = 720, 640
    screen = pygame.Surface((width, height))
    screen.fill(ui.BG)
    header = pygame.Rect(0, 0, width, ui.HEADER_H)
    game = pygame.Rect(0, ui.HEADER_H, width, height - ui.HEADER_H - ui.FOOTER_H)
    footer = pygame.Rect(0, height - ui.FOOTER_H, width, ui.FOOTER_H)

    mono = pygame.font.SysFont("consolas", 14)
    label = pygame.font.SysFont("segoeui", 12)
    title = pygame.font.SysFont("segoeui", 18, bold=True)
    body = pygame.font.SysFont("segoeui", 14)

    frame = verse.render(mode="rgb_array")
    screen.blit(ui.rgb_to_surface(frame, game.width, game.height), game.topleft)
    ui.draw_warehouse_3d_hud(screen, verse, game, mono, label, 3, action, info)
    ui.draw_header(screen, header, "warehouse_3d_world", "q", 1, 0, False, False, 8, title, body)
    ui.draw_footer(
        screen,
        footer,
        3,
        80,
        ui.action_label("warehouse_3d_world", action),
        float(result.reward),
        total_reward,
        0.0,
        0,
        0,
        body,
        [],
        label,
    )

    output = ROOT / "docs" / "images" / "warehouse_3d_viewer.png"
    pygame.image.save(screen, output)
    verse.close()
    pygame.quit()
    print(f"wrote {output} {screen.get_size()}")


if __name__ == "__main__":
    main()
