"""
tools/watch_pygame.py

Gymnasium-style graphical viewer for any Multiverse verse.
Opens a pygame window and shows the verse being trained in real time —
color-coded tile grid for spatial worlds, stats dashboard for abstract worlds.
This remains available directly, but `tools/watch.py --frontend pygame` is the
canonical watch entrypoint.

Usage:
    py tools/watch_pygame.py                                       # maze_world defaults
    py tools/watch_pygame.py --verse grid_world  --algo q --train
    py tools/watch_pygame.py --verse maze_world  --algo q --train --fps 8
    py tools/watch_pygame.py --verse cliff_world --algo q --train --fps 10
    py tools/watch_pygame.py --verse chess_world --algo q --train --fps 5
    py tools/watch_pygame.py --verse warehouse_world --algo q --train
    py tools/watch_pygame.py --verse line_world  --algo ppo --train

Keyboard controls (while running):
    SPACE         Pause / Resume
    UP / DOWN     Speed up / slow down (FPS +/-2)
    ESC / Q       Quit

Options:
    --verse       Verse name (default: maze_world)
    --algo        Agent algorithm (default: q)
    --train       Enable online learning
    --episodes    Episodes to watch (default: 0 = infinite)
    --fps         Target steps per second (default: 8)
    --seed        Random seed (default: 42)
    --vparam K=V  Verse param (repeatable)
    --aconfig K=V Agent config (repeatable)
    --width  N    Window width  (default: 720)
    --height N    Window height (default: 640)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── path bootstrap ─────────────────────────────────────────────────────────────
if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

# ── UTF-8 on Windows ──────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

try:
    import pygame
except ImportError:
    # --help only needs argparse. A missing viewer dependency should not
    # make the CI help gate fail before usage can be printed.
    pygame = None  # type: ignore[assignment]

# ── Registries ────────────────────────────────────────────────────────────────
from verses.registry import register_builtin as _reg_verses, create_verse, list_verses
from agents.registry import register_builtin_agents as _reg_agents, create_agent, _AGENT_REGISTRY
from core.types import VerseSpec, AgentSpec
from core.agent_base import ExperienceBatch, Transition

_reg_verses()
_reg_agents()


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

BG          = (12,  12,  28)
HEADER_BG   = (8,   16,  40)
FOOTER_BG   = (8,   12,  32)
GRID_BG     = (20,  20,  44)

C_WALL      = (50,  52,  68)    # walls / obstacles
C_OPEN      = (28,  28,  52)    # empty floor
C_AGENT     = (30,  100, 255)   # agent
C_GOAL      = (20,  200, 80)    # goal / exit
C_VISITED   = (36,  36,  70)    # visited floor
C_HAZARD    = (230, 70,  20)    # hazard / cliff
C_ICE       = (60,  180, 230)   # ice / water
C_TELEPORT  = (160, 60,  240)   # teleporter
C_CHARGER   = (220, 180, 30)    # charger / resource
C_BRIDGE    = (140, 100, 50)    # bridge tiles
C_SWAMP     = (60,  100, 40)    # swamp / mud
C_FOG       = (18,  18,  40)    # fog of war
C_BORDER    = (0,   80,  180)   # tile border (subtle)

C_TEXT      = (210, 215, 240)
C_DIM       = (80,  85,  110)
C_ACCENT    = (0,   180, 255)
C_GREEN     = (0,   210, 90)
C_RED       = (220, 60,  60)
C_YELLOW    = (240, 190, 50)
C_PURPLE    = (180, 80,  255)

# Map ASCII characters → tile fill colour
# (bg_color, text_color)  — text_color used when drawing the char label
CHAR_STYLE: Dict[str, Tuple[Tuple, Tuple]] = {
    "A": (C_AGENT,    (180, 220, 255)),
    "@": (C_AGENT,    (180, 220, 255)),
    "G": (C_GOAL,     (180, 255, 200)),
    "E": (C_GOAL,     (180, 255, 200)),
    "#": (C_WALL,     (80,  82,  100)),
    "+": (C_WALL,     (70,  72,  90)),
    "-": (C_WALL,     (70,  72,  90)),
    "|": (C_WALL,     (70,  72,  90)),
    ".": (C_VISITED,  (60,  65,  100)),
    " ": (C_OPEN,     C_OPEN),
    "~": (C_ICE,      (100, 210, 255)),
    "I": (C_ICE,      (100, 210, 255)),
    "T": (C_TELEPORT, (200, 140, 255)),
    "C": (C_CHARGER,  (255, 230, 120)),
    "X": (C_HAZARD,   (255, 130, 80)),
    "?": (C_FOG,      (40,  45,  80)),
    "W": (C_ICE,      (80,  180, 230)),
    "S": (C_SWAMP,    (100, 160, 60)),
    "B": (C_BRIDGE,   (180, 140, 80)),
    "P": (C_GOAL,     (140, 240, 160)),
    "F": (C_HAZARD,   (255, 160, 60)),   # fire / flood
    "M": (C_SWAMP,    (120, 80,  40)),   # mud
    "*": (C_CHARGER,  (255, 240, 100)),  # resource / star
}


def char_style(ch: str) -> Tuple[Tuple, Tuple]:
    if ch in CHAR_STYLE:
        return CHAR_STYLE[ch]
    if ch.isdigit() or ch.isalpha():
        return (C_OPEN, C_TEXT)
    return (C_WALL, C_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Font helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int) -> pygame.font.Font:
    candidates = [
        "Courier New", "Lucida Console", "Consolas", "DejaVu Sans Mono",
        "Cascadia Code", "Monospace",
    ]
    for name in candidates:
        try:
            f = pygame.font.SysFont(name, size, bold=False)
            if f is not None:
                return f
        except Exception:
            pass
    return pygame.font.Font(None, size)


def _load_bold_font(size: int) -> pygame.font.Font:
    candidates = ["Orbitron", "Courier New", "Consolas", "DejaVu Sans Mono"]
    for name in candidates:
        try:
            f = pygame.font.SysFont(name, size, bold=True)
            if f is not None:
                return f
        except Exception:
            pass
    return pygame.font.Font(None, size)


# ─────────────────────────────────────────────────────────────────────────────
# Frame parsing — ANSI text → grid of (char, bg, fg)
# ─────────────────────────────────────────────────────────────────────────────

def parse_ansi_frame(frame: str) -> List[List[Tuple[str, Tuple, Tuple]]]:
    """
    Convert ANSI-rendered text to a 2-D list of (char, bg_color, fg_color).
    Strips ANSI escape codes and applies our colour palette based on char type.
    """
    import re
    # Remove escape sequences
    clean = re.sub(r"\x1b\[[0-9;]*[mJHfABCDEFGKST]", "", frame)
    clean = re.sub(r"\x1b\[48;5;\d+m", "", clean)
    rows = clean.split("\n")

    grid: List[List[Tuple[str, Tuple, Tuple]]] = []
    for row in rows:
        cells = []
        for ch in row:
            bg, fg = char_style(ch)
            cells.append((ch, bg, fg))
        grid.append(cells)

    # Trim trailing empty rows
    while grid and all(c[0] == " " or c[0] == "" for c in grid[-1]):
        grid.pop()
    return grid


def grid_dims(grid: List[List[Any]]) -> Tuple[int, int]:
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0)
    return rows, cols


# ─────────────────────────────────────────────────────────────────────────────
# RGB-array rendering (upscale pixel grid)
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_surface(
    rgb: Any,
    target_w: int,
    target_h: int,
) -> pygame.Surface:
    """
    Convert a H×W×3 nested-list or numpy array pixel frame to a pygame Surface scaled to target_w×target_h.
    Uses surfarray if numpy array is provided for high performance.
    """
    if hasattr(rgb, "shape") and len(rgb.shape) == 3:
        try:
            import numpy as np
            transposed = np.transpose(rgb, (1, 0, 2))
            raw = pygame.surfarray.make_surface(transposed)
        except Exception:
            raw = None
    else:
        raw = None

    if raw is None:
        h = len(rgb)
        w = len(rgb[0]) if h > 0 else 1
        raw = pygame.Surface((w, h))
        for y, row in enumerate(rgb):
            for x, pixel in enumerate(row):
                r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
                raw.set_at((x, y), (r, g, b))

    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    source_w, source_h = raw.get_size()
    # Fill the viewport while preserving the source aspect ratio. A small
    # center crop is preferable to leaving empty bands around 3D scenes.
    scale = max(target_w / max(1, source_w), target_h / max(1, source_h))
    out_w = max(1, int(source_w * scale))
    out_h = max(1, int(source_h * scale))
    scaled = pygame.transform.smoothscale(raw, (out_w, out_h))
    crop_x = max(0, (out_w - target_w) // 2)
    crop_y = max(0, (out_h - target_h) // 2)
    canvas = pygame.Surface((target_w, target_h))
    canvas.blit(scaled, (0, 0), pygame.Rect(crop_x, crop_y, target_w, target_h))
    return canvas


def draw_warehouse_3d_hud(
    surface: pygame.Surface,
    verse: Any,
    area: pygame.Rect,
    mono_font: pygame.font.Font,
    label_font: pygame.font.Font,
    step: int,
    action: Any,
    info: Optional[Dict[str, Any]] = None,
) -> None:
    """Show compact robot telemetry and a top-down warehouse map."""
    if area.width < 480 or area.height < 300:
        return

    from verses.warehouse_3d_world import CRATES

    info = info or {}
    x = float(getattr(verse, "x", 0.0))
    z = float(getattr(verse, "z", 0.0))
    yaw = float(getattr(verse, "yaw", 0.0))
    goal_x = float(getattr(verse, "_goal_x", 0.0))
    goal_z = float(getattr(verse, "_goal_z", 0.0))
    distance = math.hypot(goal_x - x, goal_z - z)
    goal_radius = float(getattr(verse, "_goal_radius", 0.45))
    arena = max(1.0, float(getattr(verse, "_arena", 5.5)))

    # Keep telemetry over the open lower-left floor so the crates and robot stay visible.
    panel_w, panel_h = 266, 142
    panel_x, panel_y = area.x + 14, area.bottom - panel_h - 14
    panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
    panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
    panel.fill((9, 17, 31, 224))
    surface.blit(panel, panel_rect.topleft)
    pygame.draw.rect(surface, (70, 94, 110), panel_rect, 1, border_radius=8)
    pygame.draw.line(surface, C_ACCENT, (panel_x + 1, panel_y + 1), (panel_x + panel_w - 2, panel_y + 1), 2)

    title = label_font.render("WAREHOUSE NAVIGATION", True, C_ACCENT)
    surface.blit(title, (panel_x + 12, panel_y + 9))
    dist_color = C_GREEN if distance <= goal_radius else C_YELLOW if distance <= 2.0 else C_TEXT
    rows = (
        (f"POS   X {x:+.2f}   Z {z:+.2f}", C_TEXT),
        (f"HEAD  {math.degrees(yaw):+.1f} deg", C_YELLOW),
        (f"ACT   {action_label('warehouse_3d_world', action)}", C_YELLOW),
        (f"GOAL  X {goal_x:+.2f}   Z {goal_z:+.2f}", C_TEXT),
        (f"DIST  {distance:.2f} m", dist_color),
    )
    for index, (line, color) in enumerate(rows):
        surface.blit(mono_font.render(line, True, color), (panel_x + 12, panel_y + 30 + index * 19))

    status = "COLLISION" if info.get("collided") else "GOAL REACHED" if info.get("reached_goal") else "NAVIGATING"
    status_color = C_RED if info.get("collided") else C_GREEN if info.get("reached_goal") else (178, 195, 210)
    surface.blit(label_font.render(f"{status}   STEP {step:03d}", True, status_color), (panel_x + 12, panel_y + panel_h - 19))

    map_size = min(156, area.height - 28, area.width - panel_w - 42)
    if map_size < 118:
        return
    map_x, map_y = area.right - map_size - 14, area.bottom - map_size - 14
    map_rect = pygame.Rect(map_x, map_y, map_size, map_size)
    map_bg = pygame.Surface(map_rect.size, pygame.SRCALPHA)
    map_bg.fill((9, 17, 31, 224))
    surface.blit(map_bg, map_rect.topleft)
    pygame.draw.rect(surface, (70, 94, 110), map_rect, 1, border_radius=8)
    surface.blit(label_font.render("TOP VIEW", True, C_ACCENT), (map_x + 10, map_y + 8))
    inner = pygame.Rect(map_x + 10, map_y + 28, map_size - 20, map_size - 39)
    pygame.draw.rect(surface, (27, 39, 49), inner, border_radius=4)
    for division in range(1, 4):
        gx = inner.x + inner.width * division // 4
        gy = inner.y + inner.height * division // 4
        pygame.draw.line(surface, (48, 67, 78), (gx, inner.y), (gx, inner.bottom), 1)
        pygame.draw.line(surface, (48, 67, 78), (inner.x, gy), (inner.right, gy), 1)

    def map_point(world_x: float, world_z: float) -> Tuple[int, int]:
        px = inner.x + round((world_x + arena) / (2.0 * arena) * inner.width)
        py = inner.bottom - round((world_z + arena) / (2.0 * arena) * inner.height)
        return px, py

    for _name, cx, cz, width, _height, depth, color in getattr(verse, "_crates", CRATES):
        px, py = map_point(float(cx), float(cz))
        rw = max(3, round(float(width) / (2.0 * arena) * inner.width))
        rh = max(3, round(float(depth) / (2.0 * arena) * inner.height))
        rect = pygame.Rect(px - rw // 2, py - rh // 2, rw, rh)
        pygame.draw.rect(surface, color, rect, border_radius=2)

    goal_point = map_point(goal_x, goal_z)
    pygame.draw.circle(surface, C_GREEN, goal_point, 5)
    robot_point = map_point(x, z)
    heading_point = map_point(x + math.sin(yaw) * 0.7, z + math.cos(yaw) * 0.7)
    pygame.draw.line(surface, C_YELLOW, robot_point, heading_point, 2)
    pygame.draw.circle(surface, (255, 180, 69), robot_point, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Text-grid rendering (char-cell display)
# ─────────────────────────────────────────────────────────────────────────────

def render_text_grid(
    surface: pygame.Surface,
    grid: List[List[Tuple[str, Tuple, Tuple]]],
    area: pygame.Rect,
    mono_font: pygame.font.Font,
    cell_w: int,
    cell_h: int,
) -> None:
    """Draw the character-cell grid into the given area rect."""
    # Background fill
    surface.fill(GRID_BG, area)

    n_rows, n_cols = grid_dims(grid)
    if n_rows == 0 or n_cols == 0:
        return

    # Centre the grid inside area
    total_w = n_cols * cell_w
    total_h = n_rows * cell_h
    ox = area.x + max(0, (area.width  - total_w) // 2)
    oy = area.y + max(0, (area.height - total_h) // 2)

    for r, row in enumerate(grid):
        for c, (ch, bg, fg) in enumerate(row):
            x = ox + c * cell_w
            y = oy + r * cell_h
            rect = pygame.Rect(x, y, cell_w, cell_h)
            # Fill tile
            pygame.draw.rect(surface, bg, rect)
            # Subtle border for open tiles
            if bg == C_OPEN or bg == C_VISITED:
                pygame.draw.rect(surface, C_BORDER, rect, 1)
            # Draw char (skip space / border chars for clean look)
            if ch not in (" ", "+", "-", "|") and ch.strip():
                glyph = mono_font.render(ch, True, fg)
                gx = x + (cell_w - glyph.get_width())  // 2
                gy = y + (cell_h - glyph.get_height()) // 2
                surface.blit(glyph, (gx, gy))


# ─────────────────────────────────────────────────────────────────────────────
# Abstract-verse stats panel (chess, go, uno, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def render_stats_panel(
    surface: pygame.Surface,
    obs: Any,
    area: pygame.Rect,
    label_font: pygame.font.Font,
    val_font: pygame.font.Font,
    verse_name: str,
) -> None:
    """
    Show obs dict as a grid of metric boxes — used for abstract/non-spatial verses.
    """
    surface.fill(BG, area)

    if not isinstance(obs, dict) or not obs:
        msg = label_font.render(f"{verse_name}: no visual", True, C_DIM)
        surface.blit(msg, (area.x + 20, area.y + area.height // 2))
        return

    items = list(obs.items())
    n = len(items)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    pad = 12
    box_w = (area.width  - pad * (cols + 1)) // cols
    box_h = (area.height - pad * (rows + 1)) // rows

    for i, (key, val) in enumerate(items):
        col = i % cols
        row = i // cols
        x = area.x + pad + col * (box_w + pad)
        y = area.y + pad + row * (box_h + pad)
        box = pygame.Rect(x, y, box_w, box_h)

        # Color box by value sentiment
        try:
            v = float(val)
            fill = (20, 60, 30) if v > 0 else (60, 20, 20) if v < 0 else (20, 20, 50)
            border = C_GREEN if v > 0 else C_RED if v < 0 else C_DIM
        except (TypeError, ValueError):
            fill = (20, 20, 50)
            border = C_ACCENT

        pygame.draw.rect(surface, fill, box, border_radius=8)
        pygame.draw.rect(surface, border, box, 2, border_radius=8)

        # Key label
        k_surf = label_font.render(str(key), True, C_DIM)
        surface.blit(k_surf, (x + 8, y + 8))

        # Value
        val_str = f"{float(val):+.2f}" if isinstance(val, (int, float)) else str(val)
        v_surf = val_font.render(val_str, True, border)
        vx = x + box_w // 2 - v_surf.get_width() // 2
        vy = y + box_h // 2 - v_surf.get_height() // 2 + 6
        surface.blit(v_surf, (vx, vy))



# ─────────────────────────────────────────────────────────────────────────────
# Embodied physics & robotics vector renderers
# ─────────────────────────────────────────────────────────────────────────────

def render_robot_reacher(
    surface: pygame.Surface,
    verse: Any,
    area: pygame.Rect,
    mono_font: pygame.font.Font,
    label_font: pygame.font.Font,
) -> None:
    surface.fill((12, 14, 26), area)

    cx = area.centerx
    cy = area.centery
    scale = min(area.width, area.height) * 0.40

    # Concentric range indicators
    grid_col = (22, 28, 48)
    for r in [0.25, 0.5, 0.75, 1.0]:
        r_px = int(r * scale)
        pygame.draw.circle(surface, grid_col, (cx, cy), r_px, 1)
    pygame.draw.line(surface, grid_col, (cx - int(scale), cy), (cx + int(scale), cy), 1)
    pygame.draw.line(surface, grid_col, (cx, cy - int(scale)), (cx, cy + int(scale)), 1)

    def to_screen(x: float, y: float) -> Tuple[int, int]:
        return (int(cx + x * scale), int(cy - y * scale))

    # Workspace reach boundary (l1 + l2)
    max_reach = getattr(verse, "l1", 0.5) + getattr(verse, "l2", 0.5)
    pygame.draw.circle(surface, (45, 60, 95), (cx, cy), int(max_reach * scale), 2)

    # Target point
    tx = getattr(verse, "target_x", 0.5)
    ty = getattr(verse, "target_y", 0.5)
    t_rad = max(4, int(getattr(verse, "target_radius", 0.08) * scale))
    t_px, t_py = to_screen(tx, ty)

    # Glowing target aura
    aura_size = max(16, t_rad * 4)
    aura_surf = pygame.Surface((aura_size, aura_size), pygame.SRCALPHA)
    pygame.draw.circle(aura_surf, (0, 230, 100, 45), (aura_size // 2, aura_size // 2), aura_size // 2)
    pygame.draw.circle(aura_surf, (0, 230, 100, 120), (aura_size // 2, aura_size // 2), t_rad)
    surface.blit(aura_surf, (t_px - aura_size // 2, t_py - aura_size // 2))
    pygame.draw.circle(surface, (0, 255, 120), (t_px, t_py), max(3, t_rad // 3))

    # Motion ribbon trail
    trail = getattr(verse, "_trail", [])
    if len(trail) > 1:
        points = [to_screen(px, py) for px, py in trail]
        pygame.draw.lines(surface, (0, 160, 240), False, points, 2)

    # Arm Kinematics
    elbow = getattr(verse, "elbow", (0.35, 0.35))
    ee = getattr(verse, "end_effector", (0.5, 0.5))
    b_pt = (cx, cy)
    e_pt = to_screen(*elbow)
    ee_pt = to_screen(*ee)

    # Link 1 (metallic cyan)
    pygame.draw.line(surface, (60, 160, 240), b_pt, e_pt, 8)
    pygame.draw.line(surface, (150, 215, 255), b_pt, e_pt, 2)

    # Link 2 (metallic orange)
    pygame.draw.line(surface, (255, 140, 40), e_pt, ee_pt, 6)
    pygame.draw.line(surface, (255, 200, 120), e_pt, ee_pt, 2)

    # Base pedestal
    pygame.draw.circle(surface, (140, 150, 180), b_pt, 12)
    pygame.draw.circle(surface, (220, 225, 240), b_pt, 5)

    # Elbow joint
    pygame.draw.circle(surface, (40, 120, 200), e_pt, 9)
    pygame.draw.circle(surface, (180, 230, 255), e_pt, 4)

    # End-effector gripper
    pygame.draw.circle(surface, (255, 220, 60), ee_pt, 8)
    pygame.draw.circle(surface, (255, 255, 200), ee_pt, 3)

    # HUD Panel Overlay
    th1 = getattr(verse, "theta1", 0.0)
    th2 = getattr(verse, "theta2", 0.0)
    dist = math.hypot(ee[0] - tx, ee[1] - ty)
    hud_bg = pygame.Surface((220, 96), pygame.SRCALPHA)
    hud_bg.fill((8, 12, 24, 180))
    surface.blit(hud_bg, (area.x + 14, area.y + 14))
    pygame.draw.rect(surface, (0, 180, 255), pygame.Rect(area.x + 14, area.y + 14, 220, 96), 1, border_radius=4)

    c_dist = (0, 230, 100) if dist <= 0.08 else (240, 190, 50) if dist <= 0.25 else (240, 80, 80)
    l1 = mono_font.render(f"Joint 1 : {math.degrees(th1):+6.1f}\u00b0", True, (180, 220, 255))
    l2 = mono_font.render(f"Joint 2 : {math.degrees(th2):+6.1f}\u00b0", True, (255, 180, 100))
    l3 = mono_font.render(f"Distance: {dist:6.3f} m", True, c_dist)
    l4 = label_font.render(f"EE: ({ee[0]:+.2f}, {ee[1]:+.2f})", True, (140, 150, 180))
    surface.blit(l1, (area.x + 22, area.y + 20))
    surface.blit(l2, (area.x + 22, area.y + 40))
    surface.blit(l3, (area.x + 22, area.y + 60))
    surface.blit(l4, (area.x + 22, area.y + 82))


def render_embodied_point(
    surface: pygame.Surface,
    verse: Any,
    area: pygame.Rect,
    mono_font: pygame.font.Font,
    label_font: pygame.font.Font,
) -> None:
    surface.fill((10, 12, 22), area)

    cx = area.centerx
    cy = area.centery
    scale = min(area.width, area.height) * 0.38

    def to_screen(x: float, y: float) -> Tuple[int, int]:
        return (int(cx + x * scale), int(cy - y * scale))

    # Arena bounding box [-1, 1]
    p_tl = to_screen(-1.0, 1.0)
    p_br = to_screen(1.0, -1.0)
    box_rect = pygame.Rect(p_tl[0], p_tl[1], p_br[0] - p_tl[0], p_br[1] - p_tl[1])
    pygame.draw.rect(surface, (18, 22, 38), box_rect)
    pygame.draw.rect(surface, (0, 140, 220), box_rect, 2, border_radius=6)

    # Grid lines inside arena
    grid_col = (24, 30, 52)
    for g in [-0.5, 0.0, 0.5]:
        p0 = to_screen(g, -1.0)
        p1 = to_screen(g, 1.0)
        pygame.draw.line(surface, grid_col, p0, p1, 1)
        p2 = to_screen(-1.0, g)
        p3 = to_screen(1.0, g)
        pygame.draw.line(surface, grid_col, p2, p3, 1)

    # Goal beacon
    gx, gy = getattr(verse, "goal", (0.8, 0.8))
    g_rad = max(4, int(getattr(verse, "_goal_radius", 0.15) * scale))
    g_px, g_py = to_screen(gx, gy)

    # Goal glowing aura
    g_size = max(16, g_rad * 4)
    g_aura = pygame.Surface((g_size, g_size), pygame.SRCALPHA)
    pygame.draw.circle(g_aura, (20, 200, 80, 50), (g_size // 2, g_size // 2), g_size // 2)
    pygame.draw.circle(g_aura, (20, 200, 80, 140), (g_size // 2, g_size // 2), g_rad)
    surface.blit(g_aura, (g_px - g_size // 2, g_py - g_size // 2))
    pygame.draw.circle(surface, (40, 240, 100), (g_px, g_py), 4)

    # Trail
    trail = getattr(verse, "trail", [])
    if len(trail) > 1:
        points = [to_screen(px, py) for px, py in trail]
        pygame.draw.lines(surface, (0, 150, 240), False, points, 2)

    # Rover / particle
    rx, ry = getattr(verse, "position", (0.0, 0.0))
    vx, vy = getattr(verse, "velocity", (0.0, 0.0))
    r_px, r_py = to_screen(rx, ry)

    # 8 LIDAR Rangefinder sensor beams
    beam_col = (0, 200, 255, 35)
    beam_surf = pygame.Surface((area.width, area.height), pygame.SRCALPHA)
    for angle_deg in range(0, 360, 45):
        rad = math.radians(angle_deg)
        ray_len = 0.4
        end_x = rx + ray_len * math.cos(rad)
        end_y = ry + ray_len * math.sin(rad)
        ep = to_screen(end_x, end_y)
        pygame.draw.line(beam_surf, beam_col, (r_px - area.x, r_py - area.y), (ep[0] - area.x, ep[1] - area.y), 1)
    surface.blit(beam_surf, (area.x, area.y))

    # Velocity vector
    speed = math.hypot(vx, vy)
    if speed > 1e-3:
        v_arrow_x = r_px + int(vx * scale * 0.4)
        v_arrow_y = r_py - int(vy * scale * 0.4)
        pygame.draw.line(surface, (255, 180, 40), (r_px, r_py), (v_arrow_x, v_arrow_y), 3)

    # Rover chassis
    pygame.draw.circle(surface, (255, 210, 40), (r_px, r_py), 9)
    pygame.draw.circle(surface, (255, 255, 200), (r_px, r_py), 4)

    # HUD Panel Overlay
    dist = math.hypot(rx - gx, ry - gy)
    hud_bg = pygame.Surface((220, 96), pygame.SRCALPHA)
    hud_bg.fill((8, 12, 24, 180))
    surface.blit(hud_bg, (area.x + 14, area.y + 14))
    pygame.draw.rect(surface, (0, 180, 255), pygame.Rect(area.x + 14, area.y + 14, 220, 96), 1, border_radius=4)

    c_dist = (0, 230, 100) if dist <= getattr(verse, "_goal_radius", 0.15) else (240, 190, 50) if dist <= 0.4 else (240, 80, 80)
    l1 = mono_font.render(f"Pos : ({rx:+.2f}, {ry:+.2f})", True, (180, 220, 255))
    l2 = mono_font.render(f"Vel : ({vx:+.2f}, {vy:+.2f})", True, (255, 180, 100))
    l3 = mono_font.render(f"Distance: {dist:6.3f} m", True, c_dist)
    l4 = label_font.render(f"Goal: ({gx:+.2f}, {gy:+.2f})", True, (140, 150, 180))
    surface.blit(l1, (area.x + 22, area.y + 20))
    surface.blit(l2, (area.x + 22, area.y + 40))
    surface.blit(l3, (area.x + 22, area.y + 60))
    surface.blit(l4, (area.x + 22, area.y + 82))


# ─────────────────────────────────────────────────────────────────────────────
# Header & footer bars
# ─────────────────────────────────────────────────────────────────────────────

def draw_header(
    surface: pygame.Surface,
    rect: pygame.Rect,
    verse_name: str,
    algo: str,
    episode: int,
    total_episodes: int,
    training: bool,
    paused: bool,
    fps: float,
    title_font: pygame.font.Font,
    info_font: pygame.font.Font,
) -> None:
    surface.fill(HEADER_BG, rect)
    # Decorative top accent line
    pygame.draw.line(surface, C_ACCENT, (rect.x, rect.y), (rect.right, rect.y), 2)

    ep_str = f"ep {episode}" if total_episodes == 0 else f"ep {episode} / {total_episodes}"
    title = title_font.render(f"  {verse_name.upper()}", True, C_ACCENT)
    surface.blit(title, (rect.x + 4, rect.y + (rect.height - title.get_height()) // 2))

    mode_col  = C_PURPLE if training  else C_DIM
    pause_col = C_YELLOW if paused    else C_GREEN
    mode_str  = "TRAINING"  if training else "WATCHING"
    state_str = "PAUSED"    if paused  else f"{fps:.0f} FPS"

    right_parts = [
        (ep_str,    C_TEXT),
        (f"  |  {algo.upper()}", C_DIM),
        (f"  |  {mode_str}", mode_col),
        (f"  |  {state_str}", pause_col),
    ]
    rx = rect.right - 8
    for text, color in reversed(right_parts):
        surf = info_font.render(text, True, color)
        rx -= surf.get_width()
        surface.blit(surf, (rx, rect.y + (rect.height - surf.get_height()) // 2))


def draw_footer(
    surface: pygame.Surface,
    rect: pygame.Rect,
    step: int,
    max_steps: int,
    action_label: str,
    reward: float,
    ep_return: float,
    best_return: float,
    wins: int,
    episodes_done: int,
    info_font: pygame.font.Font,
    episode_log: List[str],
    log_font: pygame.font.Font,
) -> None:
    surface.fill(FOOTER_BG, rect)
    pygame.draw.line(surface, C_ACCENT, (rect.x, rect.y), (rect.right, rect.y), 1)

    sr = wins / episodes_done if episodes_done > 0 else 0.0

    # ── Progress bar (step) ────────────────────────────────────────────────
    bar_h = 3
    bar_y = rect.y + rect.height - bar_h
    bar_w = int(rect.width * (step / max_steps)) if max_steps > 0 else 0
    pygame.draw.rect(surface, C_ACCENT, (rect.x, bar_y, bar_w, bar_h))

    # ── Stat chips ────────────────────────────────────────────────────────
    rew_col = C_GREEN if reward > 0 else C_RED if reward < 0 else C_DIM
    ret_col = C_GREEN if ep_return > 0 else C_RED if ep_return < 0 else C_DIM
    sr_col  = C_GREEN if sr > 0.3     else C_RED

    chips = [
        (f" step {step:>3}/{max_steps} ", C_DIM),
        (f"  act: {action_label:<10}", C_YELLOW),
        (f"  rew: {reward:>+7.3f}", rew_col),
        (f"  return: {ep_return:>+8.3f}", ret_col),
        (f"  best: {best_return:>+8.3f}", C_ACCENT),
        (f"  sr: {sr:.1%}({wins}/{episodes_done})", sr_col),
    ]
    x = rect.x + 6
    y = rect.y + 6
    for text, col in chips:
        surf = info_font.render(text, True, col)
        surface.blit(surf, (x, y))
        x += surf.get_width()

    # ── Episode log (last 3) ───────────────────────────────────────────────
    if episode_log:
        log_x = rect.x + 6
        log_y = rect.y + 28
        for entry in episode_log[-3:]:
            s = log_font.render(entry, True, C_DIM)
            surface.blit(s, (log_x, log_y))
            log_x += s.get_width() + 20


# ─────────────────────────────────────────────────────────────────────────────
# Action label map (shared with watch.py)
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_LABELS: Dict[str, Dict[int, str]] = {
    "line_world":         {0: "<- left",   1: "right ->"},
    "cliff_world":        {0: "^ up",      1: "v down",   2: "< left",  3: "right >"},
    "grid_world":         {0: "^ up",      1: "v down",   2: "< left",  3: "right >"},
    "maze_world":         {0: "^ N",       1: "v S",      2: "< W",     3: "E >"},
    "labyrinth_world":    {0: "^ N",       1: "v S",      2: "< W",     3: "E >"},
    "memory_vault_world": {0: "^ N",       1: "v S",      2: "< W",     3: "E >"},
    "warehouse_world":    {0: "^ N",       1: "v S",      2: "< W",     3: "E >",    4: "charge"},
    "harvest_world":      {0: "^ N",       1: "v S",      2: "< W",     3: "E >",    4: "pick"},
    "park_world":         {0: "^ N",       1: "v S",      2: "< W",     3: "E >",    4: "park"},
    "swamp_world":        {0: "^ N",       1: "v S",      2: "< W",     3: "E >"},
    "bridge_world":       {0: "^ N",       1: "v S",      2: "< W",     3: "E >",    4: "repair"},
    "escape_world":       {0: "^ N",       1: "v S",      2: "< W",     3: "E >",    4: "hide"},
    "pursuit_world":      {0: "^ N",       1: "v S",      2: "< W",     3: "E >"},
    "factory_world":      {0: "idle",      1: "process",  2: "repair",  3: "priority"},
    "trade_world":        {0: "hold",      1: "buy",      2: "sell",    3: "short",  4: "cover"},
    "risk_tutorial_world":{0: "fold",      1: "call",     2: "raise",   3: "all-in"},
    "wind_master_world":  {0: "none",      1: "thrust",   2: "brake",   3: "stabilize"},
    "rule_flip_world":    {0: "track-A",   1: "track-B",  2: "track-C", 3: "track-D"},
    "chess_world":        {0: "build",     1: "pressure", 2: "capture", 3: "defend", 4: "tempo", 5: "convert"},
    "chess_world_v2":     {0: "build",     1: "pressure", 2: "capture", 3: "defend", 4: "tempo", 5: "convert"},
    "go_world":           {0: "expand",    1: "cut",      2: "connect", 3: "probe",  4: "invade", 5: "settle"},
    "go_world_v2":        {0: "expand",    1: "cut",      2: "connect", 3: "probe",  4: "invade", 5: "settle"},
    "uno_world":          {0: "play_low",  1: "play_high",2: "match",   3: "draw",   4: "skip"},
    "uno_world_v2":       {0: "play_low",  1: "play_high",2: "match",   3: "draw",   4: "skip"},
    "robot_reacher_world":{0: "hold",      1: "j1 +",     2: "j1 -",    3: "j2 +",   4: "j2 -"},
    "embodied_point_world":{0: "hold",     1: "+x",       2: "-x",      3: "+y",     4: "-y"},
    "warehouse_3d_world": {0: "hold",      1: "forward",  2: "backward",3: "left",   4: "right"},
}


def action_label(verse_name: str, action: Any) -> str:
    labels = _ACTION_LABELS.get(verse_name, {})
    try:
        return labels.get(int(action), str(action))
    except (TypeError, ValueError):
        return str(action)


# ─────────────────────────────────────────────────────────────────────────────
# Param parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_kv(kvs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in (kvs or []):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = float(v) if "." in v else int(v)
            continue
        except ValueError:
            pass
        try:
            parsed = json.loads(v)
            if isinstance(parsed, (list, dict)):
                out[k] = parsed
                continue
        except Exception:
            pass
        out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spatial & Embodied verse detection
# ─────────────────────────────────────────────────────────────────────────────

_SPATIAL_VERSES = {
    "line_world", "grid_world", "cliff_world", "maze_world", "labyrinth_world",
    "warehouse_world", "harvest_world", "park_world", "swamp_world",
    "bridge_world", "escape_world", "pursuit_world", "memory_vault_world",
}

_EMBODIED_VERSES = {
    "robot_reacher_world", "embodied_point_world",
}

def is_spatial(verse_name: str) -> bool:
    return verse_name in _SPATIAL_VERSES

def is_embodied(verse_name: str) -> bool:
    return verse_name in _EMBODIED_VERSES


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ViewerConfig:
    verse_name: str
    algo: str
    train: bool
    episodes: int
    fps: float
    seed: int
    vparam: Dict[str, Any]
    aconfig: Dict[str, Any]
    win_w: int
    win_h: int
    max_steps: int


# ─────────────────────────────────────────────────────────────────────────────
# Main viewer
# ─────────────────────────────────────────────────────────────────────────────

HEADER_H = 44
FOOTER_H = 64


def run_viewer(cfg: ViewerConfig) -> None:
    pygame.init()
    pygame.display.set_caption(f"Multiverse — {cfg.verse_name}")

    screen = pygame.display.set_mode((cfg.win_w, cfg.win_h), pygame.RESIZABLE)
    clock  = pygame.time.Clock()

    TITLE_FONT = _load_bold_font(18)
    INFO_FONT  = _load_font(14)
    MONO_FONT  = _load_font(15)
    LOG_FONT   = _load_font(12)
    VAL_FONT   = _load_bold_font(22)
    LABEL_FONT = _load_font(12)

    # ── Build verse ────────────────────────────────────────────────────────
    verse_params = dict(cfg.vparam)
    if cfg.max_steps > 0:
        verse_params.setdefault("max_steps", cfg.max_steps)

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=cfg.verse_name,
        verse_version="0.1",
        params=verse_params,
        tags=[],
        seed=cfg.seed,
    )
    verse = create_verse(verse_spec)

    # ── Build agent ────────────────────────────────────────────────────────
    agent_spec = AgentSpec(
        spec_version="v1",
        algo=cfg.algo,
        policy_id=f"{cfg.algo}_viewer",
        policy_version="0.0",
        config=cfg.aconfig,
    )
    agent = create_agent(agent_spec, verse.observation_space, verse.action_space)
    agent.seed(cfg.seed)

    # ── Probe render mode ──────────────────────────────────────────────────
    verse.reset()
    test_rgb = verse.render(mode="rgb_array")
    use_rgb  = test_rgb is not None
    spatial  = is_spatial(cfg.verse_name)

    # Compute cell size for text-grid mode
    test_ansi = verse.render(mode="ansi") or ""
    grid_probe = parse_ansi_frame(test_ansi)
    n_rows, n_cols = grid_dims(grid_probe)

    # ── Main state ────────────────────────────────────────────────────────
    fps           = float(cfg.fps)
    paused        = False
    episode       = 0
    best_return   = float("-inf")
    wins          = 0
    episode_log: List[str] = []

    # Episode state
    obs           = verse.reset().obs
    ep_return     = 0.0
    step          = 0
    done          = False
    truncated     = False
    transitions: List[Transition] = []
    current_action: Any = 0
    current_reward: float = 0.0
    current_info: Dict[str, Any] = {}
    max_steps     = int(verse_params.get("max_steps", 200))
    episode      += 1
    verse.seed(cfg.seed + episode)
    reset_r       = verse.reset()
    obs           = reset_r.obs
    current_info  = dict(reset_r.info or {})

    running = True
    while running:
        # ── Event handling ─────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    fps = min(fps + 2, 60)
                elif event.key == pygame.K_DOWN:
                    fps = max(fps - 2, 1)
            elif event.type == pygame.VIDEORESIZE:
                cfg.win_w, cfg.win_h = event.w, event.h

        if not running:
            break

        # ── Simulation step (if not paused) ───────────────────────────────
        if not paused:
            if not done and not truncated and step < max_steps:
                act_result     = agent.act(obs)
                current_action = act_result.action
                step_result    = verse.step(current_action)
                current_reward = step_result.reward
                current_info  = dict(step_result.info or {})
                next_obs       = step_result.obs
                done           = step_result.done
                truncated      = step_result.truncated
                ep_return     += current_reward
                step          += 1

                if cfg.train:
                    transitions.append(Transition(
                        obs=obs,
                        action=current_action,
                        reward=current_reward,
                        next_obs=next_obs,
                        done=done,
                        truncated=truncated,
                        info=dict(step_result.info),
                    ))
                obs = next_obs

            else:
                # Episode ended
                if ep_return > best_return:
                    best_return = ep_return
                success = done and not truncated and ep_return > 0
                if success:
                    wins += 1

                outcome = "WIN" if success else "lose"
                episode_log.append(
                    f"ep {episode:>4}  {outcome:<4}  ret={ep_return:>+7.3f}  steps={step:>4}"
                )

                if cfg.train and transitions:
                    try:
                        agent.learn(ExperienceBatch(transitions=transitions))
                    except (NotImplementedError, AttributeError):
                        pass

                if cfg.episodes > 0 and episode >= cfg.episodes:
                    running = False
                    break

                # Start next episode
                episode += 1
                verse.seed(cfg.seed + episode)
                reset_r    = verse.reset()
                obs        = reset_r.obs
                current_info = dict(reset_r.info or {})
                ep_return  = 0.0
                step       = 0
                done       = False
                truncated  = False
                transitions.clear()
                current_action  = 0
                current_reward  = 0.0

        # ── Layout rects ───────────────────────────────────────────────────
        w, h = screen.get_size()
        header_rect = pygame.Rect(0, 0, w, HEADER_H)
        footer_rect = pygame.Rect(0, h - FOOTER_H, w, FOOTER_H)
        game_rect   = pygame.Rect(0, HEADER_H, w, h - HEADER_H - FOOTER_H)

        screen.fill(BG)

        # ── Game area render ───────────────────────────────────────────────
        if is_embodied(cfg.verse_name):
            if cfg.verse_name == "robot_reacher_world":
                render_robot_reacher(screen, verse, game_rect, MONO_FONT, LABEL_FONT)
            else:
                render_embodied_point(screen, verse, game_rect, MONO_FONT, LABEL_FONT)

        elif use_rgb:
            rgb_frame = verse.render(mode="rgb_array")
            if rgb_frame is not None:
                surf = rgb_to_surface(rgb_frame, game_rect.width, game_rect.height)
                screen.blit(surf, (game_rect.x, game_rect.y))
            else:
                screen.fill(GRID_BG, game_rect)
            if cfg.verse_name == "warehouse_3d_world":
                draw_warehouse_3d_hud(
                    screen,
                    verse,
                    game_rect,
                    MONO_FONT,
                    LABEL_FONT,
                    step,
                    current_action,
                    current_info,
                )

        elif spatial:
            ansi_frame = verse.render(mode="ansi") or ""
            grid = parse_ansi_frame(ansi_frame)
            n_rows, n_cols = grid_dims(grid)
            if n_rows > 0 and n_cols > 0:
                cell_w = max(10, min(game_rect.width  // n_cols, 48))
                cell_h = max(10, min(game_rect.height // n_rows, 48))
                cell   = min(cell_w, cell_h)
                render_text_grid(screen, grid, game_rect, MONO_FONT, cell, cell)
            else:
                screen.fill(GRID_BG, game_rect)

        else:
            # Abstract verse — show obs stats panel
            render_stats_panel(screen, obs, game_rect, LABEL_FONT, VAL_FONT, cfg.verse_name)

        # ── Header & footer ────────────────────────────────────────────────
        draw_header(
            screen, header_rect,
            verse_name=cfg.verse_name,
            algo=cfg.algo,
            episode=episode,
            total_episodes=cfg.episodes,
            training=cfg.train,
            paused=paused,
            fps=fps,
            title_font=TITLE_FONT,
            info_font=INFO_FONT,
        )
        draw_footer(
            screen, footer_rect,
            step=step,
            max_steps=max_steps,
            action_label=action_label(cfg.verse_name, current_action),
            reward=current_reward,
            ep_return=ep_return,
            best_return=best_return,
            wins=wins,
            episodes_done=episode - 1,
            info_font=INFO_FONT,
            episode_log=episode_log,
            log_font=LOG_FONT,
        )

        pygame.display.flip()
        if fps > 0:
            clock.tick(fps)

    try:
        verse.close()
    finally:
        pygame.quit()

    # ── Summary ────────────────────────────────────────────────────────────
    total = episode
    sr = wins / total if total else 0.0
    print(f"\n{'─'*50}")
    print(f"  Verse    : {cfg.verse_name}")
    print(f"  Agent    : {cfg.algo}  (trained={cfg.train})")
    print(f"  Episodes : {total}")
    print(f"  Wins     : {wins}  ({sr:.1%} success rate)")
    print(f"  Best ret : {best_return:+.3f}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    known_verses = sorted(list_verses().keys())
    known_algos  = sorted(_AGENT_REGISTRY.keys())

    ap = argparse.ArgumentParser(
        description="Gymnasium-style graphical viewer for Multiverse verses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Verses : {', '.join(known_verses)}\nAgents : {', '.join(known_algos)}",
    )
    ap.add_argument("--verse",    default="maze_world",  help="Verse to run (default: maze_world)")
    ap.add_argument("--algo",     default="q",           help="Agent algorithm (default: q)")
    ap.add_argument("--train",    action="store_true",   help="Enable online learning")
    ap.add_argument("--episodes", type=int, default=0,   help="Episodes to watch (0=infinite)")
    ap.add_argument("--fps",      type=float, default=8, help="Steps per second (default: 8)")
    ap.add_argument("--seed",     type=int, default=42,  help="Random seed (default: 42)")
    ap.add_argument("--max-steps",type=int, default=0,   help="Override max steps (0=verse default)")
    ap.add_argument("--width",    type=int, default=720, help="Window width  (default: 720)")
    ap.add_argument("--height",   type=int, default=640, help="Window height (default: 640)")
    ap.add_argument("--vparam",   action="append", default=None, metavar="K=V")
    ap.add_argument("--aconfig",  action="append", default=None, metavar="K=V")
    args = ap.parse_args()

    if pygame is None:
        print("pygame not installed. Run: pip install pygame")
        sys.exit(1)

    if args.verse not in known_verses:
        print(f"Unknown verse '{args.verse}'. Known:\n  {chr(10).join(known_verses)}")
        sys.exit(1)
    if args.algo not in known_algos:
        print(f"Unknown algo '{args.algo}'. Known:\n  {chr(10).join(known_algos)}")
        sys.exit(1)

    cfg = ViewerConfig(
        verse_name=args.verse,
        algo=args.algo,
        train=args.train,
        episodes=max(0, args.episodes),
        fps=max(1.0, args.fps),
        seed=args.seed,
        vparam=_parse_kv(args.vparam),
        aconfig=_parse_kv(args.aconfig),
        win_w=max(400, args.width),
        win_h=max(300, args.height),
        max_steps=max(0, args.max_steps),
    )

    run_viewer(cfg)


if __name__ == "__main__":
    main()
