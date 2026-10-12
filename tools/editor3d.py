"""
tools/editor3d.py

Blender-style desktop editor for engine3d.

Create a box (or plane, sphere, cylinder), select it, and move it on X, Y, and Z
by dragging the colored arrows or scrubbing the transform numbers.

    python tools/editor3d.py

Viewport:
    Left drag an arrow   move the selection on that axis
    Left click           select an object
    Right drag           orbit the camera
    Wheel                zoom

Panel:
    Add Box              create a box and select it
    X / Y / Z rows       drag the number, or use - / +
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import numpy as np

from engine3d.geometry import Mesh, create_box, create_cylinder, create_plane, create_sphere
from engine3d.math3d import Mat4, Vec3
from engine3d.rasterizer import Rasterizer
from engine3d.scene import Camera, Node, Scene, Transform

try:
    import pygame
except ImportError:  # pragma: no cover - runtime dependency
    pygame = None


PANEL_W = 320
AXIS = (Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0))
AXIS_NAME = ("X", "Y", "Z")
AXIS_COLOR = ((220, 64, 64), (72, 190, 96), (70, 130, 220))
KIND_COLOR = {
    "box": (196, 148, 86),
    "plane": (86, 138, 176),
    "sphere": (86, 176, 124),
    "cylinder": (184, 102, 92),
}


@dataclass
class EditorObject:
    name: str
    kind: str
    position: Vec3
    rotation: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    scale: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    color: Tuple[int, int, int] = (196, 148, 86)


def mesh_for(kind: str, color: Tuple[int, int, int]) -> Mesh:
    if kind == "plane":
        return create_plane(1.0, 1.0, color, segments=1)
    if kind == "sphere":
        return create_sphere(0.5, 10, 14, color)
    if kind == "cylinder":
        return create_cylinder(0.4, 1.0, 14, color)
    return create_box(1.0, 1.0, 1.0, color)


def object_matrix(obj: EditorObject) -> Mat4:
    return Transform(
        position=Vec3(obj.position.x, obj.position.y, obj.position.z),
        rotation=Vec3(obj.rotation.x, obj.rotation.y, obj.rotation.z),
        scale=Vec3(obj.scale.x, obj.scale.y, obj.scale.z),
    ).local_matrix()


def ray_aabb(origin: Vec3, direction: Vec3, bmin: Sequence[float], bmax: Sequence[float]) -> Optional[float]:
    """Ray parameter t for origin + t * direction against an axis-aligned box. None on a miss."""
    tmin = -1e9
    tmax = 1e9
    origin_c = (origin.x, origin.y, origin.z)
    direction_c = (direction.x, direction.y, direction.z)
    for index in range(3):
        component = direction_c[index]
        start = origin_c[index]
        if abs(component) < 1e-08:
            if start < bmin[index] or start > bmax[index]:
                return None
            continue
        t1 = (bmin[index] - start) / component
        t2 = (bmax[index] - start) / component
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None
    if tmax < 0.0:
        return None
    return tmin if tmin >= 0.0 else tmax


def screen_ray(camera: Camera, sx: float, sy: float, width: int, height: int) -> Tuple[Vec3, Vec3]:
    """World ray through a viewport pixel. Direction is normalized."""
    ndc_x = (sx / max(1, width - 1)) * 2.0 - 1.0
    ndc_y = 1.0 - (sy / max(1, height - 1)) * 2.0
    view = camera.view_matrix()
    proj = camera.projection_matrix()
    inv = np.linalg.inv((proj * view).m)
    near_h = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
    far_h = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float32)
    near = near_h[:3] / near_h[3]
    far = far_h[:3] / far_h[3]
    origin = Vec3(float(near[0]), float(near[1]), float(near[2]))
    direction = Vec3(float(far[0] - near[0]), float(far[1] - near[1]), float(far[2] - near[2])).normalized()
    return origin, direction


def pick_object(
    objects: Sequence[EditorObject],
    meshes: Dict[Tuple[str, Tuple[int, int, int]], Mesh],
    camera: Camera,
    sx: float,
    sy: float,
    width: int,
    height: int,
) -> Optional[int]:
    """Return the index of the closest object under the pixel, if any."""
    origin, direction = screen_ray(camera, sx, sy, width, height)
    best_index = None
    best_dist = 1e9
    for index, obj in enumerate(objects):
        world = object_matrix(obj)
        inverse = Mat4(np.linalg.inv(world.m))
        local_origin = inverse.transform_point(origin)
        local_direction = inverse.transform_vector(direction)
        mesh = meshes[(obj.kind, obj.color)]
        vmin = mesh.vertices.min(axis=0)
        vmax = mesh.vertices.max(axis=0)
        pad = 0.06
        hit = ray_aabb(
            local_origin,
            local_direction,
            (float(vmin[0]) - pad, float(vmin[1]) - pad, float(vmin[2]) - pad),
            (float(vmax[0]) + pad, float(vmax[1]) + pad, float(vmax[2]) + pad),
        )
        if hit is None:
            continue
        point = world.transform_point(local_origin + local_direction * hit)
        dist = (point - camera.position).length()
        if dist < best_dist:
            best_dist = dist
            best_index = index
    return best_index


def axis_drag_plane(axis: Vec3, cam_forward: Vec3) -> Vec3:
    side = axis.cross(cam_forward)
    if side.length_squared() < 1e-08:
        side = axis.cross(Vec3(0.0, 1.0, 0.0))
    return side.cross(axis).normalized()


def ray_plane(origin: Vec3, direction: Vec3, point: Vec3, normal: Vec3) -> Optional[Vec3]:
    denom = direction.dot(normal)
    if abs(denom) < 1e-06:
        return None
    t = (point - origin).dot(normal) / denom
    if t < 0.0:
        return None
    return origin + direction * t


def pick_gizmo(
    origin_point: Vec3,
    camera: Camera,
    sx: float,
    sy: float,
    width: int,
    height: int,
    length: float = 1.5,
) -> Optional[int]:
    """Return 0/1/2 when the click is close to the X/Y/Z arrow."""
    origin, direction = screen_ray(camera, sx, sy, width, height)
    best = None
    best_dist = 0.18
    for index, axis in enumerate(AXIS):
        start = origin_point
        end = origin_point + axis * length
        ab = end - start
        ab_len2 = ab.length_squared()
        ray_dot_ray = direction.length_squared()
        ray_dot_ab = direction.dot(ab)
        rel = origin - start
        denom = ray_dot_ray * ab_len2 - ray_dot_ab * ray_dot_ab
        if abs(denom) < 1e-08:
            seg_u = 0.0
        else:
            seg_u = (rel.dot(direction) * ray_dot_ab - rel.dot(ab) * ray_dot_ray) / denom
        seg_u = min(1.0, max(0.0, seg_u))
        closest_on_seg = start + ab * seg_u
        ray_t = max(0.0, direction.dot(closest_on_seg - origin))
        closest_on_ray = origin + direction * ray_t
        dist = (closest_on_ray - closest_on_seg).length()
        if dist < best_dist:
            best_dist = dist
            best = index
    return best


def build_scene(
    objects: Sequence[EditorObject],
    meshes: Dict[Tuple[str, Tuple[int, int, int]], Mesh],
    camera: Camera,
    selected: Optional[int],
) -> Scene:
    scene = Scene(background_color=(22, 24, 32))
    scene.camera = camera
    scene.lights[0].ambient = 0.5
    scene.add_node(Node("floor", Transform(), create_plane(12.0, 12.0, (62, 66, 76), segments=4)))
    for i in range(-3, 4):
        tone = (96, 100, 112) if i == 0 else (74, 78, 88)
        scene.add_node(
            Node(
                f"grid_x_{i}",
                Transform(position=Vec3(0.0, 0.012, float(i))),
                create_box(8.0, 0.012, 0.02, tone),
            )
        )
        scene.add_node(
            Node(
                f"grid_z_{i}",
                Transform(position=Vec3(float(i), 0.012, 0.0)),
                create_box(0.02, 0.012, 8.0, tone),
            )
        )
    for obj in objects:
        scene.add_node(Node(obj.name, _transform(obj), meshes[(obj.kind, obj.color)]))
    if selected is not None and 0 <= selected < len(objects):
        anchor = objects[selected].position
        for index, axis in enumerate(AXIS):
            color = AXIS_COLOR[index]
            scene.add_node(
                Node(
                    f"gizmo_{AXIS_NAME[index]}",
                    Transform(position=anchor + axis * 0.75, scale=_gizmo_scale(index)),
                    create_box(1.0, 1.0, 1.0, color),
                )
            )
            scene.add_node(
                Node(
                    f"gizmo_tip_{AXIS_NAME[index]}",
                    Transform(position=anchor + axis * 1.45, scale=Vec3(0.16, 0.16, 0.16)),
                    create_box(1.0, 1.0, 1.0, color),
                )
            )
    return scene


def _transform(obj: EditorObject) -> Transform:
    return Transform(
        position=Vec3(obj.position.x, obj.position.y, obj.position.z),
        rotation=Vec3(obj.rotation.x, obj.rotation.y, obj.rotation.z),
        scale=Vec3(max(0.05, obj.scale.x), max(0.05, obj.scale.y), max(0.05, obj.scale.z)),
    )


def _gizmo_scale(axis_index: int) -> Vec3:
    if axis_index == 0:
        return Vec3(1.3, 0.06, 0.06)
    if axis_index == 1:
        return Vec3(0.06, 1.3, 0.06)
    return Vec3(0.06, 0.06, 1.3)


def _component(vec: Vec3, index: int) -> float:
    return (vec.x, vec.y, vec.z)[index]


def _set_component(vec: Vec3, index: int, value: float) -> None:
    if index == 0:
        vec.x = value
    elif index == 1:
        vec.y = value
    else:
        vec.z = value


class Editor:
    def __init__(self) -> None:
        self.objects: List[EditorObject] = [
            EditorObject("Box", "box", Vec3(0.0, 0.5, 0.0), color=KIND_COLOR["box"])
        ]
        self.selected: Optional[int] = 0
        self._counts = {"box": 1, "plane": 0, "sphere": 0, "cylinder": 0}
        self.meshes: Dict[Tuple[str, Tuple[int, int, int]], Mesh] = {}
        self._ensure_mesh(self.objects[0])
        self.camera = Camera(fov_deg=50.0, near=0.05, far=80.0)
        self.yaw = 0.75
        self.pitch = 0.55
        self.distance = 6.5
        self.frame: Optional[np.ndarray] = None
        self.dirty = True
        self.mode = "idle"
        self.drag_axis = 0
        self.drag_channel = "location"
        self.drag_start_mouse = 0
        self.drag_start_value = 0.0
        self.drag_start_pos = Vec3()
        self.drag_anchor = Vec3()
        self.drag_normal = Vec3(0.0, 1.0, 0.0)
        self.hits: List[Tuple[pygame.Rect, str, dict]] = []

    def _ensure_mesh(self, obj: EditorObject) -> None:
        key = (obj.kind, obj.color)
        if key not in self.meshes:
            self.meshes[key] = mesh_for(obj.kind, obj.color)

    def _place_camera(self) -> None:
        pitch = min(1.35, max(-0.35, self.pitch))
        self.pitch = pitch
        self.distance = min(28.0, max(1.8, self.distance))
        planar = math.cos(pitch)
        self.camera.position = Vec3(
            self.distance * planar * math.sin(self.yaw),
            0.4 + self.distance * math.sin(pitch),
            self.distance * planar * math.cos(self.yaw),
        )
        self.camera.target = Vec3(0.0, 0.4, 0.0)
        self.camera.up = Vec3(0.0, 1.0, 0.0)

    def add_object(self, kind: str) -> None:
        self._counts[kind] = self._counts.get(kind, 0) + 1
        number = self._counts[kind]
        name = kind.capitalize() if number == 1 else f"{kind.capitalize()}.{number - 1:03d}"
        spread = (len(self.objects) % 5) * 0.35
        obj = EditorObject(
            name,
            kind,
            Vec3(spread, 0.5 if kind != "plane" else 0.02, spread * 0.5),
            color=KIND_COLOR[kind],
        )
        self._ensure_mesh(obj)
        self.objects.append(obj)
        self.selected = len(self.objects) - 1
        self.dirty = True

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        del self.objects[self.selected]
        if not self.objects:
            self.selected = None
        else:
            self.selected = min(self.selected, len(self.objects) - 1)
        self.dirty = True

    def _channel_vec(self, obj: EditorObject, channel: str) -> Vec3:
        if channel == "rotation":
            return obj.rotation
        if channel == "scale":
            return obj.scale
        return obj.position

    def _display_value(self, obj: EditorObject, channel: str, axis: int) -> float:
        value = _component(self._channel_vec(obj, channel), axis)
        if channel == "rotation":
            return math.degrees(value)
        return value

    def _apply_display_value(self, obj: EditorObject, channel: str, axis: int, shown: float) -> None:
        if channel == "rotation":
            stored = math.radians(shown)
        elif channel == "scale":
            stored = max(0.05, shown)
        else:
            stored = shown
        _set_component(self._channel_vec(obj, channel), axis, stored)
        self.dirty = True

    def nudge(self, axis: int, direction: int, fine: bool) -> None:
        if self.selected is None:
            return
        step = 0.05 if fine else 0.25
        obj = self.objects[self.selected]
        _set_component(obj.position, axis, _component(obj.position, axis) + step * direction)
        self.dirty = True

    def _render_view(self, view_w: int, view_h: int) -> np.ndarray:
        self._place_camera()
        render_w = max(160, min(520, view_w))
        render_h = max(120, int(render_w * view_h / max(1, view_w)))
        scene = build_scene(self.objects, self.meshes, self.camera, self.selected)
        self.frame = Rasterizer(render_w, render_h).render(scene)
        self.dirty = False
        return self.frame

    def _viewport_click(self, sx: int, sy: int, view_w: int, view_h: int) -> None:
        self._place_camera()
        # Rays use the same aspect the rasterizer will bake in from the rendered frame.
        render_w = max(160, min(520, view_w))
        render_h = max(120, int(render_w * view_h / max(1, view_w)))
        self.camera.aspect = render_w / render_h
        px = sx * render_w / max(1, view_w)
        py = sy * render_h / max(1, view_h)
        if self.selected is not None:
            axis = pick_gizmo(self.objects[self.selected].position, self.camera, px, py, render_w, render_h)
            if axis is not None:
                self._begin_axis_drag(axis, px, py, render_w, render_h)
                return
        hit = pick_object(self.objects, self.meshes, self.camera, px, py, render_w, render_h)
        self.selected = hit
        self.dirty = True

    def _begin_axis_drag(self, axis: int, px: float, py: float, render_w: int, render_h: int) -> None:
        obj = self.objects[self.selected]
        forward = (self.camera.target - self.camera.position).normalized()
        normal = axis_drag_plane(AXIS[axis], forward)
        origin, direction = screen_ray(self.camera, px, py, render_w, render_h)
        anchor = ray_plane(origin, direction, obj.position, normal)
        if anchor is None:
            return
        self.mode = "gizmo"
        self.drag_axis = axis
        self.drag_start_pos = Vec3(obj.position.x, obj.position.y, obj.position.z)
        self.drag_anchor = anchor
        self.drag_normal = normal

    def _update_gizmo(self, sx: int, sy: int, view_w: int, view_h: int) -> None:
        if self.selected is None:
            return
        render_w = max(160, min(520, view_w))
        render_h = max(120, int(render_w * view_h / max(1, view_w)))
        self.camera.aspect = render_w / render_h
        px = sx * render_w / max(1, view_w)
        py = sy * render_h / max(1, view_h)
        origin, direction = screen_ray(self.camera, px, py, render_w, render_h)
        hit = ray_plane(origin, direction, self.drag_anchor, self.drag_normal)
        if hit is None:
            return
        delta = (hit - self.drag_anchor).dot(AXIS[self.drag_axis])
        moved = self.drag_start_pos + AXIS[self.drag_axis] * delta
        obj = self.objects[self.selected]
        obj.position = Vec3(moved.x, moved.y, moved.z)
        self.dirty = True

    def run(self) -> None:
        if pygame is None:
            raise SystemExit("pygame is not installed. Run: pip install pygame")
        pygame.init()
        pygame.display.set_caption("engine3d — create a box")
        screen = pygame.display.set_mode((1180, 740), pygame.RESIZABLE)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("segoeui", 16)
        font_sm = pygame.font.SysFont("segoeui", 13)
        font_title = pygame.font.SysFont("segoeui", 20, bold=True)
        running = True
        while running:
            view_w = max(160, screen.get_width() - PANEL_W)
            view_h = screen.get_height()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self.dirty = True
                elif event.type == pygame.KEYDOWN:
                    running = self._on_key(event) and running
                elif event.type == pygame.MOUSEWHEEL:
                    self.distance *= 0.88 if event.y > 0 else 1.14
                    self.dirty = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._on_mouse_down(event, view_w, view_h)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in (1, 3):
                        self.mode = "idle"
                elif event.type == pygame.MOUSEMOTION:
                    self._on_mouse_motion(event, view_w, view_h)

            if self.dirty or self.frame is None:
                frame = self._render_view(view_w, view_h)
            else:
                frame = self.frame
            surface = _frame_surface(frame, view_w, view_h)
            screen.fill((24, 24, 28))
            screen.blit(surface, (0, 0))
            self._draw_overlay(screen, font_sm, view_w, view_h)
            self._draw_panel(screen, font, font_sm, font_title)
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    def _on_key(self, event: "pygame.event.Event") -> bool:
        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        fine = bool(event.mod & pygame.KMOD_SHIFT)
        if event.key == pygame.K_b:
            self.add_object("box")
        elif event.key == pygame.K_DELETE or event.key == pygame.K_x and (event.mod & pygame.KMOD_CTRL):
            self.delete_selected()
        elif event.key == pygame.K_LEFT:
            self.nudge(0, -1, fine)
        elif event.key == pygame.K_RIGHT:
            self.nudge(0, 1, fine)
        elif event.key == pygame.K_UP:
            self.nudge(1, 1, fine)
        elif event.key == pygame.K_DOWN:
            self.nudge(1, -1, fine)
        elif event.key in (pygame.K_PAGEUP, pygame.K_PERIOD):
            self.nudge(2, 1, fine)
        elif event.key in (pygame.K_PAGEDOWN, pygame.K_COMMA):
            self.nudge(2, -1, fine)
        return True

    def _on_mouse_down(self, event: "pygame.event.Event", view_w: int, view_h: int) -> None:
        if event.button == 3 and event.pos[0] < view_w:
            self.mode = "orbit"
            return
        if event.button != 1:
            return
        for rect, kind, payload in reversed(self.hits):
            if rect.collidepoint(event.pos):
                self._activate_hit(kind, payload, event.pos[0])
                return
        if event.pos[0] < view_w:
            self._viewport_click(event.pos[0], event.pos[1], view_w, view_h)

    def _activate_hit(self, kind: str, payload: dict, mouse_x: int) -> None:
        if kind == "add":
            self.add_object(payload["kind"])
        elif kind == "select":
            self.selected = payload["index"]
            self.dirty = True
        elif kind == "delete":
            self.delete_selected()
        elif kind == "step":
            if self.selected is None:
                return
            obj = self.objects[self.selected]
            shown = self._display_value(obj, payload["channel"], payload["axis"])
            self._apply_display_value(obj, payload["channel"], payload["axis"], shown + payload["delta"])
        elif kind == "scrub":
            if self.selected is None:
                return
            obj = self.objects[self.selected]
            self.mode = "scrub"
            self.drag_channel = payload["channel"]
            self.drag_axis = payload["axis"]
            self.drag_start_mouse = mouse_x
            self.drag_start_value = self._display_value(obj, payload["channel"], payload["axis"])

    def _on_mouse_motion(self, event: "pygame.event.Event", view_w: int, view_h: int) -> None:
        if self.mode == "orbit":
            self.yaw += event.rel[0] * 0.01
            self.pitch += event.rel[1] * 0.008
            self.dirty = True
        elif self.mode == "gizmo":
            self._update_gizmo(event.pos[0], event.pos[1], view_w, view_h)
        elif self.mode == "scrub" and self.selected is not None:
            sensitivity = {"location": 0.01, "rotation": 0.35, "scale": 0.005}[self.drag_channel]
            shown = self.drag_start_value + (event.pos[0] - self.drag_start_mouse) * sensitivity
            self._apply_display_value(self.objects[self.selected], self.drag_channel, self.drag_axis, shown)

    def _draw_overlay(self, screen: "pygame.Surface", font: "pygame.font.Font", view_w: int, view_h: int) -> None:
        hint = "Left drag red/green/blue arrow    Right drag orbit    Wheel zoom    B add box"
        screen.blit(font.render(hint, True, (210, 214, 224)), (12, view_h - 24))
        if self.selected is None or not self.objects:
            return
        obj = self.objects[self.selected]
        label = f"{obj.name}    X {obj.position.x:.2f}    Y {obj.position.y:.2f}    Z {obj.position.z:.2f}"
        screen.blit(font.render(label, True, (236, 236, 240)), (12, 10))

    def _draw_panel(self, screen: "pygame.Surface", font: "pygame.font.Font", font_sm: "pygame.font.Font", title: "pygame.font.Font") -> None:
        width, height = screen.get_size()
        panel = pygame.Rect(width - PANEL_W, 0, PANEL_W, height)
        pygame.draw.rect(screen, (34, 34, 38), panel)
        pygame.draw.line(screen, (58, 58, 64), (panel.x, 0), (panel.x, height))
        self.hits = []
        x = panel.x + 16
        inner = PANEL_W - 32
        screen.blit(title.render("3D Editor", True, (240, 240, 244)), (x, 14))
        screen.blit(font_sm.render("Create a box and move XYZ", True, (168, 168, 176)), (x, 42))

        add = pygame.Rect(x, 70, inner, 36)
        self._button(screen, font, add, "Add Box", (214, 122, 48), "add", {"kind": "box"})
        small_w = (inner - 12) // 3
        for index, kind in enumerate(("plane", "sphere", "cylinder")):
            rect = pygame.Rect(x + index * (small_w + 6), 114, small_w, 28)
            self._button(screen, font_sm, rect, kind.capitalize(), (58, 58, 66), "add", {"kind": kind})

        screen.blit(font_sm.render("SCENE", True, (140, 140, 148)), (x, 156))
        list_top = 178
        list_bottom = height - 392
        row_h = 26
        for index, obj in enumerate(self.objects):
            y = list_top + index * (row_h + 4)
            if y + row_h > list_bottom:
                break
            rect = pygame.Rect(x, y, inner, row_h)
            selected = index == self.selected
            pygame.draw.rect(screen, (72, 56, 40) if selected else (46, 46, 52), rect, border_radius=4)
            screen.blit(font_sm.render(obj.name, True, (240, 240, 244)), (rect.x + 8, rect.y + 5))
            self.hits.append((rect, "select", {"index": index}))

        prop = height - 378
        screen.blit(font_sm.render("TRANSFORM", True, (140, 140, 148)), (x, prop))
        if self.selected is None or not self.objects:
            screen.blit(font.render("Select an object", True, (180, 180, 188)), (x, prop + 28))
            return
        obj = self.objects[self.selected]
        y = prop + 24
        for channel, label in (("location", "Location"), ("rotation", "Rotation"), ("scale", "Scale")):
            screen.blit(font_sm.render(label, True, (186, 186, 194)), (x, y))
            y += 20
            for axis in range(3):
                self._axis_row(screen, font_sm, x, y, inner, obj, channel, axis)
                y += 30
            y += 4
        delete = pygame.Rect(x, y, inner, 28)
        self._button(screen, font_sm, delete, "Delete", (92, 52, 52), "delete", {})

    def _button(self, screen, font, rect, label, color, kind, payload) -> None:
        pygame.draw.rect(screen, color, rect, border_radius=5)
        text = font.render(label, True, (248, 248, 250))
        screen.blit(text, text.get_rect(center=rect.center))
        self.hits.append((rect, kind, payload))

    def _axis_row(self, screen, font, x, y, inner, obj, channel, axis) -> None:
        color = AXIS_COLOR[axis]
        badge = pygame.Rect(x, y, 22, 24)
        pygame.draw.rect(screen, color, badge, border_radius=3)
        glyph = font.render(AXIS_NAME[axis], True, (20, 20, 24))
        screen.blit(glyph, glyph.get_rect(center=badge.center))
        minus = pygame.Rect(x + inner - 52, y, 24, 24)
        plus = pygame.Rect(x + inner - 24, y, 24, 24)
        scrub = pygame.Rect(x + 28, y, inner - 86, 24)
        pygame.draw.rect(screen, (48, 48, 54), scrub, border_radius=3)
        pygame.draw.rect(screen, (58, 58, 66), minus, border_radius=3)
        pygame.draw.rect(screen, (58, 58, 66), plus, border_radius=3)
        shown = self._display_value(obj, channel, axis)
        text = font.render(f"{shown:.2f}", True, (236, 236, 240))
        screen.blit(text, text.get_rect(center=scrub.center))
        screen.blit(font.render("-", True, (230, 230, 236)), font.render("-", True, (230, 230, 236)).get_rect(center=minus.center))
        screen.blit(font.render("+", True, (230, 230, 236)), font.render("+", True, (230, 230, 236)).get_rect(center=plus.center))
        step = 15.0 if channel == "rotation" else (0.1 if channel == "scale" else 0.25)
        self.hits.append((scrub, "scrub", {"channel": channel, "axis": axis}))
        self.hits.append((minus, "step", {"channel": channel, "axis": axis, "delta": -step}))
        self.hits.append((plus, "step", {"channel": channel, "axis": axis, "delta": step}))


def _frame_surface(frame: np.ndarray, view_w: int, view_h: int) -> "pygame.Surface":
    transposed = np.transpose(frame, (1, 0, 2))
    raw = pygame.surfarray.make_surface(transposed)
    return pygame.transform.smoothscale(raw, (view_w, view_h))


def main() -> None:
    Editor().run()


if __name__ == "__main__":
    main()
