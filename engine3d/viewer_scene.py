"""Engine-neutral snapshots consumed by the optional OpenGL viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from integrations.physics.domain_randomization import sample_domain_randomization


Color = Tuple[float, float, float]
Vector3 = Tuple[float, float, float]


@dataclass
class SceneObject:
    name: str
    position: Vector3
    scale: Vector3 = (1.0, 1.0, 1.0)
    rotation: Vector3 = (0.0, 0.0, 0.0)
    color: Color = (0.68, 0.72, 0.78)
    shape: str = "box"
    mesh_path: Optional[str] = None
    physics_body_name: Optional[str] = None
    end_position: Optional[Vector3] = None
    visible: bool = True
    inspectable: bool = True
    line_width: float = 0.036


@dataclass
class SceneFrame:
    title: str
    objects: List[SceneObject] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    selected_name: Optional[str] = None


class SceneSnapshotChannel:
    """Latest-frame channel; publishers never wait for the viewer to draw."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[SceneFrame] = None
        self._sequence = 0

    def publish(self, frame: SceneFrame) -> int:
        with self._lock:
            self._frame = frame
            self._sequence += 1
            return self._sequence

    def read_latest(self) -> Tuple[int, Optional[SceneFrame]]:
        with self._lock:
            return self._sequence, self._frame


@lru_cache(maxsize=1)
def _asset_mapping() -> Dict[str, Dict[str, Any]]:
    path = Path(__file__).resolve().parent / "assets" / "scene_mappings.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload.get("bodies", {}))
    except (OSError, ValueError, TypeError):
        return {}


def _object(name: str, position: Sequence[float], *, scale=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0), color=(0.68, 0.72, 0.78), shape="box",
            inspectable=True, line_width=0.036) -> SceneObject:
    item = SceneObject(name=name, position=tuple(map(float, position)),
                       scale=tuple(map(float, scale)), rotation=tuple(map(float, rotation)),
                       color=tuple(float(c) for c in color), shape=shape,
                       inspectable=bool(inspectable), line_width=float(line_width))
    mapped = _asset_mapping().get(name, {})
    item.physics_body_name = mapped.get("mujoco_body")
    asset = mapped.get("mesh")
    if asset:
        mesh = (Path(__file__).resolve().parent / "assets" / asset).resolve()
        if mesh.is_file():
            item.mesh_path = str(mesh)
    return item


def _apply_visual_randomization(verse: Any, frame: SceneFrame) -> SceneFrame:
    sample = getattr(verse, "_randomization_sample", None)
    if sample is None:
        params = dict(getattr(getattr(verse, "spec", None), "params", {}) or {})
        sample = sample_domain_randomization(
            getattr(getattr(verse, "spec", None), "seed", None),
            params.get("domain_randomization", False),
        )
    tint = tuple(float(value) for value in sample.visual_tint)
    for item in frame.objects:
        item.color = tuple(float(np.clip(color * factor, 0.0, 1.0)) for color, factor in zip(item.color, tint))
    return frame


def scene_from_verse(verse: Any) -> SceneFrame:
    """Make an OpenGL scene snapshot for a built-in 3D/embodied Verse."""
    class_name = type(verse).__name__
    if class_name == "Warehouse3DWorldVerse":
        from verses.warehouse_3d_world import CRATES

        objects = [_object("floor", (0.0, -0.08, 0.0), scale=(14.0, 0.16, 14.0), color=(0.105, 0.12, 0.145))]
        # Build a warehouse corner around the playable area. These are visual
        # set pieces; task collision and reward rules remain owned by the Verse.
        objects.extend((
            _object("rear_wall", (0.0, 2.45, -6.65), scale=(14.0, 4.9, 0.22),
                    color=(0.15, 0.19, 0.24), inspectable=False),
            _object("left_wall", (-6.65, 2.45, 0.0), scale=(0.22, 4.9, 14.0),
                    color=(0.13, 0.17, 0.22), inspectable=False),
            _object("rear_wall_band", (0.0, 3.85, -6.51), scale=(13.9, 0.16, 0.08),
                    color=(0.24, 0.31, 0.37), inspectable=False),
            _object("left_wall_band", (-6.51, 3.85, 0.0), scale=(0.08, 0.16, 13.9),
                    color=(0.21, 0.28, 0.34), inspectable=False),
        ))
        for x in (-6.35, -3.2, 0.0, 3.2, 6.35):
            objects.append(_object(f"rear_column_{x}", (x, 2.45, -6.43),
                                   scale=(0.14, 4.9, 0.16), color=(0.34, 0.39, 0.43),
                                   inspectable=False))
        for z in (-6.35, -3.2, 0.0, 3.2, 6.35):
            objects.append(_object(f"left_column_{z}", (-6.43, 2.45, z),
                                   scale=(0.16, 4.9, 0.14), color=(0.29, 0.35, 0.39),
                                   inspectable=False))
        # Corrugated wall ribs and overhead trusses provide scale cues.
        for x in range(-6, 7, 1):
            objects.append(_object(f"rear_rib_{x}", (float(x), 2.45, -6.515),
                                   scale=(0.035, 4.75, 0.025), color=(0.23, 0.28, 0.33),
                                   inspectable=False))
        for z in range(-6, 7, 1):
            objects.append(_object(f"left_rib_{z}", (-6.515, 2.45, float(z)),
                                   scale=(0.025, 4.75, 0.035), color=(0.2, 0.25, 0.30),
                                   inspectable=False))
        for z in (-5.2, -2.2, 0.8, 3.8):
            objects.append(_object(f"roof_truss_{z}", (0.0, 4.82, z),
                                   scale=(13.5, 0.12, 0.12), color=(0.28, 0.33, 0.38),
                                   inspectable=False))
            objects.append(_object(f"roof_light_{z}", (0.0, 4.56, z),
                                   scale=(1.4, 0.09, 0.42), color=(0.78, 0.74, 0.59),
                                   inspectable=False))
        # Two industrial rack bays along the rear wall, with stored cartons.
        for rack_x in (-3.25, 1.8, 4.8):
            for post_x in (rack_x - 1.45, rack_x + 1.45):
                objects.append(_object(f"rack_post_{rack_x}_{post_x}",
                                       (post_x, 1.9, -5.45), scale=(0.12, 3.8, 0.14),
                                       color=(0.78, 0.39, 0.12), inspectable=False))
            for level, shelf_y in enumerate((0.25, 1.48, 2.72, 3.58)):
                objects.append(_object(f"rack_beam_{rack_x}_{level}",
                                       (rack_x, shelf_y, -5.45), scale=(3.0, 0.12, 0.82),
                                       color=(0.82, 0.42, 0.13), inspectable=False))
                if level < 3:
                    objects.append(_object(f"rack_shelf_{rack_x}_{level}",
                                           (rack_x, shelf_y + 0.12, -5.48),
                                           scale=(2.82, 0.07, 0.72),
                                           color=(0.30, 0.34, 0.37), inspectable=False))
        for carton_x, carton_y, carton_z, size, carton_color in (
            (-3.25, 0.92, -5.4, (1.1, 0.95, 0.62), (0.57, 0.37, 0.22)),
            (-3.25, 2.18, -5.4, (1.28, 0.92, 0.68), (0.66, 0.49, 0.31)),
            (1.8, 0.92, -5.4, (1.28, 0.95, 0.64), (0.53, 0.35, 0.2)),
            (1.8, 2.18, -5.4, (0.98, 0.9, 0.66), (0.64, 0.44, 0.27)),
            (4.8, 0.92, -5.4, (1.0, 0.92, 0.62), (0.59, 0.4, 0.24)),
        ):
            objects.append(_object(f"stored_carton_{carton_x}_{carton_y}",
                                   (carton_x, carton_y, carton_z), scale=size,
                                   color=carton_color, inspectable=False))
        # Subtle floor grid and perimeter markings make scale and the playable
        # area legible without filling the scene tree with helper objects.
        grid_color = (0.17, 0.19, 0.22)
        for offset in range(-4, 5, 2):
            objects.append(_object(f"grid_x_{offset}", (float(offset), 0.008, 0.0),
                                   shape="line", color=grid_color, inspectable=False,
                                   line_width=0.008))
            objects[-1].end_position = (float(offset), 0.008, 5.5)
            objects[-1].position = (float(offset), 0.008, -5.5)
            objects.append(_object(f"grid_z_{offset}", (0.0, 0.008, float(offset)),
                                   shape="line", color=grid_color, inspectable=False,
                                   line_width=0.008))
            objects[-1].end_position = (5.5, 0.008, float(offset))
            objects[-1].position = (-5.5, 0.008, float(offset))
        boundary_color = (0.75, 0.58, 0.22)
        for edge in (-5.5, 5.5):
            objects.append(_object(f"boundary_x_{edge}", (edge, 0.012, 0.0),
                                   shape="line", color=boundary_color, inspectable=False,
                                   line_width=0.024))
            objects[-1].end_position = (edge, 0.012, 5.5)
            objects[-1].position = (edge, 0.012, -5.5)
            objects.append(_object(f"boundary_z_{edge}", (0.0, 0.012, edge),
                                   shape="line", color=boundary_color, inspectable=False,
                                   line_width=0.024))
            objects[-1].end_position = (5.5, 0.012, edge)
            objects[-1].position = (-5.5, 0.012, edge)
        for name, x, z, width, height, depth, rgb in CRATES:
            pallet_height = 0.12
            objects.append(_object(f"crate_{name}", (x, pallet_height + height / 2.0, z),
                                   scale=(width, height, depth),
                                   color=tuple(value / 255.0 for value in rgb)))
            for runner, runner_z in enumerate((-depth * 0.32, 0.0, depth * 0.32)):
                objects.append(_object(
                    f"pallet_{name}_{runner}", (x, pallet_height * 0.5, z + runner_z),
                    scale=(width * 0.9, pallet_height, depth * 0.16),
                    color=(0.36, 0.23, 0.13), inspectable=False,
                ))
            # Thin crate seams add shape and scale cues while staying visual-only.
            seam_y = min(height - 0.08, max(0.12, height * 0.18))
            seam_y += pallet_height
            seam_color = tuple(min(1.0, value / 255.0 * 1.18) for value in rgb)
            for face_z in (z - depth * 0.5 - 0.008, z + depth * 0.5 + 0.008):
                seam = _object(f"crate_{name}_seam_{face_z}",
                               (x, seam_y, face_z), shape="line", color=seam_color,
                               inspectable=False, line_width=0.018)
                seam.end_position = (x + width * 0.48, seam_y, face_z)
                seam.position = (x - width * 0.48, seam_y, face_z)
                objects.append(seam)
        objects.append(_object("warehouse_package", (0.55, 1.98, 2.55), shape="sphere",
                               scale=(0.22, 0.22, 0.22), color=(0.18, 0.66, 0.3)))
        objects.append(_object("warehouse_goal_pad", (verse._goal_x, 0.022, verse._goal_z),
                               scale=(0.72, 0.04, 0.72), shape="cylinder",
                               color=(0.15, 0.31, 0.22), inspectable=False))
        objects.append(_object("warehouse_robot", (verse.x, 0.36, verse.z),
                               scale=(0.95, 0.72, 0.8), rotation=(0.0, verse.yaw, 0.0),
                               color=(0.66, 0.44, 0.17)))
        cos_yaw, sin_yaw = math.cos(verse.yaw), math.sin(verse.yaw)
        for side in (-1.0, 1.0):
            for end in (-1.0, 1.0):
                local_x, local_z = side * 0.45, end * 0.24
                wheel_x = verse.x + cos_yaw * local_x + sin_yaw * local_z
                wheel_z = verse.z - sin_yaw * local_x + cos_yaw * local_z
                objects.append(_object(
                    f"warehouse_wheel_{int(side)}_{int(end)}", (wheel_x, 0.09, wheel_z),
                    scale=(0.18, 0.26, 0.18), shape="cylinder",
                    rotation=(0.0, verse.yaw, math.pi / 2.0), color=(0.075, 0.085, 0.095),
                    inspectable=False,
                ))
        objects.append(_object("warehouse_sensor", (verse.x, 0.91, verse.z),
                               scale=(0.35, 0.36, 0.35), shape="cylinder",
                               rotation=(0.0, verse.yaw, 0.0), color=(0.2, 0.54, 0.66)))
        objects.append(_object("warehouse_goal", (verse._goal_x, 0.18, verse._goal_z),
                               scale=(0.34, 0.34, 0.34), shape="sphere", color=(0.91, 0.74, 0.24)))
        return _apply_visual_randomization(verse, SceneFrame("Warehouse 3D", objects, {
            "step": int(verse._t), "distance": float(verse._distance()),
            "robot": [float(verse.x), float(verse.z), float(verse.yaw)],
            "goal": [float(verse._goal_x), float(verse._goal_z)],
        }, selected_name="warehouse_robot"))

    if class_name == "World3DVerse":
        width, depth = verse.params.width, verse.params.depth
        objects = []
        heights = getattr(verse, "_heights", [])
        pits = getattr(verse, "_pits", set())
        if heights:
            for x in range(width):
                for z in range(depth):
                    height = max(0.1, 0.15 * (int(heights[x][z]) + 1))
                    color = (0.24, 0.34, 0.38) if (x, z) in pits else (0.32 + 0.06 * heights[x][z], 0.39, 0.34)
                    objects.append(_object(f"terrain_{x}_{z}", (x, height / 2.0, z),
                                           scale=(0.96, height, 0.96), color=color))
        gx, gy = verse._gx(), verse._gy()
        objects.append(_object("terrain_agent", (verse._ax, verse._agent_z() + 0.15, verse._ay),
                               scale=(0.28, 0.3, 0.28), shape="sphere", color=(0.96, 0.69, 0.16)))
        objects.append(_object("terrain_goal", (gx, verse._elevation(gx, gy) + 0.14, gy),
                               scale=(0.25, 0.25, 0.25), shape="sphere", color=(0.18, 0.78, 0.38)))
        for index, (x, z) in enumerate(getattr(verse, "_patrols", [])):
            objects.append(_object(f"patrol_{index}", (x, verse._agent_z() + 0.12, z),
                                   scale=(0.22, 0.24, 0.22), shape="sphere", color=(0.82, 0.24, 0.28)))
        return _apply_visual_randomization(verse, SceneFrame("World 3D Terrain", objects, {
            "step": int(verse._t), "stamina": int(verse._stamina),
            "agent_cell": [int(verse._ax), int(verse._ay)], "goal_cell": [int(gx), int(gy)],
            "terrain": verse._features.summary(),
        }, selected_name="terrain_agent"))

    if class_name == "EmbodiedPointWorldVerse":
        x, z = verse.position
        objects = [
            _object("point_floor", (0.0, -0.04, 0.0), scale=(2.2, 0.08, 2.2), color=(0.18, 0.21, 0.27)),
            _object("point_agent", (x, 0.07, z), scale=(0.14, 0.14, 0.14), shape="sphere", color=(0.96, 0.69, 0.16)),
            _object("point_goal", (verse.goal[0], 0.04, verse.goal[1]), scale=(0.2, 0.08, 0.2), shape="sphere", color=(0.18, 0.78, 0.38)),
        ]
        return _apply_visual_randomization(verse, SceneFrame("Embodied Point", objects, {
            "step": int(verse._t), "distance": float(verse._distance()),
            "position": [float(x), float(z)], "velocity": list(verse.velocity),
        }, selected_name="point_agent"))

    if class_name == "RobotReacherWorldVerse":
        elbow = verse.elbow
        end_effector = verse.end_effector
        start = (0.0, 0.0, 0.0)
        elbow3 = (elbow[0], elbow[1], 0.0)
        end3 = (end_effector[0], end_effector[1], 0.0)
        objects = [
            _object("reacher_workspace", (0.0, 0.0, -0.035), scale=(2.0, 2.0, 0.03), color=(0.16, 0.19, 0.25)),
            _object("reacher_base", start, shape="sphere", scale=(0.09, 0.09, 0.09), color=(0.72, 0.76, 0.81)),
            _object("reacher_link_1", start, shape="line", color=(0.26, 0.69, 0.91)),
            _object("reacher_joint_1", elbow3, shape="sphere", scale=(0.05, 0.05, 0.05), color=(0.32, 0.8, 0.94)),
            _object("reacher_link_2", elbow3, shape="line", color=(0.95, 0.48, 0.2)),
            _object("reacher_end_effector", end3, shape="sphere", scale=(0.055, 0.055, 0.055), color=(1.0, 0.84, 0.25)),
            _object("reacher_target", (verse.target_x, verse.target_y, 0.02), shape="sphere",
                    scale=(verse.target_radius, verse.target_radius, verse.target_radius), color=(0.18, 0.82, 0.38)),
        ]
        objects[2].end_position = elbow3
        objects[4].end_position = end3
        return _apply_visual_randomization(verse, SceneFrame("Robot Reacher", objects, {
            "step": int(verse._t), "distance": float(verse._distance_to_target()),
            "joint_angles": [float(verse.theta1), float(verse.theta2)],
            "joint_velocities": [float(verse.dtheta1), float(verse.dtheta2)],
        }, selected_name="reacher_end_effector"))

    raise TypeError(f"No scene snapshot adapter for Verse class {class_name}")


__all__ = ["SceneObject", "SceneFrame", "SceneSnapshotChannel", "scene_from_verse"]
