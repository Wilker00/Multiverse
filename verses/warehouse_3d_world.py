"""
verses/warehouse_3d_world.py

Continuous 3D warehouse verse rendered by the engine3d software pipeline.

The agent drives a robot on the floor between crates. Rendering uses a
third-person follow camera or a fixed isometric camera and returns either
ANSI status text or an RGB framebuffer.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult
from engine3d.geometry import create_box, create_cylinder, create_plane, create_sphere
from engine3d.math3d import Vec3
from engine3d.rasterizer import Rasterizer
from engine3d.scene import Node, Scene, Transform


CRATES: Tuple[Tuple[str, float, float, float, float, float, Tuple[int, int, int]], ...] = (
    # name, x, z, width, height, depth, color
    ("blue", -1.6, 0.4, 1.1, 2.3, 0.95, (63, 112, 174)),
    ("red", -0.35, 1.45, 1.15, 1.15, 1.05, (184, 79, 67)),
    ("gray", 0.55, 2.55, 0.95, 1.75, 0.95, (168, 178, 180)),
    ("brown", 1.75, 1.15, 1.35, 1.45, 1.2, (178, 123, 72)),
)


def build_warehouse_scene(
    robot_x: float,
    robot_z: float,
    yaw: float,
    camera_mode: str = "isometric",
) -> Scene:
    """Assemble the warehouse scene graph for the current robot pose."""
    scene = Scene(background_color=(15, 18, 30))
    scene.lights[0].ambient = 0.38
    scene.add_node(
        Node(
            "ground",
            Transform(position=Vec3(0.0, 0.0, 0.0)),
            create_plane(14.0, 14.0, (82, 86, 98), segments=8),
        )
    )
    for name, cx, cz, width, height, depth, color in CRATES:
        scene.add_node(
            Node(
                name,
                Transform(position=Vec3(cx, height * 0.5, cz)),
                create_box(width, height, depth, color),
            )
        )
    scene.add_node(
        Node(
            "package",
            Transform(position=Vec3(0.55, 1.98, 2.55)),
            create_sphere(0.22, 8, 10, (46, 168, 78)),
        )
    )

    body_h = 0.72
    body = Node(
        "robot",
        Transform(position=Vec3(robot_x, body_h * 0.5, robot_z), rotation=Vec3(0.0, yaw, 0.0)),
        create_box(0.95, body_h, 0.8, (168, 112, 42)),
    )
    body.add_child(
        Node(
            "sensor",
            Transform(position=Vec3(0.0, body_h * 0.5 + 0.18, 0.0)),
            create_cylinder(0.22, 0.36, 12, (52, 138, 168)),
        )
    )
    scene.add_node(body)

    forward_x = math.sin(yaw)
    forward_z = math.cos(yaw)
    if camera_mode == "isometric":
        scene.camera.position = Vec3(5.2, 7.2, -4.6)
        scene.camera.target = Vec3(0.15, 0.5, 0.7)
        scene.camera.fov_deg = 42.0
    else:
        scene.camera.position = Vec3(robot_x - forward_x * 3.4, 2.35, robot_z - forward_z * 3.4)
        scene.camera.target = Vec3(robot_x + forward_x * 2.2, 0.7, robot_z + forward_z * 2.2)
        scene.camera.fov_deg = 52.0
    scene.camera.up = Vec3(0.0, 1.0, 0.0)
    return scene


class Warehouse3DWorldVerse:
    """
    Flat observation:
        x, z, yaw, goal_x, goal_z, dx, dz
    Discrete actions:
        0 hold, 1 forward, 2 backward, 3 turn left, 4 turn right
    """

    def __init__(self, spec: VerseSpec) -> None:
        tags = list(spec.tags)
        for tag in Warehouse3DWorldFactory().tags:
            if tag not in tags:
                tags.append(tag)
        self.spec = dataclasses.replace(spec, tags=tags)
        params = dict(self.spec.params or {})
        self._goal_x = float(params.get("goal_x", -2.4))
        self._goal_z = float(params.get("goal_z", 3.2))
        self._goal_radius = max(0.15, float(params.get("goal_radius", 0.45)))
        self._max_steps = max(1, int(params.get("max_steps", 80)))
        self._step_penalty = float(params.get("step_penalty", -0.02))
        self._collision_penalty = float(params.get("collision_penalty", -0.15))
        self._goal_reward = float(params.get("goal_reward", 1.0))
        self._move_speed = float(params.get("move_speed", 0.35))
        self._turn_speed = float(params.get("turn_speed", 0.35))
        self._arena = float(params.get("arena", 5.5))
        self._render_width = max(16, int(params.get("render_width", 640)))
        self._render_height = max(16, int(params.get("render_height", 426)))
        self._camera_mode = str(params.get("camera_mode", "isometric"))
        self._render_backend = str(params.get("render_backend", "auto")).lower()
        if self._render_backend not in ("auto", "panda", "panda3d", "software"):
            raise ValueError("render_backend must be 'auto', 'panda3d', or 'software'")
        self._panda_unavailable = False
        self._robot_radius = 0.42
        self.x = 1.15
        self.z = -1.8
        self.yaw = 0.15
        self._t = 0
        self._done = False
        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="vector",
            shape=(7,),
            dtype="float32",
            notes="x, z, yaw, goal_x, goal_z, dx, dz",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=5,
            notes="0 hold, 1 forward, 2 backward, 3 left, 4 right",
        )

    def seed(self, seed: Optional[int]) -> None:
        self.spec = dataclasses.replace(self.spec, seed=seed)

    def _obs(self) -> List[float]:
        return [
            float(self.x),
            float(self.z),
            float(self.yaw),
            float(self._goal_x),
            float(self._goal_z),
            float(self._goal_x - self.x),
            float(self._goal_z - self.z),
        ]

    def _distance(self) -> float:
        dx = self.x - self._goal_x
        dz = self.z - self._goal_z
        return float(math.hypot(dx, dz))

    def _blocked(self, x: float, z: float) -> bool:
        if abs(x) > self._arena or abs(z) > self._arena:
            return True
        for _name, cx, cz, width, _height, depth, _color in CRATES:
            hx = width * 0.5 + self._robot_radius
            hz = depth * 0.5 + self._robot_radius
            if abs(x - cx) <= hx and abs(z - cz) <= hz:
                return True
        return False

    def reset(self) -> ResetResult:
        self.x = 1.15
        self.z = -1.8
        self.yaw = 0.15
        self._t = 0
        self._done = False
        return ResetResult(obs=self._obs(), info={"distance": self._distance()})

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(
                obs=self._obs(),
                reward=0.0,
                done=True,
                truncated=False,
                info={"warning": "step after done"},
            )
        try:
            act = int(action)
        except (TypeError, ValueError):
            act = 0
        collided = False
        if act == 1 or act == 2:
            direction = 1.0 if act == 1 else -1.0
            nx = self.x + math.sin(self.yaw) * self._move_speed * direction
            nz = self.z + math.cos(self.yaw) * self._move_speed * direction
            if self._blocked(nx, nz):
                collided = True
            else:
                self.x, self.z = nx, nz
        elif act == 3:
            self.yaw += self._turn_speed
        elif act == 4:
            self.yaw -= self._turn_speed

        self._t += 1
        distance = self._distance()
        reached = distance <= self._goal_radius
        truncated = (not reached) and self._t >= self._max_steps
        if reached:
            reward = self._goal_reward
        elif collided:
            reward = self._collision_penalty
        else:
            reward = self._step_penalty
        self._done = bool(reached or truncated)
        return StepResult(
            obs=self._obs(),
            reward=float(reward),
            done=bool(reached),
            truncated=bool(truncated),
            info={"distance": distance, "reached_goal": bool(reached), "collided": bool(collided), "t": int(self._t)},
        )

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode == "ansi":
            return (
                f"warehouse3d robot ({self.x:.2f},{self.z:.2f}) yaw {self.yaw:.2f} "
                f"goal ({self._goal_x:.2f},{self._goal_z:.2f}) dist {self._distance():.2f}"
            )
        if mode == "rgb_array":
            if self._render_backend != "software" and not self._panda_unavailable:
                try:
                    from engine3d.panda_warehouse_renderer import render_warehouse_frame

                    return render_warehouse_frame(
                        robot_x=self.x,
                        robot_z=self.z,
                        yaw=self.yaw,
                        crates=CRATES,
                        arena=self._arena,
                        goal=(self._goal_x, self._goal_z, self._goal_radius),
                        camera_mode=self._camera_mode,
                        width=self._render_width,
                        height=self._render_height,
                    )
                except Exception as exc:
                    if self._render_backend in ("panda", "panda3d"):
                        raise RuntimeError(f"Panda3D warehouse renderer failed: {exc}") from exc
                    self._panda_unavailable = True
                    warnings.warn(
                        f"Panda3D renderer unavailable; using local software renderer ({exc})",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            scene = build_warehouse_scene(self.x, self.z, self.yaw, self._camera_mode)
            scene.add_node(
                Node(
                    "goal",
                    Transform(position=Vec3(self._goal_x, 0.18, self._goal_z)),
                    create_sphere(0.18, 6, 8, (230, 190, 60)),
                )
            )
            return Rasterizer(self._render_width, self._render_height).render(scene)
        return None

    def close(self) -> None:
        if self._render_backend == "software" or self._panda_unavailable:
            return
        try:
            from engine3d.panda_warehouse_renderer import close_warehouse_renderer

            close_warehouse_renderer()
        except ImportError:
            return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "x": float(self.x),
            "z": float(self.z),
            "yaw": float(self.yaw),
            "t": int(self._t),
            "done": bool(self._done),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self.x = float(state.get("x", self.x))
        self.z = float(state.get("z", self.z))
        self.yaw = float(state.get("yaw", self.yaw))
        self._t = int(state.get("t", 0) or 0)
        self._done = bool(state.get("done", False))


class Warehouse3DWorldFactory:
    @property
    def tags(self) -> List[str]:
        return [
            "navigation",
            "3d",
            "robotics",
            "spatial",
            "spatial_reasoning",
            "embodied",
            "warehouse",
        ]

    def create(self, spec: VerseSpec) -> Warehouse3DWorldVerse:
        return Warehouse3DWorldVerse(spec)
