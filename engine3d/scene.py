"""
engine3d/scene.py

Hierarchical scene graph, camera, lighting, and transformation nodes.
Mirrors Blender's object and scene hierarchy model.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from engine3d.geometry import Mesh
from engine3d.math3d import Mat4, Vec3


class Transform:
    """Position, rotation (Euler angles in radians), and scale."""

    __slots__ = ("position", "rotation", "scale")

    def __init__(
        self,
        position: Optional[Vec3] = None,
        rotation: Optional[Vec3] = None,
        scale: Optional[Vec3] = None,
    ) -> None:
        self.position = position if position is not None else Vec3(0.0, 0.0, 0.0)
        self.rotation = rotation if rotation is not None else Vec3(0.0, 0.0, 0.0)
        self.scale = scale if scale is not None else Vec3(1.0, 1.0, 1.0)

    def local_matrix(self) -> Mat4:
        """Computes local 4x4 matrix: T * Rz * Ry * Rx * S."""
        t_mat = Mat4.translation(self.position.x, self.position.y, self.position.z)
        r_mat = Mat4.euler(self.rotation.x, self.rotation.y, self.rotation.z)
        s_mat = Mat4.scale(self.scale.x, self.scale.y, self.scale.z)
        return t_mat * r_mat * s_mat


class Node:
    """Scene graph node supporting hierarchical parenting."""

    def __init__(
        self,
        name: str,
        transform: Optional[Transform] = None,
        mesh: Optional[Mesh] = None,
    ) -> None:
        self.name = str(name)
        self.transform = transform if transform is not None else Transform()
        self.mesh = mesh
        self.parent: Optional["Node"] = None
        self.children: List["Node"] = []

    def add_child(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def world_matrix(self) -> Mat4:
        """Recursively computes world transformation matrix."""
        local = self.transform.local_matrix()
        if self.parent is not None:
            return self.parent.world_matrix() * local
        return local


class Camera:
    """Perspective or orthographic 3D camera."""

    def __init__(
        self,
        position: Vec3 = Vec3(0.0, 3.0, 5.0),
        target: Vec3 = Vec3(0.0, 0.0, 0.0),
        up: Vec3 = Vec3(0.0, 1.0, 0.0),
        fov_deg: float = 60.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        self.position = Vec3(position.x, position.y, position.z)
        self.target = Vec3(target.x, target.y, target.z)
        self.up = Vec3(up.x, up.y, up.z)
        self.fov_deg = float(fov_deg)
        self.aspect = float(aspect)
        self.near = float(near)
        self.far = float(far)

    def view_matrix(self) -> Mat4:
        return Mat4.look_at(self.position, self.target, self.up)

    def projection_matrix(self) -> Mat4:
        return Mat4.perspective(
            fov_rad=math.radians(self.fov_deg),
            aspect=self.aspect,
            near=self.near,
            far=self.far,
        )


class DirectionalLight:
    """Directional sun-like light source."""

    def __init__(
        self,
        direction: Vec3 = Vec3(0.5, -1.0, -0.5),
        color: Vec3 = Vec3(1.0, 1.0, 1.0),
        intensity: float = 1.0,
        ambient: float = 0.25,
    ) -> None:
        self.direction = direction.normalized()
        self.color = Vec3(color.x, color.y, color.z)
        self.intensity = float(intensity)
        self.ambient = float(ambient)


class Scene:
    """Container holding scene graph, camera, lights, and environment."""

    def __init__(self, background_color: Tuple[int, int, int] = (15, 18, 30)) -> None:
        self.root = Node("root")
        self.camera = Camera()
        self.lights: List[DirectionalLight] = [DirectionalLight()]
        self.background_color = background_color

    def add_node(self, node: Node) -> Node:
        return self.root.add_child(node)

    def get_render_queue(self) -> List[Tuple[Mesh, Mat4]]:
        """Traverses the scene tree and flattens all visible meshes with their world matrices."""
        queue: List[Tuple[Mesh, Mat4]] = []

        def traverse(node: Node, parent_matrix: Optional[Mat4] = None) -> None:
            local = node.transform.local_matrix()
            current_world = parent_matrix * local if parent_matrix is not None else local
            if node.mesh is not None:
                queue.append((node.mesh, current_world))
            for child in node.children:
                traverse(child, current_world)

        traverse(self.root)
        return queue
