"""
engine3d package

Blender-style 3D creation and simulation engine core for Multiverse.
Provides linear algebra, mesh geometry, scene graphs, and Z-buffered rasterization.
"""

from engine3d.math3d import Mat4, Vec3
from engine3d.geometry import Mesh, create_box, create_cylinder, create_plane, create_sphere
from engine3d.scene import Camera, DirectionalLight, Node, Scene, Transform
from engine3d.rasterizer import Rasterizer

__all__ = [
    "Mat4",
    "Vec3",
    "Mesh",
    "create_box",
    "create_cylinder",
    "create_plane",
    "create_sphere",
    "Camera",
    "DirectionalLight",
    "Node",
    "Scene",
    "Transform",
    "Rasterizer",
]
