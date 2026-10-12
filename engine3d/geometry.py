"""
engine3d/geometry.py

Mesh data models and parametric 3D geometric primitives.
Constructs vertex buffers, index buffers, and calculates surface normals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class Mesh:
    """3D Polygonal Mesh with triangle topology."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.normals is None or len(self.normals) != len(self.faces):
            self.compute_face_normals()

    def compute_face_normals(self) -> None:
        """Computes normalized surface normals for every triangular face."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        norm = np.linalg.norm(cross, axis=1, keepdims=True)
        norm[norm == 0] = 1e-09
        self.normals = (cross / norm).astype(np.float32)


def create_box(
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
    color: Tuple[int, int, int] = (180, 180, 180),
) -> Mesh:
    """Creates a 3D box centered at origin with 12 triangles."""
    hx = width * 0.5
    hy = height * 0.5
    hz = depth * 0.5
    verts = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [4, 5, 6],
            [4, 6, 7],
            [1, 0, 3],
            [1, 3, 2],
            [2, 3, 7],
            [2, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [1, 6, 5],
            [1, 2, 6],
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=np.int32,
    )
    face_colors = np.tile(np.array(color, dtype=np.uint8), (len(faces), 1))
    return Mesh(vertices=verts, faces=faces, normals=None, colors=face_colors)


def create_plane(
    width: float = 10.0,
    depth: float = 10.0,
    color: Tuple[int, int, int] = (60, 65, 80),
    segments: int = 1,
) -> Mesh:
    """Creates a horizontal ground plane on XZ at Y=0."""
    segments = max(1, int(segments))
    verts = []
    for iz in range(segments + 1):
        z = -depth * 0.5 + depth * iz / segments
        for ix in range(segments + 1):
            x = -width * 0.5 + width * ix / segments
            verts.append([x, 0.0, z])
    faces = []
    stride = segments + 1
    for iz in range(segments):
        for ix in range(segments):
            i0 = iz * stride + ix
            i1 = i0 + 1
            i2 = i0 + stride
            i3 = i2 + 1
            faces.append([i0, i3, i1])
            faces.append([i0, i2, i3])
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    face_colors = np.tile(np.array(color, dtype=np.uint8), (len(faces), 1))
    return Mesh(vertices=verts, faces=faces, normals=None, colors=face_colors)


def create_cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    segments: int = 16,
    color: Tuple[int, int, int] = (140, 160, 200),
) -> Mesh:
    """Creates a vertical cylinder centered on Y axis."""
    hh = height * 0.5
    verts = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        c = math.cos(theta)
        s = math.sin(theta)
        verts.append([radius * c, hh, radius * s])
        verts.append([radius * c, -hh, radius * s])
    top_center_idx = len(verts)
    verts.append([0.0, hh, 0.0])
    bot_center_idx = len(verts)
    verts.append([0.0, -hh, 0.0])
    verts_arr = np.array(verts, dtype=np.float32)
    faces = []
    for i in range(segments):
        next_i = (i + 1) % segments
        t0 = i * 2
        b0 = i * 2 + 1
        t1 = next_i * 2
        b1 = next_i * 2 + 1
        faces.append([t0, b0, t1])
        faces.append([t1, b0, b1])
        faces.append([top_center_idx, t0, t1])
        faces.append([bot_center_idx, b1, b0])
    faces_arr = np.array(faces, dtype=np.int32)
    face_colors = np.tile(np.array(color, dtype=np.uint8), (len(faces_arr), 1))
    return Mesh(vertices=verts_arr, faces=faces_arr, normals=None, colors=face_colors)


def create_sphere(
    radius: float = 0.5,
    rings: int = 12,
    sectors: int = 16,
    color: Tuple[int, int, int] = (240, 140, 50),
) -> Mesh:
    """Creates a UV sphere centered at origin."""
    verts = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            verts.append([x, y, z])
    verts_arr = np.array(verts, dtype=np.float32)
    faces = []
    for r in range(rings):
        for s in range(sectors):
            cur = r * (sectors + 1) + s
            nxt = cur + (sectors + 1)
            faces.append([cur, cur + 1, nxt])
            faces.append([cur + 1, nxt + 1, nxt])
    faces_arr = np.array(faces, dtype=np.int32)
    face_colors = np.tile(np.array(color, dtype=np.uint8), (len(faces_arr), 1))
    return Mesh(vertices=verts_arr, faces=faces_arr, normals=None, colors=face_colors)
