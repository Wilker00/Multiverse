"""
engine3d/math3d.py

Linear algebra and geometric transformation core for the 3D engine.
Provides Vec3, Mat4, and camera/projection matrix generation.
Built from the ground up similar to Blender's mathematical foundation.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np


class Vec3:
    """3D Vector with linear algebra operations."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def _xyz(self, other: Union["Vec3", Sequence[float]]) -> Tuple[float, float, float]:
        if isinstance(other, Vec3):
            return other.x, other.y, other.z
        return float(other[0]), float(other[1]), float(other[2])

    def __add__(self, other: Union["Vec3", Sequence[float]]) -> "Vec3":
        ox, oy, oz = self._xyz(other)
        return Vec3(self.x + ox, self.y + oy, self.z + oz)

    def __sub__(self, other: Union["Vec3", Sequence[float]]) -> "Vec3":
        ox, oy, oz = self._xyz(other)
        return Vec3(self.x - ox, self.y - oy, self.z - oz)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        inv = 1.0 / scalar
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> "Vec3":
        length = self.length()
        if length < 1e-09:
            return Vec3(0.0, 0.0, 0.0)
        return self / length

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "Vec3":
        return cls(arr[0], arr[1], arr[2])


class Mat4:
    """4x4 Transformation Matrix stored in row-major order."""

    __slots__ = ("m",)

    def __init__(self, data: Union[np.ndarray, Sequence[Sequence[float]], None] = None) -> None:
        if data is None:
            self.m = np.eye(4, dtype=np.float32)
        else:
            self.m = np.array(data, dtype=np.float32)

    def __repr__(self) -> str:
        return f"Mat4(\n{self.m}\n)"

    @classmethod
    def identity(cls) -> "Mat4":
        return cls(np.eye(4, dtype=np.float32))

    @classmethod
    def translation(cls, tx: float, ty: float, tz: float) -> "Mat4":
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = tx
        mat[1, 3] = ty
        mat[2, 3] = tz
        return cls(mat)

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float) -> "Mat4":
        mat = np.diag([sx, sy, sz, 1.0]).astype(np.float32)
        return cls(mat)

    @classmethod
    def rotation_x(cls, rad: float) -> "Mat4":
        c = math.cos(rad)
        s = math.sin(rad)
        mat = np.eye(4, dtype=np.float32)
        mat[1, 1] = c
        mat[1, 2] = -s
        mat[2, 1] = s
        mat[2, 2] = c
        return cls(mat)

    @classmethod
    def rotation_y(cls, rad: float) -> "Mat4":
        c = math.cos(rad)
        s = math.sin(rad)
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = c
        mat[0, 2] = s
        mat[2, 0] = -s
        mat[2, 2] = c
        return cls(mat)

    @classmethod
    def rotation_z(cls, rad: float) -> "Mat4":
        c = math.cos(rad)
        s = math.sin(rad)
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = c
        mat[0, 1] = -s
        mat[1, 0] = s
        mat[1, 1] = c
        return cls(mat)

    @classmethod
    def euler(cls, rx: float, ry: float, rz: float) -> "Mat4":
        """Intrinsic Euler rotation Z * Y * X."""
        return cls.rotation_z(rz) * cls.rotation_y(ry) * cls.rotation_x(rx)

    def __mul__(self, other: "Mat4") -> "Mat4":
        return Mat4(np.matmul(self.m, other.m))

    def transform_point(self, v: Union[Vec3, Sequence[float]]) -> Vec3:
        if isinstance(v, Vec3):
            p = np.array([v.x, v.y, v.z, 1.0], dtype=np.float32)
        else:
            p = np.array([v[0], v[1], v[2], 1.0], dtype=np.float32)
        res = np.dot(self.m, p)
        w = res[3] if abs(res[3]) > 1e-09 else 1.0
        return Vec3(res[0] / w, res[1] / w, res[2] / w)

    def transform_vector(self, v: Union[Vec3, Sequence[float]]) -> Vec3:
        """Transforms direction vector (ignores translation)."""
        if isinstance(v, Vec3):
            p = np.array([v.x, v.y, v.z, 0.0], dtype=np.float32)
        else:
            p = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
        res = np.dot(self.m, p)
        return Vec3(res[0], res[1], res[2])

    @classmethod
    def look_at(cls, eye: Vec3, target: Vec3, up: Vec3) -> "Mat4":
        """
        Creates a camera view matrix looking from 'eye' to 'target'.
        Right-handed coordinate system.
        """
        forward = (target - eye).normalized()
        right = forward.cross(up).normalized()
        if right.length_squared() < 1e-06:
            right = forward.cross(Vec3(0.0, 0.0, 1.0)).normalized()
        true_up = right.cross(forward).normalized()
        view = np.eye(4, dtype=np.float32)
        view[0, 0] = right.x
        view[0, 1] = right.y
        view[0, 2] = right.z
        view[0, 3] = -right.dot(eye)
        view[1, 0] = true_up.x
        view[1, 1] = true_up.y
        view[1, 2] = true_up.z
        view[1, 3] = -true_up.dot(eye)
        view[2, 0] = -forward.x
        view[2, 1] = -forward.y
        view[2, 2] = -forward.z
        view[2, 3] = forward.dot(eye)
        return cls(view)

    @classmethod
    def perspective(cls, fov_rad: float, aspect: float, near: float, far: float) -> "Mat4":
        """
        Creates standard OpenGL-style perspective projection matrix.
        Maps view frustum to normalized device coordinates (NDC) [-1, 1]^3.
        """
        tan_half_fov = math.tan(fov_rad / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1.0 / (aspect * tan_half_fov)
        proj[1, 1] = 1.0 / tan_half_fov
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -(2.0 * far * near) / (far - near)
        proj[3, 2] = -1.0
        return cls(proj)

    @classmethod
    def orthographic(
        cls,
        left: float,
        right: float,
        bottom: float,
        top: float,
        near: float,
        far: float,
    ) -> "Mat4":
        """Creates orthographic projection matrix."""
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = 2.0 / (right - left)
        mat[1, 1] = 2.0 / (top - bottom)
        mat[2, 2] = -2.0 / (far - near)
        mat[0, 3] = -(right + left) / (right - left)
        mat[1, 3] = -(top + bottom) / (top - bottom)
        mat[2, 3] = -(far + near) / (far - near)
        mat[3, 3] = 1.0
        return cls(mat)
