"""
engine3d/rasterizer.py

Z-Buffered 3D Software Rasterizer.
Transforms 3D scene geometry through the perspective graphics pipeline,
performs backface culling, depth testing, and Lambertian lighting,
and outputs a raw RGB NumPy framebuffer.
Zero external GUI/engine dependencies.
"""

from __future__ import annotations

import math

import numpy as np

from engine3d.scene import Scene


class Rasterizer:
    """Software 3D Rasterizer with Z-buffer and Lambertian shading."""

    def __init__(self, width: int = 320, height: int = 240) -> None:
        self.width = max(16, int(width))
        self.height = max(16, int(height))

    def render(self, scene: Scene) -> np.ndarray:
        """
        Renders the given 3D scene to an RGB array of shape (height, width, 3).
        """
        w = self.width
        h = self.height
        aspect = w / h
        scene.camera.aspect = aspect

        view = scene.camera.view_matrix()
        proj = scene.camera.projection_matrix()
        vp = proj * view
        vp_m = vp.m

        color_buffer = np.full((h, w, 3), scene.background_color, dtype=np.uint8)
        depth_buffer = np.full((h, w), float("inf"), dtype=np.float32)

        if scene.lights:
            light = scene.lights[0]
        else:
            light = None
        if light:
            light_dir = (-light.direction).to_array()
            ambient = light.ambient
        else:
            light_dir = np.array([0.5, 0.8, 0.3], dtype=np.float32)
            light_dir /= np.linalg.norm(light_dir)
            ambient = 0.3

        meshes = scene.get_render_queue()
        for mesh, world_matrix in meshes:
            verts = mesh.vertices
            faces = mesh.faces
            normals = mesh.normals
            colors = mesh.colors
            wm = world_matrix.m

            verts_w = ((wm[:3, :3] @ verts.T) + wm[:3, 3:4]).T
            normals_w = (wm[:3, :3] @ normals.T).T
            norm_lens = np.linalg.norm(normals_w, axis=1, keepdims=True)
            norm_lens[norm_lens == 0] = 1e-09
            normals_w = normals_w / norm_lens

            verts_h = np.hstack([verts_w, np.ones((len(verts_w), 1), dtype=np.float32)])
            clip_v = (vp_m @ verts_h.T).T

            for f_idx in range(len(faces)):
                i0, i1, i2 = faces[f_idx]
                c0, c1, c2 = clip_v[i0], clip_v[i1], clip_v[i2]
                if c0[3] < 0.05 or c1[3] < 0.05 or c2[3] < 0.05:
                    continue

                ndc0 = c0[:3] / c0[3]
                ndc1 = c1[:3] / c1[3]
                ndc2 = c2[:3] / c2[3]

                x0 = (ndc0[0] + 1.0) * 0.5 * (w - 1)
                y0 = (1.0 - ndc0[1]) * 0.5 * (h - 1)
                z0 = ndc0[2]
                x1 = (ndc1[0] + 1.0) * 0.5 * (w - 1)
                y1 = (1.0 - ndc1[1]) * 0.5 * (h - 1)
                z1 = ndc1[2]
                x2 = (ndc2[0] + 1.0) * 0.5 * (w - 1)
                y2 = (1.0 - ndc2[1]) * 0.5 * (h - 1)
                z2 = ndc2[2]

                area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                if area <= 0:
                    continue

                norm = normals_w[f_idx]
                dot = max(0.0, float(np.dot(norm, light_dir)))
                intensity = min(1.0, ambient + (1.0 - ambient) * dot)
                base_col = colors[f_idx]
                shaded_col = (base_col.astype(np.float32) * intensity).astype(np.uint8)

                min_x = max(0, int(math.floor(min(x0, x1, x2))))
                max_x = min(w - 1, int(math.ceil(max(x0, x1, x2))))
                min_y = max(0, int(math.floor(min(y0, y1, y2))))
                max_y = min(h - 1, int(math.ceil(max(y0, y1, y2))))
                if min_x > max_x or min_y > max_y:
                    continue

                xs = np.arange(min_x, max_x + 1, dtype=np.float32)
                ys = np.arange(min_y, max_y + 1, dtype=np.float32)
                grid_x, grid_y = np.meshgrid(xs, ys)

                w0 = (x2 - x1) * (grid_y - y1) - (y2 - y1) * (grid_x - x1)
                w1 = (x0 - x2) * (grid_y - y2) - (y0 - y2) * (grid_x - x2)
                w2 = (x1 - x0) * (grid_y - y0) - (y1 - y0) * (grid_x - x0)
                inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                if not np.any(inside):
                    continue

                inv_area = 1.0 / area
                lam0 = w0 * inv_area
                lam1 = w1 * inv_area
                lam2 = w2 * inv_area
                depth = lam0 * z0 + lam1 * z1 + lam2 * z2

                sub_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
                closer = inside & (depth < sub_depth)
                if not np.any(closer):
                    continue
                sub_depth[closer] = depth[closer]
                color_buffer[min_y : max_y + 1, min_x : max_x + 1][closer] = shaded_col

        return color_buffer
