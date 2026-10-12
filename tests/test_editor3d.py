"""Picking and axis-drag math for the 3D editor."""

import unittest

from engine3d.math3d import Vec3
from engine3d.scene import Camera
from tools.editor3d import EditorObject, mesh_for, pick_object, ray_aabb, ray_plane


class TestEditorMath(unittest.TestCase):
    def test_ray_hits_box_in_front(self):
        hit = ray_aabb(Vec3(0, 0, 5), Vec3(0, 0, -1), (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit, 4.5, places=4)

    def test_ray_misses_box(self):
        hit = ray_aabb(Vec3(3, 0, 5), Vec3(0, 0, -1), (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
        self.assertIsNone(hit)

    def test_pick_selects_the_box(self):
        obj = EditorObject("Box", "box", Vec3(0.0, 0.0, 0.0))
        meshes = {("box", obj.color): mesh_for("box", obj.color)}
        camera = Camera(position=Vec3(0, 0, 5), target=Vec3(0, 0, 0), fov_deg=50, aspect=1, near=0.05, far=50)
        center = pick_object([obj], meshes, camera, 80, 80, 160, 160)
        self.assertEqual(center, 0)
        miss = pick_object([obj], meshes, camera, 4, 4, 160, 160)
        self.assertIsNone(miss)

    def test_axis_drag_moves_along_x(self):
        normal = Vec3(0, 1, 0)
        anchor = Vec3(0, 0, 0)
        hit = ray_plane(Vec3(2, 5, 0), Vec3(0, -1, 0), anchor, normal)
        self.assertIsNotNone(hit)
        delta = (hit - anchor).dot(Vec3(1, 0, 0))
        self.assertAlmostEqual(delta, 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
