"""Tests for the engine3d software creation pipeline."""

import unittest

import numpy as np

from engine3d.geometry import create_box, create_cylinder, create_plane, create_sphere
from engine3d.math3d import Mat4, Vec3
from engine3d.rasterizer import Rasterizer
from engine3d.scene import Node, Scene, Transform


class TestMath3D(unittest.TestCase):
    def test_vec_ops(self):
        a = Vec3(1, 2, 3)
        b = Vec3(0, 1, 0)
        self.assertEqual((a + b).y, 3)
        self.assertEqual((a - b).y, 1)
        self.assertAlmostEqual(a.dot(b), 2.0)
        self.assertAlmostEqual(a.cross(b).x, -3.0)
        self.assertAlmostEqual(Vec3(3, 0, 0).normalized().x, 1.0)

    def test_transform_point(self):
        moved = Mat4.translation(2, 0, -1).transform_point(Vec3(1, 1, 1))
        self.assertAlmostEqual(moved.x, 3.0)
        self.assertAlmostEqual(moved.z, 0.0)

    def test_look_at_puts_target_in_front(self):
        view = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        origin = view.transform_point(Vec3(0, 0, 0))
        self.assertLess(origin.z, 0.0)


class TestGeometry(unittest.TestCase):
    def test_primitives(self):
        box = create_box()
        self.assertEqual(len(box.faces), 12)
        self.assertEqual(len(box.normals), 12)
        self.assertEqual(len(create_plane().faces), 2)
        self.assertGreater(len(create_cylinder().faces), 8)
        self.assertGreater(len(create_sphere().faces), 8)
        up = create_plane().normals[0]
        self.assertGreater(up[1], 0.9)


class TestSceneAndRaster(unittest.TestCase):
    def test_hierarchy_translation(self):
        parent = Node("parent", Transform(position=Vec3(1, 0, 0)))
        child = Node("child", Transform(position=Vec3(0, 2, 0)), create_box())
        parent.add_child(child)
        world = child.world_matrix().transform_point(Vec3(0, 0, 0))
        self.assertAlmostEqual(world.x, 1.0, places=4)
        self.assertAlmostEqual(world.y, 2.0, places=4)

    def test_rasterizer_draws_box(self):
        scene = Scene(background_color=(0, 0, 0))
        scene.camera.position = Vec3(2.5, 2.0, 3.5)
        scene.camera.target = Vec3(0, 0, 0)
        scene.add_node(Node("box", Transform(), create_box(1, 1, 1, (220, 40, 40))))
        frame = Rasterizer(80, 60).render(scene)
        self.assertEqual(frame.shape, (60, 80, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(frame)), 0)


if __name__ == "__main__":
    unittest.main()
