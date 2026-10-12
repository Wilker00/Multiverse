"""
tests/test_rendered_worlds.py

Unit tests for built-in visual rendered worlds:
- robot_reacher_world
- embodied_point_world
- warehouse_3d_world
"""

import unittest
import numpy as np

from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


class TestRenderedWorlds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_builtin()

    def test_robot_reacher_world(self):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="robot_reacher_world",
            verse_version="0.0",
            seed=42,
        )
        verse = create_verse(spec)
        reset_res = verse.reset()
        self.assertIn("distance", reset_res.info)
        self.assertEqual(len(reset_res.obs), 8)

        # Step through all discrete actions
        for act in range(5):
            step_res = verse.step(act)
            self.assertIn("distance", step_res.info)
            self.assertEqual(len(step_res.obs), 8)

        # Verify ANSI render
        ansi = verse.render(mode="ansi")
        self.assertIsInstance(ansi, str)
        self.assertIn("Reacher", ansi)

        # Verify RGB array render
        frame = verse.render(mode="rgb_array")
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.dtype, np.uint8)

    def test_embodied_point_world(self):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="embodied_point_world",
            verse_version="0.0",
            seed=42,
        )
        verse = create_verse(spec)
        reset_res = verse.reset()
        self.assertIn("distance", reset_res.info)
        self.assertEqual(len(reset_res.obs), 6)

        # Step actions
        for act in range(5):
            step_res = verse.step(act)
            self.assertIn("distance", step_res.info)

        # Verify ANSI render
        ansi = verse.render(mode="ansi")
        self.assertIsInstance(ansi, str)
        self.assertIn("point", ansi)

        # Verify RGB array render
        frame = verse.render(mode="rgb_array")
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.dtype, np.uint8)

    def test_warehouse_3d_world(self):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="warehouse_3d_world",
            verse_version="0.0",
            seed=42,
            params={"render_width": 80, "render_height": 60},
        )
        verse = create_verse(spec)
        reset_res = verse.reset()
        self.assertIn("distance", reset_res.info)
        self.assertEqual(len(reset_res.obs), 7)

        for act in range(5):
            step_res = verse.step(act)
            self.assertIn("distance", step_res.info)
            self.assertEqual(len(step_res.obs), 7)

        ansi = verse.render(mode="ansi")
        self.assertIsInstance(ansi, str)
        self.assertIn("warehouse3d", ansi)

        frame = verse.render(mode="rgb_array")
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (60, 80, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(frame)), 0)


if __name__ == "__main__":
    unittest.main()
