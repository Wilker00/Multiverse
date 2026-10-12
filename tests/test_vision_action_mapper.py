"""
tests/test_vision_action_mapper.py

Unit tests for the embodiment grounding layer:
- SpatialIntentExtractor
- Planar2LinkIK (analytical and differential)
- VisionActionMapper
"""

import math
import unittest

from integrations.vision_action_mapper import (
    Planar2LinkIK,
    SpatialIntentExtractor,
    VisionActionMapper,
)


class TestVisionActionMapper(unittest.TestCase):
    def setUp(self):
        self.ik = Planar2LinkIK(l1=0.5, l2=0.5)
        self.mapper = VisionActionMapper(l1=0.5, l2=0.5)

    def test_spatial_intent_extractor(self):
        # 0: Up (0, 1), 1: Down (0, -1), 2: Left (-1, 0), 3: Right (1, 0)
        up = SpatialIntentExtractor.action_to_displacement(0, magnitude=0.2)
        self.assertAlmostEqual(up[0], 0.0)
        self.assertAlmostEqual(up[1], 0.2)

        right = SpatialIntentExtractor.action_to_displacement(3, magnitude=0.5)
        self.assertAlmostEqual(right[0], 0.5)
        self.assertAlmostEqual(right[1], 0.0)

    def test_forward_and_analytical_ik(self):
        # Target at (0.5, 0.5) -> radius ~ 0.707 <= 1.0 (reachable)
        target_x, target_y = 0.5, 0.5
        solution = self.ik.analytical_ik(target_x, target_y)
        self.assertIsNotNone(solution)
        th1, th2 = solution
        _, ee = self.ik.forward_kinematics(th1, th2)
        self.assertAlmostEqual(ee[0], target_x, places=4)
        self.assertAlmostEqual(ee[1], target_y, places=4)

    def test_differential_ik(self):
        th1, th2 = 0.3, 0.5
        # Request small motion in +y
        dth1, dth2 = self.ik.differential_ik(th1, th2, dx=0.0, dy=0.05)
        self.assertIsInstance(dth1, float)
        self.assertIsInstance(dth2, float)
        self.assertTrue(abs(dth1) > 0 or abs(dth2) > 0)

    def test_memory_trajectory_grounding(self):
        game_transitions = [
            {"action": 0},  # Up
            {"action": 0},  # Up
            {"action": 3},  # Right
            {"action": 3},  # Right
        ]
        robot_actions = self.mapper.ground_memory_trajectory(game_transitions, initial_th1=0.2, initial_th2=0.4)
        self.assertEqual(len(robot_actions), 4)
        for act in robot_actions:
            self.assertIn(act, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
