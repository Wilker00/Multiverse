import unittest

from core.universe_registry import (
    build_transfer_source_plan,
    primary_universe_for_verse,
    same_universe_verses,
    source_transfer_lane,
    universes_for_verse,
)


class TestUniverseRegistry(unittest.TestCase):
    def test_warehouse_maps_to_logistics_universe(self):
        self.assertEqual(primary_universe_for_verse("warehouse_world"), "logistics_universe")
        self.assertIn("logistics_universe", universes_for_verse("warehouse_world"))

    def test_warehouse_same_universe_sources_prioritize_logistics_verses(self):
        srcs = same_universe_verses("warehouse_world")
        self.assertGreaterEqual(len(srcs), 5)
        self.assertEqual(srcs[0], "factory_world")
        self.assertIn("grid_world", srcs)
        self.assertNotIn("warehouse_world", srcs)

    def test_transfer_source_plan_splits_near_and_far(self):
        plan = build_transfer_source_plan("warehouse_world")
        near = set(plan.get("near_sources", []))
        far = set(plan.get("far_sources", []))
        self.assertIn("factory_world", near)
        self.assertIn("chess_world", far)
        self.assertTrue(near.isdisjoint(far))
        self.assertEqual(plan.get("target_universe"), "logistics_universe")

    def test_lane_classification(self):
        self.assertEqual(source_transfer_lane("factory_world", "warehouse_world"), "near_universe")
        self.assertEqual(source_transfer_lane("chess_world", "warehouse_world"), "far_universe")
        self.assertEqual(source_transfer_lane("warehouse_world", "warehouse_world"), "same_verse")


if __name__ == "__main__":
    unittest.main()

