import unittest

from memory.universe_adapters import transfer_row_universe_metadata, universe_features_for_obs


class TestUniverseAdapters(unittest.TestCase):
    def test_logistics_features_for_warehouse_obs(self):
        obs = {
            "x": 1,
            "y": 2,
            "goal_x": 7,
            "goal_y": 7,
            "battery": 18,
            "nearby_obstacles": 2,
            "nearest_charger_dist": 4,
            "patrol_dist": 3,
            "t": 9,
        }
        nxt = {
            "x": 2,
            "y": 2,
            "goal_x": 7,
            "goal_y": 7,
            "battery": 17,
            "nearby_obstacles": 1,
            "nearest_charger_dist": 3,
            "patrol_dist": 4,
            "t": 10,
        }
        out = universe_features_for_obs(verse_name="warehouse_world", obs=obs, next_obs=nxt)
        self.assertIsInstance(out, dict)
        assert isinstance(out, dict)
        self.assertEqual(out.get("universe_id"), "logistics_universe")
        self.assertEqual(out.get("feature_space"), "logistics_shared_v1")
        feats = out.get("features", {})
        self.assertIn("goal_progress", feats)
        self.assertIn("hazard_proximity", feats)
        self.assertIn("resource_level", feats)
        self.assertIn("throughput", feats)
        for k in ("goal_progress", "hazard_proximity", "resource_level", "queue_pressure", "throughput", "congestion", "time_pressure"):
            self.assertGreaterEqual(float(feats.get(k, -1.0)), 0.0)
            self.assertLessEqual(float(feats.get(k, 2.0)), 1.0)

    def test_logistics_features_for_factory_obs(self):
        obs = {
            "t": 5,
            "completed": 4,
            "total_arrived": 12,
            "output_buf": 3,
            "buf_0": 1,
            "buf_1": 2,
            "buf_2": 0,
            "broken_0": 0,
            "broken_1": 1,
            "broken_2": 0,
        }
        nxt = dict(obs)
        nxt["completed"] = 6
        out = universe_features_for_obs(verse_name="factory_world", obs=obs, next_obs=nxt)
        self.assertIsNotNone(out)
        assert isinstance(out, dict)
        feats = out.get("features", {})
        self.assertGreaterEqual(float(feats.get("throughput", 0.0)), 0.5)
        self.assertGreaterEqual(float(feats.get("queue_pressure", 0.0)), 0.0)

    def test_transfer_row_universe_metadata_marks_same_universe(self):
        translated_obs = {
            "x": 3,
            "y": 3,
            "goal_x": 7,
            "goal_y": 7,
            "battery": 20,
            "nearby_obstacles": 1,
            "t": 4,
        }
        meta = transfer_row_universe_metadata(
            source_verse="factory_world",
            target_verse="warehouse_world",
            translated_obs=translated_obs,
            translated_next_obs=translated_obs,
        )
        self.assertIsNotNone(meta)
        assert isinstance(meta, dict)
        self.assertTrue(bool(meta.get("same_universe", False)))
        self.assertEqual(str(meta.get("target_universe")), "logistics_universe")
        adapter = meta.get("adapter", {})
        self.assertEqual(str(adapter.get("feature_space")), "logistics_shared_v1")


if __name__ == "__main__":
    unittest.main()

