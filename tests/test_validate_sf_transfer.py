import unittest
from types import SimpleNamespace

from tools.validate_sf_transfer import (
    EgoGridAdapter,
    _parse_int_grid,
    _parse_seed_list,
    _parse_str_list,
    _score_learned_softmax_model,
    _softmax,
)


class TestValidateSFTransfer(unittest.TestCase):
    def test_parse_helpers_dedupe_and_defaults(self):
        self.assertEqual(_parse_seed_list("3, 1; 3"), [1, 3])
        self.assertEqual(_parse_int_grid("", default=[8, 4, 8]), [4, 8])
        self.assertEqual(_parse_str_list("", default=[" maze_world ", "", "grid_world"]), ["maze_world", "grid_world"])

    def test_parse_seed_list_rejects_empty(self):
        with self.assertRaises(ValueError):
            _parse_seed_list(" , ; ")

    def test_grid_adapter_marks_obstacles_and_projects_goal_direction(self):
        adapter = EgoGridAdapter(size=5)
        verse = SimpleNamespace(
            params=SimpleNamespace(width=4, height=4),
            _obstacles={(2, 1)},
        )
        ego = adapter.from_grid_world(
            verse,
            {
                "x": 1,
                "y": 1,
                "goal_x": 10,
                "goal_y": 1,
            },
        )

        self.assertEqual(int(ego.occupancy[2, 3]), 1)
        self.assertEqual(int(ego.goal[2, 4]), 1)
        self.assertEqual(adapter.state_key(ego).count("|"), 1)

    def test_warehouse_adapter_uses_lidar_and_phi_has_expected_shape(self):
        adapter = EgoGridAdapter(size=5)
        ego = adapter.from_warehouse_world(
            {
                "x": 2,
                "y": 2,
                "goal_x": 2,
                "goal_y": -5,
                "lidar": [1, 3, 3, 3, 3, 3, 3, 2],
            }
        )

        self.assertEqual(int(ego.occupancy[1, 2]), 1)
        self.assertEqual(int(ego.goal[0, 2]), 1)
        self.assertEqual(adapter.phi(ego).shape[0], 1 + (5 * 5) + (5 * 5))

    def test_softmax_helpers_remain_stable_for_invalid_model_shapes(self):
        probs = _softmax([1000.0, 999.0, -1000.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

        scored = _score_learned_softmax_model(
            features={"hazard_gain_mean": 1.5},
            model_block={
                "feature_names": ["hazard_gain_mean", "episodes"],
                "class_names": ["sf_scratch", "sf_transfer"],
                "weights": [[1.0], [2.0], [3.0]],
                "bias": [0.2, -0.2],
            },
        )
        self.assertEqual(scored["class_names"], ["sf_scratch", "sf_transfer"])
        self.assertAlmostEqual(sum(scored["probs"]), 1.0, places=6)
        self.assertEqual(scored["feature_count"], 2)


if __name__ == "__main__":
    unittest.main()
