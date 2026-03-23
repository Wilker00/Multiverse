import unittest

from integrations.local_visual_sim import preview_local_visual_sim
from integrations.sim_registry import get_sim_provider, list_sim_provider_status


class TestMultiverseSim(unittest.TestCase):
    def test_sim_provider_status_includes_local_backend(self):
        rows = list_sim_provider_status()
        by_id = {row["provider_id"]: row for row in rows}
        self.assertIn("multiverse_local", by_id)
        self.assertTrue(bool(by_id["multiverse_local"]["available"]))
        self.assertTrue(bool(by_id["multiverse_local"]["implemented"]))
        self.assertIn("lightweight_visual", by_id["multiverse_local"]["capabilities"])

    def test_get_sim_provider_exposes_isaac_placeholder(self):
        provider = get_sim_provider("isaaclab")
        self.assertEqual(provider.provider_id, "isaaclab")
        self.assertTrue(bool(provider.external))
        self.assertTrue(bool(provider.implemented))
        self.assertIn("isaac_warehouse_nav", provider.supported_verses)

    def test_preview_local_visual_sim_returns_frames(self):
        report = preview_local_visual_sim(
            verse_name="line_world",
            verse_params={"goal_pos": 4, "max_steps": 6},
            episodes=1,
            max_steps=4,
            seed=9,
        )
        self.assertEqual(report["provider_id"], "multiverse_local")
        self.assertEqual(report["verse_name"], "line_world")
        self.assertEqual(len(report["episodes"]), 1)
        ep = report["episodes"][0]
        self.assertIn("|", ep["initial_frame"])
        self.assertGreaterEqual(len(ep["frames"]), 1)


if __name__ == "__main__":
    unittest.main()
