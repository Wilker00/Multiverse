import os
import tempfile
import unittest

from core.sim2real import build_sim2real_profile
from tools.evaluate_sim2real import assess_sim2real, summarize_sim2real_results


class TestSim2Real(unittest.TestCase):
    def test_curated_warehouse_profile_changes_dynamics(self):
        profile = build_sim2real_profile(
            verse_name="warehouse_world",
            base_params={"max_steps": 20},
            profile="moderate",
        )
        self.assertEqual(profile.support_level, "curated")
        self.assertIn("obstacle_count", profile.param_changes)
        self.assertIn("battery_drain", profile.param_changes)
        self.assertIn("lidar_range", profile.param_changes)
        self.assertGreater(int(profile.params["obstacle_count"]), 14)
        self.assertGreater(int(profile.params["battery_drain"]), 1)
        self.assertLess(int(profile.params["lidar_range"]), 4)

    def test_unknown_profile_falls_back_to_adr(self):
        profile = build_sim2real_profile(
            verse_name="line_world",
            base_params={"max_steps": 12},
            profile="mild",
        )
        self.assertEqual(profile.support_level, "adr_fallback")
        self.assertTrue(bool(profile.params.get("adr_enabled", False)))
        self.assertGreater(float(profile.params.get("adr_jitter", 0.0)), 0.0)

    def test_summary_computes_pass_fail(self):
        summary = summarize_sim2real_results(
            [
                {"name": "nominal", "metrics": {"mean_return": 10.0, "success_rate": 0.9}},
                {"name": "mild", "support_level": "curated", "metrics": {"mean_return": 9.2, "success_rate": 0.82}},
                {"name": "severe", "support_level": "curated", "metrics": {"mean_return": 6.0, "success_rate": 0.60}},
            ],
            max_success_rate_drop=0.15,
            max_return_drop=2.0,
        )
        self.assertFalse(bool(summary["passed"]))
        self.assertEqual(len(summary["comparisons"]), 2)
        by_name = {row["name"]: row for row in summary["comparisons"]}
        self.assertTrue(bool(by_name["mild"]["passed"]))
        self.assertFalse(bool(by_name["severe"]["passed"]))

    def test_assess_sim2real_smoke_random_agent(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            report = assess_sim2real(
                verse_name="warehouse_world",
                algo="random",
                episodes=2,
                max_steps=12,
                seed=7,
                runs_root=runs_root,
                profiles=["mild"],
            )
            self.assertEqual(report["verse_name"], "warehouse_world")
            self.assertEqual(len(report["profiles"]), 2)
            names = [row["name"] for row in report["profiles"]]
            self.assertEqual(names, ["nominal", "mild"])
            self.assertIn("summary", report)
            self.assertIn("passed", report["summary"])


if __name__ == "__main__":
    unittest.main()
