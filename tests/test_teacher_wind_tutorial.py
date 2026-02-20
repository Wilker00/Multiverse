import unittest

from orchestrator.teacher import HighRiskSignals, TeacherConfig, synthesize_tutorial_spec


class TestTeacherWindTutorial(unittest.TestCase):
    def test_synthesize_wind_master_tutorial(self):
        cfg = TeacherConfig.from_dict(
            {
                "target_verse": "cliff_world",
                "tutorial_verse": "wind_master_world",
                "tutorial_max_steps": 70,
            }
        )
        signals = HighRiskSignals(
            verse_name="cliff_world",
            episodes=20,
            events=1000,
            high_risk_events=120,
            high_risk_failures=72,
            high_risk_failure_rate=0.6,
            mean_episode_return=-55.0,
        )
        spec = synthesize_tutorial_spec(target_verse="cliff_world", signals=signals, cfg=cfg, seed=123)
        self.assertEqual(spec.verse_name, "wind_master_world")
        self.assertEqual(int(spec.params.get("max_steps", 0)), 70)
        self.assertIn("target_margin", spec.params)
        self.assertIn("gust_probability", spec.params)
        self.assertIn("edge_penalty", spec.params)
        self.assertIn("concept:safety_margin_navigation", list(spec.tags))


if __name__ == "__main__":
    unittest.main()
