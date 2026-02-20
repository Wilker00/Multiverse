import argparse
import unittest

from core.types import VerseSpec
from tools.run_teacher_remediation import (
    _apply_tutorial_param_overrides,
    _certificate_gate_metric,
    _should_block_next_stage,
    _stage_gate_pass,
)


class TestTeacherRemediationStageGate(unittest.TestCase):
    def test_apply_tutorial_overrides_updates_spec(self):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="wind_master_world",
            verse_version="0.1",
            params={"target_margin": 2, "edge_penalty": -2.0},
            tags=["teacher", "tutorial"],
            metadata={},
        )
        args = argparse.Namespace(
            tutorial_target_margin=4,
            tutorial_gust_probability=0.3,
            tutorial_edge_penalty=-4.2,
            tutorial_margin_reward_scale=0.16,
        )
        updated, overrides = _apply_tutorial_param_overrides(tutorial_spec=spec, args=args)
        self.assertEqual(int(updated.params["target_margin"]), 4)
        self.assertAlmostEqual(float(updated.params["gust_probability"]), 0.3, places=6)
        self.assertAlmostEqual(float(updated.params["edge_penalty"]), -4.2, places=6)
        self.assertAlmostEqual(float(updated.params["margin_reward_scale"]), 0.16, places=6)
        self.assertTrue("override:tutorial_params" in list(updated.tags))
        self.assertEqual(int(overrides["target_margin"]), 4)

    def test_gate_blocks_when_required_stage_fails(self):
        checks = [
            {"stage_index": 1, "pass": True},
            {"stage_index": 2, "pass": False},
        ]
        blocked, reason = _should_block_next_stage(
            stage_gate_enabled=True,
            required_stages=2,
            next_stage_index=3,
            checks=checks,
        )
        self.assertTrue(blocked)
        self.assertTrue(str(reason).startswith("required_stage_failed_"))

    def test_gate_allows_stage_when_requirements_met(self):
        checks = [
            {"stage_index": 1, "pass": True},
            {"stage_index": 2, "pass": True},
        ]
        blocked, reason = _should_block_next_stage(
            stage_gate_enabled=True,
            required_stages=2,
            next_stage_index=3,
            checks=checks,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "requirements_met")

    def test_certificate_metric_selection(self):
        cert = {"observed_violation_rate": 0.2, "upper_bound": 0.35}
        m_obs = _certificate_gate_metric(cert, use_upper_bound=False)
        m_upper = _certificate_gate_metric(cert, use_upper_bound=True)
        self.assertAlmostEqual(float(m_obs), 0.2, places=6)
        self.assertAlmostEqual(float(m_upper), 0.35, places=6)
        self.assertTrue(_stage_gate_pass(m_obs, 0.25))
        self.assertFalse(_stage_gate_pass(m_upper, 0.25))


if __name__ == "__main__":
    unittest.main()
