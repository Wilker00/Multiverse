import unittest

from agents.planner_recall_agent import PlannerRecallAgent
from core.types import AgentSpec, SpaceSpec


class TestPlannerRecallAgent(unittest.TestCase):
    def _agent(self, cfg=None):
        spec = AgentSpec(
            spec_version="v1",
            policy_id="planner",
            policy_version="0.1",
            algo="planner_recall",
            config=(cfg or {}),
        )
        obs_space = SpaceSpec(type="dict")
        act_space = SpaceSpec(type="discrete", n=3)
        ag = PlannerRecallAgent(spec=spec, observation_space=obs_space, action_space=act_space)
        ag.seed(7)
        return ag

    def test_phase_family_selection_early_mid_late(self):
        ag = self._agent(
            {
                "planner_force_recall": True,
                "planner_expected_horizon": 30,
                "planner_mid_start": 0.33,
                "planner_late_start": 0.66,
                "recall_cooldown_steps": 1,
            }
        )
        r0 = ag.memory_query_request(obs={"t": 0}, step_idx=0)
        r1 = ag.memory_query_request(obs={"t": 12}, step_idx=1)
        r2 = ag.memory_query_request(obs={"t": 28}, step_idx=2)
        self.assertEqual(list(r0.get("memory_families", [])), ["declarative"])
        self.assertEqual(list(r1.get("memory_families", [])), ["declarative"])
        self.assertEqual(list(r2.get("memory_families", [])), ["procedural"])
        self.assertEqual(str(r2.get("reason")), "phase_late_planner")

    def test_non_forced_mode_appends_phase_family(self):
        ag = self._agent(
            {
                "planner_force_recall": False,
                "recall_risk_threshold": 1.0,
                "recall_cooldown_steps": 1,
            }
        )
        req = ag.memory_query_request(obs={"risk": 9, "t": 0}, step_idx=0)
        self.assertIsNotNone(req)
        self.assertIn("memory_families", req)
        self.assertEqual(list(req.get("memory_families", [])), ["declarative"])
        self.assertIn("phase_early", str(req.get("reason", "")))


if __name__ == "__main__":
    unittest.main()

