import json
import os
import tempfile
import unittest

from agents.q_agent import QLearningAgent, obs_key
from core.types import AgentSpec, SpaceSpec


class TestQAgentWarmstartTransferScore(unittest.TestCase):
    def _make_agent(
        self,
        *,
        use_transfer_score: bool,
        warmstart_target: str = "immediate",
        warmstart_target_gamma: float = 0.99,
    ) -> QLearningAgent:
        spec = AgentSpec(
            spec_version="v1",
            policy_id="q_ws",
            policy_version="0.1",
            algo="q",
            seed=11,
            config={
                "warmstart_reward_scale": 1.0,
                "warmstart_use_transfer_score": bool(use_transfer_score),
                "warmstart_transfer_score_min": 0.0,
                "warmstart_transfer_score_max": 2.0,
                "warmstart_target": str(warmstart_target),
                "warmstart_target_gamma": float(warmstart_target_gamma),
            },
        )
        obs_space = SpaceSpec(type="dict")
        act_space = SpaceSpec(type="discrete", n=2)
        agent = QLearningAgent(spec=spec, observation_space=obs_space, action_space=act_space)
        agent.seed(11)
        return agent

    def test_warmstart_transfer_score_scales_q_update(self):
        with tempfile.TemporaryDirectory() as td:
            ds = os.path.join(td, "transfer.jsonl")
            row = {
                "obs": {"s": 0},
                "action": 0,
                "reward": 1.0,
                "next_obs": {"s": 1},
                "done": False,
                "transfer_score": 0.2,
            }
            with open(ds, "w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            plain = self._make_agent(use_transfer_score=False)
            weighted = self._make_agent(use_transfer_score=True)

            plain.learn_from_dataset(ds)
            m = weighted.learn_from_dataset(ds)

            key = obs_key({"s": 0})
            plain_q = float(plain.q[key][0])
            weighted_q = float(weighted.q[key][0])
            self.assertGreater(plain_q, weighted_q)
            self.assertAlmostEqual(plain_q, 1.0, places=6)
            self.assertAlmostEqual(weighted_q, 0.2, places=6)
            self.assertTrue(bool(m.get("warmstart_transfer_score_weighted", False)))

    def test_warmstart_return_to_go_changes_early_credit_assignment(self):
        with tempfile.TemporaryDirectory() as td:
            ds = os.path.join(td, "rtg.jsonl")
            rows = [
                {
                    "episode_id": "ep1",
                    "step_idx": 0,
                    "obs": {"s": 0},
                    "action": 1,
                    "reward": -0.02,
                    "next_obs": {"s": 1},
                    "done": False,
                },
                {
                    "episode_id": "ep1",
                    "step_idx": 1,
                    "obs": {"s": 1},
                    "action": 1,
                    "reward": 1.0,
                    "next_obs": {"s": 2},
                    "done": True,
                },
            ]
            with open(ds, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            immediate = self._make_agent(use_transfer_score=False, warmstart_target="immediate")
            rtg = self._make_agent(
                use_transfer_score=False,
                warmstart_target="return_to_go",
                warmstart_target_gamma=1.0,
            )

            immediate.learn_from_dataset(ds)
            m = rtg.learn_from_dataset(ds)

            s0 = obs_key({"s": 0})
            q0_immediate = float(immediate.q[s0][1])
            q0_rtg = float(rtg.q[s0][1])
            self.assertLess(q0_immediate, 0.0)
            self.assertGreater(q0_rtg, 0.0)
            self.assertEqual(str(m.get("warmstart_target_mode")), "return_to_go")

    def test_grid_obs_key_mode_can_ignore_time(self):
        a = obs_key(
            {"x": 1, "y": 2, "goal_x": 4, "goal_y": 4, "t": 3},
            grid_mode="xy_goal",
        )
        b = obs_key(
            {"x": 1, "y": 2, "goal_x": 4, "goal_y": 4, "t": 99},
            grid_mode="xy_goal",
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
