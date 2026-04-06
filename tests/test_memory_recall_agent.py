import unittest

import numpy as np

from agents.memory_recall_agent import MemoryRecallAgent
from agents.q_agent import obs_key
from core.types import AgentSpec, SpaceSpec


class TestMemoryRecallAgent(unittest.TestCase):
    def _agent(self, cfg=None):
        spec = AgentSpec(
            spec_version="v1",
            policy_id="mr",
            policy_version="0.1",
            algo="memory_recall",
            config=(cfg or {}),
        )
        obs_space = SpaceSpec(type="dict")
        act_space = SpaceSpec(type="discrete", n=3)
        ag = MemoryRecallAgent(spec=spec, observation_space=obs_space, action_space=act_space)
        ag.seed(123)
        return ag

    def test_memory_query_request_triggers_on_high_risk(self):
        ag = self._agent(
            {
                "verse_name": "chess_world",
                "recall_risk_key": "risk",
                "recall_risk_threshold": 5.0,
            }
        )
        req = ag.memory_query_request(obs={"risk": 9, "phase": 1}, step_idx=0)
        self.assertIsNotNone(req)
        self.assertEqual(str(req.get("reason")), "high_risk")
        self.assertEqual(str(req.get("verse_name")), "chess_world")

    def test_memory_query_request_respects_cooldown(self):
        ag = self._agent({"recall_cooldown_steps": 3})
        r1 = ag.memory_query_request(obs={"risk": 9}, step_idx=0)
        r2 = ag.memory_query_request(obs={"risk": 9}, step_idx=1)
        self.assertIsNotNone(r1)
        self.assertIsNone(r2)

    def test_memory_query_request_includes_bias_filters(self):
        ag = self._agent(
            {
                "verse_name": "warehouse_world",
                "recall_risk_threshold": 1.0,
                "recall_policy_ids": ["transfer_q_warehouse_world"],
                "recall_exclude_policy_ids": ["baseline_q_warehouse_world"],
                "recall_source_verse_names": ["grid_world", "cliff_world"],
                "recall_min_transfer_score": 0.7,
                "recall_min_transfer_confidence": 0.6,
            }
        )
        req = ag.memory_query_request(obs={"risk": 9}, step_idx=0)
        self.assertIsNotNone(req)
        assert isinstance(req, dict)
        self.assertEqual(list(req.get("policy_ids", [])), ["transfer_q_warehouse_world"])
        self.assertEqual(list(req.get("exclude_policy_ids", [])), ["baseline_q_warehouse_world"])
        self.assertEqual(list(req.get("source_verse_names", [])), ["cliff_world", "grid_world"])
        self.assertEqual(float(req.get("min_transfer_score", -1.0)), 0.7)
        self.assertEqual(float(req.get("min_transfer_confidence", -1.0)), 0.6)

    def test_memory_bootstrap_request_reuses_recall_filters(self):
        ag = self._agent(
            {
                "verse_name": "warehouse_world",
                "bootstrap_recall_enabled": True,
                "recall_policy_ids": ["transfer_q_warehouse_world"],
                "recall_exclude_policy_ids": ["baseline_q_warehouse_world"],
                "recall_source_verse_names": ["grid_world"],
                "recall_min_transfer_score": 0.7,
                "recall_min_transfer_confidence": 0.6,
            }
        )
        req = ag.memory_bootstrap_request(obs={"pos": 0}, step_idx=0)
        self.assertIsNotNone(req)
        assert isinstance(req, dict)
        self.assertEqual(str(req.get("reason", "")), "episode_bootstrap")
        self.assertEqual(str(req.get("verse_name", "")), "warehouse_world")
        self.assertEqual(list(req.get("policy_ids", [])), ["transfer_q_warehouse_world"])
        self.assertEqual(list(req.get("exclude_policy_ids", [])), ["baseline_q_warehouse_world"])
        self.assertEqual(list(req.get("source_verse_names", [])), ["grid_world"])
        self.assertEqual(float(req.get("min_transfer_score", -1.0)), 0.7)
        self.assertEqual(float(req.get("min_transfer_confidence", -1.0)), 0.6)

    def test_act_with_hint_uses_memory_prior(self):
        ag = self._agent(
            {
                "epsilon_start": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 1.0,
                "recall_vote_weight": 2.0,
            }
        )
        obs = {"pos": 0, "goal": 4, "t": 0}
        k = obs_key(obs)
        ag.q[k] = np.asarray([1.0, 0.1, 0.2], dtype=np.float32)
        res0 = ag.act(obs)
        self.assertEqual(int(res0.action), 0)

        hint = {
            "memory_recall": {
                "reason": "high_risk",
                "matches": [
                    {
                        "action": 2,
                        "score": 1.0,
                        "pointer_path": "runs/run_mem/events.jsonl#episode_id=e1;step_idx=3",
                    }
                ],
            }
        }
        res1 = ag.act_with_hint(obs, hint)
        self.assertEqual(int(res1.action), 2)
        self.assertTrue(bool(res1.info.get("memory_recall_used", False)))
        self.assertIn("runs/run_mem/events.jsonl", str(res1.info.get("memory_recall_pointer", "")))


if __name__ == "__main__":
    unittest.main()
