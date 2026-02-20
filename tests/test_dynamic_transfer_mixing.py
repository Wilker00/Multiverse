import json
import os
import tempfile
import unittest

from agents.q_agent import QLearningAgent
from core.agent_base import ExperienceBatch, Transition
from core.types import AgentSpec, SpaceSpec


class TestDynamicTransferMixing(unittest.TestCase):
    def _make_agent(self, *, decay_steps: int) -> QLearningAgent:
        spec = AgentSpec(
            spec_version="v1",
            policy_id="q_mix",
            policy_version="0.1",
            algo="q",
            seed=7,
            config={
                "lr": 0.2,
                "gamma": 0.9,
                "epsilon_start": 0.1,
                "epsilon_min": 0.05,
                "epsilon_decay": 0.99,
                "dynamic_transfer_mix_enabled": True,
                "transfer_mix_start": 1.0,
                "transfer_mix_end": 0.0,
                "transfer_mix_decay_steps": int(decay_steps),
                "transfer_mix_min_rows": 1,
                "transfer_replay_reward_scale": 0.5,
            },
        )
        obs_space = SpaceSpec(type="dict")
        act_space = SpaceSpec(type="discrete", n=2)
        agent = QLearningAgent(spec=spec, observation_space=obs_space, action_space=act_space)
        agent.seed(7)
        return agent

    def test_transfer_mix_replay_decays(self):
        with tempfile.TemporaryDirectory() as td:
            ds_path = os.path.join(td, "transfer.jsonl")
            rows = [
                {"obs": {"s": 0}, "action": 1, "reward": 1.0, "next_obs": {"s": 1}, "done": False},
                {"obs": {"s": 1}, "action": 0, "reward": 0.5, "next_obs": {"s": 2}, "done": False},
            ]
            with open(ds_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            agent = self._make_agent(decay_steps=2)
            warm = agent.learn_from_dataset(ds_path)
            self.assertGreaterEqual(int(warm.get("transfer_rows_loaded", 0)), 2)

            batch = ExperienceBatch(
                transitions=[
                    Transition(
                        obs={"s": 10},
                        action=0,
                        reward=0.0,
                        next_obs={"s": 11},
                        done=False,
                        truncated=False,
                        info={},
                    )
                ],
                meta={},
            )

            m1 = agent.learn(batch)
            self.assertAlmostEqual(float(m1.get("transfer_mix_ratio", 0.0)), 1.0, places=6)
            self.assertGreater(int(m1.get("transfer_replay_updates", 0)), 0)

            m2 = agent.learn(batch)
            self.assertGreater(float(m2.get("transfer_mix_ratio", 0.0)), 0.0)
            self.assertLess(float(m2.get("transfer_mix_ratio", 1.0)), 1.0)

            m3 = agent.learn(batch)
            self.assertEqual(float(m3.get("transfer_mix_ratio", 1.0)), 0.0)
            self.assertEqual(int(m3.get("transfer_replay_updates", 1)), 0)

    def test_disabled_dynamic_mix_no_replay_updates(self):
        spec = AgentSpec(
            spec_version="v1",
            policy_id="q_nomix",
            policy_version="0.1",
            algo="q",
            seed=3,
            config={
                "dynamic_transfer_mix_enabled": False,
            },
        )
        obs_space = SpaceSpec(type="dict")
        act_space = SpaceSpec(type="discrete", n=2)
        agent = QLearningAgent(spec=spec, observation_space=obs_space, action_space=act_space)
        agent.seed(3)
        batch = ExperienceBatch(
            transitions=[
                Transition(
                    obs={"s": 1},
                    action=1,
                    reward=0.2,
                    next_obs={"s": 2},
                    done=False,
                    truncated=False,
                    info={},
                )
            ],
            meta={},
        )
        m = agent.learn(batch)
        self.assertEqual(float(m.get("transfer_mix_ratio", 1.0)), 0.0)
        self.assertEqual(int(m.get("transfer_replay_updates", 1)), 0)


if __name__ == "__main__":
    unittest.main()
