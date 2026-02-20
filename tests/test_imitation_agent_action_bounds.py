import json
import os
import tempfile
import unittest

from agents.imitation_agent import ImitationLookupAgent
from core.types import AgentSpec, SpaceSpec


class TestImitationAgentActionBounds(unittest.TestCase):
    def test_learn_skips_out_of_range_discrete_actions(self):
        with tempfile.TemporaryDirectory() as td:
            ds = os.path.join(td, "mixed_actions.jsonl")
            rows = [
                {"obs": {"x": 0, "t": 0}, "action": 0, "reward": 1.0},
                {"obs": {"x": 1, "t": 1}, "action": 1, "reward": 1.0},
                # Invalid for n=2 and used to trigger MLP target OOB in mixed datasets.
                {"obs": {"x": 2, "t": 2}, "action": 4, "reward": 1.0},
            ]
            with open(ds, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            spec = AgentSpec(
                spec_version="v1",
                policy_id="imitation_test",
                policy_version="0.1",
                algo="imitation_lookup",
                seed=7,
                config={
                    "enable_mlp_generalizer": True,
                    "mlp_min_rows": 1,
                    "mlp_epochs": 1,
                    "mlp_batch_size": 2,
                },
            )
            agent = ImitationLookupAgent(
                spec=spec,
                observation_space=SpaceSpec(type="dict"),
                action_space=SpaceSpec(type="discrete", n=2),
            )
            agent.seed(7)
            stats = agent.learn_from_dataset(ds)
            self.assertGreaterEqual(int(stats.total_rows), 2)
            a0 = int(agent.act({"x": 0, "t": 0}).action)
            a1 = int(agent.act({"x": 99, "t": 9}).action)
            self.assertIn(a0, (0, 1))
            self.assertIn(a1, (0, 1))


if __name__ == "__main__":
    unittest.main()
