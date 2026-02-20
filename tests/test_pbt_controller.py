import unittest

from agents.pbt_controller import PBTConfig, PBTController
from core.types import AgentSpec


class TestPBTController(unittest.TestCase):
    def test_exploit_explore_step(self):
        base = AgentSpec(
            spec_version="v1",
            policy_id="pbt_base",
            policy_version="0.1",
            algo="q",
            config={"lr": 0.1, "gamma": 0.99},
        )
        cfg = PBTConfig(enabled=True, population_size=4, exploit_interval=1, seed=7)
        pbt = PBTController(cfg)
        pop = pbt.init_population(base)
        self.assertEqual(len(pop), 4)
        pbt.update_scores({"m0": 1.0, "m1": 0.9, "m2": 0.2, "m3": 0.1})
        out = pbt.maybe_exploit_explore()
        self.assertTrue(out.get("changed", False))
        self.assertGreater(len(out.get("changes", [])), 0)


if __name__ == "__main__":
    unittest.main()

