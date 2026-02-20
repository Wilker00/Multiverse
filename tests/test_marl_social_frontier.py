import tempfile
import unittest

from core.types import AgentSpec, VerseSpec
from orchestrator.marl_trainer import MARLConfig, MultiAgentTrainer


class TestMARLSocialFrontier(unittest.TestCase):
    def test_marl_run_returns_shared_memory_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = MultiAgentTrainer(run_root=td, schema_version="v1", auto_register_builtin=True)
            verse_specs = [
                VerseSpec(
                    spec_version="v1",
                    verse_name="line_world",
                    verse_version="0.1",
                    seed=101,
                    tags=["test"],
                    params={"goal_pos": 6, "max_steps": 12, "step_penalty": -0.01},
                ),
                VerseSpec(
                    spec_version="v1",
                    verse_name="line_world",
                    verse_version="0.1",
                    seed=202,
                    tags=["test"],
                    params={"goal_pos": 6, "max_steps": 12, "step_penalty": -0.01},
                ),
            ]
            agent_specs = [
                AgentSpec(spec_version="v1", policy_id="a0", policy_version="0.1", algo="random", seed=101),
                AgentSpec(spec_version="v1", policy_id="a1", policy_version="0.1", algo="random", seed=202),
            ]
            cfg = MARLConfig(
                episodes=2,
                max_steps=12,
                train=True,
                collect_transitions=True,
                shared_memory_enabled=True,
                shared_memory_top_k=3,
                negotiation_interval=1,
                lexicon_min_support=1,
            )
            out = trainer.run(verse_specs=verse_specs, agent_specs=agent_specs, config=cfg, seed=123)
            self.assertIn("shared_memory", out)
            snap = out["shared_memory"]
            self.assertIsInstance(snap, dict)
            self.assertIn("trajectory_count", snap)
            self.assertGreaterEqual(int(snap["trajectory_count"]), 1)


if __name__ == "__main__":
    unittest.main()

