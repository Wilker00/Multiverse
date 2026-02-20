import unittest

from core.types import AgentSpec, VerseSpec
from orchestrator.trainer import Trainer
from verses.registry import create_verse, list_verses, register_builtin


class TestStrategyVerses(unittest.TestCase):
    def test_registry_contains_strategy_verses(self):
        register_builtin()
        names = set(list_verses().keys())
        self.assertIn("chess_world", names)
        self.assertIn("go_world", names)
        self.assertIn("uno_world", names)

    def test_strategy_verses_reset_and_step(self):
        register_builtin()
        for verse_name in ("chess_world", "go_world", "uno_world"):
            spec = VerseSpec(
                spec_version="v1",
                verse_name=verse_name,
                verse_version="0.1",
                seed=7,
                params={"max_steps": 20},
            )
            verse = create_verse(spec)
            rr = verse.reset()
            self.assertIsInstance(rr.obs, dict)
            self.assertEqual(verse.action_space.type, "discrete")
            self.assertEqual(int(verse.action_space.n or 0), 6)

            sr = verse.step(0)
            self.assertIsInstance(sr.obs, dict)
            self.assertIn("t", sr.obs)
            self.assertIn("reached_goal", sr.info)
            if hasattr(verse, "legal_actions"):
                legal = verse.legal_actions(sr.obs)
                self.assertIsInstance(legal, list)
                self.assertGreaterEqual(len(legal), 1)
                self.assertTrue(all(isinstance(a, int) for a in legal))
                self.assertTrue(all(0 <= int(a) < 6 for a in legal))
            verse.close()

    def test_trainer_runs_q_on_strategy_verses(self):
        trainer = Trainer(run_root="runs_smoke", schema_version="v1", auto_register_builtin=True)
        for verse_name, max_steps in (("chess_world", 20), ("go_world", 22), ("uno_world", 18)):
            verse_spec = VerseSpec(
                spec_version="v1",
                verse_name=verse_name,
                verse_version="0.1",
                seed=13,
                params={"max_steps": max_steps},
            )
            agent_spec = AgentSpec(
                spec_version="v1",
                policy_id=f"q_{verse_name}_smoke",
                policy_version="0.1",
                algo="q",
                seed=13,
                config={"lr": 0.12, "epsilon_decay": 0.98, "epsilon_min": 0.05},
            )
            result = trainer.run(
                verse_spec=verse_spec,
                agent_spec=agent_spec,
                episodes=2,
                max_steps=max_steps,
                seed=13,
            )
            self.assertIn("run_id", result)
            self.assertGreaterEqual(int(result.get("total_steps", 0)), 1)


if __name__ == "__main__":
    unittest.main()
