import unittest

from core.mcts_search import MCTSConfig, MCTSSearch
from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


class TestMCTSSearch(unittest.TestCase):
    def test_strategy_verse_search_distribution_and_state_restore(self):
        register_builtin()
        for verse_name, max_steps in (("chess_world", 18), ("go_world", 20), ("uno_world", 16)):
            spec = VerseSpec(
                spec_version="v1",
                verse_name=verse_name,
                verse_version="0.1",
                seed=11,
                params={"max_steps": max_steps, "adr_enabled": False},
            )
            verse = create_verse(spec)
            rr = verse.reset()
            before = verse.export_state()

            search = MCTSSearch(
                verse=verse,
                config=MCTSConfig(
                    num_simulations=24,
                    max_depth=5,
                    dirichlet_epsilon=0.0,
                    forced_loss_min_visits=1,
                ),
            )
            out = search.search(
                root_obs=rr.obs,
                policy_net=lambda _obs: [1.0 for _ in range(6)],
                value_net=lambda _obs: 0.0,
            )
            after = verse.export_state()

            self.assertEqual(before, after)
            self.assertEqual(len(out.action_probs), 6)
            self.assertEqual(len(out.visit_counts), 6)
            self.assertEqual(len(out.action_values), 6)
            self.assertAlmostEqual(sum(out.action_probs), 1.0, places=5)
            self.assertGreaterEqual(int(out.best_action), 0)
            self.assertLess(int(out.best_action), 6)
            verse.close()

    def test_runtime_error_counters_capture_policy_and_value_failures(self):
        register_builtin()
        spec = VerseSpec(
            spec_version="v1",
            verse_name="chess_world",
            verse_version="0.1",
            seed=23,
            params={"max_steps": 12, "adr_enabled": False},
        )
        verse = create_verse(spec)
        rr = verse.reset()
        search = MCTSSearch(
            verse=verse,
            config=MCTSConfig(
                num_simulations=8,
                max_depth=3,
                dirichlet_epsilon=0.0,
            ),
        )

        def bad_policy(obs, history=None):
            _ = (obs, history)
            raise RuntimeError("policy failed")

        def bad_value(obs, history=None):
            _ = (obs, history)
            raise RuntimeError("value failed")

        out = search.search(root_obs=rr.obs, policy_net=bad_policy, value_net=bad_value)
        self.assertEqual(len(out.action_probs), 6)
        self.assertIn("policy_callable_error", dict(out.runtime_errors))
        self.assertIn("value_callable_error", dict(out.runtime_errors))
        verse.close()


if __name__ == "__main__":
    unittest.main()
