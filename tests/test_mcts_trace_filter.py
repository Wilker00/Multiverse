import tempfile
import unittest
from dataclasses import dataclass

from core.types import VerseSpec
from memory.embeddings import obs_to_vector
from models.meta_transformer import MetaTransformer
from orchestrator.mcts_trainer import MCTSTrainer, MCTSTrainerConfig
from verses.registry import create_verse, register_builtin


@dataclass
class _SearchResultStub:
    action_probs: list[float]
    action_values: list[float]
    forced_loss_actions: list[int]


class TestMCTSTraceFilter(unittest.TestCase):
    def _make_trainer(self, **cfg_overrides) -> MCTSTrainer:
        register_builtin()
        spec = VerseSpec(
            spec_version="v1",
            verse_name="chess_world",
            verse_version="0.1",
            seed=7,
            params={"max_steps": 8, "adr_enabled": False},
        )
        verse = create_verse(spec)
        rr = verse.reset()
        state_dim = len(obs_to_vector(rr.obs))
        action_dim = int(verse.action_space.n or 1)
        verse.close()
        model = MetaTransformer(state_dim=state_dim, action_dim=action_dim, n_embd=64)
        cfg_dict = {
            "episodes": 1,
            "max_steps": 3,
            "num_simulations": 8,
            "search_depth": 3,
            "batch_size": 8,
        }
        cfg_dict.update(cfg_overrides)
        cfg = MCTSTrainerConfig(**cfg_dict)
        return MCTSTrainer(verse_spec=spec, model=model, config=cfg)

    def test_quality_detects_significant_mcts_improvement(self):
        trainer = self._make_trainer(
            high_quality_trace_filter=True,
            min_quality_value_gain=0.05,
            min_quality_policy_shift_l1=0.20,
            min_quality_kl_divergence=0.02,
        )
        q = trainer._evaluate_trace_quality(
            initial_policy=[0.90, 0.10],
            search_result=_SearchResultStub(
                action_probs=[0.10, 0.90],
                action_values=[-0.6, 0.8],
                forced_loss_actions=[],
            ),
        )
        self.assertTrue(bool(q["significant_improvement"]))
        self.assertTrue(bool(q["high_quality"]))
        self.assertFalse(bool(q["forced_loss_avoidance"]))
        self.assertGreater(float(q["value_gain"]), 0.0)

    def test_quality_detects_forced_loss_avoidance(self):
        trainer = self._make_trainer(
            high_quality_trace_filter=True,
            min_quality_value_gain=5.0,
            min_quality_policy_shift_l1=5.0,
            min_quality_kl_divergence=5.0,
            min_forced_loss_prior_mass=0.20,
            min_forced_loss_mass_drop=0.10,
        )
        q = trainer._evaluate_trace_quality(
            initial_policy=[0.75, 0.25],
            search_result=_SearchResultStub(
                action_probs=[0.05, 0.95],
                action_values=[0.0, 0.0],
                forced_loss_actions=[0],
            ),
        )
        self.assertFalse(bool(q["significant_improvement"]))
        self.assertTrue(bool(q["forced_loss_avoidance"]))
        self.assertTrue(bool(q["high_quality"]))

    def test_sparse_replay_batching_trains_when_replay_is_small(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = self._make_trainer(
                episodes=2,
                max_steps=8,
                batch_size=128,
                high_quality_trace_filter=True,
                min_quality_value_gain=-1.0,
                min_quality_policy_shift_l1=0.0,
                min_quality_kl_divergence=0.0,
                sparse_replay_min_samples=8,
                sparse_replay_batch_size=16,
                checkpoint_path=f"{td}/meta.pt",
            )
            metrics = trainer.run()
            self.assertGreater(float(metrics["replay_size"]), 0.0)
            self.assertGreater(float(metrics["trace_rows_kept"]), 0.0)
            self.assertGreater(float(metrics["updates"]), 0.0)

    def test_strict_filter_drops_replay_rows(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = self._make_trainer(
                high_quality_trace_filter=True,
                min_quality_value_gain=10.0,
                min_quality_policy_shift_l1=10.0,
                min_quality_kl_divergence=10.0,
                min_forced_loss_prior_mass=2.0,
                min_forced_loss_mass_drop=2.0,
                checkpoint_path=f"{td}/meta.pt",
            )
            metrics = trainer.run()
            self.assertGreater(float(metrics["trace_rows_total"]), 0.0)
            self.assertEqual(float(metrics["trace_rows_kept"]), 0.0)
            self.assertEqual(float(metrics["trace_keep_rate"]), 0.0)
            self.assertEqual(float(metrics["replay_size"]), 0.0)
            self.assertEqual(float(metrics["updates"]), 0.0)


if __name__ == "__main__":
    unittest.main()
