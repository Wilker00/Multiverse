import os
import tempfile
import unittest

from core.types import VerseSpec
from memory.embeddings import obs_to_vector
from models.meta_transformer import MetaTransformer
from orchestrator.mcts_trainer import MCTSTrainer, MCTSTrainerConfig
from verses.registry import create_verse, register_builtin


class TestMCTSTrainerTrace(unittest.TestCase):
    def test_trace_dataset_is_emitted(self):
        register_builtin()
        with tempfile.TemporaryDirectory() as td:
            trace_path = os.path.join(td, "mcts_trace.jsonl")
            ckpt = os.path.join(td, "meta.pt")
            spec = VerseSpec(
                spec_version="v1",
                verse_name="chess_world",
                verse_version="0.1",
                seed=17,
                params={"max_steps": 10, "adr_enabled": False},
            )
            verse = create_verse(spec)
            rr = verse.reset()
            state_dim = len(obs_to_vector(rr.obs))
            action_dim = int(verse.action_space.n or 1)
            verse.close()

            model = MetaTransformer(state_dim=state_dim, action_dim=action_dim, n_embd=64)
            cfg = MCTSTrainerConfig(
                episodes=1,
                max_steps=4,
                num_simulations=8,
                search_depth=3,
                batch_size=8,
                checkpoint_path=ckpt,
                trace_out_path=trace_path,
                checkpoint_every_episodes=1,
            )
            trainer = MCTSTrainer(verse_spec=spec, model=model, config=cfg)
            metrics = trainer.run()
            self.assertGreaterEqual(float(metrics.get("total_steps", 0.0)), 1.0)
            self.assertTrue(os.path.isfile(trace_path))

            with open(trace_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            self.assertGreaterEqual(len(lines), 1)
            self.assertIn('"search_policy"', lines[0])
            self.assertIn('"search_visit_counts"', lines[0])


if __name__ == "__main__":
    unittest.main()
