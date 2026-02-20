import json
import os
import tempfile
import unittest

from memory.semantic_bridge import translate_dna, translate_transition
from models.contrastive_bridge import ContrastiveBridge


class TestSemanticBridgeLearned(unittest.TestCase):
    def _write_bridge_ckpt(self, path: str) -> None:
        model = ContrastiveBridge(n_embd=32, proj_dim=16, temperature_init=0.07)
        model.save(path)

    def test_translate_transition_with_learned_bridge(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "contrastive_bridge.pt")
            self._write_bridge_ckpt(ckpt)

            tr = translate_transition(
                obs={"x": 1, "y": 2, "goal_x": 4, "goal_y": 4, "t": 0},
                action=3,
                source_verse_name="grid_world",
                target_verse_name="cliff_world",
                next_obs={"x": 2, "y": 2, "goal_x": 4, "goal_y": 4, "t": 1},
                learned_bridge_enabled=True,
                learned_bridge_model_path=ckpt,
            )
            self.assertIsNotNone(tr)
            assert isinstance(tr, dict)
            self.assertIn("learned_bridge_confidence", tr)
            conf = tr.get("learned_bridge_confidence")
            self.assertIsInstance(conf, float)
            self.assertGreaterEqual(float(conf), 0.0)
            self.assertLessEqual(float(conf), 1.0)

    def test_translate_dna_learned_bridge_scores_rows(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "contrastive_bridge.pt")
            self._write_bridge_ckpt(ckpt)
            src = os.path.join(td, "dna_grid.jsonl")
            out = os.path.join(td, "synthetic_transfer_grid_to_cliff.jsonl")

            rows = [
                {
                    "episode_id": "ep_1",
                    "step_idx": 0,
                    "verse_name": "grid_world",
                    "obs": {"x": 1, "y": 1, "goal_x": 4, "goal_y": 4, "t": 0},
                    "next_obs": {"x": 2, "y": 1, "goal_x": 4, "goal_y": 4, "t": 1},
                    "action": 3,
                    "reward": 0.1,
                    "advantage": 0.5,
                    "done": False,
                },
                {
                    "episode_id": "ep_1",
                    "step_idx": 1,
                    "verse_name": "grid_world",
                    "obs": {"x": 2, "y": 1, "goal_x": 4, "goal_y": 4, "t": 1},
                    "next_obs": {"x": 3, "y": 1, "goal_x": 4, "goal_y": 4, "t": 2},
                    "action": 3,
                    "reward": 0.2,
                    "advantage": 0.2,
                    "done": False,
                },
            ]
            with open(src, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            stats = translate_dna(
                source_dna_path=src,
                target_verse_name="cliff_world",
                output_path=out,
                source_verse_name="grid_world",
                learned_bridge_enabled=True,
                learned_bridge_model_path=ckpt,
                learned_bridge_score_weight=0.5,
            )

            self.assertTrue(stats.learned_bridge_enabled)
            self.assertGreater(int(stats.learned_scored_rows), 0)
            with open(out, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
            self.assertIn("learned_bridge_confidence", first)
            self.assertIn("base_transfer_score", first)
            self.assertIn("transfer_score", first)

    def test_translate_dna_learned_bridge_missing_model_is_safe(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "dna_grid.jsonl")
            out = os.path.join(td, "synthetic_transfer_grid_to_cliff.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "episode_id": "ep_1",
                            "step_idx": 0,
                            "verse_name": "grid_world",
                            "obs": {"x": 1, "y": 1, "goal_x": 4, "goal_y": 4, "t": 0},
                            "next_obs": {"x": 2, "y": 1, "goal_x": 4, "goal_y": 4, "t": 1},
                            "action": 3,
                            "reward": 0.1,
                            "advantage": 0.5,
                            "done": False,
                        }
                    )
                    + "\n"
                )

            stats = translate_dna(
                source_dna_path=src,
                target_verse_name="cliff_world",
                output_path=out,
                source_verse_name="grid_world",
                learned_bridge_enabled=True,
                learned_bridge_model_path=os.path.join(td, "missing.pt"),
            )
            self.assertTrue(stats.learned_bridge_enabled)
            self.assertEqual(int(stats.learned_scored_rows), 0)

    def test_translate_dna_behavioral_bridge_scores_rows(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "dna_grid.jsonl")
            out = os.path.join(td, "synthetic_transfer_grid_to_cliff.jsonl")
            rows = [
                {
                    "episode_id": "ep_1",
                    "step_idx": 0,
                    "verse_name": "grid_world",
                    "obs": {"x": 1, "y": 1, "goal_x": 4, "goal_y": 4, "t": 0},
                    "next_obs": {"x": 2, "y": 1, "goal_x": 4, "goal_y": 4, "t": 1},
                    "action": 3,
                    "reward": 0.2,
                    "advantage": 0.4,
                    "done": False,
                },
                {
                    "episode_id": "ep_1",
                    "step_idx": 1,
                    "verse_name": "grid_world",
                    "obs": {"x": 2, "y": 1, "goal_x": 4, "goal_y": 4, "t": 1},
                    "next_obs": {"x": 3, "y": 1, "goal_x": 4, "goal_y": 4, "t": 2},
                    "action": 3,
                    "reward": 0.2,
                    "advantage": 0.2,
                    "done": False,
                },
            ]
            with open(src, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            stats = translate_dna(
                source_dna_path=src,
                target_verse_name="cliff_world",
                output_path=out,
                source_verse_name="grid_world",
                behavioral_bridge_enabled=True,
                behavioral_bridge_score_weight=0.5,
            )
            self.assertTrue(stats.behavioral_bridge_enabled)
            self.assertGreater(int(stats.behavioral_prototype_rows), 0)
            self.assertGreater(int(stats.behavioral_scored_rows), 0)
            with open(out, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
            self.assertIn("behavioral_bridge_confidence", first)
            self.assertIn("behavioral_bridge_similarity", first)
            self.assertIn("transfer_score", first)
            conf = first.get("behavioral_bridge_confidence")
            self.assertIsInstance(conf, float)
            self.assertGreaterEqual(float(conf), 0.0)
            self.assertLessEqual(float(conf), 1.0)


if __name__ == "__main__":
    unittest.main()
