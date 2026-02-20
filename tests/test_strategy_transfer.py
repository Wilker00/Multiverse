import json
import os
import tempfile
import unittest

from core.taxonomy import can_bridge
from memory.semantic_bridge import translate_action, translate_dna, translate_observation


class TestStrategyTransfer(unittest.TestCase):
    def test_strategy_bridge_enabled(self):
        self.assertTrue(can_bridge("chess_world", "go_world"))
        self.assertTrue(can_bridge("go_world", "uno_world"))
        self.assertTrue(can_bridge("uno_world", "chess_world"))

    def test_translate_strategy_observation_and_action(self):
        chess_obs = {
            "material_delta": 3,
            "development": 5,
            "king_safety": 6,
            "center_control": 2,
            "score_delta": 4,
            "pressure": 7,
            "risk": 1,
            "tempo": 3,
            "control": 2,
            "resource": 5,
            "phase": 1,
            "t": 9,
        }
        go_obs = translate_observation(
            obs=chess_obs,
            source_verse_name="chess_world",
            target_verse_name="go_world",
        )
        uno_obs = translate_observation(
            obs=chess_obs,
            source_verse_name="chess_world",
            target_verse_name="uno_world",
        )
        self.assertIsInstance(go_obs, dict)
        self.assertIsInstance(uno_obs, dict)
        self.assertIn("territory_delta", go_obs)
        self.assertIn("my_cards", uno_obs)

        mapped_action = translate_action(
            action=2,
            source_verse_name="chess_world",
            target_verse_name="go_world",
        )
        self.assertEqual(mapped_action, 2)

    def test_translate_dna_strategy_rows(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "dna_good.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                for i in range(3):
                    row = {
                        "episode_id": "ep_1",
                        "step_idx": i,
                        "verse_name": "chess_world",
                        "obs": {
                            "material_delta": i,
                            "development": 4,
                            "king_safety": 5,
                            "center_control": 1,
                            "score_delta": i,
                            "pressure": 4,
                            "risk": 1,
                            "tempo": 2,
                            "control": 1,
                            "resource": 4,
                            "phase": 1,
                            "t": i,
                        },
                        "next_obs": {
                            "material_delta": i + 1,
                            "development": 4,
                            "king_safety": 5,
                            "center_control": 1,
                            "score_delta": i + 1,
                            "pressure": 4,
                            "risk": 1,
                            "tempo": 2,
                            "control": 1,
                            "resource": 4,
                            "phase": 1,
                            "t": i + 1,
                        },
                        "action": i % 6,
                        "reward": 0.2,
                        "advantage": 0.7,
                        "done": False,
                    }
                    f.write(json.dumps(row) + "\n")

            out_path = os.path.join(td, "synthetic_transfer_chess_world_to_go_world.jsonl")
            st = translate_dna(
                source_dna_path=src,
                target_verse_name="go_world",
                output_path=out_path,
                source_verse_name="chess_world",
            )
            self.assertEqual(st.source_verse_name, "chess_world")
            self.assertEqual(st.target_verse_name, "go_world")
            self.assertGreater(st.translated_rows, 0)
            self.assertTrue(os.path.isfile(out_path))

            with open(out_path, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
            self.assertIn("transfer_score", first)
            self.assertIn("source_advantage", first)
            self.assertEqual(first["target_verse_name"], "go_world")


if __name__ == "__main__":
    unittest.main()
