import json
import os
import tempfile
import unittest

from tools.build_strategy_transfer import build_strategy_transfer


class TestBuildStrategyTransfer(unittest.TestCase):
    def test_build_strategy_transfer_outputs_files_and_perf(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "dna_good.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                for i in range(4):
                    row = {
                        "episode_id": "ep_x",
                        "step_idx": i,
                        "verse_name": "chess_world",
                        "obs": {
                            "material_delta": i,
                            "development": 5,
                            "king_safety": 5,
                            "center_control": 2,
                            "score_delta": i,
                            "pressure": 5,
                            "risk": 1,
                            "tempo": 3,
                            "control": 2,
                            "resource": 5,
                            "phase": 1,
                            "t": i,
                        },
                        "next_obs": {
                            "material_delta": i + 1,
                            "development": 5,
                            "king_safety": 5,
                            "center_control": 2,
                            "score_delta": i + 1,
                            "pressure": 5,
                            "risk": 1,
                            "tempo": 3,
                            "control": 2,
                            "resource": 5,
                            "phase": 1,
                            "t": i + 1,
                        },
                        "action": i % 6,
                        "reward": 0.3,
                        "advantage": 0.8,
                        "done": False,
                    }
                    f.write(json.dumps(row) + "\n")

            out_dir = os.path.join(td, "out")
            perf_out = os.path.join(td, "strategy_transfer_performance.json")
            result = build_strategy_transfer(
                source_items=[(src, "chess_world")],
                targets=["go_world", "uno_world"],
                out_dir=out_dir,
                perf_out=perf_out,
            )

            self.assertGreaterEqual(int(result["generated_files"]), 2)
            self.assertTrue(os.path.isfile(perf_out))
            with open(perf_out, "r", encoding="utf-8") as f:
                perf = json.load(f)
            self.assertIn("synthetic_transfer_chess_world_to_go_world", perf)
            self.assertIn("synthetic_transfer_chess_world_to_uno_world", perf)


if __name__ == "__main__":
    unittest.main()
