import json
import os
import tempfile
import unittest

from memory.refiner import RefinerConfig, refine_event_log


class TestRefinerStreaming(unittest.TestCase):
    def test_refiner_selects_high_advantage_episode_without_full_materialization(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "runs", "run_a")
            lessons_dir = os.path.join(td, "lessons")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")

            rows = [
                {"episode_id": "ep1", "step_idx": 0, "verse_name": "warehouse_world", "action": 1, "reward": 1.0},
                {"episode_id": "ep1", "step_idx": 1, "verse_name": "warehouse_world", "action": 2, "reward": 1.0},
                {"episode_id": "ep2", "step_idx": 0, "verse_name": "warehouse_world", "action": 0, "reward": -1.0},
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            created = refine_event_log(
                RefinerConfig(
                    run_dir=run_dir,
                    lessons_dir=lessons_dir,
                    advantage_threshold=1.0,
                    baseline_by="verse_name",
                )
            )

            self.assertEqual(len(created), 1)
            self.assertTrue(os.path.isfile(created[0]))
            with open(created[0], "r", encoding="utf-8") as f:
                txt = f.read()
            self.assertIn("SOURCE_RUN: run_a", txt)
            self.assertIn("CONTEXT: warehouse_world", txt)
            self.assertIn("DO_ACTION(", txt)


if __name__ == "__main__":
    unittest.main()
