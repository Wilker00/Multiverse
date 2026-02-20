import json
import os
import tempfile
import unittest

from memory.selection import ActiveForgettingConfig, active_forget_central_memory


class TestActiveForgetting(unittest.TestCase):
    def _write_memory_rows(self, mem_path: str) -> None:
        rows = []
        for i in range(25):
            rows.append(
                {
                    "run_id": "run_bad",
                    "episode_id": "ep_bad",
                    "step_idx": i,
                    "obs_vector": [1.0, 0.0],
                    "reward": -1.0,
                    "info": {"reached_goal": False, "fell_cliff": True},
                }
            )
        for i in range(25):
            rows.append(
                {
                    "run_id": "run_good",
                    "episode_id": "ep_good",
                    "step_idx": i,
                    "obs_vector": [0.0, 1.0],
                    "reward": 1.0,
                    "info": {"reached_goal": True},
                }
            )
        with open(mem_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_quality_filter_can_drop_unique_bad_runs(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            self._write_memory_rows(mem_path)

            st = active_forget_central_memory(
                ActiveForgettingConfig(
                    memory_dir=td,
                    similarity_threshold=0.99,
                    min_events_per_run=20,
                    write_backup=False,
                    quality_filter_enabled=True,
                    min_mean_reward=0.0,
                    min_success_rate=0.2,
                    max_hazard_rate=0.2,
                )
            )

            self.assertEqual(int(st.input_runs), 2)
            self.assertEqual(int(st.kept_runs), 1)
            self.assertEqual(int(st.dropped_runs), 1)
            self.assertEqual(int(st.dropped_low_quality_runs), 1)
            self.assertEqual(int(st.kept_rows), 25)
            self.assertEqual(int(st.dropped_rows), 25)

    def test_default_mode_is_dedupe_only(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            self._write_memory_rows(mem_path)

            st = active_forget_central_memory(
                ActiveForgettingConfig(
                    memory_dir=td,
                    similarity_threshold=0.99,
                    min_events_per_run=20,
                    write_backup=False,
                    quality_filter_enabled=False,
                )
            )

            self.assertEqual(int(st.input_runs), 2)
            self.assertEqual(int(st.kept_runs), 2)
            self.assertEqual(int(st.dropped_runs), 0)
            self.assertEqual(int(st.dropped_low_quality_runs), 0)
            self.assertEqual(int(st.kept_rows), 50)


if __name__ == "__main__":
    unittest.main()
