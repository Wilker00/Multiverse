import json
import os
import tempfile
import unittest

from memory.central_repository import CentralMemoryConfig, backfill_memory_metadata


class TestCentralRepositoryBackfill(unittest.TestCase):
    def test_backfills_metadata_and_rebuilds_tier_files(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            ltm_path = os.path.join(td, "ltm_memories.jsonl")
            stm_path = os.path.join(td, "stm_memories.jsonl")
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_a",
                            "episode_id": "ep1",
                            "step_idx": 0,
                            "verse_name": "warehouse_world",
                            "reward": -0.1,
                            "done": False,
                            "info": {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_b",
                            "episode_id": "ep2",
                            "step_idx": 1,
                            "verse": "chess_world",
                            "reward": 1.0,
                            "done": True,
                            "info": {"reached_goal": True},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            cfg = CentralMemoryConfig(root_dir=td)
            stats = backfill_memory_metadata(cfg=cfg, rebuild_tier_files=True)

            self.assertEqual(int(stats.rows_scanned), 2)
            self.assertEqual(int(stats.rows_written), 2)
            self.assertEqual(int(stats.backfilled_memory_tier), 2)
            self.assertEqual(int(stats.backfilled_memory_family), 2)
            self.assertEqual(int(stats.backfilled_memory_type), 2)
            self.assertEqual(int(stats.ltm_rows), 1)
            self.assertEqual(int(stats.stm_rows), 1)

            rows = []
            with open(mem_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        rows.append(json.loads(s))
            self.assertEqual(len(rows), 2)
            self.assertIn(rows[0].get("memory_tier"), ("ltm", "stm"))
            self.assertIn(rows[1].get("memory_tier"), ("ltm", "stm"))
            self.assertTrue(all(bool(r.get("memory_type")) for r in rows))
            self.assertTrue(all(bool(r.get("memory_family")) for r in rows))
            self.assertTrue(all(bool(r.get("verse_name")) for r in rows))

            ltm_rows = 0
            with open(ltm_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ltm_rows += 1
            stm_rows = 0
            with open(stm_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        stm_rows += 1
            self.assertEqual(ltm_rows, 1)
            self.assertEqual(stm_rows, 1)


if __name__ == "__main__":
    unittest.main()
