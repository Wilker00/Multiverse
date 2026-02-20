import json
import os
import tempfile
import unittest

from memory.high_perf_store import HighPerfMemory


class TestHighPerfStoreStreaming(unittest.TestCase):
    def test_ingest_run_streams_in_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "runs", "run_x")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            n = 5000
            with open(events_path, "w", encoding="utf-8") as f:
                for i in range(n):
                    f.write(
                        json.dumps(
                            {
                                "event_id": f"e{i}",
                                "episode_id": "ep1",
                                "step_idx": i,
                                "verse_name": "warehouse_world",
                                "policy_id": "q",
                                "obs": {"x": i % 10, "y": i % 7},
                                "next_obs": {"x": (i + 1) % 10, "y": i % 7},
                                "action": i % 4,
                                "reward": 1.0 if (i % 17 == 0) else 0.0,
                                "done": False,
                                "truncated": False,
                                "info": {},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            store = HighPerfMemory(storage_dir=os.path.join(td, "db"), use_parquet=False)
            try:
                count = store.ingest_run(run_dir)
                self.assertEqual(count, n)
                stats = store.get_statistics()
                self.assertEqual(int(stats.get("total_events", 0)), n)
                self.assertEqual(int(stats.get("total_runs", 0)), 1)
                rows = store.sql_query("SELECT run_id, source_dir, event_count FROM runs WHERE run_id=?", ["run_x"])
                self.assertEqual(len(rows), 1)
                self.assertEqual(int(rows[0]["event_count"]), n)
                self.assertTrue(str(rows[0]["source_dir"]).endswith("run_x"))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
