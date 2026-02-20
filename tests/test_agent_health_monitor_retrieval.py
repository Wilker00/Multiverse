import json
import os
import tempfile
import unittest

from tools.agent_health_monitor import (
    AgentHealthRow,
    _collect_retrieval_audit,
    _memory_inventory,
    _row_to_json,
)
from memory.embeddings import obs_to_vector


class TestAgentHealthMonitorRetrieval(unittest.TestCase):
    def test_collect_retrieval_audit_and_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "runs", "run_x")
            mem_dir = os.path.join(td, "central_memory")
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(mem_dir, exist_ok=True)

            events_path = os.path.join(run_dir, "events.jsonl")
            event_rows = [
                {
                    "episode_id": "ep1",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 1, "y": 2},
                    "action": 1,
                    "reward": 0.1,
                },
                {
                    "episode_id": "ep1",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 2, "y": 2},
                    "action": 3,
                    "reward": 0.2,
                },
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for r in event_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            mem_path = os.path.join(mem_dir, "memories.jsonl")
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_other",
                            "episode_id": "ep2",
                            "step_idx": 0,
                            "verse_name": "warehouse_world",
                            "t_ms": 1,
                            "obs": {"x": 1, "y": 2},
                            "obs_vector": [1.0, 2.0],
                            "action": 1,
                            "reward": 0.3,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with open(os.path.join(mem_dir, "dedupe_index.json"), "w", encoding="utf-8") as f:
                json.dump([], f)

            inv = _memory_inventory(central_memory_dir=mem_dir, max_scan_rows=100)
            self.assertTrue(bool(inv.get("exists", False)))
            self.assertGreaterEqual(int(inv.get("rows_scanned", 0)), 1)

            audit = _collect_retrieval_audit(
                run_dir=run_dir,
                central_memory_dir=mem_dir,
                probes_per_run=2,
                top_k=3,
                min_score=0.0,
                same_verse_only=True,
            )
            self.assertGreaterEqual(int(audit.get("probes", 0)), 1)
            self.assertGreaterEqual(float(audit.get("hit_rate", 0.0)), 0.0)
            self.assertGreater(int(audit.get("hits", 0)), 0)

    def test_collect_retrieval_audit_accepts_nested_obs_lists(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "runs", "run_x")
            mem_dir = os.path.join(td, "central_memory")
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(mem_dir, exist_ok=True)

            nested_obs = {"x": 1, "y": 2, "flat": [1.0, 2.0, 3.0]}
            events_path = os.path.join(run_dir, "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "episode_id": "ep1",
                            "step_idx": 0,
                            "verse_name": "warehouse_world",
                            "obs": nested_obs,
                            "action": 1,
                            "reward": 0.1,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            mem_path = os.path.join(mem_dir, "memories.jsonl")
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_other",
                            "episode_id": "ep2",
                            "step_idx": 0,
                            "verse_name": "warehouse_world",
                            "t_ms": 1,
                            "obs": nested_obs,
                            "obs_vector": obs_to_vector(nested_obs),
                            "action": 1,
                            "reward": 0.3,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with open(os.path.join(mem_dir, "dedupe_index.json"), "w", encoding="utf-8") as f:
                json.dump([], f)

            audit = _collect_retrieval_audit(
                run_dir=run_dir,
                central_memory_dir=mem_dir,
                probes_per_run=1,
                top_k=3,
                min_score=0.0,
                same_verse_only=True,
            )
            self.assertEqual(int(audit.get("eligible_events", 0)), 1)
            self.assertEqual(int(audit.get("probes", 0)), 1)
            self.assertGreater(int(audit.get("hits", 0)), 0)

    def test_row_to_json_contains_retrieval_audit(self):
        row = AgentHealthRow(
            agent_id="a",
            run_id="r",
            run_dir="runs/r",
            verse_name="warehouse_world",
            policy_id="q",
            mean_return=0.0,
            success_rate=0.0,
            intuition_match=None,
            memory_coherence=None,
            search_regret_kl=None,
            veto_rate=0.0,
            shield_veto_rate=0.0,
            market_reputation=None,
            intuition_score=50.0,
            search_score=50.0,
            safety_score=50.0,
            trust_score=50.0,
            total_score=50.0,
            status="watch",
            issues=[],
            recommended_actions=[],
            automated_actions=[],
            retrieval_audit={"hit_rate": 0.5, "probes": 8},
        )
        obj = _row_to_json(row)
        self.assertIn("retrieval_audit", obj)
        self.assertAlmostEqual(float(obj["retrieval_audit"]["hit_rate"]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
