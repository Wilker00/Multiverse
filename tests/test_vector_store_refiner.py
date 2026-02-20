import json
import os
import tempfile
import unittest

from tools.vector_store_refiner import VectorStoreRefinerConfig, refine_danger_map


class TestVectorStoreRefiner(unittest.TestCase):
    def test_refine_danger_map_clusters_failed_states(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            os.makedirs(runs_root, exist_ok=True)

            run_a = os.path.join(runs_root, "run_a")
            run_b = os.path.join(runs_root, "run_b")
            os.makedirs(run_a, exist_ok=True)
            os.makedirs(run_b, exist_ok=True)

            rows_a = [
                {
                    "episode_id": "ep1",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "policy_id": "transfer_q_warehouse_world",
                    "obs": {"x": 1, "y": 1, "goal_x": 7, "goal_y": 7, "t": 0},
                    "action": 2,
                    "reward": -1.0,
                    "done": False,
                    "info": {"hit_wall": True},
                },
                {
                    "episode_id": "ep1",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "policy_id": "transfer_q_warehouse_world",
                    "obs": {"x": 1, "y": 1, "goal_x": 7, "goal_y": 7, "t": 1},
                    "action": 2,
                    "reward": -1.0,
                    "done": False,
                    "info": {"hit_wall": True},
                },
                {
                    "episode_id": "ep1",
                    "step_idx": 2,
                    "verse_name": "warehouse_world",
                    "policy_id": "transfer_q_warehouse_world",
                    "obs": {"x": 2, "y": 1, "goal_x": 7, "goal_y": 7, "t": 2},
                    "action": 3,
                    "reward": 0.2,
                    "done": False,
                    "info": {"reached_goal": False},
                },
            ]
            rows_b = [
                {
                    "episode_id": "ep2",
                    "step_idx": 0,
                    "verse_name": "labyrinth_world",
                    "policy_id": "transfer_q_labyrinth_world",
                    "obs": {"x": 10, "y": 10, "battery": 20, "t": 0},
                    "action": 0,
                    "reward": -2.0,
                    "done": True,
                    "info": {"unsafe": True},
                }
            ]

            with open(os.path.join(run_a, "events.jsonl"), "w", encoding="utf-8") as f:
                for r in rows_a:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(os.path.join(run_b, "events.jsonl"), "w", encoding="utf-8") as f:
                for r in rows_b:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            out_json = os.path.join(td, "danger_map.json")
            out_sig = os.path.join(td, "danger_signatures.jsonl")
            rep = refine_danger_map(
                VectorStoreRefinerConfig(
                    runs_root=runs_root,
                    out_json=out_json,
                    out_signatures_jsonl=out_sig,
                    cluster_similarity_threshold=0.995,
                    min_cluster_size=1,
                    max_runs=10,
                )
            )

            self.assertTrue(os.path.isfile(out_json))
            self.assertTrue(os.path.isfile(out_sig))
            self.assertGreaterEqual(int(rep.get("failed_events_vectorized", 0)), 3)
            self.assertGreaterEqual(int(rep.get("clusters_kept", 0)), 2)
            self.assertGreater(int(rep.get("signatures_rows", 0)), 0)


if __name__ == "__main__":
    unittest.main()
