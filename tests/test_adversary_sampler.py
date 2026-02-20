import json
import os
import tempfile
import unittest

from core.adversary_sampler import AdversarySampler


class TestAdversarySampler(unittest.TestCase):
    def test_recent_failures_selection(self):
        with tempfile.TemporaryDirectory() as td:
            mem_dir = os.path.join(td, "central_memory")
            os.makedirs(mem_dir, exist_ok=True)
            path = os.path.join(mem_dir, "memories.jsonl")
            rows = [
                {"verse_name": "grid_world", "obs": {"x": 0, "y": 0}, "action": 1, "reward": -20.0, "done": True, "info": {}, "run_id": "r1"},
                {"verse_name": "grid_world", "obs": {"x": 1, "y": 0}, "action": 2, "reward": 0.5, "done": False, "info": {"safe_executor": {"rewound": True}}, "run_id": "r2"},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            s = AdversarySampler(memory_dir=mem_dir, runs_root=os.path.join(td, "runs"))
            b = s.sample(verse_name="grid_world", source="recent_failures", top_k=10)
            self.assertIsNotNone(b)
            self.assertEqual(b.verse_name, "grid_world")
            self.assertGreater(len(b.obs_actions), 0)


if __name__ == "__main__":
    unittest.main()

