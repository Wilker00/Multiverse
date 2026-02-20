import json
import os
import tempfile
import unittest

from tools.update_centroid import load_high_value_data, train_centroid_policy


class TestUpdateCentroid(unittest.TestCase):
    def test_load_and_train_centroid_policy(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_1")
            os.makedirs(run_dir, exist_ok=True)

            dna_path = os.path.join(run_dir, "dna_good.jsonl")
            with open(dna_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"obs": {"x": 0}, "action": 0, "advantage": 1.0}) + "\n")
                f.write("this is not json\n")
                f.write(json.dumps({"obs": {"x": 0}, "action": 1, "advantage": 4.0}) + "\n")
                f.write(json.dumps({"obs": {"x": 1}, "action": 1, "advantage": 2.0}) + "\n")
                f.write(json.dumps({"obs": {"x": 2}, "action": 0, "advantage": 0.1}) + "\n")

            data = load_high_value_data(runs_root=runs_root, min_advantage=0.5)
            self.assertEqual(len(data), 3)

            out_path = os.path.join(td, "centroid_policy.json")
            metrics = train_centroid_policy(data=data, model_path=out_path)
            self.assertTrue(metrics["saved"])
            self.assertTrue(os.path.isfile(out_path))

            with open(out_path, "r", encoding="utf-8") as f:
                artifact = json.load(f)
            self.assertEqual(artifact["format"], "centroid_policy_v1")
            self.assertEqual(artifact["default_action"], 1)
            self.assertGreaterEqual(float(artifact["default_action_confidence"]), 0.5)
            obs_entry = artifact["obs_policy"]['{"x":0}']
            self.assertEqual(obs_entry["action"], 1)
            self.assertGreater(float(obs_entry["confidence"]), 0.5)

    def test_train_centroid_policy_empty(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "centroid_policy.json")
            metrics = train_centroid_policy(data=[], model_path=out_path)
            self.assertFalse(metrics["saved"])
            self.assertEqual(metrics["reason"], "no_data")
            self.assertFalse(os.path.isfile(out_path))


if __name__ == "__main__":
    unittest.main()
