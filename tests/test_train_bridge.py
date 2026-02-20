import json
import os
import tempfile
import unittest
from argparse import Namespace

from models.contrastive_bridge import ContrastiveBridge
from tools.train_bridge import train_bridge


class TestTrainBridge(unittest.TestCase):
    def _write_memories(self, memory_dir: str) -> None:
        os.makedirs(memory_dir, exist_ok=True)
        path = os.path.join(memory_dir, "memories.jsonl")
        rows = []
        # Two verses with aligned failure-like trajectories.
        for verse_name, key_a, key_b in (
            ("line_world", "pos", "goal"),
            ("cliff_world", "x", "y"),
        ):
            for ep in range(2):
                for step in range(4):
                    done = (step == 3)
                    rows.append(
                        {
                            "run_id": f"run_{verse_name}",
                            "episode_id": f"ep_{ep}",
                            "step_idx": step,
                            "verse_name": verse_name,
                            "obs": {key_a: float(step), key_b: float(3 - step), "t": float(step)},
                            "action": int(step % 2),
                            "reward": -2.0 if done else -0.1,
                            "done": done,
                            "truncated": False,
                            "info": {"safety_violation": bool(done)},
                        }
                    )
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_train_bridge_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            memory_dir = os.path.join(td, "central_memory")
            self._write_memories(memory_dir)
            out_path = os.path.join(td, "contrastive_bridge.pt")
            args = Namespace(
                memory_dir=memory_dir,
                runs_root=os.path.join(td, "runs"),
                input_jsonl=[],
                out_path=out_path,
                max_rows=0,
                ttf_bins=8,
                failure_only=True,
                failure_score_threshold=1.0,
                val_fraction=0.2,
                min_verses_per_bin=2,
                min_samples_per_verse=1,
                epochs=1,
                steps_per_epoch=3,
                batch_size=4,
                eval_steps=2,
                lr=1e-3,
                weight_decay=1e-2,
                seed=123,
                n_embd=32,
                proj_dim=16,
                temperature_init=0.07,
                temperature_min=0.01,
                temperature_max=1.0,
                max_keys=64,
                num_key_buckets=256,
                num_heads=4,
                n_layers=1,
            )
            report = train_bridge(args)
            self.assertTrue(os.path.isfile(out_path))
            self.assertGreater(int(report.get("train_bins", 0)), 0)
            loaded = ContrastiveBridge.load(out_path)
            sims = loaded.similarity([{"pos": 1, "goal": 2}], [{"x": 1, "y": 2}])
            self.assertEqual(tuple(sims.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
