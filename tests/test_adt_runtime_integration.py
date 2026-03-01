import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.prep_adt_data import prepare_adt_data
from tools.train_adt import train_adt
import tools.train_agent


class TestADTRuntimeIntegration(unittest.TestCase):
    def test_train_agent_adt_train_cli(self):
        """
        Dedicated ADT runtime integration test for:
        tools/train_agent.py --algo adt --train --aconfig model_path=...
        """
        with tempfile.TemporaryDirectory() as td:
            # 1. Setup simple data to bootstrap a model
            runs_root = os.path.join(td, "runs_data")
            run_dir = os.path.join(runs_root, "run_init")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")

            rows = [
                {
                    "run_id": "run_init",
                    "episode_id": "ep_1",
                    "step_idx": 0,
                    "verse_name": "line_world",
                    "obs": {"pos": 0, "goal_pos": 2, "time": 0},
                    "action": 1,
                    "reward": 0.0,
                    "done": False,
                },
                {
                    "run_id": "run_init",
                    "episode_id": "ep_1",
                    "step_idx": 1,
                    "verse_name": "line_world",
                    "obs": {"pos": 1, "goal_pos": 2, "time": 1},
                    "action": 1,
                    "reward": 1.0,
                    "done": True,
                },
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            # 2. Prepare dataset
            dataset_path = os.path.join(td, "adt_data.pt")
            prepare_adt_data(
                runs_root=runs_root,
                out_path=dataset_path,
                context_len=2,
                chunk_stride=1,
                state_dim=8,
                max_timestep=64,
                gamma=1.0,
                top_return_pct=1.0,
                min_episode_steps=1,
                max_runs=0,
                verse_filter=["line_world"],
                dataset_paths=[],
                dataset_dir="",
            )

            # 3. Train initial ADT checkpoint
            ckpt_path = os.path.join(td, "decision_transformer.pt")
            train_adt(
                dataset_path=dataset_path,
                out_path=ckpt_path,
                init_model_path="",
                epochs=1,
                batch_size=2,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                val_split=0.0,
                seed=7,
                d_model=16,
                n_head=2,
                n_layer=1,
                dropout=0.0,
                max_timestep=64,
                device="cpu",
            )

            # 4. Invoke tools/train_agent.py via mocked CLI
            train_runs_root = os.path.join(td, "runs_output")
            os.makedirs(train_runs_root, exist_ok=True)

            cli_args = [
                "train_agent.py",
                "--algo", "adt",
                "--train",
                "--verse", "line_world",
                "--episodes", "2",
                "--max_steps", "10",
                "--runs_root", train_runs_root,
                "--aconfig", f"model_path={ckpt_path}",
                "--aconfig", "online_enabled=true",
                "--aconfig", "online_updates_per_learn=1",
                "--aconfig", "online_batch_size=2",
                "--aconfig", "context_len=2",
                "--aconfig", "target_return=1.0"
            ]

            with patch.object(sys, "argv", cli_args):
                tools.train_agent.main()

            # 5. Verify a run output directory was created
            output_dirs = os.listdir(train_runs_root)
            self.assertEqual(len(output_dirs), 1, "Expected exactly one run directory to be created.")
            self.assertTrue(output_dirs[0].startswith("run_"), "Run directory should have standard naming")

            run_events_path = os.path.join(train_runs_root, output_dirs[0], "events.jsonl")
            self.assertTrue(os.path.exists(run_events_path), "events.jsonl should be emitted from the run")


if __name__ == "__main__":
    unittest.main()
