import json
import os
import tempfile
import unittest

import torch

from tools.prepare_selector_data import prepare_data


class TestPrepareSelectorDataFilters(unittest.TestCase):
    def test_warehouse_success_episode_filter(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            os.makedirs(runs_root, exist_ok=True)

            run_warehouse = os.path.join(runs_root, "run_wh")
            run_line = os.path.join(runs_root, "run_line")
            os.makedirs(run_warehouse, exist_ok=True)
            os.makedirs(run_line, exist_ok=True)

            wh_rows = [
                {
                    "episode_id": "ep_good",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 0, "goal_x": 7, "goal_y": 7, "t": 0},
                    "reward": -0.1,
                    "info": {"reached_goal": False},
                },
                {
                    "episode_id": "ep_good",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 1, "y": 0, "goal_x": 7, "goal_y": 7, "t": 1},
                    "reward": 1.0,
                    "info": {"reached_goal": True},
                },
                {
                    "episode_id": "ep_fail",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 1, "goal_x": 7, "goal_y": 7, "t": 0},
                    "reward": 1.0,
                    "info": {"reached_goal": False},
                },
            ]
            with open(os.path.join(run_warehouse, "events.jsonl"), "w", encoding="utf-8") as f:
                for row in wh_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            with open(os.path.join(run_line, "events.jsonl"), "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "episode_id": "ep_line",
                            "step_idx": 0,
                            "verse_name": "line_world",
                            "obs": {"pos": 0, "goal": 1, "t": 0},
                            "reward": 1.0,
                            "info": {"reached_goal": True},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            out_path = os.path.join(td, "selector_batch.pt")
            prepare_data(
                runs_root=runs_root,
                lessons_dir=os.path.join(td, "lessons_missing"),
                output_path=out_path,
                reward_threshold=0.9,
                verse_filter=["warehouse_world"],
                episode_success_only=True,
                episode_return_threshold=0.5,
                include_all_events_from_selected_episodes=True,
            )

            self.assertTrue(os.path.isfile(out_path))
            data = torch.load(out_path, weights_only=False)
            states = data.get("states")
            labels = data.get("labels")
            self.assertEqual(int(states.shape[0]), 2)
            self.assertEqual(int(labels.shape[0]), 2)

    def test_top_return_percentile_filter(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            os.makedirs(runs_root, exist_ok=True)
            run_dir = os.path.join(runs_root, "run_wh")
            os.makedirs(run_dir, exist_ok=True)
            rows = [
                {"episode_id": "ep1", "step_idx": 0, "verse_name": "warehouse_world", "obs": {"x": 0, "y": 0, "goal_x": 7, "goal_y": 7, "t": 0}, "reward": 1.0, "info": {"reached_goal": False}},
                {"episode_id": "ep2", "step_idx": 0, "verse_name": "warehouse_world", "obs": {"x": 1, "y": 0, "goal_x": 7, "goal_y": 7, "t": 0}, "reward": 0.5, "info": {"reached_goal": False}},
                {"episode_id": "ep3", "step_idx": 0, "verse_name": "warehouse_world", "obs": {"x": 2, "y": 0, "goal_x": 7, "goal_y": 7, "t": 0}, "reward": -0.2, "info": {"reached_goal": False}},
            ]
            with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            out_path = os.path.join(td, "selector_top_pct.pt")
            prepare_data(
                runs_root=runs_root,
                lessons_dir=os.path.join(td, "lessons_missing"),
                output_path=out_path,
                reward_threshold=-999.0,
                verse_filter=["warehouse_world"],
                episode_top_return_pct=0.34,  # keep top ~1 of 3 episodes
                include_all_events_from_selected_episodes=True,
            )
            self.assertTrue(os.path.isfile(out_path))
            data = torch.load(out_path, weights_only=False)
            states = data.get("states")
            self.assertEqual(int(states.shape[0]), 1)

    def test_class_balance_equalizes_class_counts(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            os.makedirs(runs_root, exist_ok=True)
            run_line = os.path.join(runs_root, "run_line")
            run_grid = os.path.join(runs_root, "run_grid")
            os.makedirs(run_line, exist_ok=True)
            os.makedirs(run_grid, exist_ok=True)

            with open(os.path.join(run_line, "events.jsonl"), "w", encoding="utf-8") as f:
                for i in range(6):
                    f.write(
                        json.dumps(
                            {
                                "episode_id": "ep_line",
                                "step_idx": i,
                                "verse_name": "line_world",
                                "obs": {"pos": i, "goal": 10, "t": i},
                                "reward": 1.0,
                                "info": {"reached_goal": False},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            with open(os.path.join(run_grid, "events.jsonl"), "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "episode_id": "ep_grid",
                            "step_idx": 0,
                            "verse_name": "grid_world",
                            "obs": {"x": 0, "y": 0, "goal_x": 1, "goal_y": 1, "t": 0},
                            "reward": 1.0,
                            "info": {"reached_goal": False},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            out_path = os.path.join(td, "selector_balanced.pt")
            prepare_data(
                runs_root=runs_root,
                lessons_dir=os.path.join(td, "lessons_missing"),
                output_path=out_path,
                reward_threshold=-999.0,
                class_balance=True,
                class_balance_seed=7,
            )

            self.assertTrue(os.path.isfile(out_path))
            data = torch.load(out_path, weights_only=False)
            labels = data.get("labels")
            self.assertIsNotNone(labels)
            counts = torch.bincount(labels)
            nonzero = [int(x) for x in counts.tolist() if int(x) > 0]
            self.assertTrue(len(nonzero) >= 2)
            self.assertEqual(min(nonzero), max(nonzero))


if __name__ == "__main__":
    unittest.main()
