import json
import os
import tempfile
import unittest

import torch

from agents.transformer_agent import TransformerAgent
from core.agent_base import ExperienceBatch, Transition
from core.types import AgentSpec, SpaceSpec
from tools.prep_adt_data import prepare_adt_data
from tools.train_adt import train_adt


class TestADTPipeline(unittest.TestCase):
    def test_prepare_train_and_infer(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_unit")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")

            rows = [
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_1",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 0, "goal_x": 2, "goal_y": 2, "t": 0},
                    "action": 0,
                    "reward": -0.1,
                    "done": False,
                },
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_1",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 1, "y": 0, "goal_x": 2, "goal_y": 2, "t": 1},
                    "action": 1,
                    "reward": 1.0,
                    "done": True,
                },
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_2",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 1, "goal_x": 2, "goal_y": 2, "t": 0},
                    "action": 1,
                    "reward": -0.1,
                    "done": False,
                },
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_2",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 2, "goal_x": 2, "goal_y": 2, "t": 1},
                    "action": 0,
                    "reward": 1.0,
                    "done": True,
                },
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            dataset_path = os.path.join(td, "adt_data.pt")
            meta = prepare_adt_data(
                runs_root=runs_root,
                out_path=dataset_path,
                context_len=4,
                chunk_stride=2,
                state_dim=16,
                max_timestep=64,
                gamma=1.0,
                top_return_pct=1.0,
                min_episode_steps=1,
                max_runs=0,
                verse_filter=["warehouse_world"],
                dataset_paths=[],
                dataset_dir="",
            )
            self.assertEqual(int(meta["action_dim"]), 2)
            self.assertGreaterEqual(int(meta["samples"]), 2)

            ckpt_path = os.path.join(td, "decision_transformer.pt")
            ckpt = train_adt(
                dataset_path=dataset_path,
                out_path=ckpt_path,
                init_model_path="",
                epochs=2,
                batch_size=2,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                val_split=0.25,
                seed=7,
                d_model=32,
                n_head=4,
                n_layer=2,
                dropout=0.0,
                max_timestep=64,
                device="cpu",
            )
            self.assertIn("model_state_dict", ckpt)
            self.assertTrue(os.path.isfile(ckpt_path))

            obs_space = SpaceSpec(type="dict")
            act_space = SpaceSpec(type="discrete", n=2)
            spec = AgentSpec(
                spec_version="v1",
                policy_id="adt_unit",
                policy_version="0.1",
                algo="adt",
                config={
                    "model_path": ckpt_path,
                    "target_return": 1.0,
                    "context_len": 4,
                    "device": "cpu",
                },
            )
            agent = TransformerAgent(spec=spec, observation_space=obs_space, action_space=act_space)
            out = agent.act({"x": 0, "y": 0, "goal_x": 2, "goal_y": 2, "t": 0})
            self.assertIsInstance(int(out.action), int)
            self.assertGreaterEqual(int(out.action), 0)
            self.assertLess(int(out.action), 2)
            self.assertEqual(str(out.info.get("mode")), "adt")

    def test_prepare_success_only_filter(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_unit")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            rows = [
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_ok",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 0, "goal_x": 1, "goal_y": 1, "t": 0},
                    "action": 0,
                    "reward": 0.0,
                    "done": False,
                    "info": {"reached_goal": False},
                },
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_ok",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 1, "y": 1, "goal_x": 1, "goal_y": 1, "t": 1},
                    "action": 1,
                    "reward": 1.0,
                    "done": True,
                    "info": {"reached_goal": True},
                },
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_fail",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 0, "goal_x": 2, "goal_y": 2, "t": 0},
                    "action": 0,
                    "reward": -1.0,
                    "done": True,
                    "info": {"reached_goal": False},
                },
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            dataset_path = os.path.join(td, "adt_success_only.pt")
            meta = prepare_adt_data(
                runs_root=runs_root,
                out_path=dataset_path,
                context_len=4,
                chunk_stride=2,
                state_dim=16,
                max_timestep=64,
                gamma=1.0,
                top_return_pct=1.0,
                success_only=True,
                min_episode_steps=1,
                max_runs=0,
                verse_filter=["warehouse_world"],
                dataset_paths=[],
                dataset_dir="",
            )
            self.assertEqual(int(meta["episodes"]), 1)
            self.assertEqual(int(meta["episodes_after_success_only"]), 1)

    def test_online_learning_updates(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_unit")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            rows = [
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_1",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 0, "goal_x": 2, "goal_y": 2, "t": 0},
                    "action": 0,
                    "reward": -0.1,
                    "done": False,
                },
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_1",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 1, "y": 0, "goal_x": 2, "goal_y": 2, "t": 1},
                    "action": 1,
                    "reward": 1.0,
                    "done": True,
                    "info": {"reached_goal": True},
                },
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            dataset_path = os.path.join(td, "adt_data.pt")
            prepare_adt_data(
                runs_root=runs_root,
                out_path=dataset_path,
                context_len=4,
                chunk_stride=2,
                state_dim=16,
                max_timestep=64,
                gamma=1.0,
                top_return_pct=1.0,
                min_episode_steps=1,
                max_runs=0,
                verse_filter=["warehouse_world"],
                dataset_paths=[],
                dataset_dir="",
            )

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
                d_model=32,
                n_head=4,
                n_layer=2,
                dropout=0.0,
                max_timestep=64,
                device="cpu",
            )

            obs_space = SpaceSpec(type="dict")
            act_space = SpaceSpec(type="discrete", n=2)
            spec = AgentSpec(
                spec_version="v1",
                policy_id="adt_online_unit",
                policy_version="0.1",
                algo="adt",
                config={
                    "model_path": ckpt_path,
                    "target_return": 1.0,
                    "context_len": 4,
                    "device": "cpu",
                    "online_enabled": True,
                    "online_updates_per_learn": 1,
                    "online_batch_size": 2,
                },
            )
            agent = TransformerAgent(spec=spec, observation_space=obs_space, action_space=act_space)
            batch = ExperienceBatch(
                transitions=[
                    Transition(
                        obs={"x": 0, "y": 0, "goal_x": 2, "goal_y": 2, "t": 0},
                        action=0,
                        reward=-0.1,
                        next_obs={"x": 1, "y": 0, "goal_x": 2, "goal_y": 2, "t": 1},
                        done=False,
                        truncated=False,
                        info={},
                    ),
                    Transition(
                        obs={"x": 1, "y": 0, "goal_x": 2, "goal_y": 2, "t": 1},
                        action=1,
                        reward=1.0,
                        next_obs={"x": 2, "y": 0, "goal_x": 2, "goal_y": 2, "t": 2},
                        done=True,
                        truncated=False,
                        info={"env_info": {"reached_goal": True}},
                    ),
                ],
                meta={},
            )
            metrics = agent.learn(batch)
            self.assertGreaterEqual(int(metrics.get("online_updates", 0)), 1)
            self.assertGreaterEqual(int(metrics.get("online_replay_size", 0)), 1)
            self.assertIn("online_loss", metrics)

    def test_target_return_auto_inference_from_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_unit")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            rows = [
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_hi",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 0, "goal_x": 2, "goal_y": 2, "t": 0},
                    "action": 3,
                    "reward": 8.0,
                    "done": False,
                    "info": {"reached_goal": False},
                },
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_hi",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 1, "y": 0, "goal_x": 2, "goal_y": 2, "t": 1},
                    "action": 1,
                    "reward": 12.0,
                    "done": True,
                    "info": {"reached_goal": True},
                },
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            dataset_path = os.path.join(td, "adt_data.pt")
            prepare_adt_data(
                runs_root=runs_root,
                out_path=dataset_path,
                context_len=4,
                chunk_stride=2,
                state_dim=16,
                max_timestep=64,
                gamma=1.0,
                top_return_pct=1.0,
                min_episode_steps=1,
                max_runs=0,
                verse_filter=["warehouse_world"],
                dataset_paths=[],
                dataset_dir="",
            )

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
                seed=17,
                d_model=32,
                n_head=4,
                n_layer=2,
                dropout=0.0,
                max_timestep=64,
                device="cpu",
            )

            obs_space = SpaceSpec(type="dict")
            act_space = SpaceSpec(type="discrete", n=5)
            spec = AgentSpec(
                spec_version="v1",
                policy_id="adt_auto_rtg",
                policy_version="0.1",
                algo="adt",
                config={
                    "model_path": ckpt_path,
                    "context_len": 4,
                    "device": "cpu",
                    "target_return_auto": True,
                },
            )
            agent = TransformerAgent(spec=spec, observation_space=obs_space, action_space=act_space)
            self.assertGreater(float(agent._target_return), 1.0)  # inferred from high-RTG dataset

    def test_train_adt_class_weighting_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_unit")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            # Strongly imbalanced actions: mostly action=1, rare action=0.
            rows = []
            for i in range(20):
                rows.append(
                    {
                        "run_id": "run_unit",
                        "episode_id": f"ep_{i:03d}",
                        "step_idx": 0,
                        "verse_name": "warehouse_world",
                        "obs": {"x": i % 3, "y": 0, "goal_x": 2, "goal_y": 2, "t": 0},
                        "action": 1,
                        "reward": 0.1,
                        "done": True,
                        "info": {"reached_goal": False},
                    }
                )
            rows.append(
                {
                    "run_id": "run_unit",
                    "episode_id": "ep_rare",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0, "y": 1, "goal_x": 2, "goal_y": 2, "t": 0},
                    "action": 0,
                    "reward": 0.1,
                    "done": True,
                    "info": {"reached_goal": False},
                }
            )
            with open(events_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            dataset_path = os.path.join(td, "adt_data.pt")
            prepare_adt_data(
                runs_root=runs_root,
                out_path=dataset_path,
                context_len=4,
                chunk_stride=2,
                state_dim=16,
                max_timestep=64,
                gamma=1.0,
                top_return_pct=1.0,
                min_episode_steps=1,
                max_runs=0,
                verse_filter=["warehouse_world"],
                dataset_paths=[],
                dataset_dir="",
            )

            ckpt_path = os.path.join(td, "decision_transformer.pt")
            ckpt = train_adt(
                dataset_path=dataset_path,
                out_path=ckpt_path,
                init_model_path="",
                epochs=1,
                batch_size=8,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                val_split=0.0,
                seed=21,
                d_model=32,
                n_head=4,
                n_layer=2,
                dropout=0.0,
                max_timestep=64,
                device="cpu",
                class_weight_mode="inverse_sqrt",
                class_weight_min_count=1,
                class_weight_max=10.0,
            )
            meta = ckpt.get("class_weighting", {})
            self.assertEqual(str(meta.get("class_weight_mode_effective")), "inverse_sqrt")
            self.assertIn("action_counts", meta)
            self.assertIn("class_weights", meta)

    def test_prepare_action_balance_cap_ratio(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_unit")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            rows = []
            for i in range(30):
                rows.append(
                    {
                        "run_id": "run_unit",
                        "episode_id": f"ep_hi_{i:03d}",
                        "step_idx": 0,
                        "verse_name": "warehouse_world",
                        "obs": {"x": i % 5, "y": 0, "goal_x": 2, "goal_y": 2, "t": 0},
                        "action": 1,
                        "reward": 0.0,
                        "done": True,
                    }
                )
            for i in range(3):
                rows.append(
                    {
                        "run_id": "run_unit",
                        "episode_id": f"ep_lo_{i:03d}",
                        "step_idx": 0,
                        "verse_name": "warehouse_world",
                        "obs": {"x": i, "y": 1, "goal_x": 2, "goal_y": 2, "t": 0},
                        "action": 0,
                        "reward": 0.0,
                        "done": True,
                    }
                )
            with open(events_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            out_path = os.path.join(td, "adt_balanced.pt")
            meta = prepare_adt_data(
                runs_root=runs_root,
                out_path=out_path,
                context_len=1,
                chunk_stride=1,
                state_dim=16,
                max_timestep=64,
                gamma=1.0,
                top_return_pct=1.0,
                min_episode_steps=1,
                max_runs=0,
                verse_filter=["warehouse_world"],
                dataset_paths=[],
                dataset_dir="",
                action_balance_mode="cap_ratio",
                action_balance_max_ratio=1.0,
                action_balance_seed=7,
            )
            self.assertTrue(bool(meta.get("action_balance_applied", False)))
            data = torch.load(out_path, map_location="cpu", weights_only=False)
            actions = data["actions"]
            valid = actions[actions != -100]
            uniq, cnt = torch.unique(valid, return_counts=True)
            hist = {int(u.item()): int(c.item()) for u, c in zip(uniq, cnt)}
            self.assertIn(0, hist)
            self.assertIn(1, hist)
            self.assertEqual(int(hist[0]), int(hist[1]))


if __name__ == "__main__":
    unittest.main()
