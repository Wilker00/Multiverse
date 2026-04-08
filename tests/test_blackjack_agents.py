import json
import os
import tempfile
import unittest

from agents.blackjack_basic_agent import BlackjackBasicAgent
from agents.registry import DQNTrainerAdapter
from core.types import AgentSpec, SpaceSpec, VerseSpec
from orchestrator.trainer import Trainer
from tools.build_blackjack_dataset import build_blackjack_dataset_from_run
from tools.eval_blackjack import aggregate_blackjack_summaries, evaluate_blackjack_run, run_blackjack_case


class TestBlackjackAgents(unittest.TestCase):
    def _basic_agent(self) -> BlackjackBasicAgent:
        spec = AgentSpec(
            spec_version="v1",
            policy_id="blackjack_basic",
            policy_version="0.1",
            algo="blackjack_basic",
        )
        obs_space = SpaceSpec(type="dict", keys=[])
        act_space = SpaceSpec(type="discrete", n=4)
        return BlackjackBasicAgent(spec, obs_space, act_space)

    def test_basic_agent_uses_pair_and_soft_strategy(self):
        agent = self._basic_agent()

        split_obs = {
            "player_sum": 16,
            "dealer_showing": 6,
            "usable_ace": 0,
            "can_double": 1,
            "can_split": 1,
            "hand_len": 2,
            "pair_rank": 8,
        }
        self.assertEqual(int(agent.act(split_obs).action), 3)

        soft_obs = {
            "player_sum": 18,
            "dealer_showing": 6,
            "usable_ace": 1,
            "can_double": 1,
            "can_split": 0,
            "hand_len": 2,
            "pair_rank": 0,
        }
        self.assertEqual(int(agent.act(soft_obs).action), 2)

        hard_obs = {
            "player_sum": 16,
            "dealer_showing": 10,
            "usable_ace": 0,
            "can_double": 0,
            "can_split": 0,
            "hand_len": 2,
            "pair_rank": 0,
        }
        self.assertEqual(int(agent.act(hard_obs).action), 0)

    def test_dqn_adapter_masks_blackjack_illegal_actions_from_obs(self):
        spec = AgentSpec(
            spec_version="v1",
            policy_id="dqn_blackjack",
            policy_version="0.1",
            algo="dqn",
            config={"verse_name": "blackjack_world", "epsilon": 0.0, "blackjack_warmstart_samples": 512},
        )
        obs_space = SpaceSpec(type="dict", keys=[])
        act_space = SpaceSpec(type="discrete", n=4)
        agent = DQNTrainerAdapter(spec, obs_space, act_space)

        obs = {
            "player_sum": 21,
            "dealer_showing": 6,
            "can_double": 1,
            "can_split": 1,
        }
        self.assertTrue(bool(agent._warmstart_stats["enabled"]))
        self.assertGreater(int(agent._warmstart_stats["samples"]), 0)
        self.assertEqual(int(agent.act(obs).action), 1)

    def test_blackjack_evaluator_reports_domain_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = run_blackjack_case(
                algo="blackjack_basic",
                episodes=20,
                max_steps=20,
                seed=123,
                runs_root=tmpdir,
            )
            summary = evaluate_blackjack_run(run["run_dir"])

            self.assertEqual(int(summary["episodes"]), 20)
            self.assertGreaterEqual(float(summary["win_hand_rate"]), 0.0)
            self.assertLessEqual(float(summary["win_hand_rate"]), 1.0)
            self.assertGreaterEqual(float(summary["player_bust_rate"]), 0.0)
            self.assertLessEqual(float(summary["player_bust_rate"]), 1.0)
            self.assertIn("hands_total", summary)
            self.assertIn("learning_curve", summary)
            self.assertIn("action_counts", summary)

    def test_dqn_dataset_roundtrip_and_trainer_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "blackjack_dataset.jsonl")
            rows = [
                {
                    "obs": {"player_sum": 16, "dealer_showing": 10, "can_double": 0, "can_split": 0},
                    "action": 0,
                    "reward": -1.0,
                    "next_obs": {"player_sum": 22, "dealer_showing": 10, "can_double": 0, "can_split": 0},
                    "done": True,
                    "legal_actions": [1],
                },
                {
                    "obs": {"player_sum": 11, "dealer_showing": 6, "can_double": 1, "can_split": 0},
                    "action": 2,
                    "reward": 1.0,
                    "next_obs": {"player_sum": 21, "dealer_showing": 6, "can_double": 0, "can_split": 0},
                    "done": True,
                    "legal_actions": [1],
                },
            ]
            with open(dataset_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            spec = AgentSpec(
                spec_version="v1",
                policy_id="dqn_blackjack_offline",
                policy_version="0.1",
                algo="dqn",
                config={
                    "verse_name": "blackjack_world",
                    "epsilon": 0.0,
                    "blackjack_warmstart": False,
                    "batch_size": 2,
                    "dataset_train_steps": 2,
                    "behavior_clone_epochs": 1,
                    "behavior_clone_batch_size": 2,
                    "double_dqn": True,
                    "prioritized_replay": True,
                },
            )
            obs_space = SpaceSpec(type="dict", keys=[])
            act_space = SpaceSpec(type="discrete", n=4)
            agent = DQNTrainerAdapter(spec, obs_space, act_space)
            stats = agent.learn_from_dataset(dataset_path)

            self.assertEqual(int(stats["dataset_rows_added"]), 2)
            self.assertGreaterEqual(int(stats["dataset_train_steps"]), 1)
            self.assertIn("behavior_clone_loss", stats)
            self.assertEqual(len(agent._dqn.buffer), 2)

            checkpoint_dir = os.path.join(tmpdir, "agent_checkpoint")
            agent.save(checkpoint_dir)
            self.assertTrue(os.path.isfile(os.path.join(checkpoint_dir, "dqn.pt")))
            self.assertTrue(os.path.isfile(os.path.join(checkpoint_dir, "adapter_state.json")))

            spec_loaded = AgentSpec(
                spec_version="v1",
                policy_id="dqn_blackjack_loaded",
                policy_version="0.1",
                algo="dqn",
                config={
                    "verse_name": "blackjack_world",
                    "epsilon": 0.25,
                    "blackjack_warmstart": False,
                    "model_path": checkpoint_dir,
                },
            )
            loaded = DQNTrainerAdapter(spec_loaded, obs_space, act_space)
            self.assertTrue(str(loaded._loaded_model_path).endswith("dqn.pt"))
            self.assertEqual(len(loaded._dqn.buffer), 2)
            self.assertEqual(len(loaded._dqn.buffer.priorities), 2)
            self.assertEqual(int(loaded.act({"player_sum": 21, "dealer_showing": 6}).action), 1)

            eval_loaded = DQNTrainerAdapter(
                spec_loaded.evolved(config={**dict(spec_loaded.config), "train": False, "epsilon": 0.0}),
                obs_space,
                act_space,
            )
            self.assertEqual(float(eval_loaded._epsilon), 0.0)

            trainer = Trainer(run_root=tmpdir, schema_version="v1", auto_register_builtin=True)
            result = trainer.run(
                verse_spec=VerseSpec(
                    spec_version="v1",
                    verse_name="blackjack_world",
                    verse_version="0.1",
                    seed=9,
                    params={"max_steps": 20, "adr_enabled": False},
                ),
                agent_spec=spec.evolved(config={**dict(spec.config), "train": True}),
                episodes=2,
                max_steps=20,
                seed=9,
                verbose=False,
            )
            checkpoint_path = str(result.get("checkpoint_path", "") or "")
            self.assertTrue(os.path.isfile(os.path.join(checkpoint_path, "dqn.pt")))

    def test_blackjack_summary_aggregation_combines_runs(self):
        aggregate = aggregate_blackjack_summaries(
            [
                {
                    "run_id": "r1",
                    "run_dir": "runs/r1",
                    "episodes": 10,
                    "total_steps": 12,
                    "hands_total": 10,
                    "mean_return": -0.2,
                    "mean_steps": 1.2,
                    "mean_return_per_hand": -0.2,
                    "win_hand_rate": 0.4,
                    "push_hand_rate": 0.1,
                    "loss_hand_rate": 0.5,
                    "player_bust_rate": 0.2,
                    "double_hand_rate": 0.1,
                    "split_round_rate": 0.05,
                    "natural_blackjack_rate": 0.04,
                    "dealer_blackjack_rate": 0.03,
                    "push_blackjack_rate": 0.01,
                    "learning_curve": {"q1_mean_return": -0.3, "q4_mean_return": -0.1, "return_improvement": 0.2},
                    "action_counts": {"0": 6, "1": 4},
                    "outcomes": {"blackjack": 1},
                },
                {
                    "run_id": "r2",
                    "run_dir": "runs/r2",
                    "episodes": 10,
                    "total_steps": 14,
                    "hands_total": 11,
                    "mean_return": -0.1,
                    "mean_steps": 1.4,
                    "mean_return_per_hand": -0.09,
                    "win_hand_rate": 0.45,
                    "push_hand_rate": 0.1,
                    "loss_hand_rate": 0.45,
                    "player_bust_rate": 0.18,
                    "double_hand_rate": 0.12,
                    "split_round_rate": 0.06,
                    "natural_blackjack_rate": 0.05,
                    "dealer_blackjack_rate": 0.02,
                    "push_blackjack_rate": 0.01,
                    "learning_curve": {"q1_mean_return": -0.2, "q4_mean_return": 0.0, "return_improvement": 0.2},
                    "action_counts": {"0": 7, "2": 3},
                    "outcomes": {"dealer_blackjack": 1},
                },
            ]
        )
        self.assertEqual(int(aggregate["runs"]), 2)
        self.assertEqual(int(aggregate["episodes_total"]), 20)
        self.assertAlmostEqual(float(aggregate["mean_return"]), -0.15, places=6)
        self.assertEqual(int(aggregate["action_counts"]["0"]), 13)
        self.assertEqual(int(aggregate["outcomes"]["blackjack"]), 1)
        self.assertEqual(int(aggregate["outcomes"]["dealer_blackjack"]), 1)

    def test_synthetic_blackjack_rows_are_excluded_from_metrics_and_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_test")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            synthetic = {
                "run_id": "run_test",
                "episode_id": "ep_a",
                "step_idx": 0,
                "obs": {"player_sum": 21, "dealer_showing": 10},
                "action": None,
                "reward": 1.5,
                "done": True,
                "truncated": False,
                "info": {
                    "outcome": "blackjack",
                    "dealer_hand": [10, 7],
                    "player_hand": [11, 10],
                    "action_info": {"synthetic_action": True},
                },
            }
            real = {
                "run_id": "run_test",
                "episode_id": "ep_b",
                "step_idx": 0,
                "obs": {"player_sum": 16, "dealer_showing": 10},
                "action": 0,
                "reward": -1.0,
                "done": True,
                "truncated": False,
                "info": {
                    "dealer_hand": [10, 8],
                    "player_hands": [[10, 6, 9]],
                    "hand_bets": [1.0],
                },
            }
            with open(events_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(synthetic) + "\n")
                f.write(json.dumps(real) + "\n")

            summary = evaluate_blackjack_run(run_dir)
            self.assertEqual(summary["action_counts"], {"0": 1})

            out_path = os.path.join(tmpdir, "dataset.jsonl")
            payload = build_blackjack_dataset_from_run(
                run_dir=run_dir,
                out_path=out_path,
                source_algo="blackjack_basic",
                source_run_id="run_test",
                seed=123,
            )
            self.assertEqual(int(payload["rows"]), 1)
            self.assertEqual(int(payload["skipped_synthetic_rows"]), 1)
            with open(out_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["action"]), 0)


if __name__ == "__main__":
    unittest.main()
