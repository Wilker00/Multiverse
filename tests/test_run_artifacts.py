import json
import os
import tempfile
import time
import unittest

from agents.distributed_training import DistributedTrainingConfig, LocalDistributedTrainer
from core.parallel_rollout import DistributedRolloutManager, ParallelRolloutConfig
from core.run_artifacts import get_run_artifact_summary, write_run_artifact_manifest
from core.types import AgentSpec, VerseSpec
from orchestrator.trainer import Trainer


class TestRunArtifacts(unittest.TestCase):
    def test_manifest_summary_detects_stale_files(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run_x")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"episode_id\":\"1\"}\n")
            with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"loss\":0.1}\n")

            write_run_artifact_manifest(run_dir, verse_name="line_world", policy_id="random", algo="random")
            summary = get_run_artifact_summary(run_dir)
            self.assertTrue(bool(summary["manifest_present"]))
            self.assertFalse(bool(summary["manifest_stale"]))
            self.assertEqual(summary["missing_required_artifacts"], [])

            time.sleep(0.02)
            with open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write("{\"loss\":0.05}\n")

            stale_summary = get_run_artifact_summary(run_dir)
            self.assertTrue(bool(stale_summary["manifest_stale"]))

    def test_trainer_writes_run_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = Trainer(run_root=td, auto_register_builtin=True)
            verse_spec = VerseSpec(
                spec_version="v1",
                verse_name="line_world",
                verse_version="0.1",
                seed=123,
                params={"goal_pos": 4, "max_steps": 8, "step_penalty": -0.02},
            )
            agent_spec = AgentSpec(
                spec_version="v1",
                policy_id="random_smoke",
                policy_version="0.1",
                algo="random",
                seed=123,
            )
            out = trainer.run(
                verse_spec=verse_spec,
                agent_spec=agent_spec,
                episodes=1,
                max_steps=8,
                seed=123,
                verbose=False,
            )
            run_dir = os.path.join(td, str(out["run_id"]))
            manifest_path = os.path.join(run_dir, "run_manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["summary"]["verse_name"], "line_world")
            self.assertEqual(manifest["summary"]["policy_id"], "random_smoke")
            self.assertEqual(manifest["summary"]["algo"], "random")

    def test_distributed_merge_writes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = LocalDistributedTrainer(
                DistributedTrainingConfig(workers=1, run_root=td, merge_results=True)
            )
            verse_spec = VerseSpec(
                spec_version="v1",
                verse_name="line_world",
                verse_version="0.1",
                seed=123,
                params={"goal_pos": 4, "max_steps": 8, "step_penalty": -0.02},
            )
            agent_spec = AgentSpec(
                spec_version="v1",
                policy_id="dist_random",
                policy_version="0.1",
                algo="random",
                seed=123,
            )
            out = trainer.train(
                agent_spec=agent_spec,
                verse_spec=verse_spec,
                total_episodes=2,
                max_steps=8,
                seed=123,
            )
            run_dir = os.path.join(td, str(out["run_id"]))
            summary = get_run_artifact_summary(run_dir)
            self.assertTrue(bool(summary["manifest_present"]))
            self.assertEqual(summary["run_kind"], "distributed_aggregate")
            self.assertEqual(summary["summary"]["verse_name"], "line_world")

    def test_parallel_rollout_merge_writes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = DistributedRolloutManager(
                ParallelRolloutConfig(
                    num_workers=1,
                    use_ray=False,
                    run_root=td,
                    worker_auto_index=False,
                    worker_verbose=False,
                )
            )
            verse_spec = VerseSpec(
                spec_version="v1",
                verse_name="line_world",
                verse_version="0.1",
                seed=123,
                params={"goal_pos": 4, "max_steps": 8, "step_penalty": -0.02},
            )
            agent_spec = AgentSpec(
                spec_version="v1",
                policy_id="parallel_random",
                policy_version="0.1",
                algo="random",
                seed=123,
            )
            out = mgr.run_parallel_rollouts(
                agent_spec=agent_spec,
                verse_spec=verse_spec,
                total_episodes=2,
                max_steps=8,
                seed=123,
            )
            mgr.shutdown()
            run_dir = os.path.join(td, str(out["run_id"]))
            summary = get_run_artifact_summary(run_dir)
            self.assertTrue(bool(summary["manifest_present"]))
            self.assertEqual(summary["run_kind"], "parallel_aggregate")
            self.assertEqual(summary["summary"]["policy_id"], "parallel_random")


if __name__ == "__main__":
    unittest.main()
