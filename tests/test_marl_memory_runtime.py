import json
import os
import tempfile
import unittest

from core.types import AgentSpec, VerseSpec
from memory.embeddings import obs_to_vector
from orchestrator.marl_trainer import MARLConfig, MultiAgentTrainer


class TestMARLMemoryRuntime(unittest.TestCase):
    def test_marl_bootstrap_prefetches_memory_before_first_action(self):
        with tempfile.TemporaryDirectory() as td:
            mem_root = os.path.join(td, "central_memory")
            os.makedirs(mem_root, exist_ok=True)

            obs0 = {"pos": 0, "goal": 4, "t": 0}
            row = {
                "run_id": "marl_bootstrap_source",
                "episode_id": "ep_bootstrap",
                "step_idx": 0,
                "t_ms": 1,
                "verse_name": "line_world",
                "obs": obs0,
                "obs_vector": obs_to_vector(obs0),
                "action": 1,
                "reward": 1.0,
                "memory_tier": "ltm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
            with open(os.path.join(mem_root, "memories.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            trainer = MultiAgentTrainer(run_root=td, schema_version="v1", auto_register_builtin=True)
            verse_specs = [
                VerseSpec(
                    spec_version="v1",
                    verse_name="line_world",
                    verse_version="0.1",
                    seed=7,
                    tags=["test"],
                    params={"goal_pos": 4, "max_steps": 1, "start_pos": 0},
                )
            ]
            agent_specs = [
                AgentSpec(
                    spec_version="v1",
                    policy_id="marl_memory_bootstrap",
                    policy_version="0.1",
                    algo="memory_recall",
                    seed=7,
                    config={
                        "verse_name": "line_world",
                        "epsilon_start": 0.0,
                        "epsilon_min": 0.0,
                        "epsilon_decay": 1.0,
                        "bootstrap_recall_enabled": True,
                        "recall_same_verse_only": True,
                        "bootstrap_top_k": 1,
                        "bootstrap_min_score": -1.0,
                        "recall_vote_weight": 1.0,
                    },
                )
            ]
            cfg = MARLConfig(
                episodes=1,
                max_steps=1,
                train=False,
                collect_transitions=False,
                shared_memory_enabled=False,
                bootstrap_memory_enabled=True,
                on_demand_memory_enabled=True,
                on_demand_memory_root=mem_root,
                on_demand_query_budget=2,
                on_demand_min_interval=1,
            )

            out = trainer.run(verse_specs=verse_specs, agent_specs=agent_specs, config=cfg, seed=7)
            run_id = str(out.get("run_id", ""))
            events_path = os.path.join(td, run_id, "events.jsonl")
            self.assertTrue(os.path.isfile(events_path))

            with open(events_path, "r", encoding="utf-8") as f:
                event = json.loads(f.readline())

            self.assertEqual(int(event.get("action", -1)), 1)
            info = event.get("info", {})
            bootstrap = info.get("memory_bootstrap") or {}
            self.assertTrue(bool(bootstrap.get("query_executed", False)))
            self.assertEqual(str((info.get("memory_query") or {}).get("block_reason", "")), "bootstrap_prefetched")
            action_info = info.get("action_info") or {}
            self.assertTrue(bool(action_info.get("memory_recall_used", False)))
            self.assertIn(
                "runs/marl_bootstrap_source/events.jsonl",
                str(action_info.get("memory_recall_pointer", "")),
            )


if __name__ == "__main__":
    unittest.main()
