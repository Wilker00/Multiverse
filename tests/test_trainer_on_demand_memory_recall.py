import json
import os
import tempfile
import unittest

from core.types import AgentSpec, VerseSpec
from memory.embeddings import obs_to_vector
from orchestrator.trainer import Trainer


class TestTrainerOnDemandMemoryRecall(unittest.TestCase):
    def test_memory_recall_agent_queries_central_memory_and_uses_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            mem_root = os.path.join(td, "central_memory")
            os.makedirs(runs_root, exist_ok=True)
            os.makedirs(mem_root, exist_ok=True)

            obs0 = {"pos": 0, "goal": 4, "t": 0}
            row = {
                "run_id": "memory_source",
                "episode_id": "ep_mem",
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

            trainer = Trainer(run_root=runs_root, schema_version="v1", auto_register_builtin=True)
            verse = VerseSpec(
                spec_version="v1",
                verse_name="line_world",
                verse_version="0.1",
                seed=7,
                tags=["test"],
                params={"goal_pos": 4, "max_steps": 1, "start_pos": 0},
            )
            agent = AgentSpec(
                spec_version="v1",
                policy_id="memory_recaller",
                policy_version="0.1",
                algo="memory_recall",
                seed=7,
                config={
                    "train": False,
                    "epsilon_start": 0.0,
                    "epsilon_min": 0.0,
                    "epsilon_decay": 1.0,
                    "on_demand_memory_enabled": True,
                    "on_demand_memory_root": mem_root,
                    "on_demand_query_budget": 2,
                    "on_demand_min_interval": 1,
                    "recall_top_k": 1,
                    "recall_min_score": -1.0,
                    "recall_same_verse_only": True,
                    "recall_vote_weight": 1.0,
                },
            )

            out = trainer.run(
                verse_spec=verse,
                agent_spec=agent,
                episodes=1,
                max_steps=1,
                seed=7,
            )
            run_id = str(out.get("run_id", ""))
            events_path = os.path.join(runs_root, run_id, "events.jsonl")
            self.assertTrue(os.path.isfile(events_path))
            with open(events_path, "r", encoding="utf-8") as f:
                event = json.loads(f.readline())

            info = event.get("info", {})
            self.assertEqual(int(event.get("action", -1)), 1)
            self.assertTrue(int((info.get("memory_query") or {}).get("used", 0)) >= 1)
            action_info = (info.get("action_info") or {})
            self.assertTrue(bool(action_info.get("memory_recall_used", False)))
            self.assertIn(
                "runs/memory_source/events.jsonl",
                str(action_info.get("memory_recall_pointer", "")),
            )


if __name__ == "__main__":
    unittest.main()

