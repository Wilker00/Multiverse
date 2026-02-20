import json
import os
import tempfile
import unittest

from core.types import AgentSpec, VerseSpec
from memory.embeddings import obs_to_vector
from orchestrator.trainer import Trainer


class TestTrainerPlannerRecall(unittest.TestCase):
    def test_planner_recall_uses_declarative_family_in_early_phase(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            mem_root = os.path.join(td, "central_memory")
            os.makedirs(runs_root, exist_ok=True)
            os.makedirs(mem_root, exist_ok=True)

            obs0 = {"pos": 0, "goal": 4, "t": 0}
            rows = [
                {
                    "run_id": "run_decl",
                    "episode_id": "ep_decl",
                    "step_idx": 0,
                    "t_ms": 1,
                    "verse_name": "line_world",
                    "obs": obs0,
                    "obs_vector": obs_to_vector(obs0),
                    "action": 1,
                    "reward": 0.9,
                    "memory_tier": "ltm",
                    "memory_family": "declarative",
                    "memory_type": "strategic_declarative",
                },
                {
                    "run_id": "run_proc",
                    "episode_id": "ep_proc",
                    "step_idx": 0,
                    "t_ms": 2,
                    "verse_name": "line_world",
                    "obs": obs0,
                    "obs_vector": obs_to_vector(obs0),
                    "action": 0,
                    "reward": 0.8,
                    "memory_tier": "ltm",
                    "memory_family": "procedural",
                    "memory_type": "spatial_procedural",
                },
            ]
            with open(os.path.join(mem_root, "memories.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            trainer = Trainer(run_root=runs_root, schema_version="v1", auto_register_builtin=True)
            verse = VerseSpec(
                spec_version="v1",
                verse_name="line_world",
                verse_version="0.1",
                seed=9,
                tags=["test"],
                params={"goal_pos": 4, "max_steps": 1, "start_pos": 0},
            )
            agent = AgentSpec(
                spec_version="v1",
                policy_id="planner_recaller",
                policy_version="0.1",
                algo="planner_recall",
                seed=9,
                config={
                    "train": False,
                    "epsilon_start": 0.0,
                    "epsilon_min": 0.0,
                    "epsilon_decay": 1.0,
                    "planner_force_recall": True,
                    "planner_expected_horizon": 20,
                    "recall_top_k": 1,
                    "recall_min_score": -1.0,
                    "recall_same_verse_only": True,
                    "recall_vote_weight": 1.5,
                    "on_demand_memory_enabled": True,
                    "on_demand_memory_root": mem_root,
                    "on_demand_query_budget": 2,
                    "on_demand_min_interval": 1,
                },
            )

            out = trainer.run(
                verse_spec=verse,
                agent_spec=agent,
                episodes=1,
                max_steps=1,
                seed=9,
            )
            run_id = str(out.get("run_id", ""))
            events_path = os.path.join(runs_root, run_id, "events.jsonl")
            with open(events_path, "r", encoding="utf-8") as f:
                event = json.loads(f.readline())

            self.assertEqual(int(event.get("action", -1)), 1)
            info = event.get("info", {})
            ai = (info.get("action_info") or {})
            self.assertTrue(bool(ai.get("memory_recall_used", False)))
            self.assertIn("phase_early_planner", str(ai.get("memory_recall_reason", "")))
            self.assertIn("runs/run_decl/events.jsonl", str(ai.get("memory_recall_pointer", "")))


if __name__ == "__main__":
    unittest.main()

