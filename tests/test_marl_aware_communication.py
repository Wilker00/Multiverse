import json
import os
import tempfile
import unittest

from core.types import AgentSpec, VerseSpec
from orchestrator.marl_trainer import MARLConfig, MultiAgentTrainer


class TestMARLAwareCommunication(unittest.TestCase):
    def test_aware_agents_use_social_contract_and_shared_hints(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = MultiAgentTrainer(run_root=td, schema_version="v1", auto_register_builtin=True)
            verse_specs = [
                VerseSpec(
                    spec_version="v1",
                    verse_name="line_world",
                    verse_version="0.1",
                    seed=101,
                    tags=["test"],
                    params={"goal_pos": 6, "max_steps": 10, "step_penalty": -0.01},
                ),
                VerseSpec(
                    spec_version="v1",
                    verse_name="line_world",
                    verse_version="0.1",
                    seed=202,
                    tags=["test"],
                    params={"goal_pos": 6, "max_steps": 10, "step_penalty": -0.01},
                ),
            ]
            aware_cfg = {
                "verse_name": "line_world",
                "epsilon_start": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 1.0,
                "min_aware_epsilon": 0.0,
                "novelty_explore_bonus": 0.0,
                "performance_explore_boost": 0.0,
                "use_vector_memory": False,
                "shared_hint_weight": 1.0,
            }
            agent_specs = [
                AgentSpec(
                    spec_version="v1",
                    policy_id="aware_a0",
                    policy_version="0.1",
                    algo="aware",
                    seed=101,
                    config=dict(aware_cfg),
                ),
                AgentSpec(
                    spec_version="v1",
                    policy_id="aware_a1",
                    policy_version="0.1",
                    algo="aware",
                    seed=202,
                    config=dict(aware_cfg),
                ),
            ]
            cfg = MARLConfig(
                episodes=3,
                max_steps=10,
                train=True,
                collect_transitions=True,
                shared_memory_enabled=True,
                shared_memory_top_k=3,
                negotiation_interval=1,
                lexicon_min_support=1,
            )
            out = trainer.run(verse_specs=verse_specs, agent_specs=agent_specs, config=cfg, seed=123)
            snap = out.get("shared_memory", {})
            self.assertGreaterEqual(int((snap or {}).get("trajectory_count", 0)), 2)

            run_id = str(out.get("run_id", ""))
            events_path = os.path.join(td, run_id, "events.jsonl")
            self.assertTrue(os.path.isfile(events_path))

            social_contract_events = 0
            shared_hint_events = 0
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    event = json.loads(line)
                    info = event.get("info", {}) if isinstance(event, dict) else {}
                    action_info = info.get("action_info", {}) if isinstance(info, dict) else {}
                    if bool(action_info.get("social_contract_active", False)):
                        social_contract_events += 1
                    if bool(action_info.get("shared_hint_used", False)):
                        shared_hint_events += 1

            self.assertGreater(social_contract_events, 0)
            self.assertGreater(shared_hint_events, 0)


if __name__ == "__main__":
    unittest.main()

