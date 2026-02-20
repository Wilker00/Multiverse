import json
import os
import tempfile
import unittest

from core.types import AgentSpec, VerseSpec
from orchestrator.trainer import Trainer


class TestIntegrationSelfPlay(unittest.TestCase):
    def test_self_play_off_then_on_logs_tags(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            os.makedirs(runs_root, exist_ok=True)
            trainer = Trainer(run_root=runs_root, schema_version="v1", auto_register_builtin=True)

            verse_spec = VerseSpec(
                spec_version="v1",
                verse_name="grid_world",
                verse_version="0.1",
                seed=123,
                params={"max_steps": 12, "adr_enabled": False},
            )

            # Self-play OFF
            off_spec = AgentSpec(
                spec_version="v1",
                policy_id="off_random",
                policy_version="0.1",
                algo="random",
                seed=123,
                config=None,
            )
            off = trainer.run(verse_spec=verse_spec, agent_spec=off_spec, episodes=1, max_steps=12, seed=123)
            off_events = os.path.join(runs_root, str(off["run_id"]), "events.jsonl")
            self.assertTrue(os.path.isfile(off_events))

            saw_self_play_off = False
            with open(off_events, "r", encoding="utf-8") as f:
                for line in f:
                    ev = json.loads(line)
                    info = ev.get("info", {})
                    action_info = info.get("action_info") if isinstance(info, dict) else None
                    sp = action_info.get("self_play") if isinstance(action_info, dict) else None
                    if isinstance(sp, dict):
                        saw_self_play_off = True
                        break
            self.assertFalse(saw_self_play_off)

            # Self-play ON
            bundle = {
                "verse_name": "grid_world",
                "source": "recent_failures",
                "policy_id": "adversary:test",
                "created_at_ms": 0,
                "run_ids": ["r1"],
                "obs_actions": {},
                "global_actions": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0},
                "score": 1.0,
            }
            on_spec = AgentSpec(
                spec_version="v1",
                policy_id="on_random",
                policy_version="0.1",
                algo="random",
                seed=123,
                config={
                    "self_play": {
                        "enabled": True,
                        "mix_ratio": 0.5,
                        "adversary_source": "recent_failures",
                        "adversary_bundle": bundle,
                    }
                },
            )
            on = trainer.run(verse_spec=verse_spec, agent_spec=on_spec, episodes=1, max_steps=12, seed=123)
            on_events = os.path.join(runs_root, str(on["run_id"]), "events.jsonl")
            self.assertTrue(os.path.isfile(on_events))

            saw_self_play_tag = False
            with open(on_events, "r", encoding="utf-8") as f:
                for line in f:
                    ev = json.loads(line)
                    info = ev.get("info", {})
                    action_info = info.get("action_info") if isinstance(info, dict) else None
                    sp = action_info.get("self_play") if isinstance(action_info, dict) else None
                    if isinstance(sp, dict):
                        saw_self_play_tag = True
                        break
            self.assertTrue(saw_self_play_tag)


if __name__ == "__main__":
    unittest.main()
