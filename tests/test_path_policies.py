import json
import os
import tempfile
import unittest

from policies.path_agent import PathAgent
from policies.skill_path import SkillPath, SkillPathConfig, create_skill_path


class TestPathPolicies(unittest.TestCase):
    def test_skill_path_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            skill = SkillPath(
                skill_id="skill_roundtrip",
                config={"network_arch": "tabular_lookup"},
                weights={
                    "type": "tabular_vote",
                    "default_action": 2,
                    "default_confidence": 0.75,
                    "obs_policy": {
                        '{"x":1}': {"action": 1, "confidence": 1.0},
                    },
                },
                tags=["grid", "grid", ""],
            )
            saved_path = skill.save(td)
            loaded = SkillPath.load(saved_path)

            p_known = loaded.predict({"x": 1})
            p_unknown = loaded.predict({"x": 99})

            self.assertEqual(loaded.skill_id, "skill_roundtrip")
            self.assertEqual(loaded.tags, ["grid"])
            self.assertEqual(p_known["action"], 1)
            self.assertAlmostEqual(float(p_known["confidence"]), 1.0, places=6)
            self.assertEqual(p_unknown["action"], 2)
            self.assertAlmostEqual(float(p_unknown["confidence"]), 0.75, places=6)

    def test_create_skill_path_from_dna(self):
        with tempfile.TemporaryDirectory() as td:
            dna_path = os.path.join(td, "dna.jsonl")
            rows = [
                {"obs": {"cell": 1}, "action": "left", "advantage": 0.7},
                {"obs": {"cell": 1}, "action": "left", "advantage": 0.9},
                {"obs": {"cell": 2}, "action": "right", "advantage": 0.8},
                {"obs": {"cell": 2}, "action": "right", "advantage": 0.1},
            ]
            with open(dna_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            cfg = SkillPathConfig(
                dna_log_path=dna_path,
                skill_id="from_dna",
                source_verse_tags=["line_world", "line_world"],
                min_advantage=0.5,
                epochs=2,
            )
            skill = create_skill_path(cfg)

            self.assertEqual(skill.tags, ["line_world"])
            self.assertEqual(skill.predict({"cell": 1})["action"], "left")
            self.assertEqual(skill.predict({"cell": 2})["action"], "right")
            self.assertAlmostEqual(float(skill.weights["default_confidence"]), 2.0 / 3.0, places=6)

    def test_path_agent_votes_and_learns(self):
        with tempfile.TemporaryDirectory() as td:
            skill_a = SkillPath(
                skill_id="skill_a",
                config={},
                weights={
                    "type": "tabular_vote",
                    "default_action": 0,
                    "default_confidence": 0.9,
                    "obs_policy": {},
                },
                tags=[],
            )
            skill_b = SkillPath(
                skill_id="skill_b",
                config={},
                weights={
                    "type": "tabular_vote",
                    "default_action": 1,
                    "default_confidence": 0.6,
                    "obs_policy": {
                        '{"x":1}': {"action": 1, "confidence": 1.0},
                    },
                },
                tags=[],
            )
            skill_a.save(td)
            skill_b.save(td)

            agent = PathAgent(
                skill_ids=["skill_a", "skill_b"],
                skill_library_dir=td,
                controller_lr=0.2,
            )

            first_action = agent.act({"x": 1})
            self.assertEqual(first_action, 1)

            for _ in range(8):
                metrics = agent.learn([{"reward": 3.0, "skill_id": "skill_a"}])
            self.assertGreater(metrics["controller_bias"]["skill_a"], 1.0)

            second_action = agent.act({"x": 1})
            self.assertEqual(second_action, 0)

    def test_path_agent_missing_skill_file(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                PathAgent(skill_ids=["missing_skill"], skill_library_dir=td)


if __name__ == "__main__":
    unittest.main()
