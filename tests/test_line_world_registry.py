import unittest

from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


class TestLineWorldRegistry(unittest.TestCase):
    def test_registry_uses_primary_line_world_module(self):
        register_builtin()
        verse = create_verse(
            VerseSpec(
                spec_version="v1",
                verse_name="line_world",
                verse_version="0.1",
                seed=7,
                params={},
            )
        )
        self.assertEqual(type(verse).__module__, "verses.line_world")
        self.assertEqual(int(verse.params.lane_len), 10)
        self.assertEqual(int(verse.params.max_steps), 30)
        self.assertAlmostEqual(float(verse.params.step_penalty), -0.02)
        self.assertAlmostEqual(float(verse.params.goal_reward), 1.0)


if __name__ == "__main__":
    unittest.main()
