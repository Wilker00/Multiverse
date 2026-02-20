import os
import tempfile
import unittest

from orchestrator.curriculum_controller import CurriculumConfig, CurriculumController


class TestCurriculumController(unittest.TestCase):
    def test_plateau_and_collapse_adjustment(self):
        with tempfile.TemporaryDirectory() as td:
            state = os.path.join(td, "curriculum.json")
            cfg = CurriculumConfig.from_dict({"state_path": state, "plateau_window": 3, "step_size": 0.1})
            ctl = CurriculumController(cfg)

            # Plateau: stable signals should push harder eventually.
            ctl.update_from_signal(verse_name="grid_world", success_rate=0.6, mean_return=0.2)
            ctl.update_from_signal(verse_name="grid_world", success_rate=0.6, mean_return=0.2)
            rec = ctl.update_from_signal(verse_name="grid_world", success_rate=0.6, mean_return=0.2)
            self.assertIn(rec.get("mode"), ("plateau_harder", "stable"))

            # Collapse should back off.
            rec2 = ctl.update_from_signal(verse_name="grid_world", success_rate=0.01, mean_return=-1.0)
            self.assertEqual(rec2.get("mode"), "collapse_backoff")


if __name__ == "__main__":
    unittest.main()

