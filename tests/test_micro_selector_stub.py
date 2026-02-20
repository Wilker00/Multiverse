import os
import tempfile
import unittest

import torch

from models.micro_selector import train_selector_stub


def _lesson_text(title: str, context: str, utility: float, actions: list[str]) -> str:
    lines = [
        f"TITLE: {title}",
        f"CONTEXT: {context}",
        "SOURCE_RUN: run_test",
        "REWARD: 1.0",
        "ADVANTAGE: 1.0",
        f"UTILITY_SCORE: {utility}",
        "",
        "SEQUENCE:",
    ]
    for idx, action in enumerate(actions):
        lines.append(f"  {idx+1}. DO_ACTION({action})")
    lines.append("")
    return "\n".join(lines)


class TestMicroSelectorStub(unittest.TestCase):
    def test_train_selector_stub_trains_and_saves(self):
        with tempfile.TemporaryDirectory() as td:
            lessons_dir = os.path.join(td, "lessons")
            os.makedirs(lessons_dir, exist_ok=True)
            with open(os.path.join(lessons_dir, "skill_a.txt"), "w", encoding="utf-8") as f:
                f.write(_lesson_text("A route", "line_world", 2.0, ["0", "1", "1"]))
            with open(os.path.join(lessons_dir, "skill_b.txt"), "w", encoding="utf-8") as f:
                f.write(_lesson_text("B route", "grid_world", 1.0, ["2", "2"]))

            model_path = os.path.join(td, "selector.pt")
            metrics = train_selector_stub(lessons_dir=lessons_dir, state_dim=8, model_save_path=model_path)

            self.assertTrue(metrics["saved"])
            self.assertGreater(metrics["num_samples"], 0)
            self.assertGreaterEqual(metrics["train_accuracy"], 0.0)
            self.assertLessEqual(metrics["train_accuracy"], 1.0)
            self.assertTrue(os.path.isfile(model_path))

            payload = torch.load(model_path, weights_only=False)
            self.assertIn("model_state_dict", payload)
            self.assertIn("lesson_vocab", payload)
            self.assertIn("train_metrics", payload)
            self.assertEqual(len(payload["lesson_vocab"]), 2)

    def test_train_selector_stub_handles_empty_lessons(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "selector.pt")
            metrics = train_selector_stub(lessons_dir=td, state_dim=4, model_save_path=model_path)
            self.assertFalse(metrics["saved"])
            self.assertEqual(metrics["reason"], "no_lessons")
            self.assertFalse(os.path.isfile(model_path))


if __name__ == "__main__":
    unittest.main()
