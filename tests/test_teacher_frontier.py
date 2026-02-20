import json
import os
import tempfile
import unittest

from orchestrator.teacher import TeacherConfig, append_teacher_lesson, build_teacher_plan


class TestTeacherFrontier(unittest.TestCase):
    def _write_events(self, run_dir: str, rows):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "events.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_teacher_generates_tutorial_for_high_risk_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_a")
            rows = [
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep1",
                    "obs": {"risk": 8},
                    "reward": -0.8,
                    "info": {"lost_game": True},
                },
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep2",
                    "obs": {"risk": 7},
                    "reward": -0.4,
                    "info": {"reached_goal": False},
                },
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep2",
                    "obs": {"risk": 3},
                    "reward": 0.2,
                    "info": {"reached_goal": True},
                },
            ]
            self._write_events(run_dir, rows)
            cfg = TeacherConfig.from_dict(
                {
                    "enabled": True,
                    "lookback_runs": 4,
                    "min_episodes": 2,
                    "high_risk_obs_threshold": 5.0,
                    "high_risk_failure_rate_threshold": 0.5,
                }
            )
            plan = build_teacher_plan(runs_root=runs_root, target_verse="chess_world", cfg=cfg, seed=123)
            self.assertIsNotNone(plan.tutorial_spec)
            self.assertEqual(plan.tutorial_spec.verse_name, "risk_tutorial_world")
            self.assertEqual(plan.reason, "high_risk_failure_pattern_detected")

            out_path = os.path.join(td, "teacher_lessons.json")
            append_teacher_lesson(
                path=out_path,
                plan=plan,
                tutorial_run_id="run_tutorial",
                graduation_run_id="run_grad",
            )
            self.assertTrue(os.path.isfile(out_path))

    def test_teacher_skips_when_signal_is_weak(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_b")
            rows = [
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep1",
                    "obs": {"risk": 2},
                    "reward": 0.5,
                    "info": {"reached_goal": True},
                },
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep2",
                    "obs": {"risk": 3},
                    "reward": 0.3,
                    "info": {"reached_goal": True},
                },
            ]
            self._write_events(run_dir, rows)
            cfg = TeacherConfig.from_dict(
                {
                    "enabled": True,
                    "lookback_runs": 4,
                    "min_episodes": 2,
                    "high_risk_obs_threshold": 5.0,
                    "high_risk_failure_rate_threshold": 0.4,
                }
            )
            plan = build_teacher_plan(runs_root=runs_root, target_verse="chess_world", cfg=cfg, seed=123)
            self.assertIsNone(plan.tutorial_spec)
            self.assertEqual(plan.reason, "insufficient_signal")

    def test_teacher_graduation_gate_uses_mastery_windows(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_c")
            rows = [
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep1",
                    "obs": {"risk": 8},
                    "reward": -0.4,
                    "info": {"lost_game": True},
                },
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep2",
                    "obs": {"risk": 8},
                    "reward": 0.4,
                    "info": {"reached_goal": True},
                },
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep3",
                    "obs": {"risk": 7},
                    "reward": 0.3,
                    "info": {"reached_goal": True},
                },
                {
                    "verse_name": "chess_world",
                    "episode_id": "ep4",
                    "obs": {"risk": 7},
                    "reward": 0.2,
                    "info": {"reached_goal": True},
                },
            ]
            self._write_events(run_dir, rows)
            lesson_path = os.path.join(td, "teacher_lessons.json")
            history = {
                "version": "v1",
                "updated_at_ms": 1,
                "lessons": [
                    {"t_ms": 1, "plan": {"target_verse": "chess_world", "concept": "risk_management", "signals": {"high_risk_failure_rate": 0.24}}},
                    {"t_ms": 2, "plan": {"target_verse": "chess_world", "concept": "risk_management", "signals": {"high_risk_failure_rate": 0.23}}},
                    {"t_ms": 3, "plan": {"target_verse": "chess_world", "concept": "risk_management", "signals": {"high_risk_failure_rate": 0.22}}},
                    {"t_ms": 4, "plan": {"target_verse": "chess_world", "concept": "risk_management", "signals": {"high_risk_failure_rate": 0.25}}},
                ],
            }
            with open(lesson_path, "w", encoding="utf-8") as f:
                json.dump(history, f)
            cfg = TeacherConfig.from_dict(
                {
                    "enabled": True,
                    "lookback_runs": 4,
                    "min_episodes": 2,
                    "high_risk_obs_threshold": 5.0,
                    "high_risk_failure_rate_threshold": 0.2,
                    "lesson_log_path": lesson_path,
                    "mastery_window": 2,
                    "graduation_confidence_threshold": 0.6,
                }
            )
            plan = build_teacher_plan(runs_root=runs_root, target_verse="chess_world", cfg=cfg, seed=123)
            gate = plan.graduation_gate
            self.assertTrue(isinstance(gate, dict))
            self.assertIn("confidence", gate)
            self.assertTrue(bool(gate.get("stable_windows", False)))
            self.assertGreaterEqual(float(gate.get("confidence", 0.0)), 0.6)


if __name__ == "__main__":
    unittest.main()
