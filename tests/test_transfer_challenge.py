import json
import os
import tempfile
import unittest

from core.taxonomy import bridge_reason, can_bridge
from memory.semantic_bridge import translate_action, translate_observation
from tools.run_transfer_challenge import (
    _auto_tune_safe_veto_schedule,
    _auto_safe_veto_schedule_steps,
    _auto_transfer_mix_decay_steps,
    _extract_success_dna_from_events,
    _filter_transfer_dataset,
    _first_passable_episode,
    _speedup_summary,
)


class TestTransferChallenge(unittest.TestCase):
    def test_strategy_to_warehouse_bridge(self):
        obs = {
            "material_delta": 3,
            "development": 5,
            "king_safety": 6,
            "center_control": 2,
            "score_delta": 4,
            "pressure": 6,
            "risk": 1,
            "tempo": 3,
            "control": 2,
            "resource": 5,
            "phase": 1,
            "t": 7,
        }
        tr = translate_observation(
            obs=obs,
            source_verse_name="chess_world",
            target_verse_name="warehouse_world",
        )
        self.assertIsInstance(tr, dict)
        assert isinstance(tr, dict)
        self.assertIn("x", tr)
        self.assertIn("y", tr)
        self.assertIn("battery", tr)
        self.assertEqual(int(tr.get("goal_x", -1)), 7)
        self.assertEqual(int(tr.get("goal_y", -1)), 7)

        a = translate_action(
            action=3,
            source_verse_name="chess_world",
            target_verse_name="warehouse_world",
        )
        self.assertEqual(int(a), 4)  # defend -> wait/charge
        self.assertTrue(can_bridge("chess_world", "warehouse_world"))
        self.assertEqual(bridge_reason("chess_world", "warehouse_world"), "transferable_logic_projection")

    def test_extract_success_dna_from_events(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run_x")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            rows = [
                {
                    "episode_id": "ep_success",
                    "step_idx": 0,
                    "verse_name": "chess_world",
                    "obs": {"a": 1},
                    "action": 0,
                    "reward": 0.1,
                    "done": False,
                    "truncated": False,
                    "info": {"reached_goal": False},
                },
                {
                    "episode_id": "ep_success",
                    "step_idx": 1,
                    "verse_name": "chess_world",
                    "obs": {"a": 2},
                    "action": 1,
                    "reward": 0.3,
                    "done": True,
                    "truncated": False,
                    "info": {"reached_goal": True},
                },
                {
                    "episode_id": "ep_fail",
                    "step_idx": 0,
                    "verse_name": "chess_world",
                    "obs": {"a": 5},
                    "action": 2,
                    "reward": -0.2,
                    "done": True,
                    "truncated": False,
                    "info": {"reached_goal": False},
                },
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            out_path = os.path.join(run_dir, "dna_success_only.jsonl")
            written = _extract_success_dna_from_events(run_dir=run_dir, out_path=out_path, max_rows=100)
            self.assertEqual(int(written), 2)
            self.assertTrue(os.path.isfile(out_path))

            with open(out_path, "r", encoding="utf-8") as f:
                out_rows = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(out_rows), 2)
            self.assertTrue(all(str(r.get("episode_id")) == "ep_success" for r in out_rows))

    def test_passable_episode_and_speedup(self):
        transfer_curve = [
            {"return_sum": -0.2, "success": False},
            {"return_sum": 0.5, "success": True},
            {"return_sum": 0.6, "success": True},
        ]
        baseline_curve = [
            {"return_sum": -0.3, "success": False},
            {"return_sum": -0.1, "success": False},
            {"return_sum": 0.4, "success": True},
        ]
        t_first = _first_passable_episode(
            transfer_curve,
            window=2,
            passable_success_rate=0.5,
            passable_mean_return=0.3,
        )
        b_first = _first_passable_episode(
            baseline_curve,
            window=2,
            passable_success_rate=0.5,
            passable_mean_return=0.3,
        )
        self.assertEqual(int(t_first or 0), 2)
        self.assertEqual(int(b_first or 0), 3)

        cmp = _speedup_summary(
            transfer_first_passable=t_first,
            baseline_first_passable=b_first,
            transfer_hazard_rate=4.0,
            baseline_hazard_rate=10.0,
        )
        self.assertTrue(bool(cmp.get("transfer_wins_convergence", False)))
        self.assertGreater(float(cmp.get("transfer_speedup_ratio", 0.0) or 0.0), 1.0)
        self.assertGreater(float(cmp.get("hazard_improvement_pct", 0.0)), 0.0)

    def test_filter_transfer_dataset_dedup_and_normalize(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "transfer.jsonl")
            rows = [
                {
                    "obs": {"x": 1, "y": 2, "goal_x": 9, "goal_y": 9, "battery": 25, "nearby_obstacles": 1, "t": 1},
                    "action": 3,
                    "reward": 0.3,
                    "next_obs": {"x": 2, "y": 2, "goal_x": 9, "goal_y": 9, "battery": 24, "nearby_obstacles": 1, "t": 2},
                    "done": False,
                },
                {  # duplicate
                    "obs": {"x": 1, "y": 2, "goal_x": 9, "goal_y": 9, "battery": 25, "nearby_obstacles": 1, "t": 1},
                    "action": 3,
                    "reward": 0.3,
                    "next_obs": {"x": 2, "y": 2, "goal_x": 9, "goal_y": 9, "battery": 24, "nearby_obstacles": 1, "t": 2},
                    "done": False,
                },
                {  # invalid action
                    "obs": {"x": 0, "y": 0, "goal_x": 7, "goal_y": 7, "battery": 20, "nearby_obstacles": 0, "t": 0},
                    "action": 99,
                    "reward": 0.1,
                    "next_obs": {"x": 0, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 19, "nearby_obstacles": 0, "t": 1},
                    "done": False,
                },
            ]
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            fs = _filter_transfer_dataset(
                path=path,
                target_verse="warehouse_world",
                dedupe=True,
                max_rows=0,
            )
            self.assertEqual(int(fs.get("input_rows", 0)), 3)
            self.assertEqual(int(fs.get("kept_rows", 0)), 1)
            self.assertGreaterEqual(int(fs.get("dropped_duplicates", 0)), 1)
            self.assertGreaterEqual(int(fs.get("dropped_invalid", 0)), 1)
            with open(path, "r", encoding="utf-8") as f:
                kept = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(kept), 1)
            obs = kept[0]["obs"]
            self.assertEqual(int(obs["goal_x"]), 7)  # clamped
            self.assertEqual(int(obs["goal_y"]), 7)

    def test_auto_transfer_mix_decay_steps_is_stable_and_positive(self):
        d1 = _auto_transfer_mix_decay_steps(
            episodes=100,
            max_steps=80,
            transfer_rows=500,
            mix_start=1.0,
            mix_end=0.0,
        )
        d2 = _auto_transfer_mix_decay_steps(
            episodes=100,
            max_steps=80,
            transfer_rows=5000,
            mix_start=1.0,
            mix_end=0.0,
        )
        self.assertGreater(int(d1), 0)
        self.assertGreater(int(d2), 0)
        self.assertGreaterEqual(int(d2), int(d1))

    def test_auto_safe_veto_schedule_steps_is_stable_and_positive(self):
        d1 = _auto_safe_veto_schedule_steps(
            episodes=80,
            max_steps=100,
            transfer_rows=400,
        )
        d2 = _auto_safe_veto_schedule_steps(
            episodes=80,
            max_steps=100,
            transfer_rows=4000,
        )
        self.assertGreater(int(d1), 0)
        self.assertGreater(int(d2), 0)
        self.assertGreaterEqual(int(d2), int(d1))

    def test_auto_tune_safe_veto_schedule_becomes_more_conservative_when_trend_worsens(self):
        tuned = _auto_tune_safe_veto_schedule(
            base_relax_start=0.05,
            base_relax_end=0.35,
            base_schedule_steps=500,
            base_schedule_power=1.2,
            trend={
                "num_runs": 6,
                "mean_hazard_trend_ratio": 1.25,
                "mean_hazard_per_1k": 460.0,
                "improving_share": 0.20,
            },
        )
        self.assertTrue(bool(tuned.get("applied", False)))
        self.assertGreaterEqual(int(tuned.get("schedule_steps", 0)), 500)
        self.assertLessEqual(float(tuned.get("relax_end", 1.0)), 0.35)
        self.assertGreaterEqual(float(tuned.get("schedule_power", 0.0)), 1.2)


if __name__ == "__main__":
    unittest.main()
