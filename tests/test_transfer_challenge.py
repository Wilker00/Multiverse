import json
import os
import tempfile
import unittest

from core.taxonomy import bridge_reason, can_bridge
from memory.semantic_bridge import translate_action, translate_dna, translate_observation
from tools.run_transfer_challenge import (
    _SourceDNA,
    _adt_prior_rollback_decision,
    _align_with_baseline_scratch_schedule,
    _augment_translated_file_with_lane_metadata,
    _auto_tune_safe_veto_schedule,
    _auto_safe_veto_schedule_steps,
    _collect_run_eval,
    _auto_transfer_mix_decay_steps,
    _disable_transfer_warmstart,
    _extract_success_dna_from_events,
    _extract_top_return_dna_from_events,
    _filter_transfer_dataset,
    _first_passable_episode,
    _pick_robust_transfer_mode,
    _pick_sources_from_attribution,
    _speedup_summary,
    _train_td_diagnostics,
    _transfer_score_diagnostics,
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

    def test_near_universe_to_warehouse_mappings_are_translatable(self):
        lab_obs = {
            "x": 4,
            "y": 3,
            "goal_x": 13,
            "goal_y": 9,
            "battery": 72,
            "nearby_walls": 2,
            "nearby_pits": 1,
            "laser_nearby": 1,
            "nearest_charger_dist": 6,
            "t": 11,
        }
        tr_lab = translate_observation(
            obs=lab_obs,
            source_verse_name="labyrinth_world",
            target_verse_name="warehouse_world",
        )
        self.assertIsInstance(tr_lab, dict)
        assert isinstance(tr_lab, dict)
        self.assertIn("nearby_obstacles", tr_lab)
        self.assertIn("battery", tr_lab)
        self.assertTrue(0 <= int(tr_lab.get("x", -1)) <= 7)
        self.assertTrue(0 <= int(tr_lab.get("y", -1)) <= 7)

        park_obs = {"pos": 2, "goal": 5, "t": 4}
        tr_park = translate_observation(
            obs=park_obs,
            source_verse_name="park_world",
            target_verse_name="warehouse_world",
        )
        self.assertIsInstance(tr_park, dict)
        assert isinstance(tr_park, dict)
        self.assertIn("goal_x", tr_park)
        self.assertEqual(int(tr_park.get("y", -1)), 0)

        pursuit_obs = {"agent": 1, "target": 7, "t": 9}
        tr_pursuit = translate_observation(
            obs=pursuit_obs,
            source_verse_name="pursuit_world",
            target_verse_name="warehouse_world",
        )
        self.assertIsInstance(tr_pursuit, dict)
        assert isinstance(tr_pursuit, dict)
        self.assertIn("goal_x", tr_pursuit)
        self.assertEqual(int(tr_pursuit.get("y", -1)), 0)

        self.assertEqual(
            int(
                translate_action(
                    action=4,
                    source_verse_name="labyrinth_world",
                    target_verse_name="warehouse_world",
                )
            ),
            4,
        )
        self.assertEqual(
            int(
                translate_action(
                    action=2,
                    source_verse_name="park_world",
                    target_verse_name="warehouse_world",
                )
            ),
            4,
        )
        self.assertEqual(
            int(
                translate_action(
                    action=0,
                    source_verse_name="pursuit_world",
                    target_verse_name="warehouse_world",
                )
            ),
            2,
        )

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

    def test_extract_top_return_dna_from_events(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run_x")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            rows = [
                {
                    "episode_id": "ep_bad",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 0},
                    "action": 0,
                    "reward": -5.0,
                    "done": True,
                    "truncated": False,
                    "info": {"reached_goal": False},
                },
                {
                    "episode_id": "ep_mid",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 1},
                    "action": 1,
                    "reward": -1.0,
                    "done": True,
                    "truncated": False,
                    "info": {"reached_goal": False},
                },
                {
                    "episode_id": "ep_good",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "obs": {"x": 2},
                    "action": 2,
                    "reward": 2.0,
                    "done": True,
                    "truncated": False,
                    "info": {"reached_goal": True},
                },
            ]
            with open(events_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            out_path = os.path.join(run_dir, "dna_top_return.jsonl")
            written = _extract_top_return_dna_from_events(
                run_dir=run_dir,
                out_path=out_path,
                max_rows=100,
                top_return_pct=1.0 / 3.0,
            )
            self.assertEqual(int(written), 1)
            with open(out_path, "r", encoding="utf-8") as f:
                out_rows = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(out_rows), 1)
            self.assertEqual(str(out_rows[0].get("episode_id", "")), "ep_good")

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

    def test_filter_transfer_dataset_applies_min_transfer_confidence(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "transfer_conf.jsonl")
            rows = [
                {
                    "obs": {"x": 1, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 20, "nearby_obstacles": 1, "patrol_dist": -1, "t": 0},
                    "action": 3,
                    "reward": 0.1,
                    "next_obs": {"x": 2, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 19, "nearby_obstacles": 1, "patrol_dist": -1, "t": 1},
                    "done": False,
                    "transfer_confidence": 0.2,
                },
                {
                    "obs": {"x": 2, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 19, "nearby_obstacles": 0, "patrol_dist": -1, "t": 1},
                    "action": 3,
                    "reward": 0.2,
                    "next_obs": {"x": 3, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 18, "nearby_obstacles": 0, "patrol_dist": -1, "t": 2},
                    "done": False,
                    "transfer_confidence": 0.8,
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
                min_transfer_confidence=0.5,
            )
            self.assertEqual(int(fs.get("kept_rows", 0)), 1)
            self.assertEqual(int(fs.get("dropped_low_confidence", 0)), 1)

            with open(path, "r", encoding="utf-8") as f:
                kept = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(kept), 1)
            self.assertEqual(int(kept[0]["obs"].get("patrol_dist", 0)), -1)

    def test_translate_dna_estimates_next_obs_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.jsonl")
            out = os.path.join(td, "out.jsonl")
            rows = [
                {"episode_id": "ep1", "step_idx": 0, "obs": {"pos": 1, "goal": 5, "t": 0}, "action": 1},
                {"episode_id": "ep1", "step_idx": 1, "obs": {"pos": 2, "goal": 5, "t": 1}, "action": 1},
            ]
            with open(src, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            st = translate_dna(
                source_dna_path=src,
                target_verse_name="warehouse_world",
                output_path=out,
                source_verse_name="park_world",
                confidence_threshold=0.0,
            )
            self.assertGreater(int(st.translated_rows), 0)
            with open(out, "r", encoding="utf-8") as f:
                out_rows = [json.loads(ln) for ln in f if ln.strip()]
            self.assertGreaterEqual(len(out_rows), 1)
            first = out_rows[0]
            self.assertIsInstance(first.get("next_obs"), dict)

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

    def test_transfer_score_diagnostics_reports_by_lane(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "transfer.jsonl")
            rows = [
                {"transfer_score": 1.2, "source_lane": "near_universe", "universe_feature_score": 0.8},
                {"transfer_score": 0.7, "source_lane": "far_universe", "universe_feature_score": 0.5, "far_lane_weight_multiplier": 0.7},
                {"transfer_score": 0.9, "source_lane": "far_universe", "universe_feature_score": 0.6},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            d = _transfer_score_diagnostics(path)
            self.assertEqual(int(d.get("rows", 0)), 3)
            self.assertEqual(int(d.get("weighted_far_lane_rows", 0)), 1)
            by_lane = d.get("by_lane", {})
            self.assertEqual(int(by_lane.get("near_universe", {}).get("rows", 0)), 1)
            self.assertEqual(int(by_lane.get("far_universe", {}).get("rows", 0)), 2)
            self.assertIsNotNone(by_lane.get("far_universe", {}).get("transfer_score", {}).get("mean"))

    def test_augment_translated_file_applies_far_lane_weight(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "translated.jsonl")
            rows = [
                {
                    "obs": {"x": 1, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 20, "nearby_obstacles": 1, "t": 1},
                    "next_obs": {"x": 2, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 19, "nearby_obstacles": 1, "t": 2},
                    "action": 0,
                    "reward": 0.1,
                    "transfer_score": 1.0,
                }
            ]
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            src = _SourceDNA(
                verse_name="chess_world",
                path=p,
                run_id="r1",
                source_kind="explicit",
                source_lane="far_universe",
                source_universe="strategy_universe",
            )
            st = _augment_translated_file_with_lane_metadata(
                path=p,
                source=src,
                target_verse="warehouse_world",
                universe_adapter_enabled=True,
                far_lane_score_weight_enabled=True,
                far_lane_score_weight_strength=0.5,
                far_lane_min_universe_feature_score=0.0,
            )
            self.assertEqual(int(st.get("updated_rows", 0)), 1)
            self.assertEqual(int(st.get("far_lane_weighted_rows", 0)), 1)
            with open(p, "r", encoding="utf-8") as f:
                out_rows = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(out_rows), 1)
            self.assertIn("universe_transfer", out_rows[0])
            self.assertIn("universe_feature_score", out_rows[0])
            self.assertIn("transfer_score_pre_lane_weight", out_rows[0])
            self.assertLessEqual(
                float(out_rows[0]["transfer_score"]),
                float(out_rows[0]["transfer_score_pre_lane_weight"]),
            )

    def test_train_td_diagnostics_reports_replay_sampling_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run")
            os.makedirs(run_dir, exist_ok=True)
            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            rows = [
                {
                    "native_td_abs_mean": 0.2,
                    "transfer_td_abs_mean": 0.4,
                    "transfer_td_score_corr": 0.1,
                    "transfer_replay_sampled_score_mean": 0.8,
                    "transfer_replay_weighted_sampling": True,
                },
                {
                    "native_td_abs_mean": 0.3,
                    "transfer_td_abs_mean": 0.5,
                    "transfer_td_score_corr": 0.2,
                    "transfer_replay_sampled_score_mean": 1.2,
                    "transfer_replay_weighted_sampling": True,
                },
                {
                    "native_td_abs_mean": 0.9,
                    "transfer_replay_sampled_score_mean": 2.0,
                    "transfer_replay_weighted_sampling": False,
                },
            ]
            with open(metrics_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            d = _train_td_diagnostics(run_dir, early_episodes=2)
            self.assertEqual(int(d.get("episodes_logged", 0)), 3)
            self.assertEqual(int(d.get("early_episodes", 0)), 2)
            self.assertAlmostEqual(float(d.get("transfer_replay_sampled_score_mean_early", 0.0)), 1.0, places=6)
            self.assertTrue(bool(d.get("transfer_replay_weighted_sampling_enabled", False)))

    def test_disable_transfer_warmstart_zeroes_transfer_knobs(self):
        cfg = {
            "warmstart_reward_scale": 0.5,
            "dynamic_transfer_mix_enabled": True,
            "transfer_mix_start": 1.0,
            "transfer_mix_end": 0.1,
            "transfer_mix_decay_steps": 200,
            "transfer_replay_reward_scale": 0.8,
            "safe_executor": {"enabled": True},
        }
        out = _disable_transfer_warmstart(cfg)
        self.assertEqual(float(out.get("warmstart_reward_scale", -1.0)), 0.0)
        self.assertFalse(bool(out.get("dynamic_transfer_mix_enabled", True)))
        self.assertEqual(float(out.get("transfer_mix_start", -1.0)), 0.0)
        self.assertEqual(float(out.get("transfer_mix_end", -1.0)), 0.0)
        self.assertEqual(int(out.get("transfer_mix_decay_steps", 0)), 1)
        self.assertEqual(float(out.get("transfer_replay_reward_scale", -1.0)), 0.0)
        self.assertTrue(bool((out.get("safe_executor") or {}).get("enabled", False)))

    def test_align_with_baseline_scratch_schedule_overrides_exploration(self):
        cfg = {"epsilon_start": 0.8, "epsilon_min": 0.05, "epsilon_decay": 0.99, "warehouse_obs_key_mode": "direction_only"}
        baseline_cfg = {
            "epsilon_start": 0.05,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.999,
            "warehouse_obs_key_mode": "direction_only",
        }
        out = _align_with_baseline_scratch_schedule(cfg, baseline_cfg)
        self.assertEqual(float(out.get("epsilon_start", -1.0)), 0.05)
        self.assertEqual(float(out.get("epsilon_min", -1.0)), 0.01)
        self.assertEqual(float(out.get("epsilon_decay", -1.0)), 0.999)

    def test_pick_robust_transfer_mode_prefers_gated_positive_mode(self):
        decision = _pick_robust_transfer_mode(
            pilot_rows=[
                {"mode": "scratch_control", "success_rate": 0.20, "mean_return": -5.0, "hazard_per_1k": 250.0},
                {"mode": "transfer_all_lanes", "success_rate": 0.18, "mean_return": -4.0, "hazard_per_1k": 240.0},
                {"mode": "transfer_near_lane", "success_rate": 0.24, "mean_return": -3.8, "hazard_per_1k": 220.0},
            ],
            min_utility=0.0,
            min_success_delta=0.0,
            min_hazard_gain_per_1k=5.0,
            max_hazard_regression_per_1k=20.0,
            success_weight=100.0,
            return_weight=1.0,
            hazard_weight=0.02,
        )
        self.assertEqual(str(decision.get("selected_mode", "")), "transfer_near_lane")
        self.assertEqual(str(decision.get("reason", "")), "best_gated_utility")

    def test_pick_robust_transfer_mode_falls_back_to_scratch_when_gate_fails(self):
        decision = _pick_robust_transfer_mode(
            pilot_rows=[
                {"mode": "scratch_control", "success_rate": 0.30, "mean_return": -2.0, "hazard_per_1k": 200.0},
                {"mode": "transfer_all_lanes", "success_rate": 0.28, "mean_return": -1.0, "hazard_per_1k": 180.0},
            ],
            min_utility=0.0,
            min_success_delta=0.01,
            min_hazard_gain_per_1k=30.0,
            max_hazard_regression_per_1k=10.0,
            success_weight=100.0,
            return_weight=1.0,
            hazard_weight=0.02,
        )
        self.assertEqual(str(decision.get("selected_mode", "")), "scratch_control")
        self.assertIn(str(decision.get("reason", "")), {"no_candidate_passed_gate", "no_transfer_candidates"})

    def test_pick_sources_from_attribution_prunes_failed_sources(self):
        s1 = _SourceDNA("grid_world", "a.jsonl", "r1", "dna_good", "near_universe", "logistics_universe")
        s2 = _SourceDNA("chess_world", "b.jsonl", "r2", "dna_good", "far_universe", "strategy_universe")
        rows = [
            {"source_id": f"{s1.verse_name}|{s1.run_id}|{s1.source_kind}|{s1.path}", "gate_ok": True, "utility": 1.0},
            {"source_id": f"{s2.verse_name}|{s2.run_id}|{s2.source_kind}|{s2.path}", "gate_ok": False, "utility": -1.0},
        ]
        out = _pick_sources_from_attribution(
            all_sources=[s1, s2],
            attribution_rows=rows,
            min_keep_sources=1,
            keep_unscored=False,
        )
        kept = out.get("kept_sources") or []
        dropped = out.get("dropped_sources") or []
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(str(kept[0].verse_name), "grid_world")

    def test_adt_prior_rollback_decision_keeps_positive_early_transfer(self):
        d = _adt_prior_rollback_decision(
            candidate_success_rate=0.30,
            candidate_mean_return=-10.0,
            candidate_hazard_per_1k=180.0,
            baseline_success_rate=0.10,
            baseline_mean_return=-20.0,
            baseline_hazard_per_1k=220.0,
            min_success_delta=0.0,
            min_return_delta=0.0,
            max_hazard_regression_per_1k=25.0,
        )
        self.assertTrue(bool(d.get("keep_prior", False)))
        self.assertFalse(bool(d.get("rollback", True)))

    def test_adt_prior_rollback_decision_rejects_hazard_regression(self):
        d = _adt_prior_rollback_decision(
            candidate_success_rate=0.20,
            candidate_mean_return=-18.0,
            candidate_hazard_per_1k=260.0,
            baseline_success_rate=0.20,
            baseline_mean_return=-18.0,
            baseline_hazard_per_1k=220.0,
            min_success_delta=0.0,
            min_return_delta=0.0,
            max_hazard_regression_per_1k=25.0,
        )
        self.assertFalse(bool(d.get("keep_prior", True)))
        self.assertTrue(bool(d.get("rollback", False)))
        self.assertFalse(bool(d.get("hazard_regression_ok", True)))

    def test_collect_run_eval_returns_empty_safe_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            d = _collect_run_eval(td, early_episodes=5, action_first_k=3)
            self.assertIn("stats", d)
            self.assertEqual(len(d.get("curve") or []), 0)
            self.assertEqual(int((d.get("aggregate") or {}).get("episodes", -1)), 0)
            self.assertEqual(int((d.get("early_window") or {}).get("episodes", -1)), 0)


if __name__ == "__main__":
    unittest.main()
