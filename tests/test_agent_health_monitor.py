import json
import os
import tempfile
import unittest

from tools.agent_health_monitor import (
    EventRunMetrics,
    TraceVerseMetrics,
    _apply_manifest_fallback,
    _collect_event_run_metrics,
    _score_row,
)


def _write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class TestAgentHealthMonitor(unittest.TestCase):
    def test_collect_event_run_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run_x")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            rows = [
                {
                    "episode_id": "ep1",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "policy_id": "special_moe",
                    "action": 2,
                    "info": {
                        "action_info": {"selector_active": True},
                        "safe_executor": {
                            "mcts_stats": {"queries": 1, "vetoes": 0, "last_query": {"best_action": 2}},
                            "counters": {"shield_vetoes": 1},
                        },
                    },
                },
                {
                    "episode_id": "ep1",
                    "step_idx": 1,
                    "verse_name": "warehouse_world",
                    "policy_id": "special_moe",
                    "action": 1,
                    "info": {
                        "action_info": {"selector_active": True},
                        "safe_executor": {
                            "mcts_stats": {"queries": 2, "vetoes": 1, "last_query": {"best_action": 3}},
                            "counters": {"shield_vetoes": 2},
                        },
                    },
                },
                {
                    "episode_id": "ep2",
                    "step_idx": 0,
                    "verse_name": "warehouse_world",
                    "policy_id": "special_moe",
                    "action": 0,
                    "info": {
                        "action_info": {"selector_active": False},
                        "safe_executor": {
                            "mcts_stats": {"queries": 1, "vetoes": 0, "last_query": {"best_action": 0}},
                            "counters": {"shield_vetoes": 0},
                        },
                    },
                },
            ]
            _write_jsonl(events_path, rows)

            m = _collect_event_run_metrics(run_dir)
            self.assertIsNotNone(m)
            assert m is not None
            self.assertEqual(m.verse_name, "warehouse_world")
            self.assertEqual(m.policy_id, "special_moe")
            self.assertEqual(m.selector_samples, 2)
            self.assertAlmostEqual(float(m.selector_match or 0.0), 0.5, places=6)
            self.assertEqual(m.mcts_queries, 3)  # ep1=2, ep2=1
            self.assertEqual(m.mcts_vetoes, 1)
            self.assertEqual(m.shield_vetoes, 2)

    def test_score_row_flags_issues(self):
        rm = EventRunMetrics(
            run_id="run_a",
            run_dir="runs/run_a",
            verse_name="warehouse_world",
            policy_id="special_moe",
            selector_match=0.4,
            selector_samples=10,
            mcts_queries=10,
            mcts_vetoes=3,
            shield_vetoes=8,
            total_steps=20,
        )
        tr = TraceVerseMetrics(
            verse_name="warehouse_world",
            rows=100,
            mean_kl=0.2,
            prior_top1_match=0.5,
            high_quality_rate=0.3,
        )
        row = _score_row(
            run_metrics=rm,
            trace_metrics=tr,
            mean_return=-5.0,
            success_rate=0.1,
            market_reputation=0.5,
            kl_critical=0.25,
            unsafe_veto_rate=0.1,
            incoherent_match_threshold=0.55,
            stale_kl_threshold=0.12,
        )
        self.assertIn("incoherent", row.issues)
        self.assertIn("stale", row.issues)
        self.assertIn("unsafe", row.issues)
        self.assertIn("trigger_hq_trace_retraining", row.recommended_actions)
        self.assertIn("swap_manifest_to_fallback", row.recommended_actions)
        self.assertIn("retrain_micro_selector", row.recommended_actions)

    def test_apply_manifest_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            mp = os.path.join(td, "manifest.json")
            manifest = {
                "deployment_ready_defaults": {
                    "warehouse_world": {
                        "picked_run": {"run_id": "run_new", "policy": "special_moe"},
                    }
                },
                "deployment_history": {
                    "warehouse_world": [
                        {
                            "picked_run": {
                                "run_id": "run_old",
                                "policy": "imitation_lookup",
                            }
                        }
                    ]
                },
            }
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            res = _apply_manifest_fallback(manifest_path=mp, verse_name="warehouse_world")
            self.assertEqual(res, "manifest_fallback_applied")
            with open(mp, "r", encoding="utf-8") as f:
                out = json.load(f)
            picked = out["deployment_ready_defaults"]["warehouse_world"]["picked_run"]
            self.assertEqual(str(picked.get("run_id")), "run_old")
            self.assertTrue(
                isinstance(out["deployment_ready_defaults"]["warehouse_world"].get("health_override"), dict)
            )


if __name__ == "__main__":
    unittest.main()
