import json
import os
import tempfile
import unittest

from tools.mcts_telemetry_dashboard import aggregate_by_verse, analyze_events_file, load_rows


def _write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class TestMCTSTelemetryDashboard(unittest.TestCase):
    def test_analyze_events_file_tracks_mcts_deltas(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run_a")
            os.makedirs(run_dir, exist_ok=True)
            events_path = os.path.join(run_dir, "events.jsonl")
            rows = [
                {
                    "episode_id": "ep_1",
                    "step_idx": 0,
                    "verse_name": "chess_world",
                    "policy_id": "special_moe",
                    "info": {"safe_executor": {"mcts_stats": {"enabled": True, "queries": 0, "vetoes": 0}}},
                },
                {
                    "episode_id": "ep_1",
                    "step_idx": 1,
                    "verse_name": "chess_world",
                    "policy_id": "special_moe",
                    "info": {
                        "safe_executor": {
                            "mcts_stats": {
                                "enabled": True,
                                "queries": 1,
                                "vetoes": 0,
                                "last_query": {
                                    "root_value": -0.2,
                                    "avg_leaf_value": -0.1,
                                    "simulations": 64,
                                    "forced_loss_detected": True,
                                    "principal_variation": [0, 1, 2],
                                },
                            }
                        }
                    },
                },
                {
                    "episode_id": "ep_1",
                    "step_idx": 2,
                    "verse_name": "chess_world",
                    "policy_id": "special_moe",
                    "info": {
                        "safe_executor": {
                            "mcts_stats": {
                                "enabled": True,
                                "queries": 2,
                                "vetoes": 1,
                                "last_query": {
                                    "root_value": 0.4,
                                    "avg_leaf_value": 0.2,
                                    "simulations": 64,
                                    "forced_loss_detected": False,
                                    "principal_variation": [1, 0],
                                },
                            }
                        }
                    },
                },
                {
                    "episode_id": "ep_2",
                    "step_idx": 0,
                    "verse_name": "chess_world",
                    "policy_id": "special_moe",
                    "info": {"safe_executor": {"mcts_stats": {"enabled": False, "queries": 0, "vetoes": 0}}},
                },
            ]
            _write_jsonl(events_path, rows)

            out = analyze_events_file(events_path)
            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(out.verse_name, "chess_world")
            self.assertEqual(out.policy_id, "special_moe")
            self.assertEqual(out.episodes, 2)
            self.assertEqual(out.mcts_enabled_episodes, 1)
            self.assertEqual(out.query_events, 2)
            self.assertEqual(out.veto_events, 1)
            self.assertEqual(out.forced_loss_queries, 1)
            self.assertAlmostEqual(out.avg_root_value, 0.1, places=6)
            self.assertAlmostEqual(out.avg_leaf_value, 0.05, places=6)
            self.assertAlmostEqual(out.avg_simulations, 64.0, places=6)
            self.assertAlmostEqual(out.avg_pv_len, 2.5, places=6)

    def test_load_rows_and_aggregate_by_verse(self):
        with tempfile.TemporaryDirectory() as td:
            run_a = os.path.join(td, "run_a")
            run_b = os.path.join(td, "run_b")
            os.makedirs(run_a, exist_ok=True)
            os.makedirs(run_b, exist_ok=True)
            _write_jsonl(
                os.path.join(run_a, "events.jsonl"),
                [
                    {
                        "episode_id": "ep_a",
                        "verse_name": "go_world",
                        "policy_id": "gateway",
                        "info": {
                            "safe_executor": {
                                "mcts_stats": {
                                    "enabled": True,
                                    "queries": 1,
                                    "vetoes": 0,
                                    "last_query": {
                                        "root_value": 0.2,
                                        "avg_leaf_value": 0.1,
                                        "simulations": 48,
                                        "forced_loss_detected": False,
                                        "principal_variation": [1],
                                    },
                                }
                            }
                        },
                    }
                ],
            )
            _write_jsonl(
                os.path.join(run_b, "events.jsonl"),
                [
                    {
                        "episode_id": "ep_b",
                        "verse_name": "go_world",
                        "policy_id": "gateway",
                        "info": {
                            "safe_executor": {
                                "mcts_stats": {
                                    "enabled": True,
                                    "queries": 1,
                                    "vetoes": 1,
                                    "last_query": {
                                        "root_value": -0.4,
                                        "avg_leaf_value": -0.3,
                                        "simulations": 96,
                                        "forced_loss_detected": True,
                                        "principal_variation": [0, 2],
                                    },
                                }
                            }
                        },
                    }
                ],
            )

            rows = load_rows(runs_root=td, run_dir="", events_path="")
            self.assertEqual(len(rows), 2)
            by_verse = aggregate_by_verse(rows)
            self.assertEqual(len(by_verse), 1)
            row = by_verse[0]
            self.assertEqual(row["verse_name"], "go_world")
            self.assertEqual(int(row["runs"]), 2)
            self.assertEqual(int(row["query_events"]), 2)
            self.assertEqual(int(row["veto_events"]), 1)
            self.assertEqual(int(row["forced_loss_queries"]), 1)
            self.assertAlmostEqual(float(row["avg_root_value"]), -0.1, places=6)
            self.assertAlmostEqual(float(row["avg_leaf_value"]), -0.1, places=6)
            self.assertAlmostEqual(float(row["avg_simulations"]), 72.0, places=6)


if __name__ == "__main__":
    unittest.main()
