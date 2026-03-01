import json
import os
import tempfile
import unittest

from tools.run_fixed_seed_benchmark import (
    _load_existing_seed_row,
    aggregate_seed_metrics,
    extract_seed_metrics,
    parse_seed_list,
)


class TestFixedSeedBenchmark(unittest.TestCase):
    def test_parse_seed_list(self):
        seeds = parse_seed_list("123, 223;337,123")
        self.assertEqual(seeds, [123, 223, 337])

    def test_extract_seed_metrics(self):
        report = {
            "comparison": {
                "transfer_wins_convergence": True,
                "transfer_speedup_ratio": 1.8,
                "hazard_improvement_pct": 22.5,
            },
            "production_summary": {
                "safety": {
                    "transfer_hazard_per_1k": 100.0,
                    "baseline_hazard_per_1k": 200.0,
                    "transfer_mcts_veto_rate": 0.05,
                    "baseline_mcts_veto_rate": 0.12,
                },
                "health": {
                    "transfer": {"total_score": 81.0, "status": "healthy"},
                    "baseline": {"total_score": 52.0, "status": "degraded"},
                },
            },
            "transfer_agent": {"eval": {"mean_return": 2.0, "success_rate": 0.7}},
            "baseline_agent": {"eval": {"mean_return": 1.0, "success_rate": 0.4}},
        }
        m = extract_seed_metrics(report)
        self.assertTrue(bool(m.get("transfer_wins_convergence", False)))
        self.assertAlmostEqual(float(m.get("transfer_speedup_ratio", 0.0) or 0.0), 1.8, places=6)
        self.assertEqual(str(m.get("transfer_health_status", "")), "healthy")
        self.assertAlmostEqual(float(m.get("hazard_gain_per_1k", 0.0)), 100.0, places=6)
        self.assertAlmostEqual(float(m.get("success_delta", 0.0)), 0.3, places=6)
        self.assertAlmostEqual(float(m.get("return_delta", 0.0)), 1.0, places=6)

    def test_aggregate_seed_metrics(self):
        rows = [
            {
                "transfer_wins_convergence": True,
                "transfer_speedup_ratio": 2.0,
                "hazard_improvement_pct": 10.0,
                "transfer_hazard_per_1k": 80.0,
                "baseline_hazard_per_1k": 100.0,
                "transfer_health_score": 80.0,
                "baseline_health_score": 60.0,
                "transfer_health_status": "healthy",
                "baseline_health_status": "degraded",
            },
            {
                "transfer_wins_convergence": False,
                "transfer_speedup_ratio": 1.0,
                "hazard_improvement_pct": -5.0,
                "transfer_hazard_per_1k": 120.0,
                "baseline_hazard_per_1k": 110.0,
                "transfer_health_score": 50.0,
                "baseline_health_score": 55.0,
                "transfer_health_status": "degraded",
                "baseline_health_status": "degraded",
            },
        ]
        agg = aggregate_seed_metrics(rows)
        self.assertEqual(int(agg.get("num_seeds", 0)), 2)
        self.assertAlmostEqual(float(agg.get("win_rate", 0.0)), 0.5, places=6)
        self.assertIn("healthy", dict(agg.get("transfer_status_counts", {})))
        self.assertIn("hazard_gain_per_1k_stats", agg)
        stats = dict(agg.get("hazard_gain_per_1k_stats", {}))
        self.assertIsNotNone(stats.get("mean"))
        self.assertGreaterEqual(float(stats.get("range", 0.0) or 0.0), 0.0)

    def test_load_existing_seed_row(self):
        with tempfile.TemporaryDirectory() as td:
            report_out = os.path.join(td, "transfer_seed_7.json")
            overlap_out = os.path.join(td, "transfer_seed_7_overlap.json")
            report = {
                "comparison": {
                    "transfer_wins_convergence": False,
                    "transfer_speedup_ratio": None,
                    "hazard_improvement_pct": 12.0,
                },
                "production_summary": {
                    "safety": {
                        "transfer_hazard_per_1k": 150.0,
                        "baseline_hazard_per_1k": 210.0,
                    },
                    "health": {
                        "transfer": {"status": "degraded"},
                        "baseline": {"status": "degraded"},
                    },
                },
                "transfer_agent": {"eval": {"mean_return": -5.0, "success_rate": 0.2}},
                "baseline_agent": {"eval": {"mean_return": -9.0, "success_rate": 0.0}},
            }
            with open(report_out, "w", encoding="utf-8") as f:
                json.dump(report, f)
            row = _load_existing_seed_row(seed=7, report_out=report_out, overlap_out=overlap_out)
            self.assertIsNotNone(row)
            assert isinstance(row, dict)
            self.assertTrue(bool(row.get("resumed", False)))
            self.assertEqual(int(row.get("seed", 0)), 7)
            metrics = dict(row.get("metrics", {}))
            self.assertAlmostEqual(float(metrics.get("hazard_gain_per_1k", 0.0)), 60.0, places=6)


if __name__ == "__main__":
    unittest.main()

