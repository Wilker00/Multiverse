import unittest

from tools.run_fixed_seed_benchmark import (
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


if __name__ == "__main__":
    unittest.main()

