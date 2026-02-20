import unittest

from theory.transfer_bounds import (
    analyze_transfer_report,
    calibrate_lambda_from_analyses,
    compute_transfer_bound,
    estimate_domain_divergence_from_bridge_stats,
    estimate_source_error_from_bridge_stats,
    summarize_prediction_error,
)


class TestTransferBounds(unittest.TestCase):
    def test_estimates_from_bridge_stats(self):
        bridge_stats = [
            {"input_rows": 100, "translated_rows": 80, "source_kind": "success_events"},
            {"input_rows": 50, "translated_rows": 50, "source_kind": "dna_good"},
        ]
        div = estimate_domain_divergence_from_bridge_stats(bridge_stats)
        src = estimate_source_error_from_bridge_stats(bridge_stats)
        self.assertGreaterEqual(div, 0.0)
        self.assertLessEqual(div, 1.0)
        self.assertGreaterEqual(src, 0.0)
        self.assertLessEqual(src, 1.0)

    def test_bound_and_analysis(self):
        rep = {
            "seed": 123,
            "target_verse": "warehouse_world",
            "transfer_agent": {"eval": {"success_rate": 0.25}},
            "transfer_dataset": {
                "bridge_stats": [
                    {"input_rows": 200, "translated_rows": 150, "source_kind": "success_events"},
                    {"input_rows": 100, "translated_rows": 100, "source_kind": "dna_good"},
                ]
            },
        }
        a = analyze_transfer_report(rep, lambda_term=0.1, report_path="mock.json")
        self.assertIn("bound", a)
        self.assertIn("prediction_abs_error", a)
        self.assertGreaterEqual(a["prediction_abs_error"], 0.0)

        b = compute_transfer_bound(source_error=0.1, domain_divergence=0.2, lambda_term=0.1)
        self.assertAlmostEqual(b["predicted_target_error"], 0.4, places=8)

    def test_lambda_calibration_and_summary(self):
        analyses = [
            {
                "empirical_target_error": 0.7,
                "analysis_inputs": {"source_error_estimate": 0.2, "domain_divergence_estimate": 0.3},
                "prediction_abs_error": 0.05,
            },
            {
                "empirical_target_error": 0.9,
                "analysis_inputs": {"source_error_estimate": 0.3, "domain_divergence_estimate": 0.4},
                "prediction_abs_error": 0.08,
            },
        ]
        lam = calibrate_lambda_from_analyses(analyses)
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)

        s = summarize_prediction_error(analyses, tolerance=0.1)
        self.assertEqual(s["n_reports"], 2)
        self.assertTrue(s["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
