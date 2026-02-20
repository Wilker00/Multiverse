import unittest

from tools.validation_stats import compute_rate_stats, compute_validation_stats


class TestValidationStats(unittest.TestCase):
    def test_compute_validation_stats_basic(self):
        stats = compute_validation_stats([1.0, 2.0, 3.0, 4.0], min_detectable_delta=0.5)
        self.assertEqual(stats["current_n"], 4)
        self.assertAlmostEqual(stats["mean"], 2.5, places=6)
        self.assertIn("ci_95", stats)
        self.assertEqual(len(stats["ci_95"]), 2)
        self.assertGreaterEqual(stats["required_n"], 1)

    def test_compute_validation_stats_empty(self):
        stats = compute_validation_stats([])
        self.assertEqual(stats["current_n"], 0)
        self.assertFalse(stats["is_sufficient"])

    def test_compute_rate_stats(self):
        stats = compute_rate_stats([1, 0, 1, 1, 0], min_detectable_delta=0.1)
        self.assertEqual(stats["current_n"], 5)
        self.assertAlmostEqual(stats["rate"], 0.6, places=6)
        self.assertIn("ci_95", stats)


if __name__ == "__main__":
    unittest.main()
