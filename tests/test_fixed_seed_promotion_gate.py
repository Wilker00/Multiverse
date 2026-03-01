import unittest

from tools.run_fixed_seed_benchmark import compute_transfer_promotion_gate


class TestFixedSeedPromotionGate(unittest.TestCase):
    def test_promotion_gate_passes_on_consistent_positive_deltas(self):
        rows = [
            {"metrics": {"transfer_success_rate": 0.5, "baseline_success_rate": 0.3, "transfer_hazard_per_1k": 120.0, "baseline_hazard_per_1k": 180.0, "transfer_mean_return": -5.0, "baseline_mean_return": -9.0}},
            {"metrics": {"transfer_success_rate": 0.6, "baseline_success_rate": 0.4, "transfer_hazard_per_1k": 100.0, "baseline_hazard_per_1k": 170.0, "transfer_mean_return": -4.0, "baseline_mean_return": -8.0}},
            {"metrics": {"transfer_success_rate": 0.55, "baseline_success_rate": 0.35, "transfer_hazard_per_1k": 110.0, "baseline_hazard_per_1k": 175.0, "transfer_mean_return": -4.5, "baseline_mean_return": -8.5}},
        ]
        gate = compute_transfer_promotion_gate(
            seed_rows=rows,
            n_boot=1000,
            seed=7,
            min_success_delta=0.0,
            min_hazard_gain_per_1k=0.0,
            min_return_delta=0.0,
        )
        self.assertTrue(bool(gate.get("ok", False)))

    def test_promotion_gate_fails_when_hazard_regresses(self):
        rows = [
            {"metrics": {"transfer_success_rate": 0.6, "baseline_success_rate": 0.4, "transfer_hazard_per_1k": 250.0, "baseline_hazard_per_1k": 180.0, "transfer_mean_return": -4.0, "baseline_mean_return": -8.0}},
            {"metrics": {"transfer_success_rate": 0.65, "baseline_success_rate": 0.45, "transfer_hazard_per_1k": 260.0, "baseline_hazard_per_1k": 170.0, "transfer_mean_return": -3.0, "baseline_mean_return": -7.5}},
            {"metrics": {"transfer_success_rate": 0.62, "baseline_success_rate": 0.42, "transfer_hazard_per_1k": 255.0, "baseline_hazard_per_1k": 175.0, "transfer_mean_return": -3.5, "baseline_mean_return": -7.8}},
        ]
        gate = compute_transfer_promotion_gate(
            seed_rows=rows,
            n_boot=1000,
            seed=7,
            min_success_delta=0.0,
            min_hazard_gain_per_1k=0.0,
            min_return_delta=0.0,
        )
        self.assertFalse(bool(gate.get("ok", True)))
        checks = gate.get("checks", {})
        self.assertFalse(bool(checks.get("hazard_gain_ci95_low_ge_threshold", True)))


if __name__ == "__main__":
    unittest.main()

