import os
import tempfile
import unittest

from tools.production_readiness_gate import _validate_benchmark, _validate_manifest


class TestProductionReadinessGate(unittest.TestCase):
    def test_manifest_passes_min_requirements(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run_a")
            os.makedirs(run_dir, exist_ok=True)
            manifest = {
                "deployment_ready_defaults": {
                    "line_world": {
                        "picked_run": {
                            "run_id": "run_a",
                            "run_dir": run_dir,
                            "policy": "special_moe",
                            "mean_return": 0.75,
                            "success_rate": 0.9,
                            "episodes": 80,
                        }
                    }
                }
            }
            ok, errs, details = _validate_manifest(
                manifest,
                min_verses=1,
                min_episodes=50,
                min_success_rate=0.6,
                require_run_dirs=True,
            )
            self.assertTrue(ok)
            self.assertEqual(errs, [])
            self.assertEqual(details["count"], 1)

    def test_manifest_fails_low_success(self):
        manifest = {
            "deployment_ready_defaults": {
                "line_world": {
                    "picked_run": {
                        "run_id": "run_a",
                        "run_dir": "runs/run_a",
                        "policy": "special_moe",
                        "mean_return": 0.1,
                        "success_rate": 0.2,
                        "episodes": 80,
                    }
                }
            }
        }
        ok, errs, _ = _validate_manifest(
            manifest,
            min_verses=1,
            min_episodes=50,
            min_success_rate=0.6,
            require_run_dirs=False,
        )
        self.assertFalse(ok)
        self.assertTrue(any("success_rate" in e for e in errs))

    def test_benchmark_fails_safety(self):
        bench = {
            "overall_pass": True,
            "created_at_iso": "2026-02-09T20:31:36Z",
            "by_verse": {
                "cliff_world": {
                    "passed": True,
                    "candidate": {
                        "success_rate": 0.9,
                        "safety_violation_rate": 0.5,
                    },
                }
            },
        }
        ok, errs, _ = _validate_benchmark(
            bench,
            max_age_hours=100000,
            min_success_rate=0.6,
            max_safety_violation_rate=0.2,
        )
        self.assertFalse(ok)
        self.assertTrue(any("safety_violation_rate" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
