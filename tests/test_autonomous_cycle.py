import argparse
import unittest

from tools.run_autonomous_cycle import (
    _cycle_summary,
    build_benchmark_cmd,
    build_health_cmd,
)


class TestAutonomousCycle(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            runs_root="runs",
            target_verse="warehouse_world",
            episodes=40,
            max_steps=80,
            transfer_algo="q",
            baseline_algo="q",
            seeds="123,223",
            health_trace_root="models/expert_datasets",
            benchmark_report_dir="models/benchmarks/fixed_seed",
            challenge_arg=["--safe_transfer"],
            manifest_path="models/default_policy_set.json",
            central_memory_dir="central_memory",
            health_latest_runs=12,
            health_limit=20,
            auto_heal=True,
            auto_heal_max_agents=2,
        )

    def test_build_commands_include_expected_tokens(self):
        args = self._args()
        bcmd = build_benchmark_cmd(py="python", args=args, out_json="out/bench.json")
        hcmd = build_health_cmd(py="python", args=args, out_json="out/health.json")
        self.assertIn("run_fixed_seed_benchmark.py", " ".join(bcmd))
        self.assertIn("--safe_transfer", bcmd)
        self.assertIn("agent_health_monitor.py", " ".join(hcmd))
        self.assertIn("--auto_heal", hcmd)

    def test_cycle_summary(self):
        bench = {"aggregate": {"win_rate": 0.66, "mean_hazard_improvement_pct": 12.0}}
        health = {"rows": [{"status": "healthy"}, {"status": "critical"}, {"status": "watch"}]}
        s = _cycle_summary(bench, health)
        self.assertAlmostEqual(float(s.get("bench_win_rate", 0.0)), 0.66, places=6)
        self.assertEqual(int(s.get("health_critical_count", 0)), 1)


if __name__ == "__main__":
    unittest.main()
