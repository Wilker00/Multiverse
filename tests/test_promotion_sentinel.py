import argparse
import os
import unittest

from tools.promotion_sentinel import (
    _decide_cycle,
    _health_summary,
    _render_status,
    _readiness_summary,
    _summary_path_from_args,
    build_deploy_cmd,
    build_health_cmd,
    build_readiness_cmd,
)


class TestPromotionSentinel(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            runs_root="runs",
            manifest_path=os.path.join("models", "default_policy_set.json"),
            central_memory_dir="central_memory",
            health_trace_root=os.path.join("models", "expert_datasets"),
            health_latest_runs=12,
            health_limit=20,
            auto_heal=True,
            auto_heal_max_agents=2,
            bench_json=os.path.join("models", "benchmarks", "latest.json"),
            require_benchmark=True,
            require_run_dirs=False,
            min_verses=2,
            min_episodes=50,
            min_success_rate=0.6,
            max_bench_age_hours=72.0,
            max_safety_violation_rate=0.2,
            max_critical_agents=0,
            max_unhealthy_agents=1,
            deploy_on_pass=True,
            deploy_verse="line_world",
            deploy_episodes=40,
            deploy_seed=123,
            deploy_skip_eval_gate=False,
            deploy_skip_promotion_board=True,
            deploy_ingest_memory=True,
            out_dir=os.path.join("models", "tuning", "promotion_sentinel"),
            cycles=1,
            sleep_seconds=0.0,
        )

    def test_build_health_cmd(self):
        args = self._args()
        cmd = build_health_cmd(py="python", args=args, out_json="health.json")
        self.assertIn("agent_health_monitor.py", cmd[1].replace("\\", "/"))
        self.assertIn("--auto_heal", cmd)
        self.assertIn("--out_json", cmd)
        self.assertIn("health.json", cmd)

    def test_build_readiness_cmd(self):
        args = self._args()
        cmd = build_readiness_cmd(py="python", args=args, out_json="readiness.json")
        self.assertIn("production_readiness_gate.py", cmd[1].replace("\\", "/"))
        self.assertIn("--require_benchmark", cmd)
        self.assertIn("--min_success_rate", cmd)
        self.assertIn("readiness.json", cmd)

    def test_build_deploy_cmd(self):
        args = self._args()
        cmd = build_deploy_cmd(py="python", args=args)
        self.assertIn("deploy_agent.py", cmd[1].replace("\\", "/"))
        self.assertIn("--verse", cmd)
        self.assertIn("line_world", cmd)
        self.assertIn("--skip_promotion_board", cmd)
        self.assertIn("--ingest_memory", cmd)

    def test_health_summary_counts_statuses(self):
        out = _health_summary(
            {
                "rows": [
                    {"status": "healthy"},
                    {"status": "critical"},
                    {"status": "unhealthy"},
                    {"status": "healthy"},
                ]
            }
        )
        self.assertEqual(int(out["agents_scored"]), 4)
        self.assertEqual(int(out["healthy_count"]), 2)
        self.assertEqual(int(out["critical_count"]), 1)
        self.assertEqual(int(out["unhealthy_count"]), 1)

    def test_readiness_summary_extracts_pass_state(self):
        out = _readiness_summary(
            {
                "passed": True,
                "errors": [],
                "checks": {
                    "manifest": {"passed": True},
                    "benchmark": {"passed": False},
                },
            }
        )
        self.assertTrue(bool(out["passed"]))
        self.assertTrue(bool(out["manifest_passed"]))
        self.assertFalse(bool(out["benchmark_passed"]))
        self.assertEqual(int(out["error_count"]), 0)

    def test_decide_cycle_blocks_on_health_or_readiness_failure(self):
        decision = _decide_cycle(
            health={"rows": [{"status": "critical"}]},
            readiness={"passed": False, "errors": ["x"], "checks": {}},
            max_critical_agents=0,
            max_unhealthy_agents=0,
        )
        self.assertFalse(bool(decision["deploy_allowed"]))
        self.assertIn("readiness_failed", list(decision["block_reasons"]))
        self.assertIn("critical_agents_exceeded", list(decision["block_reasons"]))

    def test_decide_cycle_allows_deploy_when_checks_pass(self):
        decision = _decide_cycle(
            health={"rows": [{"status": "healthy"}]},
            readiness={"passed": True, "errors": [], "checks": {"manifest": {"passed": True}, "benchmark": {"passed": True}}},
            max_critical_agents=0,
            max_unhealthy_agents=0,
        )
        self.assertTrue(bool(decision["health_ok"]))
        self.assertTrue(bool(decision["readiness_ok"]))
        self.assertTrue(bool(decision["deploy_allowed"]))

    def test_summary_path_defaults_to_out_dir(self):
        args = self._args()
        args.summary_json = ""
        path = _summary_path_from_args(args)
        self.assertIn("promotion_sentinel_summary.json", path.replace("\\", "/"))

    def test_render_status_includes_latest_cycle_fields(self):
        text = _render_status(
            {
                "created_at_iso": "2026-03-11T00:00:00Z",
                "cycles": 2,
                "cycle_rows": [
                    {
                        "cycle": 2,
                        "decision": {
                            "readiness_ok": True,
                            "health_ok": False,
                            "deploy_allowed": False,
                            "block_reasons": ["critical_agents_exceeded"],
                            "health": {"critical_count": 1, "unhealthy_count": 0},
                        },
                        "deploy": {"attempted": False, "returncode": None},
                    }
                ],
            }
        )
        self.assertIn("Promotion Sentinel", text)
        self.assertIn("Latest cycle   : 2", text)
        self.assertIn("Deploy allowed : False", text)
        self.assertIn("critical_agents_exceeded", text)


if __name__ == "__main__":
    unittest.main()
