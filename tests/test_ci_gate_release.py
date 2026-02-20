import unittest
from argparse import Namespace
from pathlib import Path
import tempfile
from unittest.mock import patch

from tools.ci_gate import _apply_profile_defaults, exception_debt_gate, workspace_hygiene_gate


class TestCiGateRelease(unittest.TestCase):
    def test_fast_profile_sets_lightweight_defaults(self):
        ns = Namespace(
            profile="fast",
            skip_compile=True,
            skip_help=False,
            skip_model=False,
            bootstrap_run=True,
            run_comm_gate=False,
            run_production_readiness=True,
            run_memory_soak=True,
            enforce_workspace_hygiene=True,
        )
        _apply_profile_defaults(ns)
        self.assertFalse(bool(ns.skip_compile))
        self.assertTrue(bool(ns.skip_help))
        self.assertTrue(bool(ns.skip_model))
        self.assertFalse(bool(ns.bootstrap_run))
        self.assertTrue(bool(ns.run_comm_gate))
        self.assertFalse(bool(ns.run_production_readiness))
        self.assertFalse(bool(ns.run_memory_soak))
        self.assertTrue(bool(ns.run_exception_debt_gate))
        self.assertFalse(bool(ns.run_artifact_hygiene))
        self.assertFalse(bool(ns.enforce_workspace_hygiene))

    def test_full_profile_sets_comprehensive_defaults(self):
        ns = Namespace(
            profile="full",
            skip_compile=True,
            skip_help=True,
            skip_model=True,
            bootstrap_run=False,
            run_comm_gate=False,
        )
        _apply_profile_defaults(ns)
        self.assertFalse(bool(ns.skip_compile))
        self.assertFalse(bool(ns.skip_help))
        self.assertFalse(bool(ns.skip_model))
        self.assertTrue(bool(ns.bootstrap_run))
        self.assertTrue(bool(ns.run_comm_gate))
        self.assertTrue(bool(ns.run_exception_debt_gate))
        self.assertTrue(bool(ns.run_artifact_hygiene))

    def test_profile_respects_explicit_cli_flags(self):
        ns = Namespace(
            profile="full",
            skip_compile=False,
            skip_help=False,
            skip_model=True,
            bootstrap_run=False,
            run_comm_gate=False,
            run_exception_debt_gate=False,
            run_artifact_hygiene=False,
        )
        _apply_profile_defaults(ns, cli_args=["--skip-model"])
        self.assertTrue(bool(ns.skip_model))
        self.assertFalse(bool(ns.skip_compile))
        self.assertFalse(bool(ns.skip_help))
        self.assertTrue(bool(ns.bootstrap_run))
        self.assertTrue(bool(ns.run_comm_gate))
        self.assertTrue(bool(ns.run_exception_debt_gate))
        self.assertTrue(bool(ns.run_artifact_hygiene))

    def test_exception_debt_gate_uses_baseline_and_fails_on_regression(self):
        with tempfile.TemporaryDirectory() as td:
            baseline_path = Path(td) / "baseline.json"
            baseline_path.write_text('{"broad_exception_max": 10}', encoding="utf-8")
            with patch("tools.ci_gate._python_files", return_value=[]):
                with patch("tools.ci_gate._count_broad_exception_blocks", return_value=12):
                    with self.assertRaises(RuntimeError):
                        exception_debt_gate(
                            baseline_json=str(baseline_path),
                            max_broad=-1,
                            fail_on_regression=True,
                        )

    def test_workspace_hygiene_allows_new_paths_under_allowlist(self):
        baseline = {"a.txt"}
        with patch("tools.ci_gate._repo_diff_paths", return_value={"a.txt", ".ci_artifacts/out.json"}):
            workspace_hygiene_gate(baseline=baseline, allow_prefixes=[".ci_artifacts"])

    def test_workspace_hygiene_rejects_new_paths_outside_allowlist(self):
        baseline = {"a.txt"}
        with patch("tools.ci_gate._repo_diff_paths", return_value={"a.txt", "runs/run_abc/events.jsonl"}):
            with self.assertRaises(RuntimeError):
                workspace_hygiene_gate(baseline=baseline, allow_prefixes=[".ci_artifacts"])


if __name__ == "__main__":
    unittest.main()
