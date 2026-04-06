import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout

from core.artifact_index import register_artifact
from core.run_artifacts import write_run_artifact_manifest
from memory.embeddings import obs_to_vector
from tools.multiverse_cli import (
    _normalize_remainder,
    apply_distributed_profile,
    apply_train_profile,
    build_doctor_report,
    build_operator_status_report,
    build_parser,
    build_promotion_sentinel_cmd,
    build_sim_control_cmd,
    build_sim2real_cmd,
    build_train_agent_cmd,
    build_train_distributed_cmd,
    discover_runs,
    execute_argv,
    resolve_run_dir,
)
from tools.multiverse_cli_runs import run_snapshot
from tools.multiverse_cli_shell import InteractiveShell


class TestMultiverseCli(unittest.TestCase):
    def test_normalize_remainder_drops_separator(self):
        self.assertEqual(_normalize_remainder(["--", "--foo", "bar"]), ["--foo", "bar"])
        self.assertEqual(_normalize_remainder(["--foo"]), ["--foo"])
        self.assertEqual(_normalize_remainder(None), [])

    def test_build_train_cmd_includes_core_flags_and_passthrough(self):
        ap = build_parser()
        args = ap.parse_args(
            [
                "train",
                "--universe",
                "line_world",
                "--algo",
                "q",
                "--episodes",
                "7",
                "--max-steps",
                "11",
                "--seed",
                "9",
                "--runs-root",
                "runs_x",
                "--",
                "--aconfig",
                "epsilon=0.05",
            ]
        )
        cmd = build_train_agent_cmd(args)
        self.assertIn("tools", cmd[1].replace("\\", "/"))
        self.assertIn("train_agent.py", cmd[1].replace("\\", "/"))
        self.assertIn("--train", cmd)
        self.assertIn("--verse", cmd)
        self.assertIn("line_world", cmd)
        self.assertIn("--aconfig", cmd)
        self.assertIn("epsilon=0.05", cmd)

    def test_build_distributed_cmd_defaults_to_train(self):
        ap = build_parser()
        args = ap.parse_args(["distributed"])
        cmd = build_train_distributed_cmd(args)
        self.assertIn("train_distributed.py", cmd[1].replace("\\", "/"))
        self.assertIn("--train", cmd)
        self.assertIn("--workers", cmd)

    def test_build_sentinel_cmd_includes_core_flags(self):
        ap = build_parser()
        args = ap.parse_args(
            [
                "sentinel",
                "--cycles",
                "2",
                "--require-benchmark",
                "--deploy-on-pass",
                "--deploy-verse",
                "line_world",
            ]
        )
        cmd = build_promotion_sentinel_cmd(args)
        self.assertIn("promotion_sentinel.py", cmd[1].replace("\\", "/"))
        self.assertIn("--require_benchmark", cmd)
        self.assertIn("--deploy_on_pass", cmd)
        self.assertIn("--deploy_verse", cmd)
        self.assertIn("line_world", cmd)

    def test_build_sim2real_cmd_includes_core_flags(self):
        ap = build_parser()
        args = ap.parse_args(
            [
                "sim2real",
                "--universe",
                "warehouse_world",
                "--algo",
                "gateway",
                "--profiles",
                "mild,severe",
                "--json",
            ]
        )
        cmd = build_sim2real_cmd(args)
        self.assertIn("multiverse_sim.py", cmd[1].replace("\\", "/"))
        self.assertEqual(cmd[2], "sim2real")
        self.assertIn("--profiles", cmd)
        self.assertIn("mild,severe", cmd)
        self.assertIn("--json", cmd)

    def test_build_sim_control_cmd_includes_preview_flags(self):
        ap = build_parser()
        args = ap.parse_args(
            [
                "sim",
                "preview",
                "--provider",
                "multiverse_local",
                "--universe",
                "line_world",
                "--episodes",
                "2",
                "--show-final-frame",
            ]
        )
        cmd = build_sim_control_cmd(args)
        self.assertIn("multiverse_sim.py", cmd[1].replace("\\", "/"))
        self.assertIn("preview", cmd)
        self.assertIn("--provider", cmd)
        self.assertIn("multiverse_local", cmd)
        self.assertIn("--show-final-frame", cmd)

    def test_train_profile_applies_defaults(self):
        ap = build_parser()
        raw = ["train", "--profile", "research"]
        args = ap.parse_args(raw)
        apply_train_profile(args, raw)
        self.assertEqual(args.verse, "warehouse_world")
        self.assertEqual(args.algo, "q")
        self.assertEqual(args.episodes, 200)
        self.assertEqual(args.max_steps, 100)
        self.assertTrue(bool(args.eval))

    def test_train_profile_respects_explicit_flags(self):
        ap = build_parser()
        raw = ["train", "--profile", "research", "--episodes", "17", "--universe", "line_world"]
        args = ap.parse_args(raw)
        apply_train_profile(args, raw)
        self.assertEqual(args.episodes, 17)
        self.assertEqual(args.verse, "line_world")
        self.assertEqual(args.algo, "q")

    def test_distributed_profile_applies_defaults(self):
        ap = build_parser()
        raw = ["distributed", "--profile", "research"]
        args = ap.parse_args(raw)
        apply_distributed_profile(args, raw)
        self.assertEqual(args.mode, "pbt")
        self.assertEqual(args.verse, "warehouse_world")
        self.assertEqual(args.workers, 6)

    def test_alias_parsing(self):
        ap = build_parser()
        args = ap.parse_args(["u", "ls", "--contains", "line"])
        self.assertEqual(args.command, "u")
        self.assertEqual(args.universe_command, "ls")

        args2 = ap.parse_args(["st"])
        self.assertEqual(args2.command, "st")
        self.assertFalse(bool(args2.verbose))

        args_doctor = ap.parse_args(["check"])
        self.assertEqual(args_doctor.command, "check")

        args_sim_list = ap.parse_args(["sims", "ls"])
        self.assertEqual(args_sim_list.command, "sims")
        self.assertEqual(args_sim_list.sim_command, "ls")

        args_sim = ap.parse_args(["s2r", "--dry-run"])
        self.assertEqual(args_sim.command, "s2r")

        args3 = ap.parse_args(["shell", "--runs-root", "runs"])
        self.assertEqual(args3.command, "shell")

    def test_build_doctor_report_surfaces_core_readiness(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            mem_root = os.path.join(td, "central_memory")
            os.makedirs(runs_root, exist_ok=True)
            os.makedirs(mem_root, exist_ok=True)
            artifact_index = os.path.join(td, "artifact_index.json")
            run_dir = os.path.join(runs_root, "run_a")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"episode_id\":\"1\"}\n")
            with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"loss\":0.1}\n")
            write_run_artifact_manifest(run_dir, verse_name="line_world", policy_id="random", algo="random")
            with open(os.path.join(mem_root, "memories.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"run_id\":\"r1\"}\n")
            register_artifact(
                artifact_type="agent_health_report",
                artifact_path=os.path.join(td, "health.json"),
                status="ok",
                index_path=artifact_index,
            )

            report = build_doctor_report(
                runs_root=runs_root,
                central_memory_dir=mem_root,
                artifact_index_path=artifact_index,
            )
            self.assertIn("readiness", report)
            self.assertIn("tools", report)
            self.assertIn("next_actions", report)
            self.assertTrue(bool(report["tools"]["train_agent"]["exists"]))
            self.assertTrue(bool(report["readiness"]["core_tools_ready"]))
            self.assertTrue(bool(report["readiness"]["memory_bank_present"]))
            self.assertTrue(bool(report["readiness"]["latest_run_manifest_present"]))
            self.assertTrue(bool(report["readiness"]["latest_run_artifacts_ok"]))
            self.assertTrue(bool(report["readiness"]["artifact_index_present"]))
            self.assertTrue(bool(report["readiness"]["health_report_indexed"]))
            self.assertIn("latest_run_artifacts", report["artifacts"])

    def test_build_operator_status_report_surfaces_latest_sentinel_decision(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            mem_root = os.path.join(td, "central_memory")
            sentinel_root = os.path.join(td, "sentinel")
            artifact_index = os.path.join(td, "artifact_index.json")
            os.makedirs(runs_root, exist_ok=True)
            os.makedirs(mem_root, exist_ok=True)
            os.makedirs(os.path.join(sentinel_root, "cycle_001"), exist_ok=True)
            with open(os.path.join(mem_root, "memories.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"run_id\":\"r1\"}\n")
            summary_path = os.path.join(sentinel_root, "cycle_001", "promotion_sentinel_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(
                    """{
  "created_at_iso": "2026-03-31T00:00:00Z",
  "cycles": 1,
  "cycle_rows": [
    {
      "cycle": 1,
      "decision": {
        "readiness_ok": true,
        "health_ok": false,
        "deploy_allowed": false,
        "block_reasons": ["critical_agents_exceeded"],
        "health": {
          "agents_scored": 3,
          "critical_count": 1,
          "unhealthy_count": 0
        }
      },
      "deploy": {
        "attempted": false,
        "returncode": null
      }
    }
  ]
}"""
                )

            report = build_operator_status_report(
                runs_root=runs_root,
                central_memory_dir=mem_root,
                sentinel_out_dir=sentinel_root,
                artifact_index_path=artifact_index,
            )
            self.assertIn("latest_run", report)
            self.assertIn("indexed_artifacts", report)
            self.assertIn("sentinel", report)
            self.assertTrue(bool(report["readiness"]["sentinel_summary_present"]))
            self.assertEqual(report["sentinel"]["latest_cycle"], 1)
            self.assertTrue(bool(report["sentinel"]["readiness_ok"]))
            self.assertFalse(bool(report["sentinel"]["health_ok"]))
            self.assertEqual(report["sentinel"]["critical_agents"], 1)
            self.assertIn("critical_agents_exceeded", report["sentinel"]["block_reasons"])

    def test_build_operator_status_report_can_use_indexed_sentinel_summary(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            mem_root = os.path.join(td, "central_memory")
            sentinel_root = os.path.join(td, "sentinel")
            artifact_index = os.path.join(td, "artifact_index.json")
            os.makedirs(runs_root, exist_ok=True)
            os.makedirs(mem_root, exist_ok=True)
            os.makedirs(sentinel_root, exist_ok=True)
            with open(os.path.join(mem_root, "memories.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"run_id\":\"r1\"}\n")
            summary_path = os.path.join(td, "promotion_sentinel_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(
                    """{
  "created_at_iso": "2026-03-31T00:00:00Z",
  "cycles": 1,
  "cycle_rows": [
    {
      "cycle": 1,
      "decision": {
        "readiness_ok": true,
        "health_ok": true,
        "deploy_allowed": true,
        "block_reasons": [],
        "health": {
          "agents_scored": 1,
          "critical_count": 0,
          "unhealthy_count": 0
        }
      },
      "deploy": {
        "attempted": false,
        "returncode": null
      }
    }
  ]
}"""
                )
            register_artifact(
                artifact_type="promotion_sentinel_summary",
                artifact_path=summary_path,
                status="passed",
                index_path=artifact_index,
            )

            report = build_operator_status_report(
                runs_root=runs_root,
                central_memory_dir=mem_root,
                sentinel_out_dir=sentinel_root,
                artifact_index_path=artifact_index,
            )
            self.assertTrue(bool(report["sentinel"]["found"]))
            self.assertEqual(report["sentinel"]["summary_path"], summary_path)
            self.assertTrue(bool(report["sentinel"]["deploy_allowed"]))

    def test_execute_argv_doctor_json(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = execute_argv(["doctor", "--json"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("\"product\"", text)
        self.assertIn("\"readiness\"", text)
        self.assertIn("\"next_actions\"", text)

    def test_execute_argv_status_verbose_json(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            mem_root = os.path.join(td, "central_memory")
            sentinel_root = os.path.join(td, "sentinel")
            artifact_index = os.path.join(td, "artifact_index.json")
            os.makedirs(runs_root, exist_ok=True)
            os.makedirs(mem_root, exist_ok=True)
            os.makedirs(sentinel_root, exist_ok=True)
            with open(os.path.join(mem_root, "memories.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"run_id\":\"r1\"}\n")
            with open(os.path.join(sentinel_root, "promotion_sentinel_summary.json"), "w", encoding="utf-8") as f:
                f.write(
                    """{
  "created_at_iso": "2026-03-31T00:00:00Z",
  "cycles": 1,
  "cycle_rows": [
    {
      "cycle": 1,
      "decision": {
        "readiness_ok": false,
        "health_ok": true,
        "deploy_allowed": false,
        "block_reasons": ["readiness_failed"],
        "health": {
          "agents_scored": 2,
          "critical_count": 0,
          "unhealthy_count": 0
        }
      },
      "deploy": {
        "attempted": false,
        "returncode": null
      }
    }
  ]
}"""
                )

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = execute_argv(
                    [
                        "status",
                        "--json",
                        "--verbose",
                        "--runs-root",
                        runs_root,
                        "--central-memory-dir",
                        mem_root,
                        "--sentinel-out-dir",
                        sentinel_root,
                        "--artifact-index-path",
                        artifact_index,
                    ]
                )
            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("\"operator\"", text)
            self.assertIn("\"latest_run\"", text)
            self.assertIn("\"sentinel\"", text)
            self.assertIn("\"readiness_failed\"", text)

    def test_execute_argv_train_profile_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = execute_argv(["train", "--profile", "quick", "--dry-run"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("train_agent.py", text.replace("\\", "/"))
        self.assertIn("--episodes 20", text)
        self.assertIn("--verse line_world", text)

    def test_execute_argv_sentinel_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = execute_argv(["sentinel", "--require-benchmark", "--artifact-index-path", "artifacts/index.json", "--dry-run"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("promotion_sentinel.py", text.replace("\\", "/"))
        self.assertIn("--require_benchmark", text)
        self.assertIn("--artifact_index_path", text)

    def test_execute_argv_sentinel_status_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = execute_argv(["sentinel", "--status", "--json", "--artifact-index-path", "artifacts/index.json", "--dry-run"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("promotion_sentinel.py", text.replace("\\", "/"))
        self.assertIn("--status", text)
        self.assertIn("--json", text)
        self.assertIn("--artifact_index_path", text)

    def test_execute_argv_sim2real_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = execute_argv(["sim2real", "--profiles", "mild", "--dry-run"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("multiverse_sim.py", text.replace("\\", "/"))
        self.assertIn(" sim2real ", text)
        self.assertIn("--profiles mild", text)

    def test_execute_argv_sim_preview_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = execute_argv(["sim", "preview", "--provider", "multiverse_local", "--dry-run"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("multiverse_sim.py", text.replace("\\", "/"))
        self.assertIn("preview", text)
        self.assertIn("multiverse_local", text)

    def test_execute_argv_memory_inspect_bootstrap_json(self):
        with tempfile.TemporaryDirectory() as td:
            mem_root = os.path.join(td, "central_memory")
            os.makedirs(mem_root, exist_ok=True)
            obs0 = {"pos": 0, "goal": 4, "t": 0}
            row = {
                "run_id": "cli_bootstrap_source",
                "episode_id": "ep_bootstrap",
                "step_idx": 0,
                "t_ms": 1,
                "verse_name": "line_world",
                "obs": obs0,
                "obs_vector": obs_to_vector(obs0),
                "action": 1,
                "reward": 1.0,
                "memory_tier": "ltm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
            with open(os.path.join(mem_root, "memories.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = execute_argv(
                    [
                        "memory",
                        "inspect",
                        "--algo",
                        "memory_recall",
                        "--universe",
                        "line_world",
                        "--central-memory-dir",
                        mem_root,
                        "--obs-json",
                        json.dumps(obs0),
                        "--aconfig",
                        "bootstrap_recall_enabled=true",
                        "--aconfig",
                        "bootstrap_top_k=1",
                        "--aconfig",
                        "bootstrap_min_score=-1.0",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "resolved")
            self.assertEqual(payload["mode"], "bootstrap")
            self.assertEqual(payload["request"]["reason"], "episode_bootstrap")
            self.assertEqual(int(payload["match_count"]), 1)
            self.assertIn(
                "runs/cli_bootstrap_source/events.jsonl",
                str((payload["bundle"]["matches"][0] or {}).get("pointer_path", "")),
            )

    def test_execute_argv_memory_inspect_on_demand_json(self):
        with tempfile.TemporaryDirectory() as td:
            mem_root = os.path.join(td, "central_memory")
            os.makedirs(mem_root, exist_ok=True)
            obs0 = {"risk": 9, "pos": 0, "goal": 4, "t": 0}
            row = {
                "run_id": "cli_on_demand_source",
                "episode_id": "ep_on_demand",
                "step_idx": 0,
                "t_ms": 1,
                "verse_name": "line_world",
                "obs": obs0,
                "obs_vector": obs_to_vector(obs0),
                "action": 1,
                "reward": 1.0,
                "memory_tier": "ltm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
            with open(os.path.join(mem_root, "memories.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = execute_argv(
                    [
                        "memory",
                        "inspect",
                        "--algo",
                        "memory_recall",
                        "--mode",
                        "on_demand",
                        "--universe",
                        "line_world",
                        "--central-memory-dir",
                        mem_root,
                        "--obs-json",
                        json.dumps(obs0),
                        "--aconfig",
                        "recall_risk_threshold=1.0",
                        "--aconfig",
                        "recall_top_k=1",
                        "--aconfig",
                        "recall_min_score=-1.0",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "resolved")
            self.assertEqual(payload["mode"], "on_demand")
            self.assertEqual(payload["request"]["reason"], "high_risk")
            self.assertEqual(int(payload["match_count"]), 1)
            self.assertIn(
                "runs/cli_on_demand_source/events.jsonl",
                str((payload["bundle"]["matches"][0] or {}).get("pointer_path", "")),
            )

    def test_shell_autocomplete_and_theme_controls(self):
        sh = InteractiveShell(
            runs_root="runs",
            build_parser_fn=build_parser,
            execute_argv_fn=execute_argv,
            run_snapshot_fn=run_snapshot,
        )
        self.assertEqual(sh.theme, "dark")
        sh.input_buf = "st"
        sh._autocomplete()
        self.assertTrue(sh.input_buf.startswith("st"))

        sh._run_command(":layout full")
        self.assertEqual(sh.layout, "full")
        sh._run_command(":theme dark")
        self.assertEqual(sh.theme, "dark")
        sh._run_command(":theme matrix")
        self.assertEqual(sh.theme, "matrix")
        sh._run_command(":intensity 3")
        self.assertEqual(sh.intensity, 3)

    def test_shell_suggestion_pages_exist(self):
        sh = InteractiveShell(
            runs_root="runs",
            build_parser_fn=build_parser,
            execute_argv_fn=execute_argv,
            run_snapshot_fn=run_snapshot,
        )
        pages = sh._suggestion_pages()
        self.assertGreaterEqual(len(pages), 2)
        self.assertIn("title", pages[0])
        self.assertIn("items", pages[0])
        flat = " ".join(" ".join(page.get("items", [])) for page in pages)
        self.assertIn("doctor", flat)
        self.assertIn("sim list", flat)
        self.assertIn("sim2real", flat)

    def test_discover_and_resolve_runs(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            os.makedirs(runs_root, exist_ok=True)

            run_a = os.path.join(runs_root, "run_a")
            run_b = os.path.join(runs_root, "run_b")
            os.makedirs(run_a, exist_ok=True)
            os.makedirs(run_b, exist_ok=True)
            with open(os.path.join(run_a, "events.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"a\":1}\n")
            with open(os.path.join(run_b, "events.jsonl"), "w", encoding="utf-8") as f:
                f.write("{\"b\":1}\n")

            now = time.time()
            os.utime(os.path.join(run_a, "events.jsonl"), (now - 100, now - 100))
            os.utime(os.path.join(run_b, "events.jsonl"), (now, now))

            rows = discover_runs(runs_root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].run_id, "run_b")

            latest = resolve_run_dir(runs_root, run_id=None)
            self.assertEqual(os.path.basename(str(latest)), "run_b")

            explicit = resolve_run_dir(runs_root, run_id="run_a")
            self.assertEqual(os.path.basename(str(explicit)), "run_a")


if __name__ == "__main__":
    unittest.main()
