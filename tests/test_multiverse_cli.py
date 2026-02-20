import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout

from tools.multiverse_cli import (
    _normalize_remainder,
    InteractiveShell,
    apply_distributed_profile,
    apply_train_profile,
    build_parser,
    build_train_agent_cmd,
    build_train_distributed_cmd,
    discover_runs,
    execute_argv,
    resolve_run_dir,
)


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

        args3 = ap.parse_args(["shell", "--runs-root", "runs"])
        self.assertEqual(args3.command, "shell")

    def test_execute_argv_train_profile_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = execute_argv(["train", "--profile", "quick", "--dry-run"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("train_agent.py", text.replace("\\", "/"))
        self.assertIn("--episodes 20", text)
        self.assertIn("--verse line_world", text)

    def test_shell_autocomplete_and_theme_controls(self):
        sh = InteractiveShell(runs_root="runs")
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
        sh = InteractiveShell(runs_root="runs")
        pages = sh._suggestion_pages()
        self.assertGreaterEqual(len(pages), 2)
        self.assertIn("title", pages[0])
        self.assertIn("items", pages[0])

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
