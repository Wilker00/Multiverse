"""Platform-layer checks: live snapshot, concept retrieval, transfer log, embodied verse."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from core.concept_taxonomy import concept_tags_for_verse
from core.live_state import build_live_snapshot, write_lifecycle_event
from core.types import AgentSpec, VerseSpec
from integrations.physics.warp_backend import WarpPhysicsBackend, WarpUnavailableError, warp_backend_available
from integrations.sim_registry import provider_status
from memory.central_repository import CentralMemoryConfig, find_similar, ingest_run
from memory.skill_catalog import list_skills
from memory.transfer_log import append_transfer_records
from memory.transfer_metrics import summarize_transfer_root
from orchestrator.trainer import Trainer
from tools.multiverse_cli import build_parser
from tools.multiverse_live_web import render_page
from tools.universe_hub import UniverseHub
from verses.registry import create_verse, register_builtin


def _event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class TestLiveSnapshot(unittest.TestCase):
    def test_active_and_idle_runs_and_flows(self):
        with tempfile.TemporaryDirectory() as td:
            idle = Path(td) / "runs" / "idle_run"
            active = Path(td) / "runs" / "active_run"
            _event(
                idle / "events.jsonl",
                {"verse_name": "line_world", "run_id": "idle_run", "reward": 0.0, "done": False, "info": {}},
            )
            write_lifecycle_event(str(idle), "run_started", verse_name="line_world")
            write_lifecycle_event(str(idle), "run_finished", verse_name="line_world")
            _event(
                active / "events.jsonl",
                {
                    "verse_name": "grid_world",
                    "run_id": "active_run",
                    "reward": 1.0,
                    "done": True,
                    "step_idx": 3,
                    "info": {
                        "reached_goal": True,
                        "action_info": {
                            "memory_recall_used": True,
                            "memory_recall_match_count": 2,
                            "safe_executor": {"mode": "shield_veto"},
                        },
                        "transfer_decision_records": [
                            {
                                "decision": "accept_transfer",
                                "source": {
                                    "verse_name": "line_world",
                                    "score": 0.8,
                                    "transfer_confidence": 0.6,
                                },
                                "target": {"verse_name": "grid_world"},
                            }
                        ],
                    },
                },
            )
            write_lifecycle_event(str(active), "run_started", verse_name="grid_world")
            old = 1_000_000_000.0
            os.utime(idle / "events.jsonl", (old, old))
            os.utime(active / "events.jsonl", (old, old))

            snapshot = build_live_snapshot(
                runs_root=str(Path(td) / "runs"),
                memory_root=str(Path(td) / "memory"),
                now=old + 10_000.0,
                active_window_sec=15.0,
            )
            by_id = {row["run_id"]: row for row in snapshot["runs"]}
            self.assertEqual(by_id["idle_run"]["status"], "idle")
            self.assertEqual(by_id["active_run"]["status"], "active")
            self.assertEqual(snapshot["counts"]["recent_transfers"], 1)
            self.assertEqual(snapshot["counts"]["recent_safety"], 1)
            self.assertEqual(snapshot["flows"]["transfers"][0]["source_verse"], "line_world")
            self.assertEqual(snapshot["flows"]["safety"][0]["mode"], "shield_veto")

            text = UniverseHub(
                runs_root=str(Path(td) / "runs"),
                memory_root=str(Path(td) / "memory"),
            ).render_text(width=80)
            self.assertIn("CENTRAL MEMORY", text)
            self.assertIn("grid_world", text)
            self.assertIn("transfer", text)
            self.assertIn("safety", text)

    def test_trainer_writes_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = Trainer(run_root=td)
            result = trainer.run(
                verse_spec=VerseSpec(spec_version="v1", verse_name="line_world", verse_version="0.1", seed=1),
                agent_spec=AgentSpec(
                    spec_version="v1",
                    policy_id="random_life",
                    policy_version="0.1",
                    algo="random",
                    seed=1,
                ),
                episodes=1,
                max_steps=2,
                seed=1,
                verbose=False,
            )
            text = (Path(td) / result["run_id"] / "lifecycle.jsonl").read_text(encoding="utf-8")
            self.assertIn("run_started", text)
            self.assertIn("run_finished", text)


class TestConceptRetrieval(unittest.TestCase):
    def test_concept_filter_drops_non_overlapping_rows(self):
        self.assertIn("navigation", concept_tags_for_verse("line_world"))
        self.assertNotIn("strategy", concept_tags_for_verse("line_world"))
        self.assertIn("strategy", concept_tags_for_verse("chess_world"))
        with tempfile.TemporaryDirectory() as td:
            cfg = CentralMemoryConfig(root_dir=os.path.join(td, "memory"))
            for name, verse, action in (("line_pol", "line_world", 0), ("chess_pol", "chess_world", 1)):
                run_dir = os.path.join(td, name)
                os.makedirs(run_dir)
                event = {
                    "episode_id": name,
                    "step_idx": 0,
                    "verse_name": verse,
                    "policy_id": name,
                    "obs": [1.0, 0.0],
                    "action": action,
                    "reward": 1.0,
                    "done": True,
                    "truncated": False,
                    "info": {"reached_goal": True},
                }
                with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                ingest_run(run_dir=run_dir, cfg=cfg)

            plain = find_similar(obs=[1.0, 0.0], cfg=cfg, top_k=2, min_score=-1.0)
            same = find_similar(
                obs=[1.0, 0.0],
                cfg=cfg,
                top_k=2,
                min_score=-1.0,
                concept_weight=0.0,
            )
            strategy = find_similar(
                obs=[1.0, 0.0],
                cfg=cfg,
                top_k=2,
                min_score=-1.0,
                concept_tags={"strategy"},
            )
            self.assertEqual([row.policy_id for row in plain], [row.policy_id for row in same])
            self.assertEqual([row.score for row in plain], [row.score for row in same])
            self.assertEqual({row.policy_id for row in plain}, {"line_pol", "chess_pol"})
            self.assertEqual([row.policy_id for row in strategy], ["chess_pol"])
            skills = list_skills(cfg)
            self.assertTrue(any(item["skill_id"].startswith("skill:navigation:line_world:success") for item in skills))


class TestTransferMetrics(unittest.TestCase):
    def test_append_and_summarize(self):
        with tempfile.TemporaryDirectory() as td:
            written = append_transfer_records(
                [
                    {
                        "source_verse": "line_world",
                        "target_verse": "grid_world",
                        "concept": "navigation",
                        "similarity": 0.9,
                        "confidence": 0.5,
                        "decision": "accept_transfer",
                        "episode_return": 1.0,
                    },
                    {
                        "source_verse": "chess_world",
                        "target_verse": "grid_world",
                        "concept": "strategy",
                        "similarity": 0.2,
                        "confidence": 0.1,
                        "decision": "fallback_scratch",
                        "episode_return": -1.0,
                    },
                ],
                root_dir=td,
            )
            self.assertEqual(written, 2)
            summary = summarize_transfer_root(td)
            self.assertEqual(summary["attempts"], 2)
            self.assertEqual(summary["cross_verse_fraction"], 1.0)
            self.assertAlmostEqual(summary["mean_confidence"], 0.3)
            self.assertAlmostEqual(summary["success_rate"], 0.5)
            self.assertEqual(summary["pairs"][0]["concept"] in ("navigation", "strategy"), True)


class TestEmbodiedVerse(unittest.TestCase):
    def test_point_world_and_toy_provider(self):
        register_builtin()
        verse = create_verse(
            VerseSpec(
                spec_version="v1",
                verse_name="embodied_point_world",
                verse_version="0.1",
                seed=3,
                params={"max_steps": 5, "goal_x": 0.2, "goal_y": 0.0},
            )
        )
        try:
            reset = verse.reset()
            self.assertEqual(len(reset.obs), 6)
            stepped = verse.step(1)
            self.assertEqual(len(stepped.obs), 6)
            self.assertNotEqual(stepped.obs[0], reset.obs[0])
        finally:
            verse.close()
        status = provider_status("physics_toy")
        self.assertTrue(status["available"])
        self.assertTrue(status["implemented"])
        self.assertIn("embodied_point_world", status["supported_verses"])
        warp = provider_status("nvidia_warp")
        self.assertEqual(warp["implemented"], warp["available"])
        if not warp_backend_available():
            with self.assertRaises(WarpUnavailableError):
                WarpPhysicsBackend()
        else:
            backend = WarpPhysicsBackend()
            state = backend.reset()
            self.assertIn("x", backend.step(state, 1))


class TestPlatformCommands(unittest.TestCase):
    def test_parser_and_page(self):
        parser = build_parser()
        map_args = parser.parse_args(["map", "--once", "--runs-root", "runs"])
        self.assertEqual(map_args.command, "map")
        self.assertTrue(map_args.once)
        web_args = parser.parse_args(["map", "--web", "--dry-run", "--port", "8777"])
        self.assertTrue(web_args.web)
        self.assertEqual(web_args.port, 8777)
        transfer_args = parser.parse_args(["transfer", "overview", "--json"])
        self.assertEqual(transfer_args.transfer_command, "overview")
        self.assertTrue(transfer_args.json)
        page = render_page()
        self.assertIn("/api/snapshot", page)
        self.assertIn("Multiverse live map", page)


if __name__ == "__main__":
    unittest.main()
