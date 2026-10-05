import json
import os
import tempfile
import unittest
from unittest.mock import patch

from memory.central_repository import (
    CentralMemoryConfig,
    _open_dedupe_db,
    find_similar,
    find_similar_cached,
    ingest_run,
)


class TestCentralRepositoryRunQuality(unittest.TestCase):
    def test_weighted_ranking_prefers_quality_and_exposes_components(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = CentralMemoryConfig(root_dir=os.path.join(td, "memory"))
            for name, obs, score, action in (
                ("exact_bad", [1.0, 0.0], 0.0, 0),
                ("near_good", [0.0, 1.0], 1.0, 1),
            ):
                run_dir = os.path.join(td, name)
                os.makedirs(run_dir)
                event = {
                    "episode_id": name,
                    "step_idx": 0,
                    "verse_name": "line_world",
                    "policy_id": name,
                    "obs": obs,
                    "next_obs": [2.0, float(action)],
                    "action": action,
                    "reward": float(action),
                    "done": True,
                    "truncated": False,
                    "transfer_score": score,
                    "info": {"reached_goal": bool(action)},
                }
                with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                ingest_run(run_dir=run_dir, cfg=cfg)

            plain = find_similar(obs=[1.0, 0.0], cfg=cfg, top_k=2, min_score=-1.0)
            weighted = find_similar_cached(
                obs=[1.0, 0.0],
                cfg=cfg,
                top_k=2,
                min_score=-1.0,
                quality_weight=2.0,
            )
            self.assertEqual(plain[0].policy_id, "exact_bad")
            self.assertEqual(weighted[0].policy_id, "near_good")
            self.assertIsNotNone(weighted[0].similarity_score)
            self.assertEqual(weighted[0].quality_weight, 2.0)
            self.assertGreater(weighted[0].score, weighted[1].score)

    def test_reputation_weight_uses_provider_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = CentralMemoryConfig(root_dir=os.path.join(td, "memory"))
            for name, action in (("low_rep", 0), ("high_rep", 1)):
                run_dir = os.path.join(td, name)
                os.makedirs(run_dir)
                event = {
                    "episode_id": name,
                    "step_idx": 0,
                    "verse_name": "line_world",
                    "policy_id": name,
                    "obs": {"x": 1.0, "source": action},
                    "next_obs": {"x": 2.0, "source": action},
                    "action": action,
                    "reward": 0.0,
                    "done": True,
                    "truncated": False,
                    "transfer_score": 0.5,
                    "info": {"reached_goal": True},
                }
                with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                ingest_run(run_dir=run_dir, cfg=cfg)
            ledger = {
                "providers": {
                    "low_rep": {"reputation": 0.0},
                    "high_rep": {"reputation": 1.0},
                }
            }
            with open(os.path.join(cfg.root_dir, "knowledge_market_ledger.json"), "w", encoding="utf-8") as handle:
                json.dump(ledger, handle)

            matches = find_similar(
                obs={"x": 1.0, "source": 0},
                cfg=cfg,
                top_k=2,
                min_score=-1.0,
                reputation_weight=2.0,
            )
            self.assertEqual(matches[0].run_id, "high_rep")
            self.assertEqual(matches[0].provider_reputation, 1.0)
            self.assertEqual(matches[0].reputation_weight, 2.0)

    def test_failed_file_flush_rolls_back_key_and_file_appends(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "source_run")
            os.makedirs(run_dir)
            event = {
                "episode_id": "ep",
                "step_idx": 0,
                "verse_name": "line_world",
                "policy_id": "source",
                "obs": {"pos": 0},
                "next_obs": {"pos": 1},
                "action": 1,
                "reward": 1.0,
                "done": True,
                "truncated": False,
                "info": {"reached_goal": True},
            }
            with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            cfg = CentralMemoryConfig(root_dir=os.path.join(td, "memory"))

            with patch(
                "memory.central_repository_ingest_support.os.fsync",
                side_effect=OSError("injected flush failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected flush failure"):
                    ingest_run(run_dir=run_dir, cfg=cfg)

            conn = _open_dedupe_db(cfg)
            try:
                self.assertEqual(int(conn.execute("SELECT COUNT(*) FROM dedupe_keys").fetchone()[0]), 0)
            finally:
                conn.close()
            for filename in ("memories.jsonl", "ltm_memories.jsonl", "stm_memories.jsonl"):
                with open(os.path.join(cfg.root_dir, filename), "r", encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "")

            retry = ingest_run(run_dir=run_dir, cfg=cfg)
            self.assertEqual(retry.added_events, 1)

    def test_higher_quality_duplicate_replaces_lower_quality_row(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = CentralMemoryConfig(root_dir=os.path.join(td, "memory"))
            base = {
                "episode_id": "ep",
                "step_idx": 0,
                "verse_name": "line_world",
                "obs": {"pos": 0},
                "next_obs": {"pos": 1},
                "action": 1,
                "reward": 1.0,
                "done": True,
                "truncated": False,
            }
            for name, policy_id, success in (
                ("bad_first", "random_bad", False),
                ("good_second", "trained_good", True),
            ):
                run_dir = os.path.join(td, name)
                os.makedirs(run_dir)
                event = dict(base, policy_id=policy_id, info={"reached_goal": success})
                with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                ingest_run(run_dir=run_dir, cfg=cfg)

            with open(os.path.join(td, "memory", "memories.jsonl"), "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["policy_id"], "trained_good")
            self.assertAlmostEqual(float(rows[0]["transfer_score"]), 1.0, places=6)

    def test_equal_or_lower_quality_duplicate_is_still_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = CentralMemoryConfig(root_dir=os.path.join(td, "memory"))
            base = {
                "episode_id": "ep",
                "step_idx": 0,
                "verse_name": "line_world",
                "obs": {"pos": 0},
                "next_obs": {"pos": 1},
                "action": 1,
                "reward": 1.0,
                "done": True,
                "truncated": False,
                "info": {"reached_goal": True},
            }
            stats = []
            for name, policy_id, score in (
                ("best_first", "best", 1.0),
                ("equal_second", "equal", 1.0),
                ("lower_third", "lower", 0.5),
            ):
                run_dir = os.path.join(td, name)
                os.makedirs(run_dir)
                event = dict(base, policy_id=policy_id, transfer_score=score)
                with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                stats.append(ingest_run(run_dir=run_dir, cfg=cfg))

            self.assertEqual(stats[0].added_events, 1)
            self.assertEqual(stats[1].skipped_duplicates, 1)
            self.assertEqual(stats[2].skipped_duplicates, 1)
            with open(os.path.join(td, "memory", "memories.jsonl"), "r", encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual(row["policy_id"], "best")
            self.assertAlmostEqual(float(row["transfer_score"]), 1.0, places=6)

    def test_ingest_attaches_success_rate_and_filter_uses_it(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "source_run")
            memory_dir = os.path.join(td, "memory")
            os.makedirs(run_dir, exist_ok=True)
            events = [
                {
                    "episode_id": "success",
                    "step_idx": 0,
                    "verse_name": "line_world",
                    "policy_id": "quality_source",
                    "obs": {"pos": 0, "t": 0},
                    "next_obs": {"pos": 1, "t": 1},
                    "action": 1,
                    "reward": 1.0,
                    "done": True,
                    "truncated": False,
                    "info": {"reached_goal": True},
                },
                {
                    "episode_id": "failure",
                    "step_idx": 0,
                    "verse_name": "line_world",
                    "policy_id": "quality_source",
                    "obs": {"pos": 2, "t": 0},
                    "next_obs": {"pos": 2, "t": 1},
                    "action": 0,
                    "reward": -1.0,
                    "done": False,
                    "truncated": True,
                    "info": {"reached_goal": False},
                },
            ]
            with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")

            cfg = CentralMemoryConfig(root_dir=memory_dir)
            ingest_run(run_dir=run_dir, cfg=cfg)

            with open(os.path.join(memory_dir, "memories.jsonl"), "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertAlmostEqual(float(row["transfer_score"]), 0.5, places=6)
                self.assertAlmostEqual(float(row["source_run_success_rate"]), 0.5, places=6)
                self.assertAlmostEqual(float(row["source_run_mean_return"]), 0.0, places=6)

            accepted = find_similar(
                obs={"pos": 0, "t": 0},
                cfg=cfg,
                top_k=5,
                verse_name="line_world",
                min_score=-1.0,
                min_transfer_score=0.5,
            )
            rejected = find_similar(
                obs={"pos": 0, "t": 0},
                cfg=cfg,
                top_k=5,
                verse_name="line_world",
                min_score=-1.0,
                min_transfer_score=0.500001,
            )
            self.assertEqual(len(accepted), 2)
            self.assertEqual(rejected, [])
            cached_accepted = find_similar_cached(
                obs={"pos": 0, "t": 0},
                cfg=cfg,
                top_k=5,
                verse_name="line_world",
                min_score=-1.0,
                min_transfer_score=0.5,
            )
            cached_rejected = find_similar_cached(
                obs={"pos": 0, "t": 0},
                cfg=cfg,
                top_k=5,
                verse_name="line_world",
                min_score=-1.0,
                min_transfer_score=0.500001,
            )
            self.assertEqual(len(cached_accepted), 2)
            self.assertEqual(cached_rejected, [])

    def test_explicit_transfer_score_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "source_run")
            memory_dir = os.path.join(td, "memory")
            os.makedirs(run_dir, exist_ok=True)
            event = {
                "episode_id": "success",
                "step_idx": 0,
                "verse_name": "line_world",
                "policy_id": "explicit_source",
                "obs": {"pos": 0},
                "next_obs": {"pos": 1},
                "action": 1,
                "reward": 1.0,
                "done": True,
                "truncated": False,
                "transfer_score": 1.2,
                "info": {"reached_goal": True},
            }
            with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

            cfg = CentralMemoryConfig(root_dir=memory_dir)
            ingest_run(run_dir=run_dir, cfg=cfg)
            with open(os.path.join(memory_dir, "memories.jsonl"), "r", encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertAlmostEqual(float(row["transfer_score"]), 1.2, places=6)
            self.assertAlmostEqual(float(row["event_transfer_score"]), 1.2, places=6)
            self.assertAlmostEqual(float(row["verified_transfer_score"]), 1.0, places=6)
            self.assertEqual(row["transfer_score_provenance"], "evaluator_v1")
            self.assertAlmostEqual(float(row["source_run_success_rate"]), 1.0, places=6)

    def test_forged_event_score_cannot_pass_verified_quality_filter(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "forged")
            os.makedirs(run_dir)
            event = {
                "episode_id": "bad",
                "step_idx": 0,
                "verse_name": "line_world",
                "policy_id": "forged",
                "obs": {"pos": 0},
                "next_obs": {"pos": 0},
                "action": 0,
                "reward": 100.0,
                "done": True,
                "truncated": False,
                "transfer_score": 10.0,
                "info": {"reached_goal": False},
            }
            with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            cfg = CentralMemoryConfig(root_dir=os.path.join(td, "memory"))
            ingest_run(run_dir=run_dir, cfg=cfg)

            all_rows = find_similar(obs={"pos": 0}, cfg=cfg, min_score=-1.0)
            trusted_rows = find_similar(
                obs={"pos": 0},
                cfg=cfg,
                min_score=-1.0,
                min_verified_transfer_score=0.5,
            )
            self.assertEqual(float(all_rows[0].transfer_score), 10.0)
            self.assertEqual(float(all_rows[0].verified_transfer_score), 0.0)
            self.assertEqual(trusted_rows, [])


if __name__ == "__main__":
    unittest.main()
