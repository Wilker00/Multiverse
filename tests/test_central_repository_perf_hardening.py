import json
import os
import sqlite3
import tempfile
import unittest

from memory.central_repository import (
    CentralMemoryConfig,
    find_similar,
    ingest_run,
    get_similarity_runtime_metrics,
    reset_similarity_runtime_metrics,
    run_similarity_canaries,
    save_similarity_canary,
)
import memory.central_repository as central_repository


def _count_jsonl_rows(path: str) -> int:
    n = 0
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


class TestCentralRepositoryPerfHardening(unittest.TestCase):
    def test_ingest_run_uses_sqlite_dedupe_and_skips_replays(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = os.path.join(td, "runs")
            run_dir = os.path.join(runs_root, "run_a")
            mem_dir = os.path.join(td, "central_memory")
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(mem_dir, exist_ok=True)

            events = [
                {
                    "episode_id": "ep1",
                    "step_idx": 0,
                    "verse_name": "grid_world",
                    "obs": {"x": 1, "y": 1},
                    "action": 1,
                    "next_obs": {"x": 1, "y": 2},
                    "reward": 0.1,
                    "done": False,
                    "truncated": False,
                    "info": {},
                },
                {
                    # Duplicate of row 0 by dedupe key.
                    "episode_id": "epX",
                    "step_idx": 99,
                    "verse_name": "grid_world",
                    "obs": {"x": 1, "y": 1},
                    "action": 1,
                    "next_obs": {"x": 1, "y": 2},
                    "reward": 0.1,
                    "done": False,
                    "truncated": False,
                    "info": {},
                },
                {
                    "episode_id": "ep2",
                    "step_idx": 1,
                    "verse_name": "grid_world",
                    "obs": {"x": 2, "y": 1},
                    "action": 2,
                    "next_obs": {"x": 3, "y": 1},
                    "reward": 0.2,
                    "done": False,
                    "truncated": False,
                    "info": {},
                },
            ]
            with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as f:
                for r in events:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            cfg = CentralMemoryConfig(root_dir=mem_dir)
            st1 = ingest_run(run_dir=run_dir, cfg=cfg)
            self.assertEqual(int(st1.input_events), 3)
            self.assertEqual(int(st1.selected_events), 3)
            self.assertEqual(int(st1.added_events), 2)
            self.assertEqual(int(st1.skipped_duplicates), 1)

            st2 = ingest_run(run_dir=run_dir, cfg=cfg)
            self.assertEqual(int(st2.added_events), 0)
            self.assertEqual(int(st2.skipped_duplicates), 3)

            self.assertEqual(_count_jsonl_rows(os.path.join(mem_dir, "memories.jsonl")), 2)

            db_path = os.path.join(mem_dir, cfg.dedupe_db_filename)
            self.assertTrue(os.path.isfile(db_path))
            conn = sqlite3.connect(db_path)
            try:
                nkeys = int(conn.execute("SELECT COUNT(*) FROM dedupe_keys").fetchone()[0] or 0)
            finally:
                conn.close()
            self.assertEqual(nkeys, 2)

    def test_find_similar_cache_invalidates_when_memory_file_changes(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            row1 = {
                "run_id": "run_old",
                "episode_id": "ep1",
                "step_idx": 0,
                "t_ms": 1,
                "verse_name": "grid_world",
                "obs": {"x": 0, "y": 1},
                "obs_vector": [0.0, 1.0],
                "action": 0,
                "reward": 0.0,
                "memory_tier": "stm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(row1, ensure_ascii=False) + "\n")

            cfg = CentralMemoryConfig(root_dir=td)
            m1 = find_similar(obs={"x": 1, "y": 0}, cfg=cfg, top_k=1, min_score=-1.0)
            self.assertEqual(len(m1), 1)
            self.assertEqual(str(m1[0].run_id), "run_old")
            self.assertTrue(os.path.isfile(mem_path + ".simcache.json"))

            row2 = {
                "run_id": "run_new",
                "episode_id": "ep2",
                "step_idx": 0,
                "t_ms": 2,
                "verse_name": "grid_world",
                "obs": {"x": 1, "y": 0},
                "obs_vector": [1.0, 0.0],
                "action": 1,
                "reward": 1.0,
                "memory_tier": "ltm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
            with open(mem_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row2, ensure_ascii=False) + "\n")

            m2 = find_similar(obs={"x": 1, "y": 0}, cfg=cfg, top_k=1, min_score=-1.0)
            self.assertEqual(len(m2), 1)
            self.assertEqual(str(m2[0].run_id), "run_new")

    def test_find_similar_ignores_legacy_pickle_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            row = {
                "run_id": "run_a",
                "episode_id": "ep1",
                "step_idx": 0,
                "t_ms": 1,
                "verse_name": "grid_world",
                "obs": {"x": 1, "y": 0},
                "obs_vector": [1.0, 0.0],
                "action": 0,
                "reward": 0.0,
                "memory_tier": "stm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with open(mem_path + ".simcache.pkl", "wb") as f:
                f.write(b"not-a-pickle")

            cfg = CentralMemoryConfig(root_dir=td)
            out = find_similar(obs={"x": 1, "y": 0}, cfg=cfg, top_k=1, min_score=-1.0)
            self.assertEqual(len(out), 1)
            self.assertEqual(str(out[0].run_id), "run_a")
            self.assertFalse(os.path.isfile(mem_path + ".simcache.pkl"))
            self.assertTrue(os.path.isfile(mem_path + ".simcache.json"))

    def test_find_similar_ann_candidate_fallback_with_type_filter(self):
        if central_repository.NearestNeighbors is None:
            self.skipTest("scikit-learn is not available")
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            rows = []
            for i in range(300):
                row = {
                    "run_id": f"run_stm_{i}",
                    "episode_id": f"ep{i}",
                    "step_idx": i,
                    "t_ms": i + 1,
                    "verse_name": "grid_world",
                    "obs": {"x": 1, "y": 0},
                    "obs_vector": [1.0, 0.0],
                    "action": 0,
                    "reward": 0.0,
                    "memory_tier": "stm",
                    "memory_family": "procedural",
                    "memory_type": "spatial_procedural",
                }
                rows.append(row)
            rows[-1]["run_id"] = "run_ltm_target"
            rows[-1]["obs"] = {"x": 0, "y": 1}
            rows[-1]["obs_vector"] = [0.0, 1.0]
            rows[-1]["memory_type"] = "strategic_declarative"

            with open(mem_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            cfg = CentralMemoryConfig(root_dir=td)
            old_factor = os.environ.get("MULTIVERSE_SIM_ANN_FACTOR")
            old_enabled = os.environ.get("MULTIVERSE_SIM_USE_ANN")
            os.environ["MULTIVERSE_SIM_USE_ANN"] = "1"
            os.environ["MULTIVERSE_SIM_ANN_FACTOR"] = "1"
            try:
                out = find_similar(
                    obs={"x": 1, "y": 0},
                    cfg=cfg,
                    top_k=1,
                    min_score=-1.0,
                    memory_types={"strategic_declarative"},
                )
            finally:
                if old_factor is None:
                    os.environ.pop("MULTIVERSE_SIM_ANN_FACTOR", None)
                else:
                    os.environ["MULTIVERSE_SIM_ANN_FACTOR"] = old_factor
                if old_enabled is None:
                    os.environ.pop("MULTIVERSE_SIM_USE_ANN", None)
                else:
                    os.environ["MULTIVERSE_SIM_USE_ANN"] = old_enabled

            self.assertEqual(len(out), 1)
            self.assertEqual(str(out[0].run_id), "run_ltm_target")

    def test_similarity_canary_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            rows = [
                {
                    "run_id": "run_expected",
                    "episode_id": "ep1",
                    "step_idx": 0,
                    "t_ms": 1,
                    "verse_name": "grid_world",
                    "obs": {"x": 1, "y": 0},
                    "obs_vector": [1.0, 0.0],
                    "action": 0,
                    "reward": 0.0,
                    "memory_tier": "stm",
                    "memory_family": "procedural",
                    "memory_type": "spatial_procedural",
                },
                {
                    "run_id": "run_other",
                    "episode_id": "ep2",
                    "step_idx": 1,
                    "t_ms": 2,
                    "verse_name": "grid_world",
                    "obs": {"x": 0, "y": 1},
                    "obs_vector": [0.0, 1.0],
                    "action": 1,
                    "reward": 0.0,
                    "memory_tier": "stm",
                    "memory_family": "procedural",
                    "memory_type": "spatial_procedural",
                },
            ]
            with open(mem_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            cfg = CentralMemoryConfig(root_dir=td)
            save_similarity_canary(
                cfg=cfg,
                canary_id="c1",
                obs={"x": 1, "y": 0},
                expected_run_id="run_expected",
                top_k=1,
            )
            out = run_similarity_canaries(cfg=cfg)
            self.assertEqual(int(out.get("total", 0)), 1)
            self.assertEqual(int(out.get("ann_hits", 0)), 1)
            self.assertEqual(int(out.get("exact_hits", 0)), 1)
            self.assertGreaterEqual(float(out.get("agreement_rate", 0.0)), 1.0)

    def test_ann_drift_metrics_autotune_factor(self):
        old_max = os.environ.get("MULTIVERSE_SIM_ANN_MAX_DRIFT")
        try:
            os.environ["MULTIVERSE_SIM_ANN_MAX_DRIFT"] = "0.01"
            reset_similarity_runtime_metrics()
            central_repository._ann_record_drift(drift=0.20, n_rows=4096, top_n=4)
            m = get_similarity_runtime_metrics()
            self.assertIsNotNone(m.get("ann_dynamic_factor"))
            self.assertGreater(int(m.get("ann_drift_checks", 0)), 0)
            self.assertGreater(float(m.get("ann_max_drift", 0.0)), 0.01)
        finally:
            if old_max is None:
                os.environ.pop("MULTIVERSE_SIM_ANN_MAX_DRIFT", None)
            else:
                os.environ["MULTIVERSE_SIM_ANN_MAX_DRIFT"] = old_max


if __name__ == "__main__":
    unittest.main()
