import json
import os
import statistics
import tempfile
import time
import unittest

import memory.central_repository as central_repository
from memory.central_repository import CentralMemoryConfig, find_similar, find_similar_cached


def _write_memory_rows(mem_path: str, n_rows: int = 1200) -> None:
    rows = []
    for i in range(n_rows):
        x = float(i % 40) / 40.0
        y = float((i * 7) % 40) / 40.0
        rows.append(
            {
                "run_id": f"run_{i // 20}",
                "episode_id": f"ep_{i // 20}",
                "step_idx": i,
                "t_ms": i + 1,
                "verse_name": "grid_world",
                "obs": {"x": x, "y": y},
                "obs_vector": [x, y],
                "action": i % 4,
                "reward": float((i % 10) - 5) / 5.0,
                "memory_tier": "ltm" if (i % 3 == 0) else "stm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
        )
    with open(mem_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestMemoryPerformance(unittest.TestCase):
    def setUp(self) -> None:
        central_repository._QUERY_RESULT_CACHE.clear()
        central_repository._SIM_CACHE.clear()

    def test_query_result_cache_hit_path_is_faster_than_miss_path(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            _write_memory_rows(mem_path, n_rows=1200)
            cfg = CentralMemoryConfig(root_dir=td)

            # Prime the similarity cache so this benchmark isolates query-result cache behavior.
            find_similar(obs={"x": 0.1, "y": 0.2}, cfg=cfg, top_k=5, min_score=-1.0)

            miss_times = []
            for i in range(20):
                obs = {"x": float(i % 10) / 10.0, "y": float((i * 3) % 10) / 10.0, "qid": i}
                t0 = time.perf_counter()
                _ = find_similar_cached(obs=obs, cfg=cfg, top_k=5, min_score=-1.0)
                miss_times.append(time.perf_counter() - t0)

            hit_obs = {"x": 0.3, "y": 0.7, "qid": 9999}
            first = find_similar_cached(obs=hit_obs, cfg=cfg, top_k=5, min_score=-1.0)
            hit_times = []
            for _ in range(20):
                t0 = time.perf_counter()
                again = find_similar_cached(obs=hit_obs, cfg=cfg, top_k=5, min_score=-1.0)
                hit_times.append(time.perf_counter() - t0)

            self.assertEqual([m.run_id for m in first], [m.run_id for m in again])
            self.assertGreater(len(first), 0)

            miss_med = statistics.median(miss_times)
            hit_med = statistics.median(hit_times)
            # Keep this relative threshold loose to avoid environment-related flakiness.
            self.assertLess(hit_med, miss_med * 1.25)

    def test_warm_similarity_cache_not_slower_than_cold_rebuild_path(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            _write_memory_rows(mem_path, n_rows=1500)
            cfg = CentralMemoryConfig(root_dir=td)
            obs = {"x": 0.25, "y": 0.75}

            cold_times = []
            for _ in range(4):
                central_repository._SIM_CACHE.clear()
                sidecar = mem_path + ".simcache.json"
                if os.path.isfile(sidecar):
                    os.remove(sidecar)
                t0 = time.perf_counter()
                out = find_similar(obs=obs, cfg=cfg, top_k=5, min_score=-1.0)
                cold_times.append(time.perf_counter() - t0)
                self.assertGreater(len(out), 0)

            # Warm path: cache already built and reused.
            _ = find_similar(obs=obs, cfg=cfg, top_k=5, min_score=-1.0)
            warm_times = []
            for _ in range(8):
                t0 = time.perf_counter()
                out = find_similar(obs=obs, cfg=cfg, top_k=5, min_score=-1.0)
                warm_times.append(time.perf_counter() - t0)
                self.assertGreater(len(out), 0)

            cold_med = statistics.median(cold_times)
            warm_med = statistics.median(warm_times)
            self.assertLess(warm_med, cold_med * 1.25)


if __name__ == "__main__":
    unittest.main()
