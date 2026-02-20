import json
import os
import tempfile
import unittest

from memory.central_repository import CentralMemoryConfig, find_similar
from memory.embeddings import obs_to_universal_vector, obs_to_vector


class TestCentralRepositoryUniversalFallback(unittest.TestCase):
    def test_find_similar_uses_universal_vector_when_raw_dims_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = CentralMemoryConfig(root_dir=td)
            mem_path = os.path.join(td, "memories.jsonl")
            row_obs = {"a": 1, "b": [2, 3]}
            row = {
                "run_id": "run_src",
                "episode_id": "ep1",
                "step_idx": 0,
                "t_ms": 1,
                "verse_name": "grid_world",
                "obs": row_obs,
                "obs_vector": obs_to_vector(row_obs),
                "obs_vector_u": obs_to_universal_vector(row_obs),
                "action": 1,
                "reward": 1.0,
                "memory_tier": "ltm",
                "memory_family": "procedural",
                "memory_type": "spatial_procedural",
            }
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Provide required companion files expected by repository helpers.
            for name in ("ltm_memories.jsonl", "stm_memories.jsonl"):
                with open(os.path.join(td, name), "w", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Mismatched raw dimensionality (5 vs 3) should still retrieve via universal encoder.
            query_obs = {"a": 1, "b": [2, 3], "extra": [0, 0]}
            matches = find_similar(
                obs=query_obs,
                cfg=cfg,
                top_k=1,
                verse_name=None,
                min_score=-1.0,
            )
            self.assertEqual(len(matches), 1)
            self.assertEqual(str(matches[0].run_id), "run_src")


if __name__ == "__main__":
    unittest.main()
