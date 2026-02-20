import json
import os
import tempfile
import unittest

from memory.central_repository import CentralMemoryConfig
from memory.semantic_bridge import translate_observation
from models.contrastive_bridge import ContrastiveBridge
from orchestrator.scenario_matcher import ScenarioRequest, recommend_action


class TestScenarioMatcherLearnedBridge(unittest.TestCase):
    def _write_bridge_ckpt(self, path: str) -> None:
        model = ContrastiveBridge(n_embd=32, proj_dim=16, temperature_init=0.07)
        model.save(path)

    def test_semantic_fallback_accepts_learned_bridge_options(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "contrastive_bridge.pt")
            self._write_bridge_ckpt(ckpt)

            row_obs = {"pos": 3, "goal": 8, "t": 0}
            query_obs = translate_observation(
                obs=row_obs,
                source_verse_name="line_world",
                target_verse_name="grid_world",
            )
            self.assertIsInstance(query_obs, dict)

            mem_path = os.path.join(td, "memories.jsonl")
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_line",
                            "episode_id": "ep_1",
                            "step_idx": 0,
                            "t_ms": 0,
                            "verse_name": "line_world",
                            "obs": row_obs,
                            "action": 1,
                            "reward": 0.1,
                            "tags": ["navigation"],
                        }
                    )
                    + "\n"
                )

            advice = recommend_action(
                request=ScenarioRequest(
                    obs=query_obs,
                    verse_name="grid_world",
                    top_k=1,
                    min_score=-1.0,
                    enable_semantic_bridge=True,
                    enable_tag_fallback=True,
                    enable_knowledge_graph=False,
                    enable_confidence_auditor=False,
                    learned_bridge_enabled=True,
                    learned_bridge_model_path=ckpt,
                    learned_bridge_score_weight=0.5,
                ),
                cfg=CentralMemoryConfig(root_dir=td),
            )
            self.assertIsNotNone(advice)
            assert advice is not None
            self.assertEqual(str(advice.strategy), "semantic_bridge")


if __name__ == "__main__":
    unittest.main()
