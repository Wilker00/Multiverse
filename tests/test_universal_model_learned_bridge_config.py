import os
import tempfile
import unittest

from models.contrastive_bridge import ContrastiveBridge
from models.universal_model import UniversalModel, UniversalModelConfig


class TestUniversalModelLearnedBridgeConfig(unittest.TestCase):
    def _write_bridge_ckpt(self, path: str) -> None:
        model = ContrastiveBridge(n_embd=32, proj_dim=16, temperature_init=0.07)
        model.save(path)

    def test_save_and_load_preserves_learned_bridge_config(self):
        with tempfile.TemporaryDirectory() as td:
            memory_dir = os.path.join(td, "central_memory")
            os.makedirs(memory_dir, exist_ok=True)
            with open(os.path.join(memory_dir, "memories.jsonl"), "w", encoding="utf-8"):
                pass
            with open(os.path.join(memory_dir, "dedupe_index.json"), "w", encoding="utf-8") as f:
                f.write("[]")

            bridge_ckpt = os.path.join(td, "bridge.pt")
            self._write_bridge_ckpt(bridge_ckpt)

            model = UniversalModel(
                UniversalModelConfig(
                    memory_dir=memory_dir,
                    learned_bridge_enabled=True,
                    learned_bridge_model_path=bridge_ckpt,
                    learned_bridge_score_weight=0.42,
                )
            )
            out_dir = os.path.join(td, "model_pkg")
            model.save(out_dir, snapshot_memory=True)

            loaded = UniversalModel.load(out_dir)
            self.assertTrue(loaded.config.learned_bridge_enabled)
            self.assertAlmostEqual(float(loaded.config.learned_bridge_score_weight), 0.42, places=6)
            self.assertIsNotNone(loaded.config.learned_bridge_model_path)
            assert loaded.config.learned_bridge_model_path is not None
            self.assertTrue(os.path.isfile(loaded.config.learned_bridge_model_path))


if __name__ == "__main__":
    unittest.main()
