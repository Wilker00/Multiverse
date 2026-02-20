import os
import tempfile
import unittest

import torch

from models.meta_transformer import MetaTransformer
from models.universal_model import UniversalModel, UniversalModelConfig


class TestUniversalModelGeneralizedMeta(unittest.TestCase):
    def _write_ckpt(self, path: str, *, use_generalized_input: bool) -> None:
        model = MetaTransformer(
            state_dim=1,
            action_dim=5,
            n_embd=32,
            use_generalized_input=use_generalized_input,
        )
        payload = {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "history_len": 4,
        }
        torch.save(payload, path)

    def test_predict_meta_with_generalized_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "meta_generalized.pt")
            self._write_ckpt(ckpt_path, use_generalized_input=True)

            model = UniversalModel(
                UniversalModelConfig(
                    memory_dir=td,
                    meta_model_path=ckpt_path,
                    prefer_meta_policy=True,
                )
            )
            out = model._predict_meta(
                obs={"agent": {"x": 2, "y": 1}, "battery": 0.75},
                recent_history=[
                    {"obs": {"agent": {"x": 1, "y": 1}, "battery": 0.9}, "action": 2, "reward": 0.1}
                ],
            )
            self.assertIsNotNone(out)
            self.assertIn("action", out)
            self.assertTrue(0 <= int(out["action"]) < 5)

    def test_predict_meta_with_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "meta_legacy.pt")
            self._write_ckpt(ckpt_path, use_generalized_input=False)

            model = UniversalModel(
                UniversalModelConfig(
                    memory_dir=td,
                    meta_model_path=ckpt_path,
                    prefer_meta_policy=True,
                )
            )
            out = model._predict_meta(
                obs={"x": 1.0, "y": -1.0},
                recent_history=[
                    {"obs": {"x": 0.0, "y": 0.0}, "action": 1, "reward": 0.0},
                ],
            )
            self.assertIsNotNone(out)
            self.assertIn("action", out)
            self.assertTrue(0 <= int(out["action"]) < 5)


if __name__ == "__main__":
    unittest.main()
