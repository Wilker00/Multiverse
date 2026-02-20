import os
import tempfile
import unittest

import torch

from models.contrastive_bridge import ContrastiveBridge, ContrastiveBridgeConfig


class TestContrastiveBridge(unittest.TestCase):
    def test_forward_and_loss(self):
        model = ContrastiveBridge(n_embd=32, proj_dim=16, temperature_init=0.1)
        obs_a = [
            {"agent": {"x": 1, "y": 2}, "battery": 0.7},
            {"material": -2.0, "king_safety": 1.0},
            {"pos": 4, "goal": 8, "t": 3},
        ]
        obs_b = [
            {"x": 3, "y": 1, "nearest_charger_dist": 4.0},
            {"territory_delta": -3, "risk": 2},
            {"pos": 2, "goal": 8, "t": 5},
        ]
        out = model(obs_a, obs_b, return_loss=True)
        self.assertIn("logits", out)
        self.assertIn("loss", out)
        self.assertEqual(tuple(out["logits"].shape), (3, 3))
        self.assertTrue(torch.isfinite(out["loss"]).item())

    def test_save_and_load(self):
        model = ContrastiveBridge(n_embd=24, proj_dim=24, temperature_init=0.07)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bridge.pt")
            model.save(path, extra={"note": "unit-test"})
            loaded = ContrastiveBridge.load(path)
            self.assertEqual(int(loaded.config.n_embd), 24)
            sims = loaded.similarity([{"a": 1.0}], [{"b": 2.0}])
            self.assertEqual(tuple(sims.shape), (1, 1))

    def test_config_rejects_unknown_keys_by_default(self):
        with self.assertRaises(ValueError):
            ContrastiveBridgeConfig.from_dict({"n_embd": 64, "typo_key": True})


if __name__ == "__main__":
    unittest.main()
