import os
import tempfile
import unittest

import torch

from models.confidence_monitor import (
    ConfidenceMonitor,
    ConfidenceMonitorConfig,
    load_confidence_monitor,
    save_confidence_monitor,
)


class TestConfidenceMonitorModel(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        cfg = ConfidenceMonitorConfig(input_dim=12, hidden_dim=32, hidden_layers=2, dropout=0.0)
        model = ConfidenceMonitor(cfg)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "confidence_monitor.pt")
            save_confidence_monitor(model=model, path=path, train_config={"epochs": 1}, metrics={"ok": True})
            self.assertTrue(os.path.isfile(path))
            loaded = load_confidence_monitor(path)
            x = torch.randn(4, 12)
            probs = loaded.predict_danger_prob(x)
            self.assertEqual(tuple(probs.shape), (4,))
            self.assertTrue(bool(torch.all((probs >= 0.0) & (probs <= 1.0)).item()))


if __name__ == "__main__":
    unittest.main()
