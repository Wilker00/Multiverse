import os
import tempfile
import unittest

from core.safe_executor import SafeExecutor, SafeExecutorConfig
from models.confidence_monitor import ConfidenceMonitor, ConfidenceMonitorConfig, save_confidence_monitor


class _DummyVerse:
    class _Spec:
        verse_name = "grid_world"

    spec = _Spec()


class _DummyAgent:
    def action_diagnostics(self, obs):
        _ = obs
        return {"danger_scores": [0.0, 0.0], "sample_probs": [1.0, 1.0]}


class TestSafeExecutorConfidenceModel(unittest.TestCase):
    def test_config_accepts_confidence_model_keys(self):
        cfg = SafeExecutorConfig.from_dict(
            {
                "confidence_model_path": "models/confidence_monitor.pt",
                "confidence_model_weight": 0.7,
                "confidence_model_obs_dim": 64,
            }
        )
        self.assertEqual(str(cfg.confidence_model_path), "models/confidence_monitor.pt")
        self.assertAlmostEqual(float(cfg.confidence_model_weight), 0.7, places=5)
        self.assertEqual(int(cfg.confidence_model_obs_dim), 64)

    def test_estimate_action_risk_blends_with_model(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "confidence_monitor.pt")
            model = ConfidenceMonitor(ConfidenceMonitorConfig(input_dim=66, hidden_dim=16, hidden_layers=1, dropout=0.0))
            # Bias to high danger so blended danger should increase above diagnostics.
            for p in model.parameters():
                p.data.zero_()
            for name, p in model.named_parameters():
                if name.endswith("bias"):
                    p.data.fill_(3.0)  # sigmoid(3) ~= 0.95
            save_confidence_monitor(model=model, path=model_path)

            cfg = SafeExecutorConfig.from_dict(
                {
                    "enabled": True,
                    "confidence_model_path": model_path,
                    "confidence_model_weight": 1.0,
                    "confidence_model_obs_dim": 64,
                }
            )
            se = SafeExecutor(config=cfg, verse=_DummyVerse(), fallback_agent=None)
            risk = se._estimate_action_risk(_DummyAgent(), {"x": 1, "y": 2, "t": 0}, 0)
            self.assertGreater(float(risk.get("danger", 0.0)), 0.80)
            self.assertLess(float(risk.get("confidence", 1.0)), 0.30)


if __name__ == "__main__":
    unittest.main()
