import tempfile
import unittest

import torch

from models.decision_transformer import (
    DecisionTransformer,
    DecisionTransformerConfig,
    load_decision_transformer_checkpoint,
)


class TestDecisionTransformer(unittest.TestCase):
    def test_forward_shapes(self):
        cfg = DecisionTransformerConfig(
            state_dim=8,
            action_dim=5,
            context_len=6,
            d_model=32,
            n_head=4,
            n_layer=2,
            dropout=0.0,
            max_timestep=64,
        )
        model = DecisionTransformer(cfg)
        states = torch.randn(3, 6, 8)
        rtg = torch.randn(3, 6)
        prev = torch.randint(low=0, high=6, size=(3, 6))
        t = torch.arange(0, 6).unsqueeze(0).repeat(3, 1)
        m = torch.ones(3, 6)
        logits = model(
            states=states,
            returns_to_go=rtg,
            prev_actions=prev,
            timesteps=t,
            attention_mask=m,
        )
        self.assertEqual(tuple(logits.shape), (3, 6, 5))

    def test_predict_next_action(self):
        cfg = DecisionTransformerConfig(
            state_dim=4,
            action_dim=3,
            context_len=4,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_timestep=32,
        )
        model = DecisionTransformer(cfg)
        states = torch.randn(1, 4, 4)
        rtg = torch.randn(1, 4)
        prev = torch.tensor([[3, 0, 1, 2]], dtype=torch.long)  # bos=3 by default.
        t = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        m = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        a, conf, probs = model.predict_next_action(
            states=states,
            returns_to_go=rtg,
            prev_actions=prev,
            timesteps=t,
            attention_mask=m,
            temperature=1.0,
            top_k=0,
            sample=False,
        )
        self.assertEqual(tuple(a.shape), (1,))
        self.assertEqual(tuple(conf.shape), (1,))
        self.assertEqual(tuple(probs.shape), (1, 3))
        self.assertGreaterEqual(int(a.item()), 0)
        self.assertLess(int(a.item()), 3)

    def test_checkpoint_roundtrip(self):
        cfg = DecisionTransformerConfig(
            state_dim=6,
            action_dim=4,
            context_len=5,
            d_model=24,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_timestep=64,
        )
        model = DecisionTransformer(cfg)
        with tempfile.TemporaryDirectory() as td:
            ckpt = f"{td}/dt.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.get_config(),
                },
                ckpt,
            )
            loaded, payload = load_decision_transformer_checkpoint(ckpt)
            self.assertEqual(int(loaded.get_config()["action_dim"]), 4)
            self.assertIn("model_state_dict", payload)


if __name__ == "__main__":
    unittest.main()
