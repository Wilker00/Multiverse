import unittest

import torch

from models.meta_transformer import MetaTransformer


class TestMetaTransformerValue(unittest.TestCase):
    def test_policy_and_value_heads(self):
        model = MetaTransformer(state_dim=5, action_dim=4, n_embd=32, context_layers=1, dropout=0.0)
        state = torch.randn(3, 5)
        history = torch.randn(3, 2, 7)  # state_dim + 2

        logits = model(state, history)
        self.assertEqual(tuple(logits.shape), (3, 4))

        out = model.forward_policy_value(state, history)
        self.assertIn("logits", out)
        self.assertIn("value", out)
        self.assertEqual(tuple(out["logits"].shape), (3, 4))
        self.assertEqual(tuple(out["value"].shape), (3,))

        pred = model.predict(state=state[:1], recent_history=history[:1])
        self.assertIn("value", pred)
        self.assertEqual(int(pred["action"].shape[0]), 1)


if __name__ == "__main__":
    unittest.main()
