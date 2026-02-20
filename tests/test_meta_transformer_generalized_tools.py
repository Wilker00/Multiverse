import os
import tempfile
import unittest

import torch

from models.meta_transformer import MetaTransformer
from tools.benchmark_meta_stages import _evaluate_checkpoint
from tools.train_meta_transformer import _build_generalized_samples, _generalized_batch


def _synthetic_rows() -> list[dict]:
    rows = []
    step = 0
    for ep in range(2):
        for t in range(4):
            rows.append(
                {
                    "run_id": "run_x",
                    "episode_id": str(ep),
                    "step_idx": step,
                    "obs": {
                        "agent": {"x": t, "y": ep},
                        "battery": float(10 - t),
                        "items": [t, t + 1],
                    },
                    "action": int(t % 3),
                    "reward": float(1.0 if t == 3 else -0.1),
                }
            )
            step += 1
    return rows


class TestMetaTransformerGeneralizedTools(unittest.TestCase):
    def test_generalized_sample_builder_and_batch(self):
        rows = _synthetic_rows()
        samples, action_dim = _build_generalized_samples(rows, history_len=2, gamma=0.99)
        self.assertGreater(len(samples), 0)
        self.assertGreaterEqual(action_dim, 3)

        model = MetaTransformer(
            state_dim=1,
            action_dim=action_dim,
            n_embd=32,
            use_generalized_input=True,
        )
        batch = samples[:3]
        raw_obs, t_hist, t_y, t_v = _generalized_batch(model=model, samples=batch, history_len=2)
        self.assertEqual(len(raw_obs), 3)
        self.assertEqual(tuple(t_hist.shape), (3, 2, 34))
        self.assertEqual(tuple(t_y.shape), (3,))
        self.assertEqual(tuple(t_v.shape), (3,))

        out = model.forward_policy_value(state=None, recent_history=t_hist, raw_obs=raw_obs)
        self.assertEqual(tuple(out["logits"].shape), (3, action_dim))
        self.assertEqual(tuple(out["value"].shape), (3,))

    def test_benchmark_eval_checkpoint_generalized(self):
        rows = _synthetic_rows()
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "meta_generalized.pt")
            model = MetaTransformer(
                state_dim=1,
                action_dim=3,
                n_embd=32,
                use_generalized_input=True,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.get_config(),
                    "history_len": 2,
                    "training_stage": "policy_value",
                },
                ckpt_path,
            )

            report = _evaluate_checkpoint(ckpt_path, rows, gamma=0.99, eval_limit=0)
            self.assertTrue(bool(report.get("use_generalized_input", False)))
            self.assertGreater(int(report.get("examples", 0)), 0)
            self.assertIn("action_accuracy", report)
            self.assertIn("value_mse", report)


if __name__ == "__main__":
    unittest.main()
