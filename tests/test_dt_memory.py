import os
import tempfile
import unittest

import torch

from agents.transformer_agent import TransformerAgent
from core.types import AgentSpec, SpaceSpec


class TestDTMemory(unittest.TestCase):
    def test_memory_hooks(self):
        with tempfile.TemporaryDirectory() as td:
            d_model = 16
            from models.decision_transformer import DecisionTransformer, DecisionTransformerConfig

            cfg = DecisionTransformerConfig(
                state_dim=4,
                action_dim=2,
                context_len=2,
                d_model=d_model,
                n_head=1,
                n_layer=1,
                max_timestep=100,
            )
            model = DecisionTransformer(cfg)
            ckpt_path = os.path.join(td, "dt.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.get_config(),
                },
                ckpt_path,
            )

            spec = AgentSpec(
                spec_version="v1",
                policy_id="adt_mem",
                policy_version="0.1",
                algo="adt",
                config={
                    "model_path": ckpt_path,
                    "recall_enabled": True,
                    "recall_risk_threshold": 0.5,
                },
            )
            agent = TransformerAgent(
                spec=spec,
                observation_space=SpaceSpec("dict"),
                action_space=SpaceSpec("discrete", n=2),
            )

            obs = {"x": 1, "risk": 0.8}
            req = agent.memory_query_request(obs=obs, step_idx=10)
            self.assertIsNotNone(req)
            self.assertEqual(req["reason"], "high_risk")

            hint = {
                "memory_recall": {
                    "matches": [
                        {"action": 1, "score": 0.9},
                    ]
                }
            }
            res = agent.act_with_hint(obs, hint=hint)
            self.assertEqual(res.info.get("memory_recall_eligible"), True)
            agent.close()
