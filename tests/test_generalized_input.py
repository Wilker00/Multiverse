"""
tests/test_generalized_input.py

Verification of the DeepMind-style 'Universal Input'.
Demonstrates that a single MetaTransformer instance can process completely different
observation schemas (Chess vs Warehouse) without reconfiguration.
"""

import sys
import os
import torch
import unittest

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generalized_input import GeneralizedInputEncoder
from models.meta_transformer import MetaTransformer

class TestUniversalInput(unittest.TestCase):
    def test_cross_verse_processing(self):
        print("\n[DeepMind Eval] Testing Universal Input Architecture...")
        
        # 1. Initialize 'Universal' Agent
        # state_dim is ignored (set to 1) when use_generalized_input=True
        agent = MetaTransformer(
            state_dim=1,  
            action_dim=10, 
            n_embd=64,
            use_generalized_input=True
        )
        agent.eval()
        
        # 2. Schema A: Chess-like Observation
        chess_obs = [
            {"material": 5.0, "king_safety": 1.0, "center_control": 0.5},
            {"material": -2.0, "king_safety": 0.0, "center_control": 0.1}
        ]
        
        # 3. Schema B: Warehouse-like Observation
        warehouse_obs = [
            {"x": 10.0, "y": 5.0, "battery": 20.0, "load": 1.0},
            {"x": 2.0, "y": 2.0, "battery": 5.0, "load": 0.0}
        ]
        
        print(f"  Agent initialized. Processing {len(chess_obs)} Chess states...")
        with torch.no_grad():
            out_chess = agent.predict(raw_obs=chess_obs)
            print(f"  Chess Logits Shape: {out_chess['probs'].shape}")
            self.assertEqual(out_chess['probs'].shape, (2, 10))
            
        print(f"  Processing {len(warehouse_obs)} Warehouse states (different schema)...")
        with torch.no_grad():
            out_warehouse = agent.predict(raw_obs=warehouse_obs)
            print(f"  Warehouse Logits Shape: {out_warehouse['probs'].shape}")
            self.assertEqual(out_warehouse['probs'].shape, (2, 10))
            
        # 4. Mixed Batch (The ultimate test: distinct schemas in same batch)
        # GeneralizedInputEncoder handles this naturally by padding keys
        mixed_obs = [chess_obs[0], warehouse_obs[0]]
        print(f"  Processing Mixed Batch (Chess + Warehouse)...")
        with torch.no_grad():
            out_mixed = agent.predict(raw_obs=mixed_obs)
            print(f"  Mixed Logits Shape: {out_mixed['probs'].shape}")
            self.assertEqual(out_mixed['probs'].shape, (2, 10))

        print("[SUCCESS] Universal Input validated. The agent is schema-agnostic.")

    def test_encoder_handles_nested_and_empty_observations(self):
        enc = GeneralizedInputEncoder(n_embd=32, max_keys=16, n_layers=1)
        enc.eval()

        obs_batch = [
            {"a": {"x": 1, "y": [2, 3]}, "b": "ignore_me"},
            {},
            {"values": [1.5, -2.0, {"z": 0.25}]},
        ]
        with torch.no_grad():
            out = enc(obs_batch)
        self.assertEqual(tuple(out.shape), (3, 32))

    def test_encoder_is_order_stable_for_equivalent_dicts(self):
        enc = GeneralizedInputEncoder(n_embd=32, max_keys=16, n_layers=1)
        enc.eval()

        obs_a = {"k1": 1.0, "k2": 2.0, "nested": {"x": 3.0, "y": 4.0}}
        obs_b = {"nested": {"y": 4.0, "x": 3.0}, "k2": 2.0, "k1": 1.0}

        with torch.no_grad():
            za = enc([obs_a])[0]
            zb = enc([obs_b])[0]
        self.assertTrue(torch.allclose(za, zb, atol=1e-6))

if __name__ == "__main__":
    unittest.main()
