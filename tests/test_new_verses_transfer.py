
import unittest
import json
from core.types import JSONValue
from core.taxonomy import can_bridge
from memory.semantic_bridge import translate_action, translate_observation

class TestNewVersesTransfer(unittest.TestCase):

    def test_new_bridge_permissions(self):
        # 1D analogy
        self.assertTrue(can_bridge("bridge_world", "line_world"), "bridge <-> line should be allowed")
        self.assertTrue(can_bridge("line_world", "bridge_world"))
        
        # Resource flow analogy
        self.assertTrue(can_bridge("factory_world", "harvest_world"), "factory <-> harvest should be allowed")
        self.assertTrue(can_bridge("harvest_world", "factory_world"))
        
        # Economic accumulation analogy
        self.assertTrue(can_bridge("trade_world", "harvest_world"), "trade <-> harvest should be allowed")
        self.assertTrue(can_bridge("harvest_world", "trade_world"))
        
        # Inventory management analogy
        self.assertTrue(can_bridge("trade_world", "factory_world"), "trade <-> factory should be allowed")
        self.assertTrue(can_bridge("factory_world", "trade_world"))
        
        # Strategy projections
        self.assertTrue(can_bridge("chess_world", "escape_world"))
        self.assertTrue(can_bridge("chess_world", "factory_world"))
        self.assertTrue(can_bridge("chess_world", "bridge_world"))
        self.assertTrue(can_bridge("chess_world", "trade_world"))

    def test_bridge_line_transfer(self):
        # Bridge -> Line
        bridge_obs = {"cursor": 3, "segments_placed": 3, "weak_count": 0, "bridge_complete": 0, "t": 10}
        line_obs = translate_observation(obs=bridge_obs, source_verse_name="bridge_world", target_verse_name="line_world")
        self.assertIsNotNone(line_obs)
        self.assertEqual(line_obs.get("pos"), 3)
        self.assertEqual(line_obs.get("goal"), 8)

        # Line -> Bridge
        line_obs_src = {"pos": 5, "goal": 10, "t": 15}
        bridge_obs_res = translate_observation(obs=line_obs_src, source_verse_name="line_world", target_verse_name="bridge_world")
        self.assertIsNotNone(bridge_obs_res)
        self.assertEqual(bridge_obs_res.get("cursor"), 5)
        self.assertEqual(bridge_obs_res.get("segments_placed"), 5)

    def test_factory_harvest_transfer(self):
        # Factory -> Harvest
        fact_obs = {"t": 10, "completed": 7, "total_arrived": 12, "buf_0": 3}
        harv_obs = translate_observation(obs=fact_obs, source_verse_name="factory_world", target_verse_name="harvest_world")
        self.assertIsNotNone(harv_obs)
        self.assertEqual(harv_obs.get("deposited"), 7) # completed maps to deposited
        self.assertEqual(harv_obs.get("nearby_fruit"), 3) # buf_0 maps to nearby_fruit

        # Harvest -> Factory
        harv_obs_src = {"t": 20, "carrying": 2, "deposited": 15, "fruit_remaining": 85}
        fact_obs_res = translate_observation(obs=harv_obs_src, source_verse_name="harvest_world", target_verse_name="factory_world")
        self.assertIsNotNone(fact_obs_res)
        self.assertEqual(fact_obs_res.get("completed"), 15) # deposited maps to completed

    def test_trade_harvest_transfer(self):
        # Trade -> Harvest
        trade_obs = {"t": 5, "price": 10.0, "cash": 120.0, "inventory": 4, "portfolio_value": 160.0, "total_profit": 20.0}
        harv_obs = translate_observation(obs=trade_obs, source_verse_name="trade_world", target_verse_name="harvest_world")
        self.assertIsNotNone(harv_obs)
        self.assertEqual(harv_obs.get("carrying"), 4) # inventory maps to carrying
        self.assertEqual(harv_obs.get("deposited"), 10) # profit/2 maps to deposited

        # Harvest -> Trade
        harv_obs_src = {"t": 8, "carrying": 5, "deposited": 10}
        trade_obs_res = translate_observation(obs=harv_obs_src, source_verse_name="harvest_world", target_verse_name="trade_world")
        self.assertIsNotNone(trade_obs_res)
        self.assertEqual(trade_obs_res.get("inventory"), 5)

    def test_strategy_projections(self):
        strategy_obs = {
            "score_delta": 5, # Progress/Score
            "risk": 2,        # Risk/Volatility
            "control": 6,     # Control/Inventory/Development
            "pressure": 3,
            "tempo": 1,
            "resource": 4,
            "t": 12
        }

        # Strategy -> Escape
        escape_obs = translate_observation(obs=strategy_obs, source_verse_name="chess_world", target_verse_name="escape_world")
        self.assertIsNotNone(escape_obs)
        # Score 5 -> prog ~ 10/15 = 0.66 -> x ~ 6
        self.assertIn("x", escape_obs)
        self.assertIn("nearest_guard_dist", escape_obs)

        # Strategy -> Factory
        factory_obs = translate_observation(obs=strategy_obs, source_verse_name="chess_world", target_verse_name="factory_world")
        self.assertIsNotNone(factory_obs)
        self.assertIn("completed", factory_obs)
        self.assertIn("buf_0", factory_obs)

        # Strategy -> Bridge
        bridge_obs = translate_observation(obs=strategy_obs, source_verse_name="chess_world", target_verse_name="bridge_world")
        self.assertIsNotNone(bridge_obs)
        self.assertIn("cursor", bridge_obs)
        self.assertIn("strong_count", bridge_obs)

        # Strategy -> Trade
        trade_obs = translate_observation(obs=strategy_obs, source_verse_name="chess_world", target_verse_name="trade_world")
        self.assertIsNotNone(trade_obs)
        self.assertIn("price", trade_obs)
        self.assertIn("cash", trade_obs)
        self.assertTrue(trade_obs["cash"] > 100.0) # check positive mapping

    def test_action_translation(self):
        # Strategy -> Escape
        # 0 (build) -> 3 (right/progress)
        act = translate_action(action=0, source_verse_name="chess_world", target_verse_name="escape_world")
        self.assertEqual(act, 3)

        # Strategy -> Factory
        # 3 (defend) -> 3 (repair)
        act = translate_action(action=3, source_verse_name="chess_world", target_verse_name="factory_world")
        self.assertEqual(act, 3)
        
        # Strategy -> Bridge
        # 2 (capture) -> 3 (cross/aggression)
        act = translate_action(action=2, source_verse_name="chess_world", target_verse_name="bridge_world")
        self.assertEqual(act, 3)

        # Strategy -> Trade
        # 5 (convert) -> 1 (sell)
        act = translate_action(action=5, source_verse_name="chess_world", target_verse_name="trade_world")
        self.assertEqual(act, 1)

if __name__ == "__main__":
    unittest.main()
