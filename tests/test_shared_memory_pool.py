import unittest

from core.communication import SharedMemoryPool


class TestSharedMemoryPool(unittest.TestCase):
    def test_trajectory_trade_and_sampling(self):
        pool = SharedMemoryPool(max_total_trajectories=10, max_per_verse=10)
        pool.publish_trajectory(
            provider_agent_id="agent_a",
            verse_name="line_world",
            transitions=[{"obs": {"x": 0}, "action": 1, "reward": 0.2, "next_obs": {"x": 1}, "done": False}],
            return_sum=1.2,
            success=True,
            episode_id="ep1",
        )
        pool.publish_trajectory(
            provider_agent_id="agent_b",
            verse_name="line_world",
            transitions=[{"obs": {"x": 1}, "action": 1, "reward": 0.1, "next_obs": {"x": 2}, "done": True}],
            return_sum=0.8,
            success=True,
            episode_id="ep2",
        )

        offers = pool.sample_trajectories(consumer_agent_id="agent_a", verse_name="line_world", top_k=3)
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0]["provider_agent_id"], "agent_b")

    def test_safety_contract_and_lexicon(self):
        pool = SharedMemoryPool(lexicon_consensus_floor=0.6)
        pool.propose_safety_boundary(
            agent_id="agent_a",
            verse_name="chess_world",
            risk_budget=0.30,
            veto_bias=0.70,
            confidence=0.8,
        )
        pool.propose_safety_boundary(
            agent_id="agent_b",
            verse_name="chess_world",
            risk_budget=0.10,
            veto_bias=0.90,
            confidence=0.7,
        )
        contract = pool.safety_contract(verse_name="chess_world")
        self.assertTrue(contract["has_contract"])
        self.assertGreater(contract["risk_budget"], 0.0)
        self.assertGreater(contract["veto_bias"], 0.0)

        pool.record_token(agent_id="agent_a", concept="risk", token="redline", confidence=0.8)
        pool.record_token(agent_id="agent_b", concept="risk", token="redline", confidence=0.7)
        pool.record_token(agent_id="agent_c", concept="risk", token="amber", confidence=0.95)
        pool.record_token(agent_id="agent_c", concept="success", token="greenlight", confidence=0.9)
        lex = pool.lexicon(min_support=2)
        self.assertIn("risk", lex)
        self.assertEqual(lex["risk"]["token"], "redline")
        self.assertGreater(float(lex["risk"].get("consensus_ratio", 0.0)), 0.6)

    def test_provider_quota_limits_single_agent_domination(self):
        pool = SharedMemoryPool(max_total_trajectories=20, max_per_verse=10, max_provider_share=0.3)
        for i in range(12):
            pool.publish_trajectory(
                provider_agent_id="dominant",
                verse_name="line_world",
                transitions=[{"obs": {"x": i}, "action": 1, "reward": 0.2, "next_obs": {"x": i + 1}, "done": True}],
                return_sum=2.0,
                success=True,
                episode_id=f"dom_{i}",
            )
        for i in range(6):
            pool.publish_trajectory(
                provider_agent_id=f"ally_{i}",
                verse_name="line_world",
                transitions=[{"obs": {"x": i}, "action": 1, "reward": 0.1, "next_obs": {"x": i + 1}, "done": True}],
                return_sum=1.0,
                success=True,
                episode_id=f"ally_{i}",
            )
        offers = pool.sample_trajectories(consumer_agent_id="consumer", verse_name="line_world", top_k=20)
        dominant = sum(1 for o in offers if str(o.get("provider_agent_id")) == "dominant")
        self.assertLessEqual(dominant, 3)

    def test_trust_weighting_prefers_reliable_provider(self):
        pool = SharedMemoryPool(max_total_trajectories=50, max_per_verse=50, trajectory_half_life_hours=1000.0)
        for i in range(8):
            pool.publish_trajectory(
                provider_agent_id="reliable",
                verse_name="grid_world",
                transitions=[{"obs": {"x": i}, "action": 1, "reward": 0.4, "next_obs": {"x": i + 1}, "done": True}],
                return_sum=3.0,
                success=True,
                episode_id=f"rel_{i}",
            )
        for i in range(8):
            pool.publish_trajectory(
                provider_agent_id="noisy",
                verse_name="grid_world",
                transitions=[{"obs": {"x": i}, "action": 1, "reward": -0.6, "next_obs": {"x": i + 1}, "done": True}],
                return_sum=4.0 if i == 7 else -2.0,
                success=False if i < 7 else True,
                episode_id=f"noisy_{i}",
            )
        offers = pool.sample_trajectories(consumer_agent_id="consumer", verse_name="grid_world", top_k=1)
        self.assertEqual(len(offers), 1)
        self.assertEqual(str(offers[0].get("provider_agent_id")), "reliable")
        self.assertGreater(float(offers[0].get("provider_trust", 0.0)), 0.5)


if __name__ == "__main__":
    unittest.main()
