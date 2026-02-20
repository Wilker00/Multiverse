import unittest
import tempfile

from core.agent_base import ExperienceBatch, Transition
from core.types import AgentSpec, SpaceSpec, VerseSpec
from memory.semantic_bridge import task_embedding_weights
from verses.registry import create_verse, register_builtin
from agents.sf_transfer_agent import SuccessorFeatureAgent
from agents.registry import create_agent, register_builtin_agents
import numpy as np


class TestSFTransferIntegration(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin()
        register_builtin_agents()

    def test_ego_grid_on_grid_world_and_warehouse_world(self) -> None:
        grid_spec = VerseSpec(
            spec_version="v1",
            verse_name="grid_world",
            verse_version="0.1",
            seed=123,
            params={"enable_ego_grid": True, "ego_grid_size": 5, "adr_enabled": False},
        )
        grid = create_verse(grid_spec)
        try:
            rr = grid.reset()
            obs = rr.obs if isinstance(rr.obs, dict) else {}
            self.assertIn("ego_grid", obs)
            ego = obs.get("ego_grid")
            self.assertIsInstance(ego, list)
            self.assertEqual(len(ego), 25)
            self.assertTrue(all(int(v) in (0, 1, 2) for v in ego))
            self.assertIn("ego_grid", (grid.observation_space.keys or []))
        finally:
            grid.close()

        wh_spec = VerseSpec(
            spec_version="v1",
            verse_name="warehouse_world",
            verse_version="0.1",
            seed=123,
            params={"enable_ego_grid": True, "ego_grid_size": 3, "adr_enabled": False},
        )
        wh = create_verse(wh_spec)
        try:
            rr = wh.reset()
            obs = rr.obs if isinstance(rr.obs, dict) else {}
            self.assertIn("ego_grid", obs)
            ego = obs.get("ego_grid")
            self.assertIsInstance(ego, list)
            self.assertEqual(len(ego), 9)
            self.assertTrue(all(int(v) in (0, 1, 2) for v in ego))
            self.assertIn("ego_grid", (wh.observation_space.keys or []))
        finally:
            wh.close()

    def test_semantic_bridge_task_embedding_profiles(self) -> None:
        safety = task_embedding_weights(target_verse_name="warehouse_world", profile="safety")
        goal = task_embedding_weights(target_verse_name="warehouse_world", profile="goal")
        self.assertLess(float(safety["obstacle"]), float(goal["obstacle"]))
        self.assertGreater(float(goal["goal"]), float(safety["goal"]))
        self.assertIn("battery", safety)

    def test_sf_transfer_agent_learns_and_exposes_diagnostics(self) -> None:
        spec = AgentSpec(
            spec_version="v1",
            policy_id="sf_test",
            policy_version="0.1",
            algo="sf_transfer",
            config={
                "verse_name": "warehouse_world",
                "use_semantic_w_init": True,
                "ego_grid_size": 3,
                "allowed_actions": [0, 1, 2, 3],
            },
        )
        obs_space = SpaceSpec(type="dict")
        act_space = SpaceSpec(type="discrete", n=5)
        ag = create_agent(spec=spec, observation_space=obs_space, action_space=act_space)
        self.assertIsInstance(ag, SuccessorFeatureAgent)

        o0 = {
            "x": 0,
            "y": 0,
            "goal_x": 2,
            "goal_y": 0,
            "battery": 20,
            "nearby_obstacles": 1,
            "ego_grid": [0, 1, 2, 0, 0, 0, 0, 0, 0],
        }
        o1 = {
            "x": 1,
            "y": 0,
            "goal_x": 2,
            "goal_y": 0,
            "battery": 19,
            "nearby_obstacles": 0,
            "ego_grid": [0, 0, 2, 0, 0, 0, 0, 0, 0],
        }
        o2 = {
            "x": 2,
            "y": 0,
            "goal_x": 2,
            "goal_y": 0,
            "battery": 18,
            "nearby_obstacles": 0,
            "ego_grid": [0, 0, 2, 0, 0, 0, 0, 0, 0],
        }

        d = ag.action_diagnostics(o0)
        self.assertLess(float(d.get("w_obstacle_mean", 0.0)), 0.0)
        self.assertGreater(float(d.get("w_goal_mean", 0.0)), 0.0)

        a = ag.act(o0).action
        self.assertIn(int(a), [0, 1, 2, 3])

        batch = ExperienceBatch(
            transitions=[
                Transition(obs=o0, action=3, reward=-0.1, next_obs=o1, done=False, truncated=False, info={}),
                Transition(obs=o1, action=3, reward=1.0, next_obs=o2, done=True, truncated=False, info={}),
            ]
        )
        m = ag.learn(batch)
        self.assertEqual(int(m.get("updates", 0)), 2)
        self.assertGreater(int(m.get("feature_dim", 0)), 0)
        self.assertIn("forward_mse_mean", m)

        with tempfile.TemporaryDirectory() as td:
            ckpt = f"{td}/sf_ckpt.json"
            ag.save(ckpt)
            ag2 = create_agent(spec=spec, observation_space=obs_space, action_space=act_space)
            self.assertIsInstance(ag2, SuccessorFeatureAgent)
            ag2.load(ckpt)
            assert isinstance(ag2, SuccessorFeatureAgent)
            self.assertIsNotNone(ag.forward_model)
            self.assertIsNotNone(ag2.forward_model)
            self.assertTrue(np.allclose(ag.forward_model, ag2.forward_model))


if __name__ == "__main__":
    unittest.main()
