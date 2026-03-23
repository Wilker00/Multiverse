import unittest

from core.types import VerseSpec
from integrations.isaaclab.adapter import (
    IsaacLabVerseAdapter,
    _normalize_isaac_action,
    _normalize_isaac_warehouse_obs,
)
from integrations.isaaclab.config import (
    DEFAULT_ISAACLAB_VERSE_BINDINGS,
    IsaacLabTaskConfig,
    get_isaaclab_binding,
)
from integrations.isaaclab.factory import (
    IsaacLabUnavailableError,
    build_isaaclab_install_hint,
    isaaclab_available,
    register_isaaclab_verses,
)
from integrations.sim_registry import provider_status


class _FakeSpace:
    def __init__(self, *, low, high, shape, dtype="float32"):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class _FakeIsaacEnv:
    def __init__(self):
        self.observation_space = object()
        self.action_space = _FakeSpace(low=[-1.0, -1.0], high=[1.0, 1.0], shape=(2,))
        self.closed = False
        self.step_calls = 0

    def reset(self, seed=None):
        del seed
        return (
            {
                "position": [1.0, 2.0, 0.5],
                "goal_position": [4.0, 6.0, 0.5],
                "velocity": [0.1, 0.0, 0.0],
                "lidar": [0.2] * 16,
                "battery": 0.9,
            },
            {"raw": "ok"},
        )

    def step(self, action):
        self.step_calls += 1
        return (
            {
                "position": [1.5, 2.0, 0.5],
                "goal_position": [4.0, 6.0, 0.5],
                "velocity": [0.2, 0.0, 0.0],
                "lidar": [0.3] * 16,
                "battery": 0.8,
            },
            1.25,
            False,
            self.step_calls >= 2,
            {"source_action": action},
        )

    def render(self):
        return [[1, 2], [3, 4]]

    def close(self):
        self.closed = True


class TestIsaacLabIntegration(unittest.TestCase):
    def test_install_hint_is_actionable(self):
        text = build_isaaclab_install_hint()
        self.assertIn("Isaac Sim", text)
        self.assertIn("optional", text)

    def test_binding_and_provider_status_expose_warehouse_lane(self):
        self.assertIn("isaac_warehouse_nav", DEFAULT_ISAACLAB_VERSE_BINDINGS)
        binding = get_isaaclab_binding("isaac_warehouse_nav")
        self.assertEqual(binding.task_id, "Isaac-Multiverse-Warehouse-Navigation-v0")
        status = provider_status("isaaclab")
        self.assertTrue(bool(status["implemented"]))
        self.assertIn("isaac_warehouse_nav", status["supported_verses"])

    def test_task_config_parses_common_params(self):
        cfg = IsaacLabTaskConfig.from_params(
            task_id="Isaac-Multiverse-Warehouse-Navigation-v0",
            params={"headless": False, "max_steps": 77, "scene": "warehouse"},
        )
        self.assertEqual(cfg.task_id, "Isaac-Multiverse-Warehouse-Navigation-v0")
        self.assertFalse(bool(cfg.headless))
        self.assertEqual(cfg.max_episode_length, 77)
        self.assertEqual(cfg.scene, "warehouse")

    def test_normalize_observation_and_action_contract(self):
        obs = _normalize_isaac_warehouse_obs(
            {
                "position": [1, 2, 3],
                "goal_position": [4, 6, 3],
                "velocity": [0.1, 0.2, 0.0],
                "lidar": [0.5] * 16,
                "battery": 0.7,
            }
        )
        self.assertEqual(obs["goal_delta"], [3.0, 4.0, 0.0])
        self.assertEqual(len(obs["flat"]), 26)
        binding = get_isaaclab_binding("isaac_warehouse_nav")
        action = _normalize_isaac_action([2.5, -4.0], action_space=binding.action_space)
        self.assertEqual(action, [1.0, -1.0])

    def test_register_hook_is_safe_without_dependency(self):
        if isaaclab_available():
            out = register_isaaclab_verses()
            self.assertIn("available_bindings", out)
            return
        with self.assertRaises(IsaacLabUnavailableError):
            register_isaaclab_verses()

    def test_adapter_works_with_injected_runtime_env(self):
        spec = VerseSpec(
            spec_version="v1",
            verse_name="isaac_warehouse_nav",
            verse_version="0.1",
            params={},
        )
        binding = get_isaaclab_binding("isaac_warehouse_nav")
        env = _FakeIsaacEnv()
        adapter = IsaacLabVerseAdapter(spec=spec, binding=binding, runtime_env=env)
        reset = adapter.reset()
        self.assertIn("position", reset.obs)
        self.assertEqual(reset.info["task_id"], binding.task_id)
        step = adapter.step([10.0, -10.0])
        self.assertAlmostEqual(step.reward, 1.25, places=6)
        self.assertFalse(bool(step.done))
        self.assertFalse(bool(step.truncated))
        step2 = adapter.step([0.0, 0.0])
        self.assertTrue(bool(step2.truncated))
        self.assertIsNotNone(adapter.render())
        adapter.close()
        self.assertTrue(bool(env.closed))


if __name__ == "__main__":
    unittest.main()
