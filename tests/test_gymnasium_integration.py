import unittest

from core.types import VerseSpec
from integrations.gymnasium.adapter import (
    GymnasiumUnavailableError,
    GymnasiumVerseAdapter,
    _space_from_gymnasium,
    _to_jsonable,
    build_gymnasium_install_hint,
    gymnasium_available,
)
from integrations.sim_registry import get_sim_provider, provider_status


class _FakeDiscreteSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        return 0


class _FakeContinuousSpace:
    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high
        self.shape = (len(low),)
        self.dtype = "float32"

    def sample(self):
        return [0.0] * len(self.low)


class _FakeGymEnv:
    """Minimal Gymnasium-compatible fake env."""

    def __init__(self, *, obs_size: int = 4, n_actions: int = 2) -> None:
        self.observation_space = _FakeContinuousSpace(
            low=[-1.0] * obs_size, high=[1.0] * obs_size
        )
        self.action_space = _FakeDiscreteSpace(n=n_actions)
        self.closed = False
        self._step_count = 0

    def reset(self, seed=None):
        del seed
        self._step_count = 0
        return ([0.1] * self.observation_space.shape[0], {"reset": True})

    def step(self, action):
        self._step_count += 1
        obs = [float(action) * 0.1] * self.observation_space.shape[0]
        reward = 1.0
        terminated = self._step_count >= 3
        truncated = False
        info = {"step": self._step_count}
        return (obs, reward, terminated, truncated, info)

    def render(self):
        return [[0, 1], [2, 3]]

    def close(self):
        self.closed = True


class TestGymnasiumAdapter(unittest.TestCase):
    def _make_spec(self, env_id: str = "CartPole-v1") -> VerseSpec:
        return VerseSpec(
            spec_version="v1",
            verse_name=env_id,
            verse_version="0.1",
            seed=42,
            params={"env_id": env_id},
        )

    def test_install_hint_is_actionable(self) -> None:
        hint = build_gymnasium_install_hint()
        self.assertIn("gymnasium", hint.lower())
        self.assertIn("pip", hint.lower())

    def test_gymnasium_available_returns_bool(self) -> None:
        result = gymnasium_available()
        self.assertIsInstance(result, bool)

    def test_sim_registry_marks_gymnasium_implemented(self) -> None:
        status = provider_status("gymnasium")
        self.assertTrue(bool(status["implemented"]))
        self.assertIn("adapter", status["capabilities"])
        self.assertIn("benchmark_friendly", status["capabilities"])

    def test_space_inference_discrete(self) -> None:
        space = _FakeDiscreteSpace(n=5)
        spec = _space_from_gymnasium(space)
        self.assertEqual(spec.type, "discrete")
        self.assertEqual(spec.n, 5)

    def test_space_inference_continuous(self) -> None:
        space = _FakeContinuousSpace(low=[-1.0, -2.0], high=[1.0, 2.0])
        spec = _space_from_gymnasium(space)
        self.assertEqual(spec.type, "continuous")
        self.assertEqual(spec.shape, (2,))
        self.assertEqual(spec.low, [-1.0, -2.0])
        self.assertEqual(spec.high, [1.0, 2.0])

    def test_to_jsonable_handles_nested(self) -> None:
        result = _to_jsonable({"a": [1, 2.0, True, None, "x"]})
        self.assertEqual(result, {"a": [1, 2.0, True, None, "x"]})

    def test_adapter_reset_and_step_with_injected_env(self) -> None:
        fake = _FakeGymEnv(obs_size=4, n_actions=2)
        spec = self._make_spec("CartPole-v1")
        adapter = GymnasiumVerseAdapter(spec=spec, runtime_env=fake)

        reset = adapter.reset()
        self.assertIsInstance(reset.obs, list)
        self.assertEqual(len(reset.obs), 4)
        self.assertEqual(reset.info["sim_provider"], "gymnasium")
        self.assertEqual(reset.info["env_id"], "CartPole-v1")

        step = adapter.step(1)
        self.assertIsInstance(step.obs, list)
        self.assertAlmostEqual(step.reward, 1.0, places=6)
        self.assertFalse(bool(step.done))
        self.assertFalse(bool(step.truncated))

    def test_adapter_terminates_after_max_steps(self) -> None:
        fake = _FakeGymEnv(obs_size=4, n_actions=2)
        spec = self._make_spec()
        adapter = GymnasiumVerseAdapter(spec=spec, runtime_env=fake)
        adapter.reset()

        done = False
        for _ in range(10):
            sr = adapter.step(0)
            if sr.done or sr.truncated:
                done = True
                break
        self.assertTrue(done)

    def test_adapter_observation_and_action_spaces_inferred(self) -> None:
        fake = _FakeGymEnv(obs_size=4, n_actions=3)
        spec = self._make_spec()
        adapter = GymnasiumVerseAdapter(spec=spec, runtime_env=fake)
        self.assertEqual(adapter.observation_space.type, "continuous")
        self.assertEqual(adapter.action_space.type, "discrete")
        self.assertEqual(adapter.action_space.n, 3)

    def test_adapter_render_returns_value(self) -> None:
        fake = _FakeGymEnv()
        spec = self._make_spec()
        adapter = GymnasiumVerseAdapter(spec=spec, runtime_env=fake)
        adapter.reset()
        frame = adapter.render()
        self.assertIsNotNone(frame)

    def test_adapter_export_import_state(self) -> None:
        fake = _FakeGymEnv()
        spec = self._make_spec()
        adapter = GymnasiumVerseAdapter(spec=spec, runtime_env=fake)
        adapter.reset()
        adapter.step(0)
        adapter.step(0)

        state = adapter.export_state()
        self.assertEqual(int(state["episode_step"]), 2)
        self.assertFalse(bool(state["closed"]))

        adapter.import_state({"episode_step": 99, "closed": False, "last_obs": [0.0, 0.0, 0.0, 0.0]})
        self.assertEqual(adapter._state.episode_step, 99)

    def test_adapter_close_sets_closed(self) -> None:
        fake = _FakeGymEnv()
        spec = self._make_spec()
        adapter = GymnasiumVerseAdapter(spec=spec, runtime_env=fake)
        adapter.close()
        self.assertTrue(bool(fake.closed))
        self.assertTrue(bool(adapter._state.closed))

    def test_unavailable_error_raised_without_runtime_env_when_not_installed(self) -> None:
        if gymnasium_available():
            self.skipTest("gymnasium is installed; unavailability cannot be tested")
        spec = self._make_spec()
        with self.assertRaises(GymnasiumUnavailableError):
            GymnasiumVerseAdapter(spec=spec)


class TestGymnasiumRegistry(unittest.TestCase):
    def test_provider_spec_fields(self) -> None:
        provider = get_sim_provider("gymnasium")
        self.assertEqual(provider.provider_id, "gymnasium")
        self.assertTrue(bool(provider.external))
        self.assertTrue(bool(provider.implemented))
        self.assertIn("gymnasium", provider.modules)

    def test_local_provider_now_lists_all_builtin_verses(self) -> None:
        provider = get_sim_provider("multiverse_local")
        self.assertIn("warehouse_world", provider.supported_verses)
        self.assertIn("labyrinth_world", provider.supported_verses)
        self.assertIn("bridge_world", provider.supported_verses)
        self.assertIn("trade_world", provider.supported_verses)
        self.assertIn("maze_world", provider.supported_verses)
        self.assertGreater(len(provider.supported_verses), 10)


if __name__ == "__main__":
    unittest.main()
