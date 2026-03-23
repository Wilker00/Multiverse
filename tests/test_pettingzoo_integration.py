import unittest

from core.types import VerseSpec
from integrations.pettingzoo.adapter import (
    PettingZooUnavailableError,
    PettingZooVerseAdapter,
    _space_from_pettingzoo,
    build_pettingzoo_install_hint,
    pettingzoo_available,
)
from integrations.sim_registry import get_sim_provider, provider_status


class _FakeDiscreteSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        return 0


class _FakePettingZooAECEnv:
    """
    Minimal PettingZoo AEC-compatible fake environment.

    Two agents: "player_0" (focal) and "player_1" (auto-played).
    Each episode lasts up to 3 full cycles.
    """

    def __init__(self) -> None:
        self.possible_agents = ["player_0", "player_1"]
        self.agents: list = []
        self._agent_order = list(self.possible_agents)
        self._cycle_idx: int = 0
        self._round: int = 0
        self._max_rounds: int = 3
        self._obs: dict = {}
        self._rewards: dict = {}
        self._terminations: dict = {}
        self._truncations: dict = {}
        self._infos: dict = {}
        self.closed = False

        # Space accessors (callable form)
        self._action_spaces = {a: _FakeDiscreteSpace(n=4) for a in self.possible_agents}
        self._obs_spaces = {a: _FakeDiscreteSpace(n=8) for a in self.possible_agents}

    def observation_space(self, agent: str) -> _FakeDiscreteSpace:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> _FakeDiscreteSpace:
        return self._action_spaces[agent]

    @property
    def agent_selection(self) -> str:
        if not self.agents:
            return self.possible_agents[0]
        return self.agents[self._cycle_idx % len(self.agents)]

    def reset(self, seed=None) -> None:
        del seed
        self.agents = list(self.possible_agents)
        self._cycle_idx = 0
        self._round = 0
        for a in self.agents:
            self._obs[a] = 0
            self._rewards[a] = 0.0
            self._terminations[a] = False
            self._truncations[a] = False
            self._infos[a] = {}

    def last(self):
        agent = self.agent_selection
        return (
            self._obs.get(agent, 0),
            self._rewards.get(agent, 0.0),
            self._terminations.get(agent, False),
            self._truncations.get(agent, False),
            self._infos.get(agent, {}),
        )

    def step(self, action) -> None:
        agent = self.agent_selection
        # Assign a reward proportional to action (for testability)
        self._rewards[agent] = float(action) if action is not None else 0.0
        self._obs[agent] = (self._obs.get(agent, 0) + 1) % 8
        self._infos[agent] = {"action": action}

        # Advance cycle
        self._cycle_idx += 1

        # After a full round (all agents acted), increment round counter
        if self._cycle_idx % len(self.agents) == 0:
            self._round += 1
            if self._round >= self._max_rounds:
                for a in list(self.agents):
                    self._terminations[a] = True
                self.agents = []

    def render(self) -> str:
        return f"round={self._round}"

    def close(self) -> None:
        self.closed = True


class TestPettingZooAdapter(unittest.TestCase):
    def _make_spec(self, focal_agent: str = "") -> VerseSpec:
        params = {}
        if focal_agent:
            params["focal_agent"] = focal_agent
        return VerseSpec(
            spec_version="v1",
            verse_name="test_pz_env",
            verse_version="0.1",
            seed=7,
            params=params,
        )

    def test_install_hint_is_actionable(self) -> None:
        hint = build_pettingzoo_install_hint()
        self.assertIn("pettingzoo", hint.lower())
        self.assertIn("pip", hint.lower())

    def test_pettingzoo_available_returns_bool(self) -> None:
        result = pettingzoo_available()
        self.assertIsInstance(result, bool)

    def test_sim_registry_marks_pettingzoo_implemented(self) -> None:
        status = provider_status("pettingzoo")
        self.assertTrue(bool(status["implemented"]))
        self.assertIn("multi_agent", status["capabilities"])
        self.assertIn("adapter", status["capabilities"])

    def test_space_inference_discrete(self) -> None:
        space = _FakeDiscreteSpace(n=6)
        spec = _space_from_pettingzoo(space)
        self.assertEqual(spec.type, "discrete")
        self.assertEqual(spec.n, 6)

    def test_adapter_default_focal_agent_is_first(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec()
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        self.assertEqual(adapter._focal_agent, "player_0")

    def test_adapter_explicit_focal_agent(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec(focal_agent="player_1")
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        self.assertEqual(adapter._focal_agent, "player_1")

    def test_adapter_reset_returns_focal_obs(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec()
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        reset = adapter.reset()
        self.assertIsNotNone(reset.obs)
        self.assertEqual(reset.info["sim_provider"], "pettingzoo")
        self.assertEqual(reset.info["focal_agent"], "player_0")

    def test_adapter_step_advances_episode(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec()
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        adapter.reset()

        step = adapter.step(2)
        self.assertIsNotNone(step.obs)
        self.assertIsInstance(step.reward, float)
        self.assertFalse(bool(step.done))

    def test_adapter_episode_terminates(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec()
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        adapter.reset()

        done = False
        for _ in range(20):
            sr = adapter.step(1)
            if sr.done or sr.truncated:
                done = True
                break
        self.assertTrue(done)

    def test_adapter_export_import_state(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec()
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        adapter.reset()
        adapter.step(0)

        state = adapter.export_state()
        self.assertEqual(int(state["episode_step"]), 1)
        self.assertEqual(state["focal_agent"], "player_0")
        self.assertIsInstance(state["all_agents"], list)

        adapter.import_state({"episode_step": 55, "closed": False, "last_obs": 3, "all_agents": ["player_0"]})
        self.assertEqual(adapter._state.episode_step, 55)
        self.assertEqual(adapter._state.all_agents, ["player_0"])

    def test_adapter_close(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec()
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        adapter.close()
        self.assertTrue(bool(env.closed))
        self.assertTrue(bool(adapter._state.closed))

    def test_adapter_render(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec()
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        adapter.reset()
        frame = adapter.render()
        self.assertIsNotNone(frame)

    def test_action_spaces_inferred(self) -> None:
        env = _FakePettingZooAECEnv()
        spec = self._make_spec()
        adapter = PettingZooVerseAdapter(spec=spec, runtime_env=env)
        self.assertEqual(adapter.action_space.type, "discrete")
        self.assertEqual(adapter.action_space.n, 4)
        self.assertEqual(adapter.observation_space.type, "discrete")

    def test_unavailable_error_without_runtime_env(self) -> None:
        spec = self._make_spec()
        with self.assertRaises(PettingZooUnavailableError):
            PettingZooVerseAdapter(spec=spec)


class TestPettingZooRegistry(unittest.TestCase):
    def test_provider_spec_fields(self) -> None:
        provider = get_sim_provider("pettingzoo")
        self.assertEqual(provider.provider_id, "pettingzoo")
        self.assertTrue(bool(provider.external))
        self.assertTrue(bool(provider.implemented))
        self.assertIn("pettingzoo", provider.modules)
        self.assertIn("focal_agent", provider.capabilities)


if __name__ == "__main__":
    unittest.main()
