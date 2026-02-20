import unittest
import tempfile

from agents.aware_agent import AwareAgent
from agents.evolving_agent import EvolvingAgent
from agents.q_agent import QLearningAgent
from agents.registry import _AGENT_REGISTRY, register_builtin_agents
from core.agent_base import ExperienceBatch, Transition
from core.types import AgentSpec, SpaceSpec


def _obs_space() -> SpaceSpec:
    return SpaceSpec(type="dict")


def _act_space(n: int = 4) -> SpaceSpec:
    return SpaceSpec(type="discrete", n=n)


def _agent_spec(algo: str, config=None) -> AgentSpec:
    return AgentSpec(
        spec_version="v1",
        policy_id=f"test_{algo}",
        policy_version="0.1",
        algo=algo,
        seed=123,
        config=dict(config or {}),
    )


def _batch(return_sum: float = 0.0) -> ExperienceBatch:
    tr = Transition(
        obs={"x": 0},
        action=0,
        reward=-1.0,
        next_obs={"x": 1},
        done=False,
        truncated=False,
        info={},
    )
    return ExperienceBatch(transitions=[tr], meta={"return_sum": float(return_sum)})


class TestAwareEvolvingAgents(unittest.TestCase):
    def test_registry_contains_aware_and_evolving(self):
        register_builtin_agents()
        self.assertIn("aware", _AGENT_REGISTRY)
        self.assertIn("evolving", _AGENT_REGISTRY)

    def test_aware_cliff_defaults_applied(self):
        spec = _agent_spec("aware", {"verse_name": "cliff_world"})
        agent = AwareAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(4))
        self.assertAlmostEqual(agent.lr, 0.08, places=6)
        self.assertAlmostEqual(agent.gamma, 0.98, places=6)
        self.assertAlmostEqual(agent.epsilon_decay, 0.997, places=6)
        self.assertAlmostEqual(agent._performance_floor, -20.0, places=6)
        self.assertAlmostEqual(agent.learn_success_bonus, 2.0, places=6)
        self.assertAlmostEqual(agent.learn_hazard_penalty, 2.0, places=6)

    def test_aware_user_override_wins_over_tuned_default(self):
        spec = _agent_spec("aware", {"verse_name": "labyrinth_world", "gamma": 0.91})
        agent = AwareAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(5))
        self.assertAlmostEqual(agent.gamma, 0.91, places=6)
        self.assertEqual(agent._awareness_window, 40)

    def test_evolving_labyrinth_defaults_applied(self):
        spec = _agent_spec("evolving", {"verse_name": "labyrinth_world"})
        agent = EvolvingAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(5))
        self.assertEqual(agent._evolve_interval, 20)
        self.assertAlmostEqual(agent._evolve_margin, 0.40, places=6)
        self.assertAlmostEqual(agent._mutation_scale, 0.12, places=6)
        self.assertAlmostEqual(agent.learn_success_bonus, 0.0, places=6)
        self.assertAlmostEqual(agent.learn_hazard_penalty, 0.0, places=6)

    def test_evolving_plateau_window_triggers_mutation(self):
        spec = _agent_spec(
            "evolving",
            {
                "verse_name": "line_world",
                "evolve_interval": 2,
                "evolve_margin": 100.0,
                "mutation_scale": 0.0,
                "gamma_mutation": 0.0,
                "epsilon_decay_mutation": 0.0,
            },
        )
        agent = EvolvingAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(2))
        agent.seed(7)

        m1 = agent.learn(_batch(return_sum=0.0))
        m2 = agent.learn(_batch(return_sum=0.0))
        m3 = agent.learn(_batch(return_sum=0.0))
        m4 = agent.learn(_batch(return_sum=0.0))

        self.assertEqual(m1.get("evolution_event_code"), 0.0)
        self.assertEqual(m2.get("evolution_event_code"), 1.0)  # first full window is baseline
        self.assertEqual(m3.get("evolution_event_code"), 0.0)
        self.assertEqual(m4.get("evolution_event_code"), 2.0)  # plateau window mutates
        self.assertEqual(m4.get("evolution_generation"), 1.0)

    def test_q_learning_hazard_shaping_penalizes_risky_transition(self):
        shaped = QLearningAgent(
            spec=_agent_spec("q", {"learn_hazard_penalty": 5.0, "lr": 0.1, "gamma": 0.99}),
            observation_space=_obs_space(),
            action_space=_act_space(2),
        )
        plain = QLearningAgent(
            spec=_agent_spec("q", {"learn_hazard_penalty": 0.0, "lr": 0.1, "gamma": 0.99}),
            observation_space=_obs_space(),
            action_space=_act_space(2),
        )

        tr = Transition(
            obs={"x": 0},
            action=0,
            reward=-1.0,
            next_obs={"x": 1},
            done=True,
            truncated=False,
            info={"env_info": {"fell_cliff": True}},
        )
        batch = ExperienceBatch(transitions=[tr], meta={"return_sum": -1.0})
        shaped.learn(batch)
        plain.learn(batch)

        q_shaped = float(shaped.q['{"x":0}'][0])
        q_plain = float(plain.q['{"x":0}'][0])
        self.assertLess(q_shaped, q_plain)

    def test_aware_memory_guides_unseen_state(self):
        spec = _agent_spec(
            "aware",
            {
                "epsilon_start": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 1.0,
                "aware_memory_prior_weight": 1.0,
                "aware_memory_min_context_samples": 1,
                "aware_memory_explore_bias": 1.0,
            },
        )
        agent = AwareAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(5))
        agent.seed(11)

        for _ in range(5):
            tr = Transition(
                obs={"x": 1, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 15, "nearby_obstacles": 0, "t": 0},
                action=3,
                reward=0.4,
                next_obs={"x": 2, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 14, "nearby_obstacles": 0, "t": 1},
                done=False,
                truncated=False,
                info={},
            )
            agent.learn(ExperienceBatch(transitions=[tr], meta={"return_sum": 0.4}))

        unseen_but_similar = {"x": 2, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 13, "nearby_obstacles": 0, "t": 50}
        act = agent.act(unseen_but_similar)
        self.assertEqual(int(act.action), 3)
        diag = agent.action_diagnostics(unseen_but_similar)
        probs = diag.get("sample_probs")
        self.assertIsInstance(probs, list)
        assert isinstance(probs, list)
        self.assertEqual(int(max(range(len(probs)), key=lambda i: float(probs[i]))), 3)

    def test_aware_memory_persists_across_save_load(self):
        spec = _agent_spec(
            "aware",
            {
                "epsilon_start": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 1.0,
                "aware_memory_prior_weight": 1.0,
                "aware_memory_min_context_samples": 1,
            },
        )
        agent = AwareAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(5))
        agent.seed(21)
        tr = Transition(
            obs={"x": 0, "y": 0, "goal_x": 7, "goal_y": 7, "battery": 15, "nearby_obstacles": 1, "t": 0},
            action=0,
            reward=0.2,
            next_obs={"x": 0, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 14, "nearby_obstacles": 1, "t": 1},
            done=False,
            truncated=False,
            info={},
        )
        agent.learn(ExperienceBatch(transitions=[tr], meta={"return_sum": 0.2}))

        with tempfile.TemporaryDirectory() as td:
            agent.save(td)
            reloaded = AwareAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(5))
            reloaded.load(td)
            obs = {"x": 0, "y": 1, "goal_x": 7, "goal_y": 7, "battery": 14, "nearby_obstacles": 1, "t": 99}
            act = reloaded.act(obs)
            self.assertEqual(int(act.action), 0)

    def test_aware_social_contract_caps_exploration(self):
        spec = _agent_spec(
            "aware",
            {
                "epsilon_start": 0.8,
                "epsilon_min": 0.8,
                "epsilon_decay": 1.0,
                "novelty_explore_bonus": 0.0,
                "performance_explore_boost": 0.0,
            },
        )
        agent = AwareAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(4))
        obs = {"x": 0, "y": 0, "goal_x": 5, "goal_y": 0}
        before = agent.act(obs)
        eps_before = float((before.info or {}).get("aware_epsilon", 0.0))
        self.assertAlmostEqual(eps_before, 0.8, places=4)

        agent.on_social_contract(
            {
                "verse_name": "unknown",
                "has_contract": True,
                "risk_budget": 1.0,
                "veto_bias": 1.0,
                "confidence": 1.0,
                "support": 3,
            }
        )
        after = agent.act(obs)
        eps_after = float((after.info or {}).get("aware_epsilon", 1.0))
        self.assertLess(eps_after, eps_before)
        self.assertTrue(bool((after.info or {}).get("social_contract_active", False)))

    def test_aware_learn_from_shared_influences_action(self):
        spec = _agent_spec(
            "aware",
            {
                "epsilon_start": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 1.0,
                "use_vector_memory": False,
                "shared_hint_weight": 1.0,
            },
        )
        agent = AwareAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space(4))
        obs = {"x": 0, "y": 0, "goal_x": 4, "goal_y": 0}
        pre = agent.act(obs)
        self.assertEqual(int(pre.action), 0)

        updates = agent.learn_from_shared(
            [
                {
                    "provider_agent_id": "ally_a",
                    "provider_trust": 0.9,
                    "verse_name": "line_world",
                    "transitions": [
                        {
                            "obs": {"x": 0, "y": 0, "goal_x": 4, "goal_y": 0},
                            "action": 2,
                            "reward": 0.8,
                            "done": False,
                        }
                    ],
                }
            ]
        )
        self.assertGreater(int(updates.get("shared_updates", 0)), 0)

        post = agent.act({"x": 1, "y": 0, "goal_x": 4, "goal_y": 0})
        self.assertEqual(int(post.action), 2)
        self.assertTrue(bool((post.info or {}).get("shared_hint_used", False)))


if __name__ == "__main__":
    unittest.main()
