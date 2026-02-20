import unittest

from core.agent_base import ActionResult
from core.rollout import RolloutConfig, run_episode
from core.types import AgentRef, AgentSpec, RunRef, SpaceSpec, VerseRef, VerseSpec
from core.verse_base import ResetResult, StepResult


class _ObsVerse:
    def __init__(self):
        self.spec = VerseSpec(spec_version="v1", verse_name="grid_world", verse_version="0.1")
        self.observation_space = SpaceSpec(type="dict")
        self.action_space = SpaceSpec(type="discrete", n=3)
        self._t = 0

    def seed(self, seed):
        _ = seed

    def reset(self):
        self._t = 0
        return ResetResult(obs={"x": 0, "y": 0, "t": 0}, info={})

    def step(self, action):
        _ = action
        self._t += 1
        return StepResult(obs={"x": 1, "y": 0, "t": self._t}, reward=0.1, done=True, truncated=False, info={})

    def close(self):
        return


class _SelectorAgent:
    def __init__(self):
        self.spec = AgentSpec(spec_version="v1", policy_id="sel", policy_version="0.1", algo="special_moe")
        self.observation_space = SpaceSpec(type="dict")
        self.action_space = SpaceSpec(type="discrete", n=3)

    def seed(self, seed):
        _ = seed

    def act(self, obs):
        _ = obs
        return ActionResult(
            action=1,
            info={
                "mode": "special_moe",
                "selector_active": True,
                "experts": ["grid_world.txt", "line_world.txt"],
                "weights": [0.8, 0.2],
                "selected_expert": "grid_world.txt",
                "selector_confidence": 0.8,
            },
        )


class _DeclineQueryAgent(_SelectorAgent):
    def memory_query_request(self, *, obs, step_idx):
        _ = (obs, step_idx)
        return None


class _MixedMetricsAgent(_SelectorAgent):
    def learn(self, batch):
        _ = batch
        return {"loss": 1.25, "updates": 3, "mode": "non_numeric"}


class _ErrorQueryAgent(_SelectorAgent):
    def memory_query_request(self, *, obs, step_idx):
        _ = (obs, step_idx)
        raise RuntimeError("request failed")


class _ExplodingSafeExecutor:
    def reset_episode(self, seed):
        _ = seed
        raise RuntimeError("reset failed")

    def select_action(self, agent, obs):
        return agent.act(obs)

    def post_step(self, *, obs, action_result, step_result, step_idx, primary_agent):
        _ = (obs, action_result, step_idx, primary_agent)
        return step_result


class TestRolloutObservability(unittest.TestCase):
    def test_selector_routing_telemetry_written_to_event(self):
        verse = _ObsVerse()
        agent = _SelectorAgent()
        res = run_episode(
            verse=verse,
            verse_ref=VerseRef.create("grid_world", "0.1", "abc"),
            agent=agent,
            agent_ref=AgentRef.create(policy_id="sel", policy_version="0.1"),
            run=RunRef.create(),
            config=RolloutConfig(schema_version="v1", max_steps=2),
            seed=7,
            on_step=None,
        )
        self.assertEqual(len(res.events), 1)
        info = res.events[0].info or {}
        route = dict(info.get("selector_routing") or {})
        self.assertEqual(str(route.get("selected_expert")), "grid_world.txt")
        self.assertAlmostEqual(float(route.get("confidence", 0.0)), 0.8, places=5)
        self.assertEqual(str(route.get("verse")), "grid_world")
        self.assertEqual(len(str(route.get("obs_hash", ""))), 40)

    def test_memory_query_state_reports_budget_and_block_reason(self):
        verse = _ObsVerse()
        agent = _DeclineQueryAgent()
        res = run_episode(
            verse=verse,
            verse_ref=VerseRef.create("grid_world", "0.1", "abc"),
            agent=agent,
            agent_ref=AgentRef.create(policy_id="sel", policy_version="0.1"),
            run=RunRef.create(),
            config=RolloutConfig(
                schema_version="v1",
                max_steps=2,
                on_demand_memory_enabled=True,
                on_demand_query_budget=3,
                on_demand_min_interval=1,
            ),
            seed=11,
            on_step=None,
        )
        info = res.events[0].info or {}
        mq = dict(info.get("memory_query") or {})
        self.assertTrue(bool(mq.get("enabled", False)))
        self.assertEqual(int(mq.get("budget", -1)), 3)
        self.assertEqual(int(mq.get("used", -1)), 0)
        self.assertEqual(int(mq.get("remaining", -1)), 3)
        self.assertEqual(str(mq.get("block_reason")), "agent_declined")

    def test_train_metrics_ignores_non_numeric_values(self):
        verse = _ObsVerse()
        agent = _MixedMetricsAgent()
        res = run_episode(
            verse=verse,
            verse_ref=VerseRef.create("grid_world", "0.1", "abc"),
            agent=agent,
            agent_ref=AgentRef.create(policy_id="sel", policy_version="0.1"),
            run=RunRef.create(),
            config=RolloutConfig(
                schema_version="v1",
                max_steps=2,
                train=True,
                collect_transitions=True,
            ),
            seed=13,
            on_step=None,
        )
        self.assertIn("loss", res.train_metrics)
        self.assertIn("updates", res.train_metrics)
        self.assertNotIn("mode", res.train_metrics)
        self.assertAlmostEqual(float(res.train_metrics["loss"]), 1.25, places=6)
        self.assertAlmostEqual(float(res.train_metrics["updates"]), 3.0, places=6)

    def test_runtime_errors_capture_safe_executor_reset_failures(self):
        verse = _ObsVerse()
        agent = _SelectorAgent()
        res = run_episode(
            verse=verse,
            verse_ref=VerseRef.create("grid_world", "0.1", "abc"),
            agent=agent,
            agent_ref=AgentRef.create(policy_id="sel", policy_version="0.1"),
            run=RunRef.create(),
            config=RolloutConfig(
                schema_version="v1",
                max_steps=2,
                safe_executor=_ExplodingSafeExecutor(),
            ),
            seed=17,
            on_step=None,
        )
        info = res.events[0].info or {}
        runtime = dict(info.get("runtime_errors") or {})
        counters = dict(runtime.get("counters") or {})
        self.assertEqual(int(counters.get("safe_executor_reset_error", 0)), 1)
        warnings = list(runtime.get("warnings") or [])
        self.assertTrue(any(str(w.get("code", "")) == "safe_executor_reset_error" for w in warnings))

    def test_runtime_errors_capture_memory_query_request_failures(self):
        verse = _ObsVerse()
        agent = _ErrorQueryAgent()
        res = run_episode(
            verse=verse,
            verse_ref=VerseRef.create("grid_world", "0.1", "abc"),
            agent=agent,
            agent_ref=AgentRef.create(policy_id="sel", policy_version="0.1"),
            run=RunRef.create(),
            config=RolloutConfig(
                schema_version="v1",
                max_steps=2,
                on_demand_memory_enabled=True,
                on_demand_query_budget=1,
                on_demand_min_interval=1,
            ),
            seed=19,
            on_step=None,
        )
        info = res.events[0].info or {}
        runtime = dict(info.get("runtime_errors") or {})
        counters = dict(runtime.get("counters") or {})
        self.assertGreaterEqual(int(counters.get("memory_query_request_error", 0)), 1)


if __name__ == "__main__":
    unittest.main()
