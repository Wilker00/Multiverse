import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.agent_base import ActionResult
from core.safe_executor import SafeExecutor, SafeExecutorConfig
from core.types import SpaceSpec
from core.verse_base import ResetResult, StepResult


@dataclass
class _Spec:
    verse_name: str = "dummy_world"


class _DummyVerse:
    def __init__(self):
        self.spec = _Spec()
        self.action_space = SpaceSpec(type="discrete", n=2)
        self.observation_space = SpaceSpec(type="dict")
        self._t = 0
        self._doomed = False
        self._protected = False

    def seed(self, seed: Optional[int]) -> None:
        _ = seed

    def reset(self) -> ResetResult:
        self._t = 0
        self._doomed = False
        self._protected = False
        return ResetResult(obs=self._obs(), info={})

    def step(self, action: int) -> StepResult:
        a = int(action)
        if self._t == 0 and a == 1:
            self._protected = True
        if a == 0 and not self._protected:
            self._doomed = True
        self._t += 1
        done = self._t >= 4
        if done and self._doomed:
            return StepResult(obs=self._obs(), reward=-1.0, done=True, truncated=False, info={"lost_game": True})
        if done:
            return StepResult(obs=self._obs(), reward=0.2, done=True, truncated=False, info={"reached_goal": True})
        return StepResult(obs=self._obs(), reward=0.0, done=False, truncated=False, info={})

    def export_state(self) -> Dict[str, Any]:
        return {"t": int(self._t), "doomed": bool(self._doomed), "protected": bool(self._protected)}

    def import_state(self, state: Dict[str, Any]) -> None:
        self._t = int(state.get("t", 0))
        self._doomed = bool(state.get("doomed", False))
        self._protected = bool(state.get("protected", False))

    def _obs(self) -> Dict[str, int]:
        return {"t": int(self._t), "doomed": int(self._doomed), "protected": int(self._protected)}


class _DummyAgent:
    def act(self, obs):
        _ = obs
        return ActionResult(action=0, info={})

    def action_diagnostics(self, obs):
        _ = obs
        return {"sample_probs": [0.9, 0.1], "danger_scores": [0.95, 0.10]}


class _BrokenDiagAgent(_DummyAgent):
    def action_diagnostics(self, obs):
        _ = obs
        raise RuntimeError("diag unavailable")


class TestSafeExecutorMCTS(unittest.TestCase):
    def test_mcts_vetoes_forced_loss_action(self):
        verse = _DummyVerse()
        agent = _DummyAgent()
        cfg = SafeExecutorConfig.from_dict(
            {
                "enabled": True,
                "danger_threshold": 0.90,
                "min_action_confidence": 0.0,
                "prefer_fallback_on_veto": False,
                "planner_enabled": False,
                "mcts_enabled": True,
                "mcts_num_simulations": 48,
                "mcts_max_depth": 4,
                "mcts_loss_threshold": -0.19,
                "mcts_min_visits": 2,
                "mcts_trigger_on_high_danger": True,
            }
        )
        executor = SafeExecutor(config=cfg, verse=verse, fallback_agent=None)
        executor.reset_episode(seed=123)
        rr = verse.reset()

        action_result = executor.select_action(agent, rr.obs)
        self.assertEqual(int(action_result.action), 1)
        se = dict(action_result.info.get("safe_executor") or {})
        self.assertEqual(str(se.get("mode")), "mcts_veto")
        self.assertIn(0, list(se.get("forced_loss_actions") or []))

    def test_adaptive_veto_relaxes_thresholds_after_stable_performance(self):
        verse = _DummyVerse()
        cfg = SafeExecutorConfig.from_dict(
            {
                "enabled": True,
                "danger_threshold": 0.60,
                "min_action_confidence": 0.40,
                "adaptive_veto_enabled": True,
                "adaptive_veto_relaxation": 0.50,
                "adaptive_veto_warmup_steps": 0,
                "adaptive_veto_failure_guard": 0.20,
                "min_competence_rate": 0.50,
            }
        )
        executor = SafeExecutor(config=cfg, verse=verse, fallback_agent=None)
        for _ in range(8):
            executor._monitor.observe(confidence=0.95, danger=0.10, dangerous_outcome=False, reward=0.1)  # type: ignore[attr-defined]
        executor._steps_observed = 20  # type: ignore[attr-defined]
        eff_conf, eff_danger, adaptation = executor._effective_veto_thresholds()  # type: ignore[attr-defined]
        self.assertLess(float(eff_conf), float(cfg.min_action_confidence))
        self.assertGreater(float(eff_danger), float(cfg.danger_threshold))
        self.assertGreater(float(adaptation), 0.0)

    def test_adaptive_veto_schedule_strengthens_relaxation_over_time(self):
        verse = _DummyVerse()
        cfg = SafeExecutorConfig.from_dict(
            {
                "enabled": True,
                "danger_threshold": 0.60,
                "min_action_confidence": 0.40,
                "adaptive_veto_enabled": True,
                "adaptive_veto_relaxation": 0.50,
                "adaptive_veto_warmup_steps": 0,
                "adaptive_veto_failure_guard": 0.20,
                "adaptive_veto_schedule_enabled": True,
                "adaptive_veto_relaxation_start": 0.05,
                "adaptive_veto_relaxation_end": 0.50,
                "adaptive_veto_schedule_steps": 100,
                "adaptive_veto_schedule_power": 1.0,
                "min_competence_rate": 0.50,
            }
        )
        executor = SafeExecutor(config=cfg, verse=verse, fallback_agent=None)
        for _ in range(8):
            executor._monitor.observe(confidence=0.95, danger=0.10, dangerous_outcome=False, reward=0.1)  # type: ignore[attr-defined]

        executor._steps_observed = 10  # type: ignore[attr-defined]
        eff_conf_early, eff_danger_early, adapt_early = executor._effective_veto_thresholds()  # type: ignore[attr-defined]
        executor._steps_observed = 100  # type: ignore[attr-defined]
        eff_conf_late, eff_danger_late, adapt_late = executor._effective_veto_thresholds()  # type: ignore[attr-defined]

        self.assertGreater(float(adapt_late), float(adapt_early))
        self.assertLess(float(eff_conf_late), float(eff_conf_early))
        self.assertGreater(float(eff_danger_late), float(eff_danger_early))

    def test_safe_executor_config_rejects_unknown_keys_by_default(self):
        with self.assertRaises(ValueError):
            SafeExecutorConfig.from_dict({"enabled": True, "typo_mcts_num_sim": 42})

    def test_safe_executor_config_can_disable_strict_validation(self):
        cfg = SafeExecutorConfig.from_dict(
            {
                "enabled": True,
                "strict_config_validation": False,
                "typo_mcts_num_sim": 42,
            }
        )
        self.assertTrue(bool(cfg.enabled))

    def test_ground_truth_danger_label_overrides_reward_heuristic(self):
        verse = _DummyVerse()
        cfg = SafeExecutorConfig.from_dict(
            {
                "enabled": True,
                "severe_reward_threshold": -0.5,
            }
        )
        executor = SafeExecutor(config=cfg, verse=verse, fallback_agent=None)
        action_result = ActionResult(action=0, info={"safe_executor": {"confidence": 1.0, "danger": 0.1}})
        step = StepResult(
            obs={"t": 1},
            reward=-1.0,
            done=False,
            truncated=False,
            info={"danger_label": False},
        )
        out = executor.post_step(
            obs={"t": 0},
            action_result=action_result,
            step_result=step,
            step_idx=0,
            primary_agent=None,
        )
        se = dict((out.info or {}).get("safe_executor") or {})
        self.assertFalse(bool(se.get("dangerous_outcome", True)))
        self.assertEqual(se.get("danger_label"), False)

    def test_runtime_errors_are_exposed_in_safe_executor_metadata(self):
        verse = _DummyVerse()
        cfg = SafeExecutorConfig.from_dict({"enabled": True, "prefer_fallback_on_veto": False})
        executor = SafeExecutor(config=cfg, verse=verse, fallback_agent=None)
        executor.reset_episode(seed=321)
        rr = verse.reset()

        action_result = executor.select_action(_BrokenDiagAgent(), rr.obs)
        step = verse.step(int(action_result.action))
        out = executor.post_step(
            obs=rr.obs,
            action_result=action_result,
            step_result=step,
            step_idx=0,
            primary_agent=None,
        )
        se = dict((out.info or {}).get("safe_executor") or {})
        runtime = dict(se.get("runtime_errors") or {})
        by_code = dict(runtime.get("by_code") or {})
        self.assertGreaterEqual(int(runtime.get("total", 0)), 1)
        self.assertGreaterEqual(int(by_code.get("risk_diag_error", 0)), 1)
        self.assertIn("runtime_error", list(se.get("failure_signals") or []))


if __name__ == "__main__":
    unittest.main()
