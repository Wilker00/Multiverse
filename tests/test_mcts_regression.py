import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.mcts_search import MCTSConfig, MCTSSearch
from core.types import SpaceSpec
from core.verse_base import ResetResult, StepResult


@dataclass
class _Spec:
    verse_name: str = "regression_world"


class _RegressionVerse:
    def __init__(self):
        self.spec = _Spec()
        self.action_space = SpaceSpec(type="discrete", n=3)
        self.observation_space = SpaceSpec(type="dict")
        self._t = 0
        self._score = 0
        self._done = False

    def seed(self, seed: Optional[int]) -> None:
        _ = seed

    def reset(self) -> ResetResult:
        self._t = 0
        self._score = 0
        self._done = False
        return ResetResult(obs=self._obs(), info={})

    def legal_actions(self, obs=None):
        _ = obs
        if self._done:
            return []
        return [0, 1, 2]

    def step(self, action: int) -> StepResult:
        if self._done:
            return StepResult(obs=self._obs(), reward=0.0, done=True, truncated=False, info={"warning": "done"})
        a = int(action)
        self._t += 1
        # Action 1 is consistently best, 2 is medium, 0 is bad.
        self._score += (2 if a == 1 else (1 if a == 2 else -2))
        done = self._t >= 5
        self._done = done
        reward = float(self._score / 10.0)
        info: Dict[str, Any] = {"t": int(self._t), "score": int(self._score)}
        if done:
            if self._score > 0:
                info["reached_goal"] = True
            else:
                info["lost_game"] = True
        return StepResult(obs=self._obs(), reward=reward, done=done, truncated=False, info=info)

    def export_state(self) -> Dict[str, Any]:
        return {"t": int(self._t), "score": int(self._score), "done": bool(self._done)}

    def import_state(self, state: Dict[str, Any]) -> None:
        self._t = int(state.get("t", 0))
        self._score = int(state.get("score", 0))
        self._done = bool(state.get("done", False))

    def _obs(self) -> Dict[str, int]:
        return {"t": int(self._t), "score": int(self._score)}


class TestMCTSRegression(unittest.TestCase):
    def test_deterministic_best_action(self):
        verse = _RegressionVerse()
        rr = verse.reset()
        cfg = MCTSConfig(num_simulations=64, max_depth=5, seed=123, dirichlet_epsilon=0.0)
        search = MCTSSearch(verse=verse, config=cfg)
        result = search.search(
            root_obs=rr.obs,
            policy_net=lambda _obs: [0.2, 0.6, 0.2],
            value_net=lambda _obs: 0.0,
        )
        self.assertEqual(int(result.best_action), 1)
        self.assertGreater(result.action_probs[1], result.action_probs[0])
        self.assertGreater(result.action_probs[1], result.action_probs[2])

    def test_transposition_caches_are_populated(self):
        verse = _RegressionVerse()
        rr = verse.reset()
        cfg = MCTSConfig(num_simulations=24, max_depth=4, seed=7, transposition_cache=True, dirichlet_epsilon=0.0)
        search = MCTSSearch(verse=verse, config=cfg)
        _ = search.search(
            root_obs=rr.obs,
            policy_net=lambda _obs: [0.3, 0.4, 0.3],
            value_net=lambda _obs: 0.1,
        )
        self.assertGreaterEqual(len(search._prior_cache), 1)  # noqa: SLF001 - intentional regression probe
        self.assertGreaterEqual(len(search._value_cache), 1)  # noqa: SLF001 - intentional regression probe


if __name__ == "__main__":
    unittest.main()
