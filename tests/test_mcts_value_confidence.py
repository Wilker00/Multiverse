import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.mcts_search import MCTSConfig, MCTSSearch
from core.types import SpaceSpec
from core.verse_base import ResetResult, StepResult


@dataclass
class _Spec:
    verse_name: str = "confidence_world"


class _MiniVerse:
    def __init__(self):
        self.spec = _Spec()
        self.action_space = SpaceSpec(type="discrete", n=2)
        self.observation_space = SpaceSpec(type="dict")
        self._t = 0
        self._done = False

    def reset(self):
        self._t = 0
        self._done = False
        return ResetResult(obs={"t": 0}, info={})

    def legal_actions(self, obs=None):
        _ = obs
        return [] if self._done else [0, 1]

    def step(self, action: int):
        _ = action
        self._t += 1
        done = self._t >= 2
        self._done = done
        return StepResult(obs={"t": int(self._t)}, reward=0.0, done=done, truncated=False, info={})

    def export_state(self) -> Dict[str, Any]:
        return {"t": int(self._t), "done": bool(self._done)}

    def import_state(self, state: Dict[str, Any]) -> None:
        self._t = int(state.get("t", 0))
        self._done = bool(state.get("done", False))


class _LowConfValue:
    def value_with_confidence(self, obs, history=None):
        _ = obs
        _ = history
        return {"value": 1.0, "confidence": 0.05}


class _HighConfValue:
    def value_with_confidence(self, obs, history=None):
        _ = obs
        _ = history
        return {"value": 1.0, "confidence": 0.95}


class TestMCTSValueConfidence(unittest.TestCase):
    def test_low_confidence_value_is_ignored(self):
        verse = _MiniVerse()
        rr = verse.reset()
        search = MCTSSearch(
            verse=verse,
            config=MCTSConfig(
                num_simulations=8,
                max_depth=2,
                value_confidence_threshold=0.50,
                dirichlet_epsilon=0.0,
                seed=3,
            ),
        )
        low = search.search(root_obs=rr.obs, policy_net=lambda _obs: [0.5, 0.5], value_net=_LowConfValue())
        search_hi = MCTSSearch(
            verse=verse,
            config=MCTSConfig(
                num_simulations=8,
                max_depth=2,
                value_confidence_threshold=0.50,
                dirichlet_epsilon=0.0,
                seed=3,
            ),
        )
        high = search_hi.search(root_obs=rr.obs, policy_net=lambda _obs: [0.5, 0.5], value_net=_HighConfValue())
        self.assertLessEqual(abs(float(low.avg_leaf_value)), 0.30)
        self.assertGreater(float(high.avg_leaf_value), float(low.avg_leaf_value))


if __name__ == "__main__":
    unittest.main()
