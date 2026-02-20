import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.safe_executor import SafeExecutor, SafeExecutorConfig
from core.types import SpaceSpec
from core.verse_base import ResetResult, StepResult


@dataclass
class _Spec:
    verse_name: str = "chess_world"


class _Verse:
    def __init__(self):
        self.spec = _Spec()
        self.action_space = SpaceSpec(type="discrete", n=6)
        self.observation_space = SpaceSpec(type="dict")
        self._done = False

    def seed(self, seed: Optional[int]) -> None:
        _ = seed

    def reset(self):
        self._done = False
        return ResetResult(obs={"t": 0}, info={})

    def step(self, action: int):
        _ = action
        self._done = True
        return StepResult(obs={"t": 1}, reward=0.0, done=True, truncated=False, info={})

    def export_state(self) -> Dict[str, Any]:
        return {"done": bool(self._done)}

    def import_state(self, state: Dict[str, Any]) -> None:
        self._done = bool(state.get("done", False))


class TestSafeExecutorMCTSOverrides(unittest.TestCase):
    def test_default_verse_overrides_apply(self):
        cfg = SafeExecutorConfig.from_dict({"enabled": True, "mcts_enabled": True, "mcts_num_simulations": 32})
        ex = SafeExecutor(config=cfg, verse=_Verse(), fallback_agent=None)
        self.assertTrue(ex._mcts is not None)  # noqa: SLF001
        # chess_world default override should raise simulations from 32 -> 128
        self.assertGreaterEqual(int(ex.config.mcts_num_simulations), 128)

    def test_explicit_verse_overrides_take_precedence(self):
        cfg = SafeExecutorConfig.from_dict(
            {
                "enabled": True,
                "mcts_enabled": True,
                "mcts_num_simulations": 32,
                "mcts_verse_overrides": {"chess_world": {"mcts_num_simulations": 48, "mcts_loss_threshold": -0.5}},
            }
        )
        ex = SafeExecutor(config=cfg, verse=_Verse(), fallback_agent=None)
        self.assertEqual(int(ex.config.mcts_num_simulations), 48)
        self.assertAlmostEqual(float(ex.config.mcts_loss_threshold), -0.5, places=6)


if __name__ == "__main__":
    unittest.main()
