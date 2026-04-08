"""
Deterministic blackjack basic-strategy baseline for the fixed S17/DAS ruleset.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.agent_base import ActionResult, ExperienceBatch
from core.types import AgentSpec, JSONValue, SpaceSpec


def _safe_int(obs: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(obs.get(key, default))
    except Exception:
        return int(default)


class BlackjackBasicAgent:
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

    def seed(self, seed: Optional[int]) -> None:
        return

    def act(self, obs: JSONValue) -> ActionResult:
        data = dict(obs) if isinstance(obs, dict) else {}
        legal = self._legal_actions_from_obs(data)
        action = self._choose_action(data, legal)
        return ActionResult(action=int(action), info={"mode": "blackjack_basic"})

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        return {}

    def save(self, path: str) -> None:
        return

    def load(self, path: str) -> None:
        return

    def close(self) -> None:
        return

    @staticmethod
    def _legal_actions_from_obs(obs: Dict[str, Any]) -> list[int]:
        total = _safe_int(obs, "player_sum")
        legal = [1] if total >= 21 else [0, 1]
        if _safe_int(obs, "can_double"):
            legal.append(2)
        if _safe_int(obs, "can_split"):
            legal.append(3)
        return sorted(set(int(a) for a in legal))

    def _choose_action(self, obs: Dict[str, Any], legal: list[int]) -> int:
        total = _safe_int(obs, "player_sum")
        dealer = _safe_int(obs, "dealer_showing")
        usable_ace = bool(_safe_int(obs, "usable_ace"))
        pair_rank = _safe_int(obs, "pair_rank")
        hand_len = _safe_int(obs, "hand_len", 2)

        if total >= 21:
            return 1

        recommended = None
        if hand_len == 2 and pair_rank > 0 and 3 in legal:
            recommended = self._pair_action(pair_rank, dealer)
        if recommended is None and usable_ace and hand_len >= 2:
            recommended = self._soft_action(total, dealer)
        if recommended is None:
            recommended = self._hard_action(total, dealer)

        if recommended in legal:
            return int(recommended)
        if recommended == 2 and 0 in legal:
            return 0
        if recommended == 3 and 0 in legal:
            return 0
        return 1 if 1 in legal else legal[0]

    @staticmethod
    def _pair_action(pair_rank: int, dealer: int) -> Optional[int]:
        if pair_rank == 11:
            return 3
        if pair_rank == 10:
            return 1
        if pair_rank == 9:
            return 3 if dealer in (2, 3, 4, 5, 6, 8, 9) else 1
        if pair_rank == 8:
            return 3
        if pair_rank == 7:
            return 3 if 2 <= dealer <= 7 else 0
        if pair_rank == 6:
            return 3 if 2 <= dealer <= 6 else 0
        if pair_rank == 5:
            return None
        if pair_rank == 4:
            return 3 if dealer in (5, 6) else 0
        if pair_rank in (2, 3):
            return 3 if 2 <= dealer <= 7 else 0
        return None

    @staticmethod
    def _soft_action(total: int, dealer: int) -> int:
        if total >= 19:
            return 1
        if total == 18:
            if 3 <= dealer <= 6:
                return 2
            if dealer in (2, 7, 8):
                return 1
            return 0
        if total in (17, 16):
            return 2 if 3 <= dealer <= 6 else 0
        if total in (15, 14):
            return 2 if 4 <= dealer <= 6 else 0
        if total in (13, 12):
            return 2 if 5 <= dealer <= 6 else 0
        return 0

    @staticmethod
    def _hard_action(total: int, dealer: int) -> int:
        if total >= 17:
            return 1
        if 13 <= total <= 16:
            return 1 if 2 <= dealer <= 6 else 0
        if total == 12:
            return 1 if 4 <= dealer <= 6 else 0
        if total == 11:
            return 2 if dealer != 11 else 0
        if total == 10:
            return 2 if 2 <= dealer <= 9 else 0
        if total == 9:
            return 2 if 3 <= dealer <= 6 else 0
        return 0
