"""
verses/blackjack_world.py

Blackjack (21) — full casino-rules engine.

Ruleset (locked):
  - 6 decks (312 cards), reshuffle at 75% penetration
  - Dealer stands on soft 17 (S17 rule)
  - Natural blackjack pays 3:2
  - Double down allowed on first two cards of any hand (including after split)
  - Split up to 4 hands total
  - Aces split to exactly 1 card each; no re-split aces
  - No surrender
  - No insurance

Actions (discrete, n=4):
  0 = Hit
  1 = Stand
  2 = Double Down  — legal only on a 2-card hand
  3 = Split        — legal only on a matching pair, with < 4 total hands

Observation (player-visible only — POMDP; dealer hole card is hidden):
  player_sum      : int  active hand total (4–21; bust shows actual value > 21)
  dealer_showing  : int  dealer face-up card value (2–11, where 11 = Ace)
  usable_ace      : int  1 if active hand holds an ace counted as 11
  can_double      : int  1 if double-down is legal this turn
  can_split       : int  1 if split is legal this turn
  num_hands       : int  total hands in play (1–4)
  active_hand     : int  index of the hand currently being played (0–3)
  t               : int  step counter within the episode
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


# ---------------------------------------------------------------------------
# Card / hand helpers
# ---------------------------------------------------------------------------

def _build_shoe(num_decks: int) -> List[int]:
    """
    Build a fresh unshuffled shoe.

    Encoding:
      2–9  → face value
      10   → all ten-value cards (10, J, Q, K); 16 per deck
      11   → Ace; 4 per deck
    """
    shoe: List[int] = []
    for v in range(2, 10):                        # 2–9: 4 suits × num_decks
        shoe.extend([v] * (4 * num_decks))
    shoe.extend([10] * (16 * num_decks))           # 10, J, Q, K
    shoe.extend([11] * (4 * num_decks))            # Ace
    return shoe


def _hand_value(hand: List[int]) -> Tuple[int, int]:
    """
    Return (total, usable_ace).

    usable_ace = 1 means at least one ace is being counted as 11
    in the current total (i.e. the hand is "soft").
    """
    total = sum(hand)
    aces = hand.count(11)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    usable = 1 if (aces > 0 and total <= 21) else 0
    return total, usable


def _is_natural(hand: List[int]) -> bool:
    """Two-card 21 (an Ace + any ten-value card)."""
    return len(hand) == 2 and _hand_value(hand)[0] == 21


def _is_pair(hand: List[int]) -> bool:
    """
    True if the hand is exactly two cards with the same normalized value.
    All ten-value cards (10, J, Q, K — all stored as 10) are treated as equal.
    """
    if len(hand) != 2:
        return False
    return hand[0] == hand[1]


def _pair_rank(hand: List[int]) -> int:
    if not _is_pair(hand):
        return 0
    return int(hand[0])


def _hi_lo_value(card: int) -> int:
    card = int(card)
    if 2 <= card <= 6:
        return 1
    if card in (10, 11):
        return -1
    return 0


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class BlackjackWorldParams:
    num_decks: int = 6
    reshuffle_penetration: float = 0.75  # reshuffle when this fraction of cards has been dealt
    blackjack_payout: float = 1.5        # 3:2
    win_reward: float = 1.0
    lose_reward: float = -1.0
    push_reward: float = 0.0
    step_penalty: float = 0.0            # per-step cost (0 by default)
    max_steps: int = 30                  # safety truncation


# ---------------------------------------------------------------------------
# Verse
# ---------------------------------------------------------------------------

class BlackjackWorldVerse:
    """
    Full 6-deck blackjack with S17, double-after-split, split to 4 hands.
    No surrender, no insurance.
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in BlackjackWorldFactory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        p = self.spec.params or {}
        self.params = BlackjackWorldParams(
            num_decks=max(1, int(p.get("num_decks", 6))),
            reshuffle_penetration=max(0.1, min(0.95, float(p.get("reshuffle_penetration", 0.75)))),
            blackjack_payout=float(p.get("blackjack_payout", 1.5)),
            win_reward=float(p.get("win_reward", 1.0)),
            lose_reward=float(p.get("lose_reward", -1.0)),
            push_reward=float(p.get("push_reward", 0.0)),
            step_penalty=float(p.get("step_penalty", 0.0)),
            max_steps=max(10, int(p.get("max_steps", 30))),
        )

        obs_keys = [
            "player_sum", "dealer_showing", "usable_ace",
            "can_double", "can_split", "num_hands", "active_hand", "t",
            "hand_len", "pair_rank", "split_aces_hand", "first_action",
            "running_count", "true_count", "cards_remaining",
        ]
        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=obs_keys,
            subspaces={k: SpaceSpec(type="vector", shape=(1,), dtype="int32") for k in obs_keys},
            notes="Player-visible blackjack state (POMDP — dealer hole card hidden).",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=4,
            notes="0=hit, 1=stand, 2=double_down, 3=split",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None

        # Shoe persists across episodes; reshuffled at penetration threshold
        self._shoe: List[int] = []
        self._shoe_pos: int = 0

        # Per-episode state
        self._dealer_hand: List[int] = []
        self._player_hands: List[List[int]] = []
        self._hand_bets: List[float] = []           # bet multiplier per hand (1.0 or 2.0)
        self._hand_split_aces: List[bool] = []      # True → hand came from splitting aces
        self._hand_done: List[bool] = []            # True → hand is resolved (stand/bust/double)
        self._active_hand: int = 0
        self._done: bool = False
        self._t: int = 0
        self._public_running_count: int = 0
        self._dealer_hole_revealed: bool = False

        # Deferred terminal result for natural blackjack resolved on first step
        self._pending_terminal: bool = False
        self._pending_reward: float = 0.0
        self._pending_info: Dict[str, JSONValue] = {}

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2 ** 31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._shoe = _build_shoe(self.params.num_decks)
        self._rng.shuffle(self._shoe)
        self._shoe_pos = 0

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        if self._pending_terminal and self._dealer_hand and not self._dealer_hole_revealed:
            self._reveal_dealer_hole()

        # Reshuffle when penetration threshold exceeded
        shoe_len = len(self._shoe)
        if shoe_len == 0 or self._shoe_pos >= int(shoe_len * self.params.reshuffle_penetration):
            self._shoe = _build_shoe(self.params.num_decks)
            self._rng.shuffle(self._shoe)
            self._shoe_pos = 0
            self._public_running_count = 0

        # Reset episode
        self._done = False
        self._t = 0
        self._pending_terminal = False
        self._pending_reward = 0.0
        self._pending_info = {}

        # Deal: player, dealer, player, dealer (standard order)
        c1 = self._deal_card()
        d1 = self._deal_card()
        c2 = self._deal_card()
        d2 = self._deal_card()

        self._dealer_hand = [d1, d2]
        self._player_hands = [[c1, c2]]
        self._hand_bets = [1.0]
        self._hand_split_aces = [False]
        self._hand_done = [False]
        self._active_hand = 0
        self._dealer_hole_revealed = False
        self._mark_visible_card(c1)
        self._mark_visible_card(d1)
        self._mark_visible_card(c2)

        # Check for natural blackjack — defer resolution to first step()
        player_bj = _is_natural(self._player_hands[0])
        dealer_bj = _is_natural(self._dealer_hand)

        if player_bj or dealer_bj:
            if player_bj and dealer_bj:
                r = float(self.params.push_reward)
                outcome = "push_blackjack"
            elif player_bj:
                r = float(self.params.blackjack_payout)
                outcome = "blackjack"
            else:
                r = float(self.params.lose_reward)
                outcome = "dealer_blackjack"
            self._pending_terminal = True
            self._pending_reward = r
            self._pending_info = {
                "outcome": outcome,
                "reward": float(r),
                "blackjack": True,
                "player_hand": list(self._player_hands[0]),
                "dealer_hand": list(self._dealer_hand),
            }
            self._hand_done[0] = True

        return ResetResult(
            obs=self._make_obs(),
            info={
                "seed": int(self._seed) if self._seed is not None else 0,
                "blackjack_pending": bool(self._pending_terminal),
                "player_hand": list(self._player_hands[0]),
                "dealer_showing": int(self._dealer_hand[0]),
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(
                self._make_obs(), 0.0, True, False, {"warning": "step() called after done"}
            )

        # Deliver deferred natural-blackjack outcome on first step
        if self._pending_terminal:
            self._pending_terminal = False
            self._reveal_dealer_hole()
            self._done = True
            return StepResult(
                self._make_obs(),
                float(self._pending_reward),
                True,
                False,
                dict(self._pending_info),
            )

        a = max(0, min(3, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)

        # Remap illegal actions to stand (safe fallback)
        legal = self.legal_actions()
        if a not in legal:
            a = 1

        hand = self._player_hands[self._active_hand]
        bet = float(self._hand_bets[self._active_hand])

        if a == 0:  # Hit
            card = self._deal_card()
            hand.append(card)
            self._mark_visible_card(card)
            total, _ = _hand_value(hand)
            if total >= 21:
                # Bust (> 21) or pat 21 — both auto-advance
                self._hand_done[self._active_hand] = True
                reward += self._advance_hand()

        elif a == 1:  # Stand
            self._hand_done[self._active_hand] = True
            reward += self._advance_hand()

        elif a == 2:  # Double Down
            card = self._deal_card()
            hand.append(card)
            self._mark_visible_card(card)
            self._hand_bets[self._active_hand] = bet * 2.0
            self._hand_done[self._active_hand] = True
            reward += self._advance_hand()

        elif a == 3:  # Split
            card0, card1 = hand[0], hand[1]
            splitting_aces = (card0 == 11)

            # Rebuild active hand: first card + new draw
            draw0 = self._deal_card()
            self._player_hands[self._active_hand] = [card0, draw0]
            # Insert second hand immediately after active
            ni = self._active_hand + 1
            draw1 = self._deal_card()
            self._player_hands.insert(ni, [card1, draw1])
            self._hand_bets.insert(ni, 1.0)
            self._hand_split_aces.insert(ni, splitting_aces)
            self._hand_done.insert(ni, False)
            self._hand_split_aces[self._active_hand] = splitting_aces
            self._mark_visible_card(draw0)
            self._mark_visible_card(draw1)

            if splitting_aces:
                # Aces get exactly one card each; both hands immediately done
                self._hand_done[self._active_hand] = True
                self._hand_done[ni] = True
                reward += self._advance_hand()
            else:
                current_total, _ = _hand_value(self._player_hands[self._active_hand])
                next_total, _ = _hand_value(self._player_hands[ni])
                if current_total >= 21:
                    self._hand_done[self._active_hand] = True
                if next_total >= 21:
                    self._hand_done[ni] = True
                if self._hand_done[self._active_hand]:
                    reward += self._advance_hand()

        # Truncation safety net
        if not self._done and self._t >= self.params.max_steps:
            for i in range(len(self._player_hands)):
                if not self._hand_done[i]:
                    self._hand_done[i] = True
            reward += self._resolve_dealer()
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), True, True, self._build_step_info(a)
            )

        return StepResult(
            self._make_obs(), float(reward), bool(self._done), False, self._build_step_info(a)
        )

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[int]:
        # Pending natural blackjack — step() will resolve it; action is irrelevant but must be valid
        if self._pending_terminal:
            return [1]
        if self._done or self._active_hand >= len(self._player_hands):
            return []
        if self._hand_done[self._active_hand]:
            return []

        hand = self._player_hands[self._active_hand]
        total, _ = _hand_value(hand)
        if total > 21:
            return []
        if total == 21:
            return [1]

        legal = [0, 1]  # hit and stand always available

        if len(hand) == 2:
            legal.append(2)  # double down on first two cards

        if (
            len(hand) == 2
            and _is_pair(hand)
            and len(self._player_hands) < 4
            and not self._hand_split_aces[self._active_hand]  # no re-split aces
        ):
            legal.append(3)  # split

        return sorted(set(legal))

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode != "ansi":
            return None
        idx = min(self._active_hand, len(self._player_hands) - 1)
        hand = self._player_hands[idx] if self._player_hands else []
        total, usable = _hand_value(hand) if hand else (0, 0)
        soft_tag = "(soft)" if usable else ""
        dealer_up = self._dealer_hand[0] if self._dealer_hand else "?"
        return (
            f"BlackjackWorld t={self._t} "
            f"hand[{idx}/{len(self._player_hands)}]={hand} "
            f"sum={total}{soft_tag} dealer_up={dealer_up}"
        )

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "done": bool(self._done),
            "t": int(self._t),
            "public_running_count": int(self._public_running_count),
            "dealer_hole_revealed": bool(self._dealer_hole_revealed),
            "shoe": list(self._shoe),
            "shoe_pos": int(self._shoe_pos),
            "dealer_hand": list(self._dealer_hand),
            "player_hands": [list(h) for h in self._player_hands],
            "hand_bets": [float(b) for b in self._hand_bets],
            "hand_split_aces": [bool(x) for x in self._hand_split_aces],
            "hand_done": [bool(x) for x in self._hand_done],
            "active_hand": int(self._active_hand),
            "pending_terminal": bool(self._pending_terminal),
            "pending_reward": float(self._pending_reward),
            "pending_info": dict(self._pending_info),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._done = bool(state.get("done", False))
        self._t = max(0, int(state.get("t", 0)))
        self._public_running_count = int(state.get("public_running_count", 0))
        self._dealer_hole_revealed = bool(state.get("dealer_hole_revealed", False))
        self._shoe_pos = max(0, int(state.get("shoe_pos", 0)))
        raw_shoe = state.get("shoe")
        if isinstance(raw_shoe, list):
            self._shoe = [int(x) for x in raw_shoe]
        raw_dealer = state.get("dealer_hand")
        if isinstance(raw_dealer, list):
            self._dealer_hand = [int(x) for x in raw_dealer]
        raw_hands = state.get("player_hands")
        if isinstance(raw_hands, list):
            self._player_hands = [[int(c) for c in h] for h in raw_hands]
        raw_bets = state.get("hand_bets")
        if isinstance(raw_bets, list):
            self._hand_bets = [float(b) for b in raw_bets]
        raw_split_aces = state.get("hand_split_aces")
        if isinstance(raw_split_aces, list):
            self._hand_split_aces = [bool(x) for x in raw_split_aces]
        raw_done = state.get("hand_done")
        if isinstance(raw_done, list):
            self._hand_done = [bool(x) for x in raw_done]
        self._active_hand = max(0, int(state.get("active_hand", 0)))
        self._pending_terminal = bool(state.get("pending_terminal", False))
        self._pending_reward = float(state.get("pending_reward", 0.0))
        raw_pi = state.get("pending_info")
        if isinstance(raw_pi, dict):
            self._pending_info = {str(k): v for k, v in raw_pi.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deal_card(self) -> int:
        """Draw the next card from the shoe, reshuffling on overflow."""
        if self._shoe_pos >= len(self._shoe):
            self._shoe = _build_shoe(self.params.num_decks)
            self._rng.shuffle(self._shoe)
            self._shoe_pos = 0
        card = self._shoe[self._shoe_pos]
        self._shoe_pos += 1
        return int(card)

    def _mark_visible_card(self, card: int) -> None:
        self._public_running_count += int(_hi_lo_value(card))

    def _reveal_dealer_hole(self) -> None:
        if self._dealer_hole_revealed or len(self._dealer_hand) < 2:
            return
        self._mark_visible_card(self._dealer_hand[1])
        self._dealer_hole_revealed = True

    def _advance_hand(self) -> float:
        """
        Move to the next unfinished hand.
        If all hands are resolved, trigger dealer play and return net reward.
        Otherwise return 0.0.
        """
        self._active_hand += 1
        while (
            self._active_hand < len(self._player_hands)
            and self._hand_done[self._active_hand]
        ):
            self._active_hand += 1

        if self._active_hand >= len(self._player_hands):
            r = self._resolve_dealer()
            self._done = True
            return float(r)
        return 0.0

    def _resolve_dealer(self) -> float:
        """
        Dealer draws to S17+, then compute net reward across all player hands.

        S17 rule: dealer stands on any 17, whether hard or soft.
        Bust hands lose (negative reward); win/push/loss relative to dealer total.
        """
        self._reveal_dealer_hole()
        while True:
            total, _soft = _hand_value(self._dealer_hand)
            if total < 17:
                card = self._deal_card()
                self._dealer_hand.append(card)
                self._mark_visible_card(card)
            else:
                break  # stand on all 17+, including soft 17 (S17)

        dealer_total, _ = _hand_value(self._dealer_hand)
        dealer_bust = dealer_total > 21

        net = 0.0
        for i, hand in enumerate(self._player_hands):
            bet = float(self._hand_bets[i])
            total, _ = _hand_value(hand)
            if total > 21:
                net += float(self.params.lose_reward) * bet
            elif dealer_bust or total > dealer_total:
                net += float(self.params.win_reward) * bet
            elif total == dealer_total:
                net += float(self.params.push_reward)
            else:
                net += float(self.params.lose_reward) * bet
        return float(net)

    def _build_step_info(self, action: int) -> Dict[str, JSONValue]:
        idx = min(self._active_hand, len(self._player_hands) - 1)
        hand = self._player_hands[idx] if self._player_hands else []
        total, _ = _hand_value(hand) if hand else (0, 0)
        info: Dict[str, JSONValue] = {
            "action": int(action),
            "t": int(self._t),
            "player_sum": int(total),
            "dealer_showing": int(self._dealer_hand[0]) if self._dealer_hand else 0,
            "num_hands": int(len(self._player_hands)),
            "active_hand": int(idx),
        }
        if self._done:
            info["dealer_hand"] = list(self._dealer_hand)
            info["player_hands"] = [list(h) for h in self._player_hands]
            info["hand_bets"] = [float(b) for b in self._hand_bets]
        return info

    def _make_obs(self) -> JSONValue:
        if self._player_hands:
            idx = min(self._active_hand, len(self._player_hands) - 1)
            hand = self._player_hands[idx]
        else:
            idx = 0
            hand = []

        total, usable = _hand_value(hand) if hand else (0, 0)
        dealer_showing = int(self._dealer_hand[0]) if self._dealer_hand else 0
        legal = self.legal_actions()
        pair_rank = _pair_rank(hand)
        cards_remaining = max(0, len(self._shoe) - self._shoe_pos)
        decks_remaining = max(1.0 / 52.0, float(cards_remaining) / 52.0)
        true_count = int(round(float(self._public_running_count) / float(decks_remaining)))

        return {
            "player_sum": int(total),
            "dealer_showing": int(dealer_showing),
            "usable_ace": int(usable),
            "can_double": int(2 in legal),
            "can_split": int(3 in legal),
            "num_hands": int(len(self._player_hands)),
            "active_hand": int(idx),
            "t": int(self._t),
            "hand_len": int(len(hand)),
            "pair_rank": int(pair_rank),
            "split_aces_hand": int(self._hand_split_aces[idx]) if self._hand_split_aces else 0,
            "first_action": int(1 if len(hand) == 2 else 0),
            "running_count": int(self._public_running_count),
            "true_count": int(true_count),
            "cards_remaining": int(cards_remaining),
        }


class BlackjackWorldFactory:
    @property
    def tags(self) -> List[str]:
        return [
            "strategy_games",
            "card_game",
            "blackjack",
            "probabilistic",
            "risk_sensitive",
            "partial_observable",
            "turn_based",
        ]

    def create(self, spec: VerseSpec) -> Verse:
        return BlackjackWorldVerse(spec)
