"""
verses/uno_world_v2.py

Tactical Uno with real card game mechanics:
- 60-card deck: 40 number cards (4 colors × 10 values),
  12 action cards (4 colors × Skip/Reverse/Draw2), 8 wilds
- Real hand management: play matching cards, draw from pile
- Action cards: Skip (opponent loses turn), Draw2 (opponent draws 2)
- Wild cards: playable anytime, player chooses color
- UNO call: must call when down to 1 card

Action space: 15 discrete
  0-9: play card at hand index 0-9
  10: draw a card from pile
  11: call UNO (when at 2 cards, before playing)
  12: wild as Red, 13: wild as Green, 14: wild as Blue
  (wild as Yellow = action 9 repurposed when holding wild)

Observation: hand composition + game state features

Designed for transfer learning: shares strategic concepts with chess/go
(resource management, tempo, pressure) but completely different mechanics.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse

# ---------------------------------------------------------------------------
# Card constants
# ---------------------------------------------------------------------------
# Colors
RED, GREEN, BLUE, YELLOW = 0, 1, 2, 3
COLOR_NAMES = ["Red", "Green", "Blue", "Yellow"]
WILD_COLOR = 4  # no specific color

# Values 0-9 are number cards, 10=Skip, 11=Reverse, 12=Draw2
SKIP, REVERSE, DRAW2 = 10, 11, 12
WILD, WILD_DRAW4 = 13, 14

N_ACTIONS = 15  # 0-9 play hand[i], 10=draw, 11=UNO, 12-14=wild color choice

@dataclass
class Card:
    color: int  # 0-3 for colored, 4 for wild
    value: int  # 0-12 for colored, 13=Wild, 14=Wild+Draw4

    def matches(self, top_color: int, top_value: int) -> bool:
        """Can this card be played on (top_color, top_value)?"""
        if self.value in (WILD, WILD_DRAW4):
            return True
        if self.color == top_color:
            return True
        if self.value == top_value and self.value <= 12:
            return True
        return False

    def __repr__(self) -> str:
        if self.value == WILD:
            return "W"
        if self.value == WILD_DRAW4:
            return "W+4"
        cname = COLOR_NAMES[self.color][0] if self.color < 4 else "?"
        if self.value == SKIP:
            return f"{cname}Skip"
        if self.value == REVERSE:
            return f"{cname}Rev"
        if self.value == DRAW2:
            return f"{cname}+2"
        return f"{cname}{self.value}"


def _build_deck() -> List[Card]:
    """Build a 60-card simplified Uno deck."""
    deck: List[Card] = []
    for color in range(4):
        # Number cards: one of each 0-9
        for val in range(10):
            deck.append(Card(color=color, value=val))
        # Action cards: one each of Skip, Reverse, Draw2
        deck.append(Card(color=color, value=SKIP))
        deck.append(Card(color=color, value=REVERSE))
        deck.append(Card(color=color, value=DRAW2))
    # 4 Wilds + 4 Wild Draw4
    for _ in range(4):
        deck.append(Card(color=WILD_COLOR, value=WILD))
    for _ in range(4):
        deck.append(Card(color=WILD_COLOR, value=WILD_DRAW4))
    return deck


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class UnoV2Params:
    max_steps: int = 200
    hand_size: int = 7
    step_penalty: float = -0.003
    win_reward: float = 2.5
    lose_penalty: float = -2.5
    card_play_reward: float = 0.03
    action_card_bonus: float = 0.08
    uno_penalty: float = -0.3  # penalty for forgetting to call UNO


# ---------------------------------------------------------------------------
# Verse implementation
# ---------------------------------------------------------------------------

class UnoWorldV2Verse(Verse):
    """
    Tactical Uno with real card game mechanics.

    15 actions, masked to legal plays.
    Agent vs. 1 opponent.
    """

    def __init__(self, spec: VerseSpec):
        merged_tags = list(spec.tags)
        for t in UnoWorldV2Factory().tags:
            if t not in merged_tags:
                merged_tags.append(t)
        self.spec = dataclasses.replace(spec, tags=merged_tags)

        self.params = UnoV2Params(
            max_steps=int(self.spec.params.get("max_steps", 200)),
            hand_size=int(self.spec.params.get("hand_size", 7)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.003)),
            win_reward=float(self.spec.params.get("win_reward", 2.5)),
            lose_penalty=float(self.spec.params.get("lose_penalty", -2.5)),
            card_play_reward=float(self.spec.params.get("card_play_reward", 0.03)),
            action_card_bonus=float(self.spec.params.get("action_card_bonus", 0.08)),
            uno_penalty=float(self.spec.params.get("uno_penalty", -0.3)),
        )
        self.params.max_steps = max(20, int(self.params.max_steps))
        self.params.hand_size = max(3, min(10, int(self.params.hand_size)))

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "hand_size", "opp_hand_size",
                "top_color", "top_value",
                "hand_colors", "hand_wilds", "hand_action_cards",
                "hand_playable", "draw_pile_size",
                "opp_said_uno",
                "score_delta", "pressure", "risk", "tempo",
                "control", "resource",
                "t",
            ],
            notes="Uno hand composition and game state features.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=N_ACTIONS,
            notes="0-9=play hand[i], 10=draw, 11=call UNO, 12-14=wild color(R/G/B/Y)",
        )

        self._rng = random.Random()
        self._seed_val: Optional[int] = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._done = False
        self._t = 0
        self._deck: List[Card] = []
        self._hand: List[Card] = []
        self._opp_hand: List[Card] = []
        self._discard: List[Card] = []
        self._top_color = 0
        self._top_value = 0
        self._said_uno = False
        self._opp_said_uno = False
        self._pending_wild: Optional[Card] = None  # wild waiting for color choice
        self._skip_next = False
        self._draw_pending = 0  # cards opponent must draw

    def seed(self, seed: Optional[int]) -> None:
        self._seed_val = seed
        if seed is not None:
            self._rng = random.Random(int(seed))

    def reset(self) -> ResetResult:
        if self._seed_val is not None:
            self._rng = random.Random(int(self._seed_val))

        self._reset_state()
        self._deck = _build_deck()
        self._rng.shuffle(self._deck)

        # Deal hands
        for _ in range(self.params.hand_size):
            if self._deck:
                self._hand.append(self._deck.pop())
            if self._deck:
                self._opp_hand.append(self._deck.pop())

        # Flip first card (must be a number card)
        while self._deck:
            card = self._deck.pop()
            if card.value <= 9:
                self._discard.append(card)
                self._top_color = card.color
                self._top_value = card.value
                break
            else:
                # Put action/wild cards back and reshuffle
                self._deck.insert(0, card)
                self._rng.shuffle(self._deck)

        return ResetResult(
            obs=self._make_obs(),
            info={"verse_name": "uno_world_v2"},
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(
                self._make_obs(), 0.0, True, False,
                {"warning": "step() called after done"},
            )

        self._t += 1
        a = int(action)
        reward = float(self.params.step_penalty)

        # ---- Handle pending wild color choice ----
        if self._pending_wild is not None:
            if 12 <= a <= 14:
                chosen_color = a - 12  # 0=R, 1=G, 2=B
                self._top_color = chosen_color
                self._pending_wild = None
            elif a <= 9:
                # Interpret as Yellow (index 3)
                self._top_color = YELLOW
                self._pending_wild = None
            else:
                # Default to most common color in hand
                self._top_color = self._best_color()
                self._pending_wild = None

            # Process draw effects from Wild+Draw4
            if self._top_value == WILD_DRAW4:
                self._draw_pending = 4

            return self._after_agent_play(reward)

        # ---- UNO call ----
        if a == 11:
            if len(self._hand) == 2:
                self._said_uno = True
                reward += 0.01
            else:
                reward -= 0.01  # called UNO with wrong hand count
            # Continue — still need to play a card, but we used the turn for UNO
            return self._after_agent_play(reward, played_card=False)

        # ---- Draw ----
        if a == 10:
            drawn = self._draw_card(self._hand)
            if drawn:
                reward -= 0.01
            return self._after_agent_play(reward, played_card=False)

        # ---- Play card from hand ----
        if 0 <= a <= 9 and a < len(self._hand):
            card = self._hand[a]
            if card.matches(self._top_color, self._top_value):
                self._hand.pop(a)
                self._discard.append(card)

                # UNO check: should have called before playing down to 1
                if len(self._hand) == 1 and not self._said_uno:
                    reward += float(self.params.uno_penalty)
                    # Draw 2 as penalty
                    self._draw_card(self._hand)
                    self._draw_card(self._hand)

                self._said_uno = False

                # Update top card
                if card.value in (WILD, WILD_DRAW4):
                    self._top_value = card.value
                    self._pending_wild = card
                    reward += float(self.params.card_play_reward)
                    # Return — waiting for color choice
                    return StepResult(
                        self._make_obs(), float(reward), False, False,
                        {"t": self._t, "played": repr(card), "awaiting_color": True},
                    )

                self._top_color = card.color
                self._top_value = card.value
                reward += float(self.params.card_play_reward)

                # Action card effects
                if card.value == SKIP:
                    self._skip_next = True
                    reward += float(self.params.action_card_bonus)
                elif card.value == REVERSE:
                    # In 2-player, reverse acts as skip
                    self._skip_next = True
                    reward += float(self.params.action_card_bonus)
                elif card.value == DRAW2:
                    self._draw_pending = 2
                    reward += float(self.params.action_card_bonus)

                # Win check
                if len(self._hand) == 0:
                    reward += float(self.params.win_reward)
                    self._done = True
                    return StepResult(
                        self._make_obs(), float(reward), True, False,
                        {"reached_goal": True, "reason": "empty_hand", "t": self._t},
                    )

                return self._after_agent_play(reward)
            else:
                # Card doesn't match — penalty, treat as draw
                reward -= 0.02
                self._draw_card(self._hand)
                return self._after_agent_play(reward, played_card=False)
        else:
            # Invalid action — draw
            reward -= 0.02
            self._draw_card(self._hand)
            return self._after_agent_play(reward, played_card=False)

    def _after_agent_play(self, reward: float, played_card: bool = True) -> StepResult:
        """Process opponent turn after agent action."""
        # Apply pending draws to opponent
        if self._draw_pending > 0:
            for _ in range(self._draw_pending):
                self._draw_card(self._opp_hand)
            self._draw_pending = 0
            self._skip_next = True  # drawing also skips turn

        # Opponent turn (unless skipped)
        if not self._skip_next:
            opp_reward_delta = self._opponent_turn()
            reward += opp_reward_delta

            # Opponent win check
            if len(self._opp_hand) == 0:
                reward += float(self.params.lose_penalty)
                self._done = True
                return StepResult(
                    self._make_obs(), float(reward), True, False,
                    {"reached_goal": False, "lost_game": True, "reason": "opp_empty_hand", "t": self._t},
                )
        else:
            self._skip_next = False

        # Truncation
        if self._t >= int(self.params.max_steps):
            # Score by hand size difference
            diff = len(self._opp_hand) - len(self._hand)
            if diff > 0:
                reward += 0.5
                reached = True
            elif diff < 0:
                reward -= 0.5
                reached = False
            else:
                reached = False
            self._done = True
            return StepResult(
                self._make_obs(), float(reward), False, True,
                {"reached_goal": reached, "reason": "time_limit", "hand_diff": diff, "t": self._t},
            )

        # Shaping: fewer cards is better
        hand_diff = len(self._opp_hand) - len(self._hand)
        reward += 0.002 * hand_diff

        return StepResult(
            self._make_obs(), float(reward), False, False,
            {"t": self._t, "hand_size": len(self._hand), "opp_hand_size": len(self._opp_hand)},
        )

    def _opponent_turn(self) -> float:
        """Simple opponent: play the first matching card, prefer action cards."""
        penalty = 0.0

        # UNO call check
        if len(self._opp_hand) == 2:
            self._opp_said_uno = True

        # Find playable cards
        playable = []
        for i, card in enumerate(self._opp_hand):
            if card.matches(self._top_color, self._top_value):
                # Prefer action cards
                priority = 0
                if card.value >= SKIP:
                    priority = 2
                elif card.value >= 7:
                    priority = 1
                playable.append((i, card, priority))

        if playable:
            # Sort by priority (action cards first), then by value (high first)
            playable.sort(key=lambda x: (x[2], x[1].value), reverse=True)

            # 30% chance of random choice for variety
            if self._rng.random() < 0.30 and len(playable) > 1:
                idx, card, _ = self._rng.choice(playable)
            else:
                idx, card, _ = playable[0]

            self._opp_hand.pop(idx)
            self._discard.append(card)

            if card.value in (WILD, WILD_DRAW4):
                # Opponent picks their most common color
                color_counts = [0, 0, 0, 0]
                for c in self._opp_hand:
                    if c.color < 4:
                        color_counts[c.color] += 1
                self._top_color = color_counts.index(max(color_counts)) if any(c > 0 for c in color_counts) else 0
                self._top_value = card.value

                if card.value == WILD_DRAW4:
                    # Agent draws 4
                    for _ in range(4):
                        self._draw_card(self._hand)
                    penalty -= 0.15

            else:
                self._top_color = card.color
                self._top_value = card.value

                if card.value == SKIP or card.value == REVERSE:
                    # In 2-player, this skips agent's next turn
                    # We'll handle by giving opponent another turn
                    penalty -= 0.05
                elif card.value == DRAW2:
                    for _ in range(2):
                        self._draw_card(self._hand)
                    penalty -= 0.10

            # UNO penalty check for opponent
            if len(self._opp_hand) == 1 and not self._opp_said_uno:
                # Opponent forgot UNO — they draw 2
                self._draw_card(self._opp_hand)
                self._draw_card(self._opp_hand)
                self._opp_said_uno = False
                penalty += 0.05

        else:
            # Opponent draws
            self._draw_card(self._opp_hand)

        return penalty

    def _draw_card(self, hand: List[Card]) -> bool:
        """Draw a card into *hand*. Reshuffles discard if deck empty."""
        if not self._deck:
            if len(self._discard) <= 1:
                return False  # no cards left
            top = self._discard.pop()
            self._deck = self._discard
            self._discard = [top]
            self._rng.shuffle(self._deck)

        if self._deck:
            hand.append(self._deck.pop())
            return True
        return False

    def _best_color(self) -> int:
        """Return the most common color in agent's hand."""
        counts = [0, 0, 0, 0]
        for c in self._hand:
            if c.color < 4:
                counts[c.color] += 1
        return counts.index(max(counts)) if any(c > 0 for c in counts) else 0

    def legal_actions(self, obs: Optional[JSONValue] = None) -> List[JSONValue]:
        if self._done:
            return []

        # Pending wild color choice
        if self._pending_wild is not None:
            return [12, 13, 14]  # choose R, G, B (or 0-9 for Y)

        actions: List[int] = []
        for i, card in enumerate(self._hand):
            if i >= 10:
                break
            if card.matches(self._top_color, self._top_value):
                actions.append(i)

        actions.append(10)  # draw is always legal

        if len(self._hand) == 2:
            actions.append(11)  # UNO call

        return actions

    def _make_obs(self) -> JSONValue:
        # Hand composition
        color_counts = [0, 0, 0, 0]
        wilds = 0
        action_cards = 0
        playable = 0
        hand_value = 0

        for card in self._hand:
            if card.color < 4:
                color_counts[card.color] += 1
            if card.value in (WILD, WILD_DRAW4):
                wilds += 1
            if card.value >= SKIP:
                action_cards += 1
            if card.matches(self._top_color, self._top_value):
                playable += 1
            hand_value += min(card.value, 10)  # cap at 10 for face cards

        opp_hand_value = sum(min(c.value, 10) for c in self._opp_hand)

        # Abstract strategy features (for semantic bridge)
        hand_size = len(self._hand)
        opp_size = len(self._opp_hand)
        score_delta = opp_size - hand_size  # positive = we're winning
        pressure = min(16, max(-16, playable * 2 + action_cards * 3 - opp_size))
        risk = max(0, min(16, hand_size * 2 - playable * 3 + max(0, hand_size - 3)))
        tempo = max(0, min(10, playable + wilds))
        control = min(16, max(-16, action_cards * 2 + wilds * 3 - opp_size // 2))
        resource = max(0, min(16, hand_size + wilds * 2))

        return {
            "hand_size": int(hand_size),
            "opp_hand_size": int(opp_size),
            "top_color": int(self._top_color),
            "top_value": int(self._top_value),
            "hand_colors": list(color_counts),
            "hand_wilds": int(wilds),
            "hand_action_cards": int(action_cards),
            "hand_playable": int(playable),
            "draw_pile_size": int(len(self._deck)),
            "opp_said_uno": int(self._opp_said_uno),
            # Abstract strategy features for semantic bridge
            "score_delta": int(score_delta),
            "pressure": int(pressure),
            "risk": int(risk),
            "tempo": int(tempo),
            "control": int(control),
            "resource": int(resource),
            "t": int(self._t),
        }

    def render(self, mode: str = "ansi") -> str:
        lines = []
        lines.append(f"Top: {COLOR_NAMES[self._top_color]}({self._top_value})")
        hand_str = ", ".join(repr(c) for c in self._hand)
        lines.append(f"Your hand ({len(self._hand)}): [{hand_str}]")
        lines.append(f"Opponent cards: {len(self._opp_hand)}")
        lines.append(f"Draw pile: {len(self._deck)}")
        if self._opp_said_uno:
            lines.append("Opponent said UNO!")
        return "\n".join(lines)

    def close(self) -> None:
        pass

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "hand": [(c.color, c.value) for c in self._hand],
            "opp_hand": [(c.color, c.value) for c in self._opp_hand],
            "deck": [(c.color, c.value) for c in self._deck],
            "discard": [(c.color, c.value) for c in self._discard],
            "top_color": int(self._top_color),
            "top_value": int(self._top_value),
            "t": int(self._t),
            "done": bool(self._done),
            "said_uno": bool(self._said_uno),
            "opp_said_uno": bool(self._opp_said_uno),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._hand = [Card(color=c, value=v) for c, v in state["hand"]]
        self._opp_hand = [Card(color=c, value=v) for c, v in state["opp_hand"]]
        self._deck = [Card(color=c, value=v) for c, v in state["deck"]]
        self._discard = [Card(color=c, value=v) for c, v in state["discard"]]
        self._top_color = int(state["top_color"])
        self._top_value = int(state["top_value"])
        self._t = int(state["t"])
        self._done = bool(state.get("done", False))
        self._said_uno = bool(state.get("said_uno", False))
        self._opp_said_uno = bool(state.get("opp_said_uno", False))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class UnoWorldV2Factory:
    @property
    def tags(self) -> List[str]:
        return ["strategy_games", "card_game", "uno", "hand_management", "v2"]

    def create(self, spec: VerseSpec) -> UnoWorldV2Verse:
        return UnoWorldV2Verse(spec)
