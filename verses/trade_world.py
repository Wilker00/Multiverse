"""
verses/trade_world.py

Economics / trading verse.
The agent operates in a market with cyclical price fluctuations.
It buys goods when prices are low, holds inventory, and sells when
prices are high. Transaction costs, inventory limits, and random
market shocks add complexity.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class TradeParams:
    max_steps: int = 120
    starting_cash: float = 100.0
    max_inventory: int = 10
    transaction_cost: float = 1.0
    # Price dynamics
    base_price: float = 20.0
    price_amplitude: float = 8.0     # how much price oscillates
    cycle_length: int = 20           # steps per full sine cycle
    noise_stddev: float = 2.0        # random price noise
    shock_probability: float = 0.05  # prob of a market crash/boom each step
    shock_magnitude: float = 10.0    # how big a shock is
    # Rewards
    step_penalty: float = -0.01
    profit_scale: float = 1.0        # multiplier on realized profit


class TradeWorldVerse:
    """
    Market trading:
    - Price follows a noisy sine wave with random shocks
    - Agent can buy (if cash >= price + cost), sell (if inventory > 0), or hold
    - Transaction cost on every buy/sell
    - Inventory has a max limit
    - Reward = realized profit from sales - costs - step penalty
    - Episode ends after max_steps; final reward includes liquidation value
    """

    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in TradeWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        cfg = self.spec.params
        self.params = TradeParams(
            max_steps=max(10, int(cfg.get("max_steps", 120))),
            starting_cash=max(1.0, float(cfg.get("starting_cash", 100.0))),
            max_inventory=max(1, int(cfg.get("max_inventory", 10))),
            transaction_cost=max(0.0, float(cfg.get("transaction_cost", 1.0))),
            base_price=max(1.0, float(cfg.get("base_price", 20.0))),
            price_amplitude=max(0.0, float(cfg.get("price_amplitude", 8.0))),
            cycle_length=max(2, int(cfg.get("cycle_length", 20))),
            noise_stddev=max(0.0, float(cfg.get("noise_stddev", 2.0))),
            shock_probability=max(0.0, min(1.0, float(cfg.get("shock_probability", 0.05)))),
            shock_magnitude=max(0.0, float(cfg.get("shock_magnitude", 10.0))),
            step_penalty=float(cfg.get("step_penalty", -0.01)),
            profit_scale=max(0.0, float(cfg.get("profit_scale", 1.0))),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=[
                "price", "price_delta", "cash", "inventory", "avg_buy_price",
                "portfolio_value", "cycle_phase", "t",
            ],
            subspaces={
                "price": SpaceSpec(type="vector", shape=(1,), dtype="float32"),
                "price_delta": SpaceSpec(type="vector", shape=(1,), dtype="float32"),
                "cash": SpaceSpec(type="vector", shape=(1,), dtype="float32"),
                "inventory": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "avg_buy_price": SpaceSpec(type="vector", shape=(1,), dtype="float32"),
                "portfolio_value": SpaceSpec(type="vector", shape=(1,), dtype="float32"),
                "cycle_phase": SpaceSpec(type="vector", shape=(1,), dtype="float32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="TradeWorld obs dict",
        )

        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=3,
            notes="0=buy,1=sell,2=hold",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._t = 0
        self._done = False
        self._cash = 0.0
        self._inventory = 0
        self._price = 0.0
        self._prev_price = 0.0
        self._total_buy_cost = 0.0    # total spent on current inventory
        self._total_profit = 0.0
        self._shock_offset = 0.0      # persistent shock that decays

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        self._t = 0
        self._done = False
        self._cash = float(self.params.starting_cash)
        self._inventory = 0
        self._total_buy_cost = 0.0
        self._total_profit = 0.0
        self._shock_offset = 0.0
        self._price = self._compute_price()
        self._prev_price = self._price

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "starting_cash": self.params.starting_cash,
            "max_inventory": self.params.max_inventory,
        }
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        a = max(0, min(2, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": int(a)}

        if a == 0:
            # Buy
            total_cost = self._price + self.params.transaction_cost
            if self._cash >= total_cost and self._inventory < self.params.max_inventory:
                self._cash -= total_cost
                self._inventory += 1
                self._total_buy_cost += self._price
                info["bought_at"] = round(self._price, 2)
            else:
                info["buy_failed"] = True
                if self._cash < total_cost:
                    info["reason"] = "insufficient_cash"
                else:
                    info["reason"] = "inventory_full"
        elif a == 1:
            # Sell
            if self._inventory > 0:
                revenue = self._price - self.params.transaction_cost
                avg_cost = self._total_buy_cost / max(1, self._inventory)
                profit = revenue - avg_cost
                self._cash += max(0.0, revenue)
                self._total_buy_cost -= avg_cost
                self._total_buy_cost = max(0.0, self._total_buy_cost)
                self._inventory -= 1
                self._total_profit += profit
                reward += float(profit * self.params.profit_scale)
                info["sold_at"] = round(self._price, 2)
                info["profit"] = round(profit, 2)
            else:
                info["sell_failed"] = True
                info["reason"] = "no_inventory"
        # else: hold — no action

        # Update price
        self._prev_price = self._price

        # Random shock
        if self._rng.random() < self.params.shock_probability:
            direction = 1.0 if self._rng.random() < 0.5 else -1.0
            self._shock_offset += direction * self.params.shock_magnitude
            info["market_shock"] = round(direction * self.params.shock_magnitude, 2)

        # Shock decay
        self._shock_offset *= 0.85

        self._price = self._compute_price()

        done = False
        truncated = bool(self._t >= self.params.max_steps)

        # Liquidation at end
        if truncated and self._inventory > 0:
            liquidation_value = self._inventory * (self._price - self.params.transaction_cost)
            avg_cost = self._total_buy_cost / max(1, self._inventory) * self._inventory
            liquidation_profit = liquidation_value - avg_cost
            self._cash += max(0.0, liquidation_value)
            self._total_profit += liquidation_profit
            reward += float(liquidation_profit * self.params.profit_scale * 0.5)  # 50% discount for forced liquidation
            info["liquidated"] = self._inventory
            info["liquidation_profit"] = round(liquidation_profit, 2)
            self._inventory = 0
            self._total_buy_cost = 0.0

        self._done = bool(done or truncated)
        info["t"] = int(self._t)
        info["cash"] = round(self._cash, 2)
        info["inventory"] = int(self._inventory)
        info["total_profit"] = round(self._total_profit, 2)

        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        delta = self._price - self._prev_price
        direction = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
        return (
            f"t={self._t} price={self._price:.1f}{direction} "
            f"cash={self._cash:.1f} inv={self._inventory} "
            f"profit={self._total_profit:.1f}"
        )

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "t": int(self._t), "done": bool(self._done),
            "cash": float(self._cash), "inventory": int(self._inventory),
            "price": float(self._price), "prev_price": float(self._prev_price),
            "total_buy_cost": float(self._total_buy_cost),
            "total_profit": float(self._total_profit),
            "shock_offset": float(self._shock_offset),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        self._cash = max(0.0, float(state.get("cash", self._cash)))
        self._inventory = max(0, int(state.get("inventory", self._inventory)))
        self._price = max(0.1, float(state.get("price", self._price)))
        self._prev_price = float(state.get("prev_price", self._prev_price))
        self._total_buy_cost = max(0.0, float(state.get("total_buy_cost", self._total_buy_cost)))
        self._total_profit = float(state.get("total_profit", self._total_profit))
        self._shock_offset = float(state.get("shock_offset", self._shock_offset))

    def _compute_price(self) -> float:
        """Noisy sinusoidal price with shock offset."""
        phase = 2.0 * math.pi * float(self._t) / float(self.params.cycle_length)
        base = self.params.base_price + self.params.price_amplitude * math.sin(phase)
        noise = self._rng.gauss(0.0, self.params.noise_stddev)
        return max(1.0, base + noise + self._shock_offset)

    def _make_obs(self) -> JSONValue:
        avg_buy = 0.0
        if self._inventory > 0:
            avg_buy = self._total_buy_cost / float(self._inventory)
        portfolio = self._cash + self._inventory * self._price
        phase = (float(self._t) % float(self.params.cycle_length)) / float(self.params.cycle_length)
        return {
            "price": round(self._price, 2),
            "price_delta": round(self._price - self._prev_price, 2),
            "cash": round(self._cash, 2),
            "inventory": int(self._inventory),
            "avg_buy_price": round(avg_buy, 2),
            "portfolio_value": round(portfolio, 2),
            "cycle_phase": round(phase, 3),
            "t": int(self._t),
        }


class TradeWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["economics", "trading", "temporal_pattern", "risk_management", "inventory"]

    def create(self, spec: VerseSpec) -> Verse:
        return TradeWorldVerse(spec)
