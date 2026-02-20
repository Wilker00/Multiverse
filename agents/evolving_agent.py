"""
agents/evolving_agent.py

Evolving agent built on top of AwareAgent.

The agent monitors rolling episode returns. On stagnation windows, it mutates
core learning hyperparameters around the best-known settings.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Dict

from core.agent_base import ExperienceBatch
from core.types import AgentSpec, JSONValue, SpaceSpec

from agents.aware_agent import AwareAgent, _safe_float, _safe_int


def _safe_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
    return bool(default)


def _evolving_defaults_for_verse(verse_name: str) -> Dict[str, float]:
    verse = str(verse_name or "").strip().lower()
    if verse == "cliff_world":
        return {
            "evolve_interval": 16,
            "evolve_margin": 0.50,
            "mutation_scale": 0.15,
            "gamma_mutation": 0.015,
            "epsilon_decay_mutation": 0.010,
            # Keep base Q-shaping neutral for evolving on cliff to avoid over-coupling
            # mutation with hazard shaping.
            "learn_success_bonus": 0.0,
            "learn_hazard_penalty": 0.0,
        }
    if verse == "labyrinth_world":
        return {
            "evolve_interval": 20,
            "evolve_margin": 0.40,
            "mutation_scale": 0.12,
            "gamma_mutation": 0.010,
            "epsilon_decay_mutation": 0.008,
            "learn_success_bonus": 0.0,
            "learn_hazard_penalty": 0.0,
        }
    return {}


class EvolvingAgent(AwareAgent):
    """
    Awareness-augmented Q agent with lightweight hyperparameter evolution.

    Config keys:
    - evolve_interval: int episodes per evaluation window (default 12)
    - evolve_margin: float minimum improvement to mark window as better (default 0.02)
    - mutation_scale: float multiplicative mutation range for lr (default 0.20)
    - gamma_mutation: float additive mutation range for gamma (default 0.02)
    - epsilon_decay_mutation: float additive mutation range for epsilon_decay (default 0.015)
    - mutate_on_plateau: bool-like (default true)
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        cfg_in = dict(spec.config) if isinstance(spec.config, dict) else {}
        verse_name = str(cfg_in.get("verse_name", "")).strip().lower()
        tuned_pre = _evolving_defaults_for_verse(verse_name)
        if tuned_pre:
            merged = dict(cfg_in)
            for k, v in tuned_pre.items():
                merged.setdefault(k, v)
            spec = spec.evolved(config=merged)

        super().__init__(spec=spec, observation_space=observation_space, action_space=action_space)

        cfg = spec.config if isinstance(spec.config, dict) else {}
        tuned = _evolving_defaults_for_verse(verse_name)

        self._evolve_interval = max(1, _safe_int(cfg.get("evolve_interval", tuned.get("evolve_interval", 12)), 12))
        self._evolve_margin = _safe_float(cfg.get("evolve_margin", tuned.get("evolve_margin", 0.02)), 0.02)
        self._mutation_scale = min(
            0.95,
            max(0.0, _safe_float(cfg.get("mutation_scale", tuned.get("mutation_scale", 0.20)), 0.20)),
        )
        self._gamma_mutation = min(
            0.2,
            max(0.0, _safe_float(cfg.get("gamma_mutation", tuned.get("gamma_mutation", 0.02)), 0.02)),
        )
        self._epsilon_decay_mutation = min(
            0.1,
            max(
                0.0,
                _safe_float(
                    cfg.get("epsilon_decay_mutation", tuned.get("epsilon_decay_mutation", 0.015)),
                    0.015,
                ),
            ),
        )
        self._mutate_on_plateau = _safe_bool(cfg.get("mutate_on_plateau", True), True)

        self._episodes_seen = 0
        self._generation = 0
        self._window_returns: deque[float] = deque(maxlen=self._evolve_interval)
        self._best_window_return = float("-inf")
        self._best_params: Dict[str, float] = {
            "lr": float(self.lr),
            "gamma": float(self.gamma),
            "epsilon_decay": float(self.epsilon_decay),
        }
        self._last_event_code = 0.0

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        metrics = super().learn(batch)
        self._episodes_seen += 1

        ret = None
        if isinstance(batch.meta, dict):
            ret = batch.meta.get("return_sum")
        if isinstance(ret, (int, float)):
            self._window_returns.append(float(ret))

        event_code = 0.0  # idle
        if len(self._window_returns) >= self._evolve_interval:
            window_mean = float(sum(self._window_returns) / float(len(self._window_returns)))
            improved = (not math.isfinite(self._best_window_return)) or (
                window_mean >= (self._best_window_return + self._evolve_margin)
            )

            if improved:
                self._best_window_return = float(window_mean)
                self._best_params = self._current_params()
                event_code = 1.0  # promote
            elif self._mutate_on_plateau:
                self._mutate_from_best()
                self._generation += 1
                event_code = 2.0  # mutate

            self._window_returns.clear()

        self._last_event_code = float(event_code)

        metrics["evolution_generation"] = float(self._generation)
        metrics["evolution_event_code"] = float(event_code)
        metrics["evolution_best_window_return"] = float(
            0.0 if not math.isfinite(self._best_window_return) else self._best_window_return
        )
        metrics["evolution_lr"] = float(self.lr)
        metrics["evolution_gamma"] = float(self.gamma)
        metrics["evolution_epsilon_decay"] = float(self.epsilon_decay)
        return metrics

    def save(self, path: str) -> None:
        super().save(path)

        fp = os.path.join(path, "q_table.json")
        if not os.path.isfile(fp):
            return
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["evolving"] = {
            "episodes_seen": int(self._episodes_seen),
            "generation": int(self._generation),
            "window_returns": [float(x) for x in list(self._window_returns)],
            "best_window_return": (
                float(self._best_window_return) if math.isfinite(self._best_window_return) else None
            ),
            "best_params": {k: float(v) for k, v in self._best_params.items()},
            "last_event_code": float(self._last_event_code),
            "evolve_interval": int(self._evolve_interval),
        }

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        super().load(path)

        fp = os.path.join(path, "q_table.json")
        if not os.path.isfile(fp):
            return
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        evo = payload.get("evolving")
        if not isinstance(evo, dict):
            return

        self._episodes_seen = max(0, _safe_int(evo.get("episodes_seen", 0), 0))
        self._generation = max(0, _safe_int(evo.get("generation", 0), 0))

        raw_window = evo.get("window_returns")
        if isinstance(raw_window, list):
            self._window_returns = deque(
                [float(x) for x in raw_window if isinstance(x, (int, float))],
                maxlen=self._evolve_interval,
            )

        self._best_window_return = _safe_float(
            evo.get("best_window_return", self._best_window_return),
            self._best_window_return,
        )
        raw_best = evo.get("best_params")
        if isinstance(raw_best, dict):
            self._best_params = {
                "lr": _safe_float(raw_best.get("lr", self.lr), self.lr),
                "gamma": _safe_float(raw_best.get("gamma", self.gamma), self.gamma),
                "epsilon_decay": _safe_float(raw_best.get("epsilon_decay", self.epsilon_decay), self.epsilon_decay),
            }
        self._last_event_code = _safe_float(evo.get("last_event_code", 0.0), 0.0)

    def _mutate_from_best(self) -> None:
        base = dict(self._best_params) if self._best_params else self._current_params()
        scale = float(self._mutation_scale)

        lr_base = _safe_float(base.get("lr", self.lr), self.lr)
        gamma_base = _safe_float(base.get("gamma", self.gamma), self.gamma)
        decay_base = _safe_float(base.get("epsilon_decay", self.epsilon_decay), self.epsilon_decay)

        lr_mult = 1.0 + float(self._rng.uniform(-scale, scale))
        self.lr = float(min(1.0, max(1e-4, lr_base * lr_mult)))

        gamma_delta = float(self._rng.uniform(-self._gamma_mutation, self._gamma_mutation))
        self.gamma = float(min(0.9999, max(0.5, gamma_base + gamma_delta)))

        decay_delta = float(self._rng.uniform(-self._epsilon_decay_mutation, self._epsilon_decay_mutation))
        self.epsilon_decay = float(min(0.9999, max(0.85, decay_base + decay_delta)))

        # After mutation, reopen exploration slightly so new params can express.
        self.stats.epsilon = float(min(1.0, max(self.epsilon_min, self.stats.epsilon + 0.05)))

    def _current_params(self) -> Dict[str, float]:
        return {
            "lr": float(self.lr),
            "gamma": float(self.gamma),
            "epsilon_decay": float(self.epsilon_decay),
        }
