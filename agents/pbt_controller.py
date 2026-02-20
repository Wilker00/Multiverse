"""
agents/pbt_controller.py

Population Based Training (PBT) controller with memory consequence rules.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.types import AgentSpec


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@dataclass
class PBTRules:
    improve_mult_up: float = 1.10
    stagnation_mult_down: float = 0.92
    stagnation_patience: int = 2


@dataclass
class PBTConfig:
    enabled: bool = False
    population_size: int = 4
    exploit_interval: int = 2
    mutation_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "lr": (0.5, 1.5),
            "epsilon_decay": (0.9, 1.1),
            "gamma": (0.98, 1.01),
        }
    )
    memory_retention_multiplier_rules: PBTRules = field(default_factory=PBTRules)
    retention_file: str = os.path.join("central_memory", "retention_multipliers.json")
    seed: int = 123

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "PBTConfig":
        c = cfg if isinstance(cfg, dict) else {}
        rules_raw = c.get("memory_retention_multiplier_rules", {})
        rules_raw = rules_raw if isinstance(rules_raw, dict) else {}
        rules = PBTRules(
            improve_mult_up=max(1.0, _safe_float(rules_raw.get("improve_mult_up", 1.10), 1.10)),
            stagnation_mult_down=max(0.1, min(1.0, _safe_float(rules_raw.get("stagnation_mult_down", 0.92), 0.92))),
            stagnation_patience=max(1, _safe_int(rules_raw.get("stagnation_patience", 2), 2)),
        )
        mut = c.get("mutation_ranges", {})
        mut = mut if isinstance(mut, dict) else {}
        parsed: Dict[str, Tuple[float, float]] = {}
        for k, v in mut.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                parsed[str(k)] = (float(v[0]), float(v[1]))
        return PBTConfig(
            enabled=bool(c.get("enabled", False)),
            population_size=max(2, _safe_int(c.get("population_size", 4), 4)),
            exploit_interval=max(1, _safe_int(c.get("exploit_interval", 2), 2)),
            mutation_ranges=parsed or PBTConfig().mutation_ranges,
            memory_retention_multiplier_rules=rules,
            retention_file=str(c.get("retention_file", os.path.join("central_memory", "retention_multipliers.json"))),
            seed=_safe_int(c.get("seed", 123), 123),
        )


@dataclass
class Member:
    member_id: str
    agent_spec: AgentSpec
    score: float = -1e18
    best_score: float = -1e18
    stagnation_steps: int = 0
    retention_multiplier: float = 1.0


class PBTController:
    def __init__(self, cfg: Optional[PBTConfig] = None):
        self.cfg = cfg or PBTConfig()
        self.rng = random.Random(int(self.cfg.seed))
        self.population: List[Member] = []
        self.generation = 0
        os.makedirs(os.path.dirname(self.cfg.retention_file) or ".", exist_ok=True)

    def init_population(self, base_spec: AgentSpec) -> List[Member]:
        self.population = []
        for i in range(max(2, int(self.cfg.population_size))):
            cfg = dict(base_spec.config or {})
            # Small deterministic jitter for initial diversity.
            cfg = self._mutate_config(cfg, intensity=0.10)
            member_spec = base_spec.evolved(
                policy_id=f"{base_spec.policy_id}_pbt_{i}",
                config=cfg,
            )
            self.population.append(Member(member_id=f"m{i}", agent_spec=member_spec))
        self._write_retention_registry()
        return list(self.population)

    def update_scores(self, scores_by_member: Dict[str, float]) -> None:
        for m in self.population:
            if m.member_id not in scores_by_member:
                continue
            s = float(scores_by_member[m.member_id])
            m.score = s
            if s > m.best_score + 1e-9:
                m.best_score = s
                m.stagnation_steps = 0
                m.retention_multiplier *= float(self.cfg.memory_retention_multiplier_rules.improve_mult_up)
            else:
                m.stagnation_steps += 1
                if m.stagnation_steps >= int(self.cfg.memory_retention_multiplier_rules.stagnation_patience):
                    m.retention_multiplier *= float(self.cfg.memory_retention_multiplier_rules.stagnation_mult_down)
                    m.stagnation_steps = 0
            m.retention_multiplier = max(0.1, min(5.0, float(m.retention_multiplier)))
        self._write_retention_registry()

    def maybe_exploit_explore(self) -> Dict[str, Any]:
        self.generation += 1
        if self.generation % int(self.cfg.exploit_interval) != 0:
            return {"generation": self.generation, "changed": False}
        if len(self.population) < 2:
            return {"generation": self.generation, "changed": False}

        ranked = sorted(self.population, key=lambda m: float(m.score), reverse=True)
        half = max(1, len(ranked) // 2)
        elites = ranked[:half]
        laggards = ranked[half:]

        changes = []
        for lg in laggards:
            donor = self.rng.choice(elites)
            donor_cfg = dict(donor.agent_spec.config or {})
            new_cfg = self._mutate_config(copy.deepcopy(donor_cfg), intensity=0.25)
            lg.agent_spec = lg.agent_spec.evolved(config=new_cfg)
            lg.score = donor.score
            # Exploit+explore copied from donor; carry reduced retention until it proves itself.
            lg.retention_multiplier = max(0.1, donor.retention_multiplier * 0.95)
            changes.append({"member": lg.member_id, "donor": donor.member_id})

        self._write_retention_registry()
        return {"generation": self.generation, "changed": True, "changes": changes}

    def _mutate_config(self, cfg: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        out = dict(cfg)
        for key, (lo, hi) in self.cfg.mutation_ranges.items():
            if key not in out:
                continue
            try:
                cur = float(out[key])
            except Exception:
                continue
            span_lo = 1.0 + (float(lo) - 1.0) * float(intensity)
            span_hi = 1.0 + (float(hi) - 1.0) * float(intensity)
            mult = self.rng.uniform(span_lo, span_hi)
            out[key] = float(cur * mult)
        return out

    def _write_retention_registry(self) -> None:
        payload = {
            "updated_at_ms": int(time.time() * 1000),
            "generation": int(self.generation),
            "members": [
                {
                    "member_id": m.member_id,
                    "policy_id": m.agent_spec.policy_id,
                    "score": float(m.score),
                    "best_score": float(m.best_score),
                    "retention_multiplier": float(m.retention_multiplier),
                }
                for m in self.population
            ],
        }
        with open(self.cfg.retention_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

