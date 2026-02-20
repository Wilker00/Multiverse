"""
experiment/config.py

Experiment configuration helpers for u.ai.

This module keeps experiments portable and JSON serializable while
reusing core VerseSpec and AgentSpec definitions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from core.types import VerseSpec, AgentSpec


@dataclass
class ExperimentConfig:
    """
    Minimal experiment configuration.
    """
    name: str
    verse_spec: VerseSpec
    agent_spec: AgentSpec
    episodes: int = 50
    max_steps: int = 40
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "verse_spec": self.verse_spec.to_dict(),
            "agent_spec": self.agent_spec.to_dict(),
            "episodes": int(self.episodes),
            "max_steps": int(self.max_steps),
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExperimentConfig":
        return ExperimentConfig(
            name=str(data.get("name", "experiment")),
            verse_spec=VerseSpec(**data.get("verse_spec", {})),
            agent_spec=AgentSpec(**data.get("agent_spec", {})),
            episodes=int(data.get("episodes", 50)),
            max_steps=int(data.get("max_steps", 40)),
            seed=data.get("seed"),
        )


def load_experiment(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return ExperimentConfig.from_dict(payload)


def save_experiment(cfg: ExperimentConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
