"""
policies/skill_path.py

Defines reusable, frozen neural submodules (Skill Paths) inspired by PathNet.
These are trained on specific, high-value DNA segments to master a narrow skill.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _obs_key(obs: Any) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return repr(obs)


def _action_key(action: Any) -> str:
    try:
        return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return repr(action)


def _safe_confidence(value: Any, default: float = 0.5) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = float(default)
    return max(0.0, min(1.0, conf))


def _normalize_tags(tags: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in tags:
        txt = str(tag).strip()
        if not txt or txt in seen:
            continue
        out.append(txt)
        seen.add(txt)
    return out


class SkillPath:
    """A frozen, reusable skill submodule with a lightweight tabular policy."""

    def __init__(self, skill_id: str, config: Dict[str, Any], weights: Any, tags: List[str]):
        sid = str(skill_id).strip()
        if not sid:
            raise ValueError("skill_id must be non-empty")
        self.skill_id = sid
        self.config = dict(config or {})
        self.weights = weights
        self.tags = _normalize_tags(tags or [])

    def save(self, skill_library_dir: str) -> str:
        """Saves the skill path's config, weights, and tags."""
        path = os.path.join(skill_library_dir, f"{self.skill_id}.json")
        os.makedirs(skill_library_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "skill_id": self.skill_id,
                    "config": self.config,
                    "weights": self.weights,
                    "tags": self.tags,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return path

    @staticmethod
    def load(skill_path: str) -> "SkillPath":
        """Loads a SkillPath from a file."""
        with open(skill_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        weights = data.get("weights")
        if weights is None:
            weights = {
                "type": "tabular_vote",
                "default_action": data.get("default_action"),
                "default_confidence": 0.5,
                "obs_policy": {},
            }

        return SkillPath(
            skill_id=data["skill_id"],
            config=data.get("config", {}),
            weights=weights,
            tags=data.get("tags", []),
        )

    def predict(self, obs: Any) -> Dict[str, Any]:
        """Predicts an action with an associated confidence score."""
        weights = self.weights if isinstance(self.weights, dict) else {}
        default_action = weights.get("default_action")
        default_conf = _safe_confidence(weights.get("default_confidence", 0.5), 0.5)

        obs_policy = weights.get("obs_policy", {})
        action = default_action
        confidence = default_conf

        if isinstance(obs_policy, dict):
            entry = obs_policy.get(_obs_key(obs))
            if isinstance(entry, dict):
                action = entry.get("action", action)
                confidence = _safe_confidence(entry.get("confidence", confidence), confidence)

        return {
            "skill_id": self.skill_id,
            "action": action,
            "confidence": confidence,
        }


@dataclass
class SkillPathConfig:
    """Configuration for creating a SkillPath."""

    dna_log_path: str
    skill_id: str
    # Tags inherited from the verse where the DNA was generated
    source_verse_tags: List[str] = field(default_factory=list)
    # Filter criteria
    min_advantage: float = 1.0
    # Training parameters
    epochs: int = 10

def create_skill_path(cfg: SkillPathConfig) -> SkillPath:
    """
    Factory function to train and freeze a SkillPath from a DNA log.
    """
    if not os.path.isfile(cfg.dna_log_path):
        raise FileNotFoundError(f"DNA log not found: {cfg.dna_log_path}")

    training_data: List[Dict[str, Any]] = []
    with open(cfg.dna_log_path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            try:
                dna = json.loads(row)
            except json.JSONDecodeError:
                continue
            if not isinstance(dna, dict):
                continue
            try:
                advantage = float(dna.get("advantage", 0.0))
            except Exception:
                continue
            if advantage < float(cfg.min_advantage):
                continue
            if "action" not in dna:
                continue
            training_data.append(
                {
                    "obs": dna.get("obs"),
                    "action": dna.get("action"),
                }
            )

    if not training_data:
        raise ValueError("No suitable DNA found for skill path training.")

    action_lookup: Dict[str, Any] = {}
    global_counts: Counter[str] = Counter()
    obs_action_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for sample in training_data:
        obs = sample.get("obs")
        action = sample.get("action")
        key = _action_key(action)
        action_lookup[key] = action
        global_counts[key] += 1
        obs_action_counts[_obs_key(obs)][key] += 1

    default_action_key, default_count = global_counts.most_common(1)[0]
    default_action = action_lookup[default_action_key]
    default_conf = float(default_count) / float(len(training_data))

    obs_policy: Dict[str, Dict[str, Any]] = {}
    for obs_k, counts in obs_action_counts.items():
        best_action_key, best_count = counts.most_common(1)[0]
        total = sum(int(v) for v in counts.values())
        obs_policy[obs_k] = {
            "action": action_lookup[best_action_key],
            "confidence": float(best_count) / float(total) if total > 0 else 0.5,
        }

    frozen_weights = {
        "type": "tabular_vote",
        "default_action": default_action,
        "default_confidence": default_conf,
        "obs_policy": obs_policy,
        "action_counts": {k: int(v) for k, v in global_counts.items()},
    }

    skill_path = SkillPath(
        skill_id=cfg.skill_id,
        config={
            "network_arch": "tabular_lookup",
            "epochs": int(cfg.epochs),
            "min_advantage": float(cfg.min_advantage),
            "num_examples": int(len(training_data)),
        },
        weights=frozen_weights,
        tags=cfg.source_verse_tags,
    )

    return skill_path
