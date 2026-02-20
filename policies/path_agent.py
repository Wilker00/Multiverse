"""
policies/path_agent.py

An agent architecture that can load and compose pre-trained, frozen Skill Paths.
"""

from __future__ import annotations
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional

from policies.skill_path import SkillPath


def _action_key(action: Any) -> str:
    try:
        return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return repr(action)


def _try_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


class PathAgent:
    """
    Composes frozen SkillPath modules with a lightweight trainable gate.

    The gate keeps one scalar bias per skill and combines it with each skill's
    confidence estimate at inference time.
    """

    def __init__(
        self,
        skill_ids: List[str],
        skill_library_dir: str = "skills",
        controller_lr: float = 0.1,
        temperature: float = 1.0,
    ):
        if not skill_ids:
            raise ValueError("PathAgent requires at least one skill_id")
        self.loaded_skills = self._load_skills(skill_ids, skill_library_dir)
        self.controller_lr = max(0.0, min(1.0, float(controller_lr)))
        self.temperature = max(1e-6, float(temperature))
        self.controller_bias: Dict[str, float] = {skill.skill_id: 0.0 for skill in self.loaded_skills}
        self._last_selected_skill: Optional[str] = None
        self._last_action: Any = None

    def _resolve_skill_path(self, skill_id: str, skill_library_dir: str) -> str:
        raw = str(skill_id).strip()
        if not raw:
            raise ValueError("skill_id entries must be non-empty strings")

        if os.path.isfile(raw):
            return raw
        filename = raw if raw.endswith(".json") else f"{raw}.json"
        return os.path.join(skill_library_dir, filename)

    def _load_skills(self, skill_ids: List[str], skill_library_dir: str) -> List[SkillPath]:
        """Loads frozen skill modules from disk."""
        skills: List[SkillPath] = []
        for skill_id in skill_ids:
            path = self._resolve_skill_path(skill_id, skill_library_dir)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Skill file not found for '{skill_id}': {path}")
            skills.append(SkillPath.load(path))
        return skills

    def _softmax(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        total = sum(exps)
        if total <= 0.0:
            return [1.0 / float(len(scores)) for _ in scores]
        return [x / total for x in exps]

    def act(self, obs: Any) -> Any:
        """
        Selects an action by weighted voting across loaded skills.
        """
        predictions = [skill.predict(obs) for skill in self.loaded_skills]
        logits: List[float] = []
        confidences: List[float] = []
        for pred in predictions:
            sid = str(pred.get("skill_id", ""))
            conf = _try_float(pred.get("confidence"))
            clamped_conf = max(0.0, min(1.0, conf if conf is not None else 0.0))
            confidences.append(clamped_conf)
            logits.append((clamped_conf + self.controller_bias.get(sid, 0.0)) / self.temperature)

        gates = self._softmax(logits)

        action_scores: Dict[str, float] = {}
        action_values: Dict[str, Any] = {}
        for pred, gate, conf in zip(predictions, gates, confidences):
            action = pred.get("action")
            key = _action_key(action)
            action_values[key] = action
            action_scores[key] = action_scores.get(key, 0.0) + (gate * max(conf, 1e-6))

        if not action_scores:
            raise RuntimeError("PathAgent could not derive any action from loaded skills")

        best_action_key = max(action_scores.items(), key=lambda kv: kv[1])[0]
        self._last_action = action_values[best_action_key]

        best_skill_idx = max(range(len(gates)), key=lambda i: gates[i])
        self._last_selected_skill = self.loaded_skills[best_skill_idx].skill_id
        return self._last_action

    def _iter_records(self, transitions: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(transitions, dict):
            yield transitions
            return
        if isinstance(transitions, (list, tuple)):
            for item in transitions:
                if isinstance(item, dict):
                    yield item
                    continue
                rec: Dict[str, Any] = {}
                for key in ("reward", "skill_id", "selected_skill", "skill"):
                    if hasattr(item, key):
                        rec[key] = getattr(item, key)
                if rec:
                    yield rec

    def learn(self, transitions: Any) -> Dict[str, Any]:
        """
        Updates gating biases from reward-labeled transitions.
        """
        updates = 0
        reward_sum = 0.0
        for rec in self._iter_records(transitions):
            reward = _try_float(rec.get("reward"))
            if reward is None:
                continue

            skill_id = (
                rec.get("skill_id")
                or rec.get("selected_skill")
                or rec.get("skill")
                or self._last_selected_skill
            )
            sid = str(skill_id) if skill_id is not None else ""
            if sid not in self.controller_bias:
                continue

            self.controller_bias[sid] += self.controller_lr * math.tanh(reward)
            updates += 1
            reward_sum += reward

        return {
            "updated_skills": updates,
            "mean_reward": (reward_sum / float(updates)) if updates > 0 else 0.0,
            "controller_bias": dict(self.controller_bias),
        }
