"""
agents/memory_recall_agent.py

Special agent with true on-demand recall:
- the agent can request memory lookups at specific steps,
- rollout executes the query and returns memory pointers + matches,
- agent uses recalled actions as a soft prior when selecting actions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np

from agents.q_agent import QLearningAgent, obs_key
from core.agent_base import ActionResult
from core.types import AgentSpec, JSONValue, SpaceSpec


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_set(raw: Any) -> Optional[Set[str]]:
    if raw is None:
        return None
    if isinstance(raw, (set, list, tuple)):
        out = set(str(x).strip().lower() for x in raw if str(x).strip())
        return out if out else None
    s = str(raw).strip().lower()
    if not s:
        return None
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    return set(parts) if parts else None


class MemoryRecallAgent(QLearningAgent):
    """
    Q-learning + on-demand memory recall controls.

    Config keys:
    - verse_name
    - recall_enabled (bool, default true)
    - recall_top_k (int, default 5)
    - recall_min_score (float, default -0.2)
    - recall_same_verse_only (bool, default true)
    - recall_memory_types (list/set/csv string)
    - recall_vote_weight (float, default 0.75)
    - recall_risk_key (str, default "risk")
    - recall_risk_threshold (float, default 6.0)
    - recall_uncertainty_margin (float, default 0.10)
    - recall_cooldown_steps (int, default 2)
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        super().__init__(spec=spec, observation_space=observation_space, action_space=action_space)
        cfg = spec.config if isinstance(spec.config, dict) else {}
        self._verse_name = str(cfg.get("verse_name", "")).strip().lower()
        self._recall_enabled = bool(cfg.get("recall_enabled", True))
        self._recall_top_k = max(1, _safe_int(cfg.get("recall_top_k", 5), 5))
        self._recall_min_score = _safe_float(cfg.get("recall_min_score", -0.2), -0.2)
        self._recall_same_verse_only = bool(cfg.get("recall_same_verse_only", True))
        self._recall_memory_types = _as_set(cfg.get("recall_memory_types"))
        self._recall_vote_weight = max(0.0, min(3.0, _safe_float(cfg.get("recall_vote_weight", 0.75), 0.75)))
        self._recall_risk_key = str(cfg.get("recall_risk_key", "risk")).strip() or "risk"
        self._recall_risk_threshold = _safe_float(cfg.get("recall_risk_threshold", 6.0), 6.0)
        self._recall_uncertainty_margin = max(
            0.0, _safe_float(cfg.get("recall_uncertainty_margin", 0.10), 0.10)
        )
        self._recall_cooldown_steps = max(1, _safe_int(cfg.get("recall_cooldown_steps", 2), 2))
        self._last_query_step = -10**9
        self._last_bundle: Optional[Dict[str, Any]] = None
        self._recall_uses = 0
        self._last_query_diag: Dict[str, Any] = {
            "step_idx": -1,
            "risk_value": None,
            "risk_threshold": float(self._recall_risk_threshold),
            "uncertainty_margin": None,
            "uncertainty_threshold": float(self._recall_uncertainty_margin),
            "trigger_high_risk": False,
            "trigger_uncertain": False,
            "request_emitted": False,
            "block_reason": "init",
        }

    def memory_query_request(self, *, obs: JSONValue, step_idx: int) -> Optional[Dict[str, Any]]:
        if not bool(self._recall_enabled):
            self._last_query_diag = {
                "step_idx": int(step_idx),
                "risk_value": None,
                "risk_threshold": float(self._recall_risk_threshold),
                "uncertainty_margin": None,
                "uncertainty_threshold": float(self._recall_uncertainty_margin),
                "trigger_high_risk": False,
                "trigger_uncertain": False,
                "request_emitted": False,
                "block_reason": "disabled",
            }
            return None
        step = int(step_idx)
        if (step - int(self._last_query_step)) < int(self._recall_cooldown_steps):
            self._last_query_diag = {
                "step_idx": int(step),
                "risk_value": None,
                "risk_threshold": float(self._recall_risk_threshold),
                "uncertainty_margin": None,
                "uncertainty_threshold": float(self._recall_uncertainty_margin),
                "trigger_high_risk": False,
                "trigger_uncertain": False,
                "request_emitted": False,
                "block_reason": "cooldown",
            }
            return None

        risk_value = None
        if isinstance(obs, dict) and self._recall_risk_key in obs:
            risk_value = _safe_float(obs.get(self._recall_risk_key), None)  # type: ignore[arg-type]
        trigger_risk = bool(risk_value is not None and float(risk_value) >= float(self._recall_risk_threshold))

        qvals = self._get_q(obs_key(obs))
        if int(self.n_actions) <= 1:
            margin = 1.0
        else:
            ranked = np.sort(np.asarray(qvals, dtype=np.float32))
            margin = float(ranked[-1] - ranked[-2])
        trigger_uncertain = bool(margin <= float(self._recall_uncertainty_margin))

        if not (trigger_risk or trigger_uncertain):
            self._last_query_diag = {
                "step_idx": int(step),
                "risk_value": (None if risk_value is None else float(risk_value)),
                "risk_threshold": float(self._recall_risk_threshold),
                "uncertainty_margin": float(margin),
                "uncertainty_threshold": float(self._recall_uncertainty_margin),
                "trigger_high_risk": bool(trigger_risk),
                "trigger_uncertain": bool(trigger_uncertain),
                "request_emitted": False,
                "block_reason": "below_trigger",
            }
            return None

        reason = "high_risk" if trigger_risk else "uncertain_state"
        req = {
            "query_obs": obs,
            "top_k": int(self._recall_top_k),
            "min_score": float(self._recall_min_score),
            "verse_name": (self._verse_name if bool(self._recall_same_verse_only and self._verse_name) else None),
            "memory_types": (sorted(list(self._recall_memory_types)) if self._recall_memory_types else None),
            "reason": str(reason),
        }
        self._last_query_step = int(step)
        self._last_query_diag = {
            "step_idx": int(step),
            "risk_value": (None if risk_value is None else float(risk_value)),
            "risk_threshold": float(self._recall_risk_threshold),
            "uncertainty_margin": float(margin),
            "uncertainty_threshold": float(self._recall_uncertainty_margin),
            "trigger_high_risk": bool(trigger_risk),
            "trigger_uncertain": bool(trigger_uncertain),
            "request_emitted": True,
            "block_reason": "emitted",
        }
        return req

    def on_memory_response(self, payload: Dict[str, Any]) -> None:
        self._last_bundle = payload if isinstance(payload, dict) else None

    def act(self, obs: JSONValue) -> ActionResult:
        return self.act_with_hint(obs, None)

    def act_with_hint(self, obs: JSONValue, hint: Optional[Dict[str, Any]]) -> ActionResult:
        qvals = np.asarray(self._get_q(obs_key(obs)), dtype=np.float32).copy()
        recall = self._extract_recall_bundle(hint=hint)
        recall_prior = self._memory_action_prior(recall)
        recall_used = False
        recall_reason = ""
        pointer = ""
        if recall_prior is not None and float(np.max(recall_prior)) > 0.0:
            qvals = qvals + (float(self._recall_vote_weight) * recall_prior)
            recall_used = True
            self._recall_uses += 1
            recall_reason = str(recall.get("reason", "")) if isinstance(recall, dict) else ""
            pointer = self._top_pointer(recall)

        if self._rng.random() < self.stats.epsilon:
            a = int(self._rng.integers(0, self.n_actions))
            mode = "explore"
        else:
            a = int(np.argmax(qvals))
            mode = "exploit"

        return ActionResult(
            action=a,
            info={
                "mode": mode,
                "epsilon": float(self.stats.epsilon),
                "memory_recall_used": bool(recall_used),
                "memory_recall_reason": str(recall_reason),
                "memory_recall_pointer": str(pointer),
                "memory_recall_uses": int(self._recall_uses),
                "memory_recall_diag": dict(self._last_query_diag),
                "memory_recall_match_count": int(
                    len((recall or {}).get("matches", [])) if isinstance((recall or {}).get("matches"), list) else 0
                ),
            },
        )

    def _extract_recall_bundle(self, *, hint: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if isinstance(hint, dict):
            raw = hint.get("memory_recall")
            if isinstance(raw, dict):
                return raw
        if isinstance(self._last_bundle, dict):
            return self._last_bundle
        return None

    def _memory_action_prior(self, recall: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        if not isinstance(recall, dict):
            return None
        matches = recall.get("matches")
        if not isinstance(matches, list) or not matches:
            return None
        prior = np.zeros((self.n_actions,), dtype=np.float32)
        for row in matches:
            if not isinstance(row, dict):
                continue
            a = _safe_int(row.get("action"), -1)
            if a < 0 or a >= int(self.n_actions):
                continue
            score = max(0.0, _safe_float(row.get("score"), 0.0))
            prior[a] += float(score)
        mx = float(np.max(prior))
        if mx <= 0.0:
            return None
        return prior / mx

    def _top_pointer(self, recall: Optional[Dict[str, Any]]) -> str:
        if not isinstance(recall, dict):
            return ""
        matches = recall.get("matches")
        if not isinstance(matches, list) or not matches:
            return ""
        row = matches[0]
        if not isinstance(row, dict):
            return ""
        return str(row.get("pointer_path", ""))
