"""
agents/planner_recall_agent.py

Planner-style recall agent:
- chooses which memory family to query by episode phase
  (declarative early/mid, procedural late by default),
- can force recall every eligible step for high-control specialist workflows.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.types import AgentSpec, JSONValue, SpaceSpec

from agents.memory_recall_agent import MemoryRecallAgent, _safe_float, _safe_int


def _norm_family(x: Any) -> str:
    s = str(x).strip().lower()
    if s in ("declarative", "procedural"):
        return s
    return ""


class PlannerRecallAgent(MemoryRecallAgent):
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        super().__init__(spec=spec, observation_space=observation_space, action_space=action_space)
        cfg = spec.config if isinstance(spec.config, dict) else {}
        self._planner_force_recall = bool(cfg.get("planner_force_recall", False))
        self._planner_phase_key = str(cfg.get("planner_phase_key", "t")).strip() or "t"
        self._planner_expected_horizon = max(1, _safe_int(cfg.get("planner_expected_horizon", 24), 24))
        self._planner_mid_start = max(0.0, min(0.95, _safe_float(cfg.get("planner_mid_start", 0.33), 0.33)))
        self._planner_late_start = max(
            float(self._planner_mid_start),
            min(1.0, _safe_float(cfg.get("planner_late_start", 0.66), 0.66)),
        )
        self._planner_family_early = _norm_family(cfg.get("planner_family_early", "declarative")) or "declarative"
        self._planner_family_mid = _norm_family(cfg.get("planner_family_mid", "declarative")) or "declarative"
        self._planner_family_late = _norm_family(cfg.get("planner_family_late", "procedural")) or "procedural"

    def memory_query_request(self, *, obs: JSONValue, step_idx: int) -> Optional[Dict[str, Any]]:
        if not bool(self._recall_enabled):
            return None
        phase = self._phase_name(obs)
        family = self._family_for_phase(phase)

        if bool(self._planner_force_recall):
            step = int(step_idx)
            if (step - int(self._last_query_step)) < int(self._recall_cooldown_steps):
                return None
            req = {
                "query_obs": obs,
                "top_k": int(self._recall_top_k),
                "min_score": float(self._recall_min_score),
                "verse_name": (self._verse_name if bool(self._recall_same_verse_only and self._verse_name) else None),
                "memory_types": (sorted(list(self._recall_memory_types)) if self._recall_memory_types else None),
                "memory_families": [str(family)],
                "reason": f"phase_{phase}_planner",
            }
            self._last_query_step = int(step)
            return req

        req = super().memory_query_request(obs=obs, step_idx=step_idx)
        if not isinstance(req, dict):
            return req
        req["reason"] = f"{str(req.get('reason', 'agent_request'))}:phase_{phase}"
        req["memory_families"] = [str(family)]
        return req

    def _phase_name(self, obs: JSONValue) -> str:
        t = 0.0
        if isinstance(obs, dict):
            t = _safe_float(obs.get(self._planner_phase_key, obs.get("t", 0.0)), 0.0)
        ratio = max(0.0, min(1.0, float(t / float(max(1, self._planner_expected_horizon)))))
        if ratio < float(self._planner_mid_start):
            return "early"
        if ratio < float(self._planner_late_start):
            return "mid"
        return "late"

    def _family_for_phase(self, phase: str) -> str:
        p = str(phase).strip().lower()
        if p == "early":
            return str(self._planner_family_early)
        if p == "mid":
            return str(self._planner_family_mid)
        return str(self._planner_family_late)
