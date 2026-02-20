"""
agents/special_agent.py

Special Agent that blends positive DNA (imitation) with boundary logic (avoid bad DNA).
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from agents.imitation_agent import ImitationLookupAgent, obs_key
from core.agent_base import ActionResult
from core.types import AgentSpec, JSONValue, SpaceSpec
from memory.boundary import load_bad_obs


class SpecialAgent(ImitationLookupAgent):
    """
    Uses imitation from good DNA and avoids bad DNA states by sampling alternatives.
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        super().__init__(spec=spec, observation_space=observation_space, action_space=action_space)
        self._bad_obs: Set[str] = set()

        cfg = spec.config if isinstance(spec.config, dict) else {}
        bad_path = cfg.get("bad_dna_path")
        if bad_path:
            self._bad_obs = load_bad_obs(str(bad_path))

    def act(self, obs: JSONValue) -> ActionResult:
        k = obs_key(obs)
        if k in self._bad_obs:
            action = self._sample_random_action()
            return ActionResult(action=action, info={"mode": "boundary_avoid"})
        return super().act(obs)
