# this is the base class for all agents in the system
"""
core/agent_base.py

Minimal agent interface for u.ai.

Rules:
- Keep this independent from any specific RL library.
- An Agent consumes observations (JSONValue) and outputs actions (JSONValue).
- Learning is optional at first, but the interface supports it cleanly.

This file defines:
- ActionResult: output of act()
- ExperienceBatch: a lightweight training batch format
- Agent protocol + a simple RandomAgent baseline you can delete later
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np
from core.types import AgentSpec, JSONValue, SpaceSpec


@dataclass
class ActionResult:
    """
    Result of an agent's action selection.
    Contains the action itself and any auxiliary info (e.g., log probs, value estimates).
    """
    action: JSONValue
    info: Dict[str, JSONValue] = field(default_factory=dict)


@dataclass
class Transition:
    """
    One transition, suitable for on-policy or off-policy learning.
    """
    obs: JSONValue
    action: JSONValue
    reward: float
    next_obs: JSONValue
    done: bool
    truncated: bool
    info: Dict[str, JSONValue] = field(default_factory=dict)


@dataclass
class ExperienceBatch:
    """
    A simple batch container.
    Algorithms can interpret this however they want.
    """
    transitions: List[Transition]
    meta: Dict[str, JSONValue] = field(default_factory=dict)


class Agent(Protocol):
    """
    Agent instances live in agents/*.
    The orchestrator should only rely on this interface.
    """

    spec: AgentSpec
    observation_space: SpaceSpec
    action_space: SpaceSpec

    def seed(self, seed: Optional[int]) -> None:
        """
        Set randomness seed for reproducibility.
        """

    def act(self, obs: JSONValue) -> ActionResult:
        """
        Choose an action based on an observation.
        Must return JSON serializable action.
        """

    def act_with_hint(self, obs: JSONValue, hint: Optional[Dict[str, Any]]) -> ActionResult:
        """
        Optional: Choose an action given an observation and a semantic hint.
        """
        return self.act(obs)

    def set_hindsight_goal(self, goal: JSONValue) -> None:
        """
        Optional: Update the agent's internal goal representation (HER).
        """
        pass


    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        """
        Optional: perform a learning update.
        Returns training metrics.
        Implementations may raise NotImplementedError if they are inference-only.
        """

    def save(self, path: str) -> None:
        """
        Optional: save model weights and metadata.
        """

    def load(self, path: str) -> None:
        """
        Optional: load model weights and metadata.
        """

    def close(self) -> None:
        """
        Optional cleanup hook.
        """

    def get_state(self) -> Dict[str, Any]:
        """
        Optional: return internal state (e.g., LSTM hidden states).
        Used by SafeExecutor for synchronized rewinds.
        """
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Optional: restore internal state.
        """
        pass


def _space_is_discrete(space: SpaceSpec) -> bool:
    return space.type == "discrete" and isinstance(space.n, int) and space.n > 0


def _space_is_continuous(space: SpaceSpec) -> bool:
    return space.type == "continuous" and isinstance(space.low, list) and isinstance(space.high, list)


class RandomAgent:
    """
    Debug baseline: samples random actions from the action space spec.

    Not smart. Very useful for testing Verse + rollout + logging.
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space
        self._rng = None
        self._seed = None

        import random
        self._random = random

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        self._rng = self._random.Random(seed)

    def act(self, obs: JSONValue) -> ActionResult:
        a = self._sample_action()
        return ActionResult(action=a, info={"agent": "random"})

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        raise NotImplementedError("RandomAgent does not learn")

    def save(self, path: str) -> None:
        return

    def load(self, path: str) -> None:
        return

    def close(self) -> None:
        return

    def _sample_action(self) -> JSONValue:
        if self._rng is None:
            self.seed(self._seed)

        space = self.action_space

        if _space_is_discrete(space):
            return int(self._rng.randrange(space.n))  # 0..n-1

        if _space_is_continuous(space):
            # Sample uniform per dimension
            low = space.low
            high = space.high
            if len(low) != len(high):
                raise ValueError("continuous SpaceSpec low/high must be same length")
            return [float(self._rng.uniform(low[i], high[i])) for i in range(len(low))]

        if space.type == "multi_discrete":
            # Interpret shape as tuple of sizes, ex: (3, 2, 5) means each dimension is discrete
            if not space.shape:
                raise ValueError("multi_discrete SpaceSpec requires shape like (n1, n2, ...)")
            return [int(self._rng.randrange(int(n))) for n in space.shape]

        if space.type == "dict":
            if not space.subspaces:
                raise ValueError("dict SpaceSpec requires subspaces")
            out: Dict[str, JSONValue] = {}
            for k, sub in space.subspaces.items():
                out[k] = RandomAgent(self.spec, self.observation_space, sub)._sample_action()
            return out

        raise ValueError(f"Unsupported action space type: {space.type}")


class RunningMeanStd:
    """Calculates running mean and variance for online normalization."""
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)



def apply_her(transitions: List[Transition], strategy: str = "final") -> List[Transition]:
    """
    Hindsight Experience Replay implementation.
    Generates synthetic transitions where the achieved state is treated as the goal.
    """
    if not transitions:
        return []
        
    her_transitions = []
    
    # Strategy: 'final' - treat the last state of the trajectory as the goal
    if strategy == "final":
        final_obs = transitions[-1].next_obs
        for tr in transitions:
            # Create a synthetic transition where next_obs == final_obs is the 'goal'
            # Note: This requires the agent/env to support goal-conditioned observations.
            # For now, we just tag the info for the agent to use during learn().
            synthetic_tr = Transition(
                obs=tr.obs,
                action=tr.action,
                reward=1.0 if tr.next_obs == final_obs else 0.0,
                next_obs=tr.next_obs,
                done=tr.next_obs == final_obs,
                truncated=tr.truncated,
                info={**tr.info, "her_goal": final_obs}
            )
            her_transitions.append(synthetic_tr)
            
    return her_transitions


