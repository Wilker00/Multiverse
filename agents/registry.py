"""
agents/registry.py

Central registry for creating Agent instances from AgentSpec.
This decouples the Trainer from knowing about every specific Agent class.
"""

from typing import Callable, Dict

from core.types import AgentSpec, SpaceSpec
from core.agent_base import Agent

from agents.random_agent import RandomAgent
from agents.imitation_agent import ImitationLookupAgent
from agents.simple_pg_agent import SimplePolicyGradientAgent
from agents.q_agent import QLearningAgent
from agents.ppo_agent import PPOAgent
from agents.recurrent_ppo_agent import RecurrentPPOAgent
from agents.mpc_agent import MPCAgent
from agents.curious_agent import CuriousAgent
from agents.library_agent import LibraryAgent
from agents.distilled_agent import DistilledAgent
from agents.special_agent import SpecialAgent
from agents.special_moe_agent import SpecialMoEAgent
from agents.adaptive_moe import AdaptiveMoEAgent
from agents.cql_agent import CQLLookupAgent
from agents.gateway_agent import GatewayAgent
from agents.failure_aware_agent import FailureAwareAgent
from agents.aware_agent import AwareAgent
from agents.evolving_agent import EvolvingAgent
from agents.memory_recall_agent import MemoryRecallAgent
from agents.planner_recall_agent import PlannerRecallAgent
from agents.transformer_agent import TransformerAgent
from agents.sf_transfer_agent import SuccessorFeatureAgent

AgentFactory = Callable[[AgentSpec, SpaceSpec, SpaceSpec], Agent]

_AGENT_REGISTRY: Dict[str, AgentFactory] = {}


def register_agent(algo_name: str, factory: AgentFactory) -> None:
    """Register a new agent factory under a specific algorithm name."""
    _AGENT_REGISTRY[algo_name.lower()] = factory


def create_agent(
    spec: AgentSpec,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
) -> Agent:
    """
    Create an agent instance based on the spec.algo field.
    Raises ValueError if the algo is not registered.
    """
    algo = spec.algo.lower()
    if algo not in _AGENT_REGISTRY:
        raise ValueError(f"Unknown agent algorithm: '{algo}'. Available: {list(_AGENT_REGISTRY.keys())}")

    factory = _AGENT_REGISTRY[algo]
    return factory(spec, observation_space, action_space)


def register_builtin_agents() -> None:
    """Register the core agents provided by the library."""
    register_agent(
        "random",
        lambda s, o, a: RandomAgent(spec=s, observation_space=o, action_space=a),
    )

    register_agent(
        "imitation_lookup",
        lambda s, o, a: ImitationLookupAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "library",
        lambda s, o, a: LibraryAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "distilled",
        lambda s, o, a: DistilledAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "special",
        lambda s, o, a: SpecialAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "special_moe",
        lambda s, o, a: SpecialMoEAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "adaptive_moe",
        lambda s, o, a: AdaptiveMoEAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "gateway",
        lambda s, o, a: GatewayAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "cql",
        lambda s, o, a: CQLLookupAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "failure_aware",
        lambda s, o, a: FailureAwareAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "aware",
        lambda s, o, a: AwareAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "evolving",
        lambda s, o, a: EvolvingAgent(spec=s, observation_space=o, action_space=a),
    )

    register_agent(
        "simple_pg",
        lambda s, o, a: SimplePolicyGradientAgent(spec=s, observation_space=o, action_space=a),
    )

    register_agent(
        "q",
        lambda s, o, a: QLearningAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "sf_transfer",
        lambda s, o, a: SuccessorFeatureAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "memory_recall",
        lambda s, o, a: MemoryRecallAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "planner_recall",
        lambda s, o, a: PlannerRecallAgent(spec=s, observation_space=o, action_space=a),
    )
    register_agent(
        "adt",
        lambda s, o, a: TransformerAgent(spec=s, observation_space=o, action_space=a),
    )

    register_agent(
        "ppo",
        lambda s, o, a: PPOAgent(spec=s, observation_space=o, action_space=a),
    )

    register_agent(
        "recurrent_ppo",
        lambda s, o, a: RecurrentPPOAgent(spec=s, observation_space=o, action_space=a),
    )

    register_agent(
        "mpc",
        lambda s, o, a: MPCAgent(spec=s, observation_space=o, action_space=a),
    )

    register_agent(
        "curious_ppo",
        lambda s, o, a: CuriousAgent(spec=s, observation_space=o, action_space=a),
    )

    def _ppo_her_factory(s, o, a):
        # Force her_enabled in spec config
        cfg = dict(s.config) if s.config else {}
        cfg["her_enabled"] = True
        import dataclasses
        s2 = dataclasses.replace(s, config=cfg)
        return PPOAgent(spec=s2, observation_space=o, action_space=a)

    register_agent("ppo_her", _ppo_her_factory)

    def _recurrent_ppo_her_factory(s, o, a):
        cfg = dict(s.config) if s.config else {}
        cfg["her_enabled"] = True
        import dataclasses
        s2 = dataclasses.replace(s, config=cfg)
        return RecurrentPPOAgent(spec=s2, observation_space=o, action_space=a)

    register_agent("recurrent_ppo_her", _recurrent_ppo_her_factory)
