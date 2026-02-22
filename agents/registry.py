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
from agents.dqn_agent import DQNAgent, ACTION_COUNTS, _GENERIC_DIM

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


class DQNTrainerAdapter:
    """
    Adapts DQNAgent's unique API (legal_actions, store/train_step)
    to the standard Agent protocol used by the Trainer/rollout loop.
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        cfg = spec.config if isinstance(spec.config, dict) else {}
        verse_name = cfg.get("verse_name", "")
        n_actions = int(action_space.n) if action_space and hasattr(action_space, "n") and action_space.n else 0

        self._epsilon = float(cfg.get("epsilon", 0.3))
        self._epsilon_decay = float(cfg.get("epsilon_decay", 0.995))
        self._epsilon_min = float(cfg.get("epsilon_min", 0.05))

        self._dqn = DQNAgent(
            verse_name=verse_name or "__generic__",
            hidden=int(cfg.get("hidden", 128)),
            lr=float(cfg.get("lr", 1e-3)),
            gamma=float(cfg.get("gamma", 0.95)),
            buffer_size=int(cfg.get("buffer_size", 20000)),
            batch_size=int(cfg.get("batch_size", 64)),
            target_update_freq=int(cfg.get("target_update_freq", 50)),
            obs_dim=_GENERIC_DIM,
            n_actions=n_actions if n_actions > 0 else ACTION_COUNTS.get(verse_name, 4),
        )

    def seed(self, seed):
        import random as _random
        _random.seed(seed)

    def act(self, obs):
        from core.agent_base import ActionResult
        legal = list(range(self._dqn.n_actions))
        action = self._dqn.select_action(obs, legal, epsilon=self._epsilon)
        return ActionResult(action=action, info={"epsilon": self._epsilon})

    def learn(self, batch):
        if not batch.transitions:
            return {}
        losses = []
        for tr in batch.transitions:
            legal_next = list(range(self._dqn.n_actions))
            self._dqn.store(
                tr.obs, int(tr.action), float(tr.reward),
                tr.next_obs, bool(tr.done or tr.truncated), legal_next
            )
            loss = self._dqn.train_step()
            if loss is not None:
                losses.append(float(loss))
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        mean_loss = float(sum(losses) / len(losses)) if losses else 0.0
        return {"dqn_loss": mean_loss, "epsilon": self._epsilon, "buffer_size": len(self._dqn.buffer)}

    def close(self):
        pass

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

    def _dqn_factory(s: AgentSpec, o: SpaceSpec, a: SpaceSpec) -> DQNTrainerAdapter:
        cfg = dict(s.config) if isinstance(s.config, dict) else {}
        if "verse_name" not in cfg:
            # Infer verse_name from the verse being trained (injected at trainer level)
            cfg["verse_name"] = cfg.get("__verse__", "")
        import dataclasses
        s2 = dataclasses.replace(s, config=cfg)
        return DQNTrainerAdapter(s2, o, a)

    register_agent("dqn", _dqn_factory)
