"""
agents/registry.py

Central registry for creating Agent instances from AgentSpec.
This decouples the Trainer from knowing about every specific Agent class.
"""

import json
import os
from typing import Any, Callable, Dict
import numpy as np
import torch
import torch.nn.functional as F

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
from agents.blackjack_basic_agent import BlackjackBasicAgent
from agents.blackjack_warmstart import pretrain_blackjack_dqn
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
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space
        cfg = spec.config if isinstance(spec.config, dict) else {}
        self._train_enabled = bool(cfg.get("train", False))
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
            double_dqn=bool(cfg.get("double_dqn", False)),
            prioritized_replay=bool(cfg.get("prioritized_replay", False)),
            prioritized_alpha=float(cfg.get("prioritized_alpha", 0.6)),
            prioritized_beta0=float(cfg.get("prioritized_beta0", 0.4)),
            prioritized_beta_steps=int(cfg.get("prioritized_beta_steps", 10000)),
        )
        self._verse_name = str(verse_name or "").strip().lower()
        self._warmstart_stats = {"enabled": False, "samples": 0, "epochs": 0, "loss": None}
        model_path = self._resolve_checkpoint_path(str(cfg.get("model_path", "") or "").strip())
        self._loaded_model_path = None
        if model_path:
            self.load(model_path)
            if (not self._train_enabled) and ("epsilon" in cfg):
                self._epsilon = float(cfg.get("epsilon", self._epsilon))
        elif self._verse_name == "blackjack_world":
            warmstart_enabled = bool(cfg.get("blackjack_warmstart", True))
            if warmstart_enabled:
                self._warmstart_stats = pretrain_blackjack_dqn(
                    self,
                    epochs=int(cfg.get("blackjack_warmstart_epochs", 6)),
                    samples=int(cfg.get("blackjack_warmstart_samples", 4096)),
                    batch_size=int(cfg.get("blackjack_warmstart_batch_size", 256)),
                    seed=int(cfg.get("blackjack_warmstart_seed", spec.seed or 17)),
                )

    @staticmethod
    def _resolve_checkpoint_path(path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        if os.path.isdir(raw):
            return os.path.join(raw, "dqn.pt")
        if os.path.splitext(raw)[1]:
            return raw
        return os.path.join(raw, "dqn.pt")

    @staticmethod
    def _resolve_checkpoint_dir(path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            raise ValueError("checkpoint path is required")
        if os.path.isdir(raw) or not os.path.splitext(raw)[1]:
            return raw
        return os.path.dirname(raw)

    def _legal_actions_from_obs(self, obs):
        if self._verse_name == "blackjack_world" and isinstance(obs, dict):
            total = int(obs.get("player_sum", 0) or 0)
            if total >= 21:
                return [1]
            legal = [0, 1]
            if int(obs.get("can_double", 0) or 0):
                legal.append(2)
            if int(obs.get("can_split", 0) or 0):
                legal.append(3)
            return sorted(set(int(a) for a in legal))
        return list(range(self._dqn.n_actions))

    def seed(self, seed):
        import random as _random
        _random.seed(seed)

    def act(self, obs):
        from core.agent_base import ActionResult
        legal = self._legal_actions_from_obs(obs)
        action = self._dqn.select_action(obs, legal, epsilon=self._epsilon)
        return ActionResult(action=action, info={"epsilon": self._epsilon, "warmstart": dict(self._warmstart_stats)})

    def learn(self, batch):
        if not batch.transitions:
            return {}
        losses = []
        for tr in batch.transitions:
            legal_next = self._legal_actions_from_obs(tr.next_obs)
            self._dqn.store(
                tr.obs, int(tr.action), float(tr.reward),
                tr.next_obs, bool(tr.done or tr.truncated), legal_next
            )
            loss = self._dqn.train_step()
            if loss is not None:
                losses.append(float(loss))
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        mean_loss = float(sum(losses) / len(losses)) if losses else 0.0
        metrics = {"dqn_loss": mean_loss, "epsilon": self._epsilon, "buffer_size": len(self._dqn.buffer)}
        if self._warmstart_stats.get("enabled"):
            metrics["warmstart_loss"] = self._warmstart_stats.get("loss")
        return metrics

    def learn_from_dataset(self, dataset_path: str) -> Dict[str, Any]:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"dataset not found: {dataset_path}")

        cfg = self.spec.config if isinstance(self.spec.config, dict) else {}
        rows_added = 0
        legal_rows = 0
        train_steps = 0
        losses = []
        bc_obs = []
        bc_actions = []
        bc_legals = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if not isinstance(row, dict) or "obs" not in row or "action" not in row:
                    continue
                try:
                    action = int(row.get("action"))
                except Exception:
                    continue
                if action < 0 or action >= int(self._dqn.n_actions):
                    continue

                obs = row.get("obs")
                next_obs = row.get("next_obs", obs)
                reward = float(row.get("reward", 0.0) or 0.0)
                done = bool(row.get("done", False) or row.get("truncated", False))

                next_legal_raw = row.get("legal_actions")
                next_legal = []
                if isinstance(next_legal_raw, list):
                    for item in next_legal_raw:
                        try:
                            ai = int(item)
                        except Exception:
                            continue
                        if 0 <= ai < int(self._dqn.n_actions):
                            next_legal.append(ai)
                if next_legal:
                    legal_rows += 1
                else:
                    next_legal = self._legal_actions_from_obs(next_obs)

                self._dqn.store(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    next_legal=next_legal,
                )
                rows_added += 1
                bc_obs.append(obs)
                bc_actions.append(int(action))
                bc_legals.append(self._legal_actions_from_obs(obs))

        if rows_added <= 0:
            return {
                "dataset_path": str(dataset_path).replace("\\", "/"),
                "dataset_rows_added": 0,
                "dataset_train_steps": 0,
                "dataset_mean_loss": 0.0,
            }

        max_steps = int(cfg.get("dataset_train_steps", 0) or 0)
        if max_steps <= 0:
            max_steps = min(int(rows_added), 5000)
        for _ in range(max_steps):
            loss = self._dqn.train_step()
            if loss is None:
                break
            losses.append(float(loss))
            train_steps += 1

        behavior_clone_loss = None
        bc_epochs = int(cfg.get("behavior_clone_epochs", 0) or 0)
        bc_batch_size = int(cfg.get("behavior_clone_batch_size", 256) or 256)
        if bc_epochs > 0 and bc_obs:
            x = np.stack([self._dqn.encode(obs) for obs in bc_obs]).astype(np.float32)
            y = np.asarray(bc_actions, dtype=np.int64)
            legal_masks = []
            for legal in bc_legals:
                mask = np.full((self._dqn.n_actions,), -1e9, dtype=np.float32)
                if legal:
                    for action_id in legal:
                        if 0 <= int(action_id) < self._dqn.n_actions:
                            mask[int(action_id)] = 0.0
                legal_masks.append(mask)
            mask_arr = np.stack(legal_masks).astype(np.float32)
            x_t = torch.from_numpy(x).to(self._dqn.device)
            y_t = torch.from_numpy(y).to(self._dqn.device)
            mask_t = torch.from_numpy(mask_arr).to(self._dqn.device)
            rng = np.random.default_rng(int(cfg.get("behavior_clone_seed", self.spec.seed or 17)))
            idxs = np.arange(len(bc_obs))
            bc_losses = []
            for _ in range(bc_epochs):
                rng.shuffle(idxs)
                for start in range(0, len(idxs), bc_batch_size):
                    batch_idx = idxs[start : start + bc_batch_size]
                    xb = x_t[batch_idx]
                    yb = y_t[batch_idx]
                    mb = mask_t[batch_idx]
                    logits = self._dqn.q_net(xb) + mb
                    loss = F.cross_entropy(logits, yb)
                    self._dqn.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._dqn.q_net.parameters(), 10.0)
                    self._dqn.optimizer.step()
                    bc_losses.append(float(loss.item()))
            self._dqn.target_net.load_state_dict(self._dqn.q_net.state_dict())
            behavior_clone_loss = float(sum(bc_losses) / len(bc_losses)) if bc_losses else None

        result = {
            "dataset_path": str(dataset_path).replace("\\", "/"),
            "dataset_rows_added": int(rows_added),
            "dataset_legal_rows": int(legal_rows),
            "dataset_train_steps": int(train_steps),
            "dataset_mean_loss": (float(sum(losses) / len(losses)) if losses else 0.0),
            "buffer_size": int(len(self._dqn.buffer)),
        }
        if behavior_clone_loss is not None:
            result["behavior_clone_loss"] = float(behavior_clone_loss)
            result["behavior_clone_epochs"] = int(bc_epochs)
        return result

    def save(self, path: str) -> None:
        checkpoint_file = self._resolve_checkpoint_path(path)
        checkpoint_dir = self._resolve_checkpoint_dir(path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._dqn.save(checkpoint_file)
        state = {
            "epsilon": float(self._epsilon),
            "epsilon_decay": float(self._epsilon_decay),
            "epsilon_min": float(self._epsilon_min),
            "verse_name": str(self._verse_name),
            "warmstart": dict(self._warmstart_stats),
        }
        with open(os.path.join(checkpoint_dir, "adapter_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        checkpoint_file = self._resolve_checkpoint_path(path)
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(f"dqn checkpoint not found: {checkpoint_file}")
        checkpoint_dir = self._resolve_checkpoint_dir(path)
        self._dqn.load(checkpoint_file)
        state_path = os.path.join(checkpoint_dir, "adapter_state.json")
        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                self._epsilon = float(state.get("epsilon", self._epsilon))
                self._epsilon_decay = float(state.get("epsilon_decay", self._epsilon_decay))
                self._epsilon_min = float(state.get("epsilon_min", self._epsilon_min))
                warm = state.get("warmstart")
                if isinstance(warm, dict):
                    self._warmstart_stats = dict(warm)
        self._loaded_model_path = str(checkpoint_file).replace("\\", "/")

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
        "blackjack_basic",
        lambda s, o, a: BlackjackBasicAgent(spec=s, observation_space=o, action_space=a),
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
