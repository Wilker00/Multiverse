"""
orchestrator/mcts_trainer.py

Search-guided trainer:
- Uses neural-guided MCTS to produce improved policy targets (visit counts).
- Trains MetaTransformer policy/value heads to imitate search.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.mcts_search import MCTSConfig, MCTSSearch
from core.types import JSONValue, VerseSpec
from verses.registry import create_verse, register_builtin as register_builtin_verses


@dataclass
class MCTSTrainerConfig:
    episodes: int = 40
    max_steps: int = 80
    num_simulations: int = 96
    search_depth: int = 10
    c_puct: float = 1.4
    discount: float = 0.99
    temperature: float = 1.0
    value_loss_weight: float = 0.50
    train_epochs_per_episode: int = 1
    batch_size: int = 128
    replay_capacity: int = 50000
    lr: float = 1e-3
    seed: int = 123
    checkpoint_path: Optional[str] = None
    checkpoint_every_episodes: int = 0
    checkpoint_versioned_dir: Optional[str] = None
    trace_out_path: Optional[str] = None
    high_quality_trace_filter: bool = True
    min_quality_value_gain: float = 0.03
    min_quality_policy_shift_l1: float = 0.12
    min_quality_kl_divergence: float = 0.01
    min_forced_loss_prior_mass: float = 0.20
    min_forced_loss_mass_drop: float = 0.06
    sparse_replay_min_samples: int = 32
    sparse_replay_batch_size: int = 32


@dataclass
class _ReplayRow:
    state: List[float]
    history: List[List[float]]
    target_policy: List[float]
    target_value: float


class MCTSTrainer:
    def __init__(
        self,
        *,
        verse_spec: VerseSpec,
        model: Any,
        config: Optional[MCTSTrainerConfig] = None,
        policy_prior: Optional[Any] = None,
    ):
        self.verse_spec = verse_spec
        self.model = model
        self.config = config or MCTSTrainerConfig()
        self.policy_prior = policy_prior
        self.replay: List[_ReplayRow] = []

        random.seed(int(self.config.seed))

        try:
            import torch
        except Exception as e:
            raise RuntimeError("MCTSTrainer requires torch.") from e
        self._torch = torch

        self._state_dim = int(getattr(self.model, "state_dim", 0) or 0)
        self._action_dim = int(getattr(self.model, "action_dim", 0) or 0)
        self._context_dim = int(getattr(self.model, "context_input_dim", self._state_dim + 2))
        if self._state_dim <= 0 or self._action_dim <= 0:
            raise ValueError("Model must expose positive state_dim/action_dim for MCTSTrainer.")

        self._opt = torch.optim.AdamW(self.model.parameters(), lr=float(self.config.lr), weight_decay=1e-2)
        self._mse = torch.nn.MSELoss()

        from memory.embeddings import obs_to_vector

        self._obs_to_vector = obs_to_vector

    def run(self) -> Dict[str, float]:
        register_builtin_verses()
        verse = create_verse(self.verse_spec)
        action_count = int(getattr(getattr(verse, "action_space", None), "n", 0) or 0)
        if action_count <= 0:
            raise ValueError("MCTSTrainer requires discrete action space with n > 0.")

        search = MCTSSearch(
            verse=verse,
            config=MCTSConfig(
                num_simulations=int(self.config.num_simulations),
                max_depth=int(self.config.search_depth),
                c_puct=float(self.config.c_puct),
                discount=float(self.config.discount),
                seed=int(self.config.seed),
            ),
        )

        total_return = 0.0
        total_steps = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        update_count = 0
        total_trace_rows = 0
        kept_trace_rows = 0
        dropped_trace_rows = 0
        trace_fp = None
        if self.config.trace_out_path:
            os.makedirs(os.path.dirname(self.config.trace_out_path) or ".", exist_ok=True)
            trace_fp = open(self.config.trace_out_path, "a", encoding="utf-8")

        try:
            for ep in range(max(1, int(self.config.episodes))):
                ep_seed = int(self.config.seed) + int(ep)
                verse.seed(ep_seed)
                rr = verse.reset()
                obs = rr.obs
                episode_rows: List[tuple[int, _ReplayRow]] = []
                transition_rewards: List[float] = []
                history: List[Dict[str, JSONValue]] = []
                episode_return = 0.0
                kept_this_ep = 0
                dropped_this_ep = 0

                for step_idx in range(max(1, int(self.config.max_steps))):
                    initial_policy_guess = self._normalize(
                        self._pad_or_truncate(self._policy_provider(obs=obs, history=history), self._action_dim)
                    )
                    result = search.search(
                        root_obs=obs,
                        policy_net=self._policy_provider,
                        value_net=self._value_provider,
                    )
                    quality = self._evaluate_trace_quality(initial_policy=initial_policy_guess, search_result=result)
                    keep_for_training = (not bool(self.config.high_quality_trace_filter)) or bool(
                        quality.get("high_quality", False)
                    )
                    total_trace_rows += 1
                    if keep_for_training:
                        kept_trace_rows += 1
                        kept_this_ep += 1
                    else:
                        dropped_trace_rows += 1
                        dropped_this_ep += 1
                    policy_target = self._pad_or_truncate(result.action_probs, self._action_dim)
                    action = self._sample_action(result.action_probs)
                    step = verse.step(int(action))
                    if trace_fp is not None:
                        self._append_trace(
                            fp=trace_fp,
                            episode_idx=ep,
                            step_idx=step_idx,
                            obs=obs,
                            chosen_action=int(action),
                            search_result=result,
                            initial_policy_guess=initial_policy_guess,
                            quality=quality,
                            keep_for_training=bool(keep_for_training),
                            reward=float(step.reward),
                            done=bool(step.done or step.truncated),
                        )

                    if keep_for_training:
                        state_vec = self._obs_pad(obs)
                        hist_vec = self._history_rows(history)
                        episode_rows.append(
                            (
                                len(transition_rewards),
                                _ReplayRow(
                                    state=state_vec,
                                    history=hist_vec,
                                    target_policy=policy_target,
                                    target_value=0.0,  # filled after returns are known
                                ),
                            )
                        )

                    transition_rewards.append(float(step.reward))
                    episode_return += float(step.reward)
                    total_steps += 1

                    history.append({"obs": obs, "action": int(action), "reward": float(step.reward)})
                    obs = step.obs
                    if bool(step.done or step.truncated):
                        break

                returns = self._discounted_returns(transition_rewards, gamma=float(self.config.discount))
                for transition_idx, row in episode_rows:
                    if transition_idx < 0 or transition_idx >= len(returns):
                        continue
                    ret = returns[transition_idx]
                    row.target_value = float(max(-1.0, min(1.0, ret)))
                    self.replay.append(row)
                if len(self.replay) > int(self.config.replay_capacity):
                    self.replay = self.replay[-int(self.config.replay_capacity) :]

                total_return += float(episode_return)

                pl, vl, up = self._train_from_replay()
                total_policy_loss += float(pl)
                total_value_loss += float(vl)
                update_count += int(up)

                print(
                    f"[mcts_trainer] episode={ep+1}/{self.config.episodes} "
                    f"return={episode_return:.3f} replay={len(self.replay)} "
                    f"hq_kept={kept_this_ep} hq_dropped={dropped_this_ep}"
                )
                if (
                    self.config.checkpoint_path
                    and int(self.config.checkpoint_every_episodes) > 0
                    and ((ep + 1) % int(self.config.checkpoint_every_episodes) == 0)
                ):
                    self.save_checkpoint(self.config.checkpoint_path, episode_idx=(ep + 1))
                if (
                    self.config.checkpoint_versioned_dir
                    and int(self.config.checkpoint_every_episodes) > 0
                    and ((ep + 1) % int(self.config.checkpoint_every_episodes) == 0)
                ):
                    ckp = self._versioned_checkpoint_path(episode_idx=(ep + 1))
                    self.save_checkpoint(ckp, episode_idx=(ep + 1))
        finally:
            if trace_fp is not None:
                trace_fp.close()

        if self.config.checkpoint_path:
            self.save_checkpoint(self.config.checkpoint_path, episode_idx=int(self.config.episodes))

        verse.close()
        mean_policy_loss = float(total_policy_loss / float(max(1, update_count)))
        mean_value_loss = float(total_value_loss / float(max(1, update_count)))
        return {
            "episodes": float(max(1, int(self.config.episodes))),
            "total_steps": float(total_steps),
            "mean_return": float(total_return / float(max(1, int(self.config.episodes)))),
            "mean_policy_loss": mean_policy_loss,
            "mean_value_loss": mean_value_loss,
            "updates": float(update_count),
            "replay_size": float(len(self.replay)),
            "trace_rows_total": float(total_trace_rows),
            "trace_rows_kept": float(kept_trace_rows),
            "trace_rows_dropped": float(dropped_trace_rows),
            "trace_keep_rate": float(kept_trace_rows / float(max(1, total_trace_rows))),
        }

    def save_checkpoint(self, path: str, *, episode_idx: Optional[int] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "model_config": (
                self.model.get_config() if hasattr(self.model, "get_config") and callable(getattr(self.model, "get_config")) else {}
            ),
            "trainer_config": {
                "episodes": int(self.config.episodes),
                "max_steps": int(self.config.max_steps),
                "num_simulations": int(self.config.num_simulations),
                "search_depth": int(self.config.search_depth),
                "discount": float(self.config.discount),
                "high_quality_trace_filter": bool(self.config.high_quality_trace_filter),
                "min_quality_value_gain": float(self.config.min_quality_value_gain),
                "min_quality_policy_shift_l1": float(self.config.min_quality_policy_shift_l1),
                "min_quality_kl_divergence": float(self.config.min_quality_kl_divergence),
                "min_forced_loss_prior_mass": float(self.config.min_forced_loss_prior_mass),
                "min_forced_loss_mass_drop": float(self.config.min_forced_loss_mass_drop),
                "sparse_replay_min_samples": int(self.config.sparse_replay_min_samples),
                "sparse_replay_batch_size": int(self.config.sparse_replay_batch_size),
            },
            "stage": "mcts_search_guided",
            "episode_idx": (None if episode_idx is None else int(episode_idx)),
        }
        self._torch.save(payload, path)

    def _versioned_checkpoint_path(self, *, episode_idx: int) -> str:
        out_dir = str(self.config.checkpoint_versioned_dir or "").strip()
        if not out_dir:
            out_dir = os.path.join("models", "meta_transformer_versions")
        os.makedirs(out_dir, exist_ok=True)
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"meta_transformer_mcts_ep{int(episode_idx):04d}_{ts}.pt"
        return os.path.join(out_dir, name)

    def _append_trace(
        self,
        *,
        fp: Any,
        episode_idx: int,
        step_idx: int,
        obs: JSONValue,
        chosen_action: int,
        search_result: Any,
        initial_policy_guess: List[float],
        quality: Dict[str, Any],
        keep_for_training: bool,
        reward: float,
        done: bool,
    ) -> None:
        row = {
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "verse_name": str(self.verse_spec.verse_name),
            "obs": obs,
            "action": int(chosen_action),
            "search_policy": [float(x) for x in list(search_result.action_probs or [])],
            "search_visit_counts": [int(x) for x in list(search_result.visit_counts or [])],
            "search_action_values": [float(x) for x in list(search_result.action_values or [])],
            "search_root_value": float(search_result.root_value),
            "search_avg_leaf_value": float(search_result.avg_leaf_value),
            "search_simulations": int(search_result.simulations),
            "forced_loss_detected": bool(search_result.forced_loss_detected),
            "forced_loss_actions": [int(a) for a in list(search_result.forced_loss_actions or [])],
            "initial_policy_guess": [float(x) for x in list(initial_policy_guess or [])],
            "trace_quality": dict(quality),
            "high_quality_trace": bool(quality.get("high_quality", False)),
            "kept_for_training": bool(keep_for_training),
            "reward": float(reward),
            "done": bool(done),
        }
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _evaluate_trace_quality(self, *, initial_policy: List[float], search_result: Any) -> Dict[str, Any]:
        eps = 1e-8
        prior = self._normalize(
            self._pad_or_truncate([float(x) for x in list(initial_policy or [])], self._action_dim)
        )
        search_policy = self._normalize(
            self._pad_or_truncate(
                [float(x) for x in list(getattr(search_result, "action_probs", []) or [])], self._action_dim
            )
        )
        action_values = self._pad_or_truncate(
            [float(x) for x in list(getattr(search_result, "action_values", []) or [])], self._action_dim
        )

        expected_prior = 0.0
        expected_search = 0.0
        l1_shift = 0.0
        kl_div = 0.0
        for i in range(self._action_dim):
            p = float(prior[i])
            q = float(search_policy[i])
            v = float(action_values[i])
            expected_prior += p * v
            expected_search += q * v
            l1_shift += abs(q - p)
            kl_div += q * math.log((q + eps) / (p + eps))

        forced_actions = [
            int(a)
            for a in list(getattr(search_result, "forced_loss_actions", []) or [])
            if 0 <= int(a) < self._action_dim
        ]
        prior_forced = float(sum(prior[a] for a in forced_actions)) if forced_actions else 0.0
        search_forced = float(sum(search_policy[a] for a in forced_actions)) if forced_actions else 0.0
        forced_drop = float(max(0.0, prior_forced - search_forced))

        value_gain = float(expected_search - expected_prior)
        significant_improvement = bool(
            value_gain >= float(self.config.min_quality_value_gain)
            and l1_shift >= float(self.config.min_quality_policy_shift_l1)
            and kl_div >= float(self.config.min_quality_kl_divergence)
        )
        forced_loss_avoidance = bool(
            bool(forced_actions)
            and prior_forced >= float(self.config.min_forced_loss_prior_mass)
            and forced_drop >= float(self.config.min_forced_loss_mass_drop)
        )
        high_quality = bool(significant_improvement or forced_loss_avoidance)
        return {
            "high_quality": bool(high_quality),
            "significant_improvement": bool(significant_improvement),
            "forced_loss_avoidance": bool(forced_loss_avoidance),
            "expected_value_prior": float(expected_prior),
            "expected_value_search": float(expected_search),
            "value_gain": float(value_gain),
            "policy_shift_l1": float(l1_shift),
            "kl_divergence": float(kl_div),
            "prior_forced_loss_mass": float(prior_forced),
            "search_forced_loss_mass": float(search_forced),
            "forced_loss_mass_drop": float(forced_drop),
        }

    def _policy_provider(self, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]] = None) -> List[float]:
        model_probs = self._model_policy_probs(obs=obs, history=history)
        if self.policy_prior is None:
            return model_probs
        try:
            ext = self.policy_prior(obs, history)
        except Exception:
            try:
                ext = self.policy_prior(obs)
            except Exception:
                ext = None
        ext_probs = self._pad_or_truncate(self._to_float_list(ext), self._action_dim)
        if sum(ext_probs) <= 0.0:
            return model_probs
        mix = 0.5
        merged = [((1.0 - mix) * model_probs[i]) + (mix * ext_probs[i]) for i in range(self._action_dim)]
        return self._normalize(merged)

    def _value_provider(self, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]] = None) -> float:
        _, value = self._model_policy_value(obs=obs, history=history)
        return float(max(-1.0, min(1.0, value)))

    def _model_policy_probs(self, *, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]]) -> List[float]:
        probs, _ = self._model_policy_value(obs=obs, history=history)
        return probs

    def _model_policy_value(self, *, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]]) -> tuple[List[float], float]:
        t_state, t_hist = self._model_inputs(obs=obs, history=history)
        with self._torch.no_grad():
            if hasattr(self.model, "forward_policy_value"):
                out = self.model.forward_policy_value(t_state, t_hist)
                logits = out["logits"]
                value = out["value"]
            else:
                logits = self.model(t_state, t_hist)
                value = self._torch.zeros((1,), dtype=self._torch.float32)
            probs = self._torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return self._normalize(self._pad_or_truncate([float(x) for x in probs], self._action_dim)), float(value.squeeze(0).item())

    def _train_from_replay(self) -> tuple[float, float, int]:
        base_batch = max(8, int(self.config.batch_size))
        required_replay = int(base_batch)
        batch_k = int(base_batch)

        if bool(self.config.high_quality_trace_filter) and len(self.replay) < int(base_batch):
            sparse_min = max(8, min(int(self.config.sparse_replay_min_samples), int(base_batch)))
            sparse_batch = max(8, min(int(self.config.sparse_replay_batch_size), int(base_batch)))
            required_replay = int(sparse_min)
            batch_k = int(sparse_batch)

        if len(self.replay) < int(required_replay):
            return 0.0, 0.0, 0

        policy_loss_total = 0.0
        value_loss_total = 0.0
        updates = 0

        self.model.train()
        for _ in range(max(1, int(self.config.train_epochs_per_episode))):
            batch = random.sample(self.replay, k=min(len(self.replay), int(batch_k)))
            t_state = self._torch.tensor([r.state for r in batch], dtype=self._torch.float32)
            t_hist = self._torch.tensor([r.history for r in batch], dtype=self._torch.float32)
            t_pi = self._torch.tensor([r.target_policy for r in batch], dtype=self._torch.float32)
            t_v = self._torch.tensor([r.target_value for r in batch], dtype=self._torch.float32)

            if hasattr(self.model, "forward_policy_value"):
                out = self.model.forward_policy_value(t_state, t_hist)
                logits = out["logits"]
                value = out["value"]
            else:
                logits = self.model(t_state, t_hist)
                value = self._torch.zeros_like(t_v)

            logp = self._torch.log_softmax(logits, dim=-1)
            policy_loss = -(t_pi * logp).sum(dim=-1).mean()
            value_loss = self._mse(value, t_v)
            loss = policy_loss + (float(self.config.value_loss_weight) * value_loss)

            self._opt.zero_grad()
            loss.backward()
            self._torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._opt.step()

            policy_loss_total += float(policy_loss.item())
            value_loss_total += float(value_loss.item())
            updates += 1

        self.model.eval()
        return policy_loss_total, value_loss_total, updates

    def _model_inputs(self, *, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]]) -> tuple[Any, Any]:
        state = self._obs_pad(obs)
        hist_rows = self._history_rows(history)
        t_state = self._torch.tensor([state], dtype=self._torch.float32)
        t_hist = self._torch.tensor([hist_rows], dtype=self._torch.float32)
        return t_state, t_hist

    def _obs_pad(self, obs: JSONValue) -> List[float]:
        try:
            vec = self._obs_to_vector(obs)
        except Exception:
            vec = []
        return self._pad_or_truncate([float(v) for v in vec], self._state_dim)

    def _history_rows(self, history: Optional[List[Dict[str, JSONValue]]]) -> List[List[float]]:
        rows: List[List[float]] = []
        if isinstance(history, list):
            for row in history[-1:]:
                if not isinstance(row, dict):
                    continue
                h_obs = row.get("obs")
                h_action = float(row.get("action", 0.0) or 0.0)
                h_reward = float(row.get("reward", 0.0) or 0.0)
                rows.append(self._obs_pad(h_obs) + [h_action, h_reward])
        if not rows:
            rows = [[0.0] * self._context_dim]
        return rows

    def _sample_action(self, probs: List[float]) -> int:
        vec = self._normalize(self._pad_or_truncate(probs, self._action_dim))
        temp = max(1e-6, float(self.config.temperature))
        if temp <= 1e-3:
            return int(max(range(len(vec)), key=lambda i: vec[i]))
        scaled = [pow(max(1e-8, p), 1.0 / temp) for p in vec]
        scaled = self._normalize(scaled)
        return int(random.choices(range(len(scaled)), weights=scaled, k=1)[0])

    def _discounted_returns(self, rewards: List[float], *, gamma: float) -> List[float]:
        out = [0.0 for _ in rewards]
        g = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            g = float(rewards[i]) + float(gamma) * g
            out[i] = g
        return out

    def _normalize(self, values: List[float]) -> List[float]:
        s = sum(max(0.0, float(v)) for v in values)
        if s <= 0.0:
            if not values:
                return []
            u = 1.0 / float(len(values))
            return [u for _ in values]
        return [max(0.0, float(v)) / s for v in values]

    def _pad_or_truncate(self, vec: List[float], size: int) -> List[float]:
        if size <= 0:
            return []
        if len(vec) >= size:
            return [float(v) for v in vec[:size]]
        return [float(v) for v in vec] + [0.0] * (size - len(vec))

    def _to_float_list(self, value: Any) -> List[float]:
        if isinstance(value, list):
            return [float(_safe(v, 0.0)) for v in value]
        if isinstance(value, tuple):
            return self._to_float_list(list(value))
        return []


def _safe(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)
