"""
agents/transformer_agent.py

Decision Transformer runtime agent for discrete action spaces.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import torch
import torch.nn.functional as F

from core.agent_base import ActionResult, ExperienceBatch
from core.types import AgentSpec, JSONValue, SpaceSpec
from memory.embeddings import obs_to_universal_vector
from models.decision_transformer import DecisionTransformer, load_decision_transformer_checkpoint


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _load_torch_payload(path: str) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload if isinstance(payload, dict) else {}


class TransformerAgent:
    """
    Decision Transformer agent (inference + optional online fine-tuning).

    Config keys:
    - model_path: required checkpoint path
    - context_len: optional override (<= trained context)
    - target_return: optional scalar conditioning target
    - target_return_auto: infer target return from training dataset RTG (default false)
    - target_return_auto_quantile: quantile used when target_return_auto is enabled (default 0.9)
    - temperature: sampling temperature
    - top_k: top-k sampling filter (0 disables)
    - sample: bool stochastic action selection
    - device: "cpu" | "cuda" | "auto"
    - online_enabled: enable online policy updates from rollout ExperienceBatch
    - online_lr: AdamW learning rate
    - online_weight_decay: AdamW weight decay
    - online_grad_clip: gradient clip max-norm (0 disables)
    - online_batch_size: number of context windows per SGD step
    - online_updates_per_learn: SGD steps per learn() call
    - online_replay_capacity: max number of context windows kept in replay
    - online_gamma: RTG discount factor for online updates
    - online_chunk_stride: window stride over episode steps (0 => context_len)
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

        if action_space.type != "discrete" or not isinstance(action_space.n, int) or int(action_space.n) <= 0:
            raise ValueError("TransformerAgent requires discrete action_space with n > 0")
        self._n_actions = int(action_space.n)

        cfg = dict(spec.config) if isinstance(spec.config, dict) else {}
        model_path = str(cfg.get("model_path", "")).strip()
        if not model_path:
            raise ValueError("TransformerAgent requires config.model_path")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ADT checkpoint not found: {model_path}")

        device_raw = str(cfg.get("device", "auto")).strip().lower()
        if device_raw == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device_raw)

        self.model, self._checkpoint = load_decision_transformer_checkpoint(model_path, map_location=self._device)
        self.model.eval()

        model_cfg = self.model.get_config()
        self._state_dim = int(model_cfg["state_dim"])
        trained_context_len = int(model_cfg["context_len"])
        self._bos_token_id = int(model_cfg.get("bos_token_id", self.model.config.action_dim))
        self._model_action_dim = int(model_cfg["action_dim"])
        if self._model_action_dim <= 0:
            raise RuntimeError("invalid action_dim in ADT checkpoint")
        if self._model_action_dim > int(self._n_actions):
            raise ValueError(
                f"ADT checkpoint action_dim={self._model_action_dim} exceeds action_space.n={self._n_actions}"
            )

        requested_context = _safe_int(cfg.get("context_len"), trained_context_len)
        self._context_len = max(1, min(int(trained_context_len), int(requested_context)))

        if "target_return" in cfg:
            self._target_return = _safe_float(cfg.get("target_return", 1.0), 1.0)
        elif bool(cfg.get("target_return_auto", False)):
            self._target_return = self._infer_default_target_return(cfg=cfg, model_path=model_path)
        else:
            self._target_return = 1.0
        self._temperature = max(1e-6, _safe_float(cfg.get("temperature", 1.0), 1.0))
        self._top_k = max(0, _safe_int(cfg.get("top_k", 0), 0))
        self._sample = bool(cfg.get("sample", False))

        self._seed: Optional[int] = None
        self._rng = random.Random()
        self._step_count = 0
        self._state_history: Deque[List[float]] = deque(maxlen=self._context_len)
        self._action_history: Deque[int] = deque(maxlen=self._context_len)

        self._online_enabled = bool(cfg.get("online_enabled", False))
        self._online_lr = max(1e-6, _safe_float(cfg.get("online_lr", 1e-4), 1e-4))
        self._online_weight_decay = max(0.0, _safe_float(cfg.get("online_weight_decay", 1e-2), 1e-2))
        self._online_grad_clip = max(0.0, _safe_float(cfg.get("online_grad_clip", 1.0), 1.0))
        self._online_batch_size = max(1, _safe_int(cfg.get("online_batch_size"), 32))
        self._online_updates_per_learn = max(1, _safe_int(cfg.get("online_updates_per_learn"), 1))
        self._online_replay_capacity = max(1, _safe_int(cfg.get("online_replay_capacity"), 4096))
        self._online_gamma = max(0.0, min(1.0, _safe_float(cfg.get("online_gamma", 1.0), 1.0)))
        self._online_chunk_stride = _safe_int(cfg.get("online_chunk_stride"), 0)
        if int(self._online_chunk_stride) <= 0:
            self._online_chunk_stride = int(self._context_len)

        self._online_replay: Deque[Dict[str, List[Any]]] = deque(maxlen=self._online_replay_capacity)
        self._online_updates = 0
        self._optimizer: Optional[torch.optim.Optimizer] = None
        if self._online_enabled:
            self._optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self._online_lr),
                weight_decay=float(self._online_weight_decay),
            )

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        if seed is not None:
            self._rng = random.Random(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

    def set_target_return(self, target_return: float) -> None:
        self._target_return = float(target_return)

    def _reset_context(self) -> None:
        self._state_history.clear()
        self._action_history.clear()
        self._step_count = 0

    def _compute_rtg(self, rewards: List[float]) -> List[float]:
        out = [0.0 for _ in rewards]
        g = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            g = float(rewards[i]) + float(self._online_gamma) * float(g)
            out[i] = float(g)
        return out

    def _infer_default_target_return(self, *, cfg: Dict[str, Any], model_path: str) -> float:
        """
        Infer a sensible RTG conditioning target from the training dataset when
        no explicit target_return was provided.
        """
        quantile = _safe_float(cfg.get("target_return_auto_quantile", 0.9), 0.9)
        quantile = max(0.0, min(1.0, float(quantile)))
        dataset_path = str(self._checkpoint.get("dataset_path", "")).strip()
        if not dataset_path:
            return 1.0

        model_dir = os.path.dirname(os.path.abspath(model_path))
        candidates = [
            dataset_path,
            os.path.abspath(dataset_path),
            os.path.join(model_dir, os.path.basename(dataset_path)),
        ]
        chosen = None
        for p in candidates:
            if p and os.path.isfile(p):
                chosen = p
                break
        if chosen is None:
            return 1.0

        try:
            payload = _load_torch_payload(chosen)
            rtg = payload.get("returns_to_go")
            mask = payload.get("attention_mask")
            if not isinstance(rtg, torch.Tensor):
                return 1.0
            vals = rtg.float().reshape(-1)
            if isinstance(mask, torch.Tensor) and mask.shape == rtg.shape:
                vals = rtg.float()[mask.float() > 0.0]
            if vals.numel() <= 0:
                return 1.0
            q = float(torch.quantile(vals, q=float(quantile)).item())
            if math.isfinite(q) and q > 0.0:
                return float(q)
            vmax = float(torch.max(vals).item())
            return float(vmax if math.isfinite(vmax) and vmax > 0.0 else 1.0)
        except Exception:
            return 1.0

    def _build_online_windows(self, transitions: List[Any]) -> List[Dict[str, List[Any]]]:
        if not transitions:
            return []
        states: List[List[float]] = []
        actions: List[int] = []
        rewards: List[float] = []
        timesteps: List[int] = []

        for i, tr in enumerate(transitions):
            try:
                a = int(tr.action)
            except (TypeError, ValueError):
                continue
            if a < 0 or a >= int(self._model_action_dim):
                continue
            obs = tr.obs
            states.append(obs_to_universal_vector(obs, dim=self._state_dim))
            actions.append(int(a))
            rewards.append(float(tr.reward))
            t_obs = obs.get("t") if isinstance(obs, dict) else None
            t_val = _safe_int(t_obs, i)
            timesteps.append(min(int(self.model.config.max_timestep) - 1, max(0, int(t_val))))

        n = len(actions)
        if n <= 0:
            return []
        rtg = self._compute_rtg(rewards)
        K = int(self._context_len)
        stride = max(1, int(self._online_chunk_stride))

        windows: List[Dict[str, List[Any]]] = []
        for start in range(0, n, stride):
            end = min(n, start + K)
            if end <= start:
                continue
            cur_states = [[0.0] * int(self._state_dim) for _ in range(K)]
            cur_rtg = [0.0 for _ in range(K)]
            cur_prev = [int(self._bos_token_id) for _ in range(K)]
            cur_actions = [-100 for _ in range(K)]
            cur_t = [0 for _ in range(K)]
            cur_mask = [0.0 for _ in range(K)]

            for j in range(end - start):
                idx = start + j
                cur_states[j] = list(states[idx])
                cur_rtg[j] = float(rtg[idx])
                cur_actions[j] = int(actions[idx])
                cur_prev[j] = int(self._bos_token_id) if idx == 0 else int(actions[idx - 1])
                cur_t[j] = int(timesteps[idx])
                cur_mask[j] = 1.0

            windows.append(
                {
                    "states": cur_states,
                    "returns_to_go": cur_rtg,
                    "prev_actions": cur_prev,
                    "actions": cur_actions,
                    "timesteps": cur_t,
                    "attention_mask": cur_mask,
                }
            )
        return windows

    def _sample_online_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._online_replay:
            return None
        bsz = min(int(self._online_batch_size), len(self._online_replay))
        picked = self._rng.sample(list(self._online_replay), bsz)
        states = torch.tensor([x["states"] for x in picked], dtype=torch.float32, device=self._device)
        rtg = torch.tensor([x["returns_to_go"] for x in picked], dtype=torch.float32, device=self._device)
        prev = torch.tensor([x["prev_actions"] for x in picked], dtype=torch.long, device=self._device)
        tgt = torch.tensor([x["actions"] for x in picked], dtype=torch.long, device=self._device)
        t = torch.tensor([x["timesteps"] for x in picked], dtype=torch.long, device=self._device)
        m = torch.tensor([x["attention_mask"] for x in picked], dtype=torch.float32, device=self._device)
        return {
            "states": states,
            "returns_to_go": rtg,
            "prev_actions": prev,
            "actions": tgt,
            "timesteps": t,
            "attention_mask": m,
        }

    def _maybe_reset_on_obs(self, obs: JSONValue) -> None:
        # Heuristic: most verses expose "t". Reset when a fresh episode observation appears.
        if not isinstance(obs, dict):
            return
        t_raw = obs.get("t")
        if t_raw is None:
            return
        t_val = _safe_int(t_raw, -1)
        if t_val == 0 and self._step_count > 0:
            self._reset_context()

    def _build_inputs(self, current_state: List[float]) -> Dict[str, torch.Tensor]:
        hist_states = list(self._state_history)
        hist_actions = list(self._action_history)
        states_seq = hist_states + [list(current_state)]

        if len(states_seq) > self._context_len:
            overflow = len(states_seq) - self._context_len
            states_seq = states_seq[overflow:]
            hist_actions = hist_actions[overflow:]

        seq_len = len(states_seq)
        prev_actions_seq = [int(self._bos_token_id) for _ in range(seq_len)]
        for i in range(1, seq_len):
            hist_idx = i - 1
            if hist_idx < len(hist_actions):
                prev_actions_seq[i] = int(hist_actions[hist_idx])

        start_t = max(0, int(self._step_count) - seq_len + 1)
        timesteps_seq = [start_t + i for i in range(seq_len)]
        rtg_seq = [float(self._target_return) for _ in range(seq_len)]

        K = self._context_len
        states_pad = [[0.0] * self._state_dim for _ in range(K)]
        prev_pad = [int(self._bos_token_id) for _ in range(K)]
        rtg_pad = [0.0 for _ in range(K)]
        t_pad = [0 for _ in range(K)]
        m_pad = [0.0 for _ in range(K)]
        for i in range(seq_len):
            states_pad[i] = list(states_seq[i])
            prev_pad[i] = int(prev_actions_seq[i])
            rtg_pad[i] = float(rtg_seq[i])
            t_pad[i] = int(timesteps_seq[i])
            m_pad[i] = 1.0

        return {
            "states": torch.tensor([states_pad], dtype=torch.float32, device=self._device),
            "returns_to_go": torch.tensor([rtg_pad], dtype=torch.float32, device=self._device),
            "prev_actions": torch.tensor([prev_pad], dtype=torch.long, device=self._device),
            "timesteps": torch.tensor([t_pad], dtype=torch.long, device=self._device),
            "attention_mask": torch.tensor([m_pad], dtype=torch.float32, device=self._device),
            "seq_len": torch.tensor([seq_len], dtype=torch.long, device=self._device),
        }

    def act(self, obs: JSONValue) -> ActionResult:
        self._maybe_reset_on_obs(obs)
        state_vec = obs_to_universal_vector(obs, dim=self._state_dim)
        model_in = self._build_inputs(state_vec)
        with torch.no_grad():
            action_t, conf_t, _ = self.model.predict_next_action(
                states=model_in["states"],
                returns_to_go=model_in["returns_to_go"],
                prev_actions=model_in["prev_actions"],
                timesteps=model_in["timesteps"],
                attention_mask=model_in["attention_mask"],
                temperature=float(self._temperature),
                top_k=int(self._top_k),
                sample=bool(self._sample),
            )
        action = int(action_t.item())
        if action < 0 or action >= self._n_actions:
            action = max(0, min(self._n_actions - 1, action))

        self._state_history.append(list(state_vec))
        self._action_history.append(int(action))
        self._step_count += 1

        return ActionResult(
            action=int(action),
            info={
                "mode": "adt",
                "confidence": float(conf_t.item()),
                "target_return": float(self._target_return),
                "context_len": int(self._context_len),
            },
        )

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        if not self._online_enabled:
            raise NotImplementedError("TransformerAgent online updates disabled; set config.online_enabled=true")
        if self._optimizer is None:
            return {}
        windows = self._build_online_windows(list(batch.transitions))
        if not windows:
            return {}
        for w in windows:
            self._online_replay.append(w)

        loss_sum = 0.0
        acc_sum = 0.0
        n_updates = 0
        self.model.train()
        for _ in range(int(self._online_updates_per_learn)):
            sampled = self._sample_online_batch()
            if sampled is None:
                break
            logits = self.model(
                states=sampled["states"],
                returns_to_go=sampled["returns_to_go"],
                prev_actions=sampled["prev_actions"],
                timesteps=sampled["timesteps"],
                attention_mask=sampled["attention_mask"],
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                sampled["actions"].reshape(-1),
                ignore_index=-100,
            )
            self._optimizer.zero_grad()
            loss.backward()
            if float(self._online_grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self._online_grad_clip))
            self._optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                valid = sampled["actions"] != -100
                n_valid = int(valid.sum().item())
                correct = int(((pred == sampled["actions"]) & valid).sum().item())
                acc = float(correct / max(1, n_valid))
            loss_sum += float(loss.item())
            acc_sum += float(acc)
            n_updates += 1

        self.model.eval()
        if n_updates <= 0:
            return {}
        self._online_updates += int(n_updates)
        return {
            "online_updates": int(n_updates),
            "online_updates_total": int(self._online_updates),
            "online_loss": float(loss_sum / float(max(1, n_updates))),
            "online_acc": float(acc_sum / float(max(1, n_updates))),
            "online_replay_size": int(len(self._online_replay)),
            "online_windows_added": int(len(windows)),
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        weights_path = os.path.join(path, "transformer_agent_weights.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.get_config(),
            },
            weights_path,
        )
        payload = {
            "spec": self.spec.to_dict(),
            "context_len": int(self._context_len),
            "target_return": float(self._target_return),
            "temperature": float(self._temperature),
            "top_k": int(self._top_k),
            "sample": bool(self._sample),
            "model_config": self.model.get_config(),
            "weights_path": "transformer_agent_weights.pt",
            "online_enabled": bool(self._online_enabled),
            "online_lr": float(self._online_lr),
            "online_weight_decay": float(self._online_weight_decay),
            "online_grad_clip": float(self._online_grad_clip),
            "online_batch_size": int(self._online_batch_size),
            "online_updates_per_learn": int(self._online_updates_per_learn),
            "online_replay_capacity": int(self._online_replay_capacity),
            "online_gamma": float(self._online_gamma),
            "online_chunk_stride": int(self._online_chunk_stride),
            "online_updates_total": int(self._online_updates),
        }
        with open(os.path.join(path, "transformer_agent.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        # Keep parity with agent interface; loading weights is handled by config.model_path at init.
        cfg_path = os.path.join(path, "transformer_agent.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"transformer agent config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._context_len = max(1, min(self._context_len, _safe_int(payload.get("context_len"), self._context_len)))
        self._target_return = _safe_float(payload.get("target_return"), self._target_return)
        self._temperature = max(1e-6, _safe_float(payload.get("temperature"), self._temperature))
        self._top_k = max(0, _safe_int(payload.get("top_k"), self._top_k))
        self._sample = bool(payload.get("sample", self._sample))
        self._online_updates = max(0, _safe_int(payload.get("online_updates_total"), self._online_updates))

        rel_weights = str(payload.get("weights_path", "")).strip()
        if rel_weights:
            weights_path = os.path.join(path, rel_weights)
            if os.path.isfile(weights_path):
                model_loaded, _ = load_decision_transformer_checkpoint(weights_path, map_location=self._device)
                self.model.load_state_dict(model_loaded.state_dict())
                self.model.to(self._device)
                self.model.eval()

    def close(self) -> None:
        return
