"""
agents/dqn_agent.py

Minimal Deep Q-Network for Universe.AI v2 verses.
Uses PyTorch for function approximation instead of tabular Q.

Key features:
- Neural network Q-function (3-layer MLP)
- Experience replay buffer
- Target network for training stability
- Legal action masking
- Observation encoding: one-hot board + scalar features
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Observation encoders
# ---------------------------------------------------------------------------

def encode_obs_go_v2(obs: Dict[str, Any]) -> np.ndarray:
    """Encode Go v2 observation → flat float vector.
    Board: one-hot (3 states × 25 cells = 75) + 15 scalar features = 90 dims.
    """
    board = obs.get("board", [0] * 25)
    # One-hot encode: [empty, black, white] per cell
    one_hot = np.zeros(75, dtype=np.float32)
    for i, cell in enumerate(board):
        one_hot[i * 3 + int(cell)] = 1.0

    scalars = np.array([
        obs.get("my_captures", 0) / 10.0,
        obs.get("opp_captures", 0) / 10.0,
        obs.get("my_territory", 0) / 25.0,
        obs.get("opp_territory", 0) / 25.0,
        obs.get("my_groups", 0) / 10.0,
        obs.get("my_liberties_total", 0) / 20.0,
        obs.get("my_atari_groups", 0) / 5.0,
        obs.get("opp_groups", 0) / 10.0,
        obs.get("opp_liberties_total", 0) / 20.0,
        obs.get("opp_atari_groups", 0) / 5.0,
        obs.get("my_stones", 0) / 25.0,
        obs.get("opp_stones", 0) / 25.0,
        obs.get("ko_point", -1) / 25.0,
        obs.get("consecutive_passes", 0) / 3.0,
        obs.get("t", 0) / 120.0,
    ], dtype=np.float32)

    return np.concatenate([one_hot, scalars])


def encode_obs_chess_v2(obs: Dict[str, Any]) -> np.ndarray:
    """Encode Chess v2 observation → flat float vector.
    Board: one-hot (9 piece types × 25 cells = 225) + 15 scalar features = 240 dims.
    """
    board = obs.get("board", [0] * 25)
    # One-hot encode: 9 piece types (0=empty..8=b_pawn)
    one_hot = np.zeros(225, dtype=np.float32)
    for i, piece in enumerate(board):
        one_hot[i * 9 + int(piece)] = 1.0

    scalars = np.array([
        obs.get("my_material", 0) / 20.0,
        obs.get("opp_material", 0) / 20.0,
        obs.get("material_delta", 0) / 20.0,
        obs.get("my_king_pos", 0) / 25.0,
        obs.get("opp_king_pos", 0) / 25.0,
        obs.get("in_check", 0),
        obs.get("opp_in_check", 0),
        obs.get("my_legal_move_count", 0) / 50.0,
        obs.get("opp_legal_move_count", 0) / 50.0,
        obs.get("my_center_control", 0) / 9.0,
        obs.get("opp_center_control", 0) / 9.0,
        obs.get("my_pieces_count", 0) / 10.0,
        obs.get("opp_pieces_count", 0) / 10.0,
        obs.get("t", 0) / 150.0,
        0.0,  # padding
    ], dtype=np.float32)

    return np.concatenate([one_hot, scalars])


def encode_obs_uno_v2(obs: Dict[str, Any]) -> np.ndarray:
    """Encode Uno v2 observation → flat float vector. ~25 dims."""
    hand_colors = obs.get("hand_colors", [0, 0, 0, 0])
    scalars = np.array([
        obs.get("hand_size", 7) / 15.0,
        obs.get("opp_hand_size", 7) / 15.0,
        obs.get("top_color", 0) / 4.0,
        obs.get("top_value", 0) / 14.0,
        hand_colors[0] / 10.0,
        hand_colors[1] / 10.0,
        hand_colors[2] / 10.0,
        hand_colors[3] / 10.0,
        obs.get("hand_wilds", 0) / 5.0,
        obs.get("hand_action_cards", 0) / 5.0,
        obs.get("hand_playable", 0) / 10.0,
        obs.get("draw_pile_size", 0) / 60.0,
        obs.get("opp_said_uno", 0),
        obs.get("t", 0) / 200.0,
    ], dtype=np.float32)
    return scalars


def encode_obs_blackjack(obs: Dict[str, Any]) -> np.ndarray:
    """Encode blackjack observation with hand-structure and count features."""
    scalars = np.array([
        obs.get("player_sum", 0) / 21.0,
        obs.get("dealer_showing", 0) / 11.0,
        obs.get("usable_ace", 0),
        obs.get("can_double", 0),
        obs.get("can_split", 0),
        obs.get("num_hands", 1) / 4.0,
        obs.get("active_hand", 0) / 3.0,
        obs.get("t", 0) / 20.0,
        obs.get("hand_len", 0) / 10.0,
        obs.get("pair_rank", 0) / 11.0,
        obs.get("split_aces_hand", 0),
        obs.get("first_action", 0),
        obs.get("running_count", 0) / 20.0,
        obs.get("true_count", 0) / 10.0,
        obs.get("cards_remaining", 0) / 312.0,
    ], dtype=np.float32)
    return scalars


def encode_obs_generic(obs, dim: int = 32) -> np.ndarray:
    """
    Generic flat encoder for any verse observation.
    Handles dict, list, and scalar observations.
    Output is always a fixed-length float32 vector (padded or truncated to `dim`).
    """
    if isinstance(obs, dict):
        vals = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, (list, tuple)):
                for vi in v:
                    try:
                        vals.append(float(vi))
                    except (ValueError, TypeError):
                        vals.append(0.0)
            else:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(0.0)
        arr = np.array(vals, dtype=np.float32)
    elif isinstance(obs, (list, tuple)):
        try:
            arr = np.array(obs, dtype=np.float32).flatten()
        except Exception:
            arr = np.zeros(dim, dtype=np.float32)
    else:
        try:
            arr = np.array([float(obs)], dtype=np.float32)
        except Exception:
            arr = np.zeros(dim, dtype=np.float32)

    if arr.shape[0] >= dim:
        return arr[:dim]
    # Pad with zeros
    out = np.zeros(dim, dtype=np.float32)
    out[:arr.shape[0]] = arr
    return out


def _make_generic_encoder(dim: int):
    """Return a closure that encodes any obs to a fixed-length vector."""
    def _enc(obs) -> np.ndarray:
        return encode_obs_generic(obs, dim=dim)
    return _enc


# Encoder registry — verse-specific encoders for best results
OBS_ENCODERS = {
    "go_world_v2": (encode_obs_go_v2, 90),
    "chess_world_v2": (encode_obs_chess_v2, 240),
    "uno_world_v2": (encode_obs_uno_v2, 14),
    "blackjack_world": (encode_obs_blackjack, 15),
}

# Action counts — verse-specific overrides
ACTION_COUNTS = {
    "go_world_v2": 26,
    "chess_world_v2": 625,
    "uno_world_v2": 15,
    # Navigation verses
    "grid_world": 4,
    "cliff_world": 4,
    "line_world": 2,
    "maze_world": 4,
    "labyrinth_world": 4,
    "park_world": 4,
    "warehouse_world": 5,
    "swamp_world": 4,
    "escape_world": 4,
    "bridge_world": 4,
    "Wind_master_world": 4,
    "wind_master_world": 4,
    "pursuit_world": 4,
    # Strategy / card games
    "chess_world": 7,
    "go_world": 10,
    "uno_world": 15,
    "blackjack_world": 4,  # hit, stand, double_down, split
    # Planning / Economics
    "harvest_world": 4,
    "factory_world": 5,
    "trade_world": 4,
    "risk_tutorial_world": 5,
    # Memory
    "memory_vault_world": 4,
    "rule_flip_world": 4,
}

# Default generic encoding dim (covers all known verse obs sizes)
_GENERIC_DIM = 32


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Simple 3-layer MLP Q-function."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 20000,
        *,
        prioritized: bool = False,
        alpha: float = 0.6,
        priority_eps: float = 1e-5,
    ):
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, legal_actions: List[int], priority: Optional[float] = None):
        self.buffer.append((state, action, reward, next_state, done, legal_actions))
        if priority is None:
            priority = max(list(self.priorities), default=1.0)
        self.priorities.append(float(max(self.priority_eps, float(priority))))

    def sample(self, batch_size: int, *, beta: float = 0.4) -> Tuple[List[int], List, np.ndarray]:
        size = len(self.buffer)
        if size <= 0:
            return [], [], np.zeros((0,), dtype=np.float32)
        k = min(int(batch_size), size)
        if not self.prioritized:
            indices = random.sample(range(size), k)
            batch = [self.buffer[i] for i in indices]
            weights = np.ones((len(indices),), dtype=np.float32)
            return indices, batch, weights

        priorities = np.asarray(list(self.priorities), dtype=np.float64)
        scaled = np.power(np.maximum(priorities, self.priority_eps), self.alpha)
        probs = scaled / np.maximum(np.sum(scaled), 1e-12)
        indices = np.random.choice(size, size=k, replace=False, p=probs)
        batch = [self.buffer[int(i)] for i in indices]
        weights = np.power(size * probs[indices], -float(beta))
        weights = weights / np.maximum(np.max(weights), 1e-12)
        return [int(i) for i in indices.tolist()], batch, weights.astype(np.float32)

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        if not self.prioritized:
            return
        for idx, priority in zip(indices, priorities):
            if 0 <= int(idx) < len(self.priorities):
                self.priorities[int(idx)] = float(max(self.priority_eps, float(priority)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent for v2 verses.
    
    Usage:
        agent = DQNAgent("go_world_v2")
        action = agent.select_action(obs, legal_actions, epsilon=0.3)
        agent.store(obs, action, reward, next_obs, done, next_legal)
        agent.train_step()
    """

    def __init__(self, verse_name: str, hidden: int = 128, lr: float = 1e-3,
                 gamma: float = 0.95, buffer_size: int = 20000, batch_size: int = 64,
                 target_update_freq: int = 50, obs_dim: int = _GENERIC_DIM,
                 n_actions: int = 0, double_dqn: bool = False,
                 prioritized_replay: bool = False, prioritized_alpha: float = 0.6,
                 prioritized_beta0: float = 0.4, prioritized_beta_steps: int = 10000):
        # Use verse-specific encoder if available, else fall back to generic
        if verse_name in OBS_ENCODERS:
            encoder_fn, input_dim = OBS_ENCODERS[verse_name]
        else:
            input_dim = obs_dim
            encoder_fn = _make_generic_encoder(input_dim)

        # Use verse-specific action count if available, else require explicit n_actions
        if n_actions <= 0:
            if verse_name in ACTION_COUNTS:
                n_actions = ACTION_COUNTS[verse_name]
            else:
                raise ValueError(
                    f"DQNAgent: verse '{verse_name}' has no registered action count. "
                    f"Pass n_actions=<int> explicitly, or add to ACTION_COUNTS."
                )

        self.verse_name = verse_name
        self.encoder_fn = encoder_fn
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = bool(double_dqn)
        self.prioritized_replay = bool(prioritized_replay)
        self.prioritized_beta0 = float(prioritized_beta0)
        self.prioritized_beta_steps = max(1, int(prioritized_beta_steps))

        self.device = torch.device("cpu")
        self.q_net = QNetwork(input_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(input_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(
            buffer_size,
            prioritized=bool(prioritized_replay),
            alpha=float(prioritized_alpha),
        )
        self.train_steps = 0
        self.episodes = 0

    def encode(self, obs: Dict[str, Any]) -> np.ndarray:
        return self.encoder_fn(obs)

    def select_action(self, obs: Dict[str, Any], legal_actions: List[int],
                      epsilon: float = 0.1) -> int:
        if not legal_actions:
            return 0
        if random.random() < epsilon:
            return random.choice(legal_actions)

        state = torch.FloatTensor(self.encode(obs)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state).squeeze(0)
        # Mask illegal actions
        mask = torch.full((self.n_actions,), float("-inf"))
        for a in legal_actions:
            mask[a] = 0.0
        masked_q = q_values + mask
        return int(masked_q.argmax().item())

    def store(self, obs: Dict, action: int, reward: float,
              next_obs: Dict, done: bool, next_legal: List[int]):
        state = self.encode(obs)
        next_state = self.encode(next_obs)
        self.buffer.push(state, action, reward, next_state, done, next_legal)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        beta = self.prioritized_beta0 + (1.0 - self.prioritized_beta0) * min(
            1.0, float(self.train_steps) / float(self.prioritized_beta_steps)
        )
        indices, batch, weights = self.buffer.sample(self.batch_size, beta=beta)
        states, actions, rewards, next_states, dones, next_legals = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(np.array(weights)).to(self.device)

        # Current Q-values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values with legal action masking
        with torch.no_grad():
            target_next_q = self.target_net(next_states_t)
            online_next_q = self.q_net(next_states_t) if self.double_dqn else target_next_q
            for i, nl in enumerate(next_legals):
                mask = torch.full((self.n_actions,), float("-inf"))
                if nl:
                    for a in nl:
                        mask[a] = 0.0
                else:
                    mask[:] = 0.0  # if no legal info, allow all
                target_next_q[i] += mask
                online_next_q[i] += mask
            if self.double_dqn:
                next_actions = online_next_q.argmax(dim=1)
                max_next_q = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                max_next_q = target_next_q.max(dim=1)[0]
            targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        td_errors = q_values - targets
        loss = torch.mean(weights_t * torch.square(td_errors))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.buffer.update_priorities(indices, (torch.abs(td_errors).detach().cpu().numpy() + self.buffer.priority_eps).tolist())

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def end_episode(self):
        self.episodes += 1

    def get_q_values(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return Q-values for all actions (for inspection/transfer)."""
        state = torch.FloatTensor(self.encode(obs)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state).squeeze(0).numpy()

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes": self.episodes,
            "train_steps": self.train_steps,
            "replay_buffer": list(self.buffer.buffer),
            "replay_priorities": list(self.buffer.priorities),
            "replay_buffer_capacity": int(self.buffer.buffer.maxlen or len(self.buffer.buffer)),
            "prioritized_replay": bool(self.prioritized_replay),
            "prioritized_alpha": float(self.buffer.alpha),
            "prioritized_beta0": float(self.prioritized_beta0),
            "prioritized_beta_steps": int(self.prioritized_beta_steps),
            "double_dqn": bool(self.double_dqn),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episodes = ckpt.get("episodes", 0)
        self.train_steps = ckpt.get("train_steps", 0)
        replay_capacity = int(ckpt.get("replay_buffer_capacity", self.buffer.buffer.maxlen or len(self.buffer.buffer) or 1))
        self.double_dqn = bool(ckpt.get("double_dqn", self.double_dqn))
        self.prioritized_replay = bool(ckpt.get("prioritized_replay", self.prioritized_replay))
        self.prioritized_beta0 = float(ckpt.get("prioritized_beta0", self.prioritized_beta0))
        self.prioritized_beta_steps = int(ckpt.get("prioritized_beta_steps", self.prioritized_beta_steps))
        self.buffer = ReplayBuffer(
            replay_capacity,
            prioritized=bool(self.prioritized_replay),
            alpha=float(ckpt.get("prioritized_alpha", self.buffer.alpha)),
        )
        replay_priorities = list(ckpt.get("replay_priorities", []))
        replay_items = list(ckpt.get("replay_buffer", []))
        for idx, item in enumerate(replay_items):
            try:
                self.buffer.buffer.append(item)
                if idx < len(replay_priorities):
                    self.buffer.priorities.append(float(replay_priorities[idx]))
                else:
                    self.buffer.priorities.append(1.0)
            except Exception:
                continue
