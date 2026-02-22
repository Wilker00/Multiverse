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
    # Strategy
    "chess_world": 7,
    "go_world": 10,
    "uno_world": 15,
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
    def __init__(self, capacity: int = 20000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, legal_actions: List[int]):
        self.buffer.append((state, action, reward, next_state, done, legal_actions))

    def sample(self, batch_size: int) -> List:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

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
                 n_actions: int = 0):
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

        self.device = torch.device("cpu")
        self.q_net = QNetwork(input_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(input_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
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

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_legals = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values with legal action masking
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            # Mask illegal actions for each sample
            for i, nl in enumerate(next_legals):
                mask = torch.full((self.n_actions,), float("-inf"))
                if nl:
                    for a in nl:
                        mask[a] = 0.0
                else:
                    mask[:] = 0.0  # if no legal info, allow all
                next_q[i] += mask
            max_next_q = next_q.max(dim=1)[0]
            targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

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
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episodes = ckpt.get("episodes", 0)
        self.train_steps = ckpt.get("train_steps", 0)
