"""
Warm-start helpers for blackjack DQN from the deterministic basic-strategy policy.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from agents.blackjack_basic_agent import BlackjackBasicAgent


def _make_obs(rng: random.Random) -> Dict[str, Any]:
    dealer = rng.randint(2, 11)
    num_hands = rng.randint(1, 4)
    active_hand = rng.randint(0, num_hands - 1)
    cards_remaining = rng.randint(26, 312)
    running_count = rng.randint(-20, 20)
    true_count = rng.randint(-8, 8)

    scenario = rng.choice(["pair", "soft", "hard"])
    if scenario == "pair":
        pair_rank = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        player_sum = 12 if pair_rank == 11 else pair_rank * 2
        usable_ace = 1 if pair_rank == 11 else 0
        hand_len = 2
        can_split = 1
        can_double = 1
        split_aces_hand = 0
    elif scenario == "soft":
        player_sum = rng.randint(13, 20)
        usable_ace = 1
        hand_len = rng.choice([2, 2, 2, 3, 4])
        pair_rank = 0
        can_split = 0
        can_double = 1 if hand_len == 2 else 0
        split_aces_hand = 0
    else:
        player_sum = rng.randint(5, 20)
        usable_ace = 0
        hand_len = rng.choice([2, 2, 3, 4, 5])
        pair_rank = 0
        can_split = 0
        can_double = 1 if hand_len == 2 else 0
        split_aces_hand = 1 if rng.random() < 0.05 else 0

    if player_sum >= 21:
        can_double = 0
        can_split = 0
        pair_rank = 0
        hand_len = max(2, hand_len)

    return {
        "player_sum": int(player_sum),
        "dealer_showing": int(dealer),
        "usable_ace": int(usable_ace),
        "can_double": int(can_double),
        "can_split": int(can_split),
        "num_hands": int(num_hands),
        "active_hand": int(active_hand),
        "t": int(rng.randint(0, 10)),
        "hand_len": int(hand_len),
        "pair_rank": int(pair_rank),
        "split_aces_hand": int(split_aces_hand),
        "first_action": int(1 if hand_len == 2 else 0),
        "running_count": int(running_count),
        "true_count": int(true_count),
        "cards_remaining": int(cards_remaining),
    }


def pretrain_blackjack_dqn(adapter: Any, *, epochs: int, samples: int, batch_size: int, seed: int = 17) -> Dict[str, Any]:
    if int(epochs) <= 0 or int(samples) <= 0:
        return {"enabled": False, "samples": 0, "epochs": 0, "loss": None}

    rng = random.Random(int(seed))
    baseline = BlackjackBasicAgent(
        spec=adapter.spec,
        observation_space=adapter.observation_space,
        action_space=adapter.action_space,
    )
    obs_rows: List[Dict[str, Any]] = [_make_obs(rng) for _ in range(int(samples))]
    target_actions = [int(baseline.act(obs).action) for obs in obs_rows]
    x = np.stack([adapter._dqn.encode(obs) for obs in obs_rows]).astype(np.float32)
    y = np.array(target_actions, dtype=np.int64)

    x_t = torch.from_numpy(x).to(adapter._dqn.device)
    y_t = torch.from_numpy(y).to(adapter._dqn.device)

    losses: List[float] = []
    indices = list(range(len(obs_rows)))
    for _ in range(int(epochs)):
        rng.shuffle(indices)
        for start in range(0, len(indices), int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            xb = x_t[batch_idx]
            yb = y_t[batch_idx]
            logits = adapter._dqn.q_net(xb)
            loss = F.cross_entropy(logits, yb)
            adapter._dqn.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter._dqn.q_net.parameters(), 10.0)
            adapter._dqn.optimizer.step()
            losses.append(float(loss.item()))

    adapter._dqn.target_net.load_state_dict(adapter._dqn.q_net.state_dict())
    return {
        "enabled": True,
        "samples": int(samples),
        "epochs": int(epochs),
        "loss": (float(sum(losses) / len(losses)) if losses else None),
    }
