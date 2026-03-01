"""
Centroid policy training utilities.

Trains a centroid policy from high-value episodes across runs.
A centroid policy is a simple lookup-based policy that stores the most common
high-value action for each observed state.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


def load_high_value_data(
    runs_root: str,
    min_advantage: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Load high-value episodes from runs.

    Scans all run directories under runs_root, finds "dna_good.jsonl" files,
    and extracts episodes with advantage >= min_advantage.

    Args:
        runs_root: Root directory containing run subdirectories
        min_advantage: Minimum advantage threshold to include episode

    Returns:
        List of episode dictionaries with keys: obs, action, advantage
    """
    data = []

    if not os.path.isdir(runs_root):
        return data

    # Iterate through run directories
    for run_name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(run_dir):
            continue

        dna_path = os.path.join(run_dir, "dna_good.jsonl")
        if not os.path.isfile(dna_path):
            continue

        # Read JSONL file with fault tolerance
        with open(dna_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                    advantage = episode.get("advantage", 0.0)
                    if advantage >= min_advantage:
                        data.append(episode)
                except (json.JSONDecodeError, ValueError):
                    # Skip malformed JSON lines
                    continue

    return data


def train_centroid_policy(
    data: List[Dict[str, Any]],
    model_path: str,
) -> Dict[str, Any]:
    """
    Train a centroid policy from episode data.

    A centroid policy computes the most common action for each state,
    weighted by advantage. This serves as a simple imitation baseline.

    Args:
        data: List of episodes with 'obs', 'action', 'advantage' keys
        model_path: Path where to save the centroid policy

    Returns:
        Metrics dictionary with keys:
            - saved: Whether policy was successfully saved
            - reason: If not saved, reason (e.g., "no_data")
            - num_episodes: Number of episodes processed
            - num_states: Number of unique states learned
    """
    if not data:
        return {
            "saved": False,
            "reason": "no_data",
            "num_episodes": 0,
            "num_states": 0,
        }

    # Build action frequency tables by observation
    obs_to_actions = defaultdict(lambda: defaultdict(float))  # obs -> action -> total_advantage
    all_actions = defaultdict(float)  # action -> total_advantage

    for episode in data:
        obs = episode.get("obs")
        action = episode.get("action")
        advantage = episode.get("advantage", 1.0)

        if obs is None or action is None:
            continue

        # Convert obs to JSON string for consistent hashing
        obs_key = json.dumps(obs, sort_keys=True, separators=(",", ":"))

        obs_to_actions[obs_key][action] += advantage
        all_actions[action] += advantage

    if not all_actions:
        return {
            "saved": False,
            "reason": "no_valid_data",
            "num_episodes": len(data),
            "num_states": len(obs_to_actions),
        }

    # Determine default action (most common overall)
    default_action = max(all_actions.items(), key=lambda x: x[1])[0]
    default_action_total = all_actions[default_action]
    default_action_confidence = default_action_total / sum(all_actions.values())

    # Build observation-specific policy
    obs_policy = {}
    for obs_key, action_weights in obs_to_actions.items():
        best_action = max(action_weights.items(), key=lambda x: x[1])[0]
        best_weight = action_weights[best_action]
        confidence = best_weight / sum(action_weights.values())

        obs_policy[obs_key] = {
            "action": int(best_action),
            "confidence": float(confidence),
        }

    # Build artifact
    artifact = {
        "format": "centroid_policy_v1",
        "default_action": int(default_action),
        "default_action_confidence": float(default_action_confidence),
        "obs_policy": obs_policy,
        "num_states": len(obs_policy),
        "num_episodes": len(data),
    }

    # Save to file
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    return {
        "saved": True,
        "num_episodes": len(data),
        "num_states": len(obs_policy),
    }

