"""
tools/update_centroid.py

Trains a centroid policy by distilling the best behaviors from multiple runs.
This script loads high-quality DNA and summarized Lessons, then uses them
to train a generalist policy that can guide new agents.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List


def _obs_key(obs: Any) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return repr(obs)


def _action_key(action: Any) -> str:
    try:
        return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return repr(action)


def load_high_value_data(runs_root: str, min_advantage: float = 0.5) -> List[Dict[str, Any]]:
    """
    Scans all run directories and loads good DNA and lessons.
    """
    print(f"Scanning for high-value data in: {runs_root}")
    distillation_data: List[Dict[str, Any]] = []

    if not os.path.isdir(runs_root):
        print(f"Runs root does not exist: {runs_root}")
        return distillation_data

    for run_id in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, run_id)
        if not os.path.isdir(run_dir):
            continue

        dna_path = os.path.join(run_dir, "dna_good.jsonl")
        if not os.path.isfile(dna_path):
            continue

        with open(dna_path, "r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                try:
                    dna = json.loads(row)
                except json.JSONDecodeError:
                    continue
                if not isinstance(dna, dict):
                    continue
                if "action" not in dna:
                    continue
                try:
                    advantage = float(dna.get("advantage", 0.0))
                except Exception:
                    continue
                if advantage < min_advantage:
                    continue

                distillation_data.append(
                    {
                        "obs": dna.get("obs"),
                        "action": dna.get("action"),
                        "advantage": advantage,
                        "source": "dna",
                    }
                )

    print(f"Loaded {len(distillation_data)} high-advantage state-action pairs.")
    return distillation_data


def train_centroid_policy(data: List[Dict[str, Any]], model_path: str) -> Dict[str, Any]:
    """
    Distills a lightweight centroid policy artifact from state-action traces.
    """
    if not data:
        print("No data available for training. Skipping.")
        return {"saved": False, "reason": "no_data"}

    print(f"Distilling {len(data)} data points into a centroid policy...")

    action_lookup: Dict[str, Any] = {}
    global_action_counts: Counter[str] = Counter()
    obs_action_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    obs_total_weight: Dict[str, float] = defaultdict(float)

    for item in data:
        if not isinstance(item, dict):
            continue
        if "action" not in item:
            continue

        obs = item.get("obs")
        action = item.get("action")
        try:
            weight = float(item.get("advantage", 1.0) or 1.0)
        except Exception:
            weight = 1.0
        if weight <= 0.0:
            continue

        obs_k = _obs_key(obs)
        act_k = _action_key(action)
        action_lookup[act_k] = action
        global_action_counts[act_k] += weight
        obs_action_counts[obs_k][act_k] += weight
        obs_total_weight[obs_k] += weight

    if not global_action_counts:
        print("No valid examples available after parsing. Skipping.")
        return {"saved": False, "reason": "no_valid_examples"}

    default_action_k, default_weight = global_action_counts.most_common(1)[0]
    default_action = action_lookup[default_action_k]
    total_global = float(sum(global_action_counts.values()))

    obs_policy: Dict[str, Dict[str, Any]] = {}
    for obs_k, counts in obs_action_counts.items():
        best_action_k, best_weight = counts.most_common(1)[0]
        total = float(obs_total_weight.get(obs_k, 0.0))
        obs_policy[obs_k] = {
            "action": action_lookup[best_action_k],
            "confidence": float(best_weight / total) if total > 0 else 0.0,
            "support_weight": float(total),
        }

    artifact = {
        "format": "centroid_policy_v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_examples": int(len(data)),
        "num_unique_obs": int(len(obs_policy)),
        "num_unique_actions": int(len(global_action_counts)),
        "default_action": default_action,
        "default_action_confidence": float(default_weight / total_global) if total_global > 0 else 0.0,
        "global_action_weights": {k: float(v) for k, v in global_action_counts.items()},
        "obs_policy": obs_policy,
    }

    out_dir = os.path.dirname(model_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(f"Centroid policy saved to: {model_path}")
    return {
        "saved": True,
        "num_examples": int(len(data)),
        "num_unique_obs": int(len(obs_policy)),
        "num_unique_actions": int(len(global_action_counts)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs", help="Root directory containing all runs")
    ap.add_argument(
        "--model_out",
        type=str,
        default="models/centroid_policy.json",
        help="Path to save the trained centroid policy",
    )
    ap.add_argument(
        "--min_advantage",
        type=float,
        default=0.5,
        help="Minimum advantage for a DNA sample to be included",
    )
    args = ap.parse_args()

    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    distillation_data = load_high_value_data(args.runs_root, args.min_advantage)
    metrics = train_centroid_policy(distillation_data, args.model_out)
    print(f"Centroid update metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
