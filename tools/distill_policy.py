"""
tools/distill_policy.py

Train a distilled policy from an apprentice dataset.
"""

from __future__ import annotations

import argparse

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.distilled_agent import DistilledAgent
from core.types import AgentSpec, SpaceSpec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--actions", type=int, required=True, help="number of discrete actions")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    agent = DistilledAgent(
        spec=AgentSpec(
            spec_version="v1",
            policy_id="distilled",
            policy_version="0.0",
            algo="distilled",
            config={"lr": args.lr, "epochs": args.epochs},
        ),
        observation_space=SpaceSpec(type="vector", shape=(1,)),
        action_space=SpaceSpec(type="discrete", n=int(args.actions)),
    )

    stats = agent.train_from_dataset(args.dataset)
    agent.save(args.out_dir)
    print(f"Trained distilled policy: {stats}")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()




