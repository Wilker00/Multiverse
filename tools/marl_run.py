"""
tools/marl_run.py

Run multiple agents in parallel with a shared communication bus.
"""

from __future__ import annotations

import argparse
import json
from typing import List

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec, AgentSpec
from orchestrator.marl_trainer import MARLConfig, MultiAgentTrainer


def _specs(n: int, verse: str, verse_version: str, algo: str, seed: int) -> tuple[list[VerseSpec], list[AgentSpec]]:
    verse_specs: List[VerseSpec] = []
    agent_specs: List[AgentSpec] = []

    for i in range(n):
        verse_specs.append(
            VerseSpec(
                spec_version="v1",
                verse_name=verse,
                verse_version=verse_version,
                seed=seed + i,
                tags=["marl"],
                params={
                    "goal_pos": 8,
                    "max_steps": 40,
                    "step_penalty": -0.02,
                },
            )
        )

        agent_specs.append(
            AgentSpec(
                spec_version="v1",
                policy_id=f"{algo}_{i}",
                policy_version="0.0",
                algo=algo,
                seed=seed + i,
            )
        )

    return verse_specs, agent_specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--verse", type=str, default="line_world")
    ap.add_argument("--verse_version", type=str, default="0.1")
    ap.add_argument("--algo", type=str, default="random")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--shared_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shared_top_k", type=int, default=4)
    ap.add_argument("--negotiation_interval", type=int, default=1)
    ap.add_argument("--lexicon_min_support", type=int, default=2)
    ap.add_argument("--out_json", type=str, default="", help="Optional path to write MARL social summary JSON.")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    verse_specs, agent_specs = _specs(args.agents, args.verse, args.verse_version, args.algo, args.seed)

    trainer = MultiAgentTrainer(run_root=args.runs_root, schema_version="v1", auto_register_builtin=True)
    cfg = MARLConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        train=args.train,
        collect_transitions=args.train,
        shared_memory_enabled=bool(args.shared_memory),
        shared_memory_top_k=max(1, int(args.shared_top_k)),
        negotiation_interval=max(1, int(args.negotiation_interval)),
        lexicon_min_support=max(1, int(args.lexicon_min_support)),
    )

    out = trainer.run(verse_specs=verse_specs, agent_specs=agent_specs, config=cfg, seed=args.seed)
    if str(args.out_json).strip():
        path = str(args.out_json).strip()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"social_summary={path}")


if __name__ == "__main__":
    main()




