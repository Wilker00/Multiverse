#!/usr/bin/env python3
"""
tools/train_distributed.py

Local distributed training entrypoint built on agents/distributed_training.py.
No Kubernetes assumptions; runs on a single machine.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.distributed_training import LocalDistributedTrainer, DistributedTrainingConfig
from agents.pbt_controller import PBTConfig
from core.types import AgentSpec, VerseSpec


def _parse_kv_list(kvs: Optional[list[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not kvs:
        return out
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Invalid param '{item}'. Expected k=v.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except Exception:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="sharded", choices=["sharded", "pbt"])
    ap.add_argument("--algo", type=str, default="q")
    ap.add_argument("--verse", type=str, default="line_world")
    ap.add_argument("--verse_version", type=str, default="0.1")
    ap.add_argument("--policy_id", type=str, default=None)
    ap.add_argument("--policy_version", type=str, default="0.1")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--run_root", type=str, default="runs")
    ap.add_argument("--vparam", action="append", default=None)
    ap.add_argument("--aconfig", action="append", default=None)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--generations", type=int, default=3, help="PBT generations when --mode pbt.")
    ap.add_argument("--episodes_per_member", type=int, default=30, help="PBT episodes per member per generation.")
    ap.add_argument("--pbt_population_size", type=int, default=4)
    ap.add_argument("--pbt_exploit_interval", type=int, default=2)
    ap.add_argument("--pbt_mutation", action="append", default=None, help="Mutation range key=lo,hi")
    ap.add_argument("--retention_file", type=str, default=os.path.join("central_memory", "retention_multipliers.json"))
    args = ap.parse_args()

    verse_params = _parse_kv_list(args.vparam)
    if args.verse == "line_world":
        verse_params.setdefault("goal_pos", 8)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("step_penalty", -0.02)
    elif args.verse == "cliff_world":
        verse_params.setdefault("width", 12)
        verse_params.setdefault("height", 4)
        verse_params.setdefault("max_steps", args.max_steps)
        verse_params.setdefault("step_penalty", -1.0)
        verse_params.setdefault("cliff_penalty", -100.0)
        verse_params.setdefault("end_on_cliff", False)

    agent_config = _parse_kv_list(args.aconfig)
    if args.train:
        agent_config["train"] = True

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=args.verse,
        verse_version=args.verse_version,
        seed=args.seed,
        tags=["distributed_cli"],
        params=verse_params,
    )
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=args.policy_id or f"dist_{args.algo}",
        policy_version=args.policy_version,
        algo=args.algo,
        seed=args.seed,
        tags=["distributed_cli"],
        config=agent_config if agent_config else None,
    )

    trainer = LocalDistributedTrainer(
        DistributedTrainingConfig(
            workers=max(1, int(args.workers)),
            run_root=args.run_root,
            schema_version="v1",
            merge_results=True,
        )
    )
    if str(args.mode).strip().lower() == "pbt":
        mut_ranges: Dict[str, Any] = {}
        for raw in list(args.pbt_mutation or []):
            if "=" not in raw or "," not in raw:
                raise ValueError(f"Invalid --pbt_mutation '{raw}', expected key=lo,hi")
            k, v = raw.split("=", 1)
            lo_s, hi_s = v.split(",", 1)
            mut_ranges[str(k).strip()] = (float(lo_s), float(hi_s))

        pbt_cfg = PBTConfig(
            enabled=True,
            population_size=max(2, int(args.pbt_population_size)),
            exploit_interval=max(1, int(args.pbt_exploit_interval)),
            mutation_ranges=mut_ranges or PBTConfig().mutation_ranges,
            retention_file=str(args.retention_file),
            seed=int(args.seed),
        )
        out = trainer.train_pbt(
            base_agent_spec=agent_spec,
            verse_spec=verse_spec,
            generations=max(1, int(args.generations)),
            episodes_per_member=max(1, int(args.episodes_per_member)),
            max_steps=max(1, int(args.max_steps)),
            seed=int(args.seed),
            pbt_config=pbt_cfg,
        )
    else:
        out = trainer.train(
            agent_spec=agent_spec,
            verse_spec=verse_spec,
            total_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    print("")
    if str(args.mode).strip().lower() == "pbt":
        print("Distributed PBT training complete")
        best = out.get("best_member") or {}
        print(f"mode              : {out.get('mode')}")
        print(f"generations       : {out.get('generations')}")
        print(f"population_size   : {out.get('population_size')}")
        print(f"best_member_id    : {best.get('member_id')}")
        print(f"best_policy_id    : {best.get('policy_id')}")
        print(f"best_score        : {best.get('score')}")
        print(f"retention_file    : {out.get('retention_file')}")
    else:
        print("Distributed training complete")
        print(f"run_id       : {out['run_id']}")
        print(f"worker_runs  : {len(out['worker_runs'])}")
        print(f"total_steps  : {out['total_steps']}")
        print(f"total_return : {out['total_return']:.3f}")


if __name__ == "__main__":
    main()
