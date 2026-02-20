"""
tools/uas_case_study.py

Case-study CLI for the Universe Agent System (UAS/Multiverse). This tool demonstrates a simple workflow of:
- ingest run memories into a centralized repository
- match new scenarios against centralized memory
- run a multi-agent loop and auto-ingest its output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from memory.central_repository import CentralMemoryConfig, ingest_run
from memory.selection import SelectionConfig
from orchestrator.marl_trainer import MARLConfig, MultiAgentTrainer
from orchestrator.scenario_matcher import ScenarioRequest, recommend_action


def _list_run_dirs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[str] = []
    for name in os.listdir(runs_root):
        p = os.path.join(runs_root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "events.jsonl")):
            out.append(p)
    out.sort()
    return out


def _parse_obs(obs_str: str):
    value = json.loads(obs_str)
    if not isinstance(value, (dict, list, int, float)):
        raise ValueError("--obs must be a JSON dict/list/number")
    return value


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = CentralMemoryConfig(root_dir=args.memory_dir)
    selection = SelectionConfig(
        min_reward=args.min_reward,
        max_reward=args.max_reward,
        keep_top_k_per_episode=args.top_k_per_episode,
        keep_top_k_episodes=args.top_k_episodes,
        novelty_bonus=args.novelty_bonus,
    )

    run_dirs: List[str] = []
    if args.run_dir:
        run_dirs = [args.run_dir]
    else:
        run_dirs = _list_run_dirs(args.runs_root)

    if not run_dirs:
        print("No run directories found to ingest.")
        return

    total_added = 0
    for rd in run_dirs:
        stats = ingest_run(
            run_dir=rd,
            cfg=cfg,
            selection=selection,
            max_events=args.max_events,
        )
        total_added += stats.added_events
        print(
            f"{os.path.basename(rd)}: "
            f"input={stats.input_events} selected={stats.selected_events} "
            f"added={stats.added_events} skipped={stats.skipped_duplicates}"
        )
    print(f"Total added memories: {total_added}")


def cmd_match(args: argparse.Namespace) -> None:
    cfg = CentralMemoryConfig(root_dir=args.memory_dir)
    request = ScenarioRequest(
        obs=_parse_obs(args.obs),
        verse_name=args.verse,
        top_k=args.top_k,
        min_score=args.min_score,
        semantic_fallback_threshold=args.semantic_threshold,
        enable_semantic_bridge=not bool(args.no_semantic_bridge),
        enable_tag_fallback=not bool(args.no_tag_fallback),
        cross_verse_pool=args.cross_verse_pool,
        enable_knowledge_graph=not bool(args.no_knowledge_graph),
        min_graph_relatedness=args.min_graph_relatedness,
        graph_bonus_scale=args.graph_bonus_scale,
        enable_confidence_auditor=not bool(args.no_confidence_auditor),
        decay_lambda=float(args.decay_lambda),
    )
    advice = recommend_action(request=request, cfg=cfg)
    if advice is None:
        print("No matching memories found.")
        return

    print("Scenario recommendation")
    print(f"action     : {advice.action}")
    print(f"confidence : {advice.confidence:.3f}")
    print(f"strategy   : {advice.strategy}")
    print(f"bridge_src : {advice.bridge_source_verse}")
    print(f"direct_k   : {advice.direct_candidates}")
    print(f"semantic_k : {advice.semantic_candidates}")
    print(f"candidates : {len(advice.matches)}")
    print("")
    print("Top matches:")
    for m in advice.matches:
        print(
            f"  score={m.score:.3f} verse={m.verse_name} run={m.run_id} "
            f"ep={m.episode_id} step={m.step_idx} action={m.action} reward={m.reward:.3f}"
        )


def cmd_loop(args: argparse.Namespace) -> None:
    verses = [v.strip() for v in args.verses.split(",") if v.strip()]
    if not verses:
        raise ValueError("--verses must contain at least one verse name")

    verse_specs: List[VerseSpec] = []
    agent_specs: List[AgentSpec] = []
    for i in range(args.agents):
        verse_name = verses[i % len(verses)]
        verse_specs.append(
            VerseSpec(
                spec_version="v1",
                verse_name=verse_name,
                verse_version="0.1",
                seed=(args.seed + i) if args.seed is not None else None,
                tags=["uas_loop"],
                params={},
            )
        )
        agent_specs.append(
            AgentSpec(
                spec_version="v1",
                policy_id=f"{args.algo}_uas_{i}",
                policy_version="0.1",
                algo=args.algo,
                seed=(args.seed + i) if args.seed is not None else None,
                config={"train": bool(args.train)},
            )
        )

    trainer = MultiAgentTrainer(run_root=args.runs_root, schema_version="v1", auto_register_builtin=True)
    result = trainer.run(
        verse_specs=verse_specs,
        agent_specs=agent_specs,
        config=MARLConfig(
            episodes=args.episodes,
            max_steps=args.max_steps,
            train=bool(args.train),
            collect_transitions=bool(args.train),
        ),
        seed=args.seed,
    )

    run_dir = os.path.join(args.runs_root, result["run_id"])
    stats = ingest_run(
        run_dir=run_dir,
        cfg=CentralMemoryConfig(root_dir=args.memory_dir),
        selection=SelectionConfig(
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            keep_top_k_per_episode=args.top_k_per_episode,
            keep_top_k_episodes=args.top_k_episodes,
            novelty_bonus=args.novelty_bonus,
        ),
        max_events=args.max_events,
    )
    print("")
    print("Loop ingest summary")
    print(
        f"run_id={result['run_id']} added={stats.added_events} "
        f"selected={stats.selected_events} skipped={stats.skipped_duplicates}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="ingest run memories into centralized repository")
    ing.add_argument("--memory_dir", type=str, default="central_memory")
    ing.add_argument("--run_dir", type=str, default=None)
    ing.add_argument("--runs_root", type=str, default="runs")
    ing.add_argument("--max_events", type=int, default=None)
    ing.add_argument("--min_reward", type=float, default=-1e9)
    ing.add_argument("--max_reward", type=float, default=1e9)
    ing.add_argument("--top_k_per_episode", type=int, default=0)
    ing.add_argument("--top_k_episodes", type=int, default=0)
    ing.add_argument("--novelty_bonus", type=float, default=0.0)
    ing.set_defaults(func=cmd_ingest)

    mt = sub.add_parser("match", help="match a scenario against centralized memory")
    mt.add_argument("--memory_dir", type=str, default="central_memory")
    mt.add_argument("--obs", type=str, required=True, help="JSON observation payload")
    mt.add_argument("--verse", type=str, default=None)
    mt.add_argument("--top_k", type=int, default=5)
    mt.add_argument("--min_score", type=float, default=0.0)
    mt.add_argument("--semantic_threshold", type=float, default=0.35)
    mt.add_argument("--cross_verse_pool", type=int, default=250)
    mt.add_argument("--no_semantic_bridge", action="store_true")
    mt.add_argument("--no_tag_fallback", action="store_true")
    mt.add_argument("--no_knowledge_graph", action="store_true")
    mt.add_argument("--min_graph_relatedness", type=float, default=0.05)
    mt.add_argument("--graph_bonus_scale", type=float, default=0.15)
    mt.add_argument("--no_confidence_auditor", action="store_true")
    mt.add_argument("--decay_lambda", type=float, default=0.0, help="Temporal decay lambda for retrieval freshness.")
    mt.set_defaults(func=cmd_match)

    lp = sub.add_parser("loop", help="run a MARL loop and ingest resulting run")
    lp.add_argument("--memory_dir", type=str, default="central_memory")
    lp.add_argument("--runs_root", type=str, default="runs")
    lp.add_argument("--agents", type=int, default=4)
    lp.add_argument("--verses", type=str, default="line_world,grid_world,cliff_world,park_world,pursuit_world")
    lp.add_argument("--algo", type=str, default="random")
    lp.add_argument("--episodes", type=int, default=5)
    lp.add_argument("--max_steps", type=int, default=40)
    lp.add_argument("--seed", type=int, default=123)
    lp.add_argument("--train", action="store_true")
    lp.add_argument("--max_events", type=int, default=None)
    lp.add_argument("--min_reward", type=float, default=-1e9)
    lp.add_argument("--max_reward", type=float, default=1e9)
    lp.add_argument("--top_k_per_episode", type=int, default=0)
    lp.add_argument("--top_k_episodes", type=int, default=0)
    lp.add_argument("--novelty_bonus", type=float, default=0.0)
    lp.set_defaults(func=cmd_loop)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
