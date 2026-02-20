"""
tools/promotion_report.py

Compare two agents (Candidate vs Baseline) across multiple verses and generate 
 a professional markdown report with KPI deltas.
"""

import argparse
import json
import os
import sys
import numpy as np
from typing import Dict, List, Any

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from agents.registry import register_builtin_agents
from core.rollout import RolloutConfig, run_episodes
from core.types import AgentRef, RunRef, VerseRef
from verses.registry import create_verse, register_builtin as register_builtin_verses

def run_benchmark(verse_name: str, algo: str, episodes: int, max_steps: int) -> Dict[str, Any]:
    print(f"  Benchmarking {algo} on {verse_name}...")
    from core.types import VerseSpec
    v_spec_obj = VerseSpec(
        spec_version="1.0",
        verse_name=verse_name,
        verse_version="0.1",
        params={"max_steps": max_steps}
    )

    verse = create_verse(v_spec_obj)
    
    v_ref = VerseRef.create(verse_name, "0.1", "hash")
    a_ref = AgentRef.create(algo, "0.1")

    from agents.registry import create_agent
    from core.types import AgentSpec
    spec = AgentSpec(
        spec_version="1.0",
        policy_id=algo,
        policy_version="0.1",
        algo=algo,
        seed=42
    )
    agent = create_agent(spec, verse.observation_space, verse.action_space)

    
    run = RunRef.create()
    cfg = RolloutConfig(schema_version="v1", max_steps=max_steps, train=False)
    
    results = run_episodes(
        verse=verse,
        verse_ref=v_ref,
        agent=agent,
        agent_ref=a_ref,
        run=run,
        config=cfg,
        episodes=episodes,
        seed=42
    )
    
    returns = [r.return_sum for r in results]
    successes = [1 if r.return_sum > 0 else 0 for r in results] # Heuristic

    
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "episodes": episodes
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Algo name for candidate")
    parser.add_argument("--baseline", default="random", help="Algo name for baseline")
    parser.add_argument("--verses", default="grid_world,cliff_world,labyrinth_world", help="Comma separated verses")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--out", default="promotion_report.md")
    args = parser.parse_args()

    register_builtin_agents()
    register_builtin_verses()

    verse_list = [v.strip() for v in args.verses.split(",")]
    report = []
    report.append(f"# Promotion Board Report: {args.candidate} vs {args.baseline}")
    report.append(f"Date: 2026-02-11 | Samples: {args.episodes} episodes/verse\n")
    
    report.append("| Verse | Metric | Baseline | Candidate | Delta | Status |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for vname in verse_list:
        b_res = run_benchmark(vname, args.baseline, args.episodes, args.max_steps)
        c_res = run_benchmark(vname, args.candidate, args.episodes, args.max_steps)
        
        ret_delta = c_res["mean_return"] - b_res["mean_return"]
        status = "✅ Improve" if ret_delta > 0 else "❌ Regress"
        
        report.append(f"| {vname} | Return | {b_res['mean_return']:.2f} | {c_res['mean_return']:.2f} | {ret_delta:+.2f} | {status} |")
        
        sr_delta = c_res["success_rate"] - b_res["success_rate"]
        report.append(f"| | Success | {b_res['success_rate']*100:.1f}% | {c_res['success_rate']*100:.1f}% | {sr_delta*100:+.1f}% | |")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"\nReport generated: {args.out}")

if __name__ == "__main__":
    main()
