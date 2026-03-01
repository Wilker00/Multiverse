"""
tools/eval_generalist.py

Live deployment evaluation of generalist Decision Transformer models.
Rolls out the model in each verse environment and measures actual task 
performance — success rates, mean returns, steps to completion.

This is NOT held-out data evaluation. The agent acts in the environments.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.transformer_agent import TransformerAgent
from core.types import AgentSpec, VerseSpec
from verses.registry import create_verse, register_builtin

ALL_VERSES = [
    "line_world", "trade_world", "grid_world", "cliff_world",
    "maze_world", "swamp_world", "wind_master_world",
    "warehouse_world", "escape_world",
    "chess_world", "go_world", "uno_world",
    "factory_world",
]

# Default verse params matching train_agent.py defaults
VERSE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "line_world":       {"goal_pos": 8, "step_penalty": -0.02},
    "grid_world":       {},
    "cliff_world":      {"width": 12, "height": 4, "step_penalty": -1.0, "cliff_penalty": -100.0, "end_on_cliff": False},
    "maze_world":       {},
    "swamp_world":      {"width": 10, "height": 10, "flood_rate": 5, "mud_count": 6, "haven_count": 2},
    "wind_master_world":{"width": 14, "height": 7, "gust_probability": 0.12, "target_margin": 2, "margin_reward_scale": 0.05},
    "warehouse_world":  {},
    "escape_world":     {"width": 10, "height": 10, "num_guards": 3, "num_hiding_spots": 4, "guard_vision": 2},
    "trade_world":      {"starting_cash": 100.0, "max_inventory": 10, "cycle_length": 20},
    "chess_world":      {"win_material": 8, "lose_material": -8, "random_swing": 0.20},
    "go_world":         {"target_territory": 10, "random_swing": 0.25},
    "uno_world":        {"start_hand": 7, "opp_start_hand": 7, "random_swing": 0.25},
    "factory_world":    {"num_machines": 3, "buffer_size": 4, "arrival_rate": 0.6, "breakdown_prob": 0.08},
}


def evaluate_model_live(
    model_path: str,
    verses: List[str],
    episodes_per_verse: int = 20,
    max_steps: int = 80,
    seed: int = 42,
    target_return: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Deploy model in each verse and collect real performance metrics."""
    results: Dict[str, Dict[str, float]] = {}

    for verse_name in verses:
        t0 = time.time()
        params = dict(VERSE_DEFAULTS.get(verse_name, {}))
        params["max_steps"] = max_steps

        vspec = VerseSpec(
            spec_version="v1", verse_name=verse_name,
            verse_version="0.1", seed=seed, params=params,
        )
        verse = create_verse(vspec)
        verse.seed(seed)

        _TARGET_MAP = {
            "line_world": 0.85, "cliff_world": -75.0, "grid_world": -0.4, "maze_world": -1.5,
            "warehouse_world": -0.6, "swamp_world": 0.5, "wind_master_world": 0.2,
            "escape_world": 0.3, "trade_world": 0.0, "factory_world": 15.0,
            "chess_world": 0.5, "go_world": 0.5, "uno_world": 0.0
        }
        verse_target = _TARGET_MAP.get(verse_name, target_return)

        agent_spec = AgentSpec(
            spec_version="v1", policy_id="generalist_eval",
            policy_version="0.1", algo="adt", seed=seed,
            config={
                "model_path": model_path,
                "device": "cpu",
                "target_return": verse_target,
                "recall_enabled": False,   # pure model evaluation, no memory
                "sample": False,           # greedy decoding
                "verse_name": verse_name,
            },
        )
        agent = TransformerAgent(
            spec=agent_spec,
            observation_space=verse.observation_space,
            action_space=verse.action_space,
        )
        agent.seed(seed)

        successes = 0
        total_return = 0.0
        total_steps = 0
        episode_returns: List[float] = []

        for ep in range(episodes_per_verse):
            rr = verse.reset()
            obs = rr.obs
            ep_ret = 0.0
            ep_steps = 0
            ep_success = False

            for _ in range(max_steps):
                ar = agent.act(obs)
                sr = verse.step(ar.action)
                info = sr.info if isinstance(sr.info, dict) else {}
                ep_ret += float(sr.reward)
                ep_steps += 1

                if info.get("reached_goal") or info.get("success"):
                    ep_success = True
                if sr.done or sr.truncated:
                    break
                obs = sr.obs

            successes += int(ep_success)
            total_return += ep_ret
            total_steps += ep_steps
            episode_returns.append(ep_ret)

        n = max(1, episodes_per_verse)
        elapsed = time.time() - t0
        
        results[verse_name] = {
            "success_rate": float(successes) / n,
            "mean_return": float(total_return) / n,
            "mean_steps": float(total_steps) / n,
            "min_return": min(episode_returns) if episode_returns else 0.0,
            "max_return": max(episode_returns) if episode_returns else 0.0,
            "episodes": float(n),
            "elapsed_s": round(elapsed, 1),
        }

        verse.close()
        agent.close()

        sr_pct = results[verse_name]["success_rate"] * 100
        mr = results[verse_name]["mean_return"]
        ms = results[verse_name]["mean_steps"]
        print(f"  {verse_name:25s}  success={sr_pct:5.1f}%  mean_ret={mr:8.2f}  mean_steps={ms:5.1f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

    return results


def main():
    ap = argparse.ArgumentParser(description="Live deployment evaluation of ADT generalist models")
    ap.add_argument("--model", type=str, required=True, help="Path to .pt model checkpoint")
    ap.add_argument("--verses", type=str, default=",".join(ALL_VERSES),
                    help="Comma-separated verse list (default: all 13)")
    ap.add_argument("--episodes", type=int, default=20, help="Episodes per verse")
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_return", type=float, default=1.0)
    ap.add_argument("--output_json", type=str, default=None, help="Save results JSON to this path")
    ap.add_argument("--compare", type=str, default=None,
                    help="Second model path for side-by-side comparison")
    args = ap.parse_args()

    register_builtin()
    verses = [v.strip() for v in args.verses.split(",") if v.strip()]

    print(f"=" * 80)
    print(f"LIVE DEPLOYMENT EVALUATION")
    print(f"=" * 80)
    print(f"Model:    {args.model}")
    print(f"Verses:   {len(verses)}")
    print(f"Episodes: {args.episodes} per verse")
    print(f"Seed:     {args.seed}")
    print(f"=" * 80)
    print()

    # ── Model A ──────────────────────────────────────────────────────────────
    print(f"[Model A] {os.path.basename(args.model)}")
    print(f"  {'Verse':25s}  {'Success':>8s}  {'Mean Ret':>10s}  {'Mean Steps':>10s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*10}")
    sys.stdout.flush()
    
    results_a = evaluate_model_live(
        model_path=args.model, verses=verses,
        episodes_per_verse=args.episodes, max_steps=args.max_steps,
        seed=args.seed, target_return=args.target_return,
    )

    # Aggregate
    all_sr = [v["success_rate"] for v in results_a.values()]
    all_mr = [v["mean_return"] for v in results_a.values()]
    avg_sr = sum(all_sr) / max(1, len(all_sr))
    avg_mr = sum(all_mr) / max(1, len(all_mr))
    print(f"\n  {'AGGREGATE':25s}  success={avg_sr*100:5.1f}%  mean_ret={avg_mr:8.2f}")
    print()

    # ── Model B (optional comparison) ────────────────────────────────────────
    results_b = None
    if args.compare:
        print(f"[Model B] {os.path.basename(args.compare)}")
        print(f"  {'Verse':25s}  {'Success':>8s}  {'Mean Ret':>10s}  {'Mean Steps':>10s}")
        print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*10}")
        sys.stdout.flush()
        
        results_b = evaluate_model_live(
            model_path=args.compare, verses=verses,
            episodes_per_verse=args.episodes, max_steps=args.max_steps,
            seed=args.seed, target_return=args.target_return,
        )
        all_sr_b = [v["success_rate"] for v in results_b.values()]
        all_mr_b = [v["mean_return"] for v in results_b.values()]
        avg_sr_b = sum(all_sr_b) / max(1, len(all_sr_b))
        avg_mr_b = sum(all_mr_b) / max(1, len(all_mr_b))
        print(f"\n  {'AGGREGATE':25s}  success={avg_sr_b*100:5.1f}%  mean_ret={avg_mr_b:8.2f}")
        print()

        # ── Side-by-side comparison ──────────────────────────────────────────
        print(f"{'='*80}")
        print(f"COMPARISON: {os.path.basename(args.model)} vs {os.path.basename(args.compare)}")
        print(f"{'='*80}")
        print(f"  {'Verse':25s} | {'A Success':>10s}  {'A Return':>10s} | {'B Success':>10s}  {'B Return':>10s} | {'Delta SR':>9s}")
        print(f"  {'-'*25}-+-{'-'*10}--{'-'*10}-+-{'-'*10}--{'-'*10}-+-{'-'*9}")
        for vn in verses:
            va = results_a.get(vn, {})
            vb = results_b.get(vn, {})
            sr_a = va.get("success_rate", 0) * 100
            sr_b = vb.get("success_rate", 0) * 100
            mr_a = va.get("mean_return", 0)
            mr_b = vb.get("mean_return", 0)
            delta = sr_a - sr_b
            marker = ">>>" if abs(delta) > 10 else ""
            print(f"  {vn:25s} | {sr_a:9.1f}%  {mr_a:10.2f} | {sr_b:9.1f}%  {mr_b:10.2f} | {delta:+8.1f}% {marker}")
        print(f"\n  Aggregate: A={avg_sr*100:.1f}% / B={avg_sr_b*100:.1f}%  delta={(avg_sr-avg_sr_b)*100:+.1f}%")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out = {"model_a": {"path": args.model, "results": results_a}}
    if results_b is not None:
        out["model_b"] = {"path": args.compare, "results": results_b}
    
    out_path = args.output_json or "models/eval_results.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
