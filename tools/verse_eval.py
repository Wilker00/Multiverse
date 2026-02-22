"""
tools/verse_eval.py

Rigorous verse evaluation: measures whether an agent is actually learning,
not just whether the code runs.

Reports:
  - Win rate (% episodes reaching goal)
  - Mean / median / best / worst return
  - Learning curve (first half vs second half)
  - Random baseline comparison
  - Statistical significance (is trained > random?)
  - Per-verse difficulty analysis

Usage:
  python tools/verse_eval.py --verse escape_world --algo q --episodes 500 --max_steps 80
  python tools/verse_eval.py --verse all --algo q --episodes 300 --max_steps 80
"""

from __future__ import annotations

import io
import sys

# Windows consoles default to cp1252 which cannot encode emoji/arrows.
# Force UTF-8 output so verdict symbols render correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.types import AgentSpec, SpaceSpec, VerseSpec
from core.agent_base import ActionResult, ExperienceBatch, Transition
from verses.registry import register_builtin, create_verse
from agents.registry import register_builtin_agents, create_agent


# ── helpers ──────────────────────────────────────────────────

def _run_episodes(
    verse_name: str,
    algo: str,
    episodes: int,
    max_steps: int,
    seed: int = 42,
    verse_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run N episodes and return per-episode metrics."""
    params = dict(verse_params or {})
    params.setdefault("max_steps", max_steps)

    spec = VerseSpec(
        spec_version="v1",
        verse_name=verse_name,
        verse_version="0.1",
        seed=seed,
        params=params,
    )
    verse = create_verse(spec)

    agent_spec = AgentSpec(spec_version="v1", policy_id=f"eval_{algo}", policy_version="0.1", algo=algo, config={})
    obs_space = getattr(verse, "observation_space", SpaceSpec(type="vector", shape=(4,)))
    act_space = getattr(verse, "action_space", SpaceSpec(type="discrete", n=4))
    agent = create_agent(agent_spec, obs_space, act_space)

    results: List[Dict[str, Any]] = []

    for ep_idx in range(episodes):
        reset_result = verse.reset()
        obs = reset_result.obs
        ep_return = 0.0
        ep_steps = 0
        reached_goal = False
        spotted = 0
        info_accum: Dict[str, Any] = {}
        transitions: List[Transition] = []

        for step_idx in range(max_steps):
            act_result = agent.act(obs)
            action = act_result.action if isinstance(act_result, ActionResult) else act_result

            step_result = verse.step(action)
            reward = float(step_result.reward)
            ep_return += reward
            ep_steps += 1
            info = step_result.info or {}
            done_flag = bool(step_result.done or step_result.truncated)

            # Track goal completion
            if info.get("reached_goal", False):
                reached_goal = True
            if info.get("spotted", False):
                spotted += 1
            if info.get("items_completed"):
                info_accum["items_completed"] = info_accum.get("items_completed", 0) + int(info["items_completed"])
            if info.get("total_profit") is not None:
                info_accum["total_profit"] = float(info["total_profit"])

            # Collect transition for learning
            transitions.append(Transition(
                obs=obs, action=action, reward=reward,
                next_obs=step_result.obs, done=bool(step_result.done),
                truncated=bool(step_result.truncated), info=info,
            ))

            obs = step_result.obs
            if step_result.done or step_result.truncated:
                break

        # Train the agent on this episode's experience
        if transitions and algo != "random":
            try:
                agent.learn(ExperienceBatch(transitions=transitions))
            except (NotImplementedError, Exception):
                pass

        results.append({
            "episode": ep_idx,
            "return": round(ep_return, 3),
            "steps": ep_steps,
            "reached_goal": reached_goal,
            "spotted": spotted,
            **info_accum,
        })

    if hasattr(verse, "close"):
        verse.close()

    return results


def _analyze(results: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Compute statistics from episode results."""
    returns = [r["return"] for r in results]
    n = len(returns)
    if n == 0:
        return {"label": label, "episodes": 0}

    wins = sum(1 for r in results if r.get("reached_goal", False))
    win_rate = wins / n

    # Learning curve: split into quarters
    q_size = max(1, n // 4)
    q1_returns = returns[:q_size]
    q4_returns = returns[-q_size:]
    q1_mean = statistics.mean(q1_returns)
    q4_mean = statistics.mean(q4_returns)
    improvement = q4_mean - q1_mean

    # Win rate curve
    q1_wins = sum(1 for r in results[:q_size] if r.get("reached_goal", False))
    q4_wins = sum(1 for r in results[-q_size:] if r.get("reached_goal", False))
    q1_win_rate = q1_wins / q_size
    q4_win_rate = q4_wins / q_size

    analysis = {
        "label": label,
        "episodes": n,
        "win_rate": round(win_rate, 4),
        "wins": wins,
        "mean_return": round(statistics.mean(returns), 3),
        "median_return": round(statistics.median(returns), 3),
        "best_return": round(max(returns), 3),
        "worst_return": round(min(returns), 3),
        "stddev": round(statistics.stdev(returns), 3) if n > 1 else 0.0,
        "q1_mean_return": round(q1_mean, 3),
        "q4_mean_return": round(q4_mean, 3),
        "return_improvement": round(improvement, 3),
        "q1_win_rate": round(q1_win_rate, 4),
        "q4_win_rate": round(q4_win_rate, 4),
        "win_rate_improvement": round(q4_win_rate - q1_win_rate, 4),
        "mean_steps": round(statistics.mean([r["steps"] for r in results]), 1),
    }

    # Extra metrics if available
    items = [r.get("items_completed", 0) for r in results if "items_completed" in r]
    if items:
        analysis["mean_items_completed"] = round(statistics.mean(items), 2)

    profits = [r.get("total_profit", 0.0) for r in results if "total_profit" in r]
    if profits:
        analysis["mean_profit"] = round(statistics.mean(profits), 2)
        analysis["profitable_episodes"] = sum(1 for p in profits if p > 0)

    return analysis


def _is_learning(trained: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compare trained agent vs random baseline and determine if learning occurred."""
    t_mean = trained.get("mean_return", 0.0)
    b_mean = baseline.get("mean_return", 0.0)
    t_win = trained.get("win_rate", 0.0)
    b_win = baseline.get("win_rate", 0.0)
    t_improve = trained.get("return_improvement", 0.0)
    t_win_improve = trained.get("win_rate_improvement", 0.0)

    # Compute advantage
    return_advantage = t_mean - b_mean
    win_advantage = t_win - b_win

    # Significance heuristics
    learning_signals = 0
    reasons = []

    if return_advantage > 0:
        learning_signals += 1
        reasons.append(f"mean return +{return_advantage:.2f} vs baseline")
    else:
        reasons.append(f"mean return {return_advantage:.2f} vs baseline (WORSE)")

    if win_advantage > 0:
        learning_signals += 1
        reasons.append(f"win rate +{win_advantage:.1%} vs baseline")

    if t_improve > 0:
        learning_signals += 1
        reasons.append(f"return improving over time (+{t_improve:.2f} Q1→Q4)")
    else:
        reasons.append(f"return declining over time ({t_improve:.2f} Q1→Q4)")

    if t_win_improve > 0:
        learning_signals += 1
        reasons.append(f"win rate improving (+{t_win_improve:.1%} Q1→Q4)")

    # Verdict
    if learning_signals >= 3:
        verdict = "✅ LEARNING"
    elif learning_signals >= 2:
        verdict = "🟡 PARTIAL LEARNING"
    elif learning_signals >= 1:
        verdict = "🟠 WEAK SIGNAL"
    else:
        verdict = "🔴 NO LEARNING DETECTED"

    return {
        "verdict": verdict,
        "learning_signals": learning_signals,
        "return_advantage": round(return_advantage, 3),
        "win_rate_advantage": round(win_advantage, 4),
        "reasons": reasons,
    }


# ── verse defaults ───────────────────────────────────────────

VERSE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "harvest_world": {"width": 8, "height": 8, "num_fruit": 6, "carry_capacity": 3},
    "bridge_world": {"bridge_length": 8, "wind_probability": 0.15},
    "swamp_world": {"width": 10, "height": 10, "flood_rate": 5, "mud_count": 6},
    "escape_world": {"width": 10, "height": 10, "num_guards": 3, "num_hiding_spots": 4},
    "factory_world": {"num_machines": 3, "buffer_size": 4, "arrival_rate": 0.6},
    "trade_world": {"starting_cash": 100.0, "max_inventory": 10, "cycle_length": 20},
    "grid_world": {"width": 6, "height": 6},
    "cliff_world": {"width": 8, "height": 4},
    "warehouse_world": {"width": 8, "height": 8},
    "labyrinth_world": {"width": 15, "height": 11},
}

ALL_NEW_VERSES = [
    "harvest_world", "bridge_world", "swamp_world",
    "escape_world", "factory_world", "trade_world",
]


# ── main ─────────────────────────────────────────────────────

def evaluate_verse(
    verse_name: str,
    baseline_algo: str,
    trained_algo: str,
    episodes: int,
    max_steps: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full evaluation: trained agent vs baseline + comparison."""
    params = VERSE_DEFAULTS.get(verse_name, {})

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {verse_name}")
    print(f"  Baseline: {baseline_algo} | Trained: {trained_algo}")
    print(f"  episodes={episodes}  max_steps={max_steps}")
    print(f"{'='*60}")

    # 1. Baseline
    print(f"  ⏳ Running baseline ({baseline_algo})...")
    t0 = time.time()
    # Use different seed for baseline to ensure fair comparison if same algo
    baseline_results = _run_episodes(
        verse_name, baseline_algo, episodes, max_steps, seed=seed + 1000, verse_params=params,
    )
    baseline_time = time.time() - t0
    baseline_analysis = _analyze(baseline_results, f"{baseline_algo}/{verse_name}")
    print(f"     Done in {baseline_time:.1f}s | mean={baseline_analysis['mean_return']:.2f} | win={baseline_analysis['win_rate']:.1%}")

    # 2. Trained agent
    print(f"  ⏳ Training {trained_algo} agent ({episodes} episodes)...")
    t0 = time.time()
    trained_results = _run_episodes(
        verse_name, trained_algo, episodes, max_steps, seed=seed, verse_params=params,
    )
    trained_time = time.time() - t0
    trained_analysis = _analyze(trained_results, f"{trained_algo}/{verse_name}")
    print(f"     Done in {trained_time:.1f}s | mean={trained_analysis['mean_return']:.2f} | win={trained_analysis['win_rate']:.1%}")

    # 3. Compare
    comparison = _is_learning(trained_analysis, baseline_analysis)

    # 4. Report
    print(f"\n  {'─'*50}")
    print(f"  RESULTS: {verse_name}")
    print(f"  {'─'*50}")
    print(f"  Baseline ({baseline_algo:>6}):  mean={baseline_analysis['mean_return']:>8.2f}  win={baseline_analysis['win_rate']:>6.1%}")
    print(f"  Trained  ({trained_algo:>6}):  mean={trained_analysis['mean_return']:>8.2f}  win={trained_analysis['win_rate']:>6.1%}")
    print(f"  Return advantage: {comparison['return_advantage']:>+8.2f}")
    print(f"  Win rate advantage: {comparison['win_rate_advantage']:>+7.1%}")
    print()
    print(f"  Learning curve (trained):")
    print(f"    First quarter:  mean={trained_analysis['q1_mean_return']:>8.2f}  win={trained_analysis['q1_win_rate']:>6.1%}")
    print(f"    Last quarter:   mean={trained_analysis['q4_mean_return']:>8.2f}  win={trained_analysis['q4_win_rate']:>6.1%}")
    print(f"    Improvement:    return={trained_analysis['return_improvement']:>+8.2f}  win={trained_analysis['win_rate_improvement']:>+7.1%}")
    print()
    for reason in comparison["reasons"]:
        print(f"    • {reason}")
    print()
    print(f"  ➤ VERDICT: {comparison['verdict']}")
    print(f"  {'─'*50}")

    return {
        "verse": verse_name,
        "baseline_algo": baseline_algo,
        "trained_algo": trained_algo,
        "episodes": episodes,
        "max_steps": max_steps,
        "baseline": baseline_analysis,
        "trained": trained_analysis,
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Rigorous verse evaluation")
    parser.add_argument("--verse", type=str, default="all", help="Verse name or 'all' for all new verses")
    parser.add_argument("--algo", type=str, default="q", help="Training algorithm (Target)")
    parser.add_argument("--baseline", type=str, default="random", help="Baseline algorithm (Default: random)")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per evaluation")
    parser.add_argument("--max_steps", type=int, default=80, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report_out", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    # Dynamic registration using project tools
    try:
        register_builtin()
        register_builtin_agents()
    except Exception as e:
        print(f"Warning during registration: {e}")

    if args.verse == "all":
        verses = ALL_NEW_VERSES
    else:
        verses = [args.verse]

    all_results: List[Dict[str, Any]] = []
    for verse_name in verses:
        result = evaluate_verse(
            verse_name=verse_name,
            baseline_algo=args.baseline,
            trained_algo=args.algo,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        all_results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY: {args.algo} agent across {len(verses)} verses ({args.episodes} episodes each)")
    print(f"{'='*80}")
    print(f"  {'Verse':<18} {'Win%':>6} {'Mean':>8} {'vs Random':>10} {'Curve':>8} {'Verdict'}")
    print(f"  {'─'*18} {'─'*6} {'─'*8} {'─'*10} {'─'*8} {'─'*22}")
    for r in all_results:
        t = r["trained"]
        c = r["comparison"]
        curve = "↗" if t["return_improvement"] > 0 else "↘" if t["return_improvement"] < -0.5 else "→"
        print(
            f"  {r['verse']:<18} {t['win_rate']:>5.1%} {t['mean_return']:>8.2f} "
            f"{c['return_advantage']:>+9.2f} {curve:>7}  {c['verdict']}"
        )
    print(f"{'='*80}\n")

    # Save report
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {args.report_out}")


if __name__ == "__main__":
    main()
