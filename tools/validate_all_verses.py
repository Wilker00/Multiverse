"""
tools/validate_all_verses.py

Rapid smoke validation across all registered verses.

Per-verse metrics:
- success rate
- mean reward
- safety violations
- convergence score
- confidence intervals / sample adequacy
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.registry import create_agent, register_builtin_agents
from core.agent_base import ActionResult, ExperienceBatch, Transition
from core.types import AgentSpec, VerseSpec
from tools.validation_stats import compute_rate_stats, compute_validation_stats
from verses.registry import create_verse, list_verses, register_builtin


SAFETY_TRUE_KEYS = (
    "wrong_park",
    "collision",
    "crash",
    "boundary_violation",
    "unsafe",
    "failure",
    "fell_cliff",
    "fell_pit",
    "hit_laser",
    "battery_depleted",
    "hit_wall",
    "hit_obstacle",
    "battery_death",
    "safety_violation",
)

CLUSTER_ORDER = [
    "navigation",
    "strategy",
    "economics",
    "planning",
    "memory",
    "other",
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _default_max_steps(verse_name: str) -> int:
    v = str(verse_name).strip().lower()
    if v in ("line_world",):
        return 30
    if v in ("grid_world", "rule_flip_world"):
        return 60
    if v in ("park_world", "chess_world", "go_world", "uno_world", "factory_world", "trade_world"):
        return 80
    if v in ("cliff_world", "bridge_world", "pursuit_world", "warehouse_world"):
        return 100
    if v in ("labyrinth_world", "memory_vault_world", "swamp_world", "escape_world", "harvest_world"):
        return 160
    if v in ("risk_tutorial_world",):
        return 60
    if v in ("wind_master_world",):
        return 80
    return 80


def _cluster_for_verse(verse_name: str) -> str:
    v = str(verse_name).strip().lower()
    if v in (
        "line_world",
        "grid_world",
        "cliff_world",
        "labyrinth_world",
        "park_world",
        "pursuit_world",
        "warehouse_world",
        "harvest_world",
        "swamp_world",
        "escape_world",
        "bridge_world",
    ):
        return "navigation"
    if v in ("chess_world", "go_world", "uno_world"):
        return "strategy"
    if v in ("trade_world", "risk_tutorial_world"):
        return "economics"
    if v in ("wind_master_world",):
        return "navigation"
    if v in ("factory_world",):
        return "planning"
    if v in ("memory_vault_world", "rule_flip_world"):
        return "memory"
    return "other"


def _is_safety_violation(info: Dict[str, Any]) -> bool:
    for k in SAFETY_TRUE_KEYS:
        if info.get(k) is True:
            return True
    return False


def _is_success(info: Dict[str, Any]) -> bool:
    if info.get("reached_goal") is True:
        return True
    if info.get("success") is True:
        return True
    if info.get("won") is True:
        return True
    if info.get("converted_advantage") is True:
        return True
    return False


def _convergence_score(rewards: Sequence[float]) -> float:
    vals = [float(r) for r in rewards]
    n = len(vals)
    if n <= 1:
        return 0.0
    k = max(1, min(3, n // 3))
    first = vals[:k]
    last = vals[-k:]
    std = statistics.pstdev(vals) if n > 1 else 0.0
    return float((statistics.mean(last) - statistics.mean(first)) / (float(std) + 1e-8))


def smoke_test_verse(
    *,
    verse_name: str,
    algo: str,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
    alpha: float,
    power: float,
    min_detectable_delta: float,
) -> Dict[str, Any]:
    verse_max_steps = int(max_steps) if isinstance(max_steps, int) and int(max_steps) > 0 else _default_max_steps(verse_name)
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=str(verse_name),
        verse_version="0.1",
        seed=int(seed),
        params={"max_steps": int(verse_max_steps), "adr_enabled": False},
    )
    agent_cfg: Dict[str, Any] = {"train": True}
    if str(algo).strip().lower() in ("memory_recall", "planner_recall", "special_moe", "adaptive_moe", "gateway"):
        agent_cfg["verse_name"] = str(verse_name)
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=f"smoke_{str(algo).strip().lower()}_{str(verse_name).strip().lower()}",
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
        tags=["smoke_validation"],
        config=agent_cfg,
    )

    verse = create_verse(verse_spec)
    agent = create_agent(agent_spec, verse.observation_space, verse.action_space)

    results: List[Dict[str, Any]] = []
    crash_count = 0
    try:
        for ep in range(max(1, int(episodes))):
            ep_seed = int(seed) * 1000 + int(ep)
            ep_reward = 0.0
            ep_steps = 0
            ep_success = False
            ep_safety = 0
            transitions: List[Transition] = []
            crashed = False

            try:
                verse.seed(ep_seed)
                agent.seed(ep_seed)
                reset_result = verse.reset()
                obs = reset_result.obs

                done = False
                while (not done) and ep_steps < int(verse_max_steps):
                    act_result = agent.act(obs)
                    action = act_result.action if isinstance(act_result, ActionResult) else act_result

                    step_result = verse.step(action)
                    info = step_result.info if isinstance(step_result.info, dict) else {}

                    ep_reward += float(step_result.reward)
                    ep_steps += 1
                    if _is_success(info):
                        ep_success = True
                    if _is_safety_violation(info):
                        ep_safety += 1

                    transitions.append(
                        Transition(
                            obs=obs,
                            action=action,
                            reward=float(step_result.reward),
                            next_obs=step_result.obs,
                            done=bool(step_result.done),
                            truncated=bool(step_result.truncated),
                            info=info,
                        )
                    )
                    obs = step_result.obs
                    done = bool(step_result.done or step_result.truncated)

                if transitions:
                    try:
                        agent.learn(ExperienceBatch(transitions=transitions))
                    except Exception:
                        pass

            except Exception:
                crashed = True
                crash_count += 1

            results.append(
                {
                    "episode": int(ep),
                    "reward": float(ep_reward),
                    "success": bool(ep_success),
                    "safety_violations": int(ep_safety),
                    "steps": int(ep_steps),
                    "crashed": bool(crashed),
                }
            )
    finally:
        try:
            verse.close()
        except Exception:
            pass
        try:
            agent.close()
        except Exception:
            pass

    rewards = [float(r["reward"]) for r in results]
    success_bits = [1.0 if bool(r.get("success", False)) else 0.0 for r in results]
    safety_counts = [_safe_int(r.get("safety_violations", 0), 0) for r in results]
    safety_any = [1.0 if _safe_int(r.get("safety_violations", 0), 0) > 0 else 0.0 for r in results]

    reward_stats = compute_validation_stats(
        rewards,
        alpha=float(alpha),
        power=float(power),
        min_detectable_delta=float(min_detectable_delta),
    )
    success_stats = compute_rate_stats(
        success_bits,
        alpha=float(alpha),
        power=float(power),
        min_detectable_delta=float(min_detectable_delta),
    )
    safety_rate_stats = compute_rate_stats(
        safety_any,
        alpha=float(alpha),
        power=float(power),
        min_detectable_delta=float(min_detectable_delta),
    )

    return {
        "verse": str(verse_name),
        "cluster": _cluster_for_verse(verse_name),
        "episodes": int(len(results)),
        "success_rate": float(success_stats.get("rate", 0.0)),
        "mean_reward": float(reward_stats.get("mean", 0.0)),
        "mean_steps": float(statistics.mean([_safe_int(r.get("steps", 0), 0) for r in results])) if results else 0.0,
        "safety_violations": int(sum(safety_counts)),
        "safety_episode_rate": float(safety_rate_stats.get("rate", 0.0)),
        "convergence_score": float(_convergence_score(rewards)),
        "crashes": int(crash_count),
        "reward_stats": reward_stats,
        "success_stats": success_stats,
        "safety_rate_stats": safety_rate_stats,
        "episode_results": results,
    }


def generate_coverage_report(all_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(all_results)
    by_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        c = str(r.get("cluster", "other"))
        by_cluster.setdefault(c, []).append(r)

    cluster_summary: Dict[str, Any] = {}
    for cluster, items in by_cluster.items():
        cluster_summary[cluster] = {
            "verses": int(len(items)),
            "mean_success_rate": float(sum(_safe_float(x.get("success_rate", 0.0), 0.0) for x in items) / float(max(1, len(items)))),
            "mean_reward": float(sum(_safe_float(x.get("mean_reward", 0.0), 0.0) for x in items) / float(max(1, len(items)))),
            "total_crashes": int(sum(_safe_int(x.get("crashes", 0), 0) for x in items)),
        }

    return {
        "total_verses": int(len(rows)),
        "clusters": cluster_summary,
        "crash_free_verses": int(sum(1 for r in rows if _safe_int(r.get("crashes", 0), 0) == 0)),
        "overall_mean_success_rate": float(
            sum(_safe_float(r.get("success_rate", 0.0), 0.0) for r in rows) / float(max(1, len(rows)))
        ),
    }


def identify_outliers(all_results: Sequence[Dict[str, Any]], *, min_success_rate: float) -> List[Dict[str, Any]]:
    outliers: List[Dict[str, Any]] = []
    for row in all_results:
        reasons: List[str] = []
        sr = _safe_float(row.get("success_rate", 0.0), 0.0)
        crashes = _safe_int(row.get("crashes", 0), 0)
        conv = _safe_float(row.get("convergence_score", 0.0), 0.0)
        if sr < float(min_success_rate):
            reasons.append(f"low_success<{float(min_success_rate):.2f}")
        if crashes > 0:
            reasons.append("episode_crash")
        if conv < -0.10:
            reasons.append("negative_convergence")
        if reasons:
            outliers.append(
                {
                    "verse": str(row.get("verse", "")),
                    "success_rate": float(sr),
                    "crashes": int(crashes),
                    "convergence_score": float(conv),
                    "reasons": reasons,
                }
            )
    outliers.sort(key=lambda x: (x["crashes"], -x["success_rate"]), reverse=True)
    return outliers


def _parse_verses_arg(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    out = []
    for part in str(raw).replace(";", ",").split(","):
        s = str(part).strip().lower()
        if s:
            out.append(s)
    uniq = sorted(set(out))
    return uniq if uniq else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a 19-verse smoke validation sweep.")
    ap.add_argument("--algo", type=str, default="q")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = per-verse defaults")
    ap.add_argument("--verses", type=str, default=None, help="Optional comma-separated subset")
    ap.add_argument("--min_success_outlier", type=float, default=0.30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--min_detectable_delta", type=float, default=0.10)
    ap.add_argument("--out_json", type=str, default=os.path.join("models", "validation", "all_verses_smoke.json"))
    args = ap.parse_args()

    register_builtin()
    register_builtin_agents()

    available = sorted(list_verses().keys())
    wanted = _parse_verses_arg(args.verses)
    verses = [v for v in available if (wanted is None or str(v).strip().lower() in set(wanted))]
    if not verses:
        raise SystemExit("No verses selected for smoke validation.")

    by_cluster: Dict[str, List[str]] = {}
    for v in verses:
        by_cluster.setdefault(_cluster_for_verse(v), []).append(v)

    all_results: List[Dict[str, Any]] = []
    for cluster in CLUSTER_ORDER:
        items = sorted(by_cluster.get(cluster, []))
        if not items:
            continue
        print(f"\n=== Testing {cluster.upper()} Cluster ===")
        for verse_name in items:
            print(f"  {verse_name} ... ", end="", flush=True)
            result = smoke_test_verse(
                verse_name=verse_name,
                algo=str(args.algo),
                episodes=max(1, int(args.episodes)),
                seed=int(args.seed),
                max_steps=(None if int(args.max_steps) <= 0 else int(args.max_steps)),
                alpha=float(args.alpha),
                power=float(args.power),
                min_detectable_delta=float(args.min_detectable_delta),
            )
            all_results.append(result)
            print(
                f"success={float(result['success_rate']):.1%} "
                f"reward={float(result['mean_reward']):.3f} "
                f"crashes={int(result['crashes'])}"
            )

    coverage = generate_coverage_report(all_results)
    outliers = identify_outliers(all_results, min_success_rate=float(args.min_success_outlier))
    report = {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "algo": str(args.algo),
        "episodes_per_verse": int(args.episodes),
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "power": float(args.power),
        "min_detectable_delta": float(args.min_detectable_delta),
        "results": all_results,
        "coverage": coverage,
        "outliers": outliers,
    }

    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Coverage Summary ===")
    print(f"verses tested: {int(coverage['total_verses'])}")
    print(f"crash-free verses: {int(coverage['crash_free_verses'])}")
    print(f"overall mean success: {float(coverage['overall_mean_success_rate']):.1%}")
    print(f"outliers: {len(outliers)}")
    if outliers:
        print("outlier verses:")
        for row in outliers:
            print(f"  - {row['verse']}: {', '.join(row['reasons'])}")
    print(f"report: {args.out_json}")


if __name__ == "__main__":
    main()
