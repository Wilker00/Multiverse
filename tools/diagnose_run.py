"""
tools/diagnose_run.py

Quick diagnostic for any run — prints SafeExecutor, MCTS, and transfer health
metrics from a run's events.jsonl.

Usage:
    python tools/diagnose_run.py --run_dir runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def diagnose(run_dir: str) -> Dict[str, Any]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return {"error": f"events.jsonl not found in {run_dir}"}

    total = 0
    shield_vetoes = 0
    mcts_vetoes = 0
    planner_actions = 0
    fallback_actions = 0
    rewinds = 0
    hazards = 0
    successes = 0
    episodes_seen: set = set()
    mcts_queries_max = 0
    mcts_forced_loss_count = 0
    veto_adaptations: List[float] = []
    episode_returns: Dict[str, float] = {}
    episode_hazards: Dict[str, int] = {}
    episode_successes: Dict[str, bool] = {}
    confidence_values: List[float] = []
    danger_values: List[float] = []

    hazard_keys = {
        "hit_wall", "hit_obstacle", "battery_death", "battery_depleted",
        "fell_cliff", "fell_pit", "hit_laser", "collision", "crash",
    }

    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                ev = json.loads(s)
            except Exception:
                continue

            total += 1
            ep_id = str(ev.get("episode_id", ""))
            episodes_seen.add(ep_id)
            reward = _safe_float(ev.get("reward", 0.0))
            episode_returns[ep_id] = episode_returns.get(ep_id, 0.0) + reward

            info = ev.get("info")
            info = info if isinstance(info, dict) else {}

            # Hazard detection
            is_hazard = any(bool(info.get(k, False)) for k in hazard_keys)
            if is_hazard:
                hazards += 1
                episode_hazards[ep_id] = episode_hazards.get(ep_id, 0) + 1

            if bool(info.get("reached_goal", False)):
                successes += 1
                episode_successes[ep_id] = True

            # SafeExecutor diagnostics
            se = info.get("safe_executor")
            se = se if isinstance(se, dict) else {}
            mode = str(se.get("mode", "")).strip().lower()

            if mode == "shield_veto":
                shield_vetoes += 1
            elif mode == "mcts_veto":
                mcts_vetoes += 1
            elif mode == "planner_takeover":
                planner_actions += 1
            elif mode == "fallback":
                fallback_actions += 1
            elif mode == "rewind":
                rewinds += 1

            if se.get("veto_adaptation") is not None:
                veto_adaptations.append(float(_safe_float(se["veto_adaptation"])))
            if se.get("confidence") is not None:
                confidence_values.append(float(_safe_float(se["confidence"])))
            if se.get("danger") is not None:
                danger_values.append(float(_safe_float(se["danger"])))

            # MCTS diagnostics
            mcts_stats = se.get("mcts_stats")
            mcts_stats = mcts_stats if isinstance(mcts_stats, dict) else {}
            mcts_queries_max = max(mcts_queries_max, _safe_int(mcts_stats.get("queries", 0)))
            lq = mcts_stats.get("last_query")
            lq = lq if isinstance(lq, dict) else {}
            if bool(lq.get("forced_loss_detected", False)):
                mcts_forced_loss_count += 1

    num_episodes = len(episodes_seen)
    successful_eps = sum(1 for v in episode_successes.values() if v)
    hazardous_eps = sum(1 for v in episode_hazards.values() if v > 0)
    returns = list(episode_returns.values())
    mean_return = sum(returns) / max(1, len(returns)) if returns else 0.0

    result: Dict[str, Any] = {
        "run_dir": run_dir,
        "total_steps": total,
        "total_episodes": num_episodes,
        "summary": {
            "mean_return": round(mean_return, 3),
            "success_rate": round(successful_eps / max(1, num_episodes), 3),
            "hazard_rate": round(hazardous_eps / max(1, num_episodes), 3),
            "hazard_events_per_1k_steps": round(1000.0 * hazards / max(1, total), 2),
        },
        "safe_executor": {
            "shield_vetoes": shield_vetoes,
            "shield_veto_rate": round(shield_vetoes / max(1, total), 4),
            "mcts_vetoes": mcts_vetoes,
            "planner_actions": planner_actions,
            "fallback_actions": fallback_actions,
            "rewinds": rewinds,
            "mcts_queries_total": mcts_queries_max,
            "mcts_forced_loss_detections": mcts_forced_loss_count,
        },
        "adaptive_veto": {},
        "risk_profile": {},
        "issues": [],
        "recommendations": [],
    }

    # Adaptive veto analysis
    if veto_adaptations:
        result["adaptive_veto"] = {
            "initial": round(veto_adaptations[0], 4),
            "final": round(veto_adaptations[-1], 4),
            "mean": round(sum(veto_adaptations) / len(veto_adaptations), 4),
            "max": round(max(veto_adaptations), 4),
            "samples": len(veto_adaptations),
            "relaxation_improved": bool(veto_adaptations[-1] > veto_adaptations[0]),
        }

    # Risk profile
    if confidence_values:
        result["risk_profile"]["mean_confidence"] = round(
            sum(confidence_values) / len(confidence_values), 4
        )
        result["risk_profile"]["min_confidence"] = round(min(confidence_values), 4)
    if danger_values:
        result["risk_profile"]["mean_danger"] = round(
            sum(danger_values) / len(danger_values), 4
        )
        result["risk_profile"]["max_danger"] = round(max(danger_values), 4)

    # Auto-diagnose issues
    veto_rate = shield_vetoes / max(1, total)
    hazard_rate = hazards / max(1, total)
    success_rate = successful_eps / max(1, num_episodes)

    if veto_rate > 0.40:
        result["issues"].append("HIGH_VETO_RATE: >40% of actions vetoed")
        result["recommendations"].append("Lower min_action_confidence or raise adaptive_veto_relax_start")
    if hazard_rate > 0.25:
        result["issues"].append("HIGH_HAZARD_RATE: >25% of steps hit hazards")
        result["recommendations"].append("Tighten danger_threshold or retrain with more hazard DNA")
    if success_rate < 0.05 and num_episodes > 10:
        result["issues"].append("NEAR_ZERO_SUCCESS: <5% success rate")
        if veto_rate > 0.30:
            result["recommendations"].append("Agent may be frozen by safety — reduce veto sensitivity")
        else:
            result["recommendations"].append("Agent needs more training episodes or better reward shaping")
    if mcts_queries_max > 0 and mcts_forced_loss_count / max(1, mcts_queries_max) > 0.80:
        result["issues"].append("MCTS_FORCED_LOSS_DOMINANT: >80% forced-loss detections")
        result["recommendations"].append("Raise mcts_loss_threshold (closer to 0) or retrain MetaTransformer")
    if veto_adaptations and max(veto_adaptations) < 0.01:
        result["issues"].append("VETO_NOT_RELAXING: adaptation stayed near 0")
        result["recommendations"].append("Check adaptive_veto_failure_guard or warmup_steps config")

    if not result["issues"]:
        result["issues"].append("NO_CRITICAL_ISSUES_DETECTED")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose a training run")
    ap.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    ap.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    args = ap.parse_args()

    result = diagnose(str(args.run_dir))

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"\n{'='*60}")
    print(f"  RUN DIAGNOSTIC: {result['run_dir']}")
    print(f"{'='*60}")
    print(f"  Steps: {result['total_steps']:,}   Episodes: {result['total_episodes']}")
    print()

    s = result["summary"]
    print(f"  📊 PERFORMANCE")
    print(f"     Mean return:        {s['mean_return']:+.3f}")
    print(f"     Success rate:       {s['success_rate']:.1%}")
    print(f"     Hazard rate:        {s['hazard_rate']:.1%}")
    print(f"     Hazard/1k steps:    {s['hazard_events_per_1k_steps']:.1f}")
    print()

    se = result["safe_executor"]
    print(f"  🛡️  SAFETY")
    print(f"     Shield vetoes:      {se['shield_vetoes']:,} ({se['shield_veto_rate']:.1%})")
    print(f"     MCTS vetoes:        {se['mcts_vetoes']:,}")
    print(f"     Planner actions:    {se['planner_actions']:,}")
    print(f"     Fallback actions:   {se['fallback_actions']:,}")
    print(f"     Rewinds:            {se['rewinds']:,}")
    print(f"     MCTS queries:       {se['mcts_queries_total']:,}")
    print(f"     Forced-loss flags:  {se['mcts_forced_loss_detections']:,}")
    print()

    av = result.get("adaptive_veto", {})
    if av:
        print(f"  ⚙️  ADAPTIVE VETO")
        print(f"     Initial:       {av.get('initial', 'n/a')}")
        print(f"     Final:         {av.get('final', 'n/a')}")
        print(f"     Mean:          {av.get('mean', 'n/a')}")
        print(f"     Relaxing:      {'✅' if av.get('relaxation_improved') else '❌'}")
        print()

    rp = result.get("risk_profile", {})
    if rp:
        print(f"  ⚠️  RISK PROFILE")
        if "mean_confidence" in rp:
            print(f"     Mean confidence:  {rp['mean_confidence']:.3f}")
            print(f"     Min confidence:   {rp['min_confidence']:.3f}")
        if "mean_danger" in rp:
            print(f"     Mean danger:      {rp['mean_danger']:.3f}")
            print(f"     Max danger:       {rp['max_danger']:.3f}")
        print()

    issues = result.get("issues", [])
    recs = result.get("recommendations", [])
    print(f"  🔍 DIAGNOSIS")
    for issue in issues:
        icon = "✅" if "NO_CRITICAL" in issue else "🚨"
        print(f"     {icon} {issue}")
    for rec in recs:
        print(f"     💡 {rec}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
