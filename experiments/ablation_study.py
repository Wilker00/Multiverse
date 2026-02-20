"""
experiments/ablation_study.py

Phase-1 ablation runner:
1) Semantic Bridge disabled
2) SafeExecutor disabled
3) Memory retrieval disabled
4) Golden DNA disabled

Default experiment compares source->target transfer quality on:
- chess_world
- labyrinth_world
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from orchestrator.trainer import Trainer
from tools.validation_stats import compute_rate_stats, compute_validation_stats


ABLATION_CONFIGS: Dict[str, Dict[str, bool]] = {
    "baseline": {
        "semantic_bridge": True,
        "safe_executor": True,
        "memory_retrieval": True,
        "golden_dna": True,
    },
    "no_semantic_bridge": {
        "semantic_bridge": False,
        "safe_executor": True,
        "memory_retrieval": True,
        "golden_dna": True,
    },
    "no_safety": {
        "semantic_bridge": True,
        "safe_executor": False,
        "memory_retrieval": True,
        "golden_dna": True,
    },
    "no_memory": {
        "semantic_bridge": True,
        "safe_executor": True,
        "memory_retrieval": False,
        "golden_dna": True,
    },
    "no_generational": {
        "semantic_bridge": True,
        "safe_executor": True,
        "memory_retrieval": True,
        "golden_dna": False,
    },
}

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


def _iter_events(run_dir: str) -> Sequence[Dict[str, Any]]:
    events_path = os.path.join(run_dir, "events.jsonl")
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def summarize_run(run_dir: str, *, alpha: float, power: float, min_detectable_delta: float) -> Dict[str, Any]:
    by_episode: Dict[str, List[Dict[str, Any]]] = {}
    for ev in _iter_events(run_dir):
        ep = str(ev.get("episode_id", ""))
        by_episode.setdefault(ep, []).append(ev)

    returns: List[float] = []
    successes: List[float] = []
    safety_episode_flags: List[float] = []
    safety_events = 0
    mean_steps = 0.0

    for _, events in by_episode.items():
        events.sort(key=lambda x: _safe_int(x.get("step_idx", 0), 0))
        ep_return = 0.0
        ep_steps = 0
        ep_success = False
        ep_safety = False

        for ev in events:
            ep_return += _safe_float(ev.get("reward", 0.0), 0.0)
            ep_steps += 1
            info = ev.get("info") if isinstance(ev.get("info"), dict) else {}
            if _is_success(info):
                ep_success = True
            if _is_safety_violation(info):
                ep_safety = True
                safety_events += 1

        returns.append(float(ep_return))
        successes.append(1.0 if ep_success else 0.0)
        safety_episode_flags.append(1.0 if ep_safety else 0.0)
        mean_steps += float(ep_steps)

    n = max(1, len(returns))
    mean_steps = float(mean_steps / float(n))
    reward_stats = compute_validation_stats(
        returns,
        alpha=float(alpha),
        power=float(power),
        min_detectable_delta=float(min_detectable_delta),
    )
    success_stats = compute_rate_stats(
        successes,
        alpha=float(alpha),
        power=float(power),
        min_detectable_delta=float(min_detectable_delta),
    )
    safety_stats = compute_rate_stats(
        safety_episode_flags,
        alpha=float(alpha),
        power=float(power),
        min_detectable_delta=float(min_detectable_delta),
    )
    return {
        "episodes": int(len(returns)),
        "mean_return": float(reward_stats.get("mean", 0.0)),
        "mean_steps": float(mean_steps),
        "success_rate": float(success_stats.get("rate", 0.0)),
        "safety_violation_rate": float(safety_stats.get("rate", 0.0)),
        "safety_violation_events": int(safety_events),
        "reward_stats": reward_stats,
        "success_stats": success_stats,
        "safety_stats": safety_stats,
    }


def _default_params(verse_name: str, max_steps: int) -> Dict[str, Any]:
    v = str(verse_name).strip().lower()
    params: Dict[str, Any] = {"max_steps": int(max_steps), "adr_enabled": False}
    if v == "labyrinth_world":
        params.update(
            {
                "width": 15,
                "height": 11,
                "battery_capacity": 80,
                "battery_drain": 1,
                "action_noise": 0.08,
            }
        )
    elif v == "chess_world":
        params.update({"max_steps": int(max(40, max_steps))})
    return params


def _build_agent_config(
    *,
    verse_name: str,
    flags: Dict[str, bool],
    dataset_path: Optional[str],
    safe_danger_threshold: float,
    safe_min_action_confidence: float,
    safe_severe_reward_threshold: float,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "train": True,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "verse_name": str(verse_name),
        # Retrieval-augmented rollouts (Semantic Bridge path in rollout)
        "rar_enabled": bool(flags.get("semantic_bridge", True)),
        # On-demand memory path used by memory_recall/planner_recall agents
        "on_demand_memory_enabled": bool(flags.get("memory_retrieval", True)),
        "recall_enabled": bool(flags.get("memory_retrieval", True)),
    }
    if not bool(flags.get("memory_retrieval", True)):
        cfg["on_demand_query_budget"] = 0
        cfg["retrieval_interval"] = 999999

    if bool(flags.get("safe_executor", True)):
        cfg["safe_executor"] = {
            "enabled": True,
            "danger_threshold": float(safe_danger_threshold),
            "min_action_confidence": float(safe_min_action_confidence),
            "severe_reward_threshold": float(safe_severe_reward_threshold),
            "adaptive_veto_enabled": True,
            "adaptive_veto_schedule_enabled": True,
        }

    if bool(flags.get("golden_dna", True)) and dataset_path and os.path.isfile(str(dataset_path)):
        cfg["dataset_path"] = str(dataset_path)
        cfg["warmstart_reward_scale"] = 0.5

    return cfg


def compute_transfer_score(source_perf: Dict[str, Any], target_perf: Dict[str, Any]) -> float:
    """
    Harmonic mean of source and target success rates.
    """
    src = max(0.0, min(1.0, _safe_float(source_perf.get("success_rate", 0.0), 0.0)))
    tgt = max(0.0, min(1.0, _safe_float(target_perf.get("success_rate", 0.0), 0.0)))
    if (src + tgt) <= 1e-12:
        return 0.0
    return float(2.0 * src * tgt / (src + tgt))


def _retention_ratio(source_perf: Dict[str, Any], target_perf: Dict[str, Any]) -> float:
    src = _safe_float(source_perf.get("success_rate", 0.0), 0.0)
    tgt = _safe_float(target_perf.get("success_rate", 0.0), 0.0)
    if src <= 1e-12:
        return 0.0 if tgt <= 1e-12 else 1.0
    return float(tgt / src)


def run_verse(
    *,
    trainer: Trainer,
    run_root: str,
    verse_name: str,
    algo: str,
    episodes: int,
    max_steps: int,
    seed: int,
    flags: Dict[str, bool],
    dataset_path: Optional[str],
    safe_danger_threshold: float,
    safe_min_action_confidence: float,
    safe_severe_reward_threshold: float,
    alpha: float,
    power: float,
    min_detectable_delta: float,
    tag: str,
) -> Dict[str, Any]:
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=str(verse_name),
        verse_version="0.1",
        seed=int(seed),
        params=_default_params(verse_name, int(max_steps)),
        tags=["ablation", str(tag)],
    )
    agent_cfg = _build_agent_config(
        verse_name=str(verse_name),
        flags=flags,
        dataset_path=dataset_path,
        safe_danger_threshold=float(safe_danger_threshold),
        safe_min_action_confidence=float(safe_min_action_confidence),
        safe_severe_reward_threshold=float(safe_severe_reward_threshold),
    )
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=f"ablation_{str(tag)}_{str(verse_name)}",
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
        tags=["ablation", str(tag)],
        config=agent_cfg,
    )

    run = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
        auto_index=False,
        verbose=False,
    )
    run_id = str(run.get("run_id", ""))
    run_dir = os.path.join(run_root, run_id)
    summary = summarize_run(
        run_dir,
        alpha=float(alpha),
        power=float(power),
        min_detectable_delta=float(min_detectable_delta),
    )
    summary["run_id"] = run_id
    summary["run_dir"] = run_dir
    summary["agent_config"] = agent_cfg
    return summary


def run_ablation_study(args: argparse.Namespace) -> Dict[str, Any]:
    trainer = Trainer(run_root=str(args.run_root), schema_version="v1", auto_register_builtin=True)
    results: Dict[str, Any] = {}

    for config_name, flags in ABLATION_CONFIGS.items():
        print(f"\nTesting: {config_name}")
        cfg_seed = int(args.seed) + int(abs(hash(config_name)) % 10000)
        row: Dict[str, Any] = {"flags": dict(flags)}
        try:
            chess_perf = run_verse(
                trainer=trainer,
                run_root=str(args.run_root),
                verse_name=str(args.source_verse),
                algo=str(args.algo),
                episodes=int(args.episodes),
                max_steps=int(args.max_steps_source),
                seed=int(cfg_seed),
                flags=flags,
                dataset_path=(str(args.dataset_path) if args.dataset_path else None),
                safe_danger_threshold=float(args.safe_danger_threshold),
                safe_min_action_confidence=float(args.safe_min_action_confidence),
                safe_severe_reward_threshold=float(args.safe_severe_reward_threshold),
                alpha=float(args.alpha),
                power=float(args.power),
                min_detectable_delta=float(args.min_detectable_delta),
                tag=f"{config_name}_source",
            )
            target_perf = run_verse(
                trainer=trainer,
                run_root=str(args.run_root),
                verse_name=str(args.target_verse),
                algo=str(args.algo),
                episodes=int(args.episodes),
                max_steps=int(args.max_steps_target),
                seed=int(cfg_seed) + 777,
                flags=flags,
                dataset_path=(str(args.dataset_path) if args.dataset_path else None),
                safe_danger_threshold=float(args.safe_danger_threshold),
                safe_min_action_confidence=float(args.safe_min_action_confidence),
                safe_severe_reward_threshold=float(args.safe_severe_reward_threshold),
                alpha=float(args.alpha),
                power=float(args.power),
                min_detectable_delta=float(args.min_detectable_delta),
                tag=f"{config_name}_target",
            )
            transfer_quality = compute_transfer_score(chess_perf, target_perf)
            retention_ratio = _retention_ratio(chess_perf, target_perf)
            row.update(
                {
                    "source_verse": str(args.source_verse),
                    "target_verse": str(args.target_verse),
                    "source": chess_perf,
                    "target": target_perf,
                    "source_success": float(chess_perf.get("success_rate", 0.0)),
                    "target_success": float(target_perf.get("success_rate", 0.0)),
                    "transfer_quality": float(transfer_quality),
                    "retention_ratio": float(retention_ratio),
                    "golden_dna_applied": bool(flags.get("golden_dna", True) and args.dataset_path and os.path.isfile(str(args.dataset_path))),
                }
            )
            print(
                f"  source_success={float(row['source_success']):.1%} "
                f"target_success={float(row['target_success']):.1%} "
                f"transfer_quality={float(row['transfer_quality']):.3f}"
            )
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  error: {row['error']}")
        results[config_name] = row

    baseline = results.get("baseline", {})
    baseline_tq = _safe_float(baseline.get("transfer_quality", 0.0), 0.0)
    comparisons: Dict[str, Any] = {}
    for config_name, row in results.items():
        if config_name == "baseline":
            continue
        if not isinstance(row, dict) or "error" in row:
            comparisons[config_name] = {"delta_transfer_quality": None, "classification": "error"}
            continue
        delta = _safe_float(row.get("transfer_quality", 0.0), 0.0) - float(baseline_tq)
        if delta < -0.2:
            cls = "critical_degradation"
        elif delta < 0:
            cls = "degraded"
        else:
            cls = "unchanged_or_improved"
        comparisons[config_name] = {
            "delta_transfer_quality": float(delta),
            "classification": cls,
        }

    return {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "algo": str(args.algo),
        "episodes": int(args.episodes),
        "source_verse": str(args.source_verse),
        "target_verse": str(args.target_verse),
        "results": results,
        "comparisons_vs_baseline": comparisons,
        "notes": {
            "golden_dna_dataset_path": (str(args.dataset_path) if args.dataset_path else None),
            "golden_dna_dataset_exists": bool(args.dataset_path and os.path.isfile(str(args.dataset_path))),
            "transfer_quality_definition": "harmonic_mean(source_success_rate, target_success_rate)",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run phase-1 ablation study.")
    ap.add_argument("--algo", type=str, default="memory_recall")
    ap.add_argument("--source_verse", type=str, default="chess_world")
    ap.add_argument("--target_verse", type=str, default="labyrinth_world")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max_steps_source", type=int, default=80)
    ap.add_argument("--max_steps_target", type=int, default=160)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--run_root", type=str, default="runs_ablation")
    ap.add_argument("--dataset_path", type=str, default=None, help="Optional Golden DNA dataset (.jsonl)")
    ap.add_argument("--safe_danger_threshold", type=float, default=0.60)
    ap.add_argument("--safe_min_action_confidence", type=float, default=0.08)
    ap.add_argument("--safe_severe_reward_threshold", type=float, default=-50.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--min_detectable_delta", type=float, default=0.10)
    ap.add_argument("--output_json", type=str, default=os.path.join("models", "validation", "ablation_study.json"))
    args = ap.parse_args()

    report = run_ablation_study(args)

    os.makedirs(os.path.dirname(str(args.output_json)) or ".", exist_ok=True)
    with open(str(args.output_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nAblation deltas vs baseline:")
    comps = report.get("comparisons_vs_baseline", {})
    if isinstance(comps, dict):
        for name, row in comps.items():
            if not isinstance(row, dict):
                continue
            d = row.get("delta_transfer_quality")
            cls = str(row.get("classification", ""))
            if d is None:
                print(f"  {name}: Delta = n/a ({cls})")
            else:
                print(f"  {name}: Delta = {float(d):+.3f} ({cls})")
    print(f"report: {args.output_json}")


if __name__ == "__main__":
    main()
