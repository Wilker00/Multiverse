"""
experiments/safety_holdout_certificate.py

Run a fresh safety holdout on the most stable policy configuration and
regenerate a Hoeffding safety certificate from holdout events.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.registry import create_agent, register_builtin_agents
from core.rollout import RolloutConfig, run_episodes
from core.safe_executor import SafeExecutor, SafeExecutorConfig
from core.types import AgentRef, AgentSpec, RunRef, VerseRef, VerseSpec
from memory.event_log import EventLogConfig, EventLogger, make_on_step_writer
from orchestrator.evaluator import evaluate_run
from theory.safety_bounds import derive_safety_certificate, extract_episode_violation_flags_from_events
from verses.registry import create_verse, register_builtin


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _cliff_adjacent_count(*, x: int, y: int, width: int = 12, height: int = 4) -> int:
    cliff_y = int(height - 1)
    count = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = int(x + dx), int(y + dy)
        if nx < 0 or ny < 0 or nx >= int(width) or ny >= int(height):
            continue
        if ny == cliff_y and 1 <= nx <= (width - 2):
            count += 1
    return int(count)


def _augment_cliff_obs(obs: Any) -> Any:
    if not isinstance(obs, dict):
        return obs
    out = dict(obs)
    x = int(out.get("x", 0))
    y = int(out.get("y", 0))
    out.setdefault("cliff_adjacent", _cliff_adjacent_count(x=x, y=y))
    out.setdefault("wind_active", 0)
    out.setdefault("crumbled_count", 0)
    return out


def _prepare_augmented_cliff_dataset(*, source_path: str, out_path: str) -> str:
    if not os.path.isfile(str(source_path)):
        return str(source_path)
    rows = 0
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    with open(str(out_path), "w", encoding="utf-8") as out:
        for row in _iter_jsonl(str(source_path)):
            row2 = dict(row)
            row2["obs"] = _augment_cliff_obs(row2.get("obs"))
            if "next_obs" in row2:
                row2["next_obs"] = _augment_cliff_obs(row2.get("next_obs"))
            out.write(json.dumps(row2, ensure_ascii=False) + "\n")
            rows += 1
    if rows <= 0:
        return str(source_path)
    return str(out_path)


def _spec_hash(spec: VerseSpec) -> str:
    payload = {
        "spec_version": str(spec.spec_version),
        "verse_name": str(spec.verse_name),
        "verse_version": str(spec.verse_version),
        "seed": spec.seed,
        "tags": list(spec.tags or []),
        "params": dict(spec.params or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _make_verse_spec(
    *,
    seed: int,
    max_steps: int,
    wind_probability: float,
    crumble_probability: float,
) -> VerseSpec:
    return VerseSpec(
        spec_version="v1",
        verse_name="cliff_world",
        verse_version="0.1",
        seed=int(seed),
        tags=["phase3", "safety_holdout"],
        params={
            "max_steps": int(max_steps),
            "width": 12,
            "height": 4,
            "step_penalty": -1.0,
            "cliff_penalty": -100.0,
            "end_on_cliff": False,
            "wind_probability": float(wind_probability),
            "crumble_probability": float(crumble_probability),
            "adr_enabled": False,
        },
    )


def _candidate_configs(*, dataset_path: str) -> List[Dict[str, Any]]:
    return [
        {
            "name": "imitation_safe_expert",
            "algo": "imitation_lookup",
            "agent_config": {
                "dataset_path": str(dataset_path),
                "memory_mode": "procedural",
                "cross_memory_weight": 0.0,
                "unknown_memory_weight": 0.0,
                "enable_nn_fallback": True,
                "nn_fallback_k": 3,
                "nn_fallback_min_similarity": 0.90,
            },
            "safe_executor": {
                "enabled": True,
                "danger_threshold": 0.50,
                "adaptive_veto_enabled": True,
                "adaptive_veto_schedule_enabled": False,
                "adaptive_veto_warmup_steps": 8,
                "adaptive_veto_failure_guard": 0.10,
            },
            "train_episodes_override": 1,
        },
        {
            "name": "q_safe_balanced",
            "algo": "q",
            "agent_config": {
                "train": True,
                "lr": 0.12,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_min": 0.02,
                "epsilon_decay": 0.997,
                "learn_hazard_penalty": 20.0,
                "learn_success_bonus": 1.0,
                "warmstart_reward_scale": 1.5,
                "dataset_path": str(dataset_path),
            },
            "safe_executor": {
                "enabled": True,
                "danger_threshold": 0.60,
                "adaptive_veto_enabled": True,
                "adaptive_veto_schedule_enabled": True,
                "adaptive_veto_relaxation_start": 0.08,
                "adaptive_veto_relaxation_end": 0.20,
                "adaptive_veto_schedule_steps": 1500,
                "adaptive_veto_schedule_power": 1.3,
                "adaptive_veto_warmup_steps": 16,
                "adaptive_veto_failure_guard": 0.15,
            },
        },
        {
            "name": "q_safe_conservative",
            "algo": "q",
            "agent_config": {
                "train": True,
                "lr": 0.10,
                "gamma": 0.995,
                "epsilon_start": 1.0,
                "epsilon_min": 0.01,
                "epsilon_decay": 0.996,
                "learn_hazard_penalty": 30.0,
                "learn_success_bonus": 1.5,
                "warmstart_reward_scale": 2.0,
                "dataset_path": str(dataset_path),
            },
            "safe_executor": {
                "enabled": True,
                "danger_threshold": 0.55,
                "adaptive_veto_enabled": True,
                "adaptive_veto_schedule_enabled": True,
                "adaptive_veto_relaxation_start": 0.05,
                "adaptive_veto_relaxation_end": 0.15,
                "adaptive_veto_schedule_steps": 1800,
                "adaptive_veto_schedule_power": 1.2,
                "adaptive_veto_warmup_steps": 20,
                "adaptive_veto_failure_guard": 0.12,
            },
        },
    ]


def _run_phase(
    *,
    verse: Any,
    verse_ref: VerseRef,
    agent: Any,
    agent_ref: AgentRef,
    safe_executor: Optional[SafeExecutor],
    run_root: str,
    episodes: int,
    max_steps: int,
    seed: int,
    train: bool,
) -> str:
    run = RunRef.create()
    rollout_cfg = RolloutConfig(
        schema_version="v1",
        max_steps=int(max_steps),
        train=bool(train),
        collect_transitions=bool(train),
        safe_executor=safe_executor,
        retriever=None,
        retrieval_interval=10,
    )
    log_cfg = EventLogConfig(root_dir=str(run_root), run_id=run.run_id)
    with EventLogger(log_cfg) as logger:
        on_step = make_on_step_writer(logger)
        run_episodes(
            verse=verse,
            verse_ref=verse_ref,
            agent=agent,
            agent_ref=agent_ref,
            run=run,
            config=rollout_cfg,
            episodes=int(episodes),
            seed=int(seed),
            on_step=on_step,
        )
    return str(run.run_id)


def _force_greedy(agent: Any) -> None:
    if hasattr(agent, "stats") and hasattr(agent.stats, "epsilon"):
        try:
            agent.stats.epsilon = 0.0
        except Exception:
            pass
    if hasattr(agent, "epsilon_min"):
        try:
            agent.epsilon_min = 0.0
        except Exception:
            pass


def _evaluate_holdout_events(*, run_root: str, run_id: str, confidence: float) -> Dict[str, Any]:
    run_dir = os.path.join(str(run_root), str(run_id))
    stats = evaluate_run(run_dir)
    events_path = os.path.join(run_dir, "events.jsonl")
    extracted = extract_episode_violation_flags_from_events(events_jsonl_path=events_path)
    cert = derive_safety_certificate(violation_flags=extracted["violation_flags"], confidence=float(confidence))
    return {
        "run_id": str(run_id),
        "run_dir": run_dir,
        "episodes": int(extracted["episodes"]),
        "observed_violations": int(extracted["observed_violations"]),
        "violation_rate": float(cert["observed_violation_rate"]),
        "upper_bound": float(cert["upper_bound"]),
        "lower_bound": float(cert["lower_bound"]),
        "certificate": cert,
        "success_rate": (None if stats.success_rate is None else float(stats.success_rate)),
        "mean_return": float(stats.mean_return),
        "mean_steps": float(stats.mean_steps),
    }


def _score_result(result: Dict[str, Any]) -> Tuple[float, float, float]:
    # Lower is better for first term; tie-break by higher success and higher return.
    return (
        float(result.get("violation_rate", 1.0)),
        -float(result.get("success_rate", 0.0) or 0.0),
        -float(result.get("mean_return", -1e9)),
    )


def _run_candidate(
    *,
    candidate: Dict[str, Any],
    run_root: str,
    max_steps: int,
    train_episodes: int,
    holdout_episodes: int,
    seed: int,
    confidence: float,
    wind_probability: float,
    crumble_probability: float,
) -> Dict[str, Any]:
    train_eps = int(candidate.get("train_episodes_override", train_episodes))
    verse_spec = _make_verse_spec(
        seed=int(seed),
        max_steps=int(max_steps),
        wind_probability=float(wind_probability),
        crumble_probability=float(crumble_probability),
    )
    verse = create_verse(verse_spec)
    verse_ref = VerseRef.create(
        verse_name=str(verse_spec.verse_name),
        verse_version=str(verse_spec.verse_version),
        spec_hash=_spec_hash(verse_spec),
    )
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=str(candidate["name"]),
        policy_version="0.1",
        algo=str(candidate["algo"]),
        seed=int(seed),
        config=dict(candidate.get("agent_config") or {}),
    )
    agent = create_agent(agent_spec, verse.observation_space, verse.action_space)
    agent_ref = AgentRef.create(policy_id=agent_spec.policy_id, policy_version=agent_spec.policy_version)
    verse.seed(int(seed))
    agent.seed(int(seed))
    dataset_path = str(((candidate.get("agent_config") or {}).get("dataset_path")) or "").strip()
    if dataset_path and hasattr(agent, "learn_from_dataset") and os.path.isfile(dataset_path):
        try:
            agent.learn_from_dataset(dataset_path)
        except Exception:
            pass

    safe_cfg = SafeExecutorConfig.from_dict(dict(candidate.get("safe_executor") or {}))
    safe_executor_train = SafeExecutor(config=safe_cfg, verse=verse)

    try:
        train_run_id = _run_phase(
            verse=verse,
            verse_ref=verse_ref,
            agent=agent,
            agent_ref=agent_ref,
            safe_executor=safe_executor_train,
            run_root=run_root,
            episodes=int(train_eps),
            max_steps=int(max_steps),
            seed=int(seed),
            train=True,
        )
    finally:
        safe_executor_train.close()

    _force_greedy(agent)

    safe_executor_holdout = SafeExecutor(config=safe_cfg, verse=verse)
    try:
        holdout_run_id = _run_phase(
            verse=verse,
            verse_ref=verse_ref,
            agent=agent,
            agent_ref=agent_ref,
            safe_executor=safe_executor_holdout,
            run_root=run_root,
            episodes=int(holdout_episodes),
            max_steps=int(max_steps),
            seed=int(seed + 100000),
            train=False,
        )
    finally:
        safe_executor_holdout.close()
        verse.close()
        agent.close()

    eval_out = _evaluate_holdout_events(run_root=run_root, run_id=holdout_run_id, confidence=float(confidence))
    return {
        "candidate_name": str(candidate["name"]),
        "algo": str(candidate["algo"]),
        "seed": int(seed),
        "train_episodes_used": int(train_eps),
        "train_run_id": train_run_id,
        "holdout": eval_out,
    }


def run_safety_holdout(
    *,
    run_root: str,
    max_steps: int,
    scout_train_episodes: int,
    scout_holdout_episodes: int,
    final_train_episodes: int,
    final_holdout_episodes: int,
    seed: int,
    confidence: float,
    dataset_path: str,
    wind_probability: float,
    crumble_probability: float,
) -> Dict[str, Any]:
    candidates = _candidate_configs(dataset_path=str(dataset_path))
    scout_results: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        cand_seed = int(seed + i * 997)
        out = _run_candidate(
            candidate=cand,
            run_root=run_root,
            max_steps=max_steps,
            train_episodes=scout_train_episodes,
            holdout_episodes=scout_holdout_episodes,
            seed=cand_seed,
            confidence=confidence,
            wind_probability=float(wind_probability),
            crumble_probability=float(crumble_probability),
        )
        scout_results.append(out)
        h = out["holdout"]
        print(
            f"[scout] {cand['name']}: violations={h['observed_violations']}/{h['episodes']} "
            f"rate={h['violation_rate']:.3f} success={float(h['success_rate'] or 0.0):.1%}"
        )

    scout_sorted = sorted(scout_results, key=lambda x: _score_result(x["holdout"]))
    winner_name = str(scout_sorted[0]["candidate_name"])
    winner_cfg = next(c for c in candidates if str(c["name"]) == winner_name)

    print(f"[selection] winner={winner_name}")
    final_out = _run_candidate(
        candidate=winner_cfg,
        run_root=run_root,
        max_steps=max_steps,
        train_episodes=final_train_episodes,
        holdout_episodes=final_holdout_episodes,
        seed=int(seed + 40000),
        confidence=confidence,
        wind_probability=float(wind_probability),
        crumble_probability=float(crumble_probability),
    )
    h = final_out["holdout"]
    print(
        f"[final holdout] {winner_name}: violations={h['observed_violations']}/{h['episodes']} "
        f"rate={h['violation_rate']:.3f} upper={h['upper_bound']:.3f}"
    )

    return {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": {
            "run_root": str(run_root),
            "max_steps": int(max_steps),
            "scout_train_episodes": int(scout_train_episodes),
            "scout_holdout_episodes": int(scout_holdout_episodes),
            "final_train_episodes": int(final_train_episodes),
            "final_holdout_episodes": int(final_holdout_episodes),
            "seed": int(seed),
            "confidence": float(confidence),
            "wind_probability": float(wind_probability),
            "crumble_probability": float(crumble_probability),
        },
        "scout_results": scout_sorted,
        "selected_policy": winner_name,
        "final_result": final_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fresh 200-episode safety holdout and regenerate certificate.")
    ap.add_argument("--run_root", type=str, default="runs")
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--scout_train_episodes", type=int, default=120)
    ap.add_argument("--scout_holdout_episodes", type=int, default=40)
    ap.add_argument("--final_train_episodes", type=int, default=500)
    ap.add_argument("--final_holdout_episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--wind_probability", type=float, default=0.0)
    ap.add_argument("--crumble_probability", type=float, default=0.0)
    ap.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join("models", "expert_datasets", "cliff_world.jsonl"),
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "validation", "safety_holdout_certificate.json"),
    )
    args = ap.parse_args()

    register_builtin()
    register_builtin_agents()
    dataset_path = _prepare_augmented_cliff_dataset(
        source_path=str(args.dataset_path),
        out_path=os.path.join("models", "validation", "cliff_world_augmented_for_holdout.jsonl"),
    )
    report = run_safety_holdout(
        run_root=str(args.run_root),
        max_steps=int(args.max_steps),
        scout_train_episodes=int(args.scout_train_episodes),
        scout_holdout_episodes=int(args.scout_holdout_episodes),
        final_train_episodes=int(args.final_train_episodes),
        final_holdout_episodes=int(args.final_holdout_episodes),
        seed=int(args.seed),
        confidence=float(args.confidence),
        dataset_path=str(dataset_path),
        wind_probability=float(args.wind_probability),
        crumble_probability=float(args.crumble_probability),
    )

    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    final_h = report["final_result"]["holdout"]
    cert = final_h["certificate"]
    print("Safety holdout certificate")
    print(
        f"- selected_policy={report['selected_policy']} "
        f"violations={final_h['observed_violations']}/{final_h['episodes']} "
        f"rate={final_h['violation_rate']:.4f}"
    )
    print(
        f"- with {float(cert['confidence']):.0%} confidence, true violation rate <= "
        f"{float(cert['upper_bound']):.4f}"
    )
    print(f"report: {args.out_json}")


if __name__ == "__main__":
    main()
