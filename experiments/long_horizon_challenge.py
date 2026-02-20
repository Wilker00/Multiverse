"""
experiments/long_horizon_challenge.py

Phase 2.3 long-horizon sparse-reward experiment.

Compares:
- no-memory baseline
- memory-enabled agent (on-demand recall)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import random
import shutil
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentRef, AgentSpec, RunRef, VerseRef, VerseSpec, make_step_event
from memory.central_repository import CentralMemoryConfig, ingest_run
from orchestrator.trainer import Trainer
from tools.validation_stats import compute_rate_stats, compute_validation_stats
from verses.long_horizon_challenge import LongHorizonChallengeFactory
from verses.registry import create_verse, list_verses, register_builtin, register_verse


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


def ensure_registered() -> None:
    register_builtin()
    names = set(list_verses().keys())
    if "long_horizon_challenge" not in names:
        register_verse("long_horizon_challenge", LongHorizonChallengeFactory())


def _scripted_action(obs: Dict[str, Any], params: Dict[str, Any]) -> int:
    pos = _safe_int(obs.get("pos", 0), 0)
    has_key = bool(_safe_int(obs.get("has_key", 0), 0) > 0)
    door = bool(_safe_int(obs.get("door_unlocked", 0), 0) > 0)
    cp_idx = _safe_int(obs.get("checkpoint_idx", 0), 0)

    key_pos = _safe_int(params.get("key_pos", 6), 6)
    door_pos = _safe_int(params.get("door_pos", 14), 14)
    checkpoints = [
        _safe_int(params.get("checkpoint_1", 18), 18),
        _safe_int(params.get("checkpoint_2", 22), 22),
        _safe_int(params.get("checkpoint_3", 26), 26),
    ]
    treasure = _safe_int(params.get("treasure_pos", 29), 29)

    if not has_key:
        if pos < key_pos:
            return 1
        if pos > key_pos:
            return 0
        return 2
    if not door:
        if pos < door_pos:
            return 1
        if pos > door_pos:
            return 0
        return 2
    if cp_idx < len(checkpoints):
        target = checkpoints[cp_idx]
        if pos < target:
            return 1
        if pos > target:
            return 0
        return 1
    if pos < treasure:
        return 1
    if pos > treasure:
        return 0
    return 2


def _iter_events(path: str) -> Iterable[Dict[str, Any]]:
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


def _group_by_episode(path: str) -> Dict[str, List[Dict[str, Any]]]:
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for row in _iter_events(path):
        ep = str(row.get("episode_id", ""))
        by_ep.setdefault(ep, []).append(row)
    for rows in by_ep.values():
        rows.sort(key=lambda x: _safe_int(x.get("step_idx", 0), 0))
    return by_ep


def _run_stats(run_dir: str) -> Dict[str, Any]:
    events_path = os.path.join(run_dir, "events.jsonl")
    by_ep = _group_by_episode(events_path)
    returns: List[float] = []
    successes: List[float] = []
    steps: List[int] = []
    trajectories: List[Dict[str, Any]] = []

    for ep, rows in by_ep.items():
        ret = 0.0
        succ = False
        rew: List[float] = []
        acts: List[int] = []
        for row in rows:
            r = _safe_float(row.get("reward", 0.0), 0.0)
            ret += r
            rew.append(r)
            acts.append(_safe_int(row.get("action", 0), 0))
            info = row.get("info") if isinstance(row.get("info"), dict) else {}
            if bool(info.get("reached_goal", False)):
                succ = True
        returns.append(float(ret))
        successes.append(1.0 if succ else 0.0)
        steps.append(int(len(rows)))
        trajectories.append({"episode_id": ep, "rewards": rew, "actions": acts, "success": succ})

    return {
        "episodes": int(len(returns)),
        "mean_return": float(sum(returns) / float(max(1, len(returns)))) if returns else 0.0,
        "success_rate": float(sum(successes) / float(max(1, len(successes)))) if successes else 0.0,
        "mean_steps": float(sum(steps) / float(max(1, len(steps)))) if steps else 0.0,
        "return_stats": compute_validation_stats(returns, min_detectable_delta=1.0) if returns else None,
        "success_stats": compute_rate_stats(successes, min_detectable_delta=0.05) if successes else None,
        "trajectories": trajectories,
    }


def compute_eligibility_traces(
    trajectories: Sequence[Dict[str, Any]],
    *,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[List[float], List[float]]:
    traces: List[float] = []
    returns: List[float] = []
    g = float(gamma)
    lam = float(lambda_)
    for tr in trajectories:
        rewards = [float(x) for x in list(tr.get("rewards") or [])]
        if not rewards:
            continue
        # discounted returns G_t
        gts = [0.0 for _ in rewards]
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running = float(rewards[i]) + g * running
            gts[i] = float(running)
        e = 0.0
        for i in range(len(rewards)):
            e = (g * lam * e) + 1.0
            traces.append(float(e))
            returns.append(float(gts[i]))
    return traces, returns


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n <= 1:
        return 0.0
    mx = sum(float(xs[i]) for i in range(n)) / float(n)
    my = sum(float(ys[i]) for i in range(n)) / float(n)
    num = 0.0
    dx = 0.0
    dy = 0.0
    for i in range(n):
        a = float(xs[i]) - mx
        b = float(ys[i]) - my
        num += a * b
        dx += a * a
        dy += b * b
    den = math.sqrt(dx * dy)
    if den <= 1e-12:
        return 0.0
    return float(num / den)


def estimate_effective_horizon(
    *,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    threshold: float = 0.05,
) -> int:
    base = float(gamma) * float(lambda_)
    if base <= 1e-12 or base >= 0.999999:
        return 1
    # t such that base^t <= threshold
    t = math.log(max(1e-12, float(threshold))) / math.log(base)
    return int(max(1, math.ceil(t)))


def analyze_credit_assignment(trajectories: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Sparse-reward credit assignment is only meaningful on successful episodes.
    succ = [tr for tr in trajectories if bool(tr.get("success", False))]
    if not succ:
        return {
            "credit_assignment_quality": 0.0,
            "effective_horizon": int(estimate_effective_horizon(gamma=0.99, lambda_=0.95, threshold=0.05)),
            "samples": 0,
        }

    traces: List[float] = []
    rets: List[float] = []
    gamma = 0.99
    lam = 0.95
    for tr in succ:
        rewards = [float(x) for x in list(tr.get("rewards") or [])]
        if not rewards:
            continue
        t_len = len(rewards)
        # Backward eligibility proxy for sparse delayed reward.
        # Earlier actions get smaller but non-zero credit; later actions get stronger credit.
        for t in range(t_len):
            d = float(max(0, t_len - 1 - t))
            traces.append(float((gamma * lam) ** d))
            # Discounted future return from t.
            running = 0.0
            p = 1.0
            for k in range(t, t_len):
                running += p * float(rewards[k])
                p *= gamma
            rets.append(float(running))

    corr = _pearson(traces, rets)
    # Empirical horizon: distance from start to first realized non-zero reward
    # in successful sparse-reward trajectories.
    empirical_horizons: List[int] = []
    for tr in succ:
        rewards = [float(x) for x in list(tr.get("rewards") or [])]
        if not rewards:
            continue
        nz = [i for i, r in enumerate(rewards) if abs(float(r)) > 1e-12]
        if not nz:
            continue
        empirical_horizons.append(int(max(0, nz[0])))
    if empirical_horizons:
        eff = int(round(sum(empirical_horizons) / float(len(empirical_horizons))))
    else:
        eff = estimate_effective_horizon(gamma=0.99, lambda_=0.95, threshold=0.05)
    return {
        "credit_assignment_quality": float(corr),
        "effective_horizon": int(eff),
        "samples": int(min(len(traces), len(rets))),
    }


def _write_expert_run(
    *,
    out_run_dir: str,
    episodes: int,
    seed: int,
    verse_params: Dict[str, Any],
    detour_prob: float,
    random_wait_prob: float,
) -> Dict[str, Any]:
    os.makedirs(out_run_dir, exist_ok=True)
    events_path = os.path.join(out_run_dir, "events.jsonl")
    if os.path.isfile(events_path):
        os.remove(events_path)

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name="long_horizon_challenge",
        verse_version="0.1",
        seed=int(seed),
        params=dict(verse_params),
    )
    verse_ref = VerseRef.create("long_horizon_challenge", "0.1", "scripted_expert")
    run_ref = RunRef(run_id=os.path.basename(os.path.normpath(out_run_dir)))
    agent_ref = AgentRef(agent_id="agent_scripted_expert", policy_id="scripted_expert", policy_version="0.1")

    verse = create_verse(verse_spec)
    success_count = 0
    with open(events_path, "w", encoding="utf-8") as f:
        for ep in range(max(1, int(episodes))):
            ep_seed = int(seed) + (ep * 13)
            rng = random.Random(ep_seed + 777)
            verse.seed(ep_seed)
            rr = verse.reset()
            obs = rr.obs if isinstance(rr.obs, dict) else {}
            episode_id = f"expert_ep_{ep:04d}"
            for step_idx in range(_safe_int(verse_params.get("max_steps", 120), 120)):
                action = _scripted_action(obs, verse_params)
                # Inject mild detours to diversify transition signatures and reduce dedupe collapse.
                if rng.random() < float(max(0.0, min(0.9, detour_prob))):
                    pos = _safe_int(obs.get("pos", 0), 0)
                    key_pos = _safe_int(verse_params.get("key_pos", 6), 6)
                    if pos > 0 and pos != key_pos:
                        action = 0 if rng.random() < 0.7 else 2
                elif rng.random() < float(max(0.0, min(0.9, random_wait_prob))):
                    action = 2
                sr = verse.step(action)
                info = sr.info if isinstance(sr.info, dict) else {}
                ev = make_step_event(
                    schema_version="v1",
                    run=run_ref,
                    episode_id=episode_id,
                    step_idx=int(step_idx),
                    agent=agent_ref,
                    verse=verse_ref,
                    obs=obs,
                    action=int(action),
                    reward=float(sr.reward),
                    done=bool(sr.done),
                    truncated=bool(sr.truncated),
                    seed=ep_seed,
                    info=info,
                )
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
                obs = sr.obs if isinstance(sr.obs, dict) else {}
                if bool(info.get("reached_goal", False)):
                    success_count += 1
                if bool(sr.done or sr.truncated):
                    break
    verse.close()
    return {"events_path": events_path, "episodes": int(episodes), "successes": int(success_count)}


def _build_transition_dataset(
    *,
    expert_run_dir: str,
    out_path: str,
) -> Dict[str, Any]:
    events_path = os.path.join(expert_run_dir, "events.jsonl")
    by_ep = _group_by_episode(events_path)
    rows = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, evs in by_ep.items():
            for i, ev in enumerate(evs):
                obs = ev.get("obs")
                next_obs = evs[i + 1].get("obs") if i + 1 < len(evs) else obs
                row = {
                    "obs": obs,
                    "action": _safe_int(ev.get("action", 0), 0),
                    "reward": _safe_float(ev.get("reward", 0.0), 0.0),
                    "next_obs": next_obs,
                    "done": bool(ev.get("done", False)),
                    "truncated": bool(ev.get("truncated", False)),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
    return {"dataset_path": out_path, "rows": int(rows)}


def _run_agent(
    *,
    trainer: Trainer,
    run_root: str,
    policy_id: str,
    episodes: int,
    seed: int,
    memory_enabled: bool,
    memory_root: str,
    verse_params: Dict[str, Any],
    algo: str,
    dataset_path: Optional[str],
    warmstart_reward_scale: float,
) -> str:
    if bool(memory_enabled):
        epsilon_start = 0.20
        epsilon_min = 0.01
        epsilon_decay = 0.998
    else:
        epsilon_start = 1.0
        epsilon_min = 0.05
        epsilon_decay = 0.995

    cfg: Dict[str, Any] = {
        "train": True,
        "epsilon_start": float(epsilon_start),
        "epsilon_min": float(epsilon_min),
        "epsilon_decay": float(epsilon_decay),
        "verse_name": "long_horizon_challenge",
        "rar_enabled": bool(memory_enabled),
        "on_demand_memory_enabled": bool(memory_enabled),
        "on_demand_memory_root": str(memory_root),
        "on_demand_query_budget": 24 if memory_enabled else 0,
        "on_demand_min_interval": 1,
        "recall_enabled": bool(memory_enabled),
        "recall_top_k": 12,
        "recall_vote_weight": 2.00,
        "recall_same_verse_only": True,
        "recall_risk_key": "checkpoint_idx",
        "recall_risk_threshold": 1.0,
        "recall_uncertainty_margin": 1.00,
        "recall_cooldown_steps": 1,
    }
    if not memory_enabled:
        cfg["retrieval_interval"] = 999999
    if dataset_path and os.path.isfile(str(dataset_path)):
        cfg["dataset_path"] = str(dataset_path)
        cfg["warmstart_reward_scale"] = float(max(0.0, warmstart_reward_scale))

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name="long_horizon_challenge",
        verse_version="0.1",
        seed=int(seed),
        params=dict(verse_params),
    )
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=str(policy_id),
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
        config=cfg,
    )
    out = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=int(episodes),
        max_steps=_safe_int(verse_params.get("max_steps", 120), 120),
        seed=int(seed),
        auto_index=False,
        verbose=False,
    )
    run_id = str(out.get("run_id", ""))
    if not run_id:
        raise RuntimeError("trainer returned empty run_id")
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Run long-horizon challenge experiment.")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--runs_root", type=str, default="runs_long_horizon")
    ap.add_argument("--memory_root", type=str, default="central_memory_long_horizon")
    ap.add_argument("--expert_episodes", type=int, default=20)
    ap.add_argument("--expert_detour_prob", type=float, default=0.10)
    ap.add_argument("--expert_wait_prob", type=float, default=0.02)
    ap.add_argument("--baseline_algo", type=str, default="q")
    ap.add_argument("--memory_algo", type=str, default="memory_recall")
    ap.add_argument("--with_memory_warmstart", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--warmstart_reward_scale", type=float, default=1.5)
    ap.add_argument("--reset_memory_root", action="store_true")
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "validation", "long_horizon_challenge.json"),
    )
    args = ap.parse_args()

    ensure_registered()
    if bool(args.reset_memory_root) and os.path.isdir(str(args.memory_root)):
        shutil.rmtree(str(args.memory_root), ignore_errors=True)

    verse_params = {
        "width": 70,
        "max_steps": 200,
        "key_pos": 12,
        "door_pos": 30,
        "checkpoint_1": 42,
        "checkpoint_2": 52,
        "checkpoint_3": 60,
        "treasure_pos": 68,
        "final_reward": 100.0,
        "adr_enabled": False,
    }

    # Build and ingest expert memory.
    expert_run_dir = os.path.join(str(args.runs_root), f"expert_long_horizon_seed_{int(args.seed)}")
    print(f"Building scripted expert traces: {expert_run_dir}")
    expert_meta = _write_expert_run(
        out_run_dir=expert_run_dir,
        episodes=int(args.expert_episodes),
        seed=int(args.seed),
        verse_params=verse_params,
        detour_prob=float(args.expert_detour_prob),
        random_wait_prob=float(args.expert_wait_prob),
    )
    expert_dataset_path = os.path.join(str(args.runs_root), "expert_long_horizon_dataset.jsonl")
    expert_dataset = _build_transition_dataset(
        expert_run_dir=expert_run_dir,
        out_path=expert_dataset_path,
    )
    ingest_stats = ingest_run(
        run_dir=expert_run_dir,
        cfg=CentralMemoryConfig(root_dir=str(args.memory_root)),
    )
    print(f"Ingested expert memory rows: {ingest_stats.added_events}")
    print(f"Built warmstart dataset rows: {expert_dataset['rows']}")

    trainer = Trainer(run_root=str(args.runs_root), schema_version="v1", auto_register_builtin=True)

    print("Running no-memory baseline...")
    no_memory_run_id = _run_agent(
        trainer=trainer,
        run_root=str(args.runs_root),
        policy_id="memory_disabled_agent",
        episodes=int(args.episodes),
        seed=int(args.seed) + 101,
        memory_enabled=False,
        memory_root=str(args.memory_root),
        verse_params=verse_params,
        algo=str(args.baseline_algo),
        dataset_path=None,
        warmstart_reward_scale=0.0,
    )
    print("Running memory-enabled agent...")
    memory_run_id = _run_agent(
        trainer=trainer,
        run_root=str(args.runs_root),
        policy_id="memory_enabled_agent",
        episodes=int(args.episodes),
        seed=int(args.seed) + 202,
        memory_enabled=True,
        memory_root=str(args.memory_root),
        verse_params=verse_params,
        algo=str(args.memory_algo),
        dataset_path=(str(expert_dataset_path) if bool(args.with_memory_warmstart) else None),
        warmstart_reward_scale=float(args.warmstart_reward_scale),
    )

    no_memory_stats = _run_stats(os.path.join(str(args.runs_root), no_memory_run_id))
    memory_stats = _run_stats(os.path.join(str(args.runs_root), memory_run_id))

    no_credit = analyze_credit_assignment(no_memory_stats.get("trajectories") or [])
    mem_credit = analyze_credit_assignment(memory_stats.get("trajectories") or [])

    report = {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": {
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "runs_root": str(args.runs_root),
            "memory_root": str(args.memory_root),
            "expert_episodes": int(args.expert_episodes),
            "expert_detour_prob": float(args.expert_detour_prob),
            "expert_wait_prob": float(args.expert_wait_prob),
            "baseline_algo": str(args.baseline_algo),
            "memory_algo": str(args.memory_algo),
            "with_memory_warmstart": bool(args.with_memory_warmstart),
            "warmstart_reward_scale": float(args.warmstart_reward_scale),
        },
        "expert_memory": {
            "run_dir": expert_run_dir,
            "meta": expert_meta,
            "dataset": expert_dataset,
            "ingest": dataclasses.asdict(ingest_stats),
        },
        "no_memory": {
            "run_id": no_memory_run_id,
            "stats": {k: v for k, v in no_memory_stats.items() if k != "trajectories"},
            "credit_assignment": no_credit,
        },
        "with_memory": {
            "run_id": memory_run_id,
            "stats": {k: v for k, v in memory_stats.items() if k != "trajectories"},
            "credit_assignment": mem_credit,
        },
        "comparison": {
            "success_rate_delta": float(_safe_float(memory_stats.get("success_rate", 0.0), 0.0) - _safe_float(no_memory_stats.get("success_rate", 0.0), 0.0)),
            "mean_return_delta": float(_safe_float(memory_stats.get("mean_return", 0.0), 0.0) - _safe_float(no_memory_stats.get("mean_return", 0.0), 0.0)),
            "credit_quality_delta": float(_safe_float(mem_credit.get("credit_assignment_quality", 0.0), 0.0) - _safe_float(no_credit.get("credit_assignment_quality", 0.0), 0.0)),
            "effective_horizon_with_memory": int(_safe_int(mem_credit.get("effective_horizon", 0), 0)),
        },
    }

    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nLong-horizon summary")
    print(
        f"no-memory success={_safe_float(no_memory_stats.get('success_rate', 0.0), 0.0):.1%} "
        f"with-memory success={_safe_float(memory_stats.get('success_rate', 0.0), 0.0):.1%}"
    )
    print(
        f"credit quality no-memory={_safe_float(no_credit.get('credit_assignment_quality', 0.0), 0.0):.3f} "
        f"with-memory={_safe_float(mem_credit.get('credit_assignment_quality', 0.0), 0.0):.3f}"
    )
    print(f"report: {args.out_json}")


if __name__ == "__main__":
    main()
