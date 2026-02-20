"""
tools/benchmark_parallel_configs.py

Benchmark helper for:
1) comparing single-thread vs legacy-like parallel vs optimized parallel
2) calibrating per-verse min_parallel_episodes thresholds
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.parallel_rollout import DistributedRolloutManager, ParallelRolloutConfig
from core.types import AgentSpec, VerseSpec
from orchestrator.trainer import Trainer


def _parse_verses(raw: str) -> List[str]:
    out = [v.strip().lower() for v in str(raw).split(",") if v.strip()]
    if not out:
        raise ValueError("At least one verse is required.")
    return out


def _parse_episode_grid(raw: str) -> List[int]:
    out: List[int] = []
    for row in str(raw).split(","):
        s = row.strip()
        if not s:
            continue
        try:
            n = max(1, int(s))
        except Exception:
            continue
        out.append(n)
    out = sorted(set(out))
    if not out:
        raise ValueError("episode_grid must contain at least one positive integer")
    return out


def _mk_verse_spec(*, verse_name: str, seed: int, max_steps: int) -> VerseSpec:
    return VerseSpec(
        spec_version="v1",
        verse_name=str(verse_name),
        verse_version="0.1",
        seed=int(seed),
        params={"max_steps": int(max_steps)},
    )


def _mk_agent_spec(*, algo: str, seed: int) -> AgentSpec:
    return AgentSpec(
        spec_version="v1",
        policy_id=f"parallel_bench_{algo}",
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
    )


def _bench_single_thread(
    *,
    verses: List[str],
    repeats: int,
    episodes: int,
    max_steps: int,
    seed: int,
    run_root: str,
    algo: str,
) -> Dict[str, Any]:
    trainer = Trainer(run_root=run_root)
    agent_spec = _mk_agent_spec(algo=algo, seed=seed)
    started = time.perf_counter()
    total_steps = 0
    total_return = 0.0

    for rep in range(int(repeats)):
        for i, verse_name in enumerate(verses):
            verse_seed = int(seed + (rep * 1000) + i)
            verse_spec = _mk_verse_spec(verse_name=verse_name, seed=verse_seed, max_steps=max_steps)
            out = trainer.run(
                verse_spec=verse_spec,
                agent_spec=agent_spec,
                episodes=int(episodes),
                max_steps=int(max_steps),
                seed=verse_seed,
                run_id=f"single_{rep}_{verse_name}",
                auto_index=False,
                verbose=False,
            )
            total_steps += int(out.get("total_steps", 0))
            total_return += float(out.get("total_return", 0.0))

    elapsed = time.perf_counter() - started
    return {
        "profile": "single_thread",
        "elapsed_s": float(elapsed),
        "total_steps": int(total_steps),
        "total_return": float(total_return),
        "throughput_steps_per_s": float(total_steps / max(elapsed, 1e-9)),
    }


def _bench_single_case(
    *,
    trainer: Trainer,
    verse_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    algo: str,
    repeats: int,
) -> Dict[str, Any]:
    agent_spec = _mk_agent_spec(algo=algo, seed=seed)
    started = time.perf_counter()
    total_steps = 0
    total_return = 0.0
    for rep in range(max(1, int(repeats))):
        verse_seed = int(seed + rep * 1000)
        verse_spec = _mk_verse_spec(verse_name=str(verse_name), seed=verse_seed, max_steps=max_steps)
        out = trainer.run(
            verse_spec=verse_spec,
            agent_spec=agent_spec,
            episodes=int(episodes),
            max_steps=int(max_steps),
            seed=verse_seed,
            run_id=f"cal_single_{verse_name}_{episodes}_{rep}",
            auto_index=False,
            verbose=False,
        )
        total_steps += int(out.get("total_steps", 0))
        total_return += float(out.get("total_return", 0.0))
    elapsed = time.perf_counter() - started
    return {
        "elapsed_s": float(elapsed),
        "total_steps": int(total_steps),
        "total_return": float(total_return),
        "throughput_steps_per_s": float(total_steps / max(elapsed, 1e-9)),
    }


def _bench_parallel_profile(
    *,
    name: str,
    cfg: ParallelRolloutConfig,
    verses: List[str],
    repeats: int,
    episodes: int,
    max_steps: int,
    seed: int,
    algo: str,
) -> Dict[str, Any]:
    mgr = DistributedRolloutManager(cfg)
    agent_spec = _mk_agent_spec(algo=algo, seed=seed)
    started = time.perf_counter()
    total_steps = 0
    total_return = 0.0

    try:
        for rep in range(int(repeats)):
            for i, verse_name in enumerate(verses):
                verse_seed = int(seed + (rep * 1000) + i)
                verse_spec = _mk_verse_spec(verse_name=verse_name, seed=verse_seed, max_steps=max_steps)
                out = mgr.run_parallel_rollouts(
                    agent_spec=agent_spec,
                    verse_spec=verse_spec,
                    total_episodes=int(episodes),
                    max_steps=int(max_steps),
                    seed=verse_seed,
                    run_prefix=f"{name}_{rep}_{verse_name}",
                )
                total_steps += int(out.get("total_steps", 0))
                total_return += float(out.get("total_return", 0.0))
    finally:
        mgr.shutdown()

    elapsed = time.perf_counter() - started
    return {
        "profile": str(name),
        "elapsed_s": float(elapsed),
        "total_steps": int(total_steps),
        "total_return": float(total_return),
        "throughput_steps_per_s": float(total_steps / max(elapsed, 1e-9)),
        "parallel_config": asdict(cfg),
    }


def _bench_parallel_case(
    *,
    mgr: DistributedRolloutManager,
    verse_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    algo: str,
    repeats: int,
    run_prefix: str,
) -> Dict[str, Any]:
    agent_spec = _mk_agent_spec(algo=algo, seed=seed)
    started = time.perf_counter()
    total_steps = 0
    total_return = 0.0
    for rep in range(max(1, int(repeats))):
        verse_seed = int(seed + rep * 1000)
        verse_spec = _mk_verse_spec(verse_name=str(verse_name), seed=verse_seed, max_steps=max_steps)
        out = mgr.run_parallel_rollouts(
            agent_spec=agent_spec,
            verse_spec=verse_spec,
            total_episodes=int(episodes),
            max_steps=int(max_steps),
            seed=verse_seed,
            run_prefix=f"{run_prefix}_{verse_name}_{episodes}_{rep}",
        )
        total_steps += int(out.get("total_steps", 0))
        total_return += float(out.get("total_return", 0.0))
    elapsed = time.perf_counter() - started
    return {
        "elapsed_s": float(elapsed),
        "total_steps": int(total_steps),
        "total_return": float(total_return),
        "throughput_steps_per_s": float(total_steps / max(elapsed, 1e-9)),
    }


def _recommend_threshold_for_rows(
    *,
    rows: List[Dict[str, Any]],
    min_speedup: float,
    fallback_default: int,
) -> int:
    for r in rows:
        speedup = float(r.get("speedup_single_over_parallel", 0.0))
        episodes = int(r.get("episodes", fallback_default))
        if speedup >= float(min_speedup):
            return max(1, episodes)
    return max(1, int(fallback_default))


def _calibrate_per_verse_thresholds(
    *,
    verses: List[str],
    episode_grid: List[int],
    repeats: int,
    max_steps: int,
    seed: int,
    algo: str,
    run_root: str,
    workers: int,
    min_speedup: float,
    fallback_default: int,
) -> Tuple[Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    os.makedirs(run_root, exist_ok=True)
    single_root = os.path.join(run_root, "single")
    parallel_root = os.path.join(run_root, "parallel")
    os.makedirs(single_root, exist_ok=True)
    os.makedirs(parallel_root, exist_ok=True)

    trainer = Trainer(run_root=single_root)
    mgr = DistributedRolloutManager(
        ParallelRolloutConfig(
            num_workers=max(1, int(workers)),
            run_root=parallel_root,
            reuse_process_pool=True,
            min_parallel_episodes=1,
            per_verse_min_parallel_episodes={},
            auto_tune_profile_path="",
            worker_auto_index=False,
            worker_verbose=False,
            annotate_parallel_worker_id=False,
        )
    )

    recommended: Dict[str, int] = {}
    details: Dict[str, List[Dict[str, Any]]] = {}
    try:
        for i, verse in enumerate(verses):
            rows: List[Dict[str, Any]] = []
            verse_seed_base = int(seed + i * 100_000)
            for j, eps in enumerate(episode_grid):
                bench_seed = int(verse_seed_base + j * 1000)
                single = _bench_single_case(
                    trainer=trainer,
                    verse_name=str(verse),
                    episodes=int(eps),
                    max_steps=int(max_steps),
                    seed=bench_seed,
                    algo=str(algo),
                    repeats=max(1, int(repeats)),
                )
                parallel = _bench_parallel_case(
                    mgr=mgr,
                    verse_name=str(verse),
                    episodes=int(eps),
                    max_steps=int(max_steps),
                    seed=bench_seed,
                    algo=str(algo),
                    repeats=max(1, int(repeats)),
                    run_prefix="cal_parallel",
                )
                single_elapsed = float(single["elapsed_s"])
                parallel_elapsed = float(parallel["elapsed_s"])
                row = {
                    "episodes": int(eps),
                    "single_elapsed_s": float(single_elapsed),
                    "parallel_elapsed_s": float(parallel_elapsed),
                    "speedup_single_over_parallel": float(single_elapsed / max(parallel_elapsed, 1e-9)),
                    "single_throughput_steps_per_s": float(single["throughput_steps_per_s"]),
                    "parallel_throughput_steps_per_s": float(parallel["throughput_steps_per_s"]),
                }
                rows.append(row)
            details[str(verse)] = rows
            recommended[str(verse)] = int(
                _recommend_threshold_for_rows(
                    rows=rows,
                    min_speedup=float(min_speedup),
                    fallback_default=int(fallback_default),
                )
            )
    finally:
        mgr.shutdown()

    return recommended, details


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verses", type=str, default="chess_world")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--episode_grid", type=str, default="8,16,32,64,96,128")
    ap.add_argument("--threshold_min_speedup", type=float, default=1.0)
    ap.add_argument("--thresholds_out_json", type=str, default=os.path.join("models", "tuning", "parallel_thresholds.json"))
    ap.add_argument("--skip_threshold_calibration", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--algo", type=str, default="random")
    ap.add_argument("--run_root", type=str, default="runs_parallel_bench")
    ap.add_argument("--out_json", type=str, default=os.path.join("models", "tuning", "parallel_config_benchmark.json"))
    ap.add_argument("--clean_run_root", action="store_true")
    args = ap.parse_args()

    verses = _parse_verses(args.verses)
    episode_grid = _parse_episode_grid(args.episode_grid)
    run_root = str(args.run_root)
    if bool(args.clean_run_root) and os.path.isdir(run_root):
        shutil.rmtree(run_root, ignore_errors=True)
    os.makedirs(run_root, exist_ok=True)

    single_root = os.path.join(run_root, "single")
    legacy_root = os.path.join(run_root, "legacy_parallel")
    optimized_root = os.path.join(run_root, "optimized_parallel")
    for p in (single_root, legacy_root, optimized_root):
        os.makedirs(p, exist_ok=True)

    single = _bench_single_thread(
        verses=verses,
        repeats=max(1, int(args.repeats)),
        episodes=max(1, int(args.episodes)),
        max_steps=max(1, int(args.max_steps)),
        seed=int(args.seed),
        run_root=single_root,
        algo=str(args.algo),
    )

    legacy_cfg = ParallelRolloutConfig(
        num_workers=max(1, int(args.workers)),
        run_root=legacy_root,
        reuse_process_pool=False,
        min_parallel_episodes=1,
        worker_auto_index=True,
        worker_verbose=True,
        annotate_parallel_worker_id=True,
        auto_tune_profile_path="",
    )
    optimized_cfg = ParallelRolloutConfig(
        num_workers=max(1, int(args.workers)),
        run_root=optimized_root,
        reuse_process_pool=True,
        min_parallel_episodes=64,
        worker_auto_index=False,
        worker_verbose=False,
        annotate_parallel_worker_id=False,
        auto_tune_profile_path="",
    )

    legacy = _bench_parallel_profile(
        name="legacy_parallel",
        cfg=legacy_cfg,
        verses=verses,
        repeats=max(1, int(args.repeats)),
        episodes=max(1, int(args.episodes)),
        max_steps=max(1, int(args.max_steps)),
        seed=int(args.seed),
        algo=str(args.algo),
    )
    optimized = _bench_parallel_profile(
        name="optimized_parallel",
        cfg=optimized_cfg,
        verses=verses,
        repeats=max(1, int(args.repeats)),
        episodes=max(1, int(args.episodes)),
        max_steps=max(1, int(args.max_steps)),
        seed=int(args.seed),
        algo=str(args.algo),
    )

    single_elapsed = float(single["elapsed_s"])
    legacy_elapsed = float(legacy["elapsed_s"])
    optimized_elapsed = float(optimized["elapsed_s"])
    recommended_thresholds: Dict[str, int] = {}
    calibration_rows: Dict[str, List[Dict[str, Any]]] = {}
    if not bool(args.skip_threshold_calibration):
        recommended_thresholds, calibration_rows = _calibrate_per_verse_thresholds(
            verses=verses,
            episode_grid=episode_grid,
            repeats=max(1, int(args.repeats)),
            max_steps=max(1, int(args.max_steps)),
            seed=int(args.seed + 10_000_000),
            algo=str(args.algo),
            run_root=os.path.join(run_root, "threshold_calibration"),
            workers=max(1, int(args.workers)),
            min_speedup=float(args.threshold_min_speedup),
            fallback_default=max(1, int(optimized_cfg.min_parallel_episodes)),
        )

    summary = {
        "verses": verses,
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "workers": int(args.workers),
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "algo": str(args.algo),
        "single_thread": single,
        "legacy_parallel": legacy,
        "optimized_parallel": optimized,
        "episode_grid": episode_grid,
        "speedup_vs_single": {
            "legacy_parallel": float(single_elapsed / max(legacy_elapsed, 1e-9)),
            "optimized_parallel": float(single_elapsed / max(optimized_elapsed, 1e-9)),
        },
        "optimized_vs_legacy_speedup": float(legacy_elapsed / max(optimized_elapsed, 1e-9)),
        "recommended_min_parallel_episodes_by_verse": recommended_thresholds,
        "threshold_calibration": {
            "enabled": bool(not args.skip_threshold_calibration),
            "threshold_min_speedup": float(args.threshold_min_speedup),
            "rows_by_verse": calibration_rows,
        },
    }

    out_json = str(args.out_json)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    thresholds_out = str(args.thresholds_out_json)
    if recommended_thresholds:
        payload = {
            "version": "v1",
            "created_at_ms": int(time.time() * 1000),
            "source_report": out_json,
            "default_min_parallel_episodes": int(optimized_cfg.min_parallel_episodes),
            "per_verse_min_parallel_episodes": {str(k): int(v) for k, v in sorted(recommended_thresholds.items())},
        }
        os.makedirs(os.path.dirname(thresholds_out) or ".", exist_ok=True)
        with open(thresholds_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print("parallel_config_benchmark")
    print(f"out_json={out_json}")
    if recommended_thresholds:
        print(f"thresholds_out_json={thresholds_out}")
    print(f"single_thread_s={single_elapsed:.3f}")
    print(f"legacy_parallel_s={legacy_elapsed:.3f}")
    print(f"optimized_parallel_s={optimized_elapsed:.3f}")
    print(f"speedup_single_over_legacy={summary['speedup_vs_single']['legacy_parallel']:.3f}x")
    print(f"speedup_single_over_optimized={summary['speedup_vs_single']['optimized_parallel']:.3f}x")
    print(f"optimized_vs_legacy_speedup={summary['optimized_vs_legacy_speedup']:.3f}x")


if __name__ == "__main__":
    main()
