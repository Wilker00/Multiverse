"""
core/parallel_rollout.py

Local-first parallel rollout manager for Multiverse.

Design goals:
- Use current project APIs (Trainer + VerseSpec + AgentSpec).
- Work on one machine with multiprocessing by default.
- Optional Ray backend when installed.
- Produce normal run folders under runs/ plus one aggregate parallel run folder.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_THIS_DIR)
    # Avoid stdlib shadowing by `core/types.py` when executing as script.
    if sys.path and os.path.abspath(sys.path[0]) == _THIS_DIR:
        sys.path.pop(0)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import concurrent.futures
import dataclasses
import json
import argparse
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures.process import BrokenProcessPool

from core.adversary_sampler import AdversaryBundle, AdversarySampler
from core.types import AgentSpec, VerseSpec

try:
    import ray  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ray = None


def _episodes_split(total: int, workers: int) -> List[int]:
    workers = max(1, int(workers))
    total = max(0, int(total))
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _normalize_verse_name(name: str) -> str:
    return str(name).strip().lower()


def _normalize_threshold_map(raw: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        verse = _normalize_verse_name(str(k))
        if not verse:
            continue
        try:
            val = max(1, int(v))
        except Exception:
            continue
        out[verse] = val
    return out


def _load_threshold_map_from_profile(path: str) -> Dict[str, int]:
    p = str(path).strip()
    if not p or (not os.path.isfile(p)):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    direct = _normalize_threshold_map(payload.get("per_verse_min_parallel_episodes"))
    if direct:
        return direct
    recommended = _normalize_threshold_map(payload.get("recommended_min_parallel_episodes_by_verse"))
    if recommended:
        return recommended
    tuning = payload.get("tuning")
    if isinstance(tuning, dict):
        nested = _normalize_threshold_map(tuning.get("per_verse_min_parallel_episodes"))
        if nested:
            return nested
    return {}


def _worker_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    from orchestrator.trainer import Trainer

    worker_id = int(payload["worker_id"])
    episodes = int(payload["episodes"])
    if episodes <= 0:
        return {"worker_id": worker_id, "skipped": True}

    verse_spec = VerseSpec(**payload["verse_spec"])
    agent_spec = AgentSpec(**payload["agent_spec"])
    run_root = str(payload["run_root"])
    max_steps = int(payload["max_steps"])
    seed = int(payload["seed"])
    run_id = str(payload["run_id"])
    schema_version = str(payload.get("schema_version", "v1"))
    auto_index = bool(payload.get("auto_index", True))
    verbose = bool(payload.get("verbose", True))

    trainer = Trainer(run_root=run_root, schema_version=schema_version, auto_register_builtin=True)
    started = time.time()
    out = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        run_id=run_id,
        auto_index=auto_index,
        verbose=verbose,
    )
    return {
        "worker_id": worker_id,
        "skipped": False,
        "run_id": str(out["run_id"]),
        "total_return": float(out["total_return"]),
        "total_steps": int(out["total_steps"]),
        "elapsed_s": float(time.time() - started),
    }


if ray is not None:
    @ray.remote
    def _ray_worker_run(payload: Dict[str, Any]) -> Dict[str, Any]:
        return _worker_run(payload)


@dataclass
class ParallelRolloutConfig:
    num_workers: int = 4
    use_ray: bool = False
    run_root: str = "runs"
    schema_version: str = "v1"
    max_worker_timeout_s: int = 60 * 60
    self_play_enabled: bool = False
    self_play_adversary_source: str = "recent_failures"
    self_play_mix_ratio: float = 0.25
    self_play_memory_dir: str = "central_memory"
    self_play_top_k: int = 300
    reuse_process_pool: bool = True
    min_parallel_episodes: int = 64
    per_verse_min_parallel_episodes: Dict[str, int] = dataclasses.field(default_factory=dict)
    auto_tune_profile_path: str = os.path.join("models", "tuning", "parallel_thresholds.json")
    worker_auto_index: bool = False
    worker_verbose: bool = False
    annotate_parallel_worker_id: bool = False


class DistributedRolloutManager:
    def __init__(self, config: Optional[ParallelRolloutConfig] = None):
        self.config = config or ParallelRolloutConfig()
        self._ray_enabled = bool(self.config.use_ray and ray is not None)
        self._ray_started_here = False
        self._process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._per_verse_thresholds: Dict[str, int] = _normalize_threshold_map(
            self.config.per_verse_min_parallel_episodes
        )

        profile_path = str(self.config.auto_tune_profile_path or "").strip()
        if profile_path:
            prof_map = _load_threshold_map_from_profile(profile_path)
            if prof_map:
                self._per_verse_thresholds.update(prof_map)

        if self._ray_enabled:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True, include_dashboard=False)
                    self._ray_started_here = True
            except Exception:
                self._ray_enabled = False

    def _get_process_pool(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._process_pool is None:
            self._process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=max(1, int(self.config.num_workers))
            )
        return self._process_pool

    def _reset_process_pool(self) -> None:
        if self._process_pool is not None:
            try:
                self._process_pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._process_pool = None

    def _min_parallel_episodes_for_verse(self, verse_name: str) -> int:
        verse = _normalize_verse_name(verse_name)
        if verse and verse in self._per_verse_thresholds:
            return max(1, int(self._per_verse_thresholds[verse]))
        return max(1, int(self.config.min_parallel_episodes))

    def run_parallel_rollouts(
        self,
        *,
        agent_spec: AgentSpec,
        verse_spec: VerseSpec,
        total_episodes: int,
        max_steps: int = 200,
        seed: int = 123,
        run_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        os.makedirs(self.config.run_root, exist_ok=True)
        run_prefix = run_prefix or f"parallel_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        adversary_bundle: Optional[AdversaryBundle] = None
        if bool(self.config.self_play_enabled):
            try:
                sampler = AdversarySampler(memory_dir=self.config.self_play_memory_dir, runs_root=self.config.run_root)
                adversary_bundle = sampler.sample(
                    verse_name=str(verse_spec.verse_name),
                    source=str(self.config.self_play_adversary_source),
                    top_k=max(20, int(self.config.self_play_top_k)),
                )
            except Exception:
                adversary_bundle = None

        agent_spec_dict = dataclasses.asdict(agent_spec)
        if bool(self.config.self_play_enabled):
            cfg = agent_spec_dict.get("config")
            cfg = dict(cfg) if isinstance(cfg, dict) else {}
            cfg["self_play"] = {
                "enabled": True,
                "adversary_source": str(self.config.self_play_adversary_source),
                "mix_ratio": float(max(0.0, min(1.0, self.config.self_play_mix_ratio))),
                "adversary_bundle": (adversary_bundle.to_dict() if adversary_bundle is not None else None),
            }
            agent_spec_dict["config"] = cfg

        splits = _episodes_split(total_episodes, self.config.num_workers)
        payloads: List[Dict[str, Any]] = []
        for worker_id, eps in enumerate(splits):
            payloads.append(
                {
                    "worker_id": worker_id,
                    "episodes": int(eps),
                    "verse_spec": dataclasses.asdict(verse_spec),
                    "agent_spec": agent_spec_dict,
                    "run_root": self.config.run_root,
                    "max_steps": int(max_steps),
                    "seed": int(seed + worker_id * 10007),
                    "run_id": f"{run_prefix}_w{worker_id}",
                    "schema_version": self.config.schema_version,
                    "auto_index": bool(self.config.worker_auto_index),
                    "verbose": bool(self.config.worker_verbose),
                }
            )

        should_parallelize = bool(
            max(1, int(self.config.num_workers)) > 1
            and int(total_episodes) >= int(self._min_parallel_episodes_for_verse(str(verse_spec.verse_name)))
        )
        if not should_parallelize:
            payloads = [
                {
                    "worker_id": 0,
                    "episodes": int(total_episodes),
                    "verse_spec": dataclasses.asdict(verse_spec),
                    "agent_spec": agent_spec_dict,
                    "run_root": self.config.run_root,
                    "max_steps": int(max_steps),
                    "seed": int(seed),
                    "run_id": f"{run_prefix}_w0",
                    "schema_version": self.config.schema_version,
                    "auto_index": bool(self.config.worker_auto_index),
                    "verbose": bool(self.config.worker_verbose),
                }
            ]

        worker_results: List[Dict[str, Any]] = []
        if self._ray_enabled:
            refs = [_ray_worker_run.remote(p) for p in payloads if int(p["episodes"]) > 0]
            for r in ray.get(refs):
                worker_results.append(dict(r))
        else:
            active_payloads = [p for p in payloads if int(p["episodes"]) > 0]
            if not should_parallelize:
                for p in active_payloads:
                    worker_results.append(dict(_worker_run(p)))
            elif bool(self.config.reuse_process_pool):
                ex = self._get_process_pool()
                try:
                    futs = [ex.submit(_worker_run, p) for p in active_payloads]
                    for fut in concurrent.futures.as_completed(futs, timeout=self.config.max_worker_timeout_s):
                        worker_results.append(dict(fut.result()))
                except BrokenProcessPool:
                    # Recover from a dead pool once, then retry.
                    self._reset_process_pool()
                    ex = self._get_process_pool()
                    futs = [ex.submit(_worker_run, p) for p in active_payloads]
                    for fut in concurrent.futures.as_completed(futs, timeout=self.config.max_worker_timeout_s):
                        worker_results.append(dict(fut.result()))
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, self.config.num_workers)) as ex:
                    futs = [ex.submit(_worker_run, p) for p in active_payloads]
                    for fut in concurrent.futures.as_completed(futs, timeout=self.config.max_worker_timeout_s):
                        worker_results.append(dict(fut.result()))

        worker_results.sort(key=lambda x: int(x.get("worker_id", 10**9)))
        aggregate_run_id = self._merge_worker_runs(
            run_prefix=run_prefix,
            worker_results=worker_results,
            annotate_worker_id=bool(self.config.annotate_parallel_worker_id),
        )
        total_return = float(sum(float(r.get("total_return", 0.0)) for r in worker_results if not r.get("skipped")))
        total_steps = int(sum(int(r.get("total_steps", 0)) for r in worker_results if not r.get("skipped")))

        return {
            "run_id": aggregate_run_id,
            "worker_runs": [r.get("run_id") for r in worker_results if r.get("run_id")],
            "workers": worker_results,
            "total_return": total_return,
            "total_steps": total_steps,
            "parallel_decision": {
                "used_parallel_workers": bool(should_parallelize),
                "num_workers_configured": int(self.config.num_workers),
                "verse_name": str(verse_spec.verse_name),
                "episodes_requested": int(total_episodes),
                "min_parallel_episodes_effective": int(
                    self._min_parallel_episodes_for_verse(str(verse_spec.verse_name))
                ),
            },
            "self_play": {
                "enabled": bool(self.config.self_play_enabled),
                "adversary_source": str(self.config.self_play_adversary_source),
                "mix_ratio": float(self.config.self_play_mix_ratio),
                "bundle_policy_id": None if adversary_bundle is None else str(adversary_bundle.policy_id),
                "bundle_score": None if adversary_bundle is None else float(adversary_bundle.score),
            },
        }

    def _merge_worker_runs(
        self,
        *,
        run_prefix: str,
        worker_results: List[Dict[str, Any]],
        annotate_worker_id: bool = False,
    ) -> str:
        agg_run_id = f"{run_prefix}_merged"
        agg_dir = os.path.join(self.config.run_root, agg_run_id)
        os.makedirs(agg_dir, exist_ok=True)
        out_events = os.path.join(agg_dir, "events.jsonl")
        out_eps = os.path.join(agg_dir, "episodes.jsonl")

        with open(out_events, "w", encoding="utf-8") as wf:
            for wr in worker_results:
                rid = wr.get("run_id")
                if not rid:
                    continue
                src = os.path.join(self.config.run_root, str(rid), "events.jsonl")
                if not os.path.isfile(src):
                    continue
                with open(src, "r", encoding="utf-8") as rf:
                    if not annotate_worker_id:
                        shutil.copyfileobj(rf, wf)
                        continue
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                            info = ev.get("info", {}) or {}
                            if isinstance(info, dict):
                                info["parallel_worker_id"] = int(wr.get("worker_id", -1))
                                ev["info"] = info
                            wf.write(json.dumps(ev, ensure_ascii=False) + "\n")
                        except Exception:
                            wf.write(line + "\n")

        # Keep a lightweight per-worker summary.
        with open(out_eps, "w", encoding="utf-8") as ef:
            for wr in worker_results:
                ef.write(json.dumps(wr, ensure_ascii=False) + "\n")

        meta = {
            "run_id": agg_run_id,
            "created_at": int(time.time()),
            "worker_count": len(worker_results),
            "worker_runs": [wr.get("run_id") for wr in worker_results if wr.get("run_id")],
        }
        with open(os.path.join(agg_dir, "parallel_meta.json"), "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
        return agg_run_id

    def shutdown(self) -> None:
        self._reset_process_pool()
        if self._ray_enabled and self._ray_started_here and ray is not None:
            try:
                ray.shutdown()
            except Exception:
                pass


class PrioritizedRolloutScheduler:
    """
    Minimal priority scheduler:
    - explorer workers sample randomly
    - exploiter workers choose highest-priority candidate
    """

    def __init__(self, num_explorers: int = 3, num_exploiters: int = 1):
        self.num_explorers = max(0, int(num_explorers))
        self.num_exploiters = max(0, int(num_exploiters))

    def schedule(
        self,
        *,
        agent_specs: List[AgentSpec],
        verse_specs: List[VerseSpec],
        priorities: List[float],
        seed: int = 123,
    ) -> List[Dict[str, Any]]:
        import random

        rng = random.Random(seed)
        if not agent_specs or not verse_specs:
            return []
        max_idx = min(len(agent_specs), len(verse_specs), len(priorities)) - 1
        if max_idx < 0:
            return []

        out: List[Dict[str, Any]] = []
        worker_id = 0
        for _ in range(self.num_explorers):
            i = rng.randint(0, max_idx)
            out.append(
                {
                    "worker_id": worker_id,
                    "mode": "explore",
                    "agent_spec": agent_specs[i],
                    "verse_spec": verse_specs[i],
                    "priority": float(priorities[i]),
                }
            )
            worker_id += 1

        for _ in range(self.num_exploiters):
            i = int(max(range(max_idx + 1), key=lambda k: float(priorities[k])))
            out.append(
                {
                    "worker_id": worker_id,
                    "mode": "exploit",
                    "agent_spec": agent_specs[i],
                    "verse_spec": verse_specs[i],
                    "priority": float(priorities[i]),
                }
            )
            worker_id += 1
        return out


def _cli_test() -> None:
    # Smoke test with built-in random + line_world.
    v = VerseSpec(
        spec_version="v1",
        verse_name="line_world",
        verse_version="0.1",
        seed=123,
        tags=["parallel_test"],
        params={"goal_pos": 8, "max_steps": 30, "step_penalty": -0.02},
    )
    a = AgentSpec(
        spec_version="v1",
        policy_id="parallel_random",
        policy_version="0.1",
        algo="random",
        seed=123,
        tags=["parallel_test"],
        config=None,
    )
    mgr = DistributedRolloutManager(
        ParallelRolloutConfig(
            num_workers=2,
            use_ray=False,
            self_play_enabled=False,
        )
    )
    out = mgr.run_parallel_rollouts(agent_spec=a, verse_spec=v, total_episodes=6, max_steps=30, seed=123)
    mgr.shutdown()
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _main_cli() -> None:
    def _parse_threshold_rows(rows: Optional[List[str]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for row in list(rows or []):
            s = str(row).strip()
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            verse = _normalize_verse_name(k)
            if not verse:
                continue
            try:
                out[verse] = max(1, int(v))
            except Exception:
                continue
        return out

    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, default="random")
    ap.add_argument("--verse", type=str, default="line_world")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--use_ray", action="store_true")
    ap.add_argument("--reuse_process_pool", dest="reuse_process_pool", action="store_true")
    ap.add_argument("--no_reuse_process_pool", dest="reuse_process_pool", action="store_false")
    ap.add_argument("--min_parallel_episodes", type=int, default=64)
    ap.add_argument("--per_verse_threshold", action="append", default=None, help="e.g. chess_world=32")
    ap.add_argument(
        "--auto_tune_profile_path",
        type=str,
        default=os.path.join("models", "tuning", "parallel_thresholds.json"),
    )
    ap.add_argument("--worker_auto_index", action="store_true")
    ap.add_argument("--worker_verbose", action="store_true")
    ap.add_argument("--annotate_parallel_worker_id", action="store_true")
    ap.add_argument("--self_play.enabled", dest="self_play_enabled", action="store_true")
    ap.add_argument(
        "--self_play.adversary_source",
        dest="self_play_adversary_source",
        type=str,
        default="recent_failures",
        choices=["recent_failures", "near_misses", "top_regret"],
    )
    ap.add_argument("--self_play.mix_ratio", dest="self_play_mix_ratio", type=float, default=0.25)
    ap.add_argument("--self_play.memory_dir", dest="self_play_memory_dir", type=str, default="central_memory")
    ap.add_argument("--self_play.top_k", dest="self_play_top_k", type=int, default=300)
    ap.set_defaults(reuse_process_pool=True)
    args = ap.parse_args()

    verse = VerseSpec(
        spec_version="v1",
        verse_name=str(args.verse),
        verse_version="0.1",
        seed=int(args.seed),
        tags=["parallel_cli"],
        params={"max_steps": int(args.max_steps)},
    )
    agent = AgentSpec(
        spec_version="v1",
        policy_id=f"parallel_{args.algo}",
        policy_version="0.1",
        algo=str(args.algo),
        seed=int(args.seed),
        tags=["parallel_cli"],
        config={"train": True},
    )
    mgr = DistributedRolloutManager(
        ParallelRolloutConfig(
            num_workers=max(1, int(args.workers)),
            use_ray=bool(args.use_ray),
            run_root=str(args.runs_root),
            reuse_process_pool=bool(args.reuse_process_pool),
            min_parallel_episodes=max(1, int(args.min_parallel_episodes)),
            per_verse_min_parallel_episodes=_parse_threshold_rows(args.per_verse_threshold),
            auto_tune_profile_path=str(args.auto_tune_profile_path),
            worker_auto_index=bool(args.worker_auto_index),
            worker_verbose=bool(args.worker_verbose),
            annotate_parallel_worker_id=bool(args.annotate_parallel_worker_id),
            self_play_enabled=bool(args.self_play_enabled),
            self_play_adversary_source=str(args.self_play_adversary_source),
            self_play_mix_ratio=float(args.self_play_mix_ratio),
            self_play_memory_dir=str(args.self_play_memory_dir),
            self_play_top_k=max(20, int(args.self_play_top_k)),
        )
    )
    out = mgr.run_parallel_rollouts(
        agent_spec=agent,
        verse_spec=verse,
        total_episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
    )
    mgr.shutdown()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main_cli()
