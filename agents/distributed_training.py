"""
agents/distributed_training.py

Local distributed training orchestration compatible with existing Multiverse APIs.
This module parallelizes training shards across local processes and merges artifacts.

It intentionally avoids Kubernetes assumptions and works on one machine.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.pbt_controller import PBTConfig, PBTController
from core.types import AgentSpec, VerseSpec


@dataclass
class DistributedTrainingConfig:
    workers: int = 2
    run_root: str = "runs"
    schema_version: str = "v1"
    max_worker_timeout_s: int = 60 * 60
    merge_results: bool = True


def _train_shard(payload: Dict[str, Any]) -> Dict[str, Any]:
    from orchestrator.trainer import Trainer

    worker_id = int(payload["worker_id"])
    episodes = int(payload["episodes"])
    if episodes <= 0:
        return {"worker_id": worker_id, "skipped": True}

    trainer = Trainer(
        run_root=str(payload["run_root"]),
        schema_version=str(payload.get("schema_version", "v1")),
        auto_register_builtin=True,
    )
    verse_spec = VerseSpec(**payload["verse_spec"])
    agent_spec = AgentSpec(**payload["agent_spec"])
    started = time.time()
    result = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=episodes,
        max_steps=int(payload["max_steps"]),
        seed=int(payload["seed"]),
        run_id=str(payload["run_id"]),
    )
    return {
        "worker_id": worker_id,
        "skipped": False,
        "run_id": result["run_id"],
        "total_return": float(result["total_return"]),
        "total_steps": int(result["total_steps"]),
        "elapsed_s": float(time.time() - started),
    }


class LocalDistributedTrainer:
    def __init__(self, config: Optional[DistributedTrainingConfig] = None):
        self.config = config or DistributedTrainingConfig()

    def train(
        self,
        *,
        agent_spec: AgentSpec,
        verse_spec: VerseSpec,
        total_episodes: int,
        max_steps: int,
        seed: int = 123,
        run_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        os.makedirs(self.config.run_root, exist_ok=True)
        workers = max(1, int(self.config.workers))
        run_prefix = run_prefix or f"dist_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        splits = self._split(total_episodes, workers)
        payloads: List[Dict[str, Any]] = []
        for i, eps in enumerate(splits):
            payloads.append(
                {
                    "worker_id": i,
                    "episodes": int(eps),
                    "run_root": self.config.run_root,
                    "schema_version": self.config.schema_version,
                    "verse_spec": dataclasses.asdict(verse_spec),
                    "agent_spec": dataclasses.asdict(agent_spec),
                    "max_steps": int(max_steps),
                    "seed": int(seed + i * 9973),
                    "run_id": f"{run_prefix}_w{i}",
                }
            )

        results: List[Dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_train_shard, p) for p in payloads if int(p["episodes"]) > 0]
            for fut in concurrent.futures.as_completed(futs, timeout=self.config.max_worker_timeout_s):
                results.append(dict(fut.result()))

        merged_run_id = None
        if self.config.merge_results:
            merged_run_id = self._merge_runs(run_prefix=run_prefix, shard_results=results)

        return {
            "run_id": merged_run_id or run_prefix,
            "worker_runs": [r.get("run_id") for r in results if r.get("run_id")],
            "workers": results,
            "total_return": float(sum(float(r.get("total_return", 0.0)) for r in results)),
            "total_steps": int(sum(int(r.get("total_steps", 0)) for r in results)),
        }

    def train_pbt(
        self,
        *,
        base_agent_spec: AgentSpec,
        verse_spec: VerseSpec,
        generations: int,
        episodes_per_member: int,
        max_steps: int,
        seed: int = 123,
        pbt_config: Optional[PBTConfig] = None,
    ) -> Dict[str, Any]:
        pbt = PBTController(pbt_config or PBTConfig())
        members = pbt.init_population(base_agent_spec)
        history: List[Dict[str, Any]] = []

        for gen in range(max(1, int(generations))):
            scores: Dict[str, float] = {}
            gen_rows: List[Dict[str, Any]] = []
            for i, m in enumerate(members):
                out = self.train(
                    agent_spec=m.agent_spec,
                    verse_spec=verse_spec,
                    total_episodes=max(1, int(episodes_per_member)),
                    max_steps=max(1, int(max_steps)),
                    seed=int(seed + gen * 1000 + i * 17),
                    run_prefix=f"pbt_g{gen}_{m.member_id}",
                )
                score = float(out.get("total_return", 0.0)) / float(max(1, int(episodes_per_member)))
                scores[m.member_id] = float(score)
                gen_rows.append(
                    {
                        "member_id": m.member_id,
                        "policy_id": m.agent_spec.policy_id,
                        "score": float(score),
                        "run_id": str(out.get("run_id", "")),
                    }
                )

            pbt.update_scores(scores)
            x = pbt.maybe_exploit_explore()
            members = list(pbt.population)
            history.append({"generation": gen, "scores": gen_rows, "pbt_step": x})

        ranked = sorted(pbt.population, key=lambda z: float(z.score), reverse=True)
        best = ranked[0] if ranked else None
        return {
            "mode": "pbt",
            "generations": int(generations),
            "population_size": len(members),
            "best_member": None
            if best is None
            else {
                "member_id": best.member_id,
                "policy_id": best.agent_spec.policy_id,
                "score": float(best.score),
                "retention_multiplier": float(best.retention_multiplier),
            },
            "history": history,
            "retention_file": str((pbt_config or PBTConfig()).retention_file),
        }

    @staticmethod
    def _split(total: int, workers: int) -> List[int]:
        total = max(0, int(total))
        workers = max(1, int(workers))
        base = total // workers
        rem = total % workers
        return [base + (1 if i < rem else 0) for i in range(workers)]

    def _merge_runs(self, *, run_prefix: str, shard_results: List[Dict[str, Any]]) -> str:
        merged_run_id = f"{run_prefix}_merged"
        merged_dir = os.path.join(self.config.run_root, merged_run_id)
        os.makedirs(merged_dir, exist_ok=True)

        out_events = os.path.join(merged_dir, "events.jsonl")
        with open(out_events, "w", encoding="utf-8") as wf:
            for r in shard_results:
                rid = r.get("run_id")
                if not rid:
                    continue
                src = os.path.join(self.config.run_root, str(rid), "events.jsonl")
                if not os.path.isfile(src):
                    continue
                with open(src, "r", encoding="utf-8") as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                            info = ev.get("info", {}) or {}
                            if isinstance(info, dict):
                                info["distributed_worker_id"] = int(r.get("worker_id", -1))
                                ev["info"] = info
                            wf.write(json.dumps(ev, ensure_ascii=False) + "\n")
                        except Exception:
                            wf.write(line + "\n")

        with open(os.path.join(merged_dir, "distributed_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": merged_run_id,
                    "created_at_ms": int(time.time() * 1000),
                    "worker_runs": [r.get("run_id") for r in shard_results if r.get("run_id")],
                    "worker_count": len(shard_results),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return merged_run_id


def create_distributed_trainer(workers: int = 2, run_root: str = "runs") -> LocalDistributedTrainer:
    return LocalDistributedTrainer(DistributedTrainingConfig(workers=workers, run_root=run_root))


if __name__ == "__main__":
    # quick smoke
    verse = VerseSpec(
        spec_version="v1",
        verse_name="line_world",
        verse_version="0.1",
        seed=123,
        tags=["distributed_smoke"],
        params={"goal_pos": 8, "max_steps": 30, "step_penalty": -0.02},
    )
    agent = AgentSpec(
        spec_version="v1",
        policy_id="dist_smoke_random",
        policy_version="0.1",
        algo="random",
        seed=123,
        tags=["distributed_smoke"],
        config=None,
    )
    trainer = create_distributed_trainer(workers=2)
    out = trainer.train(agent_spec=agent, verse_spec=verse, total_episodes=8, max_steps=30)
    print(json.dumps(out, ensure_ascii=False, indent=2))
