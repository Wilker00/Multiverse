"""
orchestrator/trainer.py

Updated trainer that uses verses/registry instead of hardcoded imports.

This is the moment the project stops being a demo and becomes a framework.
The trainer does not know or care which Verse it runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from typing import Optional, Dict, Any

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)


from core.adversary_sampler import AdversarialMixWrapper, AdversaryBundle
from core.types import (
    RunRef,
    AgentRef,
    VerseRef,
    VerseSpec,
    AgentSpec,
    validate_agent_spec,
    validate_verse_spec,
)
from core.rollout import RolloutConfig, run_episodes
from core.safe_executor import SafeExecutor, SafeExecutorConfig
from memory.event_log import EventLogConfig, EventLogger, make_on_step_writer
from memory.retrieval import RetrievalConfig, RetrievalClient
from memory.episode_index import EpisodeIndexConfig, build_episode_index
from verses.registry import register_builtin as register_builtin_verses, create_verse

from agents.registry import register_builtin_agents, create_agent


def _hash_verse_spec(spec: VerseSpec) -> str:
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


class Trainer:
    """
    Single-agent trainer using the Verse registry.
    """

    def __init__(
        self,
        *,
        run_root: str = "runs",
        schema_version: str = "v1",
        auto_register_builtin: bool = True,
    ):
        self.run_root = run_root
        self.schema_version = schema_version

        if auto_register_builtin:
            register_builtin_verses()
            register_builtin_agents()

    @staticmethod
    def _hydrate_agent_datasets(agent: Any, spec: AgentSpec) -> None:
        algo = (spec.algo or "").strip().lower()
        if algo not in (
            "q",
            "memory_recall",
            "planner_recall",
            "aware",
            "evolving",
            "imitation_lookup",
            "library",
            "special",
            "special_moe",
            "adaptive_moe",
            "cql",
            "failure_aware",
        ):
            return

        dataset_paths = []
        dataset_dir = None
        bad_dataset_path = None
        if isinstance(spec.config, dict):
            cfg = spec.config
            dataset_dir = cfg.get("dataset_dir")
            dp = cfg.get("dataset_paths")
            if dp:
                dataset_paths = list(dp) if isinstance(dp, list) else [dp]
            if not dataset_paths and cfg.get("dataset_path"):
                dataset_paths = [cfg.get("dataset_path")]
            bad_dataset_path = cfg.get("bad_dna_path")

        if dataset_dir:
            import glob
            dataset_paths.extend(sorted(glob.glob(str(dataset_dir) + "/*.jsonl")))

        if dataset_paths:
            if hasattr(agent, "learn_from_dataset"):
                for path in dataset_paths:
                    agent.learn_from_dataset(str(path))
            else:
                raise RuntimeError(f"{algo} agent missing learn_from_dataset()")
        if bad_dataset_path and hasattr(agent, "learn_from_bad_dataset"):
            agent.learn_from_bad_dataset(str(bad_dataset_path))

    def run(
        self,
        *,
        verse_spec: VerseSpec,
        agent_spec: AgentSpec,
        episodes: int,
        max_steps: int,
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
        auto_index: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        validate_verse_spec(verse_spec)
        validate_agent_spec(agent_spec)

        # ---------------------------
        # Run identity
        # ---------------------------
        if run_id:
            run = RunRef(run_id=run_id)
        else:
            run = RunRef.create()

        # ---------------------------
        # Verse setup (via registry)
        # ---------------------------
        spec_hash = _hash_verse_spec(verse_spec)
        verse_ref = VerseRef.create(
            verse_name=verse_spec.verse_name,
            verse_version=verse_spec.verse_version,
            spec_hash=spec_hash,
        )

        verse = create_verse(verse_spec)

        # ---------------------------
        # Agent setup
        # ---------------------------
        agent_ref = AgentRef.create(
            policy_id=agent_spec.policy_id,
            policy_version=agent_spec.policy_version,
        )

        # Agent-specific config normalization before construction.
        algo = (agent_spec.algo or "").strip().lower()
        if algo in ("special_moe", "gateway", "aware", "evolving", "memory_recall", "planner_recall"):
            cfg = dict(agent_spec.config) if isinstance(agent_spec.config, dict) else {}
            cfg.setdefault("verse_name", verse_spec.verse_name)
            agent_spec = dataclasses.replace(agent_spec, config=cfg)

        runtime_cfg = dict(agent_spec.config) if isinstance(agent_spec.config, dict) else {}
        safe_cfg_raw = runtime_cfg.pop("safe_executor", None)
        self_play_cfg_raw = runtime_cfg.pop("self_play", None)
        agent_runtime_spec = dataclasses.replace(agent_spec, config=runtime_cfg if runtime_cfg else None)

        agent = create_agent(
            spec=agent_runtime_spec,
            observation_space=verse.observation_space,
            action_space=verse.action_space,
        )

        # Special hook: dataset-driven agents can be hydrated from files.
        self._hydrate_agent_datasets(agent, agent_runtime_spec)

        # Optional adversarial self-play wrapper.
        if isinstance(self_play_cfg_raw, dict) and bool(self_play_cfg_raw.get("enabled", False)):
            bundle_raw = self_play_cfg_raw.get("adversary_bundle")
            mix_ratio = float(self_play_cfg_raw.get("mix_ratio", 0.25))
            bundle = None
            if isinstance(bundle_raw, dict):
                try:
                    bundle = AdversaryBundle(
                        verse_name=str(bundle_raw.get("verse_name", verse_spec.verse_name)),
                        source=str(bundle_raw.get("source", self_play_cfg_raw.get("adversary_source", "recent_failures"))),
                        policy_id=str(bundle_raw.get("policy_id", "adversary:unknown")),
                        created_at_ms=int(bundle_raw.get("created_at_ms", 0) or 0),
                        run_ids=[str(x) for x in (bundle_raw.get("run_ids") or [])],
                        obs_actions={
                            str(k): {str(ak): float(av) for ak, av in dict(v).items()}
                            for k, v in dict(bundle_raw.get("obs_actions") or {}).items()
                        },
                        global_actions={
                            str(k): float(v) for k, v in dict(bundle_raw.get("global_actions") or {}).items()
                        },
                        score=float(bundle_raw.get("score", 0.0) or 0.0),
                    )
                except Exception:
                    bundle = None
            if bundle is not None:
                agent = AdversarialMixWrapper(agent, mix_ratio=float(mix_ratio), bundle=bundle, seed=seed)

        safe_executor = None
        if safe_cfg_raw is not None:
            safe_cfg_dict = safe_cfg_raw if isinstance(safe_cfg_raw, dict) else {}
            if (
                str(verse_spec.verse_name).strip().lower() == "cliff_world"
                and "danger_threshold" not in safe_cfg_dict
            ):
                safe_cfg_dict = dict(safe_cfg_dict)
                # Safer default for cliff-style hazards when no explicit threshold was provided.
                safe_cfg_dict["danger_threshold"] = 0.60
            safe_runtime_cfg = dict(safe_cfg_dict)
            # Wrapper-only fields are consumed here and should not leak into SafeExecutorConfig.
            safe_runtime_cfg.pop("fallback_algo", None)
            safe_runtime_cfg.pop("fallback_config", None)
            safe_cfg = SafeExecutorConfig.from_dict(safe_runtime_cfg)
            fallback_agent = None
            fallback_algo = str(safe_cfg_dict.get("fallback_algo", "")).strip().lower()
            if fallback_algo:
                fb_cfg = safe_cfg_dict.get("fallback_config")
                fb_cfg = dict(fb_cfg) if isinstance(fb_cfg, dict) else {}
                if fallback_algo in (
                    "gateway",
                    "special_moe",
                    "adaptive_moe",
                    "aware",
                    "evolving",
                    "memory_recall",
                    "planner_recall",
                ):
                    fb_cfg.setdefault("verse_name", verse_spec.verse_name)
                fallback_spec = AgentSpec(
                    spec_version=agent_runtime_spec.spec_version,
                    policy_id=f"safe_fallback:{fallback_algo}:{agent_runtime_spec.policy_id}",
                    policy_version=agent_runtime_spec.policy_version,
                    algo=fallback_algo,
                    framework=agent_runtime_spec.framework,
                    seed=agent_runtime_spec.seed,
                    tags=list(agent_runtime_spec.tags),
                    config=(fb_cfg if fb_cfg else None),
                )
                try:
                    fallback_agent = create_agent(
                        spec=fallback_spec,
                        observation_space=verse.observation_space,
                        action_space=verse.action_space,
                    )
                    self._hydrate_agent_datasets(fallback_agent, fallback_spec)
                except Exception as e:
                    print(f"safe-executor: fallback init skipped ({fallback_algo}): {e}")
                    fallback_agent = None
            safe_executor = SafeExecutor(
                config=safe_cfg,
                verse=verse,
                fallback_agent=fallback_agent,
            )

        # ---------------------------
        # Rollout + logging
        # ---------------------------
        
        # Check if training is requested in config
        config_train = False
        if isinstance(agent_spec.config, dict):
            config_train = bool(agent_spec.config.get("train", False))

        # Default logic: simple_pg always trains, others only if config says so
        train = (algo == "simple_pg") or config_train
        collect = (algo == "simple_pg") or config_train

        rollout_cfg = RolloutConfig(
            schema_version=self.schema_version,
            max_steps=max_steps,
            train=train,
            collect_transitions=collect,
            safe_executor=safe_executor,
            retriever=(
                RetrievalClient(
                    RetrievalConfig(
                        run_dir=str(
                            runtime_cfg.get("retrieval_run_dir")
                            or os.path.join(self.run_root, run.run_id)
                        )
                    )
                )
                if bool(runtime_cfg.get("rar_enabled", True))
                else None
            ),
            retrieval_interval=int(runtime_cfg.get("retrieval_interval", 10)),
            on_demand_memory_enabled=bool(
                runtime_cfg.get("on_demand_memory_enabled", algo in ("memory_recall", "planner_recall"))
            ),
            on_demand_memory_root=str(runtime_cfg.get("on_demand_memory_root", "central_memory")),
            on_demand_query_budget=int(runtime_cfg.get("on_demand_query_budget", 8)),
            on_demand_min_interval=int(runtime_cfg.get("on_demand_min_interval", 2)),
        )


        log_cfg = EventLogConfig(
            root_dir=self.run_root,
            run_id=run.run_id,
        )

        with EventLogger(log_cfg) as logger:
            on_step = make_on_step_writer(logger)

            results = run_episodes(
                verse=verse,
                verse_ref=verse_ref,
                agent=agent,
                agent_ref=agent_ref,
                run=run,
                config=rollout_cfg,
                episodes=episodes,
                seed=seed,
                on_step=on_step,
            )

            # Log episode-level metrics
            for result in results:
                if result.train_metrics:
                    logger.write_metrics(result.train_metrics)
            
            # Auto-index the run to make it searchable in the Knowledge Market.
            if bool(auto_index):
                try:
                    build_episode_index(EpisodeIndexConfig(run_dir=os.path.join(self.run_root, run.run_id)))
                except Exception as e:
                    if bool(verbose):
                        print(f"trainer: auto-indexing failed: {e}")


        verse.close()
        agent.close()
        if safe_executor is not None:
            safe_executor.close()

        # ---------------------------
        # Summary
        # ---------------------------
        total_steps = sum(r.steps for r in results)
        total_return = sum(r.return_sum for r in results)

        if bool(verbose):
            print("Run complete")
            print(f"run_id       : {run.run_id}")
            print(f"verse        : {verse_spec.verse_name}")
            print(f"episodes     : {episodes}")
            print(f"total_steps  : {total_steps}")
            print(f"total_return : {total_return:.3f}")
            print(f"log_dir      : {self.run_root}/{run.run_id}")

        return {
            "run_id": run.run_id,
            "total_return": total_return,
            "total_steps": total_steps,
        }


# -------------------------------------------------------
# CLI-style smoke test
# -------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verse", default="grid_world")
    parser.add_argument("--algo", default="ppo")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rar", action="store_true", help="Enable Retrieval-Augmented Rollouts")
    args = parser.parse_args()

    v_spec = VerseSpec(
        spec_version="v1",
        verse_name=args.verse,
        verse_version="0.1",
        seed=args.seed,
    )

    a_spec = AgentSpec(
        spec_version="v1",
        policy_id=f"{args.algo}_session",
        policy_version="0.1",
        algo=args.algo,
        seed=args.seed,
        config={"rar_enabled": args.rar, "train": True}
    )

    trainer = Trainer()
    trainer.run(
        verse_spec=v_spec,
        agent_spec=a_spec,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
