"""
orchestrator/ray_adapter.py

Ray/RLlib integration adapter for distributed training.

This module bridges Multiverse's training infrastructure with Ray for
distributed episode execution. It wraps core/parallel_rollout.py's
DistributedRolloutManager, which already supports Ray as a backend.

Usage:
    from orchestrator.ray_adapter import RayTrainer

    trainer = RayTrainer(num_workers=4)
    results = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=100,
        max_steps=200,
        seed=42,
    )

Requirements:
    pip install -r requirements_scale.txt  # includes ray>=2.9.0
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from core.types import AgentSpec, VerseSpec


def ensure_ray() -> None:
    """Verify Ray is installed and importable."""
    try:
        import ray  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Ray is not installed. Install with: pip install -r requirements_scale.txt"
        ) from exc


def is_ray_available() -> bool:
    """Check if Ray is installed without raising."""
    try:
        import ray  # type: ignore
        return True
    except ImportError:
        return False


def ray_cluster_info() -> Dict[str, Any]:
    """Return information about the current Ray cluster, or empty dict if unavailable."""
    try:
        import ray  # type: ignore
        if not ray.is_initialized():
            return {"initialized": False}
        resources = ray.cluster_resources()
        return {
            "initialized": True,
            "num_cpus": resources.get("CPU", 0),
            "num_gpus": resources.get("GPU", 0),
            "node_count": len(ray.nodes()),
        }
    except Exception:
        return {"initialized": False, "error": "Ray not available"}


class RayTrainer:
    """
    Distributed trainer that delegates to DistributedRolloutManager with Ray backend.

    This is the recommended entry point for multi-node or multi-worker training.
    For single-machine training, use orchestrator/trainer.py directly.
    """

    def __init__(
        self,
        *,
        num_workers: int = 4,
        run_root: str = "runs",
        schema_version: str = "v1",
        max_worker_timeout_s: int = 3600,
        self_play_enabled: bool = False,
        self_play_mix_ratio: float = 0.25,
        min_parallel_episodes: int = 64,
        per_verse_min_parallel_episodes: Optional[Dict[str, int]] = None,
        auto_tune_profile_path: str = "",
    ):
        ensure_ray()

        from core.parallel_rollout import DistributedRolloutManager, ParallelRolloutConfig

        self._config = ParallelRolloutConfig(
            num_workers=num_workers,
            use_ray=True,
            run_root=run_root,
            schema_version=schema_version,
            max_worker_timeout_s=max_worker_timeout_s,
            self_play_enabled=self_play_enabled,
            self_play_mix_ratio=self_play_mix_ratio,
            min_parallel_episodes=min_parallel_episodes,
            per_verse_min_parallel_episodes=per_verse_min_parallel_episodes or {},
            auto_tune_profile_path=auto_tune_profile_path,
        )
        self._manager = DistributedRolloutManager(self._config)

    def run(
        self,
        *,
        verse_spec: VerseSpec,
        agent_spec: AgentSpec,
        episodes: int,
        max_steps: int = 200,
        seed: int = 123,
        run_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run distributed episodes across Ray workers.

        Returns a dict with:
            - worker_results: list of per-worker result dicts
            - total_return: sum of all worker returns
            - total_steps: sum of all worker steps
            - run_prefix: the parallel run identifier
        """
        return self._manager.run_parallel_rollouts(
            verse_spec=verse_spec,
            agent_spec=agent_spec,
            total_episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            run_prefix=run_prefix,
        )

    def shutdown(self) -> None:
        """Shut down the Ray-backed manager and release resources."""
        self._manager.shutdown()

    @property
    def cluster_info(self) -> Dict[str, Any]:
        return ray_cluster_info()
