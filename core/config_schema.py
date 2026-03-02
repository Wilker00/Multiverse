"""
core/config_schema.py

Pydantic schemas for Multiverse configuration with YAML/JSON support.

This module provides:
- Type-safe configuration schemas
- Validation with clear error messages
- YAML/JSON serialization/deserialization
- Default value documentation
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ParallelRolloutSchema(BaseModel):
    """Parallel execution configuration."""

    num_workers: int = Field(
        default=4,
        ge=1,
        le=256,
        description="Number of parallel worker processes"
    )
    max_worker_timeout_s: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Maximum worker timeout in seconds (1 min - 24 hours)"
    )
    use_ray: bool = Field(
        default=False,
        description="Use Ray for distributed execution"
    )
    run_root: str = Field(
        default="runs",
        description="Root directory for run outputs"
    )
    reuse_process_pool: bool = Field(
        default=True,
        description="Reuse process pool across rollouts"
    )
    min_parallel_episodes: int = Field(
        default=64,
        ge=1,
        description="Minimum episodes before parallelizing"
    )


class RolloutSchema(BaseModel):
    """Episode rollout configuration."""

    retrieval_interval: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Steps between memory retrievals"
    )
    on_demand_query_budget: int = Field(
        default=8,
        ge=0,
        le=1000,
        description="Maximum memory queries per episode"
    )
    on_demand_min_interval: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Minimum steps between consecutive queries"
    )


class MCTSSchema(BaseModel):
    """Monte Carlo Tree Search configuration."""

    num_simulations: int = Field(
        default=96,
        ge=1,
        le=10000,
        description="Number of MCTS simulations per search"
    )
    max_depth: int = Field(
        default=12,
        ge=1,
        le=100,
        description="Maximum simulation depth"
    )
    c_puct: float = Field(
        default=1.4,
        ge=0.0,
        le=10.0,
        description="PUCT exploration constant"
    )
    discount: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Discount factor gamma"
    )
    transposition_max_entries: int = Field(
        default=20000,
        ge=100,
        le=10000000,
        description="Transposition table maximum entries"
    )


class CurriculumSchema(BaseModel):
    """Adaptive curriculum learning configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable adaptive curriculum"
    )
    plateau_window: int = Field(
        default=5,
        ge=2,
        le=100,
        description="Episodes to detect plateau"
    )
    step_size: float = Field(
        default=0.05,
        ge=0.001,
        le=1.0,
        description="Difficulty adjustment step size"
    )
    collapse_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Success rate collapse threshold"
    )
    max_noise: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Maximum action noise"
    )
    max_partial_obs: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Maximum partial observability"
    )
    max_distractors: int = Field(
        default=6,
        ge=0,
        le=100,
        description="Maximum distractor objects"
    )


class MemorySchema(BaseModel):
    """Central memory system configuration."""

    lock_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Lock timeout in seconds"
    )
    delta_merge_threshold: int = Field(
        default=1000,
        ge=10,
        le=1000000,
        description="Rows before delta merge"
    )
    query_cache_size: int = Field(
        default=10000,
        ge=0,
        le=1000000,
        description="LRU query cache size"
    )
    query_cache_ttl_ms: int = Field(
        default=60000,
        ge=0,
        le=3600000,
        description="Query cache TTL in milliseconds"
    )
    use_ann: bool = Field(
        default=True,
        description="Use FAISS for ANN search"
    )
    ann_candidate_count: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="ANN candidate count before reranking"
    )
    cache_limit: int = Field(
        default=150000,
        ge=1000,
        le=10000000,
        description="Maximum similarity cache entries"
    )


class SafeExecutorSchema(BaseModel):
    """Runtime safety configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable SafeExecutor"
    )
    danger_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Danger classification threshold"
    )
    min_action_confidence: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Minimum action confidence"
    )
    fallback_horizon_steps: int = Field(
        default=8,
        ge=1,
        le=1000,
        description="Fallback policy horizon"
    )
    checkpoint_interval: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Steps between checkpoints"
    )
    max_rewinds_per_episode: int = Field(
        default=8,
        ge=0,
        le=100,
        description="Maximum rewinds per episode"
    )
    planner_enabled: bool = Field(
        default=False,
        description="Enable A* planning"
    )
    planner_max_expansions: int = Field(
        default=8000,
        ge=100,
        le=1000000,
        description="Max A* node expansions"
    )
    mcts_enabled: bool = Field(
        default=False,
        description="Enable MCTS search"
    )
    mcts_num_simulations: int = Field(
        default=96,
        ge=1,
        le=10000,
        description="MCTS simulations"
    )


class MultiverseConfig(BaseModel):
    """
    Top-level Multiverse configuration.

    Supports loading from YAML/JSON with validation.
    Environment variables override file values.
    """

    parallel: ParallelRolloutSchema = Field(
        default_factory=ParallelRolloutSchema,
        description="Parallel execution settings"
    )
    rollout: RolloutSchema = Field(
        default_factory=RolloutSchema,
        description="Rollout settings"
    )
    mcts: MCTSSchema = Field(
        default_factory=MCTSSchema,
        description="MCTS settings"
    )
    curriculum: CurriculumSchema = Field(
        default_factory=CurriculumSchema,
        description="Curriculum learning settings"
    )
    memory: MemorySchema = Field(
        default_factory=MemorySchema,
        description="Memory system settings"
    )
    safe_executor: SafeExecutorSchema = Field(
        default_factory=SafeExecutorSchema,
        description="Safety system settings"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True

    @field_validator("*", mode="before")
    @classmethod
    def apply_env_overrides(cls, v, info):
        """Apply environment variable overrides to all fields."""
        # This is called for each field during validation
        # For nested models, environment variables are applied by config_loader
        return v


def load_from_dict(data: Dict) -> MultiverseConfig:
    """
    Load configuration from a dictionary (parsed from YAML/JSON).

    Args:
        data: Configuration dictionary

    Returns:
        Validated MultiverseConfig instance

    Raises:
        pydantic.ValidationError: If configuration is invalid
    """
    return MultiverseConfig(**data)


def to_dict(config: MultiverseConfig) -> Dict:
    """
    Convert configuration to dictionary (for YAML/JSON export).

    Args:
        config: MultiverseConfig instance

    Returns:
        Configuration as dictionary
    """
    return config.model_dump(exclude_none=True)


def get_schema_json() -> str:
    """
    Get JSON schema for configuration validation.

    Returns:
        JSON schema string
    """
    return MultiverseConfig.model_json_schema()
