"""
core/config_loader.py

Configuration loader with priority system:
  Command-line args > Environment Variables > YAML/JSON > Code Defaults

Supports:
- YAML and JSON configuration files
- Environment variable overrides
- Configuration validation
- Multiple config file locations
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from core.config_schema import (
    MultiverseConfig,
    ParallelRolloutSchema,
    RolloutSchema,
    MCTSSchema,
    CurriculumSchema,
    MemorySchema,
    SafeExecutorSchema,
)


class ConfigLoader:
    """
    Load and merge configuration from multiple sources.

    Priority order (highest to lowest):
    1. Environment variables (MULTIVERSE_*)
    2. YAML/JSON configuration file
    3. Default values from schema
    """

    DEFAULT_CONFIG_PATHS = [
        "multiverse.yaml",
        "multiverse.yml",
        "multiverse.json",
        "config/multiverse.yaml",
        "config/multiverse.yml",
        ".multiverse.yaml",
        ".multiverse.yml",
    ]

    @staticmethod
    def find_config_file(path: Optional[str] = None) -> Optional[Path]:
        """
        Find configuration file in standard locations.

        Args:
            path: Explicit path to config file (overrides search)

        Returns:
            Path to config file, or None if not found
        """
        if path:
            p = Path(path)
            if p.exists():
                return p
            raise FileNotFoundError(f"Config file not found: {path}")

        # Search default locations
        for default_path in ConfigLoader.DEFAULT_CONFIG_PATHS:
            p = Path(default_path)
            if p.exists():
                return p

        return None

    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            Configuration dictionary
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def load_json(path: Path) -> Dict[str, Any]:
        """
        Load JSON configuration file.

        Args:
            path: Path to JSON file

        Returns:
            Configuration dictionary
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Environment variables follow pattern: MULTIVERSE_<SECTION>_<FIELD>
        Examples:
          MULTIVERSE_PARALLEL_NUM_WORKERS=16
          MULTIVERSE_MCTS_NUM_SIMULATIONS=128
          MULTIVERSE_CURRICULUM_ENABLED=0

        Args:
            config_dict: Base configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        # Deep copy to avoid modifying input
        result = dict(config_dict)

        # Parallel rollout overrides
        if "parallel" not in result:
            result["parallel"] = {}
        parallel = result["parallel"]

        if "MULTIVERSE_PARALLEL_NUM_WORKERS" in os.environ:
            parallel["num_workers"] = int(os.environ["MULTIVERSE_PARALLEL_NUM_WORKERS"])
        if "MULTIVERSE_PARALLEL_MAX_TIMEOUT" in os.environ:
            parallel["max_worker_timeout_s"] = int(os.environ["MULTIVERSE_PARALLEL_MAX_TIMEOUT"])
        if "MULTIVERSE_PARALLEL_USE_RAY" in os.environ:
            parallel["use_ray"] = bool(int(os.environ["MULTIVERSE_PARALLEL_USE_RAY"]))

        # Rollout overrides
        if "rollout" not in result:
            result["rollout"] = {}
        rollout = result["rollout"]

        if "MULTIVERSE_ROLLOUT_RETRIEVAL_INTERVAL" in os.environ:
            rollout["retrieval_interval"] = int(os.environ["MULTIVERSE_ROLLOUT_RETRIEVAL_INTERVAL"])
        if "MULTIVERSE_ROLLOUT_QUERY_BUDGET" in os.environ:
            rollout["on_demand_query_budget"] = int(os.environ["MULTIVERSE_ROLLOUT_QUERY_BUDGET"])
        if "MULTIVERSE_ROLLOUT_MIN_INTERVAL" in os.environ:
            rollout["on_demand_min_interval"] = int(os.environ["MULTIVERSE_ROLLOUT_MIN_INTERVAL"])

        # MCTS overrides
        if "mcts" not in result:
            result["mcts"] = {}
        mcts = result["mcts"]

        if "MULTIVERSE_MCTS_NUM_SIMULATIONS" in os.environ:
            mcts["num_simulations"] = int(os.environ["MULTIVERSE_MCTS_NUM_SIMULATIONS"])
        if "MULTIVERSE_MCTS_MAX_DEPTH" in os.environ:
            mcts["max_depth"] = int(os.environ["MULTIVERSE_MCTS_MAX_DEPTH"])
        if "MULTIVERSE_MCTS_C_PUCT" in os.environ:
            mcts["c_puct"] = float(os.environ["MULTIVERSE_MCTS_C_PUCT"])
        if "MULTIVERSE_MCTS_TRANSPOSITION_MAX_ENTRIES" in os.environ:
            mcts["transposition_max_entries"] = int(os.environ["MULTIVERSE_MCTS_TRANSPOSITION_MAX_ENTRIES"])

        # Curriculum overrides
        if "curriculum" not in result:
            result["curriculum"] = {}
        curriculum = result["curriculum"]

        if "MULTIVERSE_CURRICULUM_ENABLED" in os.environ:
            curriculum["enabled"] = bool(int(os.environ["MULTIVERSE_CURRICULUM_ENABLED"]))
        if "MULTIVERSE_CURRICULUM_PLATEAU_WINDOW" in os.environ:
            curriculum["plateau_window"] = int(os.environ["MULTIVERSE_CURRICULUM_PLATEAU_WINDOW"])
        if "MULTIVERSE_CURRICULUM_STEP_SIZE" in os.environ:
            curriculum["step_size"] = float(os.environ["MULTIVERSE_CURRICULUM_STEP_SIZE"])
        if "MULTIVERSE_CURRICULUM_COLLAPSE_THRESHOLD" in os.environ:
            curriculum["collapse_threshold"] = float(os.environ["MULTIVERSE_CURRICULUM_COLLAPSE_THRESHOLD"])
        if "MULTIVERSE_CURRICULUM_MAX_NOISE" in os.environ:
            curriculum["max_noise"] = float(os.environ["MULTIVERSE_CURRICULUM_MAX_NOISE"])
        if "MULTIVERSE_CURRICULUM_MAX_PARTIAL_OBS" in os.environ:
            curriculum["max_partial_obs"] = float(os.environ["MULTIVERSE_CURRICULUM_MAX_PARTIAL_OBS"])
        if "MULTIVERSE_CURRICULUM_MAX_DISTRACTORS" in os.environ:
            curriculum["max_distractors"] = int(os.environ["MULTIVERSE_CURRICULUM_MAX_DISTRACTORS"])

        # Memory overrides
        if "memory" not in result:
            result["memory"] = {}
        memory = result["memory"]

        if "MULTIVERSE_MEMORY_LOCK_TIMEOUT" in os.environ:
            memory["lock_timeout"] = int(os.environ["MULTIVERSE_MEMORY_LOCK_TIMEOUT"])
        if "MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD" in os.environ:
            memory["delta_merge_threshold"] = int(os.environ["MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD"])
        if "MULTIVERSE_MEMORY_QUERY_CACHE_SIZE" in os.environ:
            memory["query_cache_size"] = int(os.environ["MULTIVERSE_MEMORY_QUERY_CACHE_SIZE"])
        if "MULTIVERSE_MEMORY_QUERY_CACHE_TTL_MS" in os.environ:
            memory["query_cache_ttl_ms"] = int(os.environ["MULTIVERSE_MEMORY_QUERY_CACHE_TTL_MS"])
        if "MULTIVERSE_SIM_USE_ANN" in os.environ:
            memory["use_ann"] = bool(int(os.environ["MULTIVERSE_SIM_USE_ANN"]))
        if "MULTIVERSE_SIM_ANN_CANDIDATE_COUNT" in os.environ:
            memory["ann_candidate_count"] = int(os.environ["MULTIVERSE_SIM_ANN_CANDIDATE_COUNT"])
        if "MULTIVERSE_SIM_CACHE_LIMIT" in os.environ:
            memory["cache_limit"] = int(os.environ["MULTIVERSE_SIM_CACHE_LIMIT"])

        # Safe executor overrides
        if "safe_executor" not in result:
            result["safe_executor"] = {}
        safe = result["safe_executor"]

        if "MULTIVERSE_SAFE_ENABLED" in os.environ:
            safe["enabled"] = bool(int(os.environ["MULTIVERSE_SAFE_ENABLED"]))
        if "MULTIVERSE_SAFE_DANGER_THRESHOLD" in os.environ:
            safe["danger_threshold"] = float(os.environ["MULTIVERSE_SAFE_DANGER_THRESHOLD"])
        if "MULTIVERSE_SAFE_MIN_CONFIDENCE" in os.environ:
            safe["min_action_confidence"] = float(os.environ["MULTIVERSE_SAFE_MIN_CONFIDENCE"])

        return result

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> MultiverseConfig:
        """
        Load configuration from file + environment variables.

        Priority order:
        1. Environment variables (MULTIVERSE_*)
        2. YAML/JSON file (if provided or found)
        3. Default values

        Args:
            config_path: Optional path to config file

        Returns:
            Validated MultiverseConfig instance

        Raises:
            FileNotFoundError: If explicit config_path provided but not found
            ValidationError: If configuration is invalid
        """
        # Start with empty config (will use defaults)
        config_dict: Dict[str, Any] = {}

        # Try to load from file
        file_path = cls.find_config_file(config_path)
        if file_path:
            suffix = file_path.suffix.lower()
            if suffix in [".yaml", ".yml"]:
                config_dict = cls.load_yaml(file_path)
            elif suffix == ".json":
                config_dict = cls.load_json(file_path)
            else:
                raise ValueError(f"Unsupported config file type: {suffix}")

        # Apply environment variable overrides
        config_dict = cls.apply_env_overrides(config_dict)

        # Validate and return
        return MultiverseConfig(**config_dict)


def load_config(path: Optional[str] = None) -> MultiverseConfig:
    """
    Convenience function to load configuration.

    Args:
        path: Optional path to config file

    Returns:
        Validated MultiverseConfig instance
    """
    return ConfigLoader.load(path)


def save_config(config: MultiverseConfig, path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        path: Output file path (.yaml or .json)
    """
    output_path = Path(path)
    config_dict = config.model_dump(exclude_none=True)

    if output_path.suffix.lower() in [".yaml", ".yml"]:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    elif output_path.suffix.lower() == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
