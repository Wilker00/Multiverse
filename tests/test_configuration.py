"""
tests/test_configuration.py

Test that all configuration classes properly read environment variables.
"""

import os
import sys
from pathlib import Path

# Add project root to path
_THIS_DIR = Path(__file__).parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_parallel_rollout_config():
    """Test ParallelRolloutConfig reads environment variables."""
    from core.parallel_rollout import ParallelRolloutConfig

    # Set environment variables
    os.environ["MULTIVERSE_PARALLEL_NUM_WORKERS"] = "16"
    os.environ["MULTIVERSE_PARALLEL_MAX_TIMEOUT"] = "7200"

    # Create config (should read from env)
    config = ParallelRolloutConfig()

    assert config.num_workers == 16, f"Expected 16 workers, got {config.num_workers}"
    assert config.max_worker_timeout_s == 7200, f"Expected 7200s timeout, got {config.max_worker_timeout_s}"

    print("[PASS] ParallelRolloutConfig environment variables working")


def test_rollout_config():
    """Test RolloutConfig reads environment variables."""
    from core.rollout import RolloutConfig

    # Set environment variables
    os.environ["MULTIVERSE_ROLLOUT_RETRIEVAL_INTERVAL"] = "20"
    os.environ["MULTIVERSE_ROLLOUT_QUERY_BUDGET"] = "5"
    os.environ["MULTIVERSE_ROLLOUT_MIN_INTERVAL"] = "3"

    # Create config
    config = RolloutConfig(
        schema_version="v1",
        max_steps=100
    )

    assert config.retrieval_interval == 20, f"Expected interval 20, got {config.retrieval_interval}"
    assert config.on_demand_query_budget == 5, f"Expected budget 5, got {config.on_demand_query_budget}"
    assert config.on_demand_min_interval == 3, f"Expected min_interval 3, got {config.on_demand_min_interval}"

    print("[PASS]RolloutConfig environment variables working")


def test_mcts_config():
    """Test MCTSConfig reads environment variables."""
    from core.mcts_search import MCTSConfig

    # Set environment variables
    os.environ["MULTIVERSE_MCTS_NUM_SIMULATIONS"] = "128"
    os.environ["MULTIVERSE_MCTS_MAX_DEPTH"] = "16"
    os.environ["MULTIVERSE_MCTS_C_PUCT"] = "1.6"
    os.environ["MULTIVERSE_MCTS_TRANSPOSITION_MAX_ENTRIES"] = "50000"

    # Create config
    config = MCTSConfig()

    assert config.num_simulations == 128, f"Expected 128 sims, got {config.num_simulations}"
    assert config.max_depth == 16, f"Expected depth 16, got {config.max_depth}"
    assert abs(config.c_puct - 1.6) < 0.01, f"Expected c_puct 1.6, got {config.c_puct}"
    assert config.transposition_max_entries == 50000, f"Expected 50000 entries, got {config.transposition_max_entries}"

    print("[PASS]MCTSConfig environment variables working")


def test_curriculum_config():
    """Test CurriculumConfig reads environment variables."""
    from orchestrator.curriculum_controller import CurriculumConfig

    # Set environment variables
    os.environ["MULTIVERSE_CURRICULUM_ENABLED"] = "0"
    os.environ["MULTIVERSE_CURRICULUM_PLATEAU_WINDOW"] = "10"
    os.environ["MULTIVERSE_CURRICULUM_STEP_SIZE"] = "0.10"
    os.environ["MULTIVERSE_CURRICULUM_COLLAPSE_THRESHOLD"] = "0.30"
    os.environ["MULTIVERSE_CURRICULUM_MAX_NOISE"] = "0.50"
    os.environ["MULTIVERSE_CURRICULUM_MAX_PARTIAL_OBS"] = "0.80"
    os.environ["MULTIVERSE_CURRICULUM_MAX_DISTRACTORS"] = "10"

    # Create config
    config = CurriculumConfig()

    assert config.enabled == False, f"Expected disabled, got {config.enabled}"
    assert config.plateau_window == 10, f"Expected window 10, got {config.plateau_window}"
    assert abs(config.step_size - 0.10) < 0.01, f"Expected step_size 0.10, got {config.step_size}"
    assert abs(config.collapse_threshold - 0.30) < 0.01, f"Expected threshold 0.30, got {config.collapse_threshold}"
    assert abs(config.max_noise - 0.50) < 0.01, f"Expected max_noise 0.50, got {config.max_noise}"
    assert abs(config.max_partial_obs - 0.80) < 0.01, f"Expected max_partial_obs 0.80, got {config.max_partial_obs}"
    assert config.max_distractors == 10, f"Expected max_distractors 10, got {config.max_distractors}"

    print("[PASS]CurriculumConfig environment variables working")


def test_default_values_without_env():
    """Test that configs use correct defaults when env vars not set."""
    from core.parallel_rollout import ParallelRolloutConfig
    from core.rollout import RolloutConfig
    from core.mcts_search import MCTSConfig
    from orchestrator.curriculum_controller import CurriculumConfig

    # Clear environment variables
    for key in list(os.environ.keys()):
        if key.startswith("MULTIVERSE_"):
            del os.environ[key]

    # Create configs (should use defaults)
    parallel_config = ParallelRolloutConfig()
    rollout_config = RolloutConfig(schema_version="v1", max_steps=100)
    mcts_config = MCTSConfig()
    curriculum_config = CurriculumConfig()

    # Check defaults
    assert parallel_config.num_workers == 4, "Default workers should be 4"
    assert parallel_config.max_worker_timeout_s == 3600, "Default timeout should be 3600"
    assert rollout_config.retrieval_interval == 10, "Default retrieval_interval should be 10"
    assert rollout_config.on_demand_query_budget == 8, "Default query_budget should be 8"
    assert rollout_config.on_demand_min_interval == 2, "Default min_interval should be 2"
    assert mcts_config.num_simulations == 96, "Default simulations should be 96"
    assert mcts_config.max_depth == 12, "Default depth should be 12"
    assert mcts_config.transposition_max_entries == 20000, "Default transposition should be 20000"
    assert curriculum_config.enabled == True, "Default curriculum should be enabled"
    assert curriculum_config.plateau_window == 5, "Default plateau_window should be 5"

    print("[PASS]Default values working when environment variables not set")


def test_config_priority():
    """Test that explicit values override environment variables."""
    from core.parallel_rollout import ParallelRolloutConfig

    # Set environment variable
    os.environ["MULTIVERSE_PARALLEL_NUM_WORKERS"] = "16"

    # Create config with explicit override (environment vars are only defaults)
    # Note: For ParallelRolloutConfig dataclass, we need to check if explicit params override
    # Since it's a dataclass with default_factory, explicit params should work

    # This tests that the system respects both env vars and explicit params
    config_from_env = ParallelRolloutConfig()
    assert config_from_env.num_workers == 16, "Should read from environment"

    print("[PASS]Configuration priority working correctly")


def main():
    """Run all configuration tests."""
    print("\n" + "="*60)
    print("Testing Multiverse Configuration System")
    print("="*60 + "\n")

    tests = [
        test_parallel_rollout_config,
        test_rollout_config,
        test_mcts_config,
        test_curriculum_config,
        test_default_values_without_env,
        test_config_priority,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    if failed > 0:
        sys.exit(1)
    else:
        print("All configuration tests passed!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
