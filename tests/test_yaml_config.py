"""
tests/test_yaml_config.py

Test YAML/JSON configuration loading with validation and priority.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
_THIS_DIR = Path(__file__).parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_yaml_config_loading():
    """Test loading configuration from YAML file."""
    from core.config_loader import ConfigLoader

    # Create temporary YAML config
    yaml_content = """
parallel:
  num_workers: 16
  max_worker_timeout_s: 7200

mcts:
  num_simulations: 128
  max_depth: 20

curriculum:
  enabled: false
  plateau_window: 10
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = ConfigLoader.load(temp_path)

        assert config.parallel.num_workers == 16
        assert config.parallel.max_worker_timeout_s == 7200
        assert config.mcts.num_simulations == 128
        assert config.mcts.max_depth == 20
        assert config.curriculum.enabled == False
        assert config.curriculum.plateau_window == 10

        # Check defaults are used for unspecified values
        assert config.rollout.retrieval_interval == 10  # default
        assert config.memory.use_ann == True  # default

        print("[PASS] YAML config loading working")
    finally:
        os.unlink(temp_path)


def test_json_config_loading():
    """Test loading configuration from JSON file."""
    from core.config_loader import ConfigLoader

    # Create temporary JSON config
    json_content = {
        "parallel": {
            "num_workers": 8,
            "use_ray": True
        },
        "rollout": {
            "retrieval_interval": 20,
            "on_demand_query_budget": 5
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(json_content, f)
        temp_path = f.name

    try:
        config = ConfigLoader.load(temp_path)

        assert config.parallel.num_workers == 8
        assert config.parallel.use_ray == True
        assert config.rollout.retrieval_interval == 20
        assert config.rollout.on_demand_query_budget == 5

        print("[PASS] JSON config loading working")
    finally:
        os.unlink(temp_path)


def test_env_var_override():
    """Test that environment variables override YAML config."""
    from core.config_loader import ConfigLoader

    # Create YAML with base values
    yaml_content = """
parallel:
  num_workers: 4

mcts:
  num_simulations: 96
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Set environment variables to override
        os.environ["MULTIVERSE_PARALLEL_NUM_WORKERS"] = "32"
        os.environ["MULTIVERSE_MCTS_NUM_SIMULATIONS"] = "256"

        config = ConfigLoader.load(temp_path)

        # Environment variables should override YAML
        assert config.parallel.num_workers == 32, f"Expected 32, got {config.parallel.num_workers}"
        assert config.mcts.num_simulations == 256, f"Expected 256, got {config.mcts.num_simulations}"

        print("[PASS] Environment variable override working")
    finally:
        os.unlink(temp_path)
        del os.environ["MULTIVERSE_PARALLEL_NUM_WORKERS"]
        del os.environ["MULTIVERSE_MCTS_NUM_SIMULATIONS"]


def test_validation_errors():
    """Test that invalid config raises validation errors."""
    from core.config_loader import ConfigLoader
    from pydantic import ValidationError

    # Invalid: num_workers too high
    yaml_content = """
parallel:
  num_workers: 999999  # Max is 256
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        try:
            config = ConfigLoader.load(temp_path)
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            # Expected error
            assert "num_workers" in str(e).lower()
            print("[PASS] Validation error detection working")
    finally:
        os.unlink(temp_path)


def test_example_configs():
    """Test that example YAML configs are valid."""
    from core.config_loader import ConfigLoader

    example_configs = [
        _ROOT / "configs" / "multiverse.dev.yaml",
        _ROOT / "configs" / "multiverse.staging.yaml",
        _ROOT / "configs" / "multiverse.prod.yaml",
    ]

    for config_path in example_configs:
        if not config_path.exists():
            print(f"[SKIP] {config_path.name} not found")
            continue

        try:
            config = ConfigLoader.load(str(config_path))

            # Basic sanity checks
            assert config.parallel.num_workers >= 1
            assert config.mcts.num_simulations >= 1
            assert 0.0 <= config.curriculum.collapse_threshold <= 1.0

            print(f"[PASS] {config_path.name} is valid")
        except Exception as e:
            print(f"[FAIL] {config_path.name} failed: {e}")
            raise


def test_save_and_load_roundtrip():
    """Test saving config to YAML and loading it back."""
    from core.config_loader import ConfigLoader, save_config
    from core.config_schema import MultiverseConfig

    # Create config programmatically
    config = MultiverseConfig()
    config.parallel.num_workers = 16
    config.mcts.num_simulations = 200
    config.curriculum.enabled = False

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        temp_path = f.name

    try:
        # Save to YAML
        save_config(config, temp_path)

        # Load it back
        loaded_config = ConfigLoader.load(temp_path)

        # Verify values match
        assert loaded_config.parallel.num_workers == 16
        assert loaded_config.mcts.num_simulations == 200
        assert loaded_config.curriculum.enabled == False

        print("[PASS] Save and load roundtrip working")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_partial_config():
    """Test that partial configs work (unspecified fields use defaults)."""
    from core.config_loader import ConfigLoader

    # Only specify one section
    yaml_content = """
mcts:
  num_simulations: 200
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = ConfigLoader.load(temp_path)

        # Specified value
        assert config.mcts.num_simulations == 200

        # Defaults for other sections
        assert config.parallel.num_workers == 4  # default
        assert config.rollout.retrieval_interval == 10  # default
        assert config.curriculum.enabled == True  # default

        print("[PASS] Partial config working (defaults applied)")
    finally:
        os.unlink(temp_path)


def main():
    """Run all YAML configuration tests."""
    print("\n" + "="*60)
    print("Testing YAML/JSON Configuration System")
    print("="*60 + "\n")

    tests = [
        test_yaml_config_loading,
        test_json_config_loading,
        test_env_var_override,
        test_validation_errors,
        test_example_configs,
        test_save_and_load_roundtrip,
        test_partial_config,
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
        print("All YAML configuration tests passed!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
