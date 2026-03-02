"""
tests/test_hot_reload.py

Test hot-reload configuration system with file watching.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
_THIS_DIR = Path(__file__).parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_basic_reload():
    """Test basic manual configuration reload."""
    from core.config_watcher import ConfigWatcher

    # Create temporary config
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
        # Create watcher (no auto-reload for manual testing)
        watcher = ConfigWatcher(config_path=temp_path, auto_reload=False)
        watcher.start()

        # Check initial config
        config = watcher.get_config()
        assert config.parallel.num_workers == 4
        assert config.mcts.num_simulations == 96

        # Modify config file
        new_yaml = """
parallel:
  num_workers: 16

mcts:
  num_simulations: 200
"""
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(new_yaml)

        # Manual reload
        success = watcher.reload(force=True)
        assert success, "Reload should succeed"

        # Check updated config
        config = watcher.get_config()
        assert config.parallel.num_workers == 16, f"Expected 16 workers, got {config.parallel.num_workers}"
        assert config.mcts.num_simulations == 200, f"Expected 200 sims, got {config.mcts.num_simulations}"

        # Check stats
        stats = watcher.get_stats()
        assert stats["reload_count"] == 1
        assert stats["reload_errors"] == 0

        print("[PASS] Basic manual reload working")

    finally:
        watcher.stop()
        os.unlink(temp_path)


def test_validation_prevents_bad_config():
    """Test that validation prevents invalid configs from being applied."""
    from core.config_watcher import ConfigWatcher

    # Create valid initial config
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
        watcher = ConfigWatcher(config_path=temp_path, auto_reload=False)
        watcher.start()

        # Check initial config
        config = watcher.get_config()
        assert config.parallel.num_workers == 4

        # Write invalid config (num_workers too high)
        bad_yaml = """
parallel:
  num_workers: 999999  # Max is 256

mcts:
  num_simulations: 96
"""
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(bad_yaml)

        # Try to reload (should fail)
        success = watcher.reload(force=True)
        assert not success, "Reload should fail for invalid config"

        # Config should remain unchanged (rollback)
        config = watcher.get_config()
        assert config.parallel.num_workers == 4, "Config should not change on validation error"

        # Check stats
        stats = watcher.get_stats()
        assert stats["reload_errors"] >= 1
        assert stats["last_error"] is not None

        print("[PASS] Validation prevents bad config (rollback working)")

    finally:
        watcher.stop()
        os.unlink(temp_path)


def test_change_detection():
    """Test that we correctly detect which sections changed."""
    from core.config_watcher import ConfigWatcher

    yaml_content = """
parallel:
  num_workers: 4

mcts:
  num_simulations: 96

curriculum:
  enabled: true
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        watcher = ConfigWatcher(config_path=temp_path, auto_reload=False)
        watcher.start()

        changed_sections = []

        def on_change(event):
            changed_sections.extend(event.changed_sections)

        watcher.add_listener(on_change)

        # Change only MCTS section
        new_yaml = """
parallel:
  num_workers: 4  # Unchanged

mcts:
  num_simulations: 200  # Changed

curriculum:
  enabled: true  # Unchanged
"""
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(new_yaml)

        watcher.reload(force=True)

        # Should only detect MCTS change
        assert "mcts" in changed_sections, f"Expected 'mcts' in {changed_sections}"
        assert "parallel" not in changed_sections, "parallel should not be in changed_sections"
        assert "curriculum" not in changed_sections, "curriculum should not be in changed_sections"

        print("[PASS] Change detection working correctly")

    finally:
        watcher.stop()
        os.unlink(temp_path)


def test_listener_notifications():
    """Test that listeners are notified of config changes."""
    from core.config_watcher import ConfigWatcher

    yaml_content = """
parallel:
  num_workers: 4
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        watcher = ConfigWatcher(config_path=temp_path, auto_reload=False)
        watcher.start()

        # Track listener invocations
        listener_calls = []

        def listener(event):
            listener_calls.append({
                "old_workers": event.old_config.parallel.num_workers,
                "new_workers": event.new_config.parallel.num_workers,
                "changed": event.changed_sections,
            })

        watcher.add_listener(listener)

        # Change config
        new_yaml = """
parallel:
  num_workers: 16
"""
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(new_yaml)

        watcher.reload(force=True)

        # Listener should have been called
        assert len(listener_calls) == 1, f"Expected 1 call, got {len(listener_calls)}"
        assert listener_calls[0]["old_workers"] == 4
        assert listener_calls[0]["new_workers"] == 16
        assert "parallel" in listener_calls[0]["changed"]

        print("[PASS] Listener notifications working")

    finally:
        watcher.stop()
        os.unlink(temp_path)


def test_debounce():
    """Test that rapid changes are debounced."""
    from core.config_watcher import ConfigWatcher

    yaml_content = """
parallel:
  num_workers: 4
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        watcher = ConfigWatcher(config_path=temp_path, auto_reload=False, debounce_seconds=0.5)
        watcher.start()

        # First reload
        success1 = watcher.reload(force=False)
        assert not success1, "First reload should fail (config unchanged)"

        # Immediate second reload (should be debounced)
        success2 = watcher.reload(force=False)
        assert not success2, "Second reload should be debounced"

        # Wait for debounce period
        time.sleep(0.6)

        # Modify config
        new_yaml = """
parallel:
  num_workers: 8
"""
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(new_yaml)

        # Now reload should work
        success3 = watcher.reload(force=False)
        assert success3, "Reload should succeed after debounce period"

        config = watcher.get_config()
        assert config.parallel.num_workers == 8

        print("[PASS] Debounce working correctly")

    finally:
        watcher.stop()
        os.unlink(temp_path)


def test_auto_reload():
    """Test automatic reload on file changes (file watcher)."""
    from core.config_watcher import ConfigWatcher

    yaml_content = """
parallel:
  num_workers: 4
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Enable auto-reload with short debounce
        watcher = ConfigWatcher(config_path=temp_path, auto_reload=True, debounce_seconds=0.5)
        watcher.start()

        # Track changes
        reload_events = []

        def on_reload(event):
            reload_events.append(event.new_config.parallel.num_workers)

        watcher.add_listener(on_reload)

        # Initial config
        config = watcher.get_config()
        assert config.parallel.num_workers == 4

        # Modify file (should trigger auto-reload)
        new_yaml = """
parallel:
  num_workers: 16
"""
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(new_yaml)

        # Wait for file watcher + debounce + reload
        time.sleep(2.0)

        # Config should be updated
        config = watcher.get_config()
        assert config.parallel.num_workers == 16, f"Expected auto-reload to 16, got {config.parallel.num_workers}"

        # Listener should have been notified
        assert len(reload_events) >= 1, f"Expected reload event, got {len(reload_events)}"
        assert reload_events[-1] == 16

        print("[PASS] Auto-reload on file changes working")

    finally:
        watcher.stop()
        os.unlink(temp_path)


def main():
    """Run all hot-reload tests."""
    print("\n" + "="*60)
    print("Testing Hot-Reload Configuration System")
    print("="*60 + "\n")

    tests = [
        test_basic_reload,
        test_validation_prevents_bad_config,
        test_change_detection,
        test_listener_notifications,
        test_debounce,
        test_auto_reload,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    if failed > 0:
        sys.exit(1)
    else:
        print("All hot-reload tests passed!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
