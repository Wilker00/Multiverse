# Hot-Reload Configuration Guide

**Update configuration without restarting - Zero-downtime tuning**

---

## Overview

Hot-reload allows you to update Multiverse configuration while the system is running, without restart. Changes are validated before applying and automatically rolled back on errors.

### Key Features

- ✅ **Zero Downtime** - Update configs without stopping training
- ✅ **Automatic File Watching** - Detects YAML/JSON changes instantly
- ✅ **Validation First** - Invalid configs rejected before applying
- ✅ **Automatic Rollback** - Reverts to previous config on errors
- ✅ **Change Notifications** - Event system for config updates
- ✅ **Debounce Protection** - Handles rapid file edits gracefully

---

## Quick Start

### Basic Usage

```python
from core.config_watcher import get_global_watcher

# Start watching multiverse.yaml (auto-discovers)
watcher = get_global_watcher(auto_reload=True)

# Get current config (always up-to-date)
config = watcher.get_config()
print(f"Workers: {config.parallel.num_workers}")

# Edit multiverse.yaml externally (vim, VS Code, etc.)
# Config automatically reloads!

# Access updated config
config = watcher.get_config()  # Reflects new values
```

### Convenience Function

```python
from core.config_watcher import get_config

# Simple API - uses global watcher
config = get_config()  # Auto-starts watcher if needed
```

---

## How It Works

### Architecture

```
1. File Watcher (watchdog)
   ↓
2. Debounce Timer (1 second)
   ↓
3. Load & Validate (Pydantic)
   ↓
4. Apply or Rollback
   ↓
5. Notify Listeners
```

### Validation & Safety

**Before Applying:**
- ✅ YAML/JSON syntax validation
- ✅ Pydantic schema validation
- ✅ Range checking (e.g., num_workers: 1-256)
- ✅ Type checking (int, float, bool)

**On Validation Error:**
- ❌ Config change rejected
- 🔄 Previous config remains active
- 📝 Error logged with details
- 📊 Stats track reload failures

---

## Configuration API

### ConfigWatcher Class

```python
from core.config_watcher import ConfigWatcher

watcher = ConfigWatcher(
    config_path="multiverse.yaml",  # Optional (auto-discovers)
    auto_reload=True,               # Enable file watching
    debounce_seconds=1.0,           # Delay before reloading
)

watcher.start()  # Begin watching
```

### Methods

#### `get_config() -> MultiverseConfig`

Get current configuration (thread-safe).

```python
config = watcher.get_config()
workers = config.parallel.num_workers
```

#### `reload(force: bool = False) -> bool`

Manually reload configuration.

```python
# Respects debounce
success = watcher.reload()

# Bypass debounce
success = watcher.reload(force=True)
```

#### `add_listener(callback) -> None`

Register callback for config changes.

```python
def on_config_change(event):
    print(f"Config changed: {event.changed_sections}")
    print(f"Old workers: {event.old_config.parallel.num_workers}")
    print(f"New workers: {event.new_config.parallel.num_workers}")

watcher.add_listener(on_config_change)
```

#### `get_stats() -> Dict`

Get hot-reload statistics.

```python
stats = watcher.get_stats()
# {
#     "reload_count": 5,
#     "reload_errors": 1,
#     "last_reload_time": 1709332800.123,
#     "last_error": "validation error: ...",
#     "watching": True
# }
```

#### `stop() -> None`

Stop file watching.

```python
watcher.stop()
```

---

## Use Cases

### 1. Real-Time Tuning

Adjust hyperparameters during training:

```python
# Start training
watcher = get_global_watcher(auto_reload=True)

# In another terminal, edit multiverse.yaml:
# mcts:
#   num_simulations: 128  # Changed from 96

# Next MCTS call uses updated value automatically
```

### 2. A/B Testing

Switch configurations dynamically:

```python
# multiverse.yaml
parallel:
  num_workers: 8

# Edit to test different settings:
parallel:
  num_workers: 16  # Immediately takes effect
```

### 3. Production Rollout

Gradually roll out config changes:

```bash
# Deploy new config
cp multiverse.prod-v2.yaml multiverse.yaml

# Watch logs - if errors, rollback:
cp multiverse.prod-v1.yaml multiverse.yaml  # Instant rollback
```

### 4. Event-Driven Updates

React to configuration changes:

```python
def on_worker_count_change(event):
    if "parallel" in event.changed_sections:
        old_workers = event.old_config.parallel.num_workers
        new_workers = event.new_config.parallel.num_workers

        if new_workers > old_workers:
            logger.info(f"Scaling up: {old_workers} → {new_workers}")
            # Trigger worker pool resize
        elif new_workers < old_workers:
            logger.info(f"Scaling down: {old_workers} → {new_workers}")
            # Gracefully shutdown excess workers

watcher.add_listener(on_worker_count_change)
```

---

## Advanced Features

### Manual vs Automatic Reload

```python
# Automatic (file watching enabled)
watcher = ConfigWatcher(auto_reload=True)
watcher.start()
# Reloads automatically on file changes

# Manual (for testing or controlled updates)
watcher = ConfigWatcher(auto_reload=False)
watcher.start()
# Call watcher.reload() manually when needed
```

### Debounce Configuration

Prevents reload storms from rapid edits:

```python
# Short debounce (fast feedback)
watcher = ConfigWatcher(debounce_seconds=0.5)

# Long debounce (avoid reload storms)
watcher = ConfigWatcher(debounce_seconds=5.0)
```

### Change Detection

Only reload when config actually changes:

```python
def on_change(event):
    # event.changed_sections = ["mcts", "parallel"]

    if "mcts" in event.changed_sections:
        # MCTS config changed - restart search engine
        pass

    if "memory" in event.changed_sections:
        # Memory config changed - rebuild cache
        pass

watcher.add_listener(on_change)
```

### Global Singleton Pattern

```python
# First call creates and starts watcher
watcher1 = get_global_watcher()

# Subsequent calls return same instance
watcher2 = get_global_watcher()
assert watcher1 is watcher2  # Same object

# Convenience function uses global watcher
config = get_config()  # Uses watcher1/watcher2
```

---

## Error Handling

### Validation Errors

```python
# multiverse.yaml - INVALID
parallel:
  num_workers: 999999  # Exceeds max (256)

# Reload attempt logs error:
# ERROR: Config reload failed: validation error
#   parallel.num_workers
#     Input should be less than or equal to 256

# Previous config remains active (safe rollback)
config = watcher.get_config()
assert config.parallel.num_workers == 8  # Old value preserved
```

### YAML Syntax Errors

```python
# multiverse.yaml - INVALID SYNTAX
parallel:
num_workers: 8  # Missing indentation

# Reload attempt logs error:
# ERROR: Config reload failed: yaml.scanner.ScannerError
#   mapping values are not allowed here

# Previous config remains active
```

### Missing Files

```python
watcher = ConfigWatcher(config_path="missing.yaml")
# Raises: FileNotFoundError: Config file not found: missing.yaml
```

---

## Best Practices

### 1. Use Version Control

Track config changes in Git:

```bash
# Edit config
vim multiverse.yaml

# Commit changes
git add multiverse.yaml
git commit -m "config: increase MCTS simulations to 128"

# Rollback if needed
git revert HEAD
```

### 2. Test Changes Locally

```bash
# Development
cp multiverse.yaml multiverse.yaml.backup
vim multiverse.yaml  # Make changes

# If errors:
mv multiverse.yaml.backup multiverse.yaml  # Instant rollback
```

### 3. Monitor Reload Stats

```python
import time

while True:
    stats = watcher.get_stats()
    if stats["reload_errors"] > 0:
        logger.warning(f"Config reload errors: {stats['last_error']}")

    time.sleep(60)
```

### 4. Graceful Component Updates

```python
def on_mcts_change(event):
    if "mcts" in event.changed_sections:
        # Don't interrupt ongoing search
        # Wait for current search to complete
        wait_for_mcts_idle()

        # Then apply new config
        reinitialize_mcts_with_new_config(event.new_config.mcts)

watcher.add_listener(on_mcts_change)
```

### 5. Audit Config Changes

```python
def audit_log(event):
    logger.info(f"Config changed at {event.timestamp}")
    logger.info(f"Sections: {event.changed_sections}")

    # Log to external audit system
    audit_system.log({
        "timestamp": event.timestamp,
        "changed_sections": event.changed_sections,
        "user": os.environ.get("USER"),
        "hostname": socket.gethostname(),
    })

watcher.add_listener(audit_log)
```

---

## Limitations

### What Can Be Hot-Reloaded

✅ **Safe to Hot-Reload:**
- Worker counts (parallel.num_workers)
- MCTS simulations
- Curriculum settings
- Memory cache sizes
- Safety thresholds

❌ **Not Hot-Reloadable** (require restart):
- Database connections (SQLite, Pinecone)
- Model architectures (neural net layers)
- Major algorithm changes (Q → DT)

### Platform Differences

**Linux/macOS:**
- File watching uses inotify/FSEvents (efficient)
- Instant change detection

**Windows:**
- File watching uses ReadDirectoryChangesW
- Slightly higher latency (~100-500ms)

---

## Troubleshooting

### Config Not Reloading

**Problem**: Edit multiverse.yaml but changes not applied

**Solutions**:
1. Check watcher is running: `watcher.get_stats()["watching"]`
2. Verify file path: `watcher.config_path`
3. Check logs for validation errors
4. Try manual reload: `watcher.reload(force=True)`

### Reload Errors

**Problem**: Reload count increases but config unchanged

**Solutions**:
1. Check stats: `watcher.get_stats()["last_error"]`
2. Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('multiverse.yaml'))"`
3. Check Pydantic validation: `from core.config_loader import load_config; load_config()`

### File Watcher Not Starting

**Problem**: `get_stats()["watching"] == False`

**Solutions**:
1. Ensure config file exists before calling `start()`
2. Check file permissions (readable)
3. Verify watchdog installed: `pip show watchdog`

### High CPU Usage

**Problem**: Watchdog consuming CPU

**Solutions**:
1. Increase debounce: `ConfigWatcher(debounce_seconds=5.0)`
2. Disable auto-reload for testing: `auto_reload=False`
3. Watch specific file, not entire directory

---

## Testing

### Unit Tests

```bash
python tests/test_hot_reload.py
# 6 passed, 0 failed
```

### Manual Testing

```python
# Terminal 1: Start watcher
from core.config_watcher import get_global_watcher
watcher = get_global_watcher()

# Terminal 2: Edit config
vim multiverse.yaml  # Change num_workers: 8 → 16

# Terminal 1: Verify reload
config = watcher.get_config()
print(config.parallel.num_workers)  # Should be 16
```

---

## Integration with Multiverse

### Trainer Integration

```python
from orchestrator.trainer import Trainer
from core.config_watcher import get_global_watcher

# Start watcher
watcher = get_global_watcher()

# Use config in trainer
config = watcher.get_config()
trainer = Trainer(
    run_root="runs",
    num_workers=config.parallel.num_workers,
)

# Config updates automatically reflected in subsequent calls
```

### MCTS Integration

```python
from core.mcts_search import MCTSSearch
from core.config_watcher import get_config

def create_mcts(verse):
    config = get_config()  # Always current

    return MCTSSearch(
        verse=verse,
        config=MCTSConfig(
            num_simulations=config.mcts.num_simulations,
            max_depth=config.mcts.max_depth,
            c_puct=config.mcts.c_puct,
        )
    )
```

---

## See Also

- [YAML Configuration](YAML_CONFIGURATION.md) - Structured config format
- [Configuration Reference](CONFIGURATION.md) - All parameters
- [Setup Guide](SETUP.md) - Installation
- [Watchdog Documentation](https://python-watchdog.readthedocs.io/) - File watching library

---

**Last Updated**: 2026-03-02 (Phase 4 - Hot-Reload Configuration)
