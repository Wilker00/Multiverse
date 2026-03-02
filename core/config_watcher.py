"""
core/config_watcher.py

Hot-reload configuration system with file watching and validation.

Features:
- Watch YAML/JSON config files for changes
- Validate new config before applying
- Rollback on validation errors
- Event notifications for config changes
- Thread-safe concurrent access
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from core.config_loader import ConfigLoader
from core.config_schema import MultiverseConfig

logger = logging.getLogger(__name__)


class ConfigChangeEvent:
    """Event emitted when configuration changes."""

    def __init__(
        self,
        old_config: MultiverseConfig,
        new_config: MultiverseConfig,
        changed_sections: List[str],
        timestamp: float,
    ):
        self.old_config = old_config
        self.new_config = new_config
        self.changed_sections = changed_sections
        self.timestamp = timestamp


class ConfigWatcher:
    """
    Watch configuration files and hot-reload on changes.

    Features:
    - Validates new config before applying
    - Rolls back on validation errors
    - Notifies listeners of changes
    - Debounces rapid file changes
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        auto_reload: bool = True,
        debounce_seconds: float = 1.0,
    ):
        """
        Initialize configuration watcher.

        Args:
            config_path: Path to config file (None = auto-discover)
            auto_reload: Automatically reload on file changes
            debounce_seconds: Delay before reloading (debounce rapid changes)
        """
        self.config_path = config_path
        self.auto_reload = auto_reload
        self.debounce_seconds = debounce_seconds

        # Current configuration
        self._config: Optional[MultiverseConfig] = None
        self._config_lock = threading.RLock()

        # File watching
        self._observer: Optional[Observer] = None
        self._last_reload_time = 0.0
        self._reload_timer: Optional[threading.Timer] = None

        # Event listeners
        self._listeners: List[Callable[[ConfigChangeEvent], None]] = []

        # Statistics
        self._reload_count = 0
        self._reload_errors = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """
        Start watching configuration file for changes.

        Raises:
            FileNotFoundError: If config file not found
        """
        # Initial load
        self._config = ConfigLoader.load(self.config_path)

        if not self.auto_reload:
            return

        # Find config file path
        config_file = ConfigLoader.find_config_file(self.config_path)
        if not config_file:
            logger.warning("No config file found, hot-reload disabled")
            return

        # Watch the config file directory
        watch_dir = config_file.parent
        logger.info(f"Watching config directory: {watch_dir}")

        # Create file watcher
        event_handler = ConfigFileEventHandler(
            config_file=config_file,
            on_change=self._on_file_changed,
        )

        self._observer = Observer()
        self._observer.schedule(event_handler, str(watch_dir), recursive=False)
        self._observer.start()

        logger.info(f"Hot-reload started for: {config_file}")

    def stop(self) -> None:
        """Stop watching configuration file."""
        if self._reload_timer:
            self._reload_timer.cancel()
            self._reload_timer = None

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        logger.info("Hot-reload stopped")

    def get_config(self) -> MultiverseConfig:
        """
        Get current configuration (thread-safe).

        Returns:
            Current configuration instance
        """
        with self._config_lock:
            if self._config is None:
                raise RuntimeError("Config not loaded. Call start() first.")
            return self._config

    def reload(self, force: bool = False) -> bool:
        """
        Manually reload configuration.

        Args:
            force: Force reload even if debounce not expired

        Returns:
            True if reload succeeded, False otherwise
        """
        now = time.time()

        # Debounce check
        if not force and (now - self._last_reload_time) < self.debounce_seconds:
            logger.debug("Reload debounced (too soon)")
            return False

        try:
            with self._config_lock:
                old_config = self._config

                # Load and validate new config
                new_config = ConfigLoader.load(self.config_path)

                # Detect what changed
                changed_sections = self._detect_changes(old_config, new_config)

                if not changed_sections:
                    logger.debug("Config unchanged, skipping reload")
                    return False

                # Apply new config
                self._config = new_config
                self._last_reload_time = now
                self._reload_count += 1

                logger.info(f"Config reloaded successfully (changed: {', '.join(changed_sections)})")

                # Notify listeners
                event = ConfigChangeEvent(
                    old_config=old_config,
                    new_config=new_config,
                    changed_sections=changed_sections,
                    timestamp=now,
                )
                self._notify_listeners(event)

                return True

        except Exception as e:
            self._reload_errors += 1
            self._last_error = str(e)
            logger.error(f"Config reload failed: {e}", exc_info=True)
            return False

    def add_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> None:
        """
        Add listener for configuration changes.

        Args:
            listener: Callback function(event: ConfigChangeEvent) -> None
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> None:
        """Remove configuration change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get hot-reload statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "reload_count": self._reload_count,
            "reload_errors": self._reload_errors,
            "last_reload_time": self._last_reload_time,
            "last_error": self._last_error,
            "watching": self._observer is not None and self._observer.is_alive(),
        }

    def _on_file_changed(self, path: Path) -> None:
        """
        Handle file change event (internal).

        Args:
            path: Path to changed file
        """
        logger.debug(f"Config file changed: {path}")

        # Cancel pending reload timer
        if self._reload_timer:
            self._reload_timer.cancel()

        # Schedule debounced reload
        self._reload_timer = threading.Timer(
            self.debounce_seconds,
            self.reload,
        )
        self._reload_timer.daemon = True
        self._reload_timer.start()

    def _detect_changes(
        self,
        old_config: Optional[MultiverseConfig],
        new_config: MultiverseConfig,
    ) -> List[str]:
        """
        Detect which configuration sections changed.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            List of changed section names
        """
        if old_config is None:
            return ["all"]  # Initial load

        changed = []

        # Check each section
        if old_config.parallel != new_config.parallel:
            changed.append("parallel")
        if old_config.rollout != new_config.rollout:
            changed.append("rollout")
        if old_config.mcts != new_config.mcts:
            changed.append("mcts")
        if old_config.curriculum != new_config.curriculum:
            changed.append("curriculum")
        if old_config.memory != new_config.memory:
            changed.append("memory")
        if old_config.safe_executor != new_config.safe_executor:
            changed.append("safe_executor")

        return changed

    def _notify_listeners(self, event: ConfigChangeEvent) -> None:
        """
        Notify all listeners of configuration change.

        Args:
            event: Configuration change event
        """
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Listener error: {e}", exc_info=True)


class ConfigFileEventHandler(FileSystemEventHandler):
    """Watchdog event handler for config file changes."""

    def __init__(self, config_file: Path, on_change: Callable[[Path], None]):
        super().__init__()
        self.config_file = config_file
        self.on_change = on_change

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification event."""
        if event.is_directory:
            return

        # Check if the modified file is our config file
        modified_path = Path(event.src_path)
        if modified_path.resolve() == self.config_file.resolve():
            self.on_change(modified_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation event."""
        if event.is_directory:
            return

        created_path = Path(event.src_path)
        if created_path.resolve() == self.config_file.resolve():
            self.on_change(created_path)


# Global singleton instance
_global_watcher: Optional[ConfigWatcher] = None
_global_watcher_lock = threading.Lock()


def get_global_watcher(
    config_path: Optional[str] = None,
    auto_reload: bool = True,
    auto_start: bool = True,
) -> ConfigWatcher:
    """
    Get or create global configuration watcher.

    Args:
        config_path: Path to config file
        auto_reload: Enable auto-reload on changes
        auto_start: Automatically start watching

    Returns:
        Global ConfigWatcher instance
    """
    global _global_watcher

    with _global_watcher_lock:
        if _global_watcher is None:
            _global_watcher = ConfigWatcher(
                config_path=config_path,
                auto_reload=auto_reload,
            )
            if auto_start:
                _global_watcher.start()

        return _global_watcher


def get_config() -> MultiverseConfig:
    """
    Get current global configuration.

    Returns:
        Current configuration instance
    """
    watcher = get_global_watcher()
    return watcher.get_config()
