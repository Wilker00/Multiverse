"""
memory/event_log.py

Append-only event logger for u.ai.

Writes StepEvent records to JSONL (one JSON object per line).
This is the MVP-friendly format: human readable, easy to debug, easy to parse.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.types import StepEvent


@dataclass
class EventLogConfig:
    root_dir: str = "runs"
    run_id: Optional[str] = None
    filename: str = "events.jsonl"
    metrics_filename: str = "metrics.jsonl"
    flush_every: int = 1  # flush each N events
    ensure_ascii: bool = False


class EventLogger:
    """
    Usage:
        logger = EventLogger(EventLogConfig(root_dir="runs", run_id=run.run_id))
        logger.open()
        logger.write(event)
        logger.write_metrics({"loss": 0.1})
        logger.close()
    """

    def __init__(self, config: EventLogConfig):
        self.config = config
        self._path: Optional[str] = None
        self._fh = None
        self._metrics_fh = None
        self._count = 0

    @property
    def path(self) -> Optional[str]:
        return self._path

    def open(self) -> str:
        run_id = self.config.run_id or "run_unknown"
        run_dir = os.path.join(self.config.root_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        self._path = os.path.join(run_dir, self.config.filename)
        self._fh = open(self._path, "a", encoding="utf-8")
        
        metrics_path = os.path.join(run_dir, self.config.metrics_filename)
        self._metrics_fh = open(metrics_path, "a", encoding="utf-8")
        
        self._count = 0
        return self._path

    def write(self, event: StepEvent) -> None:
        if self._fh is None:
            raise RuntimeError("EventLogger is not open(). Call open() first.")

        line = json.dumps(event.to_dict(), ensure_ascii=self.config.ensure_ascii)
        self._fh.write(line + "\n")
        self._count += 1

        if self.config.flush_every > 0 and (self._count % self.config.flush_every == 0):
            self._fh.flush()

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        if self._metrics_fh is None:
            raise RuntimeError("EventLogger is not open(). Call open() first.")
        
        line = json.dumps(metrics, ensure_ascii=self.config.ensure_ascii)
        self._metrics_fh.write(line + "\n")
        self._metrics_fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        if self._metrics_fh is not None:
            self._metrics_fh.flush()
            self._metrics_fh.close()
            self._metrics_fh = None

    def __enter__(self) -> "EventLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_on_step_writer(logger: EventLogger):
    """
    Adapter so you can pass this into rollout.run_episode(on_step=...).
    """
    def on_step(event: StepEvent) -> None:
        logger.write(event)
    return on_step
