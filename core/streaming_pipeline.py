"""
core/streaming_pipeline.py

Local-first streaming pipeline with optional Kafka backend.
If Kafka is unavailable, uses an in-process topic bus so code still runs.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_THIS_DIR)
    # Avoid stdlib shadowing by `core/types.py` when executing as script:
    #   python core/streaming_pipeline.py
    if sys.path and os.path.abspath(sys.path[0]) == _THIS_DIR:
        sys.path.pop(0)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import json
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:
    from kafka import KafkaConsumer, KafkaProducer  # type: ignore
except Exception:  # pragma: no cover
    KafkaConsumer = None
    KafkaProducer = None


class ExperienceSerializer:
    @staticmethod
    def serialize(payload: Dict[str, Any]) -> bytes:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")

    @staticmethod
    def deserialize(data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode("utf-8"))


class _InMemoryBus:
    def __init__(self, max_queue_size: int = 20000, overflow_policy: str = "drop_oldest"):
        self.max_queue_size = max(1, int(max_queue_size))
        policy = str(overflow_policy).strip().lower()
        if policy not in ("drop_oldest", "drop_newest"):
            policy = "drop_oldest"
        self.overflow_policy = policy
        self._topics: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}
        self._lock = threading.Lock()
        self._published_by_topic: Dict[str, int] = {}
        self._dropped_by_topic: Dict[str, int] = {}

    def put(self, topic: str, value: Dict[str, Any]) -> None:
        with self._lock:
            q = self._topics.setdefault(topic, queue.Queue(maxsize=self.max_queue_size))
            self._published_by_topic[topic] = int(self._published_by_topic.get(topic, 0)) + 1
        try:
            q.put_nowait(value)
            return
        except queue.Full:
            pass

        dropped = False
        if self.overflow_policy == "drop_oldest":
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                dropped = True
            try:
                q.put_nowait(value)
            except queue.Full:
                dropped = True
        else:
            dropped = True

        if dropped:
            with self._lock:
                self._dropped_by_topic[topic] = int(self._dropped_by_topic.get(topic, 0)) + 1

    def get(self, topic: str, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
        with self._lock:
            q = self._topics.setdefault(topic, queue.Queue(maxsize=self.max_queue_size))
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            queue_sizes = {k: int(v.qsize()) for k, v in self._topics.items()}
            return {
                "max_queue_size": int(self.max_queue_size),
                "overflow_policy": str(self.overflow_policy),
                "queue_sizes": queue_sizes,
                "published_by_topic": dict(self._published_by_topic),
                "dropped_by_topic": dict(self._dropped_by_topic),
            }


def _read_local_bus_max_queue() -> int:
    raw = os.environ.get("MULTIVERSE_LOCAL_BUS_MAX_QUEUE", "20000")
    try:
        return int(raw)
    except Exception:
        return 20000


def _read_local_bus_overflow_policy() -> str:
    return str(os.environ.get("MULTIVERSE_LOCAL_BUS_OVERFLOW", "drop_oldest"))


_BUS = _InMemoryBus(
    max_queue_size=_read_local_bus_max_queue(),
    overflow_policy=_read_local_bus_overflow_policy(),
)


class StreamingExperienceCollector:
    def __init__(
        self,
        *,
        bootstrap_servers: Optional[List[str]] = None,
        topic_prefix: str = "multiverse",
        batch_size: int = 128,
        force_local_bus: bool = False,
    ):
        self.topic_prefix = str(topic_prefix)
        self.experience_topic = f"{self.topic_prefix}.experiences"
        self.metrics_topic = f"{self.topic_prefix}.metrics"
        self.control_topic = f"{self.topic_prefix}.control"
        self.batch_size = max(1, int(batch_size))
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._force_local = bool(force_local_bus)
        self._kafka_enabled = False
        self._producer = None

        if (not self._force_local) and KafkaProducer is not None and bootstrap_servers:
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=ExperienceSerializer.serialize,
                    linger_ms=10,
                    batch_size=32768,
                    acks=1,
                )
                self._kafka_enabled = True
            except Exception:
                self._kafka_enabled = False
                self._producer = None

    def publish_experience(
        self,
        event: Any,
        *,
        run_id: str,
        agent_id: str,
        flush_now: bool = False,
    ) -> None:
        d = asdict(event) if is_dataclass(event) else dict(event)
        info = d.get("info")
        info = info if isinstance(info, dict) else {}
        se = info.get("safe_executor")
        se = se if isinstance(se, dict) else {}
        conf_stats = se.get("confidence_status")
        conf_stats = conf_stats if isinstance(conf_stats, dict) else {}
        age_steps = d.get("policy_age_steps", d.get("step_idx", d.get("step", 0)))
        try:
            age_steps = float(age_steps)
        except Exception:
            age_steps = 0.0

        d["_stream"] = {
            "run_id": str(run_id),
            "agent_id": str(agent_id),
            "published_at_ms": int(time.time() * 1000),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        d["_trajectory"] = {
            "policy_id": str(d.get("policy_id", "")),
            "policy_version": str(d.get("policy_version", "")),
            "policy_age_steps": float(age_steps),
            "runtime_confidence_stats": dict(conf_stats),
        }
        with self._lock:
            self._buffer.append(d)
            should_flush = len(self._buffer) >= self.batch_size or bool(flush_now)
        if should_flush:
            self.flush_buffer()

    def flush_buffer(self) -> None:
        with self._lock:
            batch = list(self._buffer)
            self._buffer.clear()
        if not batch:
            return
        if self._kafka_enabled and self._producer is not None:
            for item in batch:
                self._producer.send(self.experience_topic, item)
            self._producer.flush()
            return
        for item in batch:
            _BUS.put(self.experience_topic, item)

    def publish_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = dict(metrics)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        if self._kafka_enabled and self._producer is not None:
            self._producer.send(self.metrics_topic, payload)
            return
        _BUS.put(self.metrics_topic, payload)

    def publish_control(self, command: str, payload: Optional[Dict[str, Any]] = None) -> None:
        msg = {
            "command": str(command),
            "payload": dict(payload or {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._kafka_enabled and self._producer is not None:
            self._producer.send(self.control_topic, msg)
            return
        _BUS.put(self.control_topic, msg)

    def close(self) -> None:
        self.flush_buffer()
        if self._producer is not None:
            try:
                self._producer.flush()
                self._producer.close()
            except Exception:
                pass


class ExperienceProcessor:
    def __init__(
        self,
        *,
        bootstrap_servers: Optional[List[str]] = None,
        topic_prefix: str = "multiverse",
        group_id: str = "multiverse-processor",
        process_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        force_local_bus: bool = False,
    ):
        self.topic = f"{topic_prefix}.experiences"
        self.group_id = str(group_id)
        self.process_callback = process_callback
        self.force_local_bus = bool(force_local_bus)
        self.bootstrap_servers = bootstrap_servers or []
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.stats = {"messages_processed": 0, "errors": 0, "last_processed_ms": 0}

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        use_kafka = (
            (not self.force_local_bus)
            and KafkaConsumer is not None
            and bool(self.bootstrap_servers)
        )
        if use_kafka:
            try:
                consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    value_deserializer=ExperienceSerializer.deserialize,
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                )
                while self.running:
                    records = consumer.poll(timeout_ms=500)
                    for _, messages in records.items():
                        for m in messages:
                            self._handle(m.value)
                consumer.close()
                return
            except Exception:
                pass

        while self.running:
            msg = _BUS.get(self.topic, timeout=0.5)
            if msg is None:
                continue
            self._handle(msg)

    def _handle(self, message: Dict[str, Any]) -> None:
        try:
            self.stats["messages_processed"] += 1
            self.stats["last_processed_ms"] = int(time.time() * 1000)
            if self.process_callback is not None:
                self.process_callback(message)
        except Exception:
            self.stats["errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)


class RealTimeAggregator:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = max(5, int(window_seconds))
        self._lock = threading.Lock()
        self._events: Deque[Tuple[float, float]] = deque()
        self._reward_sum = 0.0

    def consume(self, event: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            reward = float(event.get("reward", 0.0))
            self._events.append((now, reward))
            self._reward_sum += reward
            cutoff = now - self.window_seconds
            while self._events and self._events[0][0] < cutoff:
                _, r = self._events.popleft()
                self._reward_sum -= float(r)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._events)
            if n <= 0:
                return {"count": 0, "mean_reward": 0.0, "throughput_per_s": 0.0}
            span = max(1e-9, float(self._events[-1][0] - self._events[0][0]))
            return {
                "count": n,
                "mean_reward": float(self._reward_sum / n),
                "throughput_per_s": float(n / span),
            }


def example_streaming_pipeline() -> None:
    collector = StreamingExperienceCollector(force_local_bus=True)
    agg = RealTimeAggregator(window_seconds=10)
    proc = ExperienceProcessor(process_callback=agg.consume, force_local_bus=True)
    proc.start()
    for i in range(20):
        collector.publish_experience(
            event={"obs": {"x": i}, "action": i % 2, "reward": (1.0 if i % 3 == 0 else -0.1), "done": False},
            run_id="stream_test",
            agent_id="agent_test",
            flush_now=True,
        )
        time.sleep(0.02)
    time.sleep(0.5)
    proc.stop()
    collector.close()
    print(json.dumps({"processor": proc.get_stats(), "window": agg.snapshot()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    example_streaming_pipeline()
