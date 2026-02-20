#!/usr/bin/env python3
"""
tools/scale_monitor.py

Local scaling monitor for CPU/memory/GPU and run throughput.
Works without Kubernetes and without external services.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import psutil
from orchestrator.curriculum_controller import (
    CurriculumConfig,
    CurriculumController,
)
from orchestrator.evaluator import evaluate_run

try:
    import GPUtil  # type: ignore
except Exception:  # pragma: no cover
    GPUtil = None

try:
    from prometheus_client import Gauge, start_http_server  # type: ignore
except Exception:  # pragma: no cover
    Gauge = None
    start_http_server = None


@dataclass
class MonitorSample:
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_percent: Optional[float]
    gpu_memory_percent: Optional[float]
    run_count: int
    events_lines_total: int


class ScaleMonitor:
    def __init__(
        self,
        runs_root: str = "runs",
        prometheus_port: Optional[int] = None,
        *,
        emit_curriculum: bool = False,
        curriculum_state_path: str = os.path.join("models", "curriculum_adjustments.json"),
        curriculum_window_runs: int = 16,
    ):
        self.runs_root = runs_root
        self.samples: List[MonitorSample] = []
        self.prometheus_port = prometheus_port
        self.emit_curriculum = bool(emit_curriculum)
        self.curriculum_window_runs = max(4, int(curriculum_window_runs))
        self.curriculum: Optional[CurriculumController] = None
        if self.emit_curriculum:
            self.curriculum = CurriculumController(
                CurriculumConfig.from_dict({"enabled": True, "state_path": curriculum_state_path})
            )
        self._g_cpu = self._g_mem = self._g_disk = self._g_runs = self._g_events = None
        self._g_gpu = self._g_gpu_mem = None
        if self.prometheus_port is not None and start_http_server is not None and Gauge is not None:
            start_http_server(int(self.prometheus_port))
            self._g_cpu = Gauge("multiverse_cpu_percent", "CPU usage percent")
            self._g_mem = Gauge("multiverse_memory_percent", "Memory usage percent")
            self._g_disk = Gauge("multiverse_disk_percent", "Disk usage percent")
            self._g_runs = Gauge("multiverse_runs_total", "Total run directories")
            self._g_events = Gauge("multiverse_events_lines_total", "Total events.jsonl lines across runs")
            self._g_gpu = Gauge("multiverse_gpu_percent", "GPU load percent")
            self._g_gpu_mem = Gauge("multiverse_gpu_memory_percent", "GPU memory usage percent")

    def _count_runs_and_events(self) -> tuple[int, int]:
        if not os.path.isdir(self.runs_root):
            return 0, 0
        run_count = 0
        events_total = 0
        for name in os.listdir(self.runs_root):
            run_dir = os.path.join(self.runs_root, name)
            if not os.path.isdir(run_dir):
                continue
            run_count += 1
            events_path = os.path.join(run_dir, "events.jsonl")
            if not os.path.isfile(events_path):
                continue
            try:
                with open(events_path, "r", encoding="utf-8") as f:
                    for _ in f:
                        events_total += 1
            except Exception:
                pass
        return run_count, events_total

    def collect_once(self) -> MonitorSample:
        cpu = float(psutil.cpu_percent(interval=0.2))
        mem = float(psutil.virtual_memory().percent)
        disk = float(psutil.disk_usage(".").percent)
        gpu = None
        gpu_mem = None
        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = float(gpus[0].load * 100.0)
                    gpu_mem = float(gpus[0].memoryUtil * 100.0)
            except Exception:
                pass

        run_count, events_total = self._count_runs_and_events()
        s = MonitorSample(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_percent=cpu,
            memory_percent=mem,
            disk_percent=disk,
            gpu_percent=gpu,
            gpu_memory_percent=gpu_mem,
            run_count=run_count,
            events_lines_total=events_total,
        )
        self.samples.append(s)
        if self.curriculum is not None:
            self._emit_curriculum_signals()

        if self._g_cpu is not None:
            self._g_cpu.set(cpu)
            self._g_mem.set(mem)
            self._g_disk.set(disk)
            self._g_runs.set(run_count)
            self._g_events.set(events_total)
            if gpu is not None:
                self._g_gpu.set(gpu)
            if gpu_mem is not None:
                self._g_gpu_mem.set(gpu_mem)
        return s

    def _emit_curriculum_signals(self) -> None:
        verse_signals = self._collect_verse_signals()
        if not verse_signals:
            return
        for verse_name, agg in verse_signals.items():
            try:
                self.curriculum.update_from_signal(
                    verse_name=str(verse_name),
                    success_rate=float(agg["success_rate"]),
                    mean_return=float(agg["mean_return"]),
                )
            except Exception:
                continue

    def _collect_verse_signals(self) -> Dict[str, Dict[str, float]]:
        if not os.path.isdir(self.runs_root):
            return {}
        rows: List[tuple[str, float, float]] = []
        dirs = []
        for name in os.listdir(self.runs_root):
            path = os.path.join(self.runs_root, name)
            if os.path.isdir(path):
                dirs.append(path)
        dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        dirs = dirs[: self.curriculum_window_runs]
        for run_dir in dirs:
            events_path = os.path.join(run_dir, "events.jsonl")
            if not os.path.isfile(events_path):
                continue
            verse_name = ""
            try:
                with open(events_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        ev = json.loads(line)
                        verse_name = str(ev.get("verse_name", "")).strip().lower()
                        break
            except Exception:
                verse_name = ""
            if not verse_name:
                continue
            try:
                st = evaluate_run(run_dir)
                rows.append((verse_name, float(st.mean_return), float(st.success_rate or 0.0)))
            except Exception:
                continue

        if not rows:
            return {}
        bucket: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}
        for verse_name, mean_return, success_rate in rows:
            if verse_name not in bucket:
                bucket[verse_name] = {"mean_return": 0.0, "success_rate": 0.0}
                counts[verse_name] = 0
            bucket[verse_name]["mean_return"] += float(mean_return)
            bucket[verse_name]["success_rate"] += float(success_rate)
            counts[verse_name] += 1
        for verse_name in list(bucket.keys()):
            c = max(1, counts.get(verse_name, 1))
            bucket[verse_name]["mean_return"] /= float(c)
            bucket[verse_name]["success_rate"] /= float(c)
        return bucket

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"count": 0}
        arr = self.samples
        def _avg(xs: List[float]) -> float:
            return float(sum(xs) / max(1, len(xs)))
        cpu = [s.cpu_percent for s in arr]
        mem = [s.memory_percent for s in arr]
        ev = [s.events_lines_total for s in arr]
        throughput = 0.0
        if len(arr) >= 2:
            dt = max(1e-9, (datetime.fromisoformat(arr[-1].timestamp.replace("Z", "")) -
                            datetime.fromisoformat(arr[0].timestamp.replace("Z", ""))).total_seconds())
            throughput = float((ev[-1] - ev[0]) / dt)
        return {
            "count": len(arr),
            "cpu_mean": _avg(cpu),
            "cpu_p95": sorted(cpu)[int(0.95 * (len(cpu) - 1))],
            "memory_mean": _avg(mem),
            "runs_latest": int(arr[-1].run_count),
            "events_latest": int(arr[-1].events_lines_total),
            "events_throughput_per_s": throughput,
            "latest_timestamp": arr[-1].timestamp,
        }

    def save_report(self, out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        payload = {"summary": self.summary(), "samples": [asdict(s) for s in self.samples]}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--interval_s", type=float, default=5.0)
    ap.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    ap.add_argument("--prometheus_port", type=int, default=None)
    ap.add_argument("--report_out", type=str, default="monitoring/scale_report.json")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--emit_curriculum", action="store_true")
    ap.add_argument("--curriculum_state_path", type=str, default=os.path.join("models", "curriculum_adjustments.json"))
    ap.add_argument("--curriculum_window_runs", type=int, default=16)
    args = ap.parse_args()

    mon = ScaleMonitor(
        runs_root=args.runs_root,
        prometheus_port=args.prometheus_port,
        emit_curriculum=bool(args.emit_curriculum),
        curriculum_state_path=str(args.curriculum_state_path),
        curriculum_window_runs=max(4, int(args.curriculum_window_runs)),
    )
    if args.once:
        s = mon.collect_once()
        print(json.dumps(asdict(s), ensure_ascii=False, indent=2))
        return

    i = 0
    try:
        while True:
            s = mon.collect_once()
            print(
                f"[{s.timestamp}] cpu={s.cpu_percent:.1f}% mem={s.memory_percent:.1f}% "
                f"runs={s.run_count} events={s.events_lines_total}"
            )
            i += 1
            if int(args.iterations) > 0 and i >= int(args.iterations):
                break
            time.sleep(max(0.1, float(args.interval_s)))
    except KeyboardInterrupt:
        pass

    out = mon.save_report(args.report_out)
    print(json.dumps(mon.summary(), ensure_ascii=False, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
