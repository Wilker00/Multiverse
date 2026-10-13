"""Measure native batch stepping against serial Python calls to the same MuJoCo model."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.physics.native_backend import NativeEngineUnavailableError, NativePhysicsBackend


def _serial_python(model_path: Path, controls: np.ndarray, iterations: int, substeps: int) -> float:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = [mujoco.MjData(model) for _ in range(len(controls))]
    start = time.perf_counter()
    for _ in range(iterations):
        for lane, item in enumerate(data):
            item.ctrl[:] = controls[lane]
            for _ in range(substeps):
                mujoco.mj_step(model, item)
    return time.perf_counter() - start


def _native_batch(model_path: Path, controls: np.ndarray, iterations: int,
                  substeps: int, threads: int):
    backend = NativePhysicsBackend(
        "point_mass", env_count=len(controls), worker_count=threads, substeps=substeps
    )
    try:
        backend.reset_batch()
        start = time.perf_counter()
        for _ in range(iterations):
            backend.step_controls(controls)
        return time.perf_counter() - start, backend.worker_count
    finally:
        backend.close()


def benchmark(*, env_counts: List[int], iterations: int, warmup: int, threads: int) -> Dict[str, object]:
    model_path = ROOT / "integrations" / "physics" / "models" / "point_mass.xml"
    rows = []
    for env_count in env_counts:
        controls = np.zeros((env_count, 2), dtype=np.float64)
        controls[:, 0] = np.where(np.arange(env_count) % 2 == 0, 1.5, -1.5)
        _serial_python(model_path, controls, warmup, 10)
        _native_batch(model_path, controls, warmup, 10, threads)
        serial_seconds = _serial_python(model_path, controls, iterations, 10)
        native_seconds, effective_threads = _native_batch(model_path, controls, iterations, 10, threads)
        env_steps = env_count * iterations
        serial_rate = env_steps / max(serial_seconds, 1e-12)
        native_rate = env_steps / max(native_seconds, 1e-12)
        rows.append({
            "env_count": env_count,
            "iterations": iterations,
            "substeps_per_control_step": 10,
            "native_worker_count_effective": effective_threads,
            "serial_python_env_steps_per_second": serial_rate,
            "native_cpp_env_steps_per_second": native_rate,
            "speedup": native_rate / max(serial_rate, 1e-12),
            "serial_seconds": serial_seconds,
            "native_seconds": native_seconds,
        })
    try:
        import mujoco
        mujoco_version = mujoco.__version__
    except ImportError:
        mujoco_version = "unavailable"
    return {
        "model": str(model_path.relative_to(ROOT)),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "native_worker_count_requested": threads,
        "warmup_iterations": warmup,
        "mujoco_version": mujoco_version,
        "results": rows,
        "acceptance": {
            "target": "at least 4x native C++ throughput at 64 environments",
            "passed": next((row["speedup"] >= 4.0 for row in rows if row["env_count"] == 64), False),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-counts", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0, help="0 selects up to the available CPU count")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-4x", action="store_true")
    args = parser.parse_args()
    if min(args.env_counts) < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("environment counts and iterations must be positive; warmup cannot be negative")
    try:
        report = benchmark(env_counts=args.env_counts, iterations=args.iterations,
                           warmup=args.warmup, threads=args.threads)
    except NativeEngineUnavailableError as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.require_4x and not report["acceptance"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

