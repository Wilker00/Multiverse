"""
tools/memory_soak_gate.py

Run a lightweight MARL soak and estimate RSS memory slope (MB/hour).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from core.types import AgentSpec, VerseSpec
from orchestrator.marl_trainer import MARLConfig, MultiAgentTrainer


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _linear_slope_mb_per_hour(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    n = float(len(xs))
    sx = float(sum(xs))
    sy = float(sum(ys))
    sxx = float(sum(x * x for x in xs))
    sxy = float(sum(x * y for x, y in zip(xs, ys)))
    denom = (n * sxx) - (sx * sx)
    if abs(denom) <= 1e-12:
        return 0.0
    slope_mb_per_sec = float(((n * sxy) - (sx * sy)) / denom)
    return float(slope_mb_per_sec * 3600.0)


def _rss_mb() -> float:
    if psutil is None:
        return 0.0
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss / (1024.0 * 1024.0))


def run_memory_soak(
    *,
    duration_sec: int,
    sample_interval_sec: int,
    run_root: str,
    seed: int,
) -> Dict[str, Any]:
    if psutil is None:
        raise RuntimeError("psutil is required for memory soak gate")
    trainer = MultiAgentTrainer(run_root=run_root, schema_version="v1", auto_register_builtin=True)
    points: List[Tuple[float, float]] = []
    started = time.time()

    loops = max(2, int(max(1, duration_sec) / max(1, sample_interval_sec)))
    for i in range(loops):
        verse_specs = [
            VerseSpec(
                spec_version="v1",
                verse_name="line_world",
                verse_version="0.1",
                seed=int(seed + i),
                tags=["memory_soak"],
                params={"goal_pos": 7, "max_steps": 16, "step_penalty": -0.01},
            ),
            VerseSpec(
                spec_version="v1",
                verse_name="grid_world",
                verse_version="0.1",
                seed=int(seed + i + 100),
                tags=["memory_soak"],
                params={"size": 5, "max_steps": 16, "step_penalty": -0.01},
            ),
        ]
        agent_specs = [
            AgentSpec(spec_version="v1", policy_id=f"soak_a0_{i}", policy_version="0.1", algo="random", seed=seed + i),
            AgentSpec(spec_version="v1", policy_id=f"soak_a1_{i}", policy_version="0.1", algo="random", seed=seed + i + 1000),
        ]
        cfg = MARLConfig(
            episodes=2,
            max_steps=16,
            train=True,
            collect_transitions=True,
            shared_memory_enabled=True,
            shared_memory_top_k=3,
            negotiation_interval=1,
            lexicon_min_support=1,
        )
        trainer.run(verse_specs=verse_specs, agent_specs=agent_specs, config=cfg, seed=seed + i)
        t_sec = float(time.time() - started)
        points.append((t_sec, _rss_mb()))
        time.sleep(max(0, int(sample_interval_sec)))

    slope = _linear_slope_mb_per_hour(points)
    return {
        "rss_slope_mb_per_hour": float(slope),
        "duration_sec": int(duration_sec),
        "sample_interval_sec": int(sample_interval_sec),
        "samples": [{"t_sec": float(t), "rss_mb": float(r)} for t, r in points],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration_sec", type=int, default=120)
    ap.add_argument("--sample_interval_sec", type=int, default=10)
    ap.add_argument("--max_slope_mb_per_hour", type=float, default=48.0)
    ap.add_argument("--output_json", type=str, default=os.path.join("models", "memory_health", "latest.json"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fail_on_slope", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        report = run_memory_soak(
            duration_sec=max(2, int(args.duration_sec)),
            sample_interval_sec=max(1, int(args.sample_interval_sec)),
            run_root=td,
            seed=int(args.seed),
        )
    slope = _safe_float(report.get("rss_slope_mb_per_hour", 0.0), 0.0)
    max_slope = max(0.0, _safe_float(args.max_slope_mb_per_hour, 48.0))
    leak = bool(slope > max_slope)
    report["max_allowed_mb_per_hour"] = float(max_slope)
    report["leak_detected"] = bool(leak)
    report["created_at_ms"] = int(time.time() * 1000)

    out_path = str(args.output_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"memory_health_json={out_path}")
    print(f"rss_slope_mb_per_hour={slope:.4f}")
    if bool(args.fail_on_slope) and leak:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

