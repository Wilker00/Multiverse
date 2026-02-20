"""
orchestrator/model_benchmark.py

Return-based benchmark for UniversalModel in live verse rollouts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.types import VerseSpec
from memory.confidence_auditor import (
    ConfidenceAuditConfig,
    OutcomeSample,
    record_outcomes_batch,
)
from memory.task_taxonomy import primary_task_tag
from models.universal_model import UniversalModel
from verses.registry import create_verse, register_builtin


@dataclass
class BenchmarkReport:
    verse_name: str
    episodes: int
    max_steps: int
    total_return: float
    mean_return: float
    mean_steps: float
    success_rate: Optional[float]
    audited_samples: int = 0


def _random_action(action_space: Any, rng: random.Random) -> Any:
    if getattr(action_space, "type", None) == "discrete":
        n = int(getattr(action_space, "n", 0) or 0)
        if n <= 0:
            raise ValueError("discrete action space requires n > 0")
        return int(rng.randrange(n))
    raise ValueError(f"Unsupported action space for benchmark fallback: {getattr(action_space, 'type', None)}")


def benchmark_universal_model(
    *,
    model: UniversalModel,
    verse_spec: VerseSpec,
    episodes: int,
    max_steps: int,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    seed: Optional[int] = 123,
    random_fallback: bool = True,
    audit_confidence: bool = False,
    audit_root_dir: Optional[str] = None,
) -> BenchmarkReport:
    register_builtin()
    verse = create_verse(verse_spec)
    rng = random.Random(seed if seed is not None else 123)

    total_return = 0.0
    total_steps = 0
    successes = []
    samples: list[OutcomeSample] = []

    for ep in range(int(episodes)):
        ep_seed = None if seed is None else int(seed) + ep
        verse.seed(ep_seed)
        reset = verse.reset()
        obs = reset.obs
        done = False
        ep_return = 0.0
        ep_steps = 0
        reached_goal = None
        history: list[dict[str, Any]] = []

        while not done and ep_steps < int(max_steps):
            pred = model.predict(
                obs=obs,
                verse_name=verse_spec.verse_name,
                top_k=top_k,
                min_score=min_score,
                recent_history=history,
            )
            action = pred.get("action")
            if action is None:
                if not random_fallback:
                    break
                action = _random_action(verse.action_space, rng)

            step = verse.step(action)
            ep_return += float(step.reward)
            ep_steps += 1
            done = bool(step.done or step.truncated)
            history.append({"obs": obs, "action": action, "reward": float(step.reward)})
            if len(history) > 16:
                history = history[-16:]
            obs = step.obs

            if audit_confidence and isinstance(pred, dict):
                strategy = str(pred.get("strategy", "")).strip()
                source_verse = pred.get("bridge_source_verse")
                if strategy in ("direct", "semantic_bridge", "hybrid_low_confidence"):
                    samples.append(
                        OutcomeSample(
                            strategy=strategy,
                            target_verse=verse_spec.verse_name,
                            reward=float(step.reward),
                            source_verse=(None if source_verse is None else str(source_verse)),
                            task_tag=primary_task_tag(verse_spec.verse_name),
                        )
                    )

            info = step.info or {}
            if isinstance(info, dict) and "reached_goal" in info:
                if info.get("reached_goal") is True:
                    reached_goal = True

        if reached_goal is None:
            reached_goal = False
        successes.append(bool(reached_goal))
        total_return += ep_return
        total_steps += ep_steps

    verse.close()
    if audit_confidence and samples:
        root = audit_root_dir if audit_root_dir is not None else "central_memory"
        record_outcomes_batch(samples, ConfidenceAuditConfig(root_dir=root))

    eps = max(1, int(episodes))
    success_rate = (sum(1 for s in successes if s) / float(len(successes))) if successes else None
    return BenchmarkReport(
        verse_name=verse_spec.verse_name,
        episodes=int(episodes),
        max_steps=int(max_steps),
        total_return=float(total_return),
        mean_return=float(total_return / float(eps)),
        mean_steps=float(total_steps / float(eps)),
        success_rate=success_rate,
        audited_samples=len(samples),
    )


def print_benchmark_report(report: BenchmarkReport, *, label: str = "universal_model") -> None:
    print("Model benchmark report")
    print(f"label        : {label}")
    print(f"verse        : {report.verse_name}")
    print(f"episodes     : {report.episodes}")
    print(f"max_steps    : {report.max_steps}")
    print(f"total_return : {report.total_return:.3f}")
    print(f"mean_return  : {report.mean_return:.3f}")
    print(f"mean_steps   : {report.mean_steps:.2f}")
    if report.success_rate is not None:
        print(f"success_rate : {report.success_rate:.3f}")
    else:
        print("success_rate : n/a")
    print(f"audited      : {report.audited_samples}")
