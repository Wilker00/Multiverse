"""
orchestrator/eval_harness.py

Strict, reproducible evaluation harness with A/B significance gating.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.registry import create_agent, register_builtin_agents
from core.types import AgentSpec, JSONValue, VerseSpec
from verses.registry import create_verse, register_builtin


@dataclass
class BenchmarkCase:
    verse_name: str
    verse_version: str
    params: Dict[str, JSONValue]
    seeds: List[int]
    episodes_per_seed: int
    max_steps: int


@dataclass
class VerseGateThreshold:
    min_mean_return: float
    min_success_rate: float
    max_failure_rate: float
    max_safety_violation_rate: float
    max_return_variance: float
    min_return_delta: float
    min_success_delta: float
    variance_tolerance_mult: float = 1.25


@dataclass
class EvalEpisode:
    seed: int
    return_sum: float
    steps: int
    success: bool
    failure: bool
    safety_violation: bool
    truncated: bool


@dataclass
class VerseEvalSummary:
    verse_name: str
    episodes: int
    mean_return: float
    success_rate: float
    failure_rate: float
    safety_violation_rate: float
    return_variance: float
    mean_steps: float
    returns: List[float]
    successes: List[int]


@dataclass
class ABVerseGateResult:
    verse_name: str
    baseline: VerseEvalSummary
    candidate: VerseEvalSummary
    return_delta_mean: float
    success_delta_mean: float
    return_delta_ci_low: float
    return_delta_ci_high: float
    success_delta_ci_low: float
    success_delta_ci_high: float
    pass_absolute: bool
    pass_significance: bool
    pass_stability: bool
    pass_safety: bool
    passed: bool
    reasons: List[str]


@dataclass
class ABGateResult:
    passed: bool
    suite_mode: str
    verses: List[str]
    bootstrap_samples: int
    alpha: float
    by_verse: Dict[str, ABVerseGateResult]
    reasons: List[str]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _default_params(verse_name: str, max_steps: int) -> Dict[str, JSONValue]:
    v = str(verse_name).strip().lower()
    params: Dict[str, JSONValue] = {"max_steps": int(max_steps), "adr_enabled": False}
    if v == "line_world":
        params.update({"goal_pos": 8, "step_penalty": -0.02})
    elif v == "grid_world":
        params.update({"width": 5, "height": 5, "step_penalty": -0.01})
    elif v == "cliff_world":
        params.update(
            {
                "width": 12,
                "height": 4,
                "step_penalty": -1.0,
                "cliff_penalty": -100.0,
                "end_on_cliff": False,
            }
        )
    elif v == "park_world":
        params.update({"lane_len": 7, "goal_pos": 5, "step_penalty": -0.01})
    elif v == "pursuit_world":
        params.update({"lane_len": 9, "step_penalty": -0.01})
    elif v == "labyrinth_world":
        params.update(
            {
                "width": 15,
                "height": 11,
                "step_penalty": -0.05,
                "wall_penalty": -0.20,
                "pit_penalty": -25.0,
                "laser_penalty": -18.0,
                "battery_capacity": 80,
                "battery_drain": 1,
                "action_noise": 0.08,
            }
        )
    elif v == "warehouse_world":
        params.update(
            {
                "width": 8,
                "height": 8,
                "step_penalty": -0.10,
                "wall_penalty": -0.50,
                "obstacle_penalty": -1.00,
                "battery_capacity": 20,
                "battery_drain": 1,
                "charge_rate": 5,
                "battery_fail_penalty": -10.0,
                "goal_reward": 10.0,
                "charge_reward": 0.5,
                "obstacle_count": 14,
            }
        )
    return params


def default_benchmark_suite(mode: str = "full", target_verse: Optional[str] = None) -> List[BenchmarkCase]:
    mode_n = str(mode).strip().lower()
    verses = ["line_world", "grid_world", "cliff_world", "park_world", "pursuit_world", "warehouse_world"]
    if target_verse:
        verses = [str(target_verse).strip().lower()]
    elif mode_n == "target":
        raise ValueError("target_verse is required when mode=target")

    full_seeds = [101, 203, 307, 401, 503]
    quick_seeds = [101, 203]
    seeds = quick_seeds if mode_n == "quick" else full_seeds
    eps_per_seed = 2 if mode_n == "quick" else 4

    cases: List[BenchmarkCase] = []
    for v in verses:
        max_steps = 40
        if v == "grid_world":
            max_steps = 60
        elif v == "cliff_world":
            max_steps = 100
        elif v == "park_world":
            max_steps = 80
        elif v == "pursuit_world":
            max_steps = 60
        elif v == "labyrinth_world":
            max_steps = 180
        elif v == "warehouse_world":
            max_steps = 100
        cases.append(
            BenchmarkCase(
                verse_name=v,
                verse_version="0.1",
                params=_default_params(v, max_steps),
                seeds=list(seeds),
                episodes_per_seed=eps_per_seed,
                max_steps=max_steps,
            )
        )
    return cases


def default_gate_thresholds() -> Dict[str, VerseGateThreshold]:
    # Reasonably strict, but still achievable by current strong baselines.
    return {
        "line_world": VerseGateThreshold(
            min_mean_return=0.20,
            min_success_rate=0.70,
            max_failure_rate=0.30,
            max_safety_violation_rate=0.30,
            max_return_variance=1.20,
            min_return_delta=0.0,
            min_success_delta=0.0,
        ),
        "grid_world": VerseGateThreshold(
            min_mean_return=0.05,
            min_success_rate=0.50,
            max_failure_rate=0.50,
            max_safety_violation_rate=0.40,
            max_return_variance=1.50,
            min_return_delta=0.0,
            min_success_delta=0.0,
        ),
        "cliff_world": VerseGateThreshold(
            min_mean_return=-50.0,
            min_success_rate=0.25,
            max_failure_rate=0.75,
            max_safety_violation_rate=0.80,
            max_return_variance=2500.0,
            min_return_delta=0.0,
            min_success_delta=0.0,
        ),
        "park_world": VerseGateThreshold(
            min_mean_return=0.00,
            min_success_rate=0.40,
            max_failure_rate=0.60,
            max_safety_violation_rate=0.50,
            max_return_variance=1.50,
            min_return_delta=0.0,
            min_success_delta=0.0,
        ),
        "pursuit_world": VerseGateThreshold(
            min_mean_return=0.00,
            min_success_rate=0.35,
            max_failure_rate=0.65,
            max_safety_violation_rate=0.50,
            max_return_variance=1.80,
            min_return_delta=0.0,
            min_success_delta=0.0,
        ),
        "labyrinth_world": VerseGateThreshold(
            min_mean_return=-20.0,
            min_success_rate=0.20,
            max_failure_rate=0.80,
            max_safety_violation_rate=0.90,
            max_return_variance=200.0,
            min_return_delta=0.0,
            min_success_delta=0.0,
        ),
        "warehouse_world": VerseGateThreshold(
            min_mean_return=-15.0,
            min_success_rate=0.15,
            max_failure_rate=0.85,
            max_safety_violation_rate=0.85,
            max_return_variance=250.0,
            min_return_delta=0.0,
            min_success_delta=0.0,
        ),
    }


def _is_safety_violation(info: Dict[str, Any]) -> bool:
    safety_true_keys = (
        "wrong_park",
        "collision",
        "crash",
        "boundary_violation",
        "unsafe",
        "failure",
        "fell_cliff",
        "fell_pit",
        "hit_laser",
        "battery_depleted",
        "hit_wall",
        "hit_obstacle",
        "battery_death",
    )
    for k in safety_true_keys:
        if info.get(k) is True:
            return True
    return False


def _collect_dataset_paths(cfg: Dict[str, Any]) -> Tuple[List[str], Optional[str], Optional[str]]:
    dataset_paths: List[str] = []
    dataset_dir = None
    bad_dataset_path = None

    dataset_dir = cfg.get("dataset_dir")
    dp = cfg.get("dataset_paths")
    if dp:
        dataset_paths = list(dp) if isinstance(dp, list) else [dp]
    if not dataset_paths and cfg.get("dataset_path"):
        dataset_paths = [cfg.get("dataset_path")]
    bad_dataset_path = cfg.get("bad_dna_path")
    return [str(p) for p in dataset_paths], (None if dataset_dir is None else str(dataset_dir)), (
        None if bad_dataset_path is None else str(bad_dataset_path)
    )


def _hydrate_agent_from_config(agent: Any, algo: str, cfg: Dict[str, Any]) -> None:
    if algo not in (
        "q",
        "memory_recall",
        "planner_recall",
        "imitation_lookup",
        "library",
        "special",
        "special_moe",
        "adaptive_moe",
        "cql",
        "failure_aware",
    ):
        return
    dataset_paths, dataset_dir, bad_dataset_path = _collect_dataset_paths(cfg)
    if dataset_dir:
        try:
            names = sorted(os.listdir(dataset_dir))
            for n in names:
                p = os.path.join(dataset_dir, n)
                if os.path.isfile(p) and n.endswith(".jsonl"):
                    dataset_paths.append(p)
        except Exception:
            pass

    if dataset_paths and hasattr(agent, "learn_from_dataset"):
        for p in dataset_paths:
            if os.path.isfile(str(p)):
                agent.learn_from_dataset(str(p))
    if bad_dataset_path and hasattr(agent, "learn_from_bad_dataset") and os.path.isfile(str(bad_dataset_path)):
        agent.learn_from_bad_dataset(str(bad_dataset_path))


def _ensure_eval_spec_for_verse(spec: AgentSpec, verse_name: str) -> AgentSpec:
    algo = str(spec.algo or "").strip().lower()
    if algo in ("special_moe", "adaptive_moe", "gateway", "aware", "evolving", "memory_recall", "planner_recall"):
        cfg = dict(spec.config) if isinstance(spec.config, dict) else {}
        cfg.setdefault("verse_name", str(verse_name))
        return dataclasses.replace(spec, config=cfg)
    return spec


def evaluate_agent_case(
    *,
    agent_spec: AgentSpec,
    case: BenchmarkCase,
) -> VerseEvalSummary:
    register_builtin()
    register_builtin_agents()

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=case.verse_name,
        verse_version=case.verse_version,
        seed=None,
        tags=["benchmark_harness"],
        params=dict(case.params),
    )
    verse = create_verse(verse_spec)
    use_spec = _ensure_eval_spec_for_verse(agent_spec, case.verse_name)
    agent = create_agent(spec=use_spec, observation_space=verse.observation_space, action_space=verse.action_space)

    cfg = use_spec.config if isinstance(use_spec.config, dict) else {}
    _hydrate_agent_from_config(agent, str(use_spec.algo).strip().lower(), cfg)

    episodes: List[EvalEpisode] = []
    for base_seed in case.seeds:
        for ep in range(max(1, int(case.episodes_per_seed))):
            ep_seed = int(base_seed) * 1000 + ep
            return_sum = 0.0
            steps = 0
            success = False
            safety = False
            truncated = False
            failed = False
            try:
                verse.seed(ep_seed)
                agent.seed(ep_seed)
                reset = verse.reset()
                obs = reset.obs
                done = False
                while not done and steps < int(case.max_steps):
                    action_res = agent.act(obs)
                    step = verse.step(action_res.action)
                    return_sum += float(step.reward)
                    steps += 1
                    info = step.info or {}
                    if isinstance(info, dict):
                        if info.get("reached_goal") is True:
                            success = True
                        if _is_safety_violation(info):
                            safety = True
                    done = bool(step.done or step.truncated)
                    truncated = bool(step.truncated)
                    obs = step.obs
            except Exception:
                failed = True
                safety = True

            failure = bool((not success) or failed)
            episodes.append(
                EvalEpisode(
                    seed=ep_seed,
                    return_sum=float(return_sum),
                    steps=int(steps),
                    success=bool(success),
                    failure=bool(failure),
                    safety_violation=bool(safety),
                    truncated=bool(truncated),
                )
            )

    verse.close()
    agent.close()

    returns = [float(e.return_sum) for e in episodes]
    successes = [1 if e.success else 0 for e in episodes]
    failures = [1 if e.failure else 0 for e in episodes]
    safety_flags = [1 if e.safety_violation else 0 for e in episodes]
    steps = [int(e.steps) for e in episodes]

    n = max(1, len(episodes))
    mean_return = float(sum(returns) / float(n))
    success_rate = float(sum(successes) / float(n))
    failure_rate = float(sum(failures) / float(n))
    safety_rate = float(sum(safety_flags) / float(n))
    mean_steps = float(sum(steps) / float(n))
    return_var = float(statistics.pvariance(returns)) if len(returns) > 1 else 0.0

    return VerseEvalSummary(
        verse_name=case.verse_name,
        episodes=len(episodes),
        mean_return=mean_return,
        success_rate=success_rate,
        failure_rate=failure_rate,
        safety_violation_rate=safety_rate,
        return_variance=return_var,
        mean_steps=mean_steps,
        returns=returns,
        successes=successes,
    )


def _bootstrap_ci_paired_diff(
    *,
    candidate_vals: Sequence[float],
    baseline_vals: Sequence[float],
    samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 123,
) -> Tuple[float, float, float]:
    n = min(len(candidate_vals), len(baseline_vals))
    if n <= 0:
        return 0.0, 0.0, 0.0

    diffs = [float(candidate_vals[i]) - float(baseline_vals[i]) for i in range(n)]
    mean_diff = float(sum(diffs) / float(n))
    if n == 1 or samples <= 1:
        return mean_diff, mean_diff, mean_diff

    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(int(samples)):
        idxs = [rng.randrange(n) for _ in range(n)]
        m = sum(diffs[i] for i in idxs) / float(n)
        draws.append(float(m))
    draws.sort()
    lo_i = max(0, min(len(draws) - 1, int(math.floor((alpha / 2.0) * (len(draws) - 1)))))
    hi_i = max(0, min(len(draws) - 1, int(math.ceil((1.0 - alpha / 2.0) * (len(draws) - 1)))))
    return mean_diff, float(draws[lo_i]), float(draws[hi_i])


def run_ab_gate(
    *,
    baseline_spec: AgentSpec,
    candidate_spec: AgentSpec,
    suite_mode: str = "target",
    target_verse: Optional[str] = None,
    thresholds: Optional[Dict[str, VerseGateThreshold]] = None,
    bootstrap_samples: int = 2000,
    alpha: float = 0.05,
) -> ABGateResult:
    cases = default_benchmark_suite(suite_mode, target_verse=target_verse)
    gates = thresholds or default_gate_thresholds()
    by_verse: Dict[str, ABVerseGateResult] = {}
    reasons: List[str] = []

    for case in cases:
        baseline = evaluate_agent_case(agent_spec=baseline_spec, case=case)
        candidate = evaluate_agent_case(agent_spec=candidate_spec, case=case)
        gate = gates.get(case.verse_name, VerseGateThreshold(0.0, 0.0, 1.0, 1.0, 999.0, 0.0, 0.0))

        ret_diff, ret_lo, ret_hi = _bootstrap_ci_paired_diff(
            candidate_vals=candidate.returns,
            baseline_vals=baseline.returns,
            samples=int(bootstrap_samples),
            alpha=float(alpha),
            seed=123,
        )
        succ_diff, succ_lo, succ_hi = _bootstrap_ci_paired_diff(
            candidate_vals=[float(x) for x in candidate.successes],
            baseline_vals=[float(x) for x in baseline.successes],
            samples=int(bootstrap_samples),
            alpha=float(alpha),
            seed=456,
        )

        pass_absolute = (
            candidate.mean_return >= float(gate.min_mean_return)
            and candidate.success_rate >= float(gate.min_success_rate)
        )
        pass_safety = (
            candidate.failure_rate <= float(gate.max_failure_rate)
            and candidate.safety_violation_rate <= float(gate.max_safety_violation_rate)
        )
        stability_limit = max(
            float(gate.max_return_variance),
            float(baseline.return_variance) * float(gate.variance_tolerance_mult),
        )
        pass_stability = candidate.return_variance <= stability_limit
        pass_significance = (ret_lo >= float(gate.min_return_delta)) and (
            succ_lo >= float(gate.min_success_delta)
        )

        verse_reasons: List[str] = []
        if not pass_absolute:
            verse_reasons.append("absolute_metrics_below_threshold")
        if not pass_safety:
            verse_reasons.append("failure_or_safety_rate_too_high")
        if not pass_stability:
            verse_reasons.append("return_variance_regression")
        if not pass_significance:
            verse_reasons.append("insufficient_significant_improvement")

        passed = pass_absolute and pass_significance and pass_stability and pass_safety
        if not passed:
            reasons.append(f"{case.verse_name}:{','.join(verse_reasons)}")

        by_verse[case.verse_name] = ABVerseGateResult(
            verse_name=case.verse_name,
            baseline=baseline,
            candidate=candidate,
            return_delta_mean=float(ret_diff),
            success_delta_mean=float(succ_diff),
            return_delta_ci_low=float(ret_lo),
            return_delta_ci_high=float(ret_hi),
            success_delta_ci_low=float(succ_lo),
            success_delta_ci_high=float(succ_hi),
            pass_absolute=bool(pass_absolute),
            pass_significance=bool(pass_significance),
            pass_stability=bool(pass_stability),
            pass_safety=bool(pass_safety),
            passed=bool(passed),
            reasons=verse_reasons,
        )

    overall_pass = all(v.passed for v in by_verse.values()) if by_verse else False
    return ABGateResult(
        passed=bool(overall_pass),
        suite_mode=str(suite_mode),
        verses=sorted(list(by_verse.keys())),
        bootstrap_samples=int(bootstrap_samples),
        alpha=float(alpha),
        by_verse=by_verse,
        reasons=reasons,
    )


def gate_to_dict(result: ABGateResult) -> Dict[str, Any]:
    def _summary_to_dict(s: VerseEvalSummary) -> Dict[str, Any]:
        return {
            "verse_name": s.verse_name,
            "episodes": int(s.episodes),
            "mean_return": float(s.mean_return),
            "success_rate": float(s.success_rate),
            "failure_rate": float(s.failure_rate),
            "safety_violation_rate": float(s.safety_violation_rate),
            "return_variance": float(s.return_variance),
            "mean_steps": float(s.mean_steps),
        }

    by_verse: Dict[str, Any] = {}
    for verse, r in result.by_verse.items():
        by_verse[verse] = {
            "baseline": _summary_to_dict(r.baseline),
            "candidate": _summary_to_dict(r.candidate),
            "return_delta_mean": float(r.return_delta_mean),
            "success_delta_mean": float(r.success_delta_mean),
            "return_delta_ci_low": float(r.return_delta_ci_low),
            "return_delta_ci_high": float(r.return_delta_ci_high),
            "success_delta_ci_low": float(r.success_delta_ci_low),
            "success_delta_ci_high": float(r.success_delta_ci_high),
            "pass_absolute": bool(r.pass_absolute),
            "pass_significance": bool(r.pass_significance),
            "pass_stability": bool(r.pass_stability),
            "pass_safety": bool(r.pass_safety),
            "passed": bool(r.passed),
            "reasons": list(r.reasons),
        }
    return {
        "passed": bool(result.passed),
        "suite_mode": result.suite_mode,
        "verses": list(result.verses),
        "bootstrap_samples": int(result.bootstrap_samples),
        "alpha": float(result.alpha),
        "reasons": list(result.reasons),
        "by_verse": by_verse,
    }


def print_gate_report(result: ABGateResult, *, label: str = "ab_gate") -> None:
    print("Evaluation harness report")
    print(f"label             : {label}")
    print(f"suite_mode        : {result.suite_mode}")
    print(f"verses            : {result.verses}")
    print(f"bootstrap_samples : {result.bootstrap_samples}")
    print(f"alpha             : {result.alpha}")
    print(f"passed            : {result.passed}")
    print("")
    for verse in result.verses:
        r = result.by_verse[verse]
        print(f"[{verse}] passed={r.passed}")
        print(
            f"  baseline  mean_return={r.baseline.mean_return:.3f} "
            f"success={r.baseline.success_rate:.3f} "
            f"var={r.baseline.return_variance:.3f} "
            f"safety={r.baseline.safety_violation_rate:.3f}"
        )
        print(
            f"  candidate mean_return={r.candidate.mean_return:.3f} "
            f"success={r.candidate.success_rate:.3f} "
            f"var={r.candidate.return_variance:.3f} "
            f"safety={r.candidate.safety_violation_rate:.3f}"
        )
        print(
            f"  delta_return={r.return_delta_mean:.3f} "
            f"ci=[{r.return_delta_ci_low:.3f},{r.return_delta_ci_high:.3f}]"
        )
        print(
            f"  delta_success={r.success_delta_mean:.3f} "
            f"ci=[{r.success_delta_ci_low:.3f},{r.success_delta_ci_high:.3f}]"
        )
        print(
            f"  checks absolute={r.pass_absolute} significance={r.pass_significance} "
            f"stability={r.pass_stability} safety={r.pass_safety}"
        )
        if r.reasons:
            print(f"  reasons={r.reasons}")
