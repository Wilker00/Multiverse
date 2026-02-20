"""
experiments/chaos_testing.py

Phase 2.2 chaos engineering / failure recovery experiment.

Faults:
1) worker_crash
2) memory_corrupt
3) checkpoint_loss
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.registry import create_agent, register_builtin_agents
from core.agent_base import ActionResult, ExperienceBatch, Transition
from core.types import AgentSpec, VerseSpec
from memory.central_repository import CentralMemoryConfig, SanitizeStats, sanitize_memory_file
from tools.validation_stats import compute_validation_stats
from verses.registry import create_verse, register_builtin


FAULT_TYPES = ("worker_crash", "memory_corrupt", "checkpoint_loss")
STATE_HEALTHY = 0
STATE_DEGRADED = 1
STATE_FAILED = 2
STATE_RECOVERING = 3


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _state_name(i: int) -> str:
    return {
        STATE_HEALTHY: "healthy",
        STATE_DEGRADED: "degraded",
        STATE_FAILED: "failed",
        STATE_RECOVERING: "recovering",
    }.get(int(i), "unknown")


def estimate_transition_probabilities(states: Sequence[int]) -> List[List[float]]:
    k = 4
    counts = [[0.0 for _ in range(k)] for _ in range(k)]
    if len(states) <= 1:
        return [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for i in range(len(states) - 1):
        s0 = max(0, min(k - 1, int(states[i])))
        s1 = max(0, min(k - 1, int(states[i + 1])))
        counts[s0][s1] += 1.0
    out: List[List[float]] = []
    for r in range(k):
        row_sum = float(sum(counts[r]))
        if row_sum <= 1e-12:
            out.append([1.0 if i == r else 0.0 for i in range(k)])
            continue
        out.append([float(v / row_sum) for v in counts[r]])
    return out


def _steady_state_distribution(transition_matrix: List[List[float]], max_iter: int = 2000) -> List[float]:
    k = len(transition_matrix)
    pi = [1.0 / float(max(1, k)) for _ in range(k)]
    for _ in range(max(10, int(max_iter))):
        nxt = [0.0 for _ in range(k)]
        for i in range(k):
            for j in range(k):
                nxt[j] += float(pi[i]) * float(transition_matrix[i][j])
        s = sum(nxt)
        if s > 0.0:
            nxt = [float(x / s) for x in nxt]
        diff = max(abs(float(nxt[i]) - float(pi[i])) for i in range(k))
        pi = nxt
        if diff < 1e-10:
            break
    return pi


def compute_mttf(transition_matrix: List[List[float]]) -> float:
    # Approximation: expected consecutive non-failed run length.
    # hazard ~= mean P(nonfailed -> failed)
    non_failed = (STATE_HEALTHY, STATE_DEGRADED, STATE_RECOVERING)
    hazard = sum(float(transition_matrix[s][STATE_FAILED]) for s in non_failed) / float(len(non_failed))
    if hazard <= 1e-12:
        return float("inf")
    return float(1.0 / hazard)


def compute_mttr(transition_matrix: List[List[float]]) -> float:
    # Approximation: expected time to leave failed state.
    leave = 1.0 - float(transition_matrix[STATE_FAILED][STATE_FAILED])
    if leave <= 1e-12:
        return float("inf")
    return float(1.0 / leave)


def model_as_markov_chain(system_states: Sequence[int]) -> Dict[str, Any]:
    transition_matrix = estimate_transition_probabilities(system_states)
    steady_state = _steady_state_distribution(transition_matrix)
    # Treat degraded as available for service-availability accounting.
    availability = float(steady_state[STATE_HEALTHY] + steady_state[STATE_DEGRADED] + steady_state[STATE_RECOVERING])
    mttf = compute_mttf(transition_matrix)
    mttr = compute_mttr(transition_matrix)
    return {
        "availability": float(availability),
        "failed_probability": float(steady_state[STATE_FAILED]),
        "mttf": float(mttf),
        "mttr": float(mttr),
        "transition_matrix": transition_matrix,
        "steady_state": {str(_state_name(i)): float(v) for i, v in enumerate(steady_state)},
    }


def projected_availability_from_mttr(*, mttr_steps: float, normal_fault_interval_steps: float) -> float:
    mttf = max(1.0, float(normal_fault_interval_steps))
    mttr = max(1.0, float(mttr_steps))
    return float(mttf / (mttf + mttr))


def _severity_to_ratio(severity: str) -> float:
    s = str(severity).strip().lower()
    if s == "low":
        return 0.05
    if s == "high":
        return 0.35
    return 0.18


def _default_agent_spec(*, algo: str, seed: int) -> AgentSpec:
    cfg: Dict[str, Any] = {
        "train": True,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
    }
    return AgentSpec(
        spec_version="v1",
        policy_id=f"chaos_{str(algo)}",
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
        config=cfg,
    )


def _default_verse_spec(*, verse_name: str, seed: int, max_steps: int) -> VerseSpec:
    params: Dict[str, Any] = {"max_steps": int(max_steps), "adr_enabled": False}
    if str(verse_name).strip().lower() == "cliff_world":
        params.update({"width": 12, "height": 4, "step_penalty": -1.0, "cliff_penalty": -100.0, "end_on_cliff": False})
    return VerseSpec(
        spec_version="v1",
        verse_name=str(verse_name),
        verse_version="0.1",
        seed=int(seed),
        params=params,
    )


def _relative_performance_delta(perf: float, baseline: float) -> float:
    scale = max(1.0, abs(float(baseline)))
    return float((float(perf) - float(baseline)) / scale)


def _is_recovered(perf: float, baseline: float, tolerance: float = 0.10) -> bool:
    return bool(_relative_performance_delta(perf, baseline) >= -abs(float(tolerance)))


def _classify_state(perf: float, baseline: float, had_failure: bool, was_bad: bool) -> int:
    if had_failure:
        return STATE_FAILED
    delta = _relative_performance_delta(perf, baseline)
    if delta >= -0.10:
        return STATE_HEALTHY
    if delta >= -0.30:
        if was_bad:
            return STATE_RECOVERING
        return STATE_DEGRADED
    return STATE_FAILED


def _copy_agent_state(agent: Any) -> Optional[Dict[str, Any]]:
    if hasattr(agent, "get_state"):
        try:
            st = agent.get_state()
            if isinstance(st, dict):
                return dict(st)
        except Exception:
            pass
    # q-agent fallback snapshot
    if hasattr(agent, "q") and isinstance(getattr(agent, "q"), dict):
        try:
            q = getattr(agent, "q")
            q_copy: Dict[str, Any] = {}
            for k, arr in q.items():
                try:
                    q_copy[str(k)] = [float(x) for x in list(arr)]
                except Exception:
                    continue
            return {"q": q_copy}
        except Exception:
            return None
    return None


def _restore_agent_state(agent: Any, state: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(state, dict):
        return False
    if hasattr(agent, "set_state"):
        try:
            agent.set_state(dict(state))
            return True
        except Exception:
            pass
    if "q" in state and hasattr(agent, "q") and isinstance(getattr(agent, "q"), dict):
        try:
            import numpy as np

            q_src = state.get("q")
            if isinstance(q_src, dict):
                q_dst = getattr(agent, "q")
                q_dst.clear()
                for k, vals in q_src.items():
                    if not isinstance(vals, list):
                        continue
                    q_dst[str(k)] = np.asarray([float(v) for v in vals], dtype=np.float32)
                return True
        except Exception:
            return False
    return False


def _episode_run(
    *,
    verse: Any,
    agent: Any,
    max_steps: int,
    train: bool,
    fault_type: Optional[str],
    fault_step: int,
    checkpoint_interval: int,
    baseline_step_perf: Optional[float],
) -> Dict[str, Any]:
    rr = verse.reset()
    obs = rr.obs
    transitions: List[Transition] = []
    episode_return = 0.0
    steps = 0
    crash_count = 0
    fault_triggered = False
    checkpoint: Optional[Dict[str, Any]] = None
    checkpoint_obs: Optional[Any] = None
    reward_at_fault = 0.0
    recovered_steps: Optional[int] = None
    post_fault_steps = 0
    cascade_failures = 0

    # For checkpoint_loss we remove checkpoint first then force a crash later.
    crash_step = int(fault_step)
    if str(fault_type or "") == "checkpoint_loss":
        crash_step = int(fault_step) + 2

    while steps < int(max_steps):
        try:
            if int(checkpoint_interval) > 0 and (steps % int(checkpoint_interval) == 0):
                if hasattr(verse, "export_state"):
                    checkpoint = {
                        "verse_state": verse.export_state(),
                        "agent_state": _copy_agent_state(agent),
                    }
                    checkpoint_obs = obs

            if str(fault_type or "") == "checkpoint_loss" and (not fault_triggered) and steps == int(fault_step):
                checkpoint = None
                checkpoint_obs = None
                fault_triggered = True

            if str(fault_type or "") in ("worker_crash", "checkpoint_loss") and steps == int(crash_step):
                reward_at_fault = float(episode_return)
                fault_triggered = True
                raise RuntimeError("simulated_worker_crash")

            ar = agent.act(obs)
            action = ar.action if isinstance(ar, ActionResult) else ar
            sr = verse.step(action)
            info = sr.info if isinstance(sr.info, dict) else {}

            transitions.append(
                Transition(
                    obs=obs,
                    action=action,
                    reward=float(sr.reward),
                    next_obs=sr.obs,
                    done=bool(sr.done),
                    truncated=bool(sr.truncated),
                    info=info,
                )
            )
            episode_return += float(sr.reward)
            obs = sr.obs
            steps += 1

            if str(fault_type or "") in ("worker_crash", "checkpoint_loss") and fault_triggered:
                post_fault_steps += 1
                if recovered_steps is None and post_fault_steps > 0:
                    # Recovery time for crash faults is "steps until first successful
                    # post-crash transition executes".
                    recovered_steps = int(post_fault_steps)

            if bool(sr.done or sr.truncated):
                break
        except Exception:
            crash_count += 1
            if fault_type is not None and crash_count > 1:
                cascade_failures += 1
            # Recovery: rewind to checkpoint if available, otherwise hard reset.
            restored = False
            if isinstance(checkpoint, dict) and hasattr(verse, "import_state"):
                try:
                    verse.import_state(dict(checkpoint.get("verse_state") or {}))
                    if checkpoint_obs is not None:
                        obs = checkpoint_obs
                    restored = _restore_agent_state(agent, checkpoint.get("agent_state")) or True
                except Exception:
                    restored = False
            if not restored:
                rr = verse.reset()
                obs = rr.obs
            # Continue after recovery.
            steps += 1
            if steps >= int(max_steps):
                break

    if train and transitions:
        try:
            agent.learn(ExperienceBatch(transitions=transitions))
        except Exception:
            pass

    mean_step_perf = float(episode_return / float(max(1, steps)))
    return {
        "return_sum": float(episode_return),
        "steps": int(steps),
        "mean_step_perf": float(mean_step_perf),
        "crashes": int(crash_count),
        "cascade_failures": int(cascade_failures),
        "recovered_steps": recovered_steps,
    }


class ChaosTest:
    def __init__(
        self,
        *,
        verse_name: str = "cliff_world",
        algo: str = "q",
        max_steps: int = 100,
        seed: int = 123,
        central_memory_dir: str = "central_memory_chaos",
    ):
        self.verse_name = str(verse_name)
        self.algo = str(algo)
        self.max_steps = max(10, int(max_steps))
        self.seed = int(seed)
        self.central_memory_dir = str(central_memory_dir)

    def _make_agent_verse(self, seed: int) -> Tuple[Any, Any]:
        verse = create_verse(
            _default_verse_spec(verse_name=self.verse_name, seed=int(seed), max_steps=self.max_steps)
        )
        agent_spec = _default_agent_spec(algo=self.algo, seed=int(seed))
        agent = create_agent(agent_spec, verse.observation_space, verse.action_space)
        verse.seed(int(seed))
        agent.seed(int(seed))
        return verse, agent

    def _inject_memory_corruption(self, severity: str) -> Dict[str, Any]:
        cfg = CentralMemoryConfig(root_dir=str(self.central_memory_dir))
        os.makedirs(str(cfg.root_dir), exist_ok=True)
        mem_path = os.path.join(str(cfg.root_dir), "memories.jsonl")

        valid_rows = 120
        ratio = _severity_to_ratio(severity)
        corrupt_rows = max(1, int(round(float(valid_rows) * float(ratio))))
        with open(mem_path, "w", encoding="utf-8") as f:
            for i in range(valid_rows):
                f.write(json.dumps({"run_id": "chaos_seed", "episode_id": f"ep_{i}", "obs": {"x": i}, "action": 0}) + "\n")
            for _ in range(corrupt_rows):
                f.write("{this is not valid json\n")
        st: SanitizeStats = sanitize_memory_file(cfg)
        return {
            "memory_path": mem_path,
            "input_lines": int(st.input_lines),
            "kept_lines": int(st.kept_lines),
            "dropped_lines": int(st.dropped_lines),
            "data_loss_pct": float(100.0 * float(st.dropped_lines) / float(max(1, st.input_lines))),
        }

    def run_fault_injection(
        self,
        *,
        fault_type: str,
        severity: str = "medium",
        trials: int = 10,
        warmup_episodes: int = 3,
        monitor_episodes: int = 10,
        normal_fault_interval_steps: float = 1200.0,
    ) -> Dict[str, Any]:
        ft = str(fault_type).strip().lower()
        if ft not in FAULT_TYPES:
            raise ValueError(f"unknown fault_type={fault_type}; expected one of {FAULT_TYPES}")
        tr = max(1, int(trials))
        results: List[Dict[str, Any]] = []

        for t in range(tr):
            trial_seed = int(self.seed) + (t * 7919)
            verse, agent = self._make_agent_verse(seed=trial_seed)
            trial_start = time.time()
            fault_step = max(2, int(self.max_steps * _severity_to_ratio(severity)))
            checkpoint_interval = max(1, int(self.max_steps // 12))

            # Warmup baseline.
            warm_returns: List[float] = []
            warm_step_perf: List[float] = []
            state_seq: List[int] = []
            for ep in range(max(1, int(warmup_episodes))):
                out = _episode_run(
                    verse=verse,
                    agent=agent,
                    max_steps=self.max_steps,
                    train=True,
                    fault_type=None,
                    fault_step=-1,
                    checkpoint_interval=checkpoint_interval,
                    baseline_step_perf=None,
                )
                warm_returns.append(float(out["return_sum"]))
                warm_step_perf.append(float(out["mean_step_perf"]))
            baseline_return = float(sum(warm_returns) / float(max(1, len(warm_returns))))
            baseline_step = float(sum(warm_step_perf) / float(max(1, len(warm_step_perf))))

            data_loss_pct = 0.0
            corruption_info = {}
            quarantine_steps: Optional[int] = None
            if ft == "memory_corrupt":
                corruption_info = self._inject_memory_corruption(severity=severity)
                data_loss_pct = float(corruption_info.get("data_loss_pct", 0.0))
                dropped = int(corruption_info.get("dropped_lines", 0))
                # Corruption recovery is bounded by detection+quarantine latency.
                # sanitize_memory_file runs synchronously, so model this as an
                # immediate control-plane recovery.
                if dropped > 0:
                    quarantine_steps = 1

            # Faulted episode.
            fault_out = _episode_run(
                verse=verse,
                agent=agent,
                max_steps=self.max_steps,
                train=True,
                fault_type=ft,
                fault_step=fault_step,
                checkpoint_interval=checkpoint_interval,
                baseline_step_perf=baseline_step,
            )
            was_bad = False
            st = _classify_state(
                perf=float(fault_out["return_sum"]),
                baseline=baseline_return,
                had_failure=bool(_safe_int(fault_out.get("crashes", 0), 0) > 0),
                was_bad=was_bad,
            )
            state_seq.append(int(st))
            was_bad = bool(st in (STATE_FAILED, STATE_DEGRADED))

            # Recovery monitor.
            recovery_steps_ep: Optional[int] = (
                _safe_int(fault_out.get("recovered_steps", -1), -1)
                if fault_out.get("recovered_steps") is not None
                else None
            )
            if recovery_steps_ep is None and quarantine_steps is not None:
                recovery_steps_ep = int(quarantine_steps)
            recovered = bool(recovery_steps_ep is not None)
            total_cascades = _safe_int(fault_out.get("cascade_failures", 0), 0)
            for ep in range(max(1, int(monitor_episodes))):
                out = _episode_run(
                    verse=verse,
                    agent=agent,
                    max_steps=self.max_steps,
                    train=True,
                    fault_type=None,
                    fault_step=-1,
                    checkpoint_interval=checkpoint_interval,
                    baseline_step_perf=None,
                )
                had_failure = bool(_safe_int(out.get("crashes", 0), 0) > 0)
                total_cascades += _safe_int(out.get("cascade_failures", 0), 0)
                perf = float(out["return_sum"])
                st = _classify_state(perf=perf, baseline=baseline_return, had_failure=had_failure, was_bad=was_bad)
                state_seq.append(int(st))
                was_bad = bool(st in (STATE_FAILED, STATE_DEGRADED))
                if not recovered and _is_recovered(perf=perf, baseline=baseline_return, tolerance=0.10):
                    recovered = True
                    # episode-level fallback if step-level was not captured.
                    if recovery_steps_ep is None:
                        recovery_steps_ep = int((ep + 1) * self.max_steps)

            markov = model_as_markov_chain(state_seq)
            elapsed = float(time.time() - trial_start)
            results.append(
                {
                    "trial": int(t),
                    "fault_type": ft,
                    "severity": str(severity),
                    "baseline_return": float(baseline_return),
                    "fault_return": float(fault_out["return_sum"]),
                    "fault_crashes": int(fault_out["crashes"]),
                    "recovered": bool(recovered),
                    "recovery_steps": (None if recovery_steps_ep is None else int(recovery_steps_ep)),
                    "recovery_time_sec": float(elapsed),
                    "data_loss_pct": float(data_loss_pct),
                    "cascade_failures": int(total_cascades),
                    "state_sequence": [_state_name(s) for s in state_seq],
                    "markov": markov,
                    "corruption": corruption_info,
                    "quarantine_steps": (None if quarantine_steps is None else int(quarantine_steps)),
                }
            )
            try:
                verse.close()
            except Exception:
                pass
            try:
                agent.close()
            except Exception:
                pass

        recovery_steps = [float(r["recovery_steps"]) for r in results if isinstance(r.get("recovery_steps"), int)]
        availability = [float((r.get("markov") or {}).get("availability", 0.0)) for r in results]
        projected_availability: List[float] = []
        for r in results:
            rs = r.get("recovery_steps")
            if isinstance(rs, int):
                mttr = float(max(1, rs))
            else:
                mttr = float(max(1, self.max_steps * max(1, int(monitor_episodes))))
            projected_availability.append(
                projected_availability_from_mttr(
                    mttr_steps=mttr,
                    normal_fault_interval_steps=float(normal_fault_interval_steps),
                )
            )
        mttf_vals = [float((r.get("markov") or {}).get("mttf", 0.0)) for r in results if math.isfinite(float((r.get("markov") or {}).get("mttf", 0.0)))]
        mttr_vals = [float((r.get("markov") or {}).get("mttr", 0.0)) for r in results if math.isfinite(float((r.get("markov") or {}).get("mttr", 0.0)))]
        data_loss = [float(r.get("data_loss_pct", 0.0)) for r in results]
        cascade = [float(r.get("cascade_failures", 0.0)) for r in results]

        return {
            "fault_type": ft,
            "severity": str(severity),
            "trials": int(tr),
            "results": results,
            "mean_recovery_steps": float(sum(recovery_steps) / float(max(1, len(recovery_steps)))) if recovery_steps else None,
            "recovery_steps_stats": compute_validation_stats(recovery_steps, min_detectable_delta=1.0) if recovery_steps else None,
            "recovered_rate": float(sum(1.0 for r in results if bool(r.get("recovered", False))) / float(max(1, tr))),
            "mean_data_loss_pct": float(sum(data_loss) / float(max(1, len(data_loss)))),
            "mean_cascade_failures": float(sum(cascade) / float(max(1, len(cascade)))),
            "availability_mean": float(sum(availability) / float(max(1, len(availability)))),
            "projected_availability_mean": float(sum(projected_availability) / float(max(1, len(projected_availability)))),
            "normal_fault_interval_steps": float(normal_fault_interval_steps),
            "mttf_mean": float(sum(mttf_vals) / float(max(1, len(mttf_vals)))) if mttf_vals else None,
            "mttr_mean": float(sum(mttr_vals) / float(max(1, len(mttr_vals)))) if mttr_vals else None,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run chaos testing fault-injection experiments.")
    ap.add_argument("--verse", type=str, default="cliff_world")
    ap.add_argument("--algo", type=str, default="q")
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup_episodes", type=int, default=3)
    ap.add_argument("--monitor_episodes", type=int, default=10)
    ap.add_argument("--normal_fault_interval_steps", type=float, default=1200.0)
    ap.add_argument("--faults", type=str, default="worker_crash,memory_corrupt,checkpoint_loss")
    ap.add_argument("--severities", type=str, default="low,medium,high")
    ap.add_argument("--central_memory_dir", type=str, default="central_memory_chaos")
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "validation", "chaos_testing.json"),
    )
    args = ap.parse_args()

    register_builtin()
    register_builtin_agents()

    faults = [str(x).strip().lower() for x in str(args.faults).replace(";", ",").split(",") if str(x).strip()]
    severities = [str(x).strip().lower() for x in str(args.severities).replace(";", ",").split(",") if str(x).strip()]
    if not faults:
        faults = list(FAULT_TYPES)
    if not severities:
        severities = ["medium"]

    tester = ChaosTest(
        verse_name=str(args.verse),
        algo=str(args.algo),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        central_memory_dir=str(args.central_memory_dir),
    )

    by_fault: Dict[str, Any] = {}
    for ft in faults:
        for sev in severities:
            key = f"{ft}:{sev}"
            print(f"\nRunning chaos fault={ft} severity={sev} trials={args.trials}")
            out = tester.run_fault_injection(
                fault_type=ft,
                severity=sev,
                trials=int(args.trials),
                warmup_episodes=int(args.warmup_episodes),
                monitor_episodes=int(args.monitor_episodes),
                normal_fault_interval_steps=float(args.normal_fault_interval_steps),
            )
            by_fault[key] = out
            print(
                f"  recovered_rate={_safe_float(out.get('recovered_rate', 0.0), 0.0):.1%} "
                f"mean_recovery_steps={out.get('mean_recovery_steps')} "
                f"availability(stress)={_safe_float(out.get('availability_mean', 0.0), 0.0):.3f} "
                f"availability(projected)={_safe_float(out.get('projected_availability_mean', 0.0), 0.0):.3f}"
            )

    report = {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": {
            "verse": str(args.verse),
            "algo": str(args.algo),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "trials": int(args.trials),
            "warmup_episodes": int(args.warmup_episodes),
            "monitor_episodes": int(args.monitor_episodes),
            "normal_fault_interval_steps": float(args.normal_fault_interval_steps),
            "faults": faults,
            "severities": severities,
            "central_memory_dir": str(args.central_memory_dir),
        },
        "by_fault": by_fault,
    }

    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nChaos summary")
    for key, out in by_fault.items():
        print(
            f"- {key}: recovered_rate={_safe_float(out.get('recovered_rate', 0.0), 0.0):.1%}, "
            f"mean_recovery_steps={out.get('mean_recovery_steps')}, "
            f"availability(stress)={_safe_float(out.get('availability_mean', 0.0), 0.0):.3f}, "
            f"availability(projected)={_safe_float(out.get('projected_availability_mean', 0.0), 0.0):.3f}"
        )
    print(f"report: {args.out_json}")


if __name__ == "__main__":
    main()
