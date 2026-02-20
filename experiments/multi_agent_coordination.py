"""
experiments/multi_agent_coordination.py

Phase 2.1 multi-agent coordination experiment on factory_world.

Compares three coordination modes:
1) isolated   : no shared state
2) shared_ro  : shared read-only hints (last-step summary)
3) shared_rw  : shared read-write reservations + shared hints
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec
from tools.validation_stats import compute_validation_stats
from verses.registry import create_verse, register_builtin


Mode = str
Strategy = str


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


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0.0:
        return 0.0
    h = 0.0
    for v in counts.values():
        p = float(v) / total
        if p > 0.0:
            h -= p * math.log(p + 1e-12)
    return float(h)


def _joint_entropy_from_pairs(pairs: Sequence[Tuple[int, int]]) -> float:
    counts: Dict[Tuple[int, int], int] = {}
    for p in pairs:
        counts[p] = int(counts.get(p, 0)) + 1
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    h = 0.0
    for v in counts.values():
        p = float(v) / total
        if p > 0.0:
            h -= p * math.log(p + 1e-12)
    return float(h)


def mutual_information(xs: Sequence[int], ys: Sequence[int]) -> float:
    n = min(len(xs), len(ys))
    if n <= 0:
        return 0.0
    x_counts: Dict[int, int] = {}
    y_counts: Dict[int, int] = {}
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        x = int(xs[i])
        y = int(ys[i])
        x_counts[x] = int(x_counts.get(x, 0)) + 1
        y_counts[y] = int(y_counts.get(y, 0)) + 1
        pairs.append((x, y))
    hx = _entropy_from_counts(x_counts)
    hy = _entropy_from_counts(y_counts)
    hxy = _joint_entropy_from_pairs(pairs)
    return float(max(0.0, hx + hy - hxy))


@dataclass
class EpisodeCoordinationMetrics:
    completion_time: int
    completed_items: int
    success: bool
    redundancy: float
    resource_conflicts: int
    information_utilization: float
    shared_information: float
    coordination_score: float
    per_agent_rewards: List[float]
    per_agent_actions: List[List[int]]


class CoordinationExperiment:
    def __init__(
        self,
        *,
        num_agents: int = 2,
        memory_mode: Mode = "shared_rw",
        seed: int = 123,
        max_steps: int = 220,
        target_completed: int = 12,
    ):
        self.num_agents = max(1, int(num_agents))
        self.memory_mode = str(memory_mode).strip().lower()
        self.seed = int(seed)
        self.max_steps = max(40, int(max_steps))
        self.target_completed = max(1, int(target_completed))

    def _factory_spec(self, ep_seed: int) -> VerseSpec:
        return VerseSpec(
            spec_version="v1",
            verse_name="factory_world",
            verse_version="0.1",
            seed=int(ep_seed),
            params={
                "max_steps": int(self.max_steps),
                "num_machines": 3,
                "buffer_size": 4,
                "arrival_rate": 0.60,
                "breakdown_prob": 0.08,
                "repair_steps": 3,
                "step_penalty": -0.02,
                "completion_reward": 2.0,
                "overflow_penalty": -0.5,
                "idle_penalty": -0.1,
                "adr_enabled": False,
            },
        )

    def _default_strategy_for_mode(self) -> Strategy:
        if self.memory_mode == "shared_rw":
            return "reserved_greedy"
        if self.memory_mode == "shared_ro":
            return "shared_hint_greedy"
        return "local_greedy"

    def _select_action(
        self,
        *,
        obs: Dict[str, Any],
        rng: random.Random,
        strategy: Strategy,
        reservations: Dict[int, int],
        shared_hint: Dict[str, Any],
        machine_n: int,
        agent_id: int,
    ) -> Tuple[int, bool]:
        # build machine scores
        machine_scores: List[Tuple[float, int]] = []
        for m in range(machine_n):
            buf = _safe_int(obs.get(f"buf_{m}", 0), 0)
            broken = bool(_safe_int(obs.get(f"broken_{m}", 0), 0) > 0)
            repair_timer = _safe_int(obs.get(f"repair_{m}", 0), 0)

            # Repair broken machines first when they have work waiting.
            if broken and (buf > 0 or repair_timer > 0):
                # repair action index is machine_n + m
                return machine_n + m, False

            score = float(buf)
            if broken:
                score -= 2.0
            # shared hint nudges away from repeated over-selected machines
            if strategy in ("shared_hint_greedy", "reserved_greedy"):
                pressure = float(shared_hint.get("machine_pressure", {}).get(m, 0.0))
                score -= 0.25 * pressure
            machine_scores.append((score, m))

        machine_scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        fallback = machine_n * 2  # idle action
        if not machine_scores:
            return fallback, False

        used_shared = False
        if strategy == "random":
            m = int(rng.randint(0, machine_n - 1))
            return m, False

        if strategy == "shared_hint_greedy":
            # If two top machines tie, use previous global choice bias.
            top_score = machine_scores[0][0]
            cands = [m for s, m in machine_scores if abs(float(s) - float(top_score)) <= 1e-9]
            if len(cands) > 1:
                pref = int(shared_hint.get("preferred_machine", cands[0]))
                if pref in cands:
                    used_shared = True
                    return int(pref), used_shared
            return int(machine_scores[0][1]), used_shared

        if strategy == "reserved_greedy":
            for _, m in machine_scores:
                if int(reservations.get(m, -1)) < 0:
                    reservations[m] = int(agent_id)
                    used_shared = True
                    return int(m), used_shared
            # all reserved: fallback to best machine
            m = int(machine_scores[0][1])
            return m, used_shared

        # local_greedy
        return int(machine_scores[0][1]), False

    def run_multi_agent_episode(
        self,
        *,
        episode_idx: int,
        strategy_overrides: Optional[Dict[int, Strategy]] = None,
    ) -> EpisodeCoordinationMetrics:
        ep_seed = int(self.seed) + int(episode_idx) * 1009
        rng = random.Random(ep_seed + 17)
        verse = create_verse(self._factory_spec(ep_seed))
        verse.seed(ep_seed)
        rr = verse.reset()
        obs = rr.obs if isinstance(rr.obs, dict) else {}

        default_strategy = self._default_strategy_for_mode()
        per_agent_rewards = [0.0 for _ in range(self.num_agents)]
        per_agent_actions: List[List[int]] = [[] for _ in range(self.num_agents)]

        step_count = 0
        completed_items = 0
        conflict_count = 0
        duplicate_actions = 0
        shared_use = 0
        decision_count = 0
        round_count = 0

        agent_action_series: List[List[int]] = [[] for _ in range(self.num_agents)]
        shared_hint: Dict[str, Any] = {"machine_pressure": {}}

        done = False
        while (not done) and step_count < int(self.max_steps):
            round_count += 1
            round_actions: Dict[int, int] = {}
            reservations: Dict[int, int] = {}
            machine_n = max(1, int((_safe_int(getattr(verse.action_space, "n", 1), 1) - 1) / 2))

            for aid in range(self.num_agents):
                strategy = default_strategy
                if isinstance(strategy_overrides, dict) and int(aid) in strategy_overrides:
                    strategy = str(strategy_overrides[int(aid)])

                action, used_shared = self._select_action(
                    obs=obs,
                    rng=rng,
                    strategy=strategy,
                    reservations=reservations,
                    shared_hint=shared_hint,
                    machine_n=machine_n,
                    agent_id=aid,
                )
                decision_count += 1
                if used_shared:
                    shared_use += 1

                if action < machine_n:
                    round_actions[action] = int(round_actions.get(action, 0)) + 1

                sr = verse.step(int(action))
                info = sr.info if isinstance(sr.info, dict) else {}
                reward = float(sr.reward)
                per_agent_rewards[aid] += reward
                per_agent_actions[aid].append(int(action))
                agent_action_series[aid].append(int(action))
                step_count += 1

                if "items_completed" in info:
                    completed_items += _safe_int(info.get("items_completed", 0), 0)
                # conflict proxies
                if bool(info.get("empty_buffer", False)) or bool(info.get("machine_broken", False)):
                    conflict_count += 1
                if bool(info.get("overflow", False)) or bool(info.get("arrival_blocked", False)):
                    conflict_count += 1
                if bool(info.get("unnecessary_repair", False)):
                    conflict_count += 1

                obs = sr.obs if isinstance(sr.obs, dict) else {}
                if completed_items >= int(self.target_completed):
                    done = True
                    break
                if bool(sr.done or sr.truncated):
                    done = True
                    break

            for cnt in round_actions.values():
                if int(cnt) > 1:
                    duplicate_actions += int(cnt - 1)

            # update shared hint from observed contention
            pressure = shared_hint.get("machine_pressure", {})
            if not isinstance(pressure, dict):
                pressure = {}
            for m, cnt in round_actions.items():
                old = float(pressure.get(m, 0.0))
                pressure[m] = (0.8 * old) + (0.2 * float(cnt))
            if round_actions:
                # preferred machine = least contended among chosen this round
                preferred = min(round_actions.items(), key=lambda kv: kv[1])[0]
                shared_hint["preferred_machine"] = int(preferred)
            shared_hint["machine_pressure"] = pressure

        verse.close()

        redundancy = float(duplicate_actions / float(max(1, decision_count)))
        completion_time = int(step_count)
        info_util = float(shared_use / float(max(1, decision_count)))

        # shared information proxy: average pairwise action MI across agents
        pair_mi: List[float] = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pair_mi.append(mutual_information(agent_action_series[i], agent_action_series[j]))
        shared_info = float(sum(pair_mi) / float(max(1, len(pair_mi)))) if pair_mi else 0.0

        coord_score = float(
            (1.0 - redundancy)
            * (1.0 / float(max(1, completion_time)))
            * (1.0 + min(1.0, float(completed_items) / float(max(1, self.target_completed))))
            * (1.0 / (1.0 + float(conflict_count) / float(max(1, completion_time))))
        )

        return EpisodeCoordinationMetrics(
            completion_time=int(completion_time),
            completed_items=int(completed_items),
            success=bool(completed_items >= int(self.target_completed)),
            redundancy=float(redundancy),
            resource_conflicts=int(conflict_count),
            information_utilization=float(info_util),
            shared_information=float(shared_info),
            coordination_score=float(coord_score),
            per_agent_rewards=[float(x) for x in per_agent_rewards],
            per_agent_actions=per_agent_actions,
        )

    def _best_response_gain(
        self,
        *,
        episode_idx: int,
        baseline_rewards: List[float],
    ) -> float:
        strategies = ("local_greedy", "shared_hint_greedy", "reserved_greedy", "random")
        gains: List[float] = []
        for aid in range(self.num_agents):
            base = float(baseline_rewards[aid]) if aid < len(baseline_rewards) else 0.0
            best_alt = base
            for st in strategies:
                ep = self.run_multi_agent_episode(
                    episode_idx=int(episode_idx),
                    strategy_overrides={int(aid): str(st)},
                )
                alt = float(ep.per_agent_rewards[aid]) if aid < len(ep.per_agent_rewards) else 0.0
                if alt > best_alt:
                    best_alt = alt
            gains.append(float(best_alt - base))
        return float(max(gains) if gains else 0.0)

    def run_experiment(self, *, episodes: int = 100, estimate_nash_every: int = 10) -> Dict[str, Any]:
        rows: List[EpisodeCoordinationMetrics] = []
        nash_distances: List[float] = []

        for ep in range(max(1, int(episodes))):
            m = self.run_multi_agent_episode(episode_idx=ep)
            rows.append(m)
            if int(ep) % max(1, int(estimate_nash_every)) == 0:
                nash_distances.append(
                    self._best_response_gain(episode_idx=ep, baseline_rewards=list(m.per_agent_rewards))
                )

        coord_scores = [float(r.coordination_score) for r in rows]
        comp_times = [float(r.completion_time) for r in rows]
        redund = [float(r.redundancy) for r in rows]
        conflicts = [float(r.resource_conflicts) for r in rows]
        info_util = [float(r.information_utilization) for r in rows]
        shared_info = [float(r.shared_information) for r in rows]
        success = [1.0 if bool(r.success) else 0.0 for r in rows]
        completed = [float(r.completed_items) for r in rows]

        return {
            "memory_mode": self.memory_mode,
            "num_agents": int(self.num_agents),
            "episodes": int(len(rows)),
            "mean_coordination": float(sum(coord_scores) / float(max(1, len(coord_scores)))),
            "mean_completion_time": float(sum(comp_times) / float(max(1, len(comp_times)))),
            "mean_redundancy": float(sum(redund) / float(max(1, len(redund)))),
            "mean_resource_conflicts": float(sum(conflicts) / float(max(1, len(conflicts)))),
            "mean_information_utilization": float(sum(info_util) / float(max(1, len(info_util)))),
            "mean_shared_information": float(sum(shared_info) / float(max(1, len(shared_info)))),
            "success_rate": float(sum(success) / float(max(1, len(success)))),
            "mean_completed_items": float(sum(completed) / float(max(1, len(completed)))),
            "coordination_stats": compute_validation_stats(coord_scores, min_detectable_delta=0.01),
            "completion_stats": compute_validation_stats(comp_times, min_detectable_delta=1.0),
            "nash_distance_mean": float(sum(nash_distances) / float(max(1, len(nash_distances)))) if nash_distances else 0.0,
            "nash_distance_series": [float(x) for x in nash_distances],
        }


def _efficiency_vs_single_agent(
    *,
    multi: Dict[str, Any],
    baseline_single: Dict[str, Any],
) -> float:
    # >1 means faster per-agent throughput than single-agent baseline.
    multi_time = max(1e-9, _safe_float(multi.get("mean_completion_time", 0.0), 0.0))
    single_time = max(1e-9, _safe_float(baseline_single.get("mean_completion_time", 0.0), 0.0))
    return float(single_time / multi_time)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-agent coordination experiment.")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--num_agents", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=220)
    ap.add_argument("--target_completed", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--estimate_nash_every", type=int, default=10)
    ap.add_argument(
        "--modes",
        type=str,
        default="isolated,shared_ro,shared_rw",
        help="Comma-separated modes",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "validation", "multi_agent_coordination.json"),
    )
    args = ap.parse_args()

    register_builtin()

    modes = [str(x).strip().lower() for x in str(args.modes).replace(";", ",").split(",") if str(x).strip()]
    if not modes:
        modes = ["isolated", "shared_ro", "shared_rw"]

    by_mode: Dict[str, Any] = {}
    for mode in modes:
        exp = CoordinationExperiment(
            num_agents=int(args.num_agents),
            memory_mode=str(mode),
            seed=int(args.seed),
            max_steps=int(args.max_steps),
            target_completed=int(args.target_completed),
        )
        print(f"\nRunning mode={mode} agents={args.num_agents} episodes={args.episodes}")
        by_mode[mode] = exp.run_experiment(
            episodes=int(args.episodes),
            estimate_nash_every=int(args.estimate_nash_every),
        )
        print(
            f"  coordination={_safe_float(by_mode[mode].get('mean_coordination', 0.0), 0.0):.6f} "
            f"success={_safe_float(by_mode[mode].get('success_rate', 0.0), 0.0):.1%} "
            f"nash_distance={_safe_float(by_mode[mode].get('nash_distance_mean', 0.0), 0.0):.4f}"
        )

    # Single-agent baseline for efficiency comparison.
    single = CoordinationExperiment(
        num_agents=1,
        memory_mode="isolated",
        seed=int(args.seed) + 999,
        max_steps=int(args.max_steps),
        target_completed=int(args.target_completed),
    ).run_experiment(episodes=max(10, int(args.episodes // 2)), estimate_nash_every=999999)

    for mode, row in by_mode.items():
        row["efficiency_vs_single_agent"] = _efficiency_vs_single_agent(multi=row, baseline_single=single)

    isolated = by_mode.get("isolated", {})
    shared_rw = by_mode.get("shared_rw", {})
    isolated_coord = _safe_float(isolated.get("mean_coordination", 0.0), 0.0)
    shared_rw_coord = _safe_float(shared_rw.get("mean_coordination", 0.0), 0.0)
    coord_improvement_pct = (
        0.0 if isolated_coord <= 1e-12 else ((shared_rw_coord / isolated_coord) - 1.0) * 100.0
    )

    report = {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": {
            "episodes": int(args.episodes),
            "num_agents": int(args.num_agents),
            "max_steps": int(args.max_steps),
            "target_completed": int(args.target_completed),
            "seed": int(args.seed),
            "estimate_nash_every": int(args.estimate_nash_every),
            "modes": modes,
        },
        "single_agent_baseline": single,
        "by_mode": by_mode,
        "comparisons": {
            "coordination_improvement_shared_rw_vs_isolated_pct": float(coord_improvement_pct),
            "nash_distance_delta_shared_rw_vs_isolated": float(
                _safe_float(shared_rw.get("nash_distance_mean", 0.0), 0.0)
                - _safe_float(isolated.get("nash_distance_mean", 0.0), 0.0)
            ),
        },
    }

    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nCoordination summary")
    print(
        f"shared_rw vs isolated coordination improvement: "
        f"{float(report['comparisons']['coordination_improvement_shared_rw_vs_isolated_pct']):.2f}%"
    )
    print(
        "shared_rw nash distance mean: "
        f"{_safe_float(shared_rw.get('nash_distance_mean', 0.0), 0.0):.4f}"
    )
    print(f"report: {args.out_json}")


if __name__ == "__main__":
    main()
