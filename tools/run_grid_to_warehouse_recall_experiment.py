"""Near-lane recall measurement: grid_world successes into warehouse_world.

The warehouse agent is memory_recall with training and exploration off.
One condition reads a central memory built only from successful grid episodes.
The other reads an empty memory. Same seed, same warehouse layout.

recall_same_verse_only stays off. find_similar treats a set verse_name as an
exact row filter, so leaving it on would hide every grid memory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.types import AgentSpec, VerseSpec
from memory.central_repository import CentralMemoryConfig, ingest_run
from memory.selection import SelectionConfig
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer

SOURCE_VERSE = "grid_world"
TARGET_VERSE = "warehouse_world"
SOURCE_SUCCESS_MIN = 0.5


def _load_events(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return events
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                events.append(row)
    return events


def success_episode_ids(events: Sequence[Dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for event in events:
        info = event.get("info")
        if not isinstance(info, dict) or info.get("reached_goal") is not True:
            continue
        episode_id = str(event.get("episode_id", "")).strip()
        if episode_id:
            found.add(episode_id)
    return found


def write_success_only_run(*, source_dir: str, dest_dir: str) -> int:
    """Copy events from episodes that reached the goal. Returns episodes kept."""
    os.makedirs(dest_dir, exist_ok=True)
    events = _load_events(os.path.join(source_dir, "events.jsonl"))
    keep = success_episode_ids(events)
    kept = [event for event in events if str(event.get("episode_id", "")).strip() in keep]
    with open(os.path.join(dest_dir, "events.jsonl"), "w", encoding="utf-8") as handle:
        for event in kept:
            handle.write(json.dumps(event) + "\n")
    return len(keep)


def _episode_returns(events: Sequence[Dict[str, Any]]) -> List[float]:
    order: List[str] = []
    totals: Dict[str, float] = {}
    for event in events:
        episode_id = str(event.get("episode_id", ""))
        if episode_id not in totals:
            totals[episode_id] = 0.0
            order.append(episode_id)
        try:
            totals[episode_id] += float(event.get("reward", 0.0))
        except (TypeError, ValueError):
            continue
    return [totals[episode_id] for episode_id in order]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def summarize_recall_run(run_dir: str, *, early_episodes: int = 10) -> Dict[str, Any]:
    events = _load_events(os.path.join(run_dir, "events.jsonl"))
    queries_requested = 0
    queries_executed = 0
    recalls_used = 0
    recalls_eligible = 0
    gate_passed = 0
    match_steps = 0
    grid_top_hits = 0
    positive_score_hits = 0
    top_scores: List[float] = []
    decision_reasons: Dict[str, int] = {}
    source_verses: Dict[str, int] = {}
    battery_deaths = 0
    battery_death_episodes: set[str] = set()
    hit_obstacle = 0
    hit_wall = 0
    hit_patrol = 0

    for event in events:
        info = event.get("info") if isinstance(event.get("info"), dict) else {}
        query = info.get("memory_query") if isinstance(info.get("memory_query"), dict) else {}
        action_info = info.get("action_info") if isinstance(info.get("action_info"), dict) else {}
        queries_requested += int(bool(query.get("query_requested", False)))
        queries_executed += int(bool(query.get("query_executed", False)))
        recalls_used += int(bool(action_info.get("memory_recall_used", False)))
        recalls_eligible += int(bool(action_info.get("memory_recall_eligible", False)))
        gate_passed += int(bool(action_info.get("memory_recall_gate_passed", False)))
        if bool(info.get("battery_death", False)):
            battery_deaths += 1
            battery_death_episodes.add(str(event.get("episode_id", "")))
        hit_obstacle += int(bool(info.get("hit_obstacle", False)))
        hit_wall += int(bool(info.get("hit_wall", False)))
        hit_patrol += int(bool(info.get("hit_patrol", False)))

        records = info.get("transfer_decision_records")
        record = records[0] if isinstance(records, list) and records and isinstance(records[0], dict) else {}
        reason = str(record.get("decision_reason", "")).strip()
        if reason:
            decision_reasons[reason] = int(decision_reasons.get(reason, 0)) + 1
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        verse_name = str(source.get("verse_name", "")).strip().lower()
        if verse_name and verse_name != "unknown":
            match_steps += 1
            source_verses[verse_name] = int(source_verses.get(verse_name, 0)) + 1
            if verse_name == SOURCE_VERSE:
                grid_top_hits += 1
            score = source.get("score")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = None
            if score_value is not None:
                top_scores.append(score_value)
                if score_value > 0.0:
                    positive_score_hits += 1

    returns = _episode_returns(events)
    early_n = max(1, int(early_episodes))
    return {
        "steps": int(len(events)),
        "episodes": int(len(returns)),
        "early_window_episodes": int(min(early_n, len(returns))),
        "early_mean_return": _mean(returns[:early_n]),
        "queries_requested": int(queries_requested),
        "queries_executed": int(queries_executed),
        "memory_recall_eligible": int(recalls_eligible),
        "memory_recall_gate_passed": int(gate_passed),
        "memory_recall_used": int(recalls_used),
        "match_steps": int(match_steps),
        "grid_top_hits": int(grid_top_hits),
        "positive_score_hits": int(positive_score_hits),
        "mean_top_score": (None if not top_scores else _mean(top_scores)),
        "source_verse_counts": source_verses,
        "decision_reasons": decision_reasons,
        "battery_deaths": int(battery_deaths),
        "battery_death_episodes": int(len(battery_death_episodes)),
        "hit_obstacle": int(hit_obstacle),
        "hit_wall": int(hit_wall),
        "hit_patrol": int(hit_patrol),
    }


def recall_verdict(
    *,
    source_success_rate: Optional[float],
    ingested_events: int,
    grid: Dict[str, Any],
    baseline: Dict[str, Any],
    grid_mean_return: float,
    baseline_mean_return: float,
    grid_success_rate: Optional[float],
    baseline_success_rate: Optional[float],
) -> Dict[str, Any]:
    success = 0.0 if source_success_rate is None else float(source_success_rate)
    return_lift = float(grid_mean_return) - float(baseline_mean_return)
    success_lift = float(grid_success_rate or 0.0) - float(baseline_success_rate or 0.0)
    battery_delta = int(baseline.get("battery_death_episodes", 0)) - int(grid.get("battery_death_episodes", 0))
    outcome_beats_baseline = bool(return_lift > 0.0 or success_lift > 0.0 or battery_delta > 0)

    if success < SOURCE_SUCCESS_MIN or int(ingested_events) <= 0:
        mechanism = "source_not_competent"
    elif int(grid.get("queries_executed", 0)) <= 0:
        mechanism = "recall_did_not_query"
    elif int(grid.get("grid_top_hits", 0)) <= 0:
        mechanism = "no_grid_hits"
    elif int(grid.get("memory_recall_used", 0)) <= 0 and int(grid.get("positive_score_hits", 0)) <= 0:
        mechanism = "nonpositive_similarity"
    elif int(grid.get("memory_recall_used", 0)) <= 0:
        mechanism = "hits_not_applied"
    else:
        mechanism = "recall_applied"

    if mechanism != "recall_applied":
        primary = mechanism
    elif outcome_beats_baseline:
        primary = "lift"
    else:
        primary = "applied_without_lift"

    return {
        "primary": primary,
        "mechanism": mechanism,
        "outcome_beats_baseline": outcome_beats_baseline,
        "source_success_rate": float(success),
        "source_success_min": float(SOURCE_SUCCESS_MIN),
        "ingested_events": int(ingested_events),
        "return_lift": float(return_lift),
        "success_rate_lift": float(success_lift),
        "battery_death_episode_delta": int(battery_delta),
        "pass": bool(primary == "lift"),
    }


def _grid_verse(seed: int) -> VerseSpec:
    return VerseSpec(
        spec_version="v1",
        verse_name=SOURCE_VERSE,
        verse_version="0.1",
        seed=seed,
        tags=["near_lane_recall"],
        params={
            "width": 5,
            "height": 5,
            "start_x": 0,
            "start_y": 0,
            "goal_x": -1,
            "goal_y": -1,
            "max_steps": 20,
            "obstacle_count": 1,
            "ice_count": 0,
            "teleporter_pairs": 0,
            "step_penalty": -0.01,
            "adr_enabled": False,
        },
    )


def _warehouse_verse(seed: int) -> VerseSpec:
    return VerseSpec(
        spec_version="v1",
        verse_name=TARGET_VERSE,
        verse_version="0.1",
        seed=seed,
        tags=["near_lane_recall"],
        params={
            "width": 5,
            "height": 5,
            "start_x": 0,
            "start_y": 0,
            "goal_x": -1,
            "goal_y": -1,
            "max_steps": 20,
            "obstacle_count": 1,
            "battery_capacity": 30,
            "battery_drain": 1,
            "patrol_robot": False,
            "conveyor_count": 0,
            "step_penalty": -0.10,
            "adr_enabled": False,
        },
    )


def _source_agent(seed: int) -> AgentSpec:
    return AgentSpec(
        spec_version="v1",
        policy_id="grid_world_q_source",
        policy_version="0.1",
        algo="q",
        seed=seed,
        config={
            "train": True,
            "epsilon_start": 1.0,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.97,
            "lr": 0.3,
            "gamma": 0.95,
            "learn_success_bonus": 10.0,
            "rar_enabled": False,
        },
    )


def _recall_agent(*, seed: int, memory_root: str, policy_id: str) -> AgentSpec:
    return AgentSpec(
        spec_version="v1",
        policy_id=policy_id,
        policy_version="0.1",
        algo="memory_recall",
        seed=seed,
        config={
            "train": False,
            "epsilon_start": 0.0,
            "epsilon_min": 0.0,
            "epsilon_decay": 1.0,
            "rar_enabled": False,
            "on_demand_memory_enabled": True,
            "on_demand_memory_root": memory_root,
            "on_demand_query_budget": 100000,
            "on_demand_min_interval": 1,
            "recall_top_k": 5,
            "recall_min_score": -1.0,
            "recall_same_verse_only": False,
            "recall_vote_weight": 2.0,
            "recall_uncertainty_margin": 0.1,
            "recall_cooldown_steps": 1,
            "recall_quality_weight": 0.0,
        },
    )


def _condition_block(stats: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mean_return": float(stats.mean_return),
        "success_rate": stats.success_rate,
        "mean_steps": float(stats.mean_steps),
        **summary,
    }


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs_grid_to_warehouse_recall")
    parser.add_argument("--source-seed", type=int, default=17)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--source-episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--early-episodes", type=int, default=10)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = os.path.abspath(args.output_root)
    if os.path.isdir(root):
        shutil.rmtree(root)
    runs_root = os.path.join(root, "runs")
    baseline_memory = os.path.join(root, "baseline_empty_memory")
    grid_memory = os.path.join(root, "grid_memory")
    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(baseline_memory, exist_ok=True)
    os.makedirs(grid_memory, exist_ok=True)

    trainer = Trainer(run_root=runs_root, schema_version="v1", auto_register_builtin=True)
    source_verse = _grid_verse(args.source_seed)
    target_verse = _warehouse_verse(args.eval_seed)

    source = trainer.run(
        verse_spec=source_verse,
        agent_spec=_source_agent(args.source_seed),
        episodes=args.source_episodes,
        max_steps=20,
        seed=args.source_seed,
        run_id="source_q",
        auto_index=False,
        verbose=False,
    )
    source_dir = os.path.join(runs_root, str(source["run_id"]))
    source_stats = evaluate_run(source_dir)
    success_dir = os.path.join(runs_root, "source_successes")
    success_episodes = write_success_only_run(source_dir=source_dir, dest_dir=success_dir)
    ingested_events = 0
    ingest_stats: Optional[Dict[str, Any]] = None
    if success_episodes > 0:
        stats = ingest_run(
            run_dir=success_dir,
            cfg=CentralMemoryConfig(root_dir=grid_memory),
            selection=SelectionConfig(keep_successes=True, keep_top_k_episodes=0, keep_top_k_per_episode=0),
        )
        ingested_events = int(stats.added_events)
        ingest_stats = {
            "input_events": int(stats.input_events),
            "selected_events": int(stats.selected_events),
            "added_events": int(stats.added_events),
            "skipped_duplicates": int(stats.skipped_duplicates),
        }

    def _eval(policy_id: str, memory_root: str, run_id: str) -> str:
        out = trainer.run(
            verse_spec=target_verse,
            agent_spec=_recall_agent(seed=args.eval_seed, memory_root=memory_root, policy_id=policy_id),
            episodes=args.eval_episodes,
            max_steps=20,
            seed=args.eval_seed,
            run_id=run_id,
            auto_index=False,
            verbose=False,
        )
        return os.path.join(runs_root, str(out["run_id"]))

    baseline_dir = _eval("recall_baseline_empty_memory", baseline_memory, "baseline")
    grid_dir = _eval("recall_grid_memory", grid_memory, "grid_memory")
    baseline_stats = evaluate_run(baseline_dir)
    grid_stats = evaluate_run(grid_dir)
    baseline_summary = summarize_recall_run(baseline_dir, early_episodes=args.early_episodes)
    grid_summary = summarize_recall_run(grid_dir, early_episodes=args.early_episodes)
    verdict = recall_verdict(
        source_success_rate=source_stats.success_rate,
        ingested_events=ingested_events,
        grid=grid_summary,
        baseline=baseline_summary,
        grid_mean_return=float(grid_stats.mean_return),
        baseline_mean_return=float(baseline_stats.mean_return),
        grid_success_rate=grid_stats.success_rate,
        baseline_success_rate=baseline_stats.success_rate,
    )
    result: Dict[str, Any] = {
        "design": {
            "source_verse": SOURCE_VERSE,
            "target_verse": TARGET_VERSE,
            "lane": "near_universe",
            "agent_algo": "memory_recall",
            "recall_same_verse_only": False,
            "train_on_target": False,
            "source_seed": args.source_seed,
            "evaluation_seed": args.eval_seed,
            "source_params": dict(source_verse.params),
            "target_params": dict(target_verse.params),
            "source_episodes": args.source_episodes,
            "evaluation_episodes": args.eval_episodes,
            "memory": "successful grid_world episodes only, compared with an empty repository",
        },
        "source": {
            "mean_return": source_stats.mean_return,
            "success_rate": source_stats.success_rate,
            "mean_steps": source_stats.mean_steps,
            "success_episodes": int(success_episodes),
            "ingest_stats": ingest_stats,
        },
        "baseline": _condition_block(baseline_stats, baseline_summary),
        "grid_memory": _condition_block(grid_stats, grid_summary),
        "verdict": verdict,
    }
    result_path = os.path.join(root, "result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"result_path={result_path}")
    print(f"verdict={verdict['primary']} pass={verdict['pass']}")
    return result


if __name__ == "__main__":
    main()
