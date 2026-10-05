"""Controlled native CliffWorld experiment for central-memory recall lift."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.types import AgentSpec, VerseSpec
from memory.central_repository import CentralMemoryConfig, ingest_run
from memory.selection import SelectionConfig
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer


def _count_recall_events(run_dir: str) -> Dict[str, int]:
    queries_requested = 0
    queries_executed = 0
    recalls_used = 0
    recall_actions_correct = 0
    recall_actions_adversarial = 0
    recall_actions_other = 0
    with open(os.path.join(run_dir, "events.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            info = event.get("info") or {}
            query = info.get("memory_query") or {}
            action_info = info.get("action_info") or {}
            queries_requested += int(bool(query.get("query_requested", False)))
            queries_executed += int(bool(query.get("query_executed", False)))
            recalls_used += int(bool(action_info.get("memory_recall_used", False)))
            if bool(action_info.get("memory_recall_used", False)):
                action = int(event.get("action", -1))
                if action == 3:
                    recall_actions_correct += 1
                elif action == 1:
                    recall_actions_adversarial += 1
                else:
                    recall_actions_other += 1
    return {
        "queries_requested": queries_requested,
        "queries_executed": queries_executed,
        "memory_recall_used": recalls_used,
        "recall_selected_correct_action": recall_actions_correct,
        "recall_selected_adversarial_action": recall_actions_adversarial,
        "recall_selected_other_action": recall_actions_other,
    }


def _line_verse(seed: int) -> VerseSpec:
    return VerseSpec(
        spec_version="v1",
        verse_name="line_world",
        verse_version="0.1",
        seed=seed,
        params={"lane_len": 10, "max_steps": 10, "adr_enabled": False},
    )


def _random_agent(seed: int, policy_id: str) -> AgentSpec:
    return AgentSpec(
        spec_version="v1",
        policy_id=policy_id,
        policy_version="0.1",
        algo="random",
        seed=seed,
        config={"train": False, "rar_enabled": False},
    )


def _write_adversarial_run(*, clean_memory_root: str, run_dir: str) -> int:
    os.makedirs(run_dir, exist_ok=True)
    clean_path = os.path.join(clean_memory_root, "memories.jsonl")
    rows = []
    with open(clean_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("verse_name")) == "cliff_world":
                rows.append(row)
    events_path = os.path.join(run_dir, "events.jsonl")
    with open(events_path, "w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            obs = dict(row.get("obs") or {})
            # Action 1 moves down toward/into the cliff; correct source action is 3 (right).
            next_obs = dict(obs)
            next_obs["y"] = 1
            next_obs["t"] = int(obs.get("t", 0)) + 1
            event = {
                "schema_version": "v1",
                "run_id": "adversarial_exact_match",
                "episode_id": "adversarial_ep",
                "step_idx": idx,
                "t_ms": 2_000_000_000_000 + idx,
                "verse_name": "cliff_world",
                "policy_id": "adversarial_cliff_policy",
                "obs": obs,
                "next_obs": next_obs,
                "action": 1,
                "reward": 100.0,
                "done": False,
                "truncated": False,
                "transfer_score": 10.0,
                "transfer_confidence": 1.0,
                "info": {"synthetic_adversarial": True, "reached_goal": False},
            }
            handle.write(json.dumps(event) + "\n")
    return len(rows)


def _transfer_scores_by_policy(memory_root: str) -> Dict[str, list[float]]:
    scores: Dict[str, set[float]] = {}
    with open(os.path.join(memory_root, "memories.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            score = row.get("transfer_score")
            if score is None:
                continue
            policy_id = str(row.get("policy_id", ""))
            scores.setdefault(policy_id, set()).add(float(score))
    return {policy_id: sorted(values) for policy_id, values in scores.items()}


def _verse(seed: int, *, width: int) -> VerseSpec:
    return VerseSpec(
        spec_version="v1",
        verse_name="cliff_world",
        verse_version="0.1",
        seed=seed,
        tags=["controlled_recall_experiment"],
        params={
            "width": width,
            "height": 2,
            "start_y": 0,
            "goal_y": 0,
            "max_steps": 10,
            "step_penalty": -1.0,
            "cliff_penalty": -100.0,
            "end_on_cliff": False,
            "wind_probability": 0.0,
            "crumble_probability": 0.0,
            "adr_enabled": False,
        },
    )


def _recall_agent(
    *,
    seed: int,
    memory_root: str,
    policy_id: str,
    recall_min_transfer_score: float | None = None,
    recall_quality_weight: float = 0.0,
) -> AgentSpec:
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
            "on_demand_query_budget": 100,
            "on_demand_min_interval": 1,
            "recall_top_k": 25,
            "recall_min_score": -1.0,
            "recall_same_verse_only": True,
            "recall_vote_weight": 2.0,
            "recall_uncertainty_margin": 0.1,
            "recall_cooldown_steps": 1,
            "recall_quality_weight": float(recall_quality_weight),
            **(
                {"recall_min_transfer_score": recall_min_transfer_score}
                if recall_min_transfer_score is not None
                else {}
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs_memory_recall_experiment")
    parser.add_argument("--source-seed", type=int, default=17)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--source-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    root = os.path.abspath(args.output_root)
    if os.path.isdir(root):
        shutil.rmtree(root)
    runs_root = os.path.join(root, "runs")
    baseline_memory = os.path.join(root, "baseline_empty_memory")
    clean_memory = os.path.join(root, "clean_memory")
    noisy_memory = os.path.join(root, "noisy_memory")
    adversarial_memory = os.path.join(root, "adversarial_memory")
    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(baseline_memory, exist_ok=True)
    os.makedirs(clean_memory, exist_ok=True)

    trainer = Trainer(run_root=runs_root, schema_version="v1", auto_register_builtin=True)
    source_verse = _verse(args.source_seed, width=4)
    evaluation_verse = _verse(args.eval_seed, width=5)

    source_agent = AgentSpec(
        spec_version="v1",
        policy_id="cliff_world_q_source",
        policy_version="0.1",
        algo="q",
        seed=args.source_seed,
        config={
            "train": True,
            "epsilon_start": 1.0,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.97,
            "lr": 0.3,
            "gamma": 0.95,
            "learn_success_bonus": 100.0,
            "rar_enabled": False,
        },
    )
    source = trainer.run(
        verse_spec=source_verse,
        agent_spec=source_agent,
        episodes=args.source_episodes,
        max_steps=10,
        seed=args.source_seed,
        run_id="source_q",
        auto_index=False,
        verbose=False,
    )
    source_dir = os.path.join(runs_root, str(source["run_id"]))
    source_stats = evaluate_run(source_dir)

    clean_ingest_stats = ingest_run(
        run_dir=source_dir,
        cfg=CentralMemoryConfig(root_dir=clean_memory),
        selection=SelectionConfig(
            keep_successes=True,
            keep_top_k_episodes=100,
            keep_top_k_per_episode=0,
        ),
    )

    random_cliff = trainer.run(
        verse_spec=source_verse,
        agent_spec=_random_agent(99, "random_cliff_noise"),
        episodes=500,
        max_steps=10,
        seed=99,
        run_id="noise_random_cliff",
        auto_index=False,
        verbose=False,
    )
    random_line = trainer.run(
        verse_spec=_line_verse(101),
        agent_spec=_random_agent(101, "random_line_noise"),
        episodes=500,
        max_steps=10,
        seed=101,
        run_id="noise_random_line",
        auto_index=False,
        verbose=False,
    )

    shutil.copytree(clean_memory, noisy_memory)
    noisy_cfg = CentralMemoryConfig(root_dir=noisy_memory)
    cliff_noise_stats = ingest_run(
        run_dir=os.path.join(runs_root, str(random_cliff["run_id"])),
        cfg=noisy_cfg,
        selection=SelectionConfig(),
    )
    line_noise_stats = ingest_run(
        run_dir=os.path.join(runs_root, str(random_line["run_id"])),
        cfg=noisy_cfg,
        selection=SelectionConfig(),
    )

    shutil.copytree(noisy_memory, adversarial_memory)
    adversarial_run_dir = os.path.join(runs_root, "adversarial_exact_match")
    adversarial_events_written = _write_adversarial_run(
        clean_memory_root=clean_memory,
        run_dir=adversarial_run_dir,
    )
    adversarial_ingest_stats = ingest_run(
        run_dir=adversarial_run_dir,
        cfg=CentralMemoryConfig(root_dir=adversarial_memory),
        selection=SelectionConfig(),
    )

    baseline = trainer.run(
        verse_spec=evaluation_verse,
        agent_spec=_recall_agent(
            seed=args.eval_seed,
            memory_root=baseline_memory,
            policy_id="recall_baseline_empty_memory",
        ),
        episodes=args.eval_episodes,
        max_steps=10,
        seed=args.eval_seed,
        run_id="baseline",
        auto_index=False,
        verbose=False,
    )
    condition_roots = {
        "clean": clean_memory,
        "noisy": noisy_memory,
        "adversarial": adversarial_memory,
        "filtered": noisy_memory,
    }
    condition_runs: Dict[str, str] = {}
    for condition, memory_root in condition_roots.items():
        out = trainer.run(
            verse_spec=evaluation_verse,
            agent_spec=_recall_agent(
                seed=args.eval_seed,
                memory_root=memory_root,
                policy_id=f"recall_{condition}",
                recall_min_transfer_score=(0.5 if condition == "filtered" else None),
                recall_quality_weight=1.0,
            ),
            episodes=args.eval_episodes,
            max_steps=10,
            seed=args.eval_seed,
            run_id=condition,
            auto_index=False,
            verbose=False,
        )
        condition_runs[condition] = os.path.join(runs_root, str(out["run_id"]))

    baseline_dir = os.path.join(runs_root, str(baseline["run_id"]))
    baseline_stats = evaluate_run(baseline_dir)
    condition_stats = {name: evaluate_run(path) for name, path in condition_runs.items()}
    result: Dict[str, Any] = {
        "design": {
            "verse": "cliff_world",
            "agent_algo": "memory_recall",
            "source_seed": args.source_seed,
            "evaluation_seed": args.eval_seed,
            "source_params": dict(source_verse.params),
            "evaluation_params": dict(evaluation_verse.params),
            "generalization_difference": "source width 4/goal (3,0) versus evaluation width 5/goal (4,0); evaluation requires an unseen extra x=3 to x=4 transition; seeds 17 versus 42",
            "source_episodes": args.source_episodes,
            "evaluation_episodes_per_condition": args.eval_episodes,
        },
        "source": {
            "mean_return": source_stats.mean_return,
            "success_rate": source_stats.success_rate,
            "ingest_stats": vars(clean_ingest_stats),
        },
        "memory_pool": {
            "clean_added": int(clean_ingest_stats.added_events),
            "cliff_noise_added": int(cliff_noise_stats.added_events),
            "line_noise_added": int(line_noise_stats.added_events),
            "noise_to_clean_ratio": float(
                (cliff_noise_stats.added_events + line_noise_stats.added_events)
                / max(1, clean_ingest_stats.added_events)
            ),
            "adversarial_events_written": int(adversarial_events_written),
            "adversarial_added": int(adversarial_ingest_stats.added_events),
            "transfer_scores_by_policy": _transfer_scores_by_policy(noisy_memory),
            "filtered_min_transfer_score": 0.5,
        },
        "baseline": {
            "mean_return": baseline_stats.mean_return,
            "success_rate": baseline_stats.success_rate,
            "mean_steps": baseline_stats.mean_steps,
            **_count_recall_events(baseline_dir),
        },
    }
    for name, stats in condition_stats.items():
        result[name] = {
            "mean_return": stats.mean_return,
            "success_rate": stats.success_rate,
            "mean_steps": stats.mean_steps,
            **_count_recall_events(condition_runs[name]),
            "mean_return_lift_vs_baseline": stats.mean_return - baseline_stats.mean_return,
            "success_rate_lift_vs_baseline": (
                float(stats.success_rate or 0.0) - float(baseline_stats.success_rate or 0.0)
            ),
        }
    result_path = os.path.join(root, "result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"result_path={result_path}")


if __name__ == "__main__":
    main()
