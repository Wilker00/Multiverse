"""
Blackjack-specific evaluation and baseline comparison tooling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer
from verses.blackjack_world import _hand_value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _parse_seed_list(raw: Optional[str], *, default: Optional[List[int]] = None) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        return list(default or [])
    out: List[int] = []
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _load_events(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(str(run_dir), "events.jsonl")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_synthetic_action_row(row: Dict[str, Any]) -> bool:
    info = row.get("info") or {}
    if not isinstance(info, dict):
        return False
    action_info = info.get("action_info") or {}
    if not isinstance(action_info, dict):
        return False
    return bool(action_info.get("synthetic_action", False))


def _final_event_by_episode(run_dir: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _load_events(run_dir):
        ep_id = str(row.get("episode_id", ""))
        if not ep_id:
            continue
        current = out.get(ep_id)
        if current is None or int(row.get("step_idx", 0)) >= int(current.get("step_idx", 0)):
            out[ep_id] = row
    return out


def evaluate_blackjack_run(run_dir: str) -> Dict[str, Any]:
    stats = evaluate_run(run_dir)
    finals = _final_event_by_episode(run_dir)
    returns = [float(es.return_sum) for es in stats.episode_stats]
    quarter = max(1, len(returns) // 4) if returns else 1
    q1_mean = (sum(returns[:quarter]) / quarter) if returns else 0.0
    q4_mean = (sum(returns[-quarter:]) / quarter) if returns else 0.0

    hands_total = 0
    win_hands = 0
    push_hands = 0
    loss_hands = 0
    busted_hands = 0
    doubled_hands = 0
    split_rounds = 0
    natural_blackjacks = 0
    dealer_blackjacks = 0
    push_blackjacks = 0
    outcome_counts: Dict[str, int] = defaultdict(int)
    action_counts: Dict[str, int] = defaultdict(int)

    for row in _load_events(run_dir):
        if _is_synthetic_action_row(row):
            continue
        action = row.get("action")
        if action is None:
            continue
        action_counts[str(action)] += 1

    for row in finals.values():
        info = row.get("info") or {}
        if not isinstance(info, dict):
            continue
        outcome = str(info.get("outcome", "")).strip()
        if outcome:
            outcome_counts[outcome] += 1
            if outcome == "blackjack":
                natural_blackjacks += 1
            elif outcome == "dealer_blackjack":
                dealer_blackjacks += 1
            elif outcome == "push_blackjack":
                push_blackjacks += 1

        dealer_hand = info.get("dealer_hand")
        player_hands = info.get("player_hands")
        hand_bets = info.get("hand_bets")
        if not isinstance(dealer_hand, list):
            single_player = info.get("player_hand")
            if isinstance(single_player, list) and outcome:
                player_hands = [single_player]
                hand_bets = [1.0]
                dealer_hand = info.get("dealer_hand")
        if not isinstance(dealer_hand, list) or not isinstance(player_hands, list):
            continue

        if len(player_hands) > 1:
            split_rounds += 1

        dealer_total, _ = _hand_value([int(x) for x in dealer_hand])
        dealer_bust = dealer_total > 21
        bets = hand_bets if isinstance(hand_bets, list) else [1.0] * len(player_hands)

        for idx, hand in enumerate(player_hands):
            if not isinstance(hand, list):
                continue
            hands_total += 1
            total, _ = _hand_value([int(x) for x in hand])
            bet = _safe_float(bets[idx] if idx < len(bets) else 1.0, 1.0)
            if bet > 1.0:
                doubled_hands += 1
            if total > 21:
                busted_hands += 1
                loss_hands += 1
            elif dealer_bust or total > dealer_total:
                win_hands += 1
            elif total == dealer_total:
                push_hands += 1
            else:
                loss_hands += 1

    episodes = max(1, int(stats.episodes))
    hands_total = max(1, int(hands_total))
    return {
        "run_id": str(stats.run_id),
        "run_dir": str(run_dir).replace("\\", "/"),
        "episodes": int(stats.episodes),
        "total_steps": int(stats.total_steps),
        "mean_return": float(stats.mean_return),
        "mean_steps": float(stats.mean_steps),
        "learning_curve": {
            "q1_mean_return": float(q1_mean),
            "q4_mean_return": float(q4_mean),
            "return_improvement": float(q4_mean - q1_mean),
        },
        "hands_total": int(hands_total),
        "mean_return_per_hand": float(stats.mean_return) * float(episodes) / float(hands_total),
        "win_hand_rate": float(win_hands) / float(hands_total),
        "push_hand_rate": float(push_hands) / float(hands_total),
        "loss_hand_rate": float(loss_hands) / float(hands_total),
        "player_bust_rate": float(busted_hands) / float(hands_total),
        "double_hand_rate": float(doubled_hands) / float(hands_total),
        "split_round_rate": float(split_rounds) / float(episodes),
        "natural_blackjack_rate": float(natural_blackjacks) / float(episodes),
        "dealer_blackjack_rate": float(dealer_blackjacks) / float(episodes),
        "push_blackjack_rate": float(push_blackjacks) / float(episodes),
        "action_counts": dict(action_counts),
        "outcomes": dict(outcome_counts),
    }


def compare_blackjack_runs(candidate_run_dir: str, baseline_run_dir: str) -> Dict[str, Any]:
    candidate = evaluate_blackjack_run(candidate_run_dir)
    baseline = evaluate_blackjack_run(baseline_run_dir)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta": {
            "mean_return": float(candidate["mean_return"]) - float(baseline["mean_return"]),
            "mean_return_per_hand": float(candidate["mean_return_per_hand"]) - float(baseline["mean_return_per_hand"]),
            "win_hand_rate": float(candidate["win_hand_rate"]) - float(baseline["win_hand_rate"]),
            "player_bust_rate": float(candidate["player_bust_rate"]) - float(baseline["player_bust_rate"]),
            "double_hand_rate": float(candidate["double_hand_rate"]) - float(baseline["double_hand_rate"]),
            "split_round_rate": float(candidate["split_round_rate"]) - float(baseline["split_round_rate"]),
        },
    }


def aggregate_blackjack_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        return {
            "runs": 0,
            "episodes_total": 0,
            "total_steps": 0,
            "mean_return": 0.0,
            "mean_steps": 0.0,
            "mean_return_per_hand": 0.0,
            "win_hand_rate": 0.0,
            "push_hand_rate": 0.0,
            "loss_hand_rate": 0.0,
            "player_bust_rate": 0.0,
            "double_hand_rate": 0.0,
            "split_round_rate": 0.0,
            "natural_blackjack_rate": 0.0,
            "dealer_blackjack_rate": 0.0,
            "push_blackjack_rate": 0.0,
            "learning_curve": {
                "q1_mean_return": 0.0,
                "q4_mean_return": 0.0,
                "return_improvement": 0.0,
            },
            "action_counts": {},
            "outcomes": {},
            "per_run": [],
        }

    scalar_keys = [
        "mean_return",
        "mean_steps",
        "mean_return_per_hand",
        "win_hand_rate",
        "push_hand_rate",
        "loss_hand_rate",
        "player_bust_rate",
        "double_hand_rate",
        "split_round_rate",
        "natural_blackjack_rate",
        "dealer_blackjack_rate",
        "push_blackjack_rate",
    ]
    learning_keys = ["q1_mean_return", "q4_mean_return", "return_improvement"]
    action_counts: Dict[str, int] = defaultdict(int)
    outcomes: Dict[str, int] = defaultdict(int)

    for summary in summaries:
        for k, v in dict(summary.get("action_counts") or {}).items():
            action_counts[str(k)] += int(v)
        for k, v in dict(summary.get("outcomes") or {}).items():
            outcomes[str(k)] += int(v)

    return {
        "runs": int(len(summaries)),
        "episodes_total": int(sum(int(s.get("episodes", 0) or 0) for s in summaries)),
        "total_steps": int(sum(int(s.get("total_steps", 0) or 0) for s in summaries)),
        "hands_total": int(sum(int(s.get("hands_total", 0) or 0) for s in summaries)),
        **{key: _mean([_safe_float(s.get(key, 0.0), 0.0) for s in summaries]) for key in scalar_keys},
        "learning_curve": {
            key: _mean(
                [_safe_float((s.get("learning_curve") or {}).get(key, 0.0), 0.0) for s in summaries]
            )
            for key in learning_keys
        },
        "action_counts": dict(action_counts),
        "outcomes": dict(outcomes),
        "per_run": [
            {
                "run_id": str(s.get("run_id", "")),
                "run_dir": str(s.get("run_dir", "")),
                "mean_return": _safe_float(s.get("mean_return", 0.0), 0.0),
                "win_hand_rate": _safe_float(s.get("win_hand_rate", 0.0), 0.0),
                "player_bust_rate": _safe_float(s.get("player_bust_rate", 0.0), 0.0),
            }
            for s in summaries
        ],
    }


def run_blackjack_case(
    *,
    algo: str,
    episodes: int,
    max_steps: int,
    seed: int,
    runs_root: str,
    agent_config: Optional[Dict[str, Any]] = None,
    verse_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trainer = Trainer(run_root=runs_root, schema_version="v1", auto_register_builtin=True)
    agent_cfg = dict(agent_config or {})
    algo_name = str(algo).strip().lower()
    if algo_name == "dqn":
        agent_cfg.setdefault("verse_name", "blackjack_world")
        agent_cfg.setdefault("train", True)
        agent_cfg.setdefault("lr", 5e-4)
        agent_cfg.setdefault("gamma", 0.99)
        agent_cfg.setdefault("epsilon", 0.25)
        agent_cfg.setdefault("epsilon_decay", 0.9995)
        agent_cfg.setdefault("epsilon_min", 0.02)
        agent_cfg.setdefault("batch_size", 128)
        agent_cfg.setdefault("buffer_size", 50000)
        agent_cfg.setdefault("target_update_freq", 100)
        agent_cfg.setdefault("double_dqn", True)
        agent_cfg.setdefault("prioritized_replay", True)
        agent_cfg.setdefault("prioritized_alpha", 0.6)
        agent_cfg.setdefault("prioritized_beta0", 0.4)
        agent_cfg.setdefault("prioritized_beta_steps", 20000)
        agent_cfg.setdefault("blackjack_warmstart", True)
        agent_cfg.setdefault("blackjack_warmstart_epochs", 6)
        agent_cfg.setdefault("blackjack_warmstart_samples", 4096)
        agent_cfg.setdefault("blackjack_warmstart_batch_size", 256)
        agent_cfg.setdefault("behavior_clone_epochs", 6)
        agent_cfg.setdefault("behavior_clone_batch_size", 256)
    elif algo_name == "ppo":
        agent_cfg.setdefault("train", True)
        agent_cfg.setdefault("lr", 3e-4)
        agent_cfg.setdefault("gamma", 0.99)
        agent_cfg.setdefault("gae_lambda", 0.95)
        agent_cfg.setdefault("clip_eps", 0.2)
        agent_cfg.setdefault("epochs", 8)
        agent_cfg.setdefault("hidden_dim", 128)
    elif algo_name == "recurrent_ppo":
        agent_cfg.setdefault("train", True)
        agent_cfg.setdefault("lr", 3e-4)
        agent_cfg.setdefault("gamma", 0.99)
        agent_cfg.setdefault("gae_lambda", 0.95)
        agent_cfg.setdefault("clip_eps", 0.2)
        agent_cfg.setdefault("epochs", 6)
        agent_cfg.setdefault("hidden_dim", 128)
        agent_cfg.setdefault("batch_size", 64)
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name="blackjack_world",
        verse_version="0.1",
        seed=seed,
        params={
            "num_decks": 6,
            "reshuffle_penetration": 0.75,
            "blackjack_payout": 1.5,
            "max_steps": max_steps,
            "adr_enabled": False,
            **dict(verse_params or {}),
        },
    )
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=str(algo),
        policy_version="0.1",
        algo=str(algo),
        seed=seed,
        config=agent_cfg,
    )
    result = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
    )
    run_dir = os.path.join(str(runs_root), str(result["run_id"]))
    return {
        "run_id": str(result["run_id"]),
        "run_dir": run_dir.replace("\\", "/"),
        "checkpoint_path": (
            None
            if not result.get("checkpoint_path")
            else str(result["checkpoint_path"]).replace("\\", "/")
        ),
        "summary": evaluate_blackjack_run(run_dir),
    }


def run_blackjack_suite(
    *,
    algo: str,
    seeds: List[int],
    episodes: int,
    max_steps: int,
    runs_root: str,
    agent_config: Optional[Dict[str, Any]] = None,
    verse_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runs = [
        run_blackjack_case(
            algo=algo,
            episodes=episodes,
            max_steps=max_steps,
            seed=int(seed),
            runs_root=runs_root,
            agent_config=agent_config,
            verse_params=verse_params,
        )
        for seed in seeds
    ]
    summaries = [dict(run["summary"]) for run in runs]
    return {
        "seeds": [int(seed) for seed in seeds],
        "runs": runs,
        "aggregate": aggregate_blackjack_summaries(summaries),
    }


def run_blackjack_holdout_eval(
    *,
    algo: str,
    train_seed: int,
    train_episodes: int,
    eval_episodes: int,
    eval_train_seeds: List[int],
    holdout_seeds: List[int],
    max_steps: int,
    runs_root: str,
    agent_config: Optional[Dict[str, Any]] = None,
    verse_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    algo_name = str(algo).strip().lower()
    train_cfg = dict(agent_config or {})
    train_cfg["train"] = True
    training_run = run_blackjack_case(
        algo=algo,
        episodes=int(train_episodes),
        max_steps=max_steps,
        seed=int(train_seed),
        runs_root=runs_root,
        agent_config=train_cfg,
        verse_params=verse_params,
    )

    checkpoint_path = str(training_run.get("checkpoint_path") or "").strip()
    if not checkpoint_path:
        checkpoint_path = os.path.join(str(training_run["run_dir"]), "agent_checkpoint").replace("\\", "/")

    if algo_name == "dqn":
        eval_cfg = dict(train_cfg)
        eval_cfg["train"] = False
        eval_cfg["model_path"] = checkpoint_path
        eval_cfg["epsilon"] = 0.0
        eval_cfg["blackjack_warmstart"] = False
    else:
        eval_cfg = dict(train_cfg)
        eval_cfg["train"] = False

    train_suite = run_blackjack_suite(
        algo=algo,
        seeds=list(eval_train_seeds),
        episodes=int(eval_episodes),
        max_steps=max_steps,
        runs_root=runs_root,
        agent_config=eval_cfg,
        verse_params=verse_params,
    )
    holdout_suite = run_blackjack_suite(
        algo=algo,
        seeds=list(holdout_seeds),
        episodes=int(eval_episodes),
        max_steps=max_steps,
        runs_root=runs_root,
        agent_config=eval_cfg,
        verse_params=verse_params,
    )
    return {
        "training_run": training_run,
        "checkpoint_path": checkpoint_path,
        "train_eval": train_suite,
        "holdout_eval": holdout_suite,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None, help="Analyze an existing blackjack run directory.")
    ap.add_argument("--baseline_run_dir", type=str, default=None, help="Compare an analyzed run against a baseline run.")
    ap.add_argument("--algo", type=str, default="dqn")
    ap.add_argument("--baseline_algo", type=str, default="blackjack_basic")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--train_episodes", type=int, default=None)
    ap.add_argument("--eval_episodes", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train_seed", type=int, default=None)
    ap.add_argument("--eval_train_seeds", type=str, default="", help="Comma-separated seeds for in-distribution evaluation.")
    ap.add_argument("--holdout_seeds", type=str, default="", help="Comma-separated seeds for holdout evaluation.")
    ap.add_argument("--runs_root", type=str, default="runs_smoke")
    args = ap.parse_args()

    if args.run_dir:
        if args.baseline_run_dir:
            payload = compare_blackjack_runs(args.run_dir, args.baseline_run_dir)
        else:
            payload = evaluate_blackjack_run(args.run_dir)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    train_seed = int(args.train_seed if args.train_seed is not None else args.seed)
    train_episodes = int(args.train_episodes if args.train_episodes is not None else args.episodes)
    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else args.episodes)
    eval_train_seeds = _parse_seed_list(args.eval_train_seeds, default=[train_seed])
    holdout_seeds = _parse_seed_list(args.holdout_seeds)

    if holdout_seeds:
        candidate = run_blackjack_holdout_eval(
            algo=str(args.algo),
            train_seed=int(train_seed),
            train_episodes=int(train_episodes),
            eval_episodes=int(eval_episodes),
            eval_train_seeds=list(eval_train_seeds),
            holdout_seeds=list(holdout_seeds),
            max_steps=int(args.max_steps),
            runs_root=str(args.runs_root),
            agent_config={"verse_name": "blackjack_world"} if str(args.algo).strip().lower() == "dqn" else None,
        )
        baseline_train = run_blackjack_suite(
            algo=str(args.baseline_algo),
            seeds=list(eval_train_seeds),
            episodes=int(eval_episodes),
            max_steps=int(args.max_steps),
            runs_root=str(args.runs_root),
        )
        baseline_holdout = run_blackjack_suite(
            algo=str(args.baseline_algo),
            seeds=list(holdout_seeds),
            episodes=int(eval_episodes),
            max_steps=int(args.max_steps),
            runs_root=str(args.runs_root),
        )
        payload = {
            "candidate_algo": str(args.algo),
            "baseline_algo": str(args.baseline_algo),
            "candidate_training_run": candidate["training_run"]["run_id"],
            "candidate_checkpoint_path": candidate["checkpoint_path"],
            "train_split": {
                "candidate": candidate["train_eval"]["aggregate"],
                "baseline": baseline_train["aggregate"],
                "delta": {
                    "mean_return": float(candidate["train_eval"]["aggregate"]["mean_return"]) - float(baseline_train["aggregate"]["mean_return"]),
                    "win_hand_rate": float(candidate["train_eval"]["aggregate"]["win_hand_rate"]) - float(baseline_train["aggregate"]["win_hand_rate"]),
                    "player_bust_rate": float(candidate["train_eval"]["aggregate"]["player_bust_rate"]) - float(baseline_train["aggregate"]["player_bust_rate"]),
                },
            },
            "holdout_split": {
                "candidate": candidate["holdout_eval"]["aggregate"],
                "baseline": baseline_holdout["aggregate"],
                "delta": {
                    "mean_return": float(candidate["holdout_eval"]["aggregate"]["mean_return"]) - float(baseline_holdout["aggregate"]["mean_return"]),
                    "win_hand_rate": float(candidate["holdout_eval"]["aggregate"]["win_hand_rate"]) - float(baseline_holdout["aggregate"]["win_hand_rate"]),
                    "player_bust_rate": float(candidate["holdout_eval"]["aggregate"]["player_bust_rate"]) - float(baseline_holdout["aggregate"]["player_bust_rate"]),
                },
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    baseline = run_blackjack_case(
        algo=str(args.baseline_algo),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed) + 1000,
        runs_root=str(args.runs_root),
    )
    candidate_cfg = {}
    if str(args.algo).strip().lower() == "dqn":
        candidate_cfg = {
            "verse_name": "blackjack_world",
            "train": True,
        }
    candidate = run_blackjack_case(
        algo=str(args.algo),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        runs_root=str(args.runs_root),
        agent_config=candidate_cfg,
    )
    payload = compare_blackjack_runs(candidate["run_dir"], baseline["run_dir"])
    payload["candidate_run_id"] = candidate["run_id"]
    payload["baseline_run_id"] = baseline["run_id"]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
