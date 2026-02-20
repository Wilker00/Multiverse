"""
tools/run_adt_dagger.py

Pragmatic DAgger loop for ADT:
1) warm up a Q-learning expert on the target verse
2) roll out ADT, label visited states with expert actions
3) aggregate labels into dataset
4) retrain ADT with checkpoint warm-start
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.q_agent import QLearningAgent
from agents.transformer_agent import TransformerAgent
from core.agent_base import ExperienceBatch, Transition
from core.planner_oracle import plan_actions_from_current_state
from core.types import AgentSpec, VerseSpec
from tools.prep_adt_data import prepare_adt_data
from tools.train_adt import train_adt
from verses.registry import create_verse, register_builtin
from models.decision_transformer import load_decision_transformer_checkpoint


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _parse_kv_list(kvs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not kvs:
        return out
    for item in kvs:
        s = str(item).strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except (TypeError, ValueError):
            out[k] = v
    return out


def _run_expert_training(
    *,
    verse: Any,
    expert: QLearningAgent,
    episodes: int,
    max_steps: int,
) -> Dict[str, float]:
    if int(episodes) <= 0:
        return {"episodes": 0.0, "success_rate": 0.0, "mean_return": 0.0, "mean_steps": 0.0}

    success_n = 0
    return_sum = 0.0
    steps_sum = 0
    for _ in range(int(episodes)):
        rr = verse.reset()
        obs = rr.obs
        transitions: List[Transition] = []
        ep_ret = 0.0
        ep_steps = 0
        ep_success = False

        for _t in range(int(max_steps)):
            ar = expert.act(obs)
            sr = verse.step(ar.action)
            info = sr.info if isinstance(sr.info, dict) else {}
            ep_success = ep_success or bool(info.get("reached_goal", False) or info.get("success", False))
            transitions.append(
                Transition(
                    obs=obs,
                    action=int(ar.action),
                    reward=float(sr.reward),
                    next_obs=sr.obs,
                    done=bool(sr.done),
                    truncated=bool(sr.truncated),
                    info={"env_info": dict(info)},
                )
            )
            obs = sr.obs
            ep_ret += float(sr.reward)
            ep_steps += 1
            if bool(sr.done or sr.truncated):
                break

        expert.learn(ExperienceBatch(transitions=transitions, meta={"source": "dagger_expert_warmup"}))
        success_n += int(ep_success)
        return_sum += float(ep_ret)
        steps_sum += int(ep_steps)

    n = max(1, int(episodes))
    return {
        "episodes": float(episodes),
        "success_rate": float(success_n / n),
        "mean_return": float(return_sum / n),
        "mean_steps": float(steps_sum / n),
    }


def _collect_dagger_labels(
    *,
    verse: Any,
    learner: TransformerAgent,
    expert: QLearningAgent,
    verse_name: str,
    round_idx: int,
    episodes: int,
    max_steps: int,
    mismatch_only: bool,
    failures_only: bool,
    expert_policy: str,
    planner_horizon: int,
    planner_max_expansions: int,
    planner_avoid_terminal_failures: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    success_n = 0
    return_sum = 0.0
    steps_sum = 0
    planner_labels = 0
    q_labels = 0
    old_eps = float(expert.stats.epsilon)
    expert.stats.epsilon = 0.0
    try:
        for ep in range(int(episodes)):
            rr = verse.reset()
            obs = rr.obs
            ep_rows: List[Dict[str, Any]] = []
            ep_ret = 0.0
            ep_steps = 0
            ep_success = False
            ep_planner_labels = 0
            ep_q_labels = 0

            for step_idx in range(int(max_steps)):
                learner_out = learner.act(obs)
                learner_action = int(learner_out.action)

                planner_action = None
                if str(expert_policy).strip().lower() in ("planner", "hybrid"):
                    try:
                        plan = plan_actions_from_current_state(
                            verse=verse,
                            horizon=max(1, int(planner_horizon)),
                            max_expansions=max(100, int(planner_max_expansions)),
                            avoid_terminal_failures=bool(planner_avoid_terminal_failures),
                        )
                        if plan:
                            planner_action = int(plan[0])
                    except Exception:
                        planner_action = None

                if str(expert_policy).strip().lower() == "q":
                    expert_action = int(expert.act(obs).action)
                    expert_source = "q"
                elif str(expert_policy).strip().lower() == "planner":
                    if planner_action is not None:
                        expert_action = int(planner_action)
                        expert_source = "planner"
                    else:
                        expert_action = int(expert.act(obs).action)
                        expert_source = "q_fallback"
                else:
                    if planner_action is not None:
                        expert_action = int(planner_action)
                        expert_source = "planner"
                    else:
                        expert_action = int(expert.act(obs).action)
                        expert_source = "q_fallback"

                sr = verse.step(learner_action)
                info = sr.info if isinstance(sr.info, dict) else {}
                done = bool(sr.done or sr.truncated)
                reached_goal = bool(info.get("reached_goal", False) or info.get("success", False))
                ep_success = ep_success or reached_goal

                keep = True
                if bool(mismatch_only) and learner_action == expert_action:
                    keep = False
                if keep:
                    if expert_source == "planner":
                        ep_planner_labels += 1
                    else:
                        ep_q_labels += 1
                    ep_rows.append(
                        {
                            "run_id": f"dagger_round_{int(round_idx):03d}",
                            "episode_id": f"r{int(round_idx):03d}_ep{int(ep):05d}",
                            "step_idx": int(step_idx),
                            "verse_name": str(verse_name),
                            "obs": obs,
                            "action": int(expert_action),
                            "reward": float(sr.reward),
                            "done": bool(done),
                            "truncated": bool(sr.truncated),
                            "info": {
                                "reached_goal": bool(reached_goal),
                                "source": "dagger_expert_label",
                                "expert_source": str(expert_source),
                                "learner_action": int(learner_action),
                            },
                        }
                    )

                obs = sr.obs
                ep_ret += float(sr.reward)
                ep_steps += 1
                if done:
                    break

            if not bool(failures_only) or not bool(ep_success):
                rows.extend(ep_rows)
                planner_labels += int(ep_planner_labels)
                q_labels += int(ep_q_labels)
            success_n += int(ep_success)
            return_sum += float(ep_ret)
            steps_sum += int(ep_steps)
    finally:
        expert.stats.epsilon = old_eps

    n = max(1, int(episodes))
    stats = {
        "episodes": float(episodes),
        "success_rate": float(success_n / n),
        "mean_return": float(return_sum / n),
        "mean_steps": float(steps_sum / n),
        "labeled_rows": float(len(rows)),
        "planner_labels": float(planner_labels),
        "q_labels": float(q_labels),
    }
    return rows, stats


def _append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--verse", type=str, default="warehouse_world")
    ap.add_argument("--verse_version", type=str, default="0.1")
    ap.add_argument("--vparam", action="append", default=None, help="verse param k=v (repeatable)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_steps", type=int, default=80)

    ap.add_argument("--base_model_path", type=str, required=True)
    ap.add_argument("--out_model_path", type=str, default=os.path.join("models", "decision_transformer_dagger.pt"))
    ap.add_argument("--dagger_dataset_path", type=str, default=os.path.join("models", "expert_datasets", "adt_dagger_labels.jsonl"))
    ap.add_argument("--prepared_dataset_path", type=str, default=os.path.join("models", "adt_data_dagger.pt"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--collect_episodes", type=int, default=20)
    ap.add_argument("--dagger_mismatch_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dagger_failures_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--reset_dagger_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Truncate dagger_dataset_path before collecting labels for a fresh run.",
    )
    ap.add_argument(
        "--include_runs_history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include runs_root events in prep_adt_data. Default false avoids stale run mixing.",
    )

    ap.add_argument("--expert_warmup_episodes", type=int, default=100)
    ap.add_argument("--expert_refit_episodes", type=int, default=20)
    ap.add_argument("--expert_lr", type=float, default=0.15)
    ap.add_argument("--expert_gamma", type=float, default=0.99)
    ap.add_argument("--expert_epsilon_start", type=float, default=1.0)
    ap.add_argument("--expert_epsilon_min", type=float, default=0.05)
    ap.add_argument("--expert_epsilon_decay", type=float, default=0.995)
    ap.add_argument("--expert_policy", type=str, default="hybrid", choices=["q", "planner", "hybrid"])
    ap.add_argument("--planner_horizon", type=int, default=8)
    ap.add_argument("--planner_max_expansions", type=int, default=8000)
    ap.add_argument("--planner_avoid_terminal_failures", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--context_len", type=int, default=20)
    ap.add_argument("--state_dim", type=int, default=64)
    ap.add_argument("--chunk_stride", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--top_return_pct", type=float, default=1.0)
    ap.add_argument("--success_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min_episode_steps", type=int, default=2)
    ap.add_argument("--max_runs", type=int, default=0)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--dataset_dir", type=str, default="")
    ap.add_argument("--min_action_dim", type=int, default=0, help="Lower bound for prepared dataset action_dim.")
    ap.add_argument("--action_balance_mode", type=str, default="none", choices=["none", "cap_ratio"])
    ap.add_argument("--action_balance_max_ratio", type=float, default=3.0)
    ap.add_argument("--action_balance_seed", type=int, default=123)

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_timestep", type=int, default=4096)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--class_weight_mode", type=str, default="auto", choices=["none", "auto", "inverse_sqrt", "inverse"])
    ap.add_argument("--class_weight_min_count", type=int, default=1)
    ap.add_argument("--class_weight_max", type=float, default=5.0)
    ap.add_argument("--learner_target_return", type=float, default=20.0)
    ap.add_argument("--learner_sample", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--learner_temperature", type=float, default=1.0)
    ap.add_argument("--learner_top_k", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isfile(args.base_model_path):
        raise FileNotFoundError(f"base_model_path not found: {args.base_model_path}")
    base_model, _ = load_decision_transformer_checkpoint(str(args.base_model_path), map_location="cpu")
    base_cfg = base_model.get_config()
    base_action_dim = max(0, _safe_int(base_cfg.get("action_dim"), 0))
    min_action_dim_eff = max(int(base_action_dim), max(0, int(args.min_action_dim)))

    register_builtin()
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=str(args.verse),
        verse_version=str(args.verse_version),
        seed=int(args.seed),
        tags=["dagger"],
        params=_parse_kv_list(list(args.vparam or [])),
    )
    verse = create_verse(verse_spec)
    verse.seed(int(args.seed))

    expert_spec = AgentSpec(
        spec_version="v1",
        policy_id=f"dagger_q_expert:{args.verse}",
        policy_version="0.1",
        algo="q",
        seed=int(args.seed),
        config={
            "lr": float(args.expert_lr),
            "gamma": float(args.expert_gamma),
            "epsilon_start": float(args.expert_epsilon_start),
            "epsilon_min": float(args.expert_epsilon_min),
            "epsilon_decay": float(args.expert_epsilon_decay),
        },
    )
    expert = QLearningAgent(
        spec=expert_spec,
        observation_space=verse.observation_space,
        action_space=verse.action_space,
    )
    expert.seed(int(args.seed))

    if str(args.expert_policy).strip().lower() == "planner":
        warm = {"episodes": 0.0, "success_rate": 0.0, "mean_return": 0.0, "mean_steps": 0.0}
        print("expert_warmup skipped (expert_policy=planner)")
    else:
        warm = _run_expert_training(
            verse=verse,
            expert=expert,
            episodes=max(0, int(args.expert_warmup_episodes)),
            max_steps=max(1, int(args.max_steps)),
        )
        print(
            f"expert_warmup episodes={int(warm['episodes'])} "
            f"success={float(warm['success_rate']):.3f} "
            f"mean_return={float(warm['mean_return']):.3f}"
        )

    current_model_path = str(args.base_model_path)
    dataset_paths = list(args.dataset or [])
    dataset_paths.append(str(args.dagger_dataset_path))
    os.makedirs(os.path.dirname(str(args.out_model_path)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(str(args.dagger_dataset_path)) or ".", exist_ok=True)
    if bool(args.reset_dagger_dataset):
        with open(str(args.dagger_dataset_path), "w", encoding="utf-8"):
            pass
        print(f"reset_dagger_dataset path={str(args.dagger_dataset_path).replace('\\', '/')}")

    class_weight_mode_eff = str(args.class_weight_mode)
    if str(args.action_balance_mode).strip().lower() != "none" and str(args.class_weight_mode).strip().lower() == "auto":
        class_weight_mode_eff = "none"
        print("class_weight_mode auto->none (action_balance_mode enabled)")

    runs_root_eff = str(args.runs_root)
    max_runs_eff = max(0, int(args.max_runs))
    if not bool(args.include_runs_history):
        runs_root_eff = "__adt_runs_history_disabled__"
        max_runs_eff = 0
    print(
        f"dataset_sources runs_history={'on' if bool(args.include_runs_history) else 'off'} "
        f"datasets={int(len(dataset_paths))} dataset_dir={'set' if str(args.dataset_dir).strip() else 'unset'}"
    )

    for round_idx in range(1, max(1, int(args.rounds)) + 1):
        learner_spec = AgentSpec(
            spec_version="v1",
            policy_id=f"adt_dagger_round_{int(round_idx)}",
            policy_version="0.1",
            algo="adt",
            seed=int(args.seed) + int(round_idx),
            config={
                "model_path": current_model_path,
                "device": str(args.device),
                "context_len": int(args.context_len),
                "target_return": float(args.learner_target_return),
                "sample": bool(args.learner_sample),
                "temperature": max(1e-6, float(args.learner_temperature)),
                "top_k": max(0, int(args.learner_top_k)),
            },
        )
        learner = TransformerAgent(
            spec=learner_spec,
            observation_space=verse.observation_space,
            action_space=verse.action_space,
        )
        learner.seed(int(args.seed) + int(round_idx))
        rows, col = _collect_dagger_labels(
            verse=verse,
            learner=learner,
            expert=expert,
            verse_name=str(args.verse),
            round_idx=int(round_idx),
            episodes=max(1, int(args.collect_episodes)),
            max_steps=max(1, int(args.max_steps)),
            mismatch_only=bool(args.dagger_mismatch_only),
            failures_only=bool(args.dagger_failures_only),
            expert_policy=str(args.expert_policy),
            planner_horizon=max(1, int(args.planner_horizon)),
            planner_max_expansions=max(100, int(args.planner_max_expansions)),
            planner_avoid_terminal_failures=bool(args.planner_avoid_terminal_failures),
        )
        learner.close()
        _append_jsonl(str(args.dagger_dataset_path), rows)
        print(
            f"round={int(round_idx)} collect episodes={int(col['episodes'])} "
            f"success={float(col['success_rate']):.3f} mean_return={float(col['mean_return']):.3f} "
            f"labeled_rows={int(col['labeled_rows'])} planner_labels={int(col['planner_labels'])} q_labels={int(col['q_labels'])}"
        )

        if str(args.expert_policy).strip().lower() == "planner":
            refit = {"episodes": 0.0, "success_rate": 0.0, "mean_return": 0.0, "mean_steps": 0.0}
            print(f"round={int(round_idx)} expert_refit skipped (expert_policy=planner)")
        else:
            refit = _run_expert_training(
                verse=verse,
                expert=expert,
                episodes=max(0, int(args.expert_refit_episodes)),
                max_steps=max(1, int(args.max_steps)),
            )
            print(
                f"round={int(round_idx)} expert_refit episodes={int(refit['episodes'])} "
                f"success={float(refit['success_rate']):.3f} mean_return={float(refit['mean_return']):.3f}"
            )

        meta = prepare_adt_data(
            runs_root=str(runs_root_eff),
            out_path=str(args.prepared_dataset_path),
            context_len=max(1, int(args.context_len)),
            chunk_stride=int(args.chunk_stride),
            state_dim=max(4, int(args.state_dim)),
            max_timestep=max(1, int(args.max_timestep)),
            gamma=max(0.0, min(1.0, float(args.gamma))),
            top_return_pct=max(0.0, min(1.0, float(args.top_return_pct))),
            success_only=bool(args.success_only),
            min_episode_steps=max(1, int(args.min_episode_steps)),
            max_runs=max(0, int(max_runs_eff)),
            verse_filter=[str(args.verse)],
            dataset_paths=dataset_paths,
            dataset_dir=str(args.dataset_dir),
            action_balance_mode=str(args.action_balance_mode),
            action_balance_max_ratio=max(0.1, float(args.action_balance_max_ratio)),
            action_balance_seed=int(args.action_balance_seed),
            min_action_dim=max(0, int(min_action_dim_eff)),
        )
        print(
            f"round={int(round_idx)} prep episodes={int(_safe_int(meta.get('episodes'), 0))} "
            f"samples={int(_safe_int(meta.get('samples'), 0))} "
            f"balance_applied={bool(meta.get('action_balance_applied', False))}"
        )

        round_ckpt = str(args.out_model_path)
        if int(round_idx) < int(args.rounds):
            stem, ext = os.path.splitext(str(args.out_model_path))
            ext = ext if ext else ".pt"
            round_ckpt = f"{stem}.round{int(round_idx):03d}{ext}"

        ckpt = train_adt(
            dataset_path=str(args.prepared_dataset_path),
            out_path=str(round_ckpt),
            init_model_path=str(current_model_path),
            epochs=max(1, int(args.epochs)),
            batch_size=max(1, int(args.batch_size)),
            lr=max(1e-6, float(args.lr)),
            weight_decay=max(0.0, float(args.weight_decay)),
            grad_clip=max(0.0, float(args.grad_clip)),
            val_split=max(0.0, min(0.5, float(args.val_split))),
            seed=int(args.seed) + int(round_idx),
            d_model=max(16, int(args.d_model)),
            n_head=max(1, int(args.n_head)),
            n_layer=max(1, int(args.n_layer)),
            dropout=max(0.0, min(0.9, float(args.dropout))),
            max_timestep=max(1, int(args.max_timestep)),
            device=str(args.device),
            class_weight_mode=str(class_weight_mode_eff),
            class_weight_min_count=max(1, int(args.class_weight_min_count)),
            class_weight_max=max(0.0, float(args.class_weight_max)),
        )
        current_model_path = str(round_ckpt)
        print(f"round={int(round_idx)} train best_val_loss={float(_safe_float(ckpt.get('best_val_loss'), 0.0)):.4f}")

    if os.path.abspath(current_model_path) != os.path.abspath(str(args.out_model_path)):
        shutil.copyfile(current_model_path, str(args.out_model_path))
    print(f"dagger_complete final_model={str(args.out_model_path).replace('\\', '/')}")

    verse.close()
    expert.close()


if __name__ == "__main__":
    main()
