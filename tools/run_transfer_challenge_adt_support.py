from __future__ import annotations

import os
from typing import Any, Callable, Dict

from orchestrator.trainer import Trainer


def prepare_adt_prior(
    *,
    args: Any,
    transfer_dataset: Dict[str, Any],
    transfer_cfg: Dict[str, Any],
    selected_transfer_algo: str,
    run_agent: Callable[..., str],
    parse_cfg_overrides: Callable[[Any], Dict[str, Any]],
    collect_run_eval: Callable[..., Dict[str, Any]],
    extract_success_dna_from_events: Callable[..., int],
    extract_top_return_dna_from_events: Callable[..., int],
    merge_jsonl: Callable[..., int],
    auto_transfer_mix_decay_steps: Callable[..., int],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "enabled": bool(args.adt_prior_enabled),
        "applied": False,
        "reason": "disabled",
        "model_path": str(args.adt_prior_model_path),
        "pilot": {},
        "dataset": {},
        "rollback": {},
    }
    if not bool(args.adt_prior_enabled):
        return {
            "report": report,
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
        }
    if int(transfer_dataset.get("transfer_dataset_rows", 0)) <= 0:
        report["reason"] = "empty_transfer_dataset"
        return {
            "report": report,
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
        }
    if str(selected_transfer_algo).strip().lower() != "q":
        report["reason"] = "selected_transfer_algo_not_q"
        return {
            "report": report,
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
        }

    adt_model_path = str(args.adt_prior_model_path or "").strip()
    if not adt_model_path or not os.path.isfile(adt_model_path):
        report["reason"] = "model_path_missing"
        return {
            "report": report,
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
        }

    adt_eps = max(1, min(int(args.episodes), int(max(1, args.adt_prior_episodes))))
    adt_max_steps = int(args.adt_prior_max_steps)
    if adt_max_steps <= 0:
        adt_max_steps = max(1, int(args.max_steps))
    adt_cfg: Dict[str, Any] = {
        "model_path": str(adt_model_path),
        "verse_name": str(args.target_verse),
        "train": False,
        "online_enabled": False,
        "target_return_auto": True,
    }
    adt_cfg.update(parse_cfg_overrides(args.adt_prior_cfg))
    trainer_adt = Trainer(run_root=str(args.runs_root), schema_version="v1", auto_register_builtin=True)
    adt_run = run_agent(
        trainer=trainer_adt,
        role="adt_prior",
        verse_name=str(args.target_verse),
        episodes=int(adt_eps),
        max_steps=int(adt_max_steps),
        seed=int(args.seed),
        algo="adt",
        policy_id=f"adt_prior_{args.target_verse}",
        cfg=dict(adt_cfg),
    )
    adt_run_dir = os.path.join(str(args.runs_root), adt_run)
    adt_eval = collect_run_eval(
        adt_run_dir,
        early_episodes=max(1, int(args.diagnostic_early_episodes)),
        action_first_k=max(1, int(args.diagnostic_action_agreement_first_k)),
    )
    adt_success_path = os.path.splitext(str(args.transfer_dataset_out))[0] + ".adt_prior_success.jsonl"
    adt_row_mode = str(args.adt_prior_row_mode or "").strip().lower()
    if not adt_row_mode:
        adt_row_mode = "success_only" if bool(args.adt_prior_success_only) else "all_rows"
    if adt_row_mode == "success_only":
        adt_rows = int(
            extract_success_dna_from_events(
                run_dir=adt_run_dir,
                out_path=adt_success_path,
                max_rows=max(1, int(args.adt_prior_max_rows)),
            )
        )
    elif adt_row_mode == "top_return":
        adt_rows = int(
            extract_top_return_dna_from_events(
                run_dir=adt_run_dir,
                out_path=adt_success_path,
                max_rows=max(1, int(args.adt_prior_max_rows)),
                top_return_pct=max(0.0, min(1.0, float(args.adt_prior_top_return_pct))),
            )
        )
    else:
        adt_rows = int(
            merge_jsonl(
                [os.path.join(adt_run_dir, "events.jsonl")],
                adt_success_path,
                max_rows_per_file=max(1, int(args.adt_prior_max_rows)),
            )
        )

    report["pilot"] = {
        "run_id": str(adt_run),
        "run_dir": str(adt_run_dir).replace("\\", "/"),
        "episodes": int(adt_eps),
        "max_steps": int(adt_max_steps),
        "algo": "adt",
        "config_overrides": parse_cfg_overrides(args.adt_prior_cfg),
        "eval": {
            "mean_return": float(adt_eval["stats"].mean_return),
            "success_rate": float(adt_eval["stats"].success_rate or 0.0),
            "mean_steps": float(adt_eval["stats"].mean_steps),
        },
        "curve_aggregate": adt_eval["aggregate"],
    }
    report["dataset"] = {
        "path": str(adt_success_path).replace("\\", "/"),
        "rows": int(adt_rows),
        "row_mode": str(adt_row_mode),
        "success_only": bool(str(adt_row_mode) == "success_only"),
        "top_return_pct": float(max(0.0, min(1.0, float(args.adt_prior_top_return_pct)))),
        "max_rows": int(max(1, int(args.adt_prior_max_rows))),
    }
    if adt_rows < max(1, int(args.adt_prior_min_rows)):
        report["reason"] = "insufficient_prior_rows"
        return {
            "report": report,
            "transfer_dataset": dict(transfer_dataset),
            "transfer_cfg": dict(transfer_cfg),
        }

    merged_path = os.path.splitext(str(args.transfer_dataset_out))[0] + ".with_adt_prior.jsonl"
    merged_rows = merge_jsonl(
        [str(transfer_dataset["transfer_dataset_path"]), str(adt_success_path)],
        merged_path,
        max_rows_per_file=0,
    )
    next_transfer_dataset = dict(transfer_dataset)
    next_transfer_dataset["base_transfer_dataset_path"] = str(
        next_transfer_dataset.get("transfer_dataset_path", "")
    )
    next_transfer_dataset["base_transfer_dataset_rows"] = int(
        next_transfer_dataset.get("transfer_dataset_rows", 0)
    )
    next_transfer_dataset["transfer_dataset_path"] = str(merged_path)
    next_transfer_dataset["transfer_dataset_rows"] = int(merged_rows)
    next_transfer_dataset["adt_prior"] = {
        "pilot_run_id": str(adt_run),
        "prior_rows": int(adt_rows),
        "merged_rows": int(merged_rows),
    }
    next_transfer_cfg = dict(transfer_cfg)
    next_transfer_cfg["dataset_path"] = str(merged_path)
    next_transfer_cfg["warmstart_reward_scale"] = max(
        float(next_transfer_cfg.get("warmstart_reward_scale", 0.0)),
        float(args.adt_prior_warmstart_reward_scale),
    )
    if bool(next_transfer_cfg.get("dynamic_transfer_mix_enabled", False)) and int(args.transfer_mix_decay_steps) <= 0:
        next_transfer_cfg["transfer_mix_decay_steps"] = int(
            auto_transfer_mix_decay_steps(
                episodes=max(1, int(args.episodes)),
                max_steps=max(1, int(args.max_steps)),
                transfer_rows=max(0, int(next_transfer_dataset.get("transfer_dataset_rows", 0))),
                mix_start=max(0.0, min(1.0, float(args.transfer_mix_start))),
                mix_end=max(0.0, min(1.0, float(args.transfer_mix_end))),
            )
        )
    report["applied"] = True
    report["reason"] = (
        "merged_successful_adt_target_rollouts"
        if str(adt_row_mode) == "success_only"
        else (
            "merged_top_return_adt_target_rollouts"
            if str(adt_row_mode) == "top_return"
            else "merged_adt_target_rollouts"
        )
    )
    return {
        "report": report,
        "transfer_dataset": next_transfer_dataset,
        "transfer_cfg": next_transfer_cfg,
    }


def apply_adt_prior_rollback(
    *,
    args: Any,
    adt_prior_report: Dict[str, Any],
    transfer_eval: Dict[str, Any],
    baseline_eval: Dict[str, Any],
    trainer: Any,
    transfer_run: str,
    transfer_run_dir: str,
    transfer_dataset: Dict[str, Any],
    transfer_cfg: Dict[str, Any],
    base_transfer_cfg: Dict[str, Any],
    base_selected_transfer_algo: str,
    run_agent: Callable[..., str],
    adt_prior_rollback_decision: Callable[..., Dict[str, Any]],
    collect_run_eval: Callable[..., Dict[str, Any]],
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
) -> Dict[str, Any]:
    report = dict(adt_prior_report)
    next_transfer_run = str(transfer_run)
    next_transfer_run_dir = str(transfer_run_dir)
    next_transfer_eval = dict(transfer_eval)
    next_transfer_dataset = dict(transfer_dataset)
    next_transfer_cfg = dict(transfer_cfg)
    if not bool(report.get("applied", False)):
        return {
            "report": report,
            "transfer_run": next_transfer_run,
            "transfer_run_dir": next_transfer_run_dir,
            "transfer_eval": next_transfer_eval,
            "transfer_dataset": next_transfer_dataset,
            "transfer_cfg": next_transfer_cfg,
        }
    if not bool(args.adt_prior_auto_rollback):
        report["rollback"] = {"applied": False, "reason": "disabled"}
        return {
            "report": report,
            "transfer_run": next_transfer_run,
            "transfer_run_dir": next_transfer_run_dir,
            "transfer_eval": next_transfer_eval,
            "transfer_dataset": next_transfer_dataset,
            "transfer_cfg": next_transfer_cfg,
        }

    rb = adt_prior_rollback_decision(
        candidate_success_rate=float(safe_float(transfer_eval["early_window"].get("success_rate", 0.0), 0.0)),
        candidate_mean_return=float(safe_float(transfer_eval["early_window"].get("mean_return", 0.0), 0.0)),
        candidate_hazard_per_1k=float(
            safe_float(transfer_eval["early_window"].get("hazard_events_per_1k_steps", 0.0), 0.0)
        ),
        baseline_success_rate=float(safe_float(baseline_eval["early_window"].get("success_rate", 0.0), 0.0)),
        baseline_mean_return=float(safe_float(baseline_eval["early_window"].get("mean_return", 0.0), 0.0)),
        baseline_hazard_per_1k=float(
            safe_float(baseline_eval["early_window"].get("hazard_events_per_1k_steps", 0.0), 0.0)
        ),
        min_success_delta=float(args.adt_prior_rollback_min_success_delta),
        min_return_delta=float(args.adt_prior_rollback_min_return_delta),
        max_hazard_regression_per_1k=float(args.adt_prior_rollback_max_hazard_regression_per_1k),
    )
    report["rollback"] = dict(rb)
    if bool(rb.get("rollback", False)):
        fallback_transfer_run = run_agent(
            trainer=trainer,
            role="transfer_rollback",
            verse_name=str(args.target_verse),
            episodes=max(1, int(args.episodes)),
            max_steps=max(1, int(args.max_steps)),
            seed=int(args.seed),
            algo=str(base_selected_transfer_algo),
            policy_id=f"transfer_rollback_{base_selected_transfer_algo}_{args.target_verse}",
            cfg=dict(base_transfer_cfg),
        )
        next_transfer_run = str(fallback_transfer_run)
        next_transfer_run_dir = os.path.join(str(args.runs_root), next_transfer_run)
        next_transfer_eval = collect_run_eval(
            next_transfer_run_dir,
            early_episodes=max(1, int(args.diagnostic_early_episodes)),
            action_first_k=max(1, int(args.diagnostic_action_agreement_first_k)),
        )
        next_transfer_cfg = dict(base_transfer_cfg)
        if isinstance(next_transfer_dataset.get("base_transfer_dataset_path"), str) and str(
            next_transfer_dataset.get("base_transfer_dataset_path")
        ):
            next_transfer_dataset["transfer_dataset_path"] = str(
                next_transfer_dataset.get("base_transfer_dataset_path")
            )
            next_transfer_dataset["transfer_dataset_rows"] = int(
                safe_int(next_transfer_dataset.get("base_transfer_dataset_rows", 0), 0)
            )
        if isinstance(next_transfer_dataset.get("adt_prior"), dict):
            next_transfer_dataset["adt_prior"]["rolled_back"] = True
        report["rollback"].update(
            {
                "applied": True,
                "fallback_run_id": str(fallback_transfer_run),
                "fallback_run_dir": str(next_transfer_run_dir).replace("\\", "/"),
                "fallback_algo": str(base_selected_transfer_algo),
                "reason": "early_quality_gate_failed",
            }
        )
    else:
        report["rollback"].update({"applied": False, "reason": "early_quality_gate_passed"})

    return {
        "report": report,
        "transfer_run": next_transfer_run,
        "transfer_run_dir": next_transfer_run_dir,
        "transfer_eval": next_transfer_eval,
        "transfer_dataset": next_transfer_dataset,
        "transfer_cfg": next_transfer_cfg,
    }
