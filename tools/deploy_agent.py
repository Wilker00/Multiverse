"""
tools/deploy_agent.py

Manifest-driven deployment loop with regression guard.

Behavior:
- Reads models/default_policy_set.json (or --manifest_path)
- Optionally trains a new special_moe candidate per verse
- If candidate is worse than current manifest default, rollback is automatic
- Launches deployment runs via algo=gateway (no extra policy CLI args required)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from core.skill_contracts import ContractConfig, SkillContractManager
from agents.gateway_agent import GatewayAgentConfig
from agents.special_moe_agent import SpecialMoEConfig
from memory.central_repository import CentralMemoryConfig, ingest_run
from memory.knowledge_market import KnowledgeMarket, KnowledgeMarketConfig
from memory.selection import SelectionConfig
from orchestrator.eval_harness import default_benchmark_suite, default_gate_thresholds, evaluate_agent_case
from orchestrator.eval_harness import print_gate_report, run_ab_gate
from orchestrator.evaluator import RunStats, evaluate_run
from orchestrator.promotion_board import PromotionConfig, run_promotion_board
from orchestrator.trainer import Trainer


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _validate_gateway_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(cfg) if isinstance(cfg, dict) else {}
    # Validation side effect: raises on unknown/invalid keys by default.
    GatewayAgentConfig.from_dict(raw)
    return raw


def _validate_special_moe_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(cfg) if isinstance(cfg, dict) else {}
    # Validation side effect: raises on unknown/invalid keys by default.
    SpecialMoEConfig.from_dict(raw)
    return raw


def _default_max_steps(verse: str) -> int:
    v = str(verse).strip().lower()
    if v == "line_world":
        return 40
    if v == "grid_world":
        return 60
    if v == "cliff_world":
        return 100
    if v == "park_world":
        return 80
    if v == "pursuit_world":
        return 60
    if v == "labyrinth_world":
        return 180
    if v == "chess_world":
        return 80
    if v == "go_world":
        return 90
    if v == "uno_world":
        return 70
    return 40


def _default_verse_params(verse: str, max_steps: int) -> Dict[str, Any]:
    v = str(verse).strip().lower()
    params: Dict[str, Any] = {"max_steps": int(max_steps)}
    if v == "line_world":
        params.setdefault("goal_pos", 8)
        params.setdefault("step_penalty", -0.02)
    if v == "cliff_world":
        params.setdefault("width", 12)
        params.setdefault("height", 4)
        params.setdefault("step_penalty", -1.0)
        params.setdefault("cliff_penalty", -100.0)
        params.setdefault("end_on_cliff", False)
    if v == "labyrinth_world":
        params.setdefault("width", 15)
        params.setdefault("height", 11)
        params.setdefault("step_penalty", -0.05)
        params.setdefault("wall_penalty", -0.20)
        params.setdefault("pit_penalty", -25.0)
        params.setdefault("laser_penalty", -18.0)
        params.setdefault("battery_capacity", 80)
        params.setdefault("battery_drain", 1)
        params.setdefault("action_noise", 0.08)
        params.setdefault("vision_radius", 1)
    if v == "chess_world":
        params.setdefault("win_material", 8)
        params.setdefault("lose_material", -8)
        params.setdefault("random_swing", 0.20)
    if v == "go_world":
        params.setdefault("target_territory", 10)
        params.setdefault("random_swing", 0.25)
    if v == "uno_world":
        params.setdefault("start_hand", 7)
        params.setdefault("opp_start_hand", 7)
        params.setdefault("random_swing", 0.25)
    return params


def _read_manifest(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return data


def _write_manifest(path: str, payload: Dict[str, Any], backup: bool) -> None:
    if backup and os.path.isfile(path):
        shutil.copyfile(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_once(
    *,
    verse: str,
    agent_algo: str,
    agent_config: Optional[Dict[str, Any]],
    episodes: int,
    max_steps: int,
    seed: int,
    runs_root: str,
) -> Tuple[str, RunStats]:
    trainer = Trainer(run_root=runs_root, schema_version="v1", auto_register_builtin=True)
    effective_cfg = dict(agent_config) if isinstance(agent_config, dict) else {}
    algo_norm = str(agent_algo).strip().lower()
    if algo_norm == "gateway":
        effective_cfg = _validate_gateway_cfg(effective_cfg)
    elif algo_norm == "special_moe":
        effective_cfg = _validate_special_moe_cfg(effective_cfg)

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=str(verse),
        verse_version="0.1",
        seed=int(seed),
        tags=["deploy"],
        params=_default_verse_params(verse, max_steps),
    )
    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=str(agent_algo),
        policy_version="0.0",
        algo=str(agent_algo),
        seed=int(seed),
        config=(effective_cfg if effective_cfg else None),
    )
    res = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
    )
    run_id = str(res.get("run_id", ""))
    if not run_id:
        raise RuntimeError(f"Failed to obtain run_id from trainer result: {res}")
    run_dir = os.path.join(runs_root, run_id)
    stats = evaluate_run(run_dir)
    return run_id, stats


def _entry_for_verse(manifest: Dict[str, Any], verse: str) -> Optional[Dict[str, Any]]:
    dep = _as_dict(manifest.get("deployment_ready_defaults"))
    if verse in dep and isinstance(dep.get(verse), dict):
        return dep[verse]
    robust = _as_dict(manifest.get("winners_robust"))
    if verse in robust and isinstance(robust.get(verse), dict):
        return {"picked_run": robust[verse]}
    return None


def _provider_id_from_entry(entry: Dict[str, Any]) -> Optional[str]:
    picked = _as_dict(entry.get("picked_run"))
    run_id = str(picked.get("run_id", "") or entry.get("run_id", "")).strip()
    return run_id if run_id else None


def _baseline_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    picked = _as_dict(entry.get("picked_run"))
    if not picked:
        picked = entry
    run_dir = str(picked.get("run_dir", "")).replace("/", os.sep)

    out = {
        "run_id": str(picked.get("run_id", "")),
        "run_dir": str(picked.get("run_dir", "")),
        "policy": str(picked.get("policy", "")),
        "mean_return": _safe_float(picked.get("mean_return", 0.0)),
        "success_rate": picked.get("success_rate"),
        "mean_steps": _safe_float(picked.get("mean_steps", 0.0)),
        "episodes": int(picked.get("episodes", 0) or 0),
    }

    if run_dir and os.path.isdir(run_dir):
        try:
            s = evaluate_run(run_dir)
            out["mean_return"] = float(s.mean_return)
            out["success_rate"] = s.success_rate
            out["mean_steps"] = float(s.mean_steps)
            out["episodes"] = int(s.episodes)
        except Exception:
            pass
    return out


def _is_regression(candidate: Dict[str, Any], baseline: Dict[str, Any], tol: float = 1e-9) -> bool:
    if _safe_float(candidate.get("mean_return", 0.0)) < _safe_float(baseline.get("mean_return", 0.0)) - tol:
        return True

    bsr = baseline.get("success_rate")
    csr = candidate.get("success_rate")
    if isinstance(bsr, (int, float)) and isinstance(csr, (int, float)):
        if float(csr) < float(bsr) - tol:
            return True
    return False


def _is_improvement(candidate: Dict[str, Any], baseline: Dict[str, Any], tol: float = 1e-9) -> bool:
    c_ret = _safe_float(candidate.get("mean_return", 0.0))
    b_ret = _safe_float(baseline.get("mean_return", 0.0))
    if c_ret > b_ret + tol:
        return True

    bsr = baseline.get("success_rate")
    csr = candidate.get("success_rate")
    if abs(c_ret - b_ret) <= tol and isinstance(bsr, (int, float)) and isinstance(csr, (int, float)):
        if float(csr) > float(bsr) + tol:
            return True
    return False


def _promote_special_moe(
    manifest: Dict[str, Any],
    *,
    verse: str,
    run_id: str,
    stats: RunStats,
    candidate_top_k: int,
    runs_root: str,
) -> Dict[str, Any]:
    dep = _as_dict(manifest.get("deployment_ready_defaults"))
    history = _as_dict(manifest.get("deployment_history"))
    prev_entry = dep.get(verse)
    if prev_entry is not None:
        hist_bucket = history.get(verse)
        if not isinstance(hist_bucket, list):
            hist_bucket = []
        hist_bucket.append(prev_entry)
        history[verse] = hist_bucket

    run_dir = os.path.join(str(runs_root), run_id).replace("\\", "/")
    dep[verse] = {
        "picked_run": {
            "run_id": run_id,
            "run_dir": run_dir,
            "verse": verse,
            "policy": "special_moe",
            "mean_return": float(stats.mean_return),
            "success_rate": stats.success_rate,
            "mean_steps": float(stats.mean_steps),
            "episodes": int(stats.episodes),
        },
        "command": (
            f"python tools/train_agent.py --algo special_moe --verse {verse} "
            f"--episodes 50 --max_steps {_default_max_steps(verse)} "
            f"--expert_dataset_dir models/expert_datasets --top_k_experts {int(candidate_top_k)}"
        ),
        "artifacts": [
            "models/micro_selector.pt",
            "models/expert_datasets/line_world.jsonl",
            "models/expert_datasets/grid_world.jsonl",
            "models/expert_datasets/cliff_world.jsonl",
        ],
    }
    manifest["deployment_ready_defaults"] = dep
    manifest["deployment_history"] = history
    return manifest


def _candidate_cfg(verse: str, top_k: int) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "expert_dataset_dir": os.path.join("models", "expert_datasets"),
        "top_k": int(top_k),
    }
    selector = os.path.join("models", "micro_selector.pt")
    if os.path.isfile(selector):
        cfg["selector_model_path"] = selector
    return _validate_special_moe_cfg(cfg)


def _run_behavioral_surgeon(
    *,
    cliff_lookback_runs: int,
    behavior_top_percent: float,
    cliff_reward_threshold: float,
    death_similarity_threshold: float,
    active_forgetting_similarity: float,
) -> None:
    cmd = [
        sys.executable,
        os.path.join("tools", "behavioral_surgeon.py"),
        "--cliff_lookback_runs",
        str(max(1, int(cliff_lookback_runs))),
        "--behavior_top_percent",
        str(float(behavior_top_percent)),
        "--cliff_reward_threshold",
        str(float(cliff_reward_threshold)),
        "--death_similarity_threshold",
        str(float(death_similarity_threshold)),
        "--active_forgetting_similarity",
        str(float(active_forgetting_similarity)),
    ]
    subprocess.run(cmd, check=True)


def _candidate_absolute_gate(verse: str, candidate_spec: AgentSpec) -> Tuple[bool, Dict[str, Any]]:
    verse_name = str(verse).strip().lower()
    cases = default_benchmark_suite(mode="target", target_verse=verse_name)
    if not cases:
        raise RuntimeError(f"No benchmark case found for verse: {verse_name}")
    case = cases[0]
    summary = evaluate_agent_case(agent_spec=candidate_spec, case=case)
    gate = default_gate_thresholds().get(verse_name)
    if gate is None:
        return True, {
            "verse": verse_name,
            "passed": True,
            "reason": "no_threshold_config",
            "candidate": {
                "mean_return": float(summary.mean_return),
                "success_rate": float(summary.success_rate),
                "failure_rate": float(summary.failure_rate),
                "safety_violation_rate": float(summary.safety_violation_rate),
                "return_variance": float(summary.return_variance),
            },
        }

    pass_absolute = (
        float(summary.mean_return) >= float(gate.min_mean_return)
        and float(summary.success_rate) >= float(gate.min_success_rate)
    )
    pass_safety = (
        float(summary.failure_rate) <= float(gate.max_failure_rate)
        and float(summary.safety_violation_rate) <= float(gate.max_safety_violation_rate)
    )
    pass_stability = float(summary.return_variance) <= float(gate.max_return_variance)
    passed = bool(pass_absolute and pass_safety and pass_stability)
    return passed, {
        "verse": verse_name,
        "passed": passed,
        "checks": {
            "absolute": bool(pass_absolute),
            "safety": bool(pass_safety),
            "stability": bool(pass_stability),
        },
        "thresholds": {
            "min_mean_return": float(gate.min_mean_return),
            "min_success_rate": float(gate.min_success_rate),
            "max_failure_rate": float(gate.max_failure_rate),
            "max_safety_violation_rate": float(gate.max_safety_violation_rate),
            "max_return_variance": float(gate.max_return_variance),
        },
        "candidate": {
            "mean_return": float(summary.mean_return),
            "success_rate": float(summary.success_rate),
            "failure_rate": float(summary.failure_rate),
            "safety_violation_rate": float(summary.safety_violation_rate),
            "return_variance": float(summary.return_variance),
        },
    }


def _baseline_spec_for_gate(*, verse: str, manifest_path: str, seed: int) -> AgentSpec:
    cfg = _validate_gateway_cfg({"manifest_path": str(manifest_path), "verse_name": str(verse)})
    return AgentSpec(
        spec_version="v1",
        policy_id=f"baseline_gateway:{verse}",
        policy_version="0.0",
        algo="gateway",
        seed=int(seed),
        config=cfg,
    )


def _candidate_spec_for_gate(*, verse: str, top_k: int, seed: int) -> AgentSpec:
    cfg = _candidate_cfg(verse, top_k)
    cfg["verse_name"] = str(verse)
    cfg = _validate_special_moe_cfg(cfg)
    return AgentSpec(
        spec_version="v1",
        policy_id=f"candidate_special_moe:{verse}",
        policy_version="0.0",
        algo="special_moe",
        seed=int(seed),
        config=cfg,
    )


def _ingest_to_central(run_id: str, memory_dir: str, runs_root: str) -> None:
    run_dir = os.path.join(str(runs_root), run_id)
    st = ingest_run(
        run_dir=run_dir,
        cfg=CentralMemoryConfig(root_dir=memory_dir),
        selection=SelectionConfig(keep_top_k_per_episode=40, keep_top_k_episodes=40, novelty_bonus=0.1),
    )
    print(
        f"ingest: {run_id} input={st.input_events} selected={st.selected_events} "
        f"added={st.added_events} skipped={st.skipped_duplicates}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--verse", type=str, default=None, help="Optional single-verse deploy target.")
    ap.add_argument("--episodes", type=int, default=20, help="Deployment episodes for gateway runs.")
    ap.add_argument("--max_steps", type=int, default=0, help="Override max steps (0 uses verse defaults).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--runs_root", type=str, default="runs")

    ap.add_argument("--skip_regression", action="store_true", help="Skip special_moe candidate check.")
    ap.add_argument("--candidate_episodes", type=int, default=20)
    ap.add_argument("--candidate_max_steps", type=int, default=0, help="Override candidate max steps (0 default).")
    ap.add_argument("--candidate_top_k", type=int, default=2)
    ap.add_argument("--memory_dir", type=str, default="central_memory")
    ap.add_argument("--ingest_memory", action="store_true", help="Ingest deployment/candidate runs into central memory.")
    ap.add_argument("--skip_eval_gate", action="store_true", help="Skip strict A/B benchmark gate before promotion.")
    ap.add_argument("--eval_suite_mode", type=str, default="target", choices=["target", "quick", "full"])
    ap.add_argument("--eval_bootstrap_samples", type=int, default=1000)
    ap.add_argument("--eval_alpha", type=float, default=0.05)
    ap.add_argument("--no_market_update", action="store_true", help="Disable KnowledgeMarket auto reputation updates.")
    ap.add_argument("--market_memory_dir", type=str, default="central_memory")
    ap.add_argument("--market_scale", type=float, default=0.5, help="Scale applied to return-delta reputation updates.")
    ap.add_argument("--skip_behavioral_surgeon", action="store_true", help="Skip pre-deploy Behavioral Surgeon update.")
    ap.add_argument("--surgeon_cliff_lookback_runs", type=int, default=10)
    ap.add_argument("--surgeon_behavior_top_percent", type=float, default=20.0)
    ap.add_argument("--surgeon_cliff_reward_threshold", type=float, default=-50.0)
    ap.add_argument("--surgeon_death_similarity", type=float, default=0.95)
    ap.add_argument("--surgeon_active_forgetting_similarity", type=float, default=0.95)
    ap.add_argument("--skip_park_cliff_gate", action="store_true", help="Skip mandatory park+cliff candidate gates.")
    ap.add_argument("--skip_promotion_board", action="store_true", help="Skip multi-critic promotion board.")
    ap.add_argument("--promotion_single_critic", action="store_true", help="Allow promotion if either critic passes.")
    ap.add_argument("--promotion_disagreement_policy", type=str, default="quarantine", choices=["quarantine", "reject", "allow"])
    ap.add_argument("--promotion_quarantine_dir", type=str, default=os.path.join("models", "quarantine"))
    ap.add_argument("--promotion_long_horizon_eps_per_seed", type=int, default=3)
    ap.add_argument("--promotion_external_benchmark_json", type=str, default="")
    ap.add_argument(
        "--promotion_human_decision_path",
        type=str,
        default=os.path.join("models", "promotion_board_human_decisions.json"),
    )
    ap.add_argument("--promotion_require_human_bless", action="store_true")
    ap.add_argument("--promotion_decision_ttl_hours", type=float, default=168.0)
    ap.add_argument("--skip_contracts", action="store_true", help="Skip skill contract gating on promotion.")
    ap.add_argument("--contracts_path", type=str, default=os.path.join("models", "skill_contracts.json"))
    ap.add_argument("--contracts_strict_delta", type=float, default=0.01)
    args = ap.parse_args()

    manifest = _read_manifest(args.manifest_path)
    dep = _as_dict(manifest.get("deployment_ready_defaults"))
    robust = _as_dict(manifest.get("winners_robust"))
    contracts = None
    if not args.skip_contracts:
        contracts = SkillContractManager(
            ContractConfig(
                enabled=True,
                path=str(args.contracts_path),
                strict_improvement_delta=float(args.contracts_strict_delta),
            )
        )
    market = None if args.no_market_update else KnowledgeMarket(KnowledgeMarketConfig(root_dir=args.market_memory_dir))
    market_updates: List[Dict[str, Any]] = []

    if args.verse:
        target_verses = [str(args.verse)]
    else:
        target_verses = sorted(set(list(dep.keys()) + list(robust.keys())))
    if not target_verses:
        raise RuntimeError("No verses found in manifest deployment sections.")

    print(f"Deployment target verses: {target_verses}")
    if not args.skip_behavioral_surgeon:
        print("")
        print("Running Behavioral Surgeon pre-step...")
        _run_behavioral_surgeon(
            cliff_lookback_runs=int(args.surgeon_cliff_lookback_runs),
            behavior_top_percent=float(args.surgeon_behavior_top_percent),
            cliff_reward_threshold=float(args.surgeon_cliff_reward_threshold),
            death_similarity_threshold=float(args.surgeon_death_similarity),
            active_forgetting_similarity=float(args.surgeon_active_forgetting_similarity),
        )
        print("Behavioral Surgeon pre-step complete.")

    manifest_changed = False
    per_verse_summary: List[Dict[str, Any]] = []

    for verse in target_verses:
        print("")
        print(f"[verse={verse}]")
        entry = _entry_for_verse(manifest, verse)
        if not entry:
            print("No manifest entry for verse. Skipping.")
            continue

        baseline = _baseline_metrics(entry)
        print(
            "baseline: "
            f"policy={baseline.get('policy','')} "
            f"mean_return={_safe_float(baseline.get('mean_return', 0.0)):.3f} "
            f"success={baseline.get('success_rate')}"
        )

        promoted = False
        candidate_run_id = None
        candidate_stats: Optional[RunStats] = None
        candidate_status = "not_run"

        if not args.skip_regression:
            expert_dir = os.path.join("models", "expert_datasets")
            if os.path.isdir(expert_dir):
                before = copy.deepcopy(manifest)
                try:
                    cand_max_steps = int(args.candidate_max_steps) if int(args.candidate_max_steps) > 0 else _default_max_steps(verse)
                    candidate_run_id, candidate_stats = _run_once(
                        verse=verse,
                        agent_algo="special_moe",
                        agent_config=_candidate_cfg(verse, top_k=args.candidate_top_k),
                        episodes=int(args.candidate_episodes),
                        max_steps=cand_max_steps,
                        seed=int(args.seed),
                        runs_root=args.runs_root,
                    )
                    candidate = {
                        "mean_return": float(candidate_stats.mean_return),
                        "success_rate": candidate_stats.success_rate,
                    }
                    gate_passed = True
                    gate_result = None
                    if not args.skip_eval_gate:
                        gate_result = run_ab_gate(
                            baseline_spec=_baseline_spec_for_gate(
                                verse=verse,
                                manifest_path=args.manifest_path,
                                seed=int(args.seed),
                            ),
                            candidate_spec=_candidate_spec_for_gate(
                                verse=verse,
                                top_k=int(args.candidate_top_k),
                                seed=int(args.seed),
                            ),
                            suite_mode=str(args.eval_suite_mode),
                            target_verse=(verse if str(args.eval_suite_mode) == "target" else None),
                            bootstrap_samples=int(args.eval_bootstrap_samples),
                            alpha=float(args.eval_alpha),
                        )
                        print_gate_report(gate_result, label=f"promotion_gate:{verse}")
                        gate_passed = bool(gate_result.passed)
                    if gate_passed and not args.skip_park_cliff_gate:
                        extra_gate_verses = ["park_world", "cliff_world"]
                        for gv in extra_gate_verses:
                            gv_spec = _candidate_spec_for_gate(
                                verse=gv,
                                top_k=int(args.candidate_top_k),
                                seed=int(args.seed),
                            )
                            gv_passed, gv_info = _candidate_absolute_gate(gv, gv_spec)
                            print(
                                f"absolute_gate:{gv} passed={gv_passed} "
                                f"mean_return={float(gv_info['candidate']['mean_return']):.3f} "
                                f"success={float(gv_info['candidate']['success_rate']):.3f} "
                                f"safety={float(gv_info['candidate']['safety_violation_rate']):.3f}"
                            )
                            if not gv_passed:
                                gate_passed = False
                                break
                    board_result = None
                    if gate_passed and not args.skip_promotion_board:
                        board_result = run_promotion_board(
                            baseline_spec=_baseline_spec_for_gate(
                                verse=verse,
                                manifest_path=args.manifest_path,
                                seed=int(args.seed),
                            ),
                            candidate_spec=_candidate_spec_for_gate(
                                verse=verse,
                                top_k=int(args.candidate_top_k),
                                seed=int(args.seed),
                            ),
                            target_verse=str(verse),
                            cfg=PromotionConfig.from_dict(
                                {
                                    "enabled": True,
                                    "require_multi_critic": (not bool(args.promotion_single_critic)),
                                    "disagreement_policy": str(args.promotion_disagreement_policy),
                                    "quarantine_dir": str(args.promotion_quarantine_dir),
                                    "long_horizon_episodes_per_seed": int(args.promotion_long_horizon_eps_per_seed),
                                    "bootstrap_samples": int(args.eval_bootstrap_samples),
                                    "alpha": float(args.eval_alpha),
                                    "external_benchmark_json": str(args.promotion_external_benchmark_json),
                                    "human_decision_path": str(args.promotion_human_decision_path),
                                    "require_human_bless": bool(args.promotion_require_human_bless),
                                    "decision_ttl_hours": float(args.promotion_decision_ttl_hours),
                                }
                            ),
                        )
                        gate_passed = bool(board_result.get("passed", False))
                        print(
                            "promotion_board: "
                            f"passed={gate_passed} disagreed={bool(board_result.get('disagreed', False))} "
                            f"policy={board_result.get('policy')}"
                        )
                        if board_result.get("quarantine_report"):
                            print(f"promotion_board: quarantine_report={board_result.get('quarantine_report')}")
                    if gate_passed and contracts is not None:
                        contract_spec = _candidate_spec_for_gate(
                            verse=verse,
                            top_k=int(args.candidate_top_k),
                            seed=int(args.seed),
                        )
                        abs_ok, abs_info = _candidate_absolute_gate(verse, contract_spec)
                        cand_metrics = {
                            "mean_return": float(abs_info["candidate"]["mean_return"]),
                            "success_rate": float(abs_info["candidate"]["success_rate"]),
                            "safety_violation_rate": float(abs_info["candidate"]["safety_violation_rate"]),
                        }
                        sat, reasons = contracts.check_satisfied(
                            verse_name=str(verse),
                            skill_tag="special_moe",
                            metrics=cand_metrics,
                        )
                        existing = contracts.get(verse_name=str(verse), skill_tag="special_moe")
                        if not sat:
                            gate_passed = False
                            print(f"contracts: failed reasons={reasons}")
                        else:
                            # Register/update contract; updates require strict improvement.
                            up = contracts.register_or_update(
                                verse_name=str(verse),
                                skill_tag="special_moe",
                                metrics=cand_metrics,
                                safety_invariants={"gate_absolute": bool(abs_ok)},
                                strict_improvement_delta=float(args.contracts_strict_delta),
                            )
                            if existing is not None and not bool(up.get("updated", False)):
                                gate_passed = False
                                print(f"contracts: strict improvement not met ({up.get('reason')})")
                            else:
                                print(
                                    "contracts: "
                                    f"ok updated={bool(up.get('updated', False))} reason={up.get('reason')}"
                                )

                    if not gate_passed:
                        manifest = before
                        candidate_status = "gate_failed"
                        print("regression-check: no change (promotion gate failed)")
                    elif _is_regression(candidate, baseline):
                        # Rollback: keep previous manifest defaults.
                        manifest = before
                        candidate_status = "regression"
                        print(
                            "regression-check: rollback "
                            f"(candidate run {candidate_run_id} mean_return={candidate['mean_return']:.3f})"
                        )
                    elif _is_improvement(candidate, baseline):
                        manifest = _promote_special_moe(
                            manifest,
                            verse=verse,
                            run_id=candidate_run_id,
                            stats=candidate_stats,
                            candidate_top_k=int(args.candidate_top_k),
                            runs_root=args.runs_root,
                        )
                        manifest_changed = True
                        promoted = True
                        candidate_status = "promoted"
                        print(
                            "regression-check: promoted special_moe "
                            f"(run {candidate_run_id}, mean_return={candidate['mean_return']:.3f})"
                        )
                    else:
                        manifest = before
                        candidate_status = "no_change"
                        print(
                            "regression-check: no change "
                            f"(candidate run {candidate_run_id} mean_return={candidate['mean_return']:.3f})"
                        )
                except Exception as e:
                    manifest = before
                    candidate_status = "error"
                    print(f"regression-check: skipped due to error: {e}")
            else:
                candidate_status = "no_expert_dir"
                print("regression-check: skipped (models/expert_datasets missing)")

        if market is not None and candidate_run_id and candidate_stats is not None:
            delta = (
                float(candidate_stats.mean_return) - float(_safe_float(baseline.get("mean_return", 0.0)))
            ) * float(args.market_scale)
            upd = market.update_reputation(
                provider_id=str(candidate_run_id),
                delta=float(delta),
                reason=f"candidate_eval:{verse}:{candidate_status}",
                consumer_agent_id=f"deploy_candidate:{verse}",
            )
            market_updates.append({"verse": verse, "type": "candidate", **upd})
            print(
                "market-update: "
                f"provider={upd['provider_id']} delta={float(delta):.3f} "
                f"new_rep={float(upd['new_reputation']):.3f}"
            )

        # Launch deployed gateway agent for this verse.
        deploy_max_steps = int(args.max_steps) if int(args.max_steps) > 0 else _default_max_steps(verse)
        deploy_run_id, deploy_stats = _run_once(
            verse=verse,
            agent_algo="gateway",
            agent_config={"manifest_path": args.manifest_path},
            episodes=int(args.episodes),
            max_steps=deploy_max_steps,
            seed=int(args.seed),
            runs_root=args.runs_root,
        )
        print(
            "deploy: "
            f"run={deploy_run_id} mean_return={deploy_stats.mean_return:.3f} "
            f"success={deploy_stats.success_rate}"
        )

        if market is not None:
            current_entry = _entry_for_verse(manifest, verse) or entry
            provider_id = _provider_id_from_entry(current_entry)
            provider_ref = _baseline_metrics(current_entry)
            if provider_id:
                delta = (
                    float(deploy_stats.mean_return) - float(_safe_float(provider_ref.get("mean_return", 0.0)))
                ) * float(args.market_scale)
                upd = market.update_reputation(
                    provider_id=str(provider_id),
                    delta=float(delta),
                    reason=f"deploy_eval:{verse}",
                    consumer_agent_id=f"gateway:{verse}",
                )
                market_updates.append({"verse": verse, "type": "deploy", **upd})
                print(
                    "market-update: "
                    f"provider={upd['provider_id']} delta={float(delta):.3f} "
                    f"new_rep={float(upd['new_reputation']):.3f}"
                )

        if args.ingest_memory:
            if candidate_run_id:
                _ingest_to_central(candidate_run_id, args.memory_dir, args.runs_root)
            _ingest_to_central(deploy_run_id, args.memory_dir, args.runs_root)

        per_verse_summary.append(
            {
                "verse": verse,
                "baseline_policy": baseline.get("policy"),
                "baseline_mean_return": _safe_float(baseline.get("mean_return", 0.0)),
                "candidate_run_id": candidate_run_id,
                "candidate_mean_return": None if candidate_stats is None else float(candidate_stats.mean_return),
                "promoted": bool(promoted),
                "deploy_run_id": deploy_run_id,
                "deploy_mean_return": float(deploy_stats.mean_return),
            }
        )

    if manifest_changed:
        _write_manifest(args.manifest_path, manifest, backup=True)
        print(f"\nManifest updated: {args.manifest_path} (backup: {args.manifest_path}.bak)")
    else:
        print(f"\nManifest unchanged: {args.manifest_path}")

    print("\nSummary")
    for row in per_verse_summary:
        print(
            f"- verse={row['verse']} promoted={row['promoted']} "
            f"deploy_run={row['deploy_run_id']} deploy_mean_return={row['deploy_mean_return']:.3f}"
        )
    if market_updates:
        print(f"market_updates={len(market_updates)}")


if __name__ == "__main__":
    main()
