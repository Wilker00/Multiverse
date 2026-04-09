"""
tools/validate_sf_transfer.py

Validation harness for "Multiverse V2" transfer hypothesis:
1) Universal perceptual interface via egocentric local occupancy grid.
2) Successor Features (SF) transfer: dynamics (psi) transferred separately from reward weights (w).
3) Semantic bridge style task preference vector as w initialization.

This tool is intentionally self-contained and does not modify runtime agents.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.sf_transfer_features import _adaptive_triage_direct_features
from core.sf_transfer_gate import _adaptive_gate_cfg, _adaptive_gate_decision, _adaptive_gate_model_cfg


from verses.registry import register_builtin
from tools.validate_sf_transfer_runtime import (
    EpisodeStats,
    TabularSFAgent,
    _episode_trace,
    _policy_bank_agreement_diag,
    _project_psi_table_to_actions,
    _ridge_reward_weights,
    _run_episode,
    _semantic_reward_weights,
    _slope,
    _summarize,
    _trace_delta,
    _trace_diagnostics,
    _train_then_eval,
)
from tools.validate_sf_transfer_support import (
    EgoGridAdapter,
    EgoObservation,
    _parse_int_grid,
    _parse_seed_list,
    _parse_str_list,
    _safe_float,
    _safe_int,
    _score_learned_softmax_model,
    _softmax,
)


def _triage_canary_probe_summary(row: Dict[str, Any], cond_key: str) -> Dict[str, float]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    cond = tc.get(str(cond_key), {}) if isinstance(tc.get(str(cond_key)), dict) else {}
    cvs = cond.get("canary_vs_scratch_early", {}) if isinstance(cond.get("canary_vs_scratch_early"), dict) else {}
    diag = cvs.get("diagnostics", {}) if isinstance(cvs.get("diagnostics"), dict) else {}
    # hazard_mean is transfer-minus-scratch hazard delta (positive = worse hazard); convert to gain.
    hazard_delta = _safe_float(diag.get("hazard_mean", 0.0), 0.0)
    hazard_gain = float(-hazard_delta)
    return {
        "episodes": _safe_float(diag.get("episodes", 0.0), 0.0),
        "success_delta_mean": _safe_float(diag.get("success_mean", 0.0), 0.0),
        "return_delta_mean": _safe_float(diag.get("return_mean", 0.0), 0.0),
        "hazard_delta_mean": float(hazard_delta),
        "hazard_gain_mean": float(hazard_gain),
        "success_delta_slope": _safe_float(diag.get("success_slope", 0.0), 0.0),
        "return_delta_slope": _safe_float(diag.get("return_slope", 0.0), 0.0),
        "hazard_delta_slope": _safe_float(diag.get("hazard_slope", 0.0), 0.0),
    }


def _canary_triad_override(
    *,
    row: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    selected: str,
    decision_reason: str,
) -> Tuple[str, str, Dict[str, Any]]:
    if not bool(runtime_policy.get("enable_canary_triad_override", False)):
        return str(selected), str(decision_reason), {"enabled": False}
    full = _triage_canary_probe_summary(row, "sf_transfer")
    warm = _triage_canary_probe_summary(row, "sf_transfer_warmup")
    base_policy = {
        "name": "competence",
        "success_weight": _safe_float(runtime_policy.get("canary_success_weight", 100.0), 100.0),
        "return_weight": _safe_float(runtime_policy.get("canary_return_weight", 1.0), 1.0),
        "hazard_weight": _safe_float(runtime_policy.get("canary_hazard_weight", 0.02), 0.02),
        "min_utility": _safe_float(runtime_policy.get("canary_min_utility", 0.0), 0.0),
        "min_hazard_gain": _safe_float(runtime_policy.get("canary_min_hazard_gain", -1e9), -1e9),
        "min_episodes": _safe_float(runtime_policy.get("canary_min_episodes", 1.0), 1.0),
    }
    safety_policy = {
        "name": "safety",
        "success_weight": _safe_float(runtime_policy.get("canary_safety_success_weight", base_policy["success_weight"]), base_policy["success_weight"]),
        "return_weight": _safe_float(runtime_policy.get("canary_safety_return_weight", base_policy["return_weight"]), base_policy["return_weight"]),
        "hazard_weight": _safe_float(runtime_policy.get("canary_safety_hazard_weight", max(base_policy["hazard_weight"], 0.05)), max(base_policy["hazard_weight"], 0.05)),
        "min_utility": _safe_float(runtime_policy.get("canary_safety_min_utility", base_policy["min_utility"]), base_policy["min_utility"]),
        "min_hazard_gain": _safe_float(runtime_policy.get("canary_safety_min_hazard_gain", max(0.0, base_policy["min_hazard_gain"])), max(0.0, base_policy["min_hazard_gain"])),
        "min_episodes": _safe_float(runtime_policy.get("canary_safety_min_episodes", base_policy["min_episodes"]), base_policy["min_episodes"]),
    }

    dual_enabled = bool(runtime_policy.get("enable_canary_dual_policy", False))
    dual_hg_split = _safe_float(runtime_policy.get("canary_dual_select_hazard_gain_threshold", 0.0), 0.0)
    dual_sd_split = _safe_float(runtime_policy.get("canary_dual_select_success_delta_threshold", -1e9), -1e9)
    max_probe_hg = max(_safe_float(full.get("hazard_gain_mean", 0.0), 0.0), _safe_float(warm.get("hazard_gain_mean", 0.0), 0.0))
    max_probe_sd = max(_safe_float(full.get("success_delta_mean", 0.0), 0.0), _safe_float(warm.get("success_delta_mean", 0.0), 0.0))
    if dual_enabled and (max_probe_hg < dual_hg_split or max_probe_sd < dual_sd_split):
        active_policy = safety_policy
        policy_reason = "dual_policy_safety"
    else:
        active_policy = base_policy
        policy_reason = "dual_policy_competence" if dual_enabled else "single_policy"

    sw = float(active_policy["success_weight"])
    rw = float(active_policy["return_weight"])
    hw = float(active_policy["hazard_weight"])
    min_u = float(active_policy["min_utility"])
    min_hg = float(active_policy["min_hazard_gain"])
    min_eps = float(active_policy["min_episodes"])

    def _utility(c: Dict[str, float]) -> float:
        return float(
            sw * _safe_float(c.get("success_delta_mean", 0.0), 0.0)
            + rw * _safe_float(c.get("return_delta_mean", 0.0), 0.0)
            + hw * _safe_float(c.get("hazard_gain_mean", 0.0), 0.0)
        )

    full_u = _utility(full)
    warm_u = _utility(warm)
    full_pass = bool(
        _safe_float(full.get("episodes", 0.0), 0.0) >= min_eps
        and full_u >= min_u
        and _safe_float(full.get("hazard_gain_mean", 0.0), 0.0) >= min_hg
    )
    warm_pass = bool(
        _safe_float(warm.get("episodes", 0.0), 0.0) >= min_eps
        and warm_u >= min_u
        and _safe_float(warm.get("hazard_gain_mean", 0.0), 0.0) >= min_hg
    )
    # Optional veto: require minimum canary success lift for full transfer,
    # unless hazard gain is very strong (safety override).
    full_success_floor = _safe_float(runtime_policy.get("canary_full_success_floor", -1e9), -1e9)
    full_hazard_override = _safe_float(runtime_policy.get("canary_full_hazard_gain_override", 25.0), 25.0)
    full_success_floor_applied = False
    full_success_floor_pass = True
    if float(full_success_floor) > -1e8:
        full_success_floor_applied = True
        full_success_floor_pass = bool(
            _safe_float(full.get("success_delta_mean", 0.0), 0.0) >= float(full_success_floor)
            or _safe_float(full.get("hazard_gain_mean", 0.0), 0.0) >= float(full_hazard_override)
        )
        if not full_success_floor_pass:
            full_pass = False
    prior = str(selected)
    reason = str(decision_reason)
    if bool(runtime_policy.get("canary_two_stage_global_select", False)):
        # Stage A: hard veto already encoded in full_pass/warm_pass.
        # Stage B: rank surviving modes by canary utility, with scratch utility baseline = 0.
        scratch_u = _safe_float(runtime_policy.get("canary_scratch_utility", 0.0), 0.0)
        candidates: List[Tuple[str, float]] = [("sf_scratch", float(scratch_u))]
        if full_pass:
            candidates.append(("sf_transfer", float(full_u)))
        if warm_pass:
            candidates.append(("sf_transfer_warmup", float(warm_u)))
        # Tie-break toward prior if utilities match closely.
        best_sel = "sf_scratch"
        best_u = float("-inf")
        for name, u in candidates:
            if (u > best_u) or (abs(u - best_u) <= 1e-12 and name == prior):
                best_sel = str(name)
                best_u = float(u)
        selected = str(best_sel)
        if selected == prior:
            reason = "canary_two_stage_keep_prior"
        elif selected == "sf_scratch":
            reason = "canary_two_stage_global_to_scratch"
        elif selected == "sf_transfer":
            reason = "canary_two_stage_global_to_full"
        else:
            reason = "canary_two_stage_global_to_warmup"
    elif prior == "sf_transfer" and not full_pass:
        if warm_pass and warm_u > full_u:
            selected = "sf_transfer_warmup"
            reason = "canary_override_full_to_warmup"
        else:
            selected = "sf_scratch"
            reason = "canary_override_full_to_scratch"
    elif prior == "sf_transfer_warmup" and not warm_pass:
        if full_pass and full_u > warm_u:
            selected = "sf_transfer"
            reason = "canary_override_warmup_to_full"
        else:
            selected = "sf_scratch"
            reason = "canary_override_warmup_to_scratch"
    elif prior == "sf_scratch":
        if full_pass or warm_pass:
            if full_pass and (not warm_pass or full_u >= warm_u):
                selected = "sf_transfer"
                reason = "canary_override_scratch_to_full"
            elif warm_pass:
                selected = "sf_transfer_warmup"
                reason = "canary_override_scratch_to_warmup"
    return str(selected), str(reason), {
        "enabled": True,
        "prior_selected_condition": str(prior),
        "prior_decision_reason": str(decision_reason),
        "weights": {
            "success_weight": float(sw),
            "return_weight": float(rw),
            "hazard_gain_weight": float(hw),
        },
        "active_policy": str(active_policy.get("name", "competence")),
        "policy_choice_reason": str(policy_reason),
        "dual_policy": {
            "enabled": bool(dual_enabled),
            "hazard_gain_threshold": float(dual_hg_split),
            "success_delta_threshold": float(dual_sd_split),
            "max_probe_hazard_gain": float(max_probe_hg),
            "max_probe_success_delta": float(max_probe_sd),
            "competence_policy": dict(base_policy),
            "safety_policy": dict(safety_policy),
        },
        "thresholds": {
            "min_utility": float(min_u),
            "min_hazard_gain": float(min_hg),
            "min_episodes": float(min_eps),
            "two_stage_global_select": bool(runtime_policy.get("canary_two_stage_global_select", False)),
            "scratch_utility": float(_safe_float(runtime_policy.get("canary_scratch_utility", 0.0), 0.0)),
            "full_success_floor": float(full_success_floor),
            "full_hazard_gain_override": float(full_hazard_override),
        },
        "full_probe": {
            **full,
            "utility": float(full_u),
            "pass": bool(full_pass),
            "success_floor_applied": bool(full_success_floor_applied),
            "success_floor_pass": bool(full_success_floor_pass),
        },
        "warmup_probe": {**warm, "utility": float(warm_u), "pass": bool(warm_pass)},
        "overrode": bool(str(prior) != str(selected)),
    }


def _seed_condition_summary(cond: Dict[str, Any]) -> Dict[str, Optional[float]]:
    eval_sum = cond.get("eval_summary", {}) if isinstance(cond.get("eval_summary"), dict) else {}
    early_sum = cond.get("early_train_summary", {}) if isinstance(cond.get("early_train_summary"), dict) else {}
    return {
        "eval_return": (
            None if not isinstance(eval_sum.get("mean_return"), (int, float)) else float(eval_sum.get("mean_return"))
        ),
        "eval_success_rate": (
            None
            if not isinstance(eval_sum.get("success_rate"), (int, float))
            else float(eval_sum.get("success_rate"))
        ),
        "eval_hazard_per_1k": (
            None
            if not isinstance(eval_sum.get("hazard_per_1k"), (int, float))
            else float(eval_sum.get("hazard_per_1k"))
        ),
        "eval_forward_mse": (
            None
            if not isinstance(eval_sum.get("mean_forward_mse"), (int, float))
            else float(eval_sum.get("mean_forward_mse"))
        ),
        "early_success_rate": (
            None
            if not isinstance(early_sum.get("success_rate"), (int, float))
            else float(early_sum.get("success_rate"))
        ),
        "early_return": (
            None if not isinstance(early_sum.get("mean_return"), (int, float)) else float(early_sum.get("mean_return"))
        ),
        "early_hazard_per_1k": (
            None
            if not isinstance(early_sum.get("hazard_per_1k"), (int, float))
            else float(early_sum.get("hazard_per_1k"))
        ),
        "early_forward_mse": (
            None
            if not isinstance(early_sum.get("mean_forward_mse"), (int, float))
            else float(early_sum.get("mean_forward_mse"))
        ),
    }


def _sf_universe_relation_safe(source_verse_name: str, target_verse_name: str) -> str:
    try:
        from core.taxonomy import universe_relation

        return str(universe_relation(str(source_verse_name), str(target_verse_name)))
    except Exception:
        return "unknown"


def _sf_transfer_decision_record(
    *,
    row: Dict[str, Any],
    adaptive_cfg: Dict[str, Any],
    transfer_condition_key: str,
) -> Dict[str, Any]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    scratch = tc.get("sf_scratch", {}) if isinstance(tc.get("sf_scratch"), dict) else {}
    transfer = tc.get(transfer_condition_key, {}) if isinstance(tc.get(transfer_condition_key), dict) else {}
    decision = _adaptive_gate_decision(
        row,
        adaptive_cfg,
        transfer_gate_condition_key_override=transfer_condition_key,
    )
    accept = bool(decision.get("accept_transfer", False))
    chosen_key = str(transfer_condition_key if accept else "sf_scratch")

    def _safe_block(cond: Dict[str, Any], k: str) -> Dict[str, Any]:
        v = cond.get(k, {})
        return v if isinstance(v, dict) else {}

    scratch_eval = _safe_block(scratch, "eval_summary")
    transfer_eval = _safe_block(transfer, "eval_summary")
    scratch_early = _safe_block(scratch, "early_train_summary")
    transfer_early = _safe_block(transfer, "early_train_summary")

    return {
        "schema_version": "transfer_decision_record.v1",
        "transfer_mode": "sf_transfer_adaptive_gate",
        "seed": int(row.get("seed", 0) or 0),
        "source_verse_name": str(row.get("source_verse_name", "")),
        "target_verse_name": str(row.get("target_verse_name", "")),
        "universe_relation": _sf_universe_relation_safe(
            str(row.get("source_verse_name", "")),
            str(row.get("target_verse_name", "")),
        ),
        "transfer_candidate_condition": str(transfer_condition_key),
        "decision": ("accept_transfer" if accept else "fallback_scratch"),
        "selected_condition": chosen_key,
        "decision_reason": str(decision.get("decision_reason", "")),
        "adaptive_gate": {
            "enabled": bool(adaptive_cfg.get("enabled", False)),
            "kind": str(adaptive_cfg.get("kind", "")),
            "logic": str(adaptive_cfg.get("logic", "")),
            "transfer_gate_condition_key": str(adaptive_cfg.get("transfer_gate_condition_key", "")),
            "thresholds": dict(adaptive_cfg.get("thresholds", {})) if isinstance(adaptive_cfg.get("thresholds"), dict) else {},
        },
        "gate_outcome": {
            "accept_transfer": bool(accept),
            "checks": dict(decision.get("checks", {})) if isinstance(decision.get("checks"), dict) else {},
            "model_gate": dict(decision.get("model_gate", {})) if isinstance(decision.get("model_gate"), dict) else {},
        },
        "gate_inputs": {
            "scratch_early": dict(decision.get("scratch_early", {})) if isinstance(decision.get("scratch_early"), dict) else {},
            "scratch_eval": dict(decision.get("scratch_eval", {})) if isinstance(decision.get("scratch_eval"), dict) else {},
            "transfer_early": dict(decision.get("transfer_early", {})) if isinstance(decision.get("transfer_early"), dict) else {},
            "canary_early_deltas": dict(decision.get("canary_early_deltas", {})) if isinstance(decision.get("canary_early_deltas"), dict) else {},
            "transfer_early_trends": dict(decision.get("transfer_early_trends", {})) if isinstance(decision.get("transfer_early_trends"), dict) else {},
            "canary_early_delta_trends": dict(decision.get("canary_early_delta_trends", {})) if isinstance(decision.get("canary_early_delta_trends"), dict) else {},
            "source_policy_bank_agreement": dict(decision.get("source_policy_bank_agreement", {})) if isinstance(decision.get("source_policy_bank_agreement"), dict) else {},
        },
        "counterfactual": {
            "scratch_eval_success_rate": _safe_float(scratch_eval.get("success_rate", 0.0), 0.0),
            "transfer_eval_success_rate": _safe_float(transfer_eval.get("success_rate", 0.0), 0.0),
            "transfer_minus_scratch_eval_success_rate": (
                _safe_float(transfer_eval.get("success_rate", 0.0), 0.0)
                - _safe_float(scratch_eval.get("success_rate", 0.0), 0.0)
            ),
            "scratch_eval_return": _safe_float(scratch_eval.get("mean_return", 0.0), 0.0),
            "transfer_eval_return": _safe_float(transfer_eval.get("mean_return", 0.0), 0.0),
            "transfer_minus_scratch_eval_return": (
                _safe_float(transfer_eval.get("mean_return", 0.0), 0.0)
                - _safe_float(scratch_eval.get("mean_return", 0.0), 0.0)
            ),
            "scratch_eval_hazard_per_1k": _safe_float(scratch_eval.get("hazard_per_1k", 0.0), 0.0),
            "transfer_eval_hazard_per_1k": _safe_float(transfer_eval.get("hazard_per_1k", 0.0), 0.0),
            "scratch_minus_transfer_eval_hazard_per_1k": (
                _safe_float(scratch_eval.get("hazard_per_1k", 0.0), 0.0)
                - _safe_float(transfer_eval.get("hazard_per_1k", 0.0), 0.0)
            ),
            "scratch_early_success_rate": _safe_float(scratch_early.get("success_rate", 0.0), 0.0),
            "transfer_early_success_rate": _safe_float(transfer_early.get("success_rate", 0.0), 0.0),
        },
    }


def _adaptive_gate_triage_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    if not (isinstance(cfg, dict) and bool(cfg.get("enabled", False))):
        return False
    if str(cfg.get("kind", "")).strip().lower() != "learned_gate_model":
        return False
    policy = cfg.get("model_policy", {}) if isinstance(cfg.get("model_policy"), dict) else {}
    return bool(policy.get("enable_triage", False))


def _adaptive_triage_decision(row: Dict[str, Any], adaptive_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg_full = dict(adaptive_cfg) if isinstance(adaptive_cfg, dict) else {}
    cfg_warm = dict(adaptive_cfg) if isinstance(adaptive_cfg, dict) else {}
    payload = cfg_full.get("model", {}) if isinstance(cfg_full.get("model"), dict) else {}
    runtime_policy = cfg_full.get("model_policy", {}) if isinstance(cfg_full.get("model_policy"), dict) else {}
    direct_model = payload.get("triage_direct_model", {}) if isinstance(payload.get("triage_direct_model"), dict) else {}
    if direct_model:
        feats = _adaptive_triage_direct_features(row)
        score = _score_learned_softmax_model(features=feats, model_block=direct_model)
        class_names = [str(x) for x in score.get("class_names", [])]
        probs = [float(x) for x in score.get("probs", [])]
        prob_map = {class_names[i]: probs[i] for i in range(min(len(class_names), len(probs)))}
        payload_policy = payload.get("triage_direct_policy", {}) if isinstance(payload.get("triage_direct_policy"), dict) else {}
        full_thr = _safe_float(runtime_policy.get("triage_full_accept_prob_min", payload_policy.get("full_accept_prob_min", 0.55)), 0.55)
        warm_thr = _safe_float(runtime_policy.get("triage_warmup_accept_prob_min", payload_policy.get("warmup_accept_prob_min", 0.50)), 0.50)
        prefer_higher = bool(payload_policy.get("prefer_higher_probability", True))
        p_s = _safe_float(prob_map.get("sf_scratch", 0.0), 0.0)
        p_f = _safe_float(prob_map.get("sf_transfer", 0.0), 0.0)
        p_w = _safe_float(prob_map.get("sf_transfer_warmup", 0.0), 0.0)
        full_ok = bool(p_f >= full_thr)
        warm_ok = bool(p_w >= warm_thr)
        if full_ok and warm_ok:
            if prefer_higher and p_f >= p_w:
                selected = "sf_transfer"
                reason = "direct_triad_both_accept_pick_full"
            elif prefer_higher:
                selected = "sf_transfer_warmup"
                reason = "direct_triad_both_accept_pick_warmup"
            else:
                # Conservative: prefer the better transfer mode only if it beats scratch confidence.
                if p_f >= p_s and p_f >= p_w:
                    selected = "sf_transfer"
                    reason = "direct_triad_both_accept_full_over_scratch"
                elif p_w >= p_s and p_w >= p_f:
                    selected = "sf_transfer_warmup"
                    reason = "direct_triad_both_accept_warmup_over_scratch"
                else:
                    selected = "sf_scratch"
                    reason = "direct_triad_both_accept_but_scratch_dominates"
        elif full_ok and (p_f >= p_s):
            selected = "sf_transfer"
            reason = "direct_triad_full_only"
        elif warm_ok and (p_w >= p_s):
            selected = "sf_transfer_warmup"
            reason = "direct_triad_warmup_only"
        else:
            selected = "sf_scratch"
            reason = "direct_triad_fallback_scratch"
        selected, reason, canary_override = _canary_triad_override(
            row=row,
            runtime_policy=runtime_policy,
            selected=str(selected),
            decision_reason=str(reason),
        )
        return {
            "selected_condition": str(selected),
            "decision_reason": str(reason),
            "full_decision": {
                "enabled": True,
                "accept_transfer": bool(selected == "sf_transfer"),
                "decision_reason": "direct_triad_selector",
                "model_gate": {
                    "model_type": "triage_softmax_linear",
                    "probability_help": float(p_f),
                    "probability_hard": None,
                },
            },
            "warm_decision": {
                "enabled": True,
                "accept_transfer": bool(selected == "sf_transfer_warmup"),
                "decision_reason": "direct_triad_selector",
                "model_gate": {
                    "model_type": "triage_softmax_linear",
                    "probability_help": float(p_w),
                    "probability_hard": None,
                },
            },
            "triage_policy": {
                "full_accept_prob_min": float(full_thr),
                "warmup_accept_prob_min": float(warm_thr),
                "prefer_higher_probability": bool(prefer_higher),
            },
            "direct_triad": {
                "class_names": class_names,
                "probs": [float(x) for x in probs],
                "probabilities": {
                    "sf_scratch": float(p_s),
                    "sf_transfer": float(p_f),
                    "sf_transfer_warmup": float(p_w),
                },
                "feature_count": int(score.get("feature_count", 0)),
            },
            "canary_override": canary_override,
        }
    triage_policy = payload.get("triage_policy", {}) if isinstance(payload.get("triage_policy"), dict) else {}
    if triage_policy:
        mp_full = dict(cfg_full.get("model_policy", {})) if isinstance(cfg_full.get("model_policy"), dict) else {}
        mp_warm = dict(cfg_warm.get("model_policy", {})) if isinstance(cfg_warm.get("model_policy"), dict) else {}
        if "hard_prob_min" in triage_policy:
            hp = _safe_float(triage_policy.get("hard_prob_min"), -1.0)
            mp_full["hard_prob_min"] = float(hp)
            mp_warm["hard_prob_min"] = float(hp)
        if "full_accept_prob_min" in triage_policy:
            fp = _safe_float(triage_policy.get("full_accept_prob_min"), 0.6)
            mp_full["accept_prob_min"] = float(fp)
            mp_full["warmup_prob_min"] = float(fp)
        if "warmup_accept_prob_min" in triage_policy:
            wp = _safe_float(triage_policy.get("warmup_accept_prob_min"), 0.4)
            mp_warm["accept_prob_min"] = float(wp)
            mp_warm["warmup_prob_min"] = float(wp)
        if "triage_full_accept_prob_min" in runtime_policy:
            fp_rt = _safe_float(runtime_policy.get("triage_full_accept_prob_min"), mp_full.get("accept_prob_min", 0.6))
            mp_full["accept_prob_min"] = float(fp_rt)
            mp_full["warmup_prob_min"] = float(fp_rt)
        if "triage_warmup_accept_prob_min" in runtime_policy:
            wp_rt = _safe_float(runtime_policy.get("triage_warmup_accept_prob_min"), mp_warm.get("accept_prob_min", 0.4))
            mp_warm["accept_prob_min"] = float(wp_rt)
            mp_warm["warmup_prob_min"] = float(wp_rt)
        cfg_full["model_policy"] = mp_full
        cfg_warm["model_policy"] = mp_warm
    d_full = _adaptive_gate_decision(row, cfg_full, transfer_gate_condition_key_override="sf_transfer")
    d_warm = _adaptive_gate_decision(row, cfg_warm, transfer_gate_condition_key_override="sf_transfer_warmup")
    mg_full = d_full.get("model_gate", {}) if isinstance(d_full.get("model_gate"), dict) else {}
    mg_warm = d_warm.get("model_gate", {}) if isinstance(d_warm.get("model_gate"), dict) else {}
    p_full = _safe_float(mg_full.get("probability_help", 0.0), 0.0)
    p_warm = _safe_float(mg_warm.get("probability_help", 0.0), 0.0)
    full_ok = bool(d_full.get("accept_transfer", False))
    warm_ok = bool(d_warm.get("accept_transfer", False))
    if full_ok and warm_ok:
        if p_full >= p_warm:
            selected = "sf_transfer"
            reason = "triage_both_accept_pick_full"
        else:
            selected = "sf_transfer_warmup"
            reason = "triage_both_accept_pick_warmup"
    elif full_ok:
        selected = "sf_transfer"
        reason = "triage_full_only"
    elif warm_ok:
        selected = "sf_transfer_warmup"
        reason = "triage_warmup_only"
    else:
        selected = "sf_scratch"
        reason = "triage_fallback_scratch"
    selected, reason, canary_override = _canary_triad_override(
        row=row,
        runtime_policy=runtime_policy,
        selected=str(selected),
        decision_reason=str(reason),
    )
    return {
        "enabled": True,
        "kind": "learned_gate_triad_selector",
        "selected_condition": str(selected),
        "decision_reason": str(reason),
        "triage_policy": dict(triage_policy) if isinstance(triage_policy, dict) else {},
        "canary_override": canary_override,
        "candidates": {
            "sf_transfer": {
                "accept_transfer": bool(full_ok),
                "probability_help": float(p_full),
                "model_gate": dict(mg_full),
                "checks": dict(d_full.get("checks", {})) if isinstance(d_full.get("checks"), dict) else {},
            },
            "sf_transfer_warmup": {
                "accept_transfer": bool(warm_ok),
                "probability_help": float(p_warm),
                "model_gate": dict(mg_warm),
                "checks": dict(d_warm.get("checks", {})) if isinstance(d_warm.get("checks"), dict) else {},
            },
        },
        "scratch_early": dict(d_full.get("scratch_early", {})) if isinstance(d_full.get("scratch_early"), dict) else {},
        "scratch_eval": dict(d_full.get("scratch_eval", {})) if isinstance(d_full.get("scratch_eval"), dict) else {},
    }


def _sf_transfer_triage_decision_record(
    *,
    row: Dict[str, Any],
    adaptive_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    tri = _adaptive_triage_decision(row, adaptive_cfg)
    sel = str(tri.get("selected_condition", "sf_scratch"))
    scratch_eval = (((tc.get("sf_scratch") or {}) if isinstance(tc.get("sf_scratch"), dict) else {}).get("eval_summary") or {})
    full_eval = (((tc.get("sf_transfer") or {}) if isinstance(tc.get("sf_transfer"), dict) else {}).get("eval_summary") or {})
    warm_eval = (((tc.get("sf_transfer_warmup") or {}) if isinstance(tc.get("sf_transfer_warmup"), dict) else {}).get("eval_summary") or {})
    return {
        "schema_version": "transfer_decision_record.v1",
        "transfer_mode": "sf_transfer_adaptive_triad_gate",
        "seed": int(row.get("seed", 0) or 0),
        "source_verse_name": str(row.get("source_verse_name", "")),
        "target_verse_name": str(row.get("target_verse_name", "")),
        "universe_relation": _sf_universe_relation_safe(
            str(row.get("source_verse_name", "")),
            str(row.get("target_verse_name", "")),
        ),
        "decision": ("accept_transfer" if sel != "sf_scratch" else "fallback_scratch"),
        "selected_condition": sel,
        "decision_reason": str(tri.get("decision_reason", "")),
        "triage": tri,
        "counterfactual": {
            "scratch_eval_success_rate": _safe_float(scratch_eval.get("success_rate", 0.0), 0.0),
            "transfer_eval_success_rate": _safe_float(full_eval.get("success_rate", 0.0), 0.0),
            "warmup_eval_success_rate": _safe_float(warm_eval.get("success_rate", 0.0), 0.0),
            "scratch_eval_return": _safe_float(scratch_eval.get("mean_return", 0.0), 0.0),
            "transfer_eval_return": _safe_float(full_eval.get("mean_return", 0.0), 0.0),
            "warmup_eval_return": _safe_float(warm_eval.get("mean_return", 0.0), 0.0),
            "scratch_eval_hazard_per_1k": _safe_float(scratch_eval.get("hazard_per_1k", 0.0), 0.0),
            "transfer_eval_hazard_per_1k": _safe_float(full_eval.get("hazard_per_1k", 0.0), 0.0),
            "warmup_eval_hazard_per_1k": _safe_float(warm_eval.get("hazard_per_1k", 0.0), 0.0),
        },
    }


def _summarize_condition_from_rows(rows: List[Dict[str, Any]], condition_key: str) -> Dict[str, Any]:
    def _collect_metric(path: Tuple[str, ...]) -> List[float]:
        vals: List[float] = []
        for r in rows:
            cur: Any = r
            ok = True
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if ok and isinstance(cur, (int, float)):
                vals.append(float(cur))
        return vals

    def _mean(vals: Iterable[float]) -> Optional[float]:
        arr = list(vals)
        if not arr:
            return None
        return float(sum(arr) / float(len(arr)))

    return {
        "mean_eval_return": _mean(_collect_metric(("target_conditions", condition_key, "eval_summary", "mean_return"))),
        "mean_eval_success_rate": _mean(_collect_metric(("target_conditions", condition_key, "eval_summary", "success_rate"))),
        "mean_eval_hazard_per_1k": _mean(_collect_metric(("target_conditions", condition_key, "eval_summary", "hazard_per_1k"))),
        "mean_early_success_rate": _mean(_collect_metric(("target_conditions", condition_key, "early_train_summary", "success_rate"))),
        "mean_early_return": _mean(_collect_metric(("target_conditions", condition_key, "early_train_summary", "mean_return"))),
        "mean_early_hazard_per_1k": _mean(_collect_metric(("target_conditions", condition_key, "early_train_summary", "hazard_per_1k"))),
        "mean_early_forward_mse": _mean(_collect_metric(("target_conditions", condition_key, "early_train_summary", "mean_forward_mse"))),
        "mean_eval_forward_mse": _mean(_collect_metric(("target_conditions", condition_key, "eval_summary", "mean_forward_mse"))),
    }


def _summarize_adaptive_triage(
    rows: List[Dict[str, Any]],
    *,
    adaptive_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    chosen_metrics: List[Dict[str, Optional[float]]] = []
    tri_rows: List[Dict[str, Any]] = []
    counts = {"sf_scratch": 0, "sf_transfer": 0, "sf_transfer_warmup": 0}
    for r in rows:
        tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
        if not isinstance(tc, dict):
            continue
        tri = _adaptive_triage_decision(r, adaptive_cfg)
        tri_rows.append(tri)
        sel = str(tri.get("selected_condition", "sf_scratch"))
        if sel not in counts:
            sel = "sf_scratch"
        counts[sel] += 1
        chosen = tc.get(sel, {}) if isinstance(tc.get(sel), dict) else {}
        if not isinstance(chosen, dict):
            chosen = {}
        chosen_metrics.append(_seed_condition_summary(chosen))

    def _mean_key(k: str) -> Optional[float]:
        vals = [float(m[k]) for m in chosen_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            return None
        return float(sum(vals) / float(len(vals)))

    summary = {
        "mean_eval_return": _mean_key("eval_return"),
        "mean_eval_success_rate": _mean_key("eval_success_rate"),
        "mean_eval_hazard_per_1k": _mean_key("eval_hazard_per_1k"),
        "mean_early_success_rate": _mean_key("early_success_rate"),
        "mean_early_return": _mean_key("early_return"),
        "mean_early_hazard_per_1k": _mean_key("early_hazard_per_1k"),
        "mean_early_forward_mse": _mean_key("early_forward_mse"),
        "mean_eval_forward_mse": _mean_key("eval_forward_mse"),
    }
    n_eval = max(1, len(tri_rows))
    gate_info = {
        "enabled": True,
        "kind": "learned_gate_triad_selector",
        "num_seeds": int(len(rows)),
        "evaluated_seeds": int(len(tri_rows)),
        "selected_counts": {k: int(v) for k, v in counts.items()},
        "accept_transfer_count": int(counts["sf_transfer"] + counts["sf_transfer_warmup"]),
        "fallback_to_scratch_count": int(counts["sf_scratch"]),
        "accept_transfer_rate": float((counts["sf_transfer"] + counts["sf_transfer_warmup"]) / float(n_eval)),
        "full_transfer_rate": float(counts["sf_transfer"] / float(n_eval)),
        "warmup_transfer_rate": float(counts["sf_transfer_warmup"] / float(n_eval)),
    }
    return summary, gate_info


def _summarize_adaptive_condition(
    rows: List[Dict[str, Any]],
    *,
    transfer_condition_key: str,
    adaptive_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    chosen_metrics: List[Dict[str, Optional[float]]] = []
    accept_count = 0
    fallback_count = 0
    gate_rows: List[Dict[str, Any]] = []
    for r in rows:
        tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
        scratch = tc.get("sf_scratch", {}) if isinstance(tc.get("sf_scratch"), dict) else {}
        transfer = tc.get(transfer_condition_key, {}) if isinstance(tc.get(transfer_condition_key), dict) else {}
        if not isinstance(scratch, dict) or not isinstance(transfer, dict):
            continue
        decision = _adaptive_gate_decision(
            r,
            adaptive_cfg,
            transfer_gate_condition_key_override=transfer_condition_key,
        )
        gate_rows.append(decision)
        if bool(decision.get("accept_transfer", False)):
            chosen = transfer
            accept_count += 1
        else:
            chosen = scratch
            fallback_count += 1
        chosen_metrics.append(_seed_condition_summary(chosen))

    def _mean_key(k: str) -> Optional[float]:
        vals = [float(m[k]) for m in chosen_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            return None
        return float(sum(vals) / float(len(vals)))

    summary = {
        "mean_eval_return": _mean_key("eval_return"),
        "mean_eval_success_rate": _mean_key("eval_success_rate"),
        "mean_eval_hazard_per_1k": _mean_key("eval_hazard_per_1k"),
        "mean_early_success_rate": _mean_key("early_success_rate"),
        "mean_early_return": _mean_key("early_return"),
        "mean_early_hazard_per_1k": _mean_key("early_hazard_per_1k"),
        "mean_early_forward_mse": _mean_key("early_forward_mse"),
        "mean_eval_forward_mse": _mean_key("eval_forward_mse"),
    }

    gate_info = {
        "enabled": bool(adaptive_cfg.get("enabled", False)),
        "kind": str(adaptive_cfg.get("kind", "hybrid_hardness_quality_gate")),
        "logic": str(adaptive_cfg.get("logic", "any")),
        "transfer_gate_condition_key": str(transfer_condition_key),
        "thresholds": dict(adaptive_cfg.get("thresholds", {})) if isinstance(adaptive_cfg.get("thresholds"), dict) else {},
        "num_seeds": int(len(rows)),
        "evaluated_seeds": int(len(gate_rows)),
        "accept_transfer_count": int(accept_count),
        "fallback_to_scratch_count": int(fallback_count),
        "accept_transfer_rate": (
            float(accept_count / float(max(1, accept_count + fallback_count)))
            if (accept_count + fallback_count) > 0
            else 0.0
        ),
        "mean_scratch_early_success_rate": (
            float(sum(_safe_float((g.get("scratch_early") or {}).get("success_rate", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_early_return": (
            float(sum(_safe_float((g.get("scratch_early") or {}).get("mean_return", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_early_hazard_per_1k": (
            float(sum(_safe_float((g.get("scratch_early") or {}).get("hazard_per_1k", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_eval_success_rate": (
            float(sum(_safe_float((g.get("scratch_eval") or {}).get("success_rate", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_eval_return": (
            float(sum(_safe_float((g.get("scratch_eval") or {}).get("mean_return", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_eval_hazard_per_1k": (
            float(sum(_safe_float((g.get("scratch_eval") or {}).get("hazard_per_1k", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_transfer_early_success_rate": (
            float(sum(_safe_float((g.get("transfer_early") or {}).get("success_rate", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_transfer_early_return": (
            float(sum(_safe_float((g.get("transfer_early") or {}).get("mean_return", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_transfer_early_hazard_per_1k": (
            float(sum(_safe_float((g.get("transfer_early") or {}).get("hazard_per_1k", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_transfer_early_forward_mse": (
            float(sum(_safe_float((g.get("transfer_early") or {}).get("mean_forward_mse", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_canary_early_delta_success_rate": (
            float(sum(_safe_float((g.get("canary_early_deltas") or {}).get("transfer_minus_scratch_success_rate", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_canary_early_delta_return": (
            float(sum(_safe_float((g.get("canary_early_deltas") or {}).get("transfer_minus_scratch_mean_return", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_canary_early_hazard_gain_per_1k": (
            float(sum(_safe_float((g.get("canary_early_deltas") or {}).get("scratch_minus_transfer_hazard_per_1k", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_canary_early_forward_mse_gain": (
            float(sum(_safe_float((g.get("canary_early_deltas") or {}).get("scratch_minus_transfer_forward_mse", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
    }
    return summary, gate_info


def _aggregate_seed_block(rows: List[Dict[str, Any]], adaptive_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    def _collect(path: Tuple[str, ...]) -> List[float]:
        vals: List[float] = []
        for r in rows:
            cur: Any = r
            ok = True
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if ok and isinstance(cur, (int, float)):
                vals.append(float(cur))
        return vals

    def _mean(vals: Iterable[float]) -> Optional[float]:
        arr = list(vals)
        if not arr:
            return None
        return float(sum(arr) / float(len(arr)))

    out: Dict[str, Any] = {"num_seeds": int(len(rows))}
    for cond in ("sf_scratch", "sf_transfer", "sf_transfer_warmup"):
        out[cond] = _summarize_condition_from_rows(rows, cond)

    if isinstance(adaptive_cfg, dict) and bool(adaptive_cfg.get("enabled", False)):
        adap_full, gate_info_full = _summarize_adaptive_condition(
            rows,
            transfer_condition_key="sf_transfer",
            adaptive_cfg=adaptive_cfg,
        )
        adap_warm, gate_info_warm = _summarize_adaptive_condition(
            rows,
            transfer_condition_key="sf_transfer_warmup",
            adaptive_cfg=adaptive_cfg,
        )
        out["sf_adaptive_transfer"] = adap_full
        out["sf_adaptive_transfer_warmup"] = adap_warm
        out["adaptive_gate"] = {
            "sf_transfer": gate_info_full,
            "sf_transfer_warmup": gate_info_warm,
        }
        if _adaptive_gate_triage_enabled(adaptive_cfg):
            adap_triage, gate_info_triage = _summarize_adaptive_triage(rows, adaptive_cfg=adaptive_cfg)
            out["sf_adaptive_triad"] = adap_triage
            out["adaptive_gate"]["triage"] = gate_info_triage

    def _success_harm_rate(*, transfer_key: str) -> Optional[float]:
        harms = 0
        total = 0
        for r in rows:
            tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
            s_eval = (((tc.get("sf_scratch") or {}) if isinstance(tc.get("sf_scratch"), dict) else {}).get("eval_summary") or {})
            t_eval = (((tc.get(transfer_key) or {}) if isinstance(tc.get(transfer_key), dict) else {}).get("eval_summary") or {})
            if not isinstance(s_eval, dict) or not isinstance(t_eval, dict):
                continue
            s_sr = s_eval.get("success_rate")
            t_sr = t_eval.get("success_rate")
            if not isinstance(s_sr, (int, float)) or not isinstance(t_sr, (int, float)):
                continue
            total += 1
            if float(t_sr) < float(s_sr):
                harms += 1
        if total <= 0:
            return None
        return float(harms / float(total))

    def _adaptive_success_harm_rate(*, transfer_key: str) -> Optional[float]:
        if not (isinstance(adaptive_cfg, dict) and bool(adaptive_cfg.get("enabled", False))):
            return None
        harms = 0
        total = 0
        for r in rows:
            tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
            s_eval = (((tc.get("sf_scratch") or {}) if isinstance(tc.get("sf_scratch"), dict) else {}).get("eval_summary") or {})
            t_eval = (((tc.get(transfer_key) or {}) if isinstance(tc.get(transfer_key), dict) else {}).get("eval_summary") or {})
            if not isinstance(s_eval, dict) or not isinstance(t_eval, dict):
                continue
            s_sr = s_eval.get("success_rate")
            t_sr = t_eval.get("success_rate")
            if not isinstance(s_sr, (int, float)) or not isinstance(t_sr, (int, float)):
                continue
            d = _adaptive_gate_decision(
                r,
                adaptive_cfg,
                transfer_gate_condition_key_override=transfer_key,
            )
            chosen_sr = float(t_sr) if bool(d.get("accept_transfer", False)) else float(s_sr)
            total += 1
            if chosen_sr < float(s_sr):
                harms += 1
        if total <= 0:
            return None
        return float(harms / float(total))

    def _adaptive_triage_success_harm_rate() -> Optional[float]:
        if not (isinstance(adaptive_cfg, dict) and bool(adaptive_cfg.get("enabled", False)) and _adaptive_gate_triage_enabled(adaptive_cfg)):
            return None
        harms = 0
        total = 0
        for r in rows:
            tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
            s_eval = (((tc.get("sf_scratch") or {}) if isinstance(tc.get("sf_scratch"), dict) else {}).get("eval_summary") or {})
            if not isinstance(s_eval, dict):
                continue
            s_sr = s_eval.get("success_rate")
            if not isinstance(s_sr, (int, float)):
                continue
            tri = _adaptive_triage_decision(r, adaptive_cfg)
            sel = str(tri.get("selected_condition", "sf_scratch"))
            t_eval = (((tc.get(sel) or {}) if isinstance(tc.get(sel), dict) else {}).get("eval_summary") or {})
            t_sr = t_eval.get("success_rate")
            if not isinstance(t_sr, (int, float)):
                continue
            total += 1
            if float(t_sr) < float(s_sr):
                harms += 1
        if total <= 0:
            return None
        return float(harms / float(total))

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(a - b)

    transfer_ret = out["sf_transfer"]["mean_eval_return"]
    scratch_ret = out["sf_scratch"]["mean_eval_return"]
    transfer_warmup_ret = out["sf_transfer_warmup"]["mean_eval_return"]
    transfer_haz = out["sf_transfer"]["mean_eval_hazard_per_1k"]
    scratch_haz = out["sf_scratch"]["mean_eval_hazard_per_1k"]
    transfer_warmup_haz = out["sf_transfer_warmup"]["mean_eval_hazard_per_1k"]
    transfer_sr = out["sf_transfer"]["mean_eval_success_rate"]
    scratch_sr = out["sf_scratch"]["mean_eval_success_rate"]
    transfer_warmup_sr = out["sf_transfer_warmup"]["mean_eval_success_rate"]

    out["comparison"] = {
        "transfer_minus_scratch_eval_return": _delta(transfer_ret, scratch_ret),
        "transfer_warmup_minus_scratch_eval_return": _delta(transfer_warmup_ret, scratch_ret),
        "transfer_minus_scratch_eval_success_rate": _delta(transfer_sr, scratch_sr),
        "transfer_warmup_minus_scratch_eval_success_rate": _delta(transfer_warmup_sr, scratch_sr),
        "scratch_minus_transfer_eval_hazard_per_1k": _delta(scratch_haz, transfer_haz),
        "scratch_minus_transfer_warmup_eval_hazard_per_1k": _delta(scratch_haz, transfer_warmup_haz),
        "negative_transfer_rate_success_sf_transfer": _success_harm_rate(transfer_key="sf_transfer"),
        "negative_transfer_rate_success_sf_transfer_warmup": _success_harm_rate(transfer_key="sf_transfer_warmup"),
    }
    if "sf_adaptive_transfer" in out:
        adap_ret = out["sf_adaptive_transfer"]["mean_eval_return"]
        adap_haz = out["sf_adaptive_transfer"]["mean_eval_hazard_per_1k"]
        adap_sr = out["sf_adaptive_transfer"]["mean_eval_success_rate"]
        adap_warm_ret = out["sf_adaptive_transfer_warmup"]["mean_eval_return"]
        adap_warm_haz = out["sf_adaptive_transfer_warmup"]["mean_eval_hazard_per_1k"]
        adap_warm_sr = out["sf_adaptive_transfer_warmup"]["mean_eval_success_rate"]
        out["comparison"].update(
            {
                "adaptive_transfer_minus_scratch_eval_return": _delta(adap_ret, scratch_ret),
                "adaptive_transfer_warmup_minus_scratch_eval_return": _delta(adap_warm_ret, scratch_ret),
                "adaptive_transfer_minus_scratch_eval_success_rate": _delta(adap_sr, scratch_sr),
                "adaptive_transfer_warmup_minus_scratch_eval_success_rate": _delta(adap_warm_sr, scratch_sr),
                "scratch_minus_adaptive_transfer_eval_hazard_per_1k": _delta(scratch_haz, adap_haz),
                "scratch_minus_adaptive_transfer_warmup_eval_hazard_per_1k": _delta(scratch_haz, adap_warm_haz),
                "negative_transfer_rate_success_sf_adaptive_transfer": _adaptive_success_harm_rate(
                    transfer_key="sf_transfer"
                ),
                "negative_transfer_rate_success_sf_adaptive_transfer_warmup": _adaptive_success_harm_rate(
                    transfer_key="sf_transfer_warmup"
                ),
            }
        )
    if "sf_adaptive_triad" in out:
        adap_tri_ret = out["sf_adaptive_triad"]["mean_eval_return"]
        adap_tri_haz = out["sf_adaptive_triad"]["mean_eval_hazard_per_1k"]
        adap_tri_sr = out["sf_adaptive_triad"]["mean_eval_success_rate"]
        out["comparison"].update(
            {
                "adaptive_triad_minus_scratch_eval_return": _delta(adap_tri_ret, scratch_ret),
                "adaptive_triad_minus_scratch_eval_success_rate": _delta(adap_tri_sr, scratch_sr),
                "scratch_minus_adaptive_triad_eval_hazard_per_1k": _delta(scratch_haz, adap_tri_haz),
                "negative_transfer_rate_success_sf_adaptive_triad": _adaptive_triage_success_harm_rate(),
            }
        )
    return out


def _profile_params(*, profile: str, max_steps: int) -> Dict[str, Any]:
    p = str(profile).strip().lower()
    if p == "near_transfer":
        return {
            "profile": p,
            "description": "Navigation-isolated warehouse profile (near-transfer).",
            "source_verse_name": "grid_world",
            "target_verse_name": "warehouse_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 32,
            "source_policy_snapshots": 3,
            "source_params": {
                "width": 8,
                "height": 8,
                "max_steps": int(max_steps),
                "obstacle_count": 10,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.02,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 8,
                "height": 8,
                "max_steps": int(max_steps),
                "obstacle_count": 14,
                "patrol_robot": False,
                "conveyor_count": 0,
                "battery_drain": 0,
                "lidar_range": 4,
                "step_penalty": -0.10,
                "adr_enabled": False,
            },
        }
    if p == "default_like":
        return {
            "profile": p,
            "description": "Warehouse default-like profile with patrol, conveyor, and battery costs.",
            "source_verse_name": "grid_world",
            "target_verse_name": "warehouse_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 32,
            "source_policy_snapshots": 3,
            "source_params": {
                "width": 8,
                "height": 8,
                "max_steps": int(max_steps),
                "obstacle_count": 12,
                "ice_count": 2,
                "teleporter_pairs": 1,
                "step_penalty": -0.02,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 8,
                "height": 8,
                "max_steps": int(max_steps),
                "obstacle_count": 14,
                "patrol_robot": True,
                "conveyor_count": 3,
                "battery_drain": 1,
                "lidar_range": 4,
                "step_penalty": -0.10,
                "adr_enabled": False,
            },
        }
    if p == "maze_near_transfer":
        return {
            "profile": p,
            "description": "Task-solving transfer curriculum stage: grid_world -> maze_world (5x5, no hazards).",
            "source_verse_name": "grid_world",
            "target_verse_name": "maze_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 24,
            "source_policy_snapshots": 3,
            "source_params": {
                "width": 5,
                "height": 5,
                "max_steps": int(max_steps),
                "obstacle_count": 3,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.01,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 5,
                "height": 5,
                "max_steps": int(max_steps),
                "step_penalty": -0.005,
                "bump_penalty": -0.02,
                "explore_bonus": 0.0,
                "hazard_count": 0,
                "fog_of_war": False,
                "adr_enabled": False,
            },
        }
    if p == "grid_same_transfer":
        return {
            "profile": p,
            "description": "Rung-1 task-solving transfer: grid_world -> harder held-out grid_world.",
            "source_verse_name": "grid_world",
            "target_verse_name": "grid_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 20,
            "source_policy_snapshots": 4,
            "source_params": {
                "width": 6,
                "height": 6,
                "max_steps": int(max_steps),
                "obstacle_count": 3,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.01,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 6,
                "height": 6,
                "max_steps": int(max_steps),
                "obstacle_count": 4,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.01,
                "adr_enabled": False,
            },
        }
    if p == "maze_to_grid_near_transfer":
        return {
            "profile": p,
            "description": "Same-universe reverse curriculum stage: maze_world -> grid_world (5x5 core navigation).",
            "source_verse_name": "maze_world",
            "target_verse_name": "grid_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 24,
            "source_policy_snapshots": 3,
            "source_params": {
                "width": 5,
                "height": 5,
                "max_steps": int(max_steps),
                "step_penalty": -0.005,
                "bump_penalty": -0.02,
                "explore_bonus": 0.0,
                "hazard_count": 0,
                "fog_of_war": False,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 5,
                "height": 5,
                "max_steps": int(max_steps),
                "obstacle_count": 3,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.01,
                "adr_enabled": False,
            },
        }
    raise ValueError(
        "Unknown profile: "
        f"{profile}. Expected near_transfer, default_like, maze_near_transfer, grid_same_transfer, or maze_to_grid_near_transfer."
    )


def _phase2_score(summary: Dict[str, Any]) -> float:
    cmp = summary.get("comparison", {}) if isinstance(summary.get("comparison"), dict) else {}
    ret_delta = _safe_float(cmp.get("transfer_warmup_minus_scratch_eval_return", 0.0), 0.0)
    haz_gain = _safe_float(cmp.get("scratch_minus_transfer_warmup_eval_hazard_per_1k", 0.0), 0.0)
    sr_delta = _safe_float(cmp.get("transfer_warmup_minus_scratch_eval_success_rate", 0.0), 0.0)
    # Return remains primary; hazard and success are tie-break style bonuses.
    return float(ret_delta + 0.02 * haz_gain + 20.0 * sr_delta)


def _transfer_pair_policy_check(
    *,
    source_verse_name: str,
    target_verse_name: str,
    disable_universe_policy: bool,
    allow_adjacent_universe_transfer: bool,
    allow_cross_universe_transfer: bool,
) -> Dict[str, Any]:
    src = str(source_verse_name or "").strip().lower()
    tgt = str(target_verse_name or "").strip().lower()
    relation = "unknown"
    try:
        from core.taxonomy import universe_relation

        relation = str(universe_relation(src, tgt))
    except Exception:
        relation = "unknown"

    if bool(disable_universe_policy):
        return {
            "enabled": False,
            "relation": relation,
            "allowed": True,
            "reason": "universe_policy_disabled",
        }

    allowed = False
    reason = "blocked_unknown_relation"
    if relation.startswith("same_universe:"):
        allowed = True
        reason = "same_universe_allowed"
    elif relation.startswith("adjacent_universe:"):
        allowed = bool(allow_adjacent_universe_transfer)
        reason = "adjacent_universe_override" if allowed else "adjacent_universe_blocked"
    elif relation.startswith("cross_universe:"):
        allowed = bool(allow_cross_universe_transfer)
        reason = "cross_universe_override" if allowed else "cross_universe_blocked"

    return {
        "enabled": True,
        "relation": relation,
        "allowed": bool(allowed),
        "reason": reason,
        "allow_adjacent_universe_transfer": bool(allow_adjacent_universe_transfer),
        "allow_cross_universe_transfer": bool(allow_cross_universe_transfer),
    }


def _gate_check_from_comparison(
    *,
    comparison: Dict[str, Any],
    min_return_delta: float,
    min_hazard_gain: float,
    min_success_delta: float,
) -> Dict[str, Any]:
    ret = _safe_float(comparison.get("transfer_warmup_minus_scratch_eval_return", 0.0), 0.0)
    haz = _safe_float(comparison.get("scratch_minus_transfer_warmup_eval_hazard_per_1k", 0.0), 0.0)
    sr = _safe_float(comparison.get("transfer_warmup_minus_scratch_eval_success_rate", 0.0), 0.0)
    ok = bool(
        ret >= float(min_return_delta)
        and haz >= float(min_hazard_gain)
        and sr >= float(min_success_delta)
    )
    return {
        "ok": ok,
        "transfer_warmup_minus_scratch_eval_return": float(ret),
        "scratch_minus_transfer_warmup_eval_hazard_per_1k": float(haz),
        "transfer_warmup_minus_scratch_eval_success_rate": float(sr),
        "thresholds": {
            "min_return_delta": float(min_return_delta),
            "min_hazard_gain": float(min_hazard_gain),
            "min_success_delta": float(min_success_delta),
        },
    }


def _evaluate_config(
    *,
    seeds: Sequence[int],
    profile_params: Dict[str, Any],
    ego_size: int,
    source_train_episodes: int,
    target_train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    warmup_psi_episodes: int,
    target_w_estimation_steps: int = 0,
    source_policy_snapshots: int = 3,
    adaptive_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    adapter = EgoGridAdapter(size=int(ego_size))
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        row = _train_then_eval(
            seed=int(s),
            adapter=adapter,
            source_verse_name=str(profile_params.get("source_verse_name", "grid_world")),
            target_verse_name=str(profile_params.get("target_verse_name", "warehouse_world")),
            source_params=dict(profile_params["source_params"]),
            target_params=dict(profile_params["target_params"]),
            source_train_episodes=int(source_train_episodes),
            target_train_episodes=int(target_train_episodes),
            eval_episodes=int(eval_episodes),
            max_steps=int(max_steps),
            warmup_psi_episodes=int(warmup_psi_episodes),
            source_allowed_actions=profile_params.get("source_allowed_actions"),
            target_allowed_actions=profile_params.get("target_allowed_actions"),
            target_w_estimation_steps=int(
                profile_params.get("target_w_estimation_steps", target_w_estimation_steps)
            ),
            source_policy_snapshots=int(
                profile_params.get("source_policy_snapshots", source_policy_snapshots)
            ),
        )
        if isinstance(adaptive_gate, dict) and bool(adaptive_gate.get("enabled", False)):
            row["adaptive_transfer_decision_records"] = [
                _sf_transfer_decision_record(
                    row=row,
                    adaptive_cfg=adaptive_gate,
                    transfer_condition_key="sf_transfer",
                ),
                _sf_transfer_decision_record(
                    row=row,
                    adaptive_cfg=adaptive_gate,
                    transfer_condition_key="sf_transfer_warmup",
                ),
            ]
            if _adaptive_gate_triage_enabled(adaptive_gate):
                row["adaptive_transfer_triage_decision_record"] = _sf_transfer_triage_decision_record(
                    row=row,
                    adaptive_cfg=adaptive_gate,
                )
        per_seed.append(row)

    summary = _aggregate_seed_block(per_seed, adaptive_cfg=adaptive_gate)
    out = {
        "config": {
            "source_verse_name": str(profile_params.get("source_verse_name", "grid_world")),
            "target_verse_name": str(profile_params.get("target_verse_name", "warehouse_world")),
            "ego_size": int(ego_size),
            "source_train_episodes": int(source_train_episodes),
            "target_train_episodes": int(target_train_episodes),
            "eval_episodes": int(eval_episodes),
            "warmup_psi_episodes": int(warmup_psi_episodes),
            "max_steps": int(max_steps),
            "target_w_estimation_steps": int(
                profile_params.get("target_w_estimation_steps", target_w_estimation_steps)
            ),
            "source_policy_snapshots": int(
                profile_params.get("source_policy_snapshots", source_policy_snapshots)
            ),
        },
        "summary": summary,
        "score": _phase2_score(summary),
        "per_seed": per_seed,
    }
    if isinstance(adaptive_gate, dict) and bool(adaptive_gate.get("enabled", False)):
        out["adaptive_gate"] = dict(adaptive_gate)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="phase2", choices=["single", "phase2"])
    ap.add_argument("--seeds", type=str, default="123,223,337")
    ap.add_argument("--profiles", type=str, default="near_transfer,default_like")
    ap.add_argument("--ego_size", type=int, default=5)
    ap.add_argument("--warmup_psi_episodes", type=int, default=8)
    ap.add_argument("--source_train_episodes", type=int, default=120)
    ap.add_argument("--target_train_episodes", type=int, default=80)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--target_w_estimation_steps", type=int, default=32)
    ap.add_argument("--source_policy_snapshots", type=int, default=3)
    ap.add_argument("--sweep_ego_sizes", type=str, default="3,5,7")
    ap.add_argument("--sweep_source_episodes", type=str, default="80,120")
    ap.add_argument("--sweep_warmup_episodes", type=str, default="0,8,16")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--strict_gate", action="store_true")
    ap.add_argument("--gate_min_return_delta", type=float, default=0.0)
    ap.add_argument("--gate_min_hazard_gain", type=float, default=0.0)
    ap.add_argument("--gate_min_success_delta", type=float, default=0.0)
    ap.add_argument("--adaptive_gate_enabled", action="store_true")
    ap.add_argument("--adaptive_gate_logic", type=str, default="any", choices=["any", "all"])
    ap.add_argument("--adaptive_gate_scratch_early_success_max", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_scratch_early_return_max", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_scratch_early_hazard_min", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_scratch_eval_success_max", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_scratch_eval_return_max", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_scratch_eval_hazard_min", type=float, default=-1.0)
    ap.add_argument(
        "--adaptive_gate_transfer_condition",
        type=str,
        default="sf_transfer_warmup",
        choices=["sf_transfer", "sf_transfer_warmup"],
    )
    ap.add_argument("--adaptive_gate_transfer_early_success_min", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_transfer_early_return_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_early_hazard_max", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_transfer_early_forward_mse_max", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_transfer_minus_scratch_early_success_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_minus_scratch_early_return_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_scratch_minus_transfer_early_hazard_max", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_scratch_minus_transfer_early_forward_mse_max", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_early_return_slope_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_early_success_slope_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_early_hazard_slope_max", type=float, default=1e9)
    ap.add_argument("--adaptive_gate_transfer_early_forward_mse_slope_max", type=float, default=1e9)
    ap.add_argument("--adaptive_gate_canary_delta_return_slope_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_canary_delta_success_slope_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_canary_delta_hazard_slope_max", type=float, default=1e9)
    ap.add_argument("--adaptive_gate_canary_delta_forward_mse_slope_max", type=float, default=1e9)
    ap.add_argument("--adaptive_gate_source_policy_bank_majority_min", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_json", type=str, default="")
    ap.add_argument("--adaptive_gate_model_accept_prob", type=float, default=0.5)
    ap.add_argument("--adaptive_gate_model_warmup_prob", type=float, default=0.35)
    ap.add_argument("--adaptive_gate_model_hard_prob", type=float, default=-2.0)
    ap.add_argument("--adaptive_gate_model_triage_full_prob", type=float, default=-2.0)
    ap.add_argument("--adaptive_gate_model_triage_warmup_prob", type=float, default=-2.0)
    ap.add_argument("--adaptive_gate_model_enable_canary_triad_override", action="store_true")
    ap.add_argument("--adaptive_gate_model_canary_success_weight", type=float, default=100.0)
    ap.add_argument("--adaptive_gate_model_canary_return_weight", type=float, default=1.0)
    ap.add_argument("--adaptive_gate_model_canary_hazard_weight", type=float, default=0.02)
    ap.add_argument("--adaptive_gate_model_canary_min_utility", type=float, default=0.0)
    ap.add_argument("--adaptive_gate_model_canary_min_hazard_gain", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_min_episodes", type=float, default=1.0)
    ap.add_argument("--adaptive_gate_model_canary_two_stage_global_select", action="store_true")
    ap.add_argument("--adaptive_gate_model_canary_scratch_utility", type=float, default=0.0)
    ap.add_argument("--adaptive_gate_model_enable_canary_dual_policy", action="store_true")
    ap.add_argument("--adaptive_gate_model_canary_dual_hazard_gain_threshold", type=float, default=0.0)
    ap.add_argument("--adaptive_gate_model_canary_dual_success_delta_threshold", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_success_weight", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_return_weight", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_hazard_weight", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_min_utility", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_min_hazard_gain", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_min_episodes", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_canary_full_success_floor", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_full_hazard_gain_override", type=float, default=25.0)
    ap.add_argument("--adaptive_gate_model_enable_triage", action="store_true")
    ap.add_argument("--disable_universe_policy", action="store_true")
    ap.add_argument("--allow_adjacent_universe_transfer", action="store_true")
    ap.add_argument("--allow_cross_universe_transfer", action="store_true")
    ap.add_argument("--out_json", type=str, default="models/validation/sf_transfer_validation_v2_phase2.json")
    args = ap.parse_args()

    register_builtin()
    seeds = _parse_seed_list(args.seeds)
    profiles = _parse_str_list(args.profiles, default=["near_transfer", "default_like"])
    adaptive_gate_cfg = _adaptive_gate_cfg(
        enabled=bool(args.adaptive_gate_enabled),
        scratch_early_success_max=float(args.adaptive_gate_scratch_early_success_max),
        scratch_early_return_max=float(args.adaptive_gate_scratch_early_return_max),
        scratch_early_hazard_min=float(args.adaptive_gate_scratch_early_hazard_min),
        scratch_eval_success_max=float(args.adaptive_gate_scratch_eval_success_max),
        scratch_eval_return_max=float(args.adaptive_gate_scratch_eval_return_max),
        scratch_eval_hazard_min=float(args.adaptive_gate_scratch_eval_hazard_min),
        transfer_gate_condition_key=str(args.adaptive_gate_transfer_condition),
        transfer_early_success_min=float(args.adaptive_gate_transfer_early_success_min),
        transfer_early_return_min=float(args.adaptive_gate_transfer_early_return_min),
        transfer_early_hazard_max=float(args.adaptive_gate_transfer_early_hazard_max),
        transfer_early_forward_mse_max=float(args.adaptive_gate_transfer_early_forward_mse_max),
        transfer_minus_scratch_early_success_min=float(args.adaptive_gate_transfer_minus_scratch_early_success_min),
        transfer_minus_scratch_early_return_min=float(args.adaptive_gate_transfer_minus_scratch_early_return_min),
        scratch_minus_transfer_early_hazard_max=float(args.adaptive_gate_scratch_minus_transfer_early_hazard_max),
        scratch_minus_transfer_early_forward_mse_max=float(args.adaptive_gate_scratch_minus_transfer_early_forward_mse_max),
        transfer_early_return_slope_min=float(args.adaptive_gate_transfer_early_return_slope_min),
        transfer_early_success_slope_min=float(args.adaptive_gate_transfer_early_success_slope_min),
        transfer_early_hazard_slope_max=float(args.adaptive_gate_transfer_early_hazard_slope_max),
        transfer_early_forward_mse_slope_max=float(args.adaptive_gate_transfer_early_forward_mse_slope_max),
        canary_delta_return_slope_min=float(args.adaptive_gate_canary_delta_return_slope_min),
        canary_delta_success_slope_min=float(args.adaptive_gate_canary_delta_success_slope_min),
        canary_delta_hazard_slope_max=float(args.adaptive_gate_canary_delta_hazard_slope_max),
        canary_delta_forward_mse_slope_max=float(args.adaptive_gate_canary_delta_forward_mse_slope_max),
        source_policy_bank_majority_min=float(args.adaptive_gate_source_policy_bank_majority_min),
        logic=str(args.adaptive_gate_logic),
    )
    if str(args.adaptive_gate_model_json or "").strip():
        model_path = str(args.adaptive_gate_model_json).strip()
        with open(model_path, "r", encoding="utf-8") as f:
            model_payload = json.load(f)
        adaptive_gate_cfg = _adaptive_gate_model_cfg(
            enabled=bool(args.adaptive_gate_enabled or True),
            transfer_gate_condition_key=str(args.adaptive_gate_transfer_condition),
            model_payload=model_payload,
            accept_prob_min=float(args.adaptive_gate_model_accept_prob),
            warmup_prob_min=float(args.adaptive_gate_model_warmup_prob),
        )
        if float(args.adaptive_gate_model_hard_prob) > -1.5:
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["hard_prob_min"] = float(args.adaptive_gate_model_hard_prob)
        if float(args.adaptive_gate_model_triage_full_prob) > -1.5:
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["triage_full_accept_prob_min"] = float(args.adaptive_gate_model_triage_full_prob)
        if float(args.adaptive_gate_model_triage_warmup_prob) > -1.5:
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["triage_warmup_accept_prob_min"] = float(args.adaptive_gate_model_triage_warmup_prob)
        if bool(args.adaptive_gate_model_enable_canary_triad_override):
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["enable_canary_triad_override"] = True
            adaptive_gate_cfg["model_policy"]["canary_success_weight"] = float(args.adaptive_gate_model_canary_success_weight)
            adaptive_gate_cfg["model_policy"]["canary_return_weight"] = float(args.adaptive_gate_model_canary_return_weight)
            adaptive_gate_cfg["model_policy"]["canary_hazard_weight"] = float(args.adaptive_gate_model_canary_hazard_weight)
            adaptive_gate_cfg["model_policy"]["canary_min_utility"] = float(args.adaptive_gate_model_canary_min_utility)
            adaptive_gate_cfg["model_policy"]["canary_min_hazard_gain"] = float(args.adaptive_gate_model_canary_min_hazard_gain)
            adaptive_gate_cfg["model_policy"]["canary_min_episodes"] = float(args.adaptive_gate_model_canary_min_episodes)
            adaptive_gate_cfg["model_policy"]["canary_two_stage_global_select"] = bool(args.adaptive_gate_model_canary_two_stage_global_select)
            adaptive_gate_cfg["model_policy"]["canary_scratch_utility"] = float(args.adaptive_gate_model_canary_scratch_utility)
            adaptive_gate_cfg["model_policy"]["enable_canary_dual_policy"] = bool(args.adaptive_gate_model_enable_canary_dual_policy)
            adaptive_gate_cfg["model_policy"]["canary_dual_select_hazard_gain_threshold"] = float(args.adaptive_gate_model_canary_dual_hazard_gain_threshold)
            adaptive_gate_cfg["model_policy"]["canary_dual_select_success_delta_threshold"] = float(args.adaptive_gate_model_canary_dual_success_delta_threshold)
            adaptive_gate_cfg["model_policy"]["canary_full_success_floor"] = float(args.adaptive_gate_model_canary_full_success_floor)
            adaptive_gate_cfg["model_policy"]["canary_full_hazard_gain_override"] = float(args.adaptive_gate_model_canary_full_hazard_gain_override)
            if float(args.adaptive_gate_model_canary_safety_success_weight) > -0.5:
                adaptive_gate_cfg["model_policy"]["canary_safety_success_weight"] = float(args.adaptive_gate_model_canary_safety_success_weight)
            if float(args.adaptive_gate_model_canary_safety_return_weight) > -0.5:
                adaptive_gate_cfg["model_policy"]["canary_safety_return_weight"] = float(args.adaptive_gate_model_canary_safety_return_weight)
            if float(args.adaptive_gate_model_canary_safety_hazard_weight) > -0.5:
                adaptive_gate_cfg["model_policy"]["canary_safety_hazard_weight"] = float(args.adaptive_gate_model_canary_safety_hazard_weight)
            if float(args.adaptive_gate_model_canary_safety_min_utility) > -1e8:
                adaptive_gate_cfg["model_policy"]["canary_safety_min_utility"] = float(args.adaptive_gate_model_canary_safety_min_utility)
            if float(args.adaptive_gate_model_canary_safety_min_hazard_gain) > -1e8:
                adaptive_gate_cfg["model_policy"]["canary_safety_min_hazard_gain"] = float(args.adaptive_gate_model_canary_safety_min_hazard_gain)
            if float(args.adaptive_gate_model_canary_safety_min_episodes) > -0.5:
                adaptive_gate_cfg["model_policy"]["canary_safety_min_episodes"] = float(args.adaptive_gate_model_canary_safety_min_episodes)
        if bool(args.adaptive_gate_model_enable_triage):
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["enable_triage"] = True

    if args.mode == "single":
        if len(profiles) != 1:
            raise ValueError("Single mode requires exactly one profile in --profiles.")
        prof = _profile_params(profile=profiles[0], max_steps=int(args.max_steps))
        pair_policy = _transfer_pair_policy_check(
            source_verse_name=str(prof.get("source_verse_name", "")),
            target_verse_name=str(prof.get("target_verse_name", "")),
            disable_universe_policy=bool(args.disable_universe_policy),
            allow_adjacent_universe_transfer=bool(args.allow_adjacent_universe_transfer),
            allow_cross_universe_transfer=bool(args.allow_cross_universe_transfer),
        )
        if not bool(pair_policy.get("allowed", False)):
            raise ValueError(
                "SF competence transfer pair blocked by universe policy: "
                f"{prof.get('source_verse_name')} -> {prof.get('target_verse_name')} "
                f"({pair_policy.get('relation')}; {pair_policy.get('reason')}). "
                "Use --allow_adjacent_universe_transfer / --allow_cross_universe_transfer, "
                "or --disable_universe_policy to override."
            )
        run = _evaluate_config(
            seeds=seeds,
            profile_params=prof,
            ego_size=int(args.ego_size),
            source_train_episodes=int(args.source_train_episodes),
            target_train_episodes=int(args.target_train_episodes),
            eval_episodes=int(args.eval_episodes),
            max_steps=int(args.max_steps),
            warmup_psi_episodes=int(args.warmup_psi_episodes),
            target_w_estimation_steps=int(args.target_w_estimation_steps),
            source_policy_snapshots=int(args.source_policy_snapshots),
            adaptive_gate=adaptive_gate_cfg,
        )
        artifact = {
            "experiment": "sf_transfer_validation_v2_single",
            "notes": [
                "Ego-grid interface supports grid_world (global-map slice), warehouse_world (lidar approximation), and maze_world (wall-sensor local occupancy).",
                "Successor Features transfer: psi copied from source pretraining; target learns reward weights w.",
                "Auxiliary dynamics objective: one-step next-feature prediction (forward model) is always trained.",
            ],
            "config": {
                "mode": "single",
                "seeds": seeds,
                "profile": profiles[0],
                "profile_description": prof["description"],
                "source_verse_name": prof.get("source_verse_name"),
                "target_verse_name": prof.get("target_verse_name"),
                "run_config": run["config"],
                "source_params": prof["source_params"],
                "target_params": prof["target_params"],
                "transfer_pair_policy": pair_policy,
                "adaptive_gate": adaptive_gate_cfg,
            },
            "summary": run["summary"],
            "per_seed": run["per_seed"],
            "score": run["score"],
        }
        if bool(args.strict_gate):
            gate = _gate_check_from_comparison(
                comparison=artifact["summary"].get("comparison", {}),
                min_return_delta=float(args.gate_min_return_delta),
                min_hazard_gain=float(args.gate_min_hazard_gain),
                min_success_delta=float(args.gate_min_success_delta),
            )
            artifact["strict_gate"] = gate
            if not bool(gate.get("ok", False)):
                print("[gate] FAIL(single):", json.dumps(gate, ensure_ascii=False))
            else:
                print("[gate] PASS(single):", json.dumps(gate, ensure_ascii=False))
        print(json.dumps(artifact["summary"], ensure_ascii=False, indent=2))
    else:
        ego_grid = _parse_int_grid(args.sweep_ego_sizes, default=[3, 5, 7])
        source_grid = _parse_int_grid(args.sweep_source_episodes, default=[80, 120])
        warmup_grid = _parse_int_grid(args.sweep_warmup_episodes, default=[0, 8, 16])

        profile_results: Dict[str, Any] = {}
        for p in profiles:
            prof = _profile_params(profile=p, max_steps=int(args.max_steps))
            pair_policy = _transfer_pair_policy_check(
                source_verse_name=str(prof.get("source_verse_name", "")),
                target_verse_name=str(prof.get("target_verse_name", "")),
                disable_universe_policy=bool(args.disable_universe_policy),
                allow_adjacent_universe_transfer=bool(args.allow_adjacent_universe_transfer),
                allow_cross_universe_transfer=bool(args.allow_cross_universe_transfer),
            )
            if not bool(pair_policy.get("allowed", False)):
                raise ValueError(
                    "SF competence transfer pair blocked by universe policy: "
                    f"{prof.get('source_verse_name')} -> {prof.get('target_verse_name')} "
                    f"({pair_policy.get('relation')}; {pair_policy.get('reason')}). "
                    "Use --allow_adjacent_universe_transfer / --allow_cross_universe_transfer, "
                    "or --disable_universe_policy to override."
                )
            sweep_rows: List[Dict[str, Any]] = []
            for ego_size in ego_grid:
                for src_eps in source_grid:
                    for warmup in warmup_grid:
                        row = _evaluate_config(
                            seeds=seeds,
                            profile_params=prof,
                            ego_size=int(ego_size),
                            source_train_episodes=int(src_eps),
                            target_train_episodes=int(args.target_train_episodes),
                            eval_episodes=int(args.eval_episodes),
                            max_steps=int(args.max_steps),
                            warmup_psi_episodes=int(warmup),
                            target_w_estimation_steps=int(args.target_w_estimation_steps),
                            source_policy_snapshots=int(args.source_policy_snapshots),
                            adaptive_gate=adaptive_gate_cfg,
                        )
                        sweep_rows.append(row)

            ranked = sorted(sweep_rows, key=lambda x: float(x.get("score", -1e9)), reverse=True)
            top_k = max(1, int(args.top_k))
            top_rows = ranked[:top_k]
            table = []
            for r in top_rows:
                cmp = r["summary"]["comparison"]
                table.append(
                    {
                        "score": float(r["score"]),
                        "ego_size": int(r["config"]["ego_size"]),
                        "source_train_episodes": int(r["config"]["source_train_episodes"]),
                        "warmup_psi_episodes": int(r["config"]["warmup_psi_episodes"]),
                        "transfer_warmup_minus_scratch_eval_return": _safe_float(
                            cmp.get("transfer_warmup_minus_scratch_eval_return", 0.0), 0.0
                        ),
                        "scratch_minus_transfer_warmup_eval_hazard_per_1k": _safe_float(
                            cmp.get("scratch_minus_transfer_warmup_eval_hazard_per_1k", 0.0), 0.0
                        ),
                        "transfer_warmup_minus_scratch_eval_success_rate": _safe_float(
                            cmp.get("transfer_warmup_minus_scratch_eval_success_rate", 0.0), 0.0
                        ),
                    }
                )

            profile_results[p] = {
                "profile_description": prof["description"],
                "transfer_pair_policy": pair_policy,
                "source_params": prof["source_params"],
                "target_params": prof["target_params"],
                "num_configs": len(sweep_rows),
                "best": ranked[0] if ranked else None,
                "top_table": table,
            }

        global_best: List[Dict[str, Any]] = []
        for p, block in profile_results.items():
            best = block.get("best")
            if isinstance(best, dict):
                global_best.append(
                    {
                        "profile": p,
                        "score": float(best.get("score", 0.0)),
                        "config": best.get("config", {}),
                        "comparison": (best.get("summary", {}) or {}).get("comparison", {}),
                    }
                )
        global_best = sorted(global_best, key=lambda x: float(x.get("score", -1e9)), reverse=True)

        artifact = {
            "experiment": "sf_transfer_validation_v2_phase2",
            "notes": [
                "Phase 2 adds multi-profile validation and hyperparameter sweep.",
                "Profiles can target warehouse or maze transfer pairs depending on --profiles.",
                "Sweep axes: ego_grid size, source SF pretrain episodes, transfer warmup freeze episodes.",
            ],
            "config": {
                "mode": "phase2",
                "seeds": seeds,
                "profiles": profiles,
                "sweep_ego_sizes": ego_grid,
                "sweep_source_episodes": source_grid,
                "sweep_warmup_episodes": warmup_grid,
                "target_train_episodes": int(args.target_train_episodes),
                "eval_episodes": int(args.eval_episodes),
                "max_steps": int(args.max_steps),
                "target_w_estimation_steps": int(args.target_w_estimation_steps),
                "source_policy_snapshots": int(args.source_policy_snapshots),
                "adaptive_gate": adaptive_gate_cfg,
                "universe_policy": {
                    "enabled": not bool(args.disable_universe_policy),
                    "allow_adjacent_universe_transfer": bool(args.allow_adjacent_universe_transfer),
                    "allow_cross_universe_transfer": bool(args.allow_cross_universe_transfer),
                },
                "top_k": int(args.top_k),
            },
            "phase2": {
                "by_profile": profile_results,
                "global_best": global_best,
            },
        }
        if bool(args.strict_gate):
            gate_by_profile: Dict[str, Any] = {}
            all_ok = True
            for p, block in profile_results.items():
                best = block.get("best")
                cmp = (
                    ((best.get("summary", {}) if isinstance(best, dict) else {}) or {}).get("comparison", {})
                    if isinstance(best, dict)
                    else {}
                )
                gate = _gate_check_from_comparison(
                    comparison=cmp if isinstance(cmp, dict) else {},
                    min_return_delta=float(args.gate_min_return_delta),
                    min_hazard_gain=float(args.gate_min_hazard_gain),
                    min_success_delta=float(args.gate_min_success_delta),
                )
                gate_by_profile[p] = gate
                if not bool(gate.get("ok", False)):
                    all_ok = False
            artifact["strict_gate"] = {
                "ok": bool(all_ok),
                "by_profile": gate_by_profile,
            }
            print("[gate] phase2:", json.dumps(artifact["strict_gate"], ensure_ascii=False))
            if not all_ok:
                out_path = str(args.out_json)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(artifact, f, ensure_ascii=False, indent=2)
                print(f"[ok] wrote: {out_path}")
                raise SystemExit(2)
        for p, block in profile_results.items():
            print(f"[profile={p}] top configs:")
            print(json.dumps(block.get("top_table", []), ensure_ascii=False, indent=2))

    out_path = str(args.out_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote: {out_path}")

    if bool(args.strict_gate) and args.mode == "single":
        gate = artifact.get("strict_gate", {})
        if isinstance(gate, dict) and not bool(gate.get("ok", False)):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
