from __future__ import annotations

import math
from typing import Any, Dict, Optional

from core.sf_transfer_features import _adaptive_gate_model_features


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _adaptive_gate_cfg(
    *,
    enabled: bool,
    scratch_early_success_max: float,
    scratch_early_return_max: float,
    scratch_early_hazard_min: float,
    scratch_eval_success_max: float,
    scratch_eval_return_max: float,
    scratch_eval_hazard_min: float,
    transfer_gate_condition_key: str,
    transfer_early_success_min: float,
    transfer_early_return_min: float,
    transfer_early_hazard_max: float,
    transfer_early_forward_mse_max: float,
    transfer_minus_scratch_early_success_min: float,
    transfer_minus_scratch_early_return_min: float,
    scratch_minus_transfer_early_hazard_max: float,
    scratch_minus_transfer_early_forward_mse_max: float,
    transfer_early_return_slope_min: float,
    transfer_early_success_slope_min: float,
    transfer_early_hazard_slope_max: float,
    transfer_early_forward_mse_slope_max: float,
    canary_delta_return_slope_min: float,
    canary_delta_success_slope_min: float,
    canary_delta_hazard_slope_max: float,
    canary_delta_forward_mse_slope_max: float,
    source_policy_bank_majority_min: float,
    logic: str,
) -> Dict[str, Any]:
    logic_n = str(logic or "any").strip().lower()
    if logic_n not in {"any", "all"}:
        logic_n = "any"
    return {
        "enabled": bool(enabled),
        "kind": "hybrid_hardness_quality_gate",
        "logic": logic_n,
        "transfer_gate_condition_key": str(transfer_gate_condition_key or "sf_transfer_warmup"),
        "thresholds": {
            "scratch_early_success_max": float(scratch_early_success_max),
            "scratch_early_return_max": float(scratch_early_return_max),
            "scratch_early_hazard_min": float(scratch_early_hazard_min),
            "scratch_eval_success_max": float(scratch_eval_success_max),
            "scratch_eval_return_max": float(scratch_eval_return_max),
            "scratch_eval_hazard_min": float(scratch_eval_hazard_min),
            "transfer_early_success_min": float(transfer_early_success_min),
            "transfer_early_return_min": float(transfer_early_return_min),
            "transfer_early_hazard_max": float(transfer_early_hazard_max),
            "transfer_early_forward_mse_max": float(transfer_early_forward_mse_max),
            "transfer_minus_scratch_early_success_min": float(transfer_minus_scratch_early_success_min),
            "transfer_minus_scratch_early_return_min": float(transfer_minus_scratch_early_return_min),
            "scratch_minus_transfer_early_hazard_max": float(scratch_minus_transfer_early_hazard_max),
            "scratch_minus_transfer_early_forward_mse_max": float(scratch_minus_transfer_early_forward_mse_max),
            "transfer_early_return_slope_min": float(transfer_early_return_slope_min),
            "transfer_early_success_slope_min": float(transfer_early_success_slope_min),
            "transfer_early_hazard_slope_max": float(transfer_early_hazard_slope_max),
            "transfer_early_forward_mse_slope_max": float(transfer_early_forward_mse_slope_max),
            "canary_delta_return_slope_min": float(canary_delta_return_slope_min),
            "canary_delta_success_slope_min": float(canary_delta_success_slope_min),
            "canary_delta_hazard_slope_max": float(canary_delta_hazard_slope_max),
            "canary_delta_forward_mse_slope_max": float(canary_delta_forward_mse_slope_max),
            "source_policy_bank_majority_min": float(source_policy_bank_majority_min),
        },
    }


def _adaptive_gate_model_cfg(
    *,
    enabled: bool,
    transfer_gate_condition_key: str,
    model_payload: Dict[str, Any],
    accept_prob_min: float,
    warmup_prob_min: float,
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "kind": "learned_gate_model",
        "logic": "probability_policy",
        "transfer_gate_condition_key": str(transfer_gate_condition_key or "sf_transfer_warmup"),
        "thresholds": {},
        "model": dict(model_payload) if isinstance(model_payload, dict) else {},
        "model_policy": {
            "accept_prob_min": float(accept_prob_min),
            "warmup_prob_min": float(warmup_prob_min),
            "enable_triage": False,
        },
    }


def _sigmoid(x: float) -> float:
    z = max(-60.0, min(60.0, float(x)))
    return float(1.0 / (1.0 + math.exp(-z)))


def _score_learned_linear_model(
    *,
    features: Dict[str, Any],
    model_block: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(model_block, dict):
        return {"probability": 0.5, "logit_score": 0.0, "feature_names": [], "feature_count": 0}
    feature_names = [str(x) for x in model_block.get("feature_names", [])]
    weights = [float(x) for x in model_block.get("weights", [])]
    bias = _safe_float(model_block.get("bias", 0.0), 0.0)
    norm = model_block.get("normalization", {}) if isinstance(model_block.get("normalization"), dict) else {}
    means = [float(x) for x in norm.get("mean", [])] if isinstance(norm.get("mean"), list) else []
    scales = [float(x) for x in norm.get("scale", [])] if isinstance(norm.get("scale"), list) else []
    z = float(bias)
    used_features: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        x = _safe_float(features.get(name, 0.0), 0.0)
        mu = means[i] if i < len(means) else 0.0
        sc = scales[i] if i < len(scales) and abs(scales[i]) > 1e-12 else 1.0
        xn = float((x - mu) / sc)
        w = weights[i] if i < len(weights) else 0.0
        z += float(w * xn)
        used_features[name] = float(x)
    return {
        "probability": float(_sigmoid(z)),
        "logit_score": float(z),
        "feature_names": feature_names,
        "feature_count": int(len(feature_names)),
        "used_features": used_features,
    }


def _adaptive_gate_model_decision(row: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool((cfg or {}).get("enabled", False))
    transfer_gate_condition_key = str(
        (cfg or {}).get("transfer_gate_condition_key", "sf_transfer_warmup") or "sf_transfer_warmup"
    )
    payload = (cfg or {}).get("model", {})
    if not isinstance(payload, dict):
        payload = {}
    condition_models = payload.get("condition_models", {})
    model_block: Dict[str, Any] = {}
    if isinstance(condition_models, dict) and isinstance(condition_models.get(transfer_gate_condition_key), dict):
        model_block = dict(condition_models.get(transfer_gate_condition_key) or {})
    elif isinstance(payload.get("model"), dict):
        model_block = dict(payload.get("model") or {})
    elif isinstance(payload.get("default_model"), dict):
        model_block = dict(payload.get("default_model") or {})
    feature_payload = _adaptive_gate_model_features(row, transfer_gate_condition_key=transfer_gate_condition_key)
    feats = feature_payload.get("features", {}) if isinstance(feature_payload.get("features"), dict) else {}

    model_gate_details: Dict[str, Any] = {}
    if not enabled:
        accept = True
        reason = "adaptive_disabled"
        prob_help = 1.0
        z = 0.0
    elif not model_block:
        accept = True
        reason = "missing_learned_model"
        prob_help = 1.0
        z = 0.0
    else:
        effect_score = _score_learned_linear_model(features=feats, model_block=model_block)
        prob_help = _safe_float(effect_score.get("probability", 0.5), 0.5)
        z = _safe_float(effect_score.get("logit_score", 0.0), 0.0)
        policy_cfg = (cfg or {}).get("model_policy", {})
        payload_policy = payload.get("policy", {}) if isinstance(payload.get("policy"), dict) else {}
        model_policy = model_block.get("policy", {}) if isinstance(model_block.get("policy"), dict) else {}
        accept_prob_min = _safe_float(
            (policy_cfg.get("accept_prob_min") if isinstance(policy_cfg, dict) else None),
            _safe_float(model_policy.get("accept_prob_min", payload_policy.get("accept_prob_min", 0.5)), 0.5),
        )
        warmup_prob_min = _safe_float(
            (policy_cfg.get("warmup_prob_min") if isinstance(policy_cfg, dict) else None),
            _safe_float(model_policy.get("warmup_prob_min", payload_policy.get("warmup_prob_min", 0.35)), 0.35),
        )
        hard_model_block = (
            model_block.get("hardness_aux_model", {})
            if isinstance(model_block.get("hardness_aux_model"), dict)
            else {}
        )
        prob_hard = None
        hard_z = None
        hard_prob_min = _safe_float(
            (policy_cfg.get("hard_prob_min") if isinstance(policy_cfg, dict) else None),
            _safe_float(model_policy.get("hard_prob_min", payload_policy.get("hard_prob_min", -1.0)), -1.0),
        )
        hardness_pass = True
        if isinstance(hard_model_block, dict) and hard_model_block:
            hard_score = _score_learned_linear_model(features=feats, model_block=hard_model_block)
            prob_hard = _safe_float(hard_score.get("probability", 0.5), 0.5)
            hard_z = _safe_float(hard_score.get("logit_score", 0.0), 0.0)
            if hard_prob_min >= 0.0:
                hardness_pass = bool(prob_hard >= hard_prob_min)
        model_gate_details = {
            "effect_model": {
                "probability_help": float(prob_help),
                "logit_score": float(z),
                "feature_names": list(effect_score.get("feature_names", [])),
                "feature_count": int(effect_score.get("feature_count", 0)),
            }
        }
        if prob_hard is not None:
            model_gate_details["hardness_model"] = {
                "probability_hard": float(prob_hard),
                "logit_score": float(hard_z if hard_z is not None else 0.0),
                "hard_prob_min": float(hard_prob_min),
                "pass": bool(hardness_pass),
                "label_definition": str(hard_model_block.get("label_definition", "scratch_eval_success_leq")),
            }
        if transfer_gate_condition_key == "sf_transfer_warmup":
            accept = bool(hardness_pass and (prob_help >= warmup_prob_min))
            reason = "learned_two_stage_warmup" if accept else "learned_two_stage_scratch"
            recommendation = "accept_transfer" if accept else "fallback_scratch"
        else:
            accept = bool(hardness_pass and (prob_help >= accept_prob_min))
            if accept:
                recommendation = "accept_transfer"
                reason = "learned_two_stage_transfer"
            elif hardness_pass and (prob_help >= warmup_prob_min):
                recommendation = "warmup_only_recommended"
                reason = "learned_two_stage_warmup_band"
            else:
                recommendation = "fallback_scratch"
                reason = "learned_two_stage_scratch"
    if "recommendation" not in locals():
        recommendation = "accept_transfer" if accept else "fallback_scratch"
    policy_cfg = (cfg or {}).get("model_policy", {})
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}
    accept_prob_min = _safe_float(policy_cfg.get("accept_prob_min", 0.5), 0.5)
    warmup_prob_min = _safe_float(policy_cfg.get("warmup_prob_min", 0.35), 0.35)
    checks: Dict[str, Optional[bool]] = {
        "model_probability_above_accept": bool(prob_help >= accept_prob_min),
        "model_probability_above_warmup": bool(prob_help >= warmup_prob_min),
    }
    if isinstance(model_gate_details.get("hardness_model"), dict):
        checks["hardness_probability_above_threshold"] = bool(
            (model_gate_details.get("hardness_model") or {}).get("pass", True)
        )
    out = {
        "enabled": bool(enabled),
        "accept_transfer": bool(accept),
        "decision_reason": str(reason),
        "logic": "probability_policy",
        "thresholds": {},
        "scratch_early": dict(feature_payload.get("scratch_early", {})),
        "scratch_eval": dict(feature_payload.get("scratch_eval", {})),
        "transfer_gate_condition_key": transfer_gate_condition_key,
        "transfer_early": dict(feature_payload.get("transfer_early", {})),
        "canary_early_deltas": dict(feature_payload.get("canary_early_deltas", {})),
        "transfer_early_trends": dict(feature_payload.get("transfer_early_trends", {})),
        "canary_early_delta_trends": dict(feature_payload.get("canary_early_delta_trends", {})),
        "source_policy_bank_agreement": dict(feature_payload.get("source_policy_bank_agreement", {})),
        "checks": checks,
        "model_gate": {
            "model_schema_version": str(payload.get("schema_version", "")),
            "model_type": str(payload.get("model_type", "logistic_linear")),
            "transfer_gate_condition_key": transfer_gate_condition_key,
            "probability_help": float(prob_help),
            "logit_score": float(z),
            "recommendation": str(recommendation),
            "policy": {
                "accept_prob_min": float(accept_prob_min),
                "warmup_prob_min": float(warmup_prob_min),
                "hard_prob_min": float(
                    _safe_float(
                        (policy_cfg.get("hard_prob_min") if isinstance(policy_cfg, dict) else None),
                        _safe_float(
                            (
                                model_block.get("policy", {})
                                if isinstance(model_block.get("policy"), dict)
                                else {}
                            ).get(
                                "hard_prob_min",
                                (
                                    payload.get("policy", {})
                                    if isinstance(payload.get("policy"), dict)
                                    else {}
                                ).get("hard_prob_min", -1.0),
                            ),
                            -1.0,
                        ),
                    )
                ),
            },
        },
    }
    if model_gate_details:
        out["model_gate"].update(model_gate_details)
    if isinstance(model_block, dict) and "feature_names" not in out["model_gate"]:
        out["model_gate"]["feature_names"] = [str(x) for x in model_block.get("feature_names", [])]
        out["model_gate"]["feature_count"] = int(len(out["model_gate"]["feature_names"]))
    return out


def _adaptive_gate_decision(
    row: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    transfer_gate_condition_key_override: Optional[str] = None,
) -> Dict[str, Any]:
    if transfer_gate_condition_key_override is not None:
        cfg_local = dict(cfg) if isinstance(cfg, dict) else {}
        cfg_local["transfer_gate_condition_key"] = str(transfer_gate_condition_key_override)
    else:
        cfg_local = cfg
    if str((cfg_local or {}).get("kind", "")).strip().lower() == "learned_gate_model":
        return _adaptive_gate_model_decision(row, cfg_local)
    enabled = bool((cfg_local or {}).get("enabled", False))
    thresholds = (
        dict((cfg_local or {}).get("thresholds", {}))
        if isinstance((cfg_local or {}).get("thresholds", {}), dict)
        else {}
    )
    logic = str((cfg_local or {}).get("logic", "any")).strip().lower()
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    scratch = tc.get("sf_scratch", {}) if isinstance(tc.get("sf_scratch"), dict) else {}
    transfer_gate_condition_key = str(
        (cfg_local or {}).get("transfer_gate_condition_key", "sf_transfer_warmup") or "sf_transfer_warmup"
    )
    transfer_cond = tc.get(transfer_gate_condition_key, {}) if isinstance(tc.get(transfer_gate_condition_key), dict) else {}
    early = scratch.get("early_train_summary", {}) if isinstance(scratch.get("early_train_summary"), dict) else {}
    eval_sum = scratch.get("eval_summary", {}) if isinstance(scratch.get("eval_summary"), dict) else {}
    transfer_early = (
        transfer_cond.get("early_train_summary", {})
        if isinstance(transfer_cond.get("early_train_summary"), dict)
        else {}
    )
    transfer_early_diag = (
        transfer_cond.get("early_train_diagnostics", {})
        if isinstance(transfer_cond.get("early_train_diagnostics"), dict)
        else {}
    )
    canary_delta = (
        (transfer_cond.get("canary_vs_scratch_early") or {}).get("diagnostics", {})
        if isinstance(transfer_cond.get("canary_vs_scratch_early"), dict)
        and isinstance((transfer_cond.get("canary_vs_scratch_early") or {}).get("diagnostics"), dict)
        else {}
    )
    src_bank_agreement = (
        row.get("source_policy_bank_agreement", {})
        if isinstance(row.get("source_policy_bank_agreement"), dict)
        else {}
    )
    early_sr = _safe_float(early.get("success_rate", 0.0), 0.0)
    early_ret = _safe_float(early.get("mean_return", 0.0), 0.0)
    early_haz = _safe_float(early.get("hazard_per_1k", 0.0), 0.0)
    eval_sr = _safe_float(eval_sum.get("success_rate", 0.0), 0.0)
    eval_ret = _safe_float(eval_sum.get("mean_return", 0.0), 0.0)
    eval_haz = _safe_float(eval_sum.get("hazard_per_1k", 0.0), 0.0)
    transfer_early_sr = _safe_float(transfer_early.get("success_rate", 0.0), 0.0)
    transfer_early_ret = _safe_float(transfer_early.get("mean_return", 0.0), 0.0)
    transfer_early_haz = _safe_float(transfer_early.get("hazard_per_1k", 0.0), 0.0)
    scratch_early_fwd = _safe_float(early.get("mean_forward_mse", 0.0), 0.0)
    transfer_early_fwd = _safe_float(transfer_early.get("mean_forward_mse", 0.0), 0.0)
    d_early_success = float(transfer_early_sr - early_sr)
    d_early_return = float(transfer_early_ret - early_ret)
    d_early_hazard_gain = float(early_haz - transfer_early_haz)
    d_early_forward_mse_gain = float(scratch_early_fwd - transfer_early_fwd)
    transfer_early_return_slope = _safe_float(transfer_early_diag.get("return_slope", 0.0), 0.0)
    transfer_early_success_slope = _safe_float(transfer_early_diag.get("success_slope", 0.0), 0.0)
    transfer_early_hazard_slope = _safe_float(transfer_early_diag.get("hazard_slope", 0.0), 0.0)
    transfer_early_forward_mse_slope = _safe_float(transfer_early_diag.get("forward_mse_slope", 0.0), 0.0)
    canary_return_slope = _safe_float(canary_delta.get("return_slope", 0.0), 0.0)
    canary_success_slope = _safe_float(canary_delta.get("success_slope", 0.0), 0.0)
    canary_hazard_slope = _safe_float(canary_delta.get("hazard_slope", 0.0), 0.0)
    canary_forward_mse_slope = _safe_float(canary_delta.get("forward_mse_slope", 0.0), 0.0)
    source_bank_majority = _safe_float(src_bank_agreement.get("mean_majority_fraction", 0.0), 0.0)

    sr_max = _safe_float(thresholds.get("scratch_early_success_max", -1.0), -1.0)
    ret_max = _safe_float(thresholds.get("scratch_early_return_max", -1e9), -1e9)
    haz_min = _safe_float(thresholds.get("scratch_early_hazard_min", -1.0), -1.0)
    eval_sr_max = _safe_float(thresholds.get("scratch_eval_success_max", -1.0), -1.0)
    eval_ret_max = _safe_float(thresholds.get("scratch_eval_return_max", -1e9), -1e9)
    eval_haz_min = _safe_float(thresholds.get("scratch_eval_hazard_min", -1.0), -1.0)
    tr_early_sr_min = _safe_float(thresholds.get("transfer_early_success_min", -1.0), -1.0)
    tr_early_ret_min = _safe_float(thresholds.get("transfer_early_return_min", -1e9), -1e9)
    tr_early_haz_max = _safe_float(thresholds.get("transfer_early_hazard_max", -1.0), -1.0)
    tr_early_fwd_max = _safe_float(thresholds.get("transfer_early_forward_mse_max", -1.0), -1.0)
    d_tr_sc_early_sr_min = _safe_float(thresholds.get("transfer_minus_scratch_early_success_min", -1e9), -1e9)
    d_tr_sc_early_ret_min = _safe_float(thresholds.get("transfer_minus_scratch_early_return_min", -1e9), -1e9)
    d_sc_tr_early_haz_max = _safe_float(thresholds.get("scratch_minus_transfer_early_hazard_max", -1e9), -1e9)
    d_sc_tr_early_fwd_max = _safe_float(
        thresholds.get("scratch_minus_transfer_early_forward_mse_max", -1e9),
        -1e9,
    )
    tr_ret_slope_min = _safe_float(thresholds.get("transfer_early_return_slope_min", -1e9), -1e9)
    tr_succ_slope_min = _safe_float(thresholds.get("transfer_early_success_slope_min", -1e9), -1e9)
    tr_haz_slope_max = _safe_float(thresholds.get("transfer_early_hazard_slope_max", 1e9), 1e9)
    tr_fwd_slope_max = _safe_float(thresholds.get("transfer_early_forward_mse_slope_max", 1e9), 1e9)
    can_ret_slope_min = _safe_float(thresholds.get("canary_delta_return_slope_min", -1e9), -1e9)
    can_succ_slope_min = _safe_float(thresholds.get("canary_delta_success_slope_min", -1e9), -1e9)
    can_haz_slope_max = _safe_float(thresholds.get("canary_delta_hazard_slope_max", 1e9), 1e9)
    can_fwd_slope_max = _safe_float(thresholds.get("canary_delta_forward_mse_slope_max", 1e9), 1e9)
    src_bank_majority_min = _safe_float(thresholds.get("source_policy_bank_majority_min", -1.0), -1.0)
    checks: Dict[str, Optional[bool]] = {
        "scratch_early_success_hard": (None if sr_max < 0.0 else bool(early_sr <= sr_max)),
        "scratch_early_return_hard": (None if ret_max <= -1e8 else bool(early_ret <= ret_max)),
        "scratch_early_hazard_hard": (None if haz_min < 0.0 else bool(early_haz >= haz_min)),
        "scratch_eval_success_hard": (None if eval_sr_max < 0.0 else bool(eval_sr <= eval_sr_max)),
        "scratch_eval_return_hard": (None if eval_ret_max <= -1e8 else bool(eval_ret <= eval_ret_max)),
        "scratch_eval_hazard_hard": (None if eval_haz_min < 0.0 else bool(eval_haz >= eval_haz_min)),
        "transfer_early_success_good": (None if tr_early_sr_min < 0.0 else bool(transfer_early_sr >= tr_early_sr_min)),
        "transfer_early_return_good": (None if tr_early_ret_min <= -1e8 else bool(transfer_early_ret >= tr_early_ret_min)),
        "transfer_early_hazard_good": (None if tr_early_haz_max < 0.0 else bool(transfer_early_haz <= tr_early_haz_max)),
        "transfer_early_forward_mse_good": (None if tr_early_fwd_max < 0.0 else bool(transfer_early_fwd <= tr_early_fwd_max)),
        "canary_delta_early_success_good": (
            None if d_tr_sc_early_sr_min <= -1e8 else bool(d_early_success >= d_tr_sc_early_sr_min)
        ),
        "canary_delta_early_return_good": (
            None if d_tr_sc_early_ret_min <= -1e8 else bool(d_early_return >= d_tr_sc_early_ret_min)
        ),
        "canary_delta_early_hazard_not_too_large": (
            None if d_sc_tr_early_haz_max <= -1e8 else bool(d_early_hazard_gain <= d_sc_tr_early_haz_max)
        ),
        "canary_delta_early_forward_mse_not_too_large": (
            None if d_sc_tr_early_fwd_max <= -1e8 else bool(d_early_forward_mse_gain <= d_sc_tr_early_fwd_max)
        ),
        "transfer_early_return_slope_good": (
            None if tr_ret_slope_min <= -1e8 else bool(transfer_early_return_slope >= tr_ret_slope_min)
        ),
        "transfer_early_success_slope_good": (
            None if tr_succ_slope_min <= -1e8 else bool(transfer_early_success_slope >= tr_succ_slope_min)
        ),
        "transfer_early_hazard_slope_good": (
            None if tr_haz_slope_max >= 1e8 else bool(transfer_early_hazard_slope <= tr_haz_slope_max)
        ),
        "transfer_early_forward_mse_slope_good": (
            None if tr_fwd_slope_max >= 1e8 else bool(transfer_early_forward_mse_slope <= tr_fwd_slope_max)
        ),
        "canary_delta_return_slope_good": (
            None if can_ret_slope_min <= -1e8 else bool(canary_return_slope >= can_ret_slope_min)
        ),
        "canary_delta_success_slope_good": (
            None if can_succ_slope_min <= -1e8 else bool(canary_success_slope >= can_succ_slope_min)
        ),
        "canary_delta_hazard_slope_good": (
            None if can_haz_slope_max >= 1e8 else bool(canary_hazard_slope <= can_haz_slope_max)
        ),
        "canary_delta_forward_mse_slope_good": (
            None if can_fwd_slope_max >= 1e8 else bool(canary_forward_mse_slope <= can_fwd_slope_max)
        ),
        "source_policy_bank_agreement_good": (
            None if src_bank_majority_min < 0.0 else bool(source_bank_majority >= src_bank_majority_min)
        ),
    }
    active = [bool(v) for v in checks.values() if isinstance(v, bool)]
    if not enabled:
        accept = True
        reason = "adaptive_disabled"
    elif not active:
        accept = True
        reason = "no_active_thresholds"
    elif logic == "all":
        accept = bool(all(active))
        reason = "all_hardness_checks"
    else:
        accept = bool(any(active))
        reason = "any_hardness_check"
    return {
        "enabled": bool(enabled),
        "accept_transfer": bool(accept),
        "decision_reason": str(reason),
        "logic": str(logic),
        "thresholds": {
            "scratch_early_success_max": float(sr_max),
            "scratch_early_return_max": float(ret_max),
            "scratch_early_hazard_min": float(haz_min),
            "scratch_eval_success_max": float(eval_sr_max),
            "scratch_eval_return_max": float(eval_ret_max),
            "scratch_eval_hazard_min": float(eval_haz_min),
            "transfer_early_success_min": float(tr_early_sr_min),
            "transfer_early_return_min": float(tr_early_ret_min),
            "transfer_early_hazard_max": float(tr_early_haz_max),
            "transfer_early_forward_mse_max": float(tr_early_fwd_max),
            "transfer_minus_scratch_early_success_min": float(d_tr_sc_early_sr_min),
            "transfer_minus_scratch_early_return_min": float(d_tr_sc_early_ret_min),
            "scratch_minus_transfer_early_hazard_max": float(d_sc_tr_early_haz_max),
            "scratch_minus_transfer_early_forward_mse_max": float(d_sc_tr_early_fwd_max),
            "transfer_early_return_slope_min": float(tr_ret_slope_min),
            "transfer_early_success_slope_min": float(tr_succ_slope_min),
            "transfer_early_hazard_slope_max": float(tr_haz_slope_max),
            "transfer_early_forward_mse_slope_max": float(tr_fwd_slope_max),
            "canary_delta_return_slope_min": float(can_ret_slope_min),
            "canary_delta_success_slope_min": float(can_succ_slope_min),
            "canary_delta_hazard_slope_max": float(can_haz_slope_max),
            "canary_delta_forward_mse_slope_max": float(can_fwd_slope_max),
            "source_policy_bank_majority_min": float(src_bank_majority_min),
        },
        "scratch_early": {
            "success_rate": float(early_sr),
            "mean_return": float(early_ret),
            "hazard_per_1k": float(early_haz),
        },
        "scratch_eval": {
            "success_rate": float(eval_sr),
            "mean_return": float(eval_ret),
            "hazard_per_1k": float(eval_haz),
        },
        "transfer_gate_condition_key": transfer_gate_condition_key,
        "transfer_early": {
            "success_rate": float(transfer_early_sr),
            "mean_return": float(transfer_early_ret),
            "hazard_per_1k": float(transfer_early_haz),
            "mean_forward_mse": float(transfer_early_fwd),
        },
        "canary_early_deltas": {
            "transfer_minus_scratch_success_rate": float(d_early_success),
            "transfer_minus_scratch_mean_return": float(d_early_return),
            "scratch_minus_transfer_hazard_per_1k": float(d_early_hazard_gain),
            "scratch_minus_transfer_forward_mse": float(d_early_forward_mse_gain),
        },
        "transfer_early_trends": {
            "return_slope": float(transfer_early_return_slope),
            "success_slope": float(transfer_early_success_slope),
            "hazard_slope": float(transfer_early_hazard_slope),
            "forward_mse_slope": float(transfer_early_forward_mse_slope),
        },
        "canary_early_delta_trends": {
            "return_slope": float(canary_return_slope),
            "success_slope": float(canary_success_slope),
            "hazard_slope": float(canary_hazard_slope),
            "forward_mse_slope": float(canary_forward_mse_slope),
        },
        "source_policy_bank_agreement": {
            "mean_majority_fraction": float(source_bank_majority),
            "mean_unique_actions": _safe_float(src_bank_agreement.get("mean_unique_actions", 0.0), 0.0),
            "mean_vote_margin": _safe_float(src_bank_agreement.get("mean_vote_margin", 0.0), 0.0),
            "num_snapshots": int(src_bank_agreement.get("num_snapshots", 0) or 0),
            "evaluated_probes": int(src_bank_agreement.get("evaluated_probes", 0) or 0),
        },
        "checks": checks,
    }
