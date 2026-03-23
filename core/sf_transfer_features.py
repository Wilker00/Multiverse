from __future__ import annotations

from typing import Any, Dict


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _adaptive_gate_model_features(
    row: Dict[str, Any],
    *,
    transfer_gate_condition_key: str,
) -> Dict[str, Any]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    scratch = tc.get("sf_scratch", {}) if isinstance(tc.get("sf_scratch"), dict) else {}
    transfer_cond = tc.get(str(transfer_gate_condition_key), {})
    if not isinstance(transfer_cond, dict):
        transfer_cond = {}
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
    source_bank_vote_margin = _safe_float(src_bank_agreement.get("mean_vote_margin", 0.0), 0.0)
    source_bank_unique_actions = _safe_float(src_bank_agreement.get("mean_unique_actions", 0.0), 0.0)
    features = {
        "scratch_early_success_rate": float(early_sr),
        "scratch_early_return": float(early_ret),
        "scratch_early_hazard_per_1k": float(early_haz),
        "scratch_early_forward_mse": float(scratch_early_fwd),
        "scratch_eval_success_rate": float(eval_sr),
        "scratch_eval_return": float(eval_ret),
        "scratch_eval_hazard_per_1k": float(eval_haz),
        "transfer_early_success_rate": float(transfer_early_sr),
        "transfer_early_return": float(transfer_early_ret),
        "transfer_early_hazard_per_1k": float(transfer_early_haz),
        "transfer_early_forward_mse": float(transfer_early_fwd),
        "transfer_minus_scratch_early_success_rate": float(d_early_success),
        "transfer_minus_scratch_early_return": float(d_early_return),
        "scratch_minus_transfer_early_hazard_per_1k": float(d_early_hazard_gain),
        "scratch_minus_transfer_early_forward_mse": float(d_early_forward_mse_gain),
        "transfer_early_return_slope": float(transfer_early_return_slope),
        "transfer_early_success_slope": float(transfer_early_success_slope),
        "transfer_early_hazard_slope": float(transfer_early_hazard_slope),
        "transfer_early_forward_mse_slope": float(transfer_early_forward_mse_slope),
        "canary_delta_return_slope": float(canary_return_slope),
        "canary_delta_success_slope": float(canary_success_slope),
        "canary_delta_hazard_slope": float(canary_hazard_slope),
        "canary_delta_forward_mse_slope": float(canary_forward_mse_slope),
        "source_policy_bank_majority_fraction": float(source_bank_majority),
        "source_policy_bank_vote_margin": float(source_bank_vote_margin),
        "source_policy_bank_unique_actions": float(source_bank_unique_actions),
    }
    return {
        "features": features,
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
            "mean_unique_actions": float(source_bank_unique_actions),
            "mean_vote_margin": float(source_bank_vote_margin),
            "num_snapshots": int(src_bank_agreement.get("num_snapshots", 0) or 0),
            "evaluated_probes": int(src_bank_agreement.get("evaluated_probes", 0) or 0),
        },
    }


def _adaptive_triage_direct_features(row: Dict[str, Any]) -> Dict[str, float]:
    f_full = _adaptive_gate_model_features(row, transfer_gate_condition_key="sf_transfer")
    f_warm = _adaptive_gate_model_features(row, transfer_gate_condition_key="sf_transfer_warmup")
    ff = f_full.get("features", {}) if isinstance(f_full.get("features"), dict) else {}
    fw = f_warm.get("features", {}) if isinstance(f_warm.get("features"), dict) else {}
    out: Dict[str, float] = {}
    for k, v in ff.items():
        out[f"full::{k}"] = _safe_float(v, 0.0)
    for k, v in fw.items():
        out[f"warm::{k}"] = _safe_float(v, 0.0)
    for k in sorted(set(ff.keys()) | set(fw.keys())):
        out[f"full_minus_warm::{k}"] = float(_safe_float(ff.get(k, 0.0), 0.0) - _safe_float(fw.get(k, 0.0), 0.0))
    return out
