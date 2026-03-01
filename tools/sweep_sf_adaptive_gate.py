from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_sf_transfer import _adaptive_gate_cfg, _adaptive_gate_decision


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _parse_float_list(raw: str, *, default: Sequence[float]) -> List[float]:
    if not str(raw or "").strip():
        return [float(x) for x in default]
    out: List[float] = []
    for p in str(raw).split(","):
        s = p.strip()
        if not s:
            continue
        out.append(float(s))
    return out if out else [float(x) for x in default]


def _get_cond(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    c = tc.get(key, {})
    return c if isinstance(c, dict) else {}


def _metric(cond: Dict[str, Any], phase: str, metric: str) -> Optional[float]:
    block = cond.get(phase, {}) if isinstance(cond.get(phase), dict) else {}
    v = block.get(metric)
    return None if not isinstance(v, (int, float)) else float(v)


def _mean(vals: Iterable[float]) -> Optional[float]:
    arr = list(vals)
    if not arr:
        return None
    return float(sum(arr) / float(len(arr)))


def _bootstrap_ci(values: Sequence[float], *, n: int, seed: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    arr = [float(x) for x in values]
    if not arr:
        return (None, None)
    if len(arr) == 1:
        return (arr[0], arr[0])
    rng = random.Random(int(seed))
    means: List[float] = []
    m = len(arr)
    for _ in range(max(1, int(n))):
        sample = [arr[rng.randrange(m)] for _ in range(m)]
        means.append(float(sum(sample) / float(m)))
    means.sort()
    lo_idx = max(0, min(len(means) - 1, int((alpha / 2.0) * len(means))))
    hi_idx = max(0, min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)) - 1))
    return (float(means[lo_idx]), float(means[hi_idx]))


def _evaluate_gate(
    *,
    rows: Sequence[Dict[str, Any]],
    gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    chosen_full_sr_deltas: List[float] = []
    chosen_full_ret_deltas: List[float] = []
    chosen_full_haz_gains: List[float] = []
    chosen_warm_sr_deltas: List[float] = []
    chosen_warm_ret_deltas: List[float] = []
    chosen_warm_haz_gains: List[float] = []
    accept_count = 0
    fallback_count = 0
    decisions: List[Dict[str, Any]] = []

    for row in rows:
        d = _adaptive_gate_decision(row, gate_cfg)
        decisions.append(d)
        use_transfer = bool(d.get("accept_transfer", False))
        if use_transfer:
            accept_count += 1
        else:
            fallback_count += 1

        scratch = _get_cond(row, "sf_scratch")
        full = _get_cond(row, "sf_transfer")
        warm = _get_cond(row, "sf_transfer_warmup")

        s_eval_sr = _safe_float(_metric(scratch, "eval_summary", "success_rate"), 0.0)
        s_eval_ret = _safe_float(_metric(scratch, "eval_summary", "mean_return"), 0.0)
        s_eval_h = _safe_float(_metric(scratch, "eval_summary", "hazard_per_1k"), 0.0)

        f_eval_sr = _safe_float(_metric(full, "eval_summary", "success_rate"), s_eval_sr if not use_transfer else 0.0)
        f_eval_ret = _safe_float(_metric(full, "eval_summary", "mean_return"), s_eval_ret if not use_transfer else 0.0)
        f_eval_h = _safe_float(_metric(full, "eval_summary", "hazard_per_1k"), s_eval_h if not use_transfer else 0.0)

        w_eval_sr = _safe_float(_metric(warm, "eval_summary", "success_rate"), s_eval_sr if not use_transfer else 0.0)
        w_eval_ret = _safe_float(_metric(warm, "eval_summary", "mean_return"), s_eval_ret if not use_transfer else 0.0)
        w_eval_h = _safe_float(_metric(warm, "eval_summary", "hazard_per_1k"), s_eval_h if not use_transfer else 0.0)

        if not use_transfer:
            f_eval_sr = s_eval_sr
            f_eval_ret = s_eval_ret
            f_eval_h = s_eval_h
            w_eval_sr = s_eval_sr
            w_eval_ret = s_eval_ret
            w_eval_h = s_eval_h

        chosen_full_sr_deltas.append(float(f_eval_sr - s_eval_sr))
        chosen_full_ret_deltas.append(float(f_eval_ret - s_eval_ret))
        chosen_full_haz_gains.append(float(s_eval_h - f_eval_h))
        chosen_warm_sr_deltas.append(float(w_eval_sr - s_eval_sr))
        chosen_warm_ret_deltas.append(float(w_eval_ret - s_eval_ret))
        chosen_warm_haz_gains.append(float(s_eval_h - w_eval_h))

    n = int(len(rows))
    return {
        "num_seeds": n,
        "accept_transfer_count": int(accept_count),
        "fallback_to_scratch_count": int(fallback_count),
        "accept_transfer_rate": (float(accept_count / float(n)) if n > 0 else 0.0),
        "adaptive_full": {
            "success_delta_mean": _mean(chosen_full_sr_deltas),
            "return_delta_mean": _mean(chosen_full_ret_deltas),
            "hazard_gain_mean": _mean(chosen_full_haz_gains),
            "success_deltas": chosen_full_sr_deltas,
            "return_deltas": chosen_full_ret_deltas,
            "hazard_gains": chosen_full_haz_gains,
        },
        "adaptive_warmup": {
            "success_delta_mean": _mean(chosen_warm_sr_deltas),
            "return_delta_mean": _mean(chosen_warm_ret_deltas),
            "hazard_gain_mean": _mean(chosen_warm_haz_gains),
            "success_deltas": chosen_warm_sr_deltas,
            "return_deltas": chosen_warm_ret_deltas,
            "hazard_gains": chosen_warm_haz_gains,
        },
        "sample_decisions": decisions[:5],
    }


def _score(result: Dict[str, Any], *, mode: str) -> float:
    block = result.get("adaptive_full", {}) if str(mode) == "full" else result.get("adaptive_warmup", {})
    sr = _safe_float((block or {}).get("success_delta_mean", 0.0), 0.0)
    ret = _safe_float((block or {}).get("return_delta_mean", 0.0), 0.0)
    haz = _safe_float((block or {}).get("hazard_gain_mean", 0.0), 0.0)
    # Primary: success delta. Secondary: hazard, return.
    return float(100.0 * sr + 0.02 * haz + ret)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_json", type=str, required=True)
    ap.add_argument("--logic_values", type=str, default="all")
    ap.add_argument("--transfer_conditions", type=str, default="sf_transfer_warmup,sf_transfer")
    ap.add_argument("--transfer_early_return_mins", type=str, default="-1e9,-0.2,0.0,0.2")
    ap.add_argument("--transfer_early_hazard_maxs", type=str, default="-1,100,150,200,300,500")
    ap.add_argument("--transfer_early_success_mins", type=str, default="-1")
    ap.add_argument("--transfer_early_forward_mse_maxs", type=str, default="-1")
    ap.add_argument("--canary_transfer_minus_scratch_early_success_mins", type=str, default="-1e9")
    ap.add_argument("--canary_transfer_minus_scratch_early_return_mins", type=str, default="-1e9")
    ap.add_argument("--canary_scratch_minus_transfer_early_hazard_maxs", type=str, default="-1e9")
    ap.add_argument("--canary_scratch_minus_transfer_early_forward_mse_maxs", type=str, default="-1e9")
    ap.add_argument("--transfer_early_return_slope_mins", type=str, default="-1e9")
    ap.add_argument("--transfer_early_success_slope_mins", type=str, default="-1e9")
    ap.add_argument("--transfer_early_hazard_slope_maxs", type=str, default="1e9")
    ap.add_argument("--transfer_early_forward_mse_slope_maxs", type=str, default="1e9")
    ap.add_argument("--canary_delta_return_slope_mins", type=str, default="-1e9")
    ap.add_argument("--canary_delta_success_slope_mins", type=str, default="-1e9")
    ap.add_argument("--canary_delta_hazard_slope_maxs", type=str, default="1e9")
    ap.add_argument("--canary_delta_forward_mse_slope_maxs", type=str, default="1e9")
    ap.add_argument("--source_policy_bank_majority_mins", type=str, default="-1")
    ap.add_argument("--oracle_scratch_eval_success_maxs", type=str, default="-1")
    ap.add_argument("--oracle_scratch_early_success_maxs", type=str, default="-1")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--score_mode", type=str, default="full", choices=["full", "warmup"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=7)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    artifact_path = Path(args.artifact_json)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = payload.get("per_seed", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Artifact missing per_seed rows.")

    logic_values = [s.strip() for s in str(args.logic_values).split(",") if s.strip()]
    transfer_conditions = [s.strip() for s in str(args.transfer_conditions).split(",") if s.strip()]
    tr_ret_mins = _parse_float_list(args.transfer_early_return_mins, default=[-1e9])
    tr_haz_maxs = _parse_float_list(args.transfer_early_hazard_maxs, default=[-1.0])
    tr_sr_mins = _parse_float_list(args.transfer_early_success_mins, default=[-1.0])
    tr_fwd_maxs = _parse_float_list(args.transfer_early_forward_mse_maxs, default=[-1.0])
    canary_sr_mins = _parse_float_list(args.canary_transfer_minus_scratch_early_success_mins, default=[-1e9])
    canary_ret_mins = _parse_float_list(args.canary_transfer_minus_scratch_early_return_mins, default=[-1e9])
    canary_haz_maxs = _parse_float_list(args.canary_scratch_minus_transfer_early_hazard_maxs, default=[-1e9])
    canary_fwd_maxs = _parse_float_list(args.canary_scratch_minus_transfer_early_forward_mse_maxs, default=[-1e9])
    tr_ret_slope_mins = _parse_float_list(args.transfer_early_return_slope_mins, default=[-1e9])
    tr_succ_slope_mins = _parse_float_list(args.transfer_early_success_slope_mins, default=[-1e9])
    tr_haz_slope_maxs = _parse_float_list(args.transfer_early_hazard_slope_maxs, default=[1e9])
    tr_fwd_slope_maxs = _parse_float_list(args.transfer_early_forward_mse_slope_maxs, default=[1e9])
    canary_ret_slope_mins = _parse_float_list(args.canary_delta_return_slope_mins, default=[-1e9])
    canary_succ_slope_mins = _parse_float_list(args.canary_delta_success_slope_mins, default=[-1e9])
    canary_haz_slope_maxs = _parse_float_list(args.canary_delta_hazard_slope_maxs, default=[1e9])
    canary_fwd_slope_maxs = _parse_float_list(args.canary_delta_forward_mse_slope_maxs, default=[1e9])
    src_bank_maj_mins = _parse_float_list(args.source_policy_bank_majority_mins, default=[-1.0])
    sc_eval_sr_maxs = _parse_float_list(args.oracle_scratch_eval_success_maxs, default=[-1.0])
    sc_early_sr_maxs = _parse_float_list(args.oracle_scratch_early_success_maxs, default=[-1.0])

    results: List[Dict[str, Any]] = []
    for logic in logic_values:
        for cond in transfer_conditions:
            for tr_ret in tr_ret_mins:
                for tr_haz in tr_haz_maxs:
                    for tr_sr in tr_sr_mins:
                        for tr_fwd in tr_fwd_maxs:
                            for canary_sr in canary_sr_mins:
                                for canary_ret in canary_ret_mins:
                                    for canary_haz in canary_haz_maxs:
                                        for canary_fwd in canary_fwd_maxs:
                                            for tr_ret_slope in tr_ret_slope_mins:
                                                for tr_succ_slope in tr_succ_slope_mins:
                                                    for tr_haz_slope in tr_haz_slope_maxs:
                                                        for tr_fwd_slope in tr_fwd_slope_maxs:
                                                            for can_ret_slope in canary_ret_slope_mins:
                                                                for can_succ_slope in canary_succ_slope_mins:
                                                                    for can_haz_slope in canary_haz_slope_maxs:
                                                                        for can_fwd_slope in canary_fwd_slope_maxs:
                                                                            for src_bank_maj in src_bank_maj_mins:
                                                                                for sc_eval_sr in sc_eval_sr_maxs:
                                                                                    for sc_early_sr in sc_early_sr_maxs:
                                                                                        cfg = _adaptive_gate_cfg(
                                                                                            enabled=True,
                                                                                            scratch_early_success_max=float(sc_early_sr),
                                                                                            scratch_early_return_max=-1e9,
                                                                                            scratch_early_hazard_min=-1.0,
                                                                                            scratch_eval_success_max=float(sc_eval_sr),
                                                                                            scratch_eval_return_max=-1e9,
                                                                                            scratch_eval_hazard_min=-1.0,
                                                                                            transfer_gate_condition_key=str(cond),
                                                                                            transfer_early_success_min=float(tr_sr),
                                                                                            transfer_early_return_min=float(tr_ret),
                                                                                            transfer_early_hazard_max=float(tr_haz),
                                                                                            transfer_early_forward_mse_max=float(tr_fwd),
                                                                                            transfer_minus_scratch_early_success_min=float(canary_sr),
                                                                                            transfer_minus_scratch_early_return_min=float(canary_ret),
                                                                                            scratch_minus_transfer_early_hazard_max=float(canary_haz),
                                                                                            scratch_minus_transfer_early_forward_mse_max=float(canary_fwd),
                                                                                            transfer_early_return_slope_min=float(tr_ret_slope),
                                                                                            transfer_early_success_slope_min=float(tr_succ_slope),
                                                                                            transfer_early_hazard_slope_max=float(tr_haz_slope),
                                                                                            transfer_early_forward_mse_slope_max=float(tr_fwd_slope),
                                                                                            canary_delta_return_slope_min=float(can_ret_slope),
                                                                                            canary_delta_success_slope_min=float(can_succ_slope),
                                                                                            canary_delta_hazard_slope_max=float(can_haz_slope),
                                                                                            canary_delta_forward_mse_slope_max=float(can_fwd_slope),
                                                                                            source_policy_bank_majority_min=float(src_bank_maj),
                                                                                            logic=str(logic),
                                                                                        )
                                                                                        res = _evaluate_gate(rows=rows, gate_cfg=cfg)
                                                                                        score = _score(res, mode=str(args.score_mode))

                                                    full = res["adaptive_full"]
                                                    warm = res["adaptive_warmup"]
                                                    full_sr_ci = _bootstrap_ci(full["success_deltas"], n=int(args.bootstrap), seed=int(args.bootstrap_seed) + 1)
                                                    full_ret_ci = _bootstrap_ci(full["return_deltas"], n=int(args.bootstrap), seed=int(args.bootstrap_seed) + 2)
                                                    full_h_ci = _bootstrap_ci(full["hazard_gains"], n=int(args.bootstrap), seed=int(args.bootstrap_seed) + 3)
                                                    warm_sr_ci = _bootstrap_ci(warm["success_deltas"], n=int(args.bootstrap), seed=int(args.bootstrap_seed) + 4)
                                                    warm_ret_ci = _bootstrap_ci(warm["return_deltas"], n=int(args.bootstrap), seed=int(args.bootstrap_seed) + 5)
                                                    warm_h_ci = _bootstrap_ci(warm["hazard_gains"], n=int(args.bootstrap), seed=int(args.bootstrap_seed) + 6)

                                                    results.append(
                                                        {
                                                            "score": float(score),
                                                            "gate_cfg": cfg,
                                                            "accept_transfer_count": int(res["accept_transfer_count"]),
                                                            "fallback_to_scratch_count": int(res["fallback_to_scratch_count"]),
                                                            "accept_transfer_rate": float(res["accept_transfer_rate"]),
                                                            "adaptive_full": {
                                                                "success_delta_mean": full["success_delta_mean"],
                                                                "success_delta_ci95": [full_sr_ci[0], full_sr_ci[1]],
                                                                "return_delta_mean": full["return_delta_mean"],
                                                                "return_delta_ci95": [full_ret_ci[0], full_ret_ci[1]],
                                                                "hazard_gain_mean": full["hazard_gain_mean"],
                                                                "hazard_gain_ci95": [full_h_ci[0], full_h_ci[1]],
                                                            },
                                                            "adaptive_warmup": {
                                                                "success_delta_mean": warm["success_delta_mean"],
                                                                "success_delta_ci95": [warm_sr_ci[0], warm_sr_ci[1]],
                                                                "return_delta_mean": warm["return_delta_mean"],
                                                                "return_delta_ci95": [warm_ret_ci[0], warm_ret_ci[1]],
                                                                "hazard_gain_mean": warm["hazard_gain_mean"],
                                                                "hazard_gain_ci95": [warm_h_ci[0], warm_h_ci[1]],
                                                            },
                                                        }
                                                    )

    ranked = sorted(results, key=lambda x: float(x.get("score", -1e9)), reverse=True)
    top_k = max(1, int(args.top_k))
    out = {
        "experiment": "sf_adaptive_gate_posthoc_sweep",
        "artifact_json": str(artifact_path),
        "num_seeds": int(len(rows)),
        "score_mode": str(args.score_mode),
        "num_gate_configs": int(len(results)),
        "top_results": ranked[:top_k],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if str(args.out_json or "").strip():
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
