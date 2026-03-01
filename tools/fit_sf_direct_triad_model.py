import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fit_sf_gate_model import (
    DEFAULT_FEATURE_NAMES,
    _artifact_cv_splits,
    _extract_eval_metrics,
    _load_artifact_rows,
    _mean,
    _parse_str_list,
    _safe_float,
    _train_test_split_indices,
)
from tools.validate_sf_transfer import _adaptive_triage_direct_features


def _softmax_np(z: np.ndarray) -> np.ndarray:
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z.astype(np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    z = np.clip(z, -60.0, 60.0)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return e / s


def _build_direct_feature_names(base_names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for n in base_names:
        out.append(f"full::{n}")
    for n in base_names:
        out.append(f"warm::{n}")
    for n in base_names:
        out.append(f"full_minus_warm::{n}")
    return out


def _triad_utilities(
    row: Dict[str, Any],
    *,
    success_w: float,
    return_w: float,
    hazard_w: float,
) -> Dict[str, float]:
    s = _extract_eval_metrics(row, "sf_scratch")
    f = _extract_eval_metrics(row, "sf_transfer")
    w = _extract_eval_metrics(row, "sf_transfer_warmup")

    def util(t: Dict[str, float]) -> float:
        ds = float(t["success_rate"] - s["success_rate"])
        dr = float(t["mean_return"] - s["mean_return"])
        hg = float(s["hazard_per_1k"] - t["hazard_per_1k"])
        return float(float(success_w) * ds + float(return_w) * dr + float(hazard_w) * hg)

    return {
        "sf_scratch": 0.0,
        "sf_transfer": util(f),
        "sf_transfer_warmup": util(w),
    }


def _triad_label(
    row: Dict[str, Any],
    *,
    success_w: float,
    return_w: float,
    hazard_w: float,
    require_positive_gain: bool,
) -> int:
    utils = _triad_utilities(row, success_w=success_w, return_w=return_w, hazard_w=hazard_w)
    order = ["sf_scratch", "sf_transfer", "sf_transfer_warmup"]
    best_name = max(order, key=lambda k: (float(utils.get(k, 0.0)), 0 if k == "sf_scratch" else 1))
    if bool(require_positive_gain) and best_name != "sf_scratch" and float(utils.get(best_name, 0.0)) <= 0.0:
        best_name = "sf_scratch"
    return {"sf_scratch": 0, "sf_transfer": 1, "sf_transfer_warmup": 2}[best_name]


def _build_direct_dataset(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_names: Sequence[str],
    success_w: float,
    return_w: float,
    hazard_w: float,
    require_positive_gain: bool,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Dict[str, Any]]]:
    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    seeds: List[int] = []
    kept: List[Dict[str, Any]] = []
    for row in rows:
        feats = _adaptive_triage_direct_features(row)
        vec = [_safe_float(feats.get(n, 0.0), 0.0) for n in feature_names]
        X_rows.append(vec)
        y_rows.append(
            _triad_label(
                row,
                success_w=float(success_w),
                return_w=float(return_w),
                hazard_w=float(hazard_w),
                require_positive_gain=bool(require_positive_gain),
            )
        )
        seeds.append(int(row.get("seed", 0) or 0))
        kept.append(row)
    return np.asarray(X_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64), seeds, kept


def _fit_softmax(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_idx: np.ndarray,
    lr: float,
    l2: float,
    epochs: int,
    class_weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    X_train = X[train_idx]
    y_train = y[train_idx]
    n, d = X_train.shape
    k = int(max(3, int(np.max(y)) + 1))
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    Xn = (X_train - mean) / scale
    priors = np.bincount(y_train.astype(np.int64), minlength=k).astype(np.float64) + 1e-3
    priors /= np.sum(priors)
    W = np.zeros((d, k), dtype=np.float64)
    b = np.log(priors)
    cw = np.asarray(class_weights if class_weights is not None else np.ones((k,), dtype=np.float64), dtype=np.float64)
    if cw.size < k:
        cw = np.pad(cw, (0, k - cw.size), constant_values=1.0)
    cw = np.where(cw <= 0.0, 1.0, cw)
    Y = np.eye(k, dtype=np.float64)[y_train.astype(np.int64)]
    sample_w = cw[y_train.astype(np.int64)]
    for _ in range(max(1, int(epochs))):
        logits = Xn @ W + b
        P = _softmax_np(logits)
        err = (P - Y) * sample_w[:, None]
        denom = float(np.sum(sample_w))
        if denom <= 0.0:
            denom = float(n)
        gW = (Xn.T @ err) / denom + float(l2) * W
        gb = np.sum(err, axis=0) / denom
        W -= float(lr) * gW
        b -= float(lr) * gb
    return {"weights": W, "bias": b, "mean": mean, "scale": scale}


def _predict_softmax(X: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    W = np.asarray(model["weights"], dtype=np.float64)
    b = np.asarray(model["bias"], dtype=np.float64)
    Xn = (X - mean) / np.where(np.abs(scale) < 1e-8, 1.0, scale)
    return _softmax_np(Xn @ W + b)


def _triad_policy_eval(
    rows: Sequence[Dict[str, Any]],
    probs: np.ndarray,
    *,
    full_accept_prob_min: float,
    warmup_accept_prob_min: float,
) -> Dict[str, Any]:
    sr_deltas: List[float] = []
    ret_deltas: List[float] = []
    haz_gains: List[float] = []
    actions: List[str] = []
    for i, row in enumerate(rows):
        p = probs[i] if i < probs.shape[0] else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        p_s = float(p[0]) if p.size > 0 else 1.0
        p_f = float(p[1]) if p.size > 1 else 0.0
        p_w = float(p[2]) if p.size > 2 else 0.0
        if (p_f >= float(full_accept_prob_min)) and (p_f >= p_w) and (p_f >= p_s):
            chosen_key = "sf_transfer"
            action = "accept_transfer"
        elif (p_w >= float(warmup_accept_prob_min)) and (p_w >= p_f) and (p_w >= p_s):
            chosen_key = "sf_transfer_warmup"
            action = "warmup_only"
        else:
            chosen_key = "sf_scratch"
            action = "fallback_scratch"
        s = _extract_eval_metrics(row, "sf_scratch")
        c = _extract_eval_metrics(row, chosen_key)
        actions.append(action)
        sr_deltas.append(float(c["success_rate"] - s["success_rate"]))
        ret_deltas.append(float(c["mean_return"] - s["mean_return"]))
        haz_gains.append(float(s["hazard_per_1k"] - c["hazard_per_1k"]))
    n = max(1, len(rows))
    return {
        "num_rows": int(len(rows)),
        "accept_transfer_count": int(sum(1 for a in actions if a == "accept_transfer")),
        "warmup_only_count": int(sum(1 for a in actions if a == "warmup_only")),
        "fallback_to_scratch_count": int(sum(1 for a in actions if a == "fallback_scratch")),
        "accept_transfer_rate": float(sum(1 for a in actions if a == "accept_transfer") / float(n)),
        "warmup_only_rate": float(sum(1 for a in actions if a == "warmup_only") / float(n)),
        "fallback_to_scratch_rate": float(sum(1 for a in actions if a == "fallback_scratch") / float(n)),
        "success_delta_mean": _mean(sr_deltas),
        "return_delta_mean": _mean(ret_deltas),
        "hazard_gain_mean": _mean(haz_gains),
        "negative_transfer_rate_success": float(sum(1 for d in sr_deltas if float(d) < 0.0) / float(n)) if len(rows) else 0.0,
    }


def _calibrate_direct_triad_policy(
    *,
    rows: Sequence[Dict[str, Any]],
    split_records: Sequence[Dict[str, Any]],
    min_total_transfer_rate: float,
    target_total_transfer_rate: float,
    total_transfer_penalty: float,
    min_success_delta: float,
    min_hazard_gain: float,
    max_negative_transfer_rate: float,
) -> Dict[str, Any]:
    if not split_records:
        return {"used": False}
    best: Optional[Dict[str, Any]] = None
    full_grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    warm_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    for fthr in full_grid:
        for wthr in warm_grid:
            evals: List[Dict[str, Any]] = []
            for rec in split_records:
                idx = [int(i) for i in rec.get("test_idx", [])]
                if not idx:
                    continue
                pr = np.asarray(rec.get("triad_test_probs", []), dtype=np.float64)
                evals.append(
                    _triad_policy_eval(
                        [rows[i] for i in idx],
                        pr,
                        full_accept_prob_min=float(fthr),
                        warmup_accept_prob_min=float(wthr),
                    )
                )
            if not evals:
                continue
            sr = _mean(e["success_delta_mean"] for e in evals if isinstance(e.get("success_delta_mean"), (int, float)))
            ret = _mean(e["return_delta_mean"] for e in evals if isinstance(e.get("return_delta_mean"), (int, float)))
            haz = _mean(e["hazard_gain_mean"] for e in evals if isinstance(e.get("hazard_gain_mean"), (int, float)))
            neg = _mean(e["negative_transfer_rate_success"] for e in evals if isinstance(e.get("negative_transfer_rate_success"), (int, float)))
            full_rate = _mean(e["accept_transfer_rate"] for e in evals if isinstance(e.get("accept_transfer_rate"), (int, float)))
            warm_rate = _mean(e["warmup_only_rate"] for e in evals if isinstance(e.get("warmup_only_rate"), (int, float)))
            if sr is None or ret is None or haz is None or neg is None:
                continue
            full_rate = 0.0 if full_rate is None else float(full_rate)
            warm_rate = 0.0 if warm_rate is None else float(warm_rate)
            total_rate = float(full_rate + warm_rate)
            if total_rate < float(min_total_transfer_rate):
                continue
            if float(sr) < float(min_success_delta):
                continue
            if float(haz) < float(min_hazard_gain):
                continue
            if float(neg) > float(max_negative_transfer_rate):
                continue
            score = float(100.0 * sr + ret + 0.02 * haz - 5.0 * neg)
            if float(target_total_transfer_rate) >= 0.0 and float(total_transfer_penalty) > 0.0:
                score -= float(total_transfer_penalty) * abs(total_rate - float(target_total_transfer_rate))
            cand = {
                "score": float(score),
                "full_accept_prob_min": float(fthr),
                "warmup_accept_prob_min": float(wthr),
                "cv_policy_mean": {
                    "success_delta_mean": float(sr),
                    "return_delta_mean": float(ret),
                    "hazard_gain_mean": float(haz),
                    "negative_transfer_rate_success": float(neg),
                    "full_transfer_rate": float(full_rate),
                    "warmup_transfer_rate": float(warm_rate),
                    "total_transfer_rate": float(total_rate),
                },
            }
            if best is None or float(cand["score"]) > float(best["score"]):
                best = cand
    return {
        "used": bool(best),
        "best": best,
        "constraints": {
            "min_total_transfer_rate": float(min_total_transfer_rate),
            "target_total_transfer_rate": float(target_total_transfer_rate),
            "total_transfer_penalty": float(total_transfer_penalty),
            "min_success_delta": float(min_success_delta),
            "min_hazard_gain": float(min_hazard_gain),
            "max_negative_transfer_rate": float(max_negative_transfer_rate),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_json", type=str, default="")
    ap.add_argument("--artifact_jsons", type=str, default="")
    ap.add_argument("--feature_names", type=str, default="")
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--cv_split_mode", type=str, default="artifact", choices=["row", "artifact", "auto"])
    ap.add_argument("--test_frac", type=float, default=0.33)
    ap.add_argument("--split_seed", type=int, default=7)
    ap.add_argument("--success_weight", type=float, default=100.0)
    ap.add_argument("--return_weight", type=float, default=1.0)
    ap.add_argument("--hazard_weight", type=float, default=0.02)
    ap.add_argument("--require_positive_gain", action="store_true")
    ap.add_argument("--calibrate_policy_from_cv", action="store_true")
    ap.add_argument("--calib_min_total_transfer_rate", type=float, default=0.1)
    ap.add_argument("--calib_target_total_transfer_rate", type=float, default=0.2)
    ap.add_argument("--calib_total_transfer_penalty", type=float, default=1.0)
    ap.add_argument("--calib_min_cv_success_delta", type=float, default=-1e9)
    ap.add_argument("--calib_min_cv_hazard_gain", type=float, default=-1e9)
    ap.add_argument("--calib_max_cv_negative_transfer_rate", type=float, default=1.0)
    ap.add_argument("--full_accept_prob", type=float, default=0.55)
    ap.add_argument("--warmup_accept_prob", type=float, default=0.50)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    rows, artifact_paths = _load_artifact_rows(artifact_json=str(args.artifact_json or ""), artifact_jsons=str(args.artifact_jsons or ""))
    base_names = _parse_str_list(str(args.feature_names or ""), default=DEFAULT_FEATURE_NAMES)
    feature_names = _build_direct_feature_names(base_names)
    X, y, seeds, kept_rows = _build_direct_dataset(
        rows,
        feature_names=feature_names,
        success_w=float(args.success_weight),
        return_w=float(args.return_weight),
        hazard_w=float(args.hazard_weight),
        require_positive_gain=bool(args.require_positive_gain),
    )
    n = int(X.shape[0])
    if n < 8:
        raise ValueError("Not enough rows for direct triage training.")

    train_idx, test_idx = _train_test_split_indices(n, test_frac=float(args.test_frac), seed=int(args.split_seed))
    counts = np.bincount(y.astype(np.int64), minlength=3).astype(np.float64)
    class_weights = np.where(counts > 0.0, float(n) / np.maximum(1.0, 3.0 * counts), 1.0)
    fit_split = _fit_softmax(X, y, train_idx=train_idx, lr=float(args.lr), l2=float(args.l2), epochs=int(args.epochs), class_weights=class_weights)
    p_train = _predict_softmax(X[train_idx], fit_split)
    p_test = _predict_softmax(X[test_idx], fit_split) if test_idx.size > 0 else np.zeros((0, 3), dtype=np.float64)

    cv_records: List[Dict[str, Any]] = []
    split_mode = str(args.cv_split_mode or "artifact").strip().lower()
    artifact_splits = _artifact_cv_splits(kept_rows) if split_mode in {"artifact", "auto"} else []
    if artifact_splits:
        splits = artifact_splits
    else:
        # fallback single random split
        splits = [{"split_kind": "row_split", "split_name": f"seed={int(args.split_seed)}", "train_idx": train_idx, "test_idx": test_idx}]
    for sidx, split in enumerate(splits):
        tri = np.asarray(split.get("train_idx", []), dtype=np.int64)
        tei = np.asarray(split.get("test_idx", []), dtype=np.int64)
        if tri.size == 0 or tei.size == 0:
            continue
        fit_cv = _fit_softmax(X, y, train_idx=tri, lr=float(args.lr), l2=float(args.l2), epochs=int(args.epochs), class_weights=class_weights)
        probs_test = _predict_softmax(X[tei], fit_cv)
        pred = np.argmax(probs_test, axis=1).astype(np.int64)
        acc = float(np.mean(pred == y[tei])) if tei.size > 0 else None
        rec = {
            "split_seed": int(sidx),
            "split_kind": str(split.get("split_kind", "")),
            "split_name": str(split.get("split_name", "")),
            "train_n": int(tri.size),
            "test_n": int(tei.size),
            "test_idx": [int(i) for i in tei.tolist()],
            "triad_test_probs": probs_test.astype(np.float64).tolist(),
            "triad_test_accuracy": acc,
            "triad_policy_eval_test": _triad_policy_eval(
                [kept_rows[int(i)] for i in tei.tolist()],
                probs_test,
                full_accept_prob_min=float(args.full_accept_prob),
                warmup_accept_prob_min=float(args.warmup_accept_prob),
            ),
        }
        cv_records.append(rec)

    triad_cal = (
        _calibrate_direct_triad_policy(
            rows=kept_rows,
            split_records=cv_records,
            min_total_transfer_rate=float(args.calib_min_total_transfer_rate),
            target_total_transfer_rate=float(args.calib_target_total_transfer_rate),
            total_transfer_penalty=float(args.calib_total_transfer_penalty),
            min_success_delta=float(args.calib_min_cv_success_delta),
            min_hazard_gain=float(args.calib_min_cv_hazard_gain),
            max_negative_transfer_rate=float(args.calib_max_cv_negative_transfer_rate),
        )
        if bool(args.calibrate_policy_from_cv)
        else {"used": False}
    )
    best = (triad_cal.get("best") or {}) if isinstance(triad_cal, dict) else {}
    eff_full = float(best.get("full_accept_prob_min", args.full_accept_prob)) if best else float(args.full_accept_prob)
    eff_warm = float(best.get("warmup_accept_prob_min", args.warmup_accept_prob)) if best else float(args.warmup_accept_prob)

    fit_full = _fit_softmax(X, y, train_idx=np.arange(n, dtype=np.int64), lr=float(args.lr), l2=float(args.l2), epochs=int(args.epochs), class_weights=class_weights)
    p_full = _predict_softmax(X, fit_full)

    out = {
        "schema_version": "sf_adaptive_gate_model.v1",
        "model_type": "triage_softmax_linear",
        "artifact_json": str(args.artifact_json),
        "artifact_jsons": artifact_paths,
        "num_training_rows": int(n),
        "split": {
            "cv_split_mode": str(args.cv_split_mode),
            "test_frac": float(args.test_frac),
            "split_seed": int(args.split_seed),
        },
        "triage_labeling": {
            "success_weight": float(args.success_weight),
            "return_weight": float(args.return_weight),
            "hazard_weight": float(args.hazard_weight),
            "require_positive_gain": bool(args.require_positive_gain),
        },
        "triage_direct_model": {
            "feature_names": [str(x) for x in feature_names],
            "class_names": ["sf_scratch", "sf_transfer", "sf_transfer_warmup"],
            "weights": np.asarray(fit_full["weights"]).astype(float).tolist(),
            "bias": np.asarray(fit_full["bias"]).astype(float).tolist(),
            "normalization": {
                "mean": np.asarray(fit_full["mean"]).astype(float).tolist(),
                "scale": np.asarray(fit_full["scale"]).astype(float).tolist(),
            },
            "train_class_counts": [int(x) for x in np.bincount(y.astype(np.int64), minlength=3).tolist()],
        },
        "triage_direct_policy": {
            "full_accept_prob_min": float(eff_full),
            "warmup_accept_prob_min": float(eff_warm),
            "prefer_higher_probability": True,
        },
        "triage_direct_calibration": triad_cal,
        "triage_direct_eval": {
            "split_test_accuracy": float(np.mean(np.argmax(p_test, axis=1) == y[test_idx])) if test_idx.size > 0 else None,
            "split_policy_eval_test": _triad_policy_eval(
                [kept_rows[int(i)] for i in test_idx.tolist()],
                p_test,
                full_accept_prob_min=float(eff_full),
                warmup_accept_prob_min=float(eff_warm),
            ) if test_idx.size > 0 else {"num_rows": 0},
            "full_export_policy_eval": _triad_policy_eval(
                kept_rows,
                p_full,
                full_accept_prob_min=float(eff_full),
                warmup_accept_prob_min=float(eff_warm),
            ),
            "cv": {
                "num_splits": int(len(cv_records)),
                "triad_test_accuracy_mean": _mean(
                    _safe_float(r.get("triad_test_accuracy"), 0.0) for r in cv_records if isinstance(r.get("triad_test_accuracy"), (int, float))
                ),
                "triad_policy_eval_test_mean": {
                    "accept_transfer_rate": _mean(_safe_float((r.get("triad_policy_eval_test") or {}).get("accept_transfer_rate"), 0.0) for r in cv_records if isinstance(r.get("triad_policy_eval_test"), dict)),
                    "warmup_only_rate": _mean(_safe_float((r.get("triad_policy_eval_test") or {}).get("warmup_only_rate"), 0.0) for r in cv_records if isinstance(r.get("triad_policy_eval_test"), dict)),
                    "fallback_to_scratch_rate": _mean(_safe_float((r.get("triad_policy_eval_test") or {}).get("fallback_to_scratch_rate"), 0.0) for r in cv_records if isinstance(r.get("triad_policy_eval_test"), dict)),
                    "success_delta_mean": _mean(_safe_float((r.get("triad_policy_eval_test") or {}).get("success_delta_mean"), 0.0) for r in cv_records if isinstance(r.get("triad_policy_eval_test"), dict)),
                    "return_delta_mean": _mean(_safe_float((r.get("triad_policy_eval_test") or {}).get("return_delta_mean"), 0.0) for r in cv_records if isinstance(r.get("triad_policy_eval_test"), dict)),
                    "hazard_gain_mean": _mean(_safe_float((r.get("triad_policy_eval_test") or {}).get("hazard_gain_mean"), 0.0) for r in cv_records if isinstance(r.get("triad_policy_eval_test"), dict)),
                    "negative_transfer_rate_success": _mean(_safe_float((r.get("triad_policy_eval_test") or {}).get("negative_transfer_rate_success"), 0.0) for r in cv_records if isinstance(r.get("triad_policy_eval_test"), dict)),
                },
                "splits": cv_records,
            },
        },
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact_json": str(args.artifact_json),
        "artifact_jsons": artifact_paths,
        "num_training_rows": int(n),
        "out_json": str(out_path),
        "triage_direct_policy": out["triage_direct_policy"],
        "cv_num_splits": int(len(cv_records)),
    }, indent=2))


if __name__ == "__main__":
    main()
