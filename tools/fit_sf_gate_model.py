from __future__ import annotations

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

from tools.validate_sf_transfer import _adaptive_gate_model_features


DEFAULT_FEATURE_NAMES: List[str] = [
    "scratch_early_success_rate",
    "scratch_early_return",
    "scratch_early_hazard_per_1k",
    "scratch_early_forward_mse",
    "transfer_early_success_rate",
    "transfer_early_return",
    "transfer_early_hazard_per_1k",
    "transfer_early_forward_mse",
    "transfer_minus_scratch_early_success_rate",
    "transfer_minus_scratch_early_return",
    "scratch_minus_transfer_early_hazard_per_1k",
    "scratch_minus_transfer_early_forward_mse",
    "transfer_early_return_slope",
    "transfer_early_success_slope",
    "transfer_early_hazard_slope",
    "transfer_early_forward_mse_slope",
    "canary_delta_return_slope",
    "canary_delta_success_slope",
    "canary_delta_hazard_slope",
    "canary_delta_forward_mse_slope",
    "source_policy_bank_majority_fraction",
    "source_policy_bank_vote_margin",
    "source_policy_bank_unique_actions",
]

DEFAULT_HARDNESS_FEATURE_NAMES: List[str] = [
    "scratch_early_success_rate",
    "scratch_early_return",
    "scratch_early_hazard_per_1k",
    "scratch_early_forward_mse",
    "source_policy_bank_majority_fraction",
    "source_policy_bank_vote_margin",
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _parse_str_list(raw: str, *, default: Sequence[str]) -> List[str]:
    txt = str(raw or "").strip()
    if not txt:
        return [str(x) for x in default]
    out = [s.strip() for s in txt.replace(";", ",").split(",") if s.strip()]
    return out or [str(x) for x in default]


def _row_identity(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(row.get("source_verse_name", "")),
        str(row.get("target_verse_name", "")),
        int(row.get("seed", 0) or 0),
    )


def _load_artifact_rows(*, artifact_json: str, artifact_jsons: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    paths: List[str] = []
    if str(artifact_jsons or "").strip():
        paths.extend(_parse_str_list(str(artifact_jsons), default=[]))
    elif str(artifact_json or "").strip():
        paths.append(str(artifact_json).strip())
    else:
        raise ValueError("No artifact path(s) provided.")
    all_rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int]] = set()
    used_paths: List[str] = []
    for p in paths:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        rows = payload.get("per_seed", [])
        if not isinstance(rows, list) or not rows:
            continue
        used_paths.append(str(p))
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = _row_identity(row)
            if rid in seen:
                continue
            seen.add(rid)
            r = dict(row)
            r["_training_artifact_json"] = str(p)
            all_rows.append(r)
    if not all_rows:
        raise ValueError("No usable per_seed rows found in artifact(s).")
    return all_rows, used_paths


def _mean(vals: Iterable[float]) -> Optional[float]:
    arr = [float(v) for v in vals]
    if not arr:
        return None
    return float(sum(arr) / float(len(arr)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-zc))


def _logit(p: float) -> float:
    x = min(1.0 - 1e-6, max(1e-6, float(p)))
    return float(math.log(x / (1.0 - x)))


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if y_true.size == 0:
        return None
    pos = int(np.sum(y_true > 0.5))
    neg = int(np.sum(y_true <= 0.5))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    pos_rank_sum = float(np.sum(ranks[y_true > 0.5]))
    auc = (pos_rank_sum - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(auc)


def _extract_eval_metrics(row: Dict[str, Any], cond_key: str) -> Dict[str, float]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    cond = tc.get(cond_key, {}) if isinstance(tc.get(cond_key), dict) else {}
    ev = cond.get("eval_summary", {}) if isinstance(cond.get("eval_summary"), dict) else {}
    return {
        "success_rate": _safe_float(ev.get("success_rate", 0.0), 0.0),
        "mean_return": _safe_float(ev.get("mean_return", 0.0), 0.0),
        "hazard_per_1k": _safe_float(ev.get("hazard_per_1k", 0.0), 0.0),
    }


def _label_row(
    row: Dict[str, Any],
    *,
    cond_key: str,
    label_mode: str,
    min_success_delta: float,
    min_hazard_gain: float,
) -> int:
    s = _extract_eval_metrics(row, "sf_scratch")
    t = _extract_eval_metrics(row, cond_key)
    sr_delta = float(t["success_rate"] - s["success_rate"])
    haz_gain = float(s["hazard_per_1k"] - t["hazard_per_1k"])
    if sr_delta <= float(min_success_delta):
        return 0
    if label_mode == "success_and_hazard_noninferior" and haz_gain < float(min_hazard_gain):
        return 0
    return 1


def _label_hard_target(
    row: Dict[str, Any],
    *,
    scratch_eval_success_max: float,
) -> int:
    s = _extract_eval_metrics(row, "sf_scratch")
    return 1 if float(s["success_rate"]) <= float(scratch_eval_success_max) else 0


def _build_dataset(
    rows: Sequence[Dict[str, Any]],
    *,
    cond_key: str,
    feature_names: Sequence[str],
    label_mode: str,
    min_success_delta: float,
    min_hazard_gain: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Dict[str, Any]]]:
    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    seeds: List[int] = []
    kept_rows: List[Dict[str, Any]] = []
    for row in rows:
        feats_payload = _adaptive_gate_model_features(row, transfer_gate_condition_key=cond_key)
        feats = feats_payload.get("features", {}) if isinstance(feats_payload.get("features"), dict) else {}
        vec = [_safe_float(feats.get(name, 0.0), 0.0) for name in feature_names]
        X_rows.append(vec)
        y_rows.append(
            _label_row(
                row,
                cond_key=cond_key,
                label_mode=label_mode,
                min_success_delta=min_success_delta,
                min_hazard_gain=min_hazard_gain,
            )
        )
        seeds.append(int(row.get("seed", 0) or 0))
        kept_rows.append(row)
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    return X, y, seeds, kept_rows


def _build_feature_matrix(
    rows: Sequence[Dict[str, Any]],
    *,
    cond_key: str,
    feature_names: Sequence[str],
) -> np.ndarray:
    mats: List[List[float]] = []
    for row in rows:
        feats_payload = _adaptive_gate_model_features(row, transfer_gate_condition_key=cond_key)
        feats = feats_payload.get("features", {}) if isinstance(feats_payload.get("features"), dict) else {}
        mats.append([_safe_float(feats.get(name, 0.0), 0.0) for name in feature_names])
    return np.asarray(mats, dtype=np.float64)


def _train_test_split_indices(n: int, *, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = list(range(int(n)))
    rng = random.Random(int(seed))
    rng.shuffle(idx)
    n_test = max(1, int(round(float(test_frac) * n))) if n > 1 else 0
    n_test = min(max(0, n_test), n - 1) if n > 1 else 0
    test_idx = np.asarray(sorted(idx[:n_test]), dtype=np.int64)
    train_idx = np.asarray(sorted(idx[n_test:]), dtype=np.int64)
    return train_idx, test_idx


def _artifact_cv_splits(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        g = str(r.get("_training_artifact_json", "") or "")
        if not g:
            g = "<unknown_artifact>"
        groups.setdefault(g, []).append(int(i))
    if len(groups) <= 1:
        return []
    splits: List[Dict[str, Any]] = []
    all_idx = set(range(len(rows)))
    for gname, gidx in sorted(groups.items(), key=lambda kv: kv[0]):
        test_set = set(int(i) for i in gidx)
        train = sorted(int(i) for i in all_idx if i not in test_set)
        test = sorted(int(i) for i in test_set)
        if not train or not test:
            continue
        splits.append(
            {
                "split_kind": "artifact_holdout",
                "split_name": str(gname),
                "train_idx": np.asarray(train, dtype=np.int64),
                "test_idx": np.asarray(test, dtype=np.int64),
            }
        )
    return splits


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_idx: np.ndarray,
    lr: float,
    l2: float,
    epochs: int,
    positive_weight: float,
) -> Dict[str, Any]:
    X_train = X[train_idx]
    y_train = y[train_idx]
    n, d = X_train.shape
    if n <= 0:
        raise ValueError("Empty training split.")
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    Xn = (X_train - mean) / scale

    p0 = float(np.clip(np.mean(y_train), 1e-6, 1.0 - 1e-6))
    w = np.zeros(d, dtype=np.float64)
    b = _logit(p0)
    pos_w = float(max(1.0, positive_weight))

    for _ in range(max(1, int(epochs))):
        z = Xn @ w + b
        p = _sigmoid(z)
        sample_w = np.where(y_train > 0.5, pos_w, 1.0)
        err = (p - y_train) * sample_w
        denom = float(np.sum(sample_w))
        if denom <= 0.0:
            denom = float(n)
        grad_w = (Xn.T @ err) / denom + float(l2) * w
        grad_b = float(np.sum(err) / denom)
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b

    return {
        "weights": w,
        "bias": float(b),
        "mean": mean,
        "scale": scale,
    }


def _predict_probs(X: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    w = np.asarray(model["weights"], dtype=np.float64)
    b = float(model["bias"])
    Xn = (X - mean) / np.where(np.abs(scale) < 1e-8, 1.0, scale)
    return _sigmoid(Xn @ w + b)


def _binary_metrics(y: np.ndarray, p: np.ndarray, *, threshold: float = 0.5) -> Dict[str, Any]:
    if len(y) == 0:
        return {
            "n": 0,
            "positive_rate": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "auc_roc": None,
            "logloss": None,
            "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        }
    yb = (y > 0.5).astype(np.int64)
    pred = (p >= float(threshold)).astype(np.int64)
    tp = int(np.sum((pred == 1) & (yb == 1)))
    tn = int(np.sum((pred == 0) & (yb == 0)))
    fp = int(np.sum((pred == 1) & (yb == 0)))
    fn = int(np.sum((pred == 0) & (yb == 1)))
    acc = float((tp + tn) / float(len(y))) if len(y) else 0.0
    prec = float(tp / float(tp + fp)) if (tp + fp) > 0 else None
    rec = float(tp / float(tp + fn)) if (tp + fn) > 0 else None
    auc = _auc_roc(y.astype(np.float64), p.astype(np.float64))
    eps = 1e-9
    logloss = float(
        -np.mean(y * np.log(np.clip(p, eps, 1.0 - eps)) + (1.0 - y) * np.log(np.clip(1.0 - p, eps, 1.0 - eps)))
    )
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y.astype(np.float64))) if len(y) else 0.0,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc_roc": auc,
        "logloss": logloss,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _cv_aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"num_splits": 0}
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "auc_roc",
        "logloss",
        "positive_rate",
    ]
    policy_keys = [
        "accept_transfer_rate",
        "warmup_only_rate",
        "fallback_to_scratch_rate",
        "success_delta_mean",
        "return_delta_mean",
        "hazard_gain_mean",
        "negative_transfer_rate_success",
    ]
    out: Dict[str, Any] = {"num_splits": int(len(records))}
    test_metric_records = [r.get("test_metrics", {}) for r in records if isinstance(r.get("test_metrics"), dict)]
    policy_records = [r.get("policy_eval_test", {}) for r in records if isinstance(r.get("policy_eval_test"), dict)]
    out["test_metrics_mean"] = {
        k: _mean(
            _safe_float(m.get(k), 0.0)
            for m in test_metric_records
            if isinstance(m.get(k), (int, float))
        )
        for k in metric_keys
    }
    out["policy_eval_test_mean"] = {
        k: _mean(
            _safe_float(m.get(k), 0.0)
            for m in policy_records
            if isinstance(m.get(k), (int, float))
        )
        for k in policy_keys
    }
    out["splits"] = records
    return out


def _parse_int_list(raw: str, *, default: Sequence[int]) -> List[int]:
    txt = str(raw or "").strip()
    if not txt:
        return [int(x) for x in default]
    out: List[int] = []
    for p in txt.replace(";", ",").split(","):
        s = p.strip()
        if not s:
            continue
        out.append(int(s))
    return out or [int(x) for x in default]


def _policy_eval(
    rows: Sequence[Dict[str, Any]],
    probs: np.ndarray,
    *,
    accept_prob: float,
    warmup_prob: float,
) -> Dict[str, Any]:
    sr_deltas: List[float] = []
    ret_deltas: List[float] = []
    haz_gains: List[float] = []
    actions: List[str] = []
    for row, p in zip(rows, probs.tolist()):
        s = _extract_eval_metrics(row, "sf_scratch")
        f = _extract_eval_metrics(row, "sf_transfer")
        w = _extract_eval_metrics(row, "sf_transfer_warmup")
        if p >= float(accept_prob):
            chosen = f
            action = "accept_transfer"
        elif p >= float(warmup_prob):
            chosen = w
            action = "warmup_only"
        else:
            chosen = s
            action = "fallback_scratch"
        actions.append(action)
        sr_deltas.append(float(chosen["success_rate"] - s["success_rate"]))
        ret_deltas.append(float(chosen["mean_return"] - s["mean_return"]))
        haz_gains.append(float(s["hazard_per_1k"] - chosen["hazard_per_1k"]))
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
        "negative_transfer_rate_success": (
            float(sum(1 for d in sr_deltas if float(d) < 0.0) / float(n)) if len(rows) else 0.0
        ),
    }


def _policy_eval_two_stage(
    rows: Sequence[Dict[str, Any]],
    effect_probs: np.ndarray,
    hardness_probs: Optional[np.ndarray],
    *,
    hard_prob: float,
    accept_prob: float,
    warmup_prob: float,
) -> Dict[str, Any]:
    sr_deltas: List[float] = []
    ret_deltas: List[float] = []
    haz_gains: List[float] = []
    actions: List[str] = []
    for i, row in enumerate(rows):
        p_eff = float(effect_probs[i]) if i < len(effect_probs) else 0.0
        p_hard = float(hardness_probs[i]) if (hardness_probs is not None and i < len(hardness_probs)) else 1.0
        s = _extract_eval_metrics(row, "sf_scratch")
        f = _extract_eval_metrics(row, "sf_transfer")
        w = _extract_eval_metrics(row, "sf_transfer_warmup")
        hard_ok = True if float(hard_prob) < 0.0 else bool(p_hard >= float(hard_prob))
        if (not hard_ok) or (p_eff < float(warmup_prob)):
            chosen = s
            action = "fallback_scratch"
        elif p_eff >= float(accept_prob):
            chosen = f
            action = "accept_transfer"
        else:
            chosen = w
            action = "warmup_only"
        actions.append(action)
        sr_deltas.append(float(chosen["success_rate"] - s["success_rate"]))
        ret_deltas.append(float(chosen["mean_return"] - s["mean_return"]))
        haz_gains.append(float(s["hazard_per_1k"] - chosen["hazard_per_1k"]))
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


def _policy_eval_joint_triad(
    rows: Sequence[Dict[str, Any]],
    full_probs: np.ndarray,
    warm_probs: np.ndarray,
    hardness_probs: Optional[np.ndarray],
    *,
    hard_prob: float,
    full_accept_prob: float,
    warm_accept_prob: float,
) -> Dict[str, Any]:
    sr_deltas: List[float] = []
    ret_deltas: List[float] = []
    haz_gains: List[float] = []
    actions: List[str] = []
    for i, row in enumerate(rows):
        pf = float(full_probs[i]) if i < len(full_probs) else 0.0
        pw = float(warm_probs[i]) if i < len(warm_probs) else 0.0
        ph = float(hardness_probs[i]) if (hardness_probs is not None and i < len(hardness_probs)) else 1.0
        s = _extract_eval_metrics(row, "sf_scratch")
        f = _extract_eval_metrics(row, "sf_transfer")
        w = _extract_eval_metrics(row, "sf_transfer_warmup")
        hard_ok = True if float(hard_prob) < 0.0 else bool(ph >= float(hard_prob))
        full_ok = bool(hard_ok and pf >= float(full_accept_prob))
        warm_ok = bool(hard_ok and pw >= float(warm_accept_prob))
        if full_ok and warm_ok:
            if pf >= pw:
                chosen = f
                action = "accept_transfer"
            else:
                chosen = w
                action = "warmup_only"
        elif full_ok:
            chosen = f
            action = "accept_transfer"
        elif warm_ok:
            chosen = w
            action = "warmup_only"
        else:
            chosen = s
            action = "fallback_scratch"
        actions.append(action)
        sr_deltas.append(float(chosen["success_rate"] - s["success_rate"]))
        ret_deltas.append(float(chosen["mean_return"] - s["mean_return"]))
        haz_gains.append(float(s["hazard_per_1k"] - chosen["hazard_per_1k"]))
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


def _serialize_fit_model(
    fit: Dict[str, Any],
    *,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    return {
        "feature_names": [str(x) for x in feature_names],
        "weights": [float(x) for x in np.asarray(fit["weights"]).tolist()],
        "bias": float(fit["bias"]),
        "normalization": {
            "mean": [float(x) for x in np.asarray(fit["mean"]).tolist()],
            "scale": [float(x) for x in np.asarray(fit["scale"]).tolist()],
        },
    }


def _calibrate_joint_triad_policy(
    *,
    rows: Sequence[Dict[str, Any]],
    full_split_records: Sequence[Dict[str, Any]],
    warm_split_records: Sequence[Dict[str, Any]],
    min_accept_rate: float = 0.0,
    min_total_transfer_rate: float = 0.0,
    target_accept_rate: float = -1.0,
    accept_rate_penalty: float = 0.0,
    min_success_delta: float = -1e9,
    min_hazard_gain: float = -1e9,
    max_negative_transfer_rate: float = 1.0,
) -> Dict[str, Any]:
    if not full_split_records or not warm_split_records:
        return {"used": False}
    warm_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for rec in warm_split_records:
        k = (str(rec.get("split_kind", "")), str(rec.get("split_name", "")), int(rec.get("test_n", 0)))
        warm_map[k] = rec
    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for rec_f in full_split_records:
        k = (str(rec_f.get("split_kind", "")), str(rec_f.get("split_name", "")), int(rec_f.get("test_n", 0)))
        rec_w = warm_map.get(k)
        if not isinstance(rec_w, dict):
            continue
        if list(rec_f.get("test_idx", [])) != list(rec_w.get("test_idx", [])):
            continue
        aligned.append((rec_f, rec_w))
    if not aligned:
        return {"used": False}
    best: Optional[Dict[str, Any]] = None
    hard_grid = [-1.0, 0.45, 0.55, 0.65]
    full_grid = [0.55, 0.60, 0.65, 0.70, 0.75]
    warm_grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    for h in hard_grid:
        for fthr in full_grid:
            for wthr in warm_grid:
                evals: List[Dict[str, Any]] = []
                for rec_f, rec_w in aligned:
                    test_idx = [int(i) for i in rec_f.get("test_idx", [])]
                    if not test_idx:
                        continue
                    full_probs = np.asarray(rec_f.get("effect_test_probs", []), dtype=np.float64)
                    warm_probs = np.asarray(rec_w.get("effect_test_probs", []), dtype=np.float64)
                    hard_probs = (
                        np.asarray(rec_f.get("hardness_test_probs", []), dtype=np.float64)
                        if isinstance(rec_f.get("hardness_test_probs"), list)
                        else None
                    )
                    test_rows = [rows[i] for i in test_idx]
                    evals.append(
                        _policy_eval_joint_triad(
                            test_rows,
                            full_probs,
                            warm_probs,
                            hard_probs,
                            hard_prob=float(h),
                            full_accept_prob=float(fthr),
                            warm_accept_prob=float(wthr),
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
                if full_rate is None:
                    full_rate = 0.0
                if warm_rate is None:
                    warm_rate = 0.0
                total_rate = float(full_rate + warm_rate)
                if total_rate < float(min_total_transfer_rate):
                    continue
                if full_rate < float(min_accept_rate):
                    continue
                if float(sr) < float(min_success_delta):
                    continue
                if float(haz) < float(min_hazard_gain):
                    continue
                if float(neg) > float(max_negative_transfer_rate):
                    continue
                score = float(100.0 * sr + ret + 0.02 * haz - 5.0 * neg)
                if float(target_accept_rate) >= 0.0 and float(accept_rate_penalty) > 0.0:
                    score -= float(accept_rate_penalty) * abs(float(total_rate) - float(target_accept_rate))
                cand = {
                    "score": float(score),
                    "hard_prob_min": float(h),
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
            "min_accept_rate": float(min_accept_rate),
            "min_total_transfer_rate": float(min_total_transfer_rate),
            "target_accept_rate": float(target_accept_rate),
            "accept_rate_penalty": float(accept_rate_penalty),
            "min_success_delta": float(min_success_delta),
            "min_hazard_gain": float(min_hazard_gain),
            "max_negative_transfer_rate": float(max_negative_transfer_rate),
        },
        "aligned_splits": int(len(aligned)),
    }


def _calibrate_two_stage_policy(
    *,
    rows: Sequence[Dict[str, Any]],
    split_records: Sequence[Dict[str, Any]],
    min_accept_rate: float = 0.0,
    min_total_transfer_rate: float = 0.0,
    target_accept_rate: float = -1.0,
    accept_rate_penalty: float = 0.0,
    min_success_delta: float = -1e9,
    min_hazard_gain: float = -1e9,
    max_negative_transfer_rate: float = 1.0,
) -> Dict[str, Any]:
    if not split_records:
        return {"used": False}
    best: Optional[Dict[str, Any]] = None
    hard_grid = [-1.0, 0.45, 0.55, 0.65]
    accept_grid = [0.55, 0.60, 0.65, 0.70, 0.75]
    warm_grid = [0.35, 0.40, 0.45, 0.50, 0.55]
    for h in hard_grid:
        for a in accept_grid:
            for w in warm_grid:
                if float(w) > float(a):
                    continue
                evals: List[Dict[str, Any]] = []
                for rec in split_records:
                    test_idx = [int(i) for i in rec.get("test_idx", [])]
                    if not test_idx:
                        continue
                    eff = np.asarray(rec.get("effect_test_probs", []), dtype=np.float64)
                    hard = np.asarray(rec.get("hardness_test_probs", []), dtype=np.float64) if isinstance(rec.get("hardness_test_probs"), list) else None
                    test_rows = [rows[i] for i in test_idx]
                    evals.append(
                        _policy_eval_two_stage(
                            test_rows,
                            eff,
                            hard,
                            hard_prob=float(h),
                            accept_prob=float(a),
                            warmup_prob=float(w),
                        )
                    )
                if not evals:
                    continue
                sr = _mean(e["success_delta_mean"] for e in evals if isinstance(e.get("success_delta_mean"), (int, float)))
                ret = _mean(e["return_delta_mean"] for e in evals if isinstance(e.get("return_delta_mean"), (int, float)))
                haz = _mean(e["hazard_gain_mean"] for e in evals if isinstance(e.get("hazard_gain_mean"), (int, float)))
                neg = _mean(e["negative_transfer_rate_success"] for e in evals if isinstance(e.get("negative_transfer_rate_success"), (int, float)))
                acc_rate = _mean(e["accept_transfer_rate"] for e in evals if isinstance(e.get("accept_transfer_rate"), (int, float)))
                warm_rate = _mean(e["warmup_only_rate"] for e in evals if isinstance(e.get("warmup_only_rate"), (int, float)))
                if sr is None or ret is None or haz is None or neg is None:
                    continue
                if acc_rate is None:
                    acc_rate = 0.0
                if warm_rate is None:
                    warm_rate = 0.0
                total_transfer_rate = float(acc_rate + warm_rate)
                if float(acc_rate) < float(min_accept_rate):
                    continue
                if float(total_transfer_rate) < float(min_total_transfer_rate):
                    continue
                if float(sr) < float(min_success_delta):
                    continue
                if float(haz) < float(min_hazard_gain):
                    continue
                if float(neg) > float(max_negative_transfer_rate):
                    continue
                score = float(100.0 * sr + ret + 0.02 * haz - 5.0 * neg)
                if float(target_accept_rate) >= 0.0 and float(accept_rate_penalty) > 0.0:
                    score -= float(accept_rate_penalty) * abs(float(acc_rate) - float(target_accept_rate))
                cand = {
                    "score": score,
                    "hard_prob_min": float(h),
                    "accept_prob_min": float(a),
                    "warmup_prob_min": float(w),
                    "cv_policy_mean": {
                        "success_delta_mean": float(sr),
                        "return_delta_mean": float(ret),
                        "hazard_gain_mean": float(haz),
                        "negative_transfer_rate_success": float(neg),
                        "accept_transfer_rate": float(acc_rate),
                        "warmup_only_rate": float(warm_rate),
                        "total_transfer_rate": float(total_transfer_rate),
                    },
                }
                if best is None or float(cand["score"]) > float(best["score"]):
                    best = cand
    return {
        "used": bool(best),
        "best": best,
        "constraints": {
            "min_accept_rate": float(min_accept_rate),
            "min_total_transfer_rate": float(min_total_transfer_rate),
            "target_accept_rate": float(target_accept_rate),
            "accept_rate_penalty": float(accept_rate_penalty),
            "min_success_delta": float(min_success_delta),
            "min_hazard_gain": float(min_hazard_gain),
            "max_negative_transfer_rate": float(max_negative_transfer_rate),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_json", type=str, default="")
    ap.add_argument("--artifact_jsons", type=str, default="")
    ap.add_argument("--conditions", type=str, default="sf_transfer,sf_transfer_warmup")
    ap.add_argument("--feature_names", type=str, default="")
    ap.add_argument("--label_mode", type=str, default="success_only", choices=["success_only", "success_and_hazard_noninferior"])
    ap.add_argument("--min_success_delta", type=float, default=0.0)
    ap.add_argument("--min_hazard_gain", type=float, default=0.0)
    ap.add_argument("--test_frac", type=float, default=0.33)
    ap.add_argument("--split_seed", type=int, default=7)
    ap.add_argument("--cv_split_seeds", type=str, default="")
    ap.add_argument("--cv_num_splits", type=int, default=0)
    ap.add_argument("--cv_seed_start", type=int, default=7)
    ap.add_argument("--cv_split_mode", type=str, default="row", choices=["row", "artifact", "auto"])
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--positive_weight", type=float, default=2.0)
    ap.add_argument("--accept_prob", type=float, default=0.6)
    ap.add_argument("--warmup_prob", type=float, default=0.4)
    ap.add_argument("--train_hardness_aux", action="store_true")
    ap.add_argument("--hardness_scratch_eval_success_max", type=float, default=0.25)
    ap.add_argument("--hardness_feature_names", type=str, default="")
    ap.add_argument("--hardness_positive_weight", type=float, default=1.5)
    ap.add_argument("--hard_prob", type=float, default=-1.0)
    ap.add_argument("--calibrate_policy_from_cv", action="store_true")
    ap.add_argument("--calib_min_accept_rate", type=float, default=0.0)
    ap.add_argument("--calib_min_total_transfer_rate", type=float, default=0.0)
    ap.add_argument("--calib_target_accept_rate", type=float, default=-1.0)
    ap.add_argument("--calib_accept_rate_penalty", type=float, default=0.0)
    ap.add_argument("--calib_min_cv_success_delta", type=float, default=-1e9)
    ap.add_argument("--calib_min_cv_hazard_gain", type=float, default=-1e9)
    ap.add_argument("--calib_max_cv_negative_transfer_rate", type=float, default=1.0)
    ap.add_argument("--export_fit", type=str, default="full", choices=["full", "split"])
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    rows, artifact_paths = _load_artifact_rows(
        artifact_json=str(args.artifact_json or ""),
        artifact_jsons=str(args.artifact_jsons or ""),
    )

    feature_names = _parse_str_list(args.feature_names, default=DEFAULT_FEATURE_NAMES)
    conditions = _parse_str_list(args.conditions, default=["sf_transfer", "sf_transfer_warmup"])
    out: Dict[str, Any] = {
        "schema_version": "sf_adaptive_gate_model.v1",
        "model_type": "logistic_linear",
        "artifact_json": str(args.artifact_json),
        "artifact_jsons": artifact_paths,
        "num_training_rows": int(len(rows)),
        "labeling": {
            "label_mode": str(args.label_mode),
            "min_success_delta": float(args.min_success_delta),
            "min_hazard_gain": float(args.min_hazard_gain),
        },
        "split": {
            "test_frac": float(args.test_frac),
            "split_seed": int(args.split_seed),
            "cv_split_mode": str(args.cv_split_mode),
            "cv_split_seeds": _parse_int_list(args.cv_split_seeds, default=[]) if str(args.cv_split_seeds).strip() else [],
            "cv_num_splits": int(args.cv_num_splits),
            "cv_seed_start": int(args.cv_seed_start),
        },
        "optimizer": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "positive_weight": float(args.positive_weight),
        },
        "policy": {
            "hard_prob_min": float(args.hard_prob),
            "accept_prob_min": float(args.accept_prob),
            "warmup_prob_min": float(args.warmup_prob),
        },
        "triage_policy": {},
        "condition_models": {},
    }
    cv_records_by_condition: Dict[str, List[Dict[str, Any]]] = {}

    for cond_key in conditions:
        hardness_feature_names = _parse_str_list(args.hardness_feature_names, default=DEFAULT_HARDNESS_FEATURE_NAMES)
        X, y, seeds, kept_rows = _build_dataset(
            rows,
            cond_key=cond_key,
            feature_names=feature_names,
            label_mode=str(args.label_mode),
            min_success_delta=float(args.min_success_delta),
            min_hazard_gain=float(args.min_hazard_gain),
        )
        if X.shape[0] < 4:
            raise ValueError(f"Not enough rows to train model for {cond_key}.")
        train_idx, test_idx = _train_test_split_indices(X.shape[0], test_frac=float(args.test_frac), seed=int(args.split_seed))
        fit_split = _fit_logistic(
            X,
            y,
            train_idx=train_idx,
            lr=float(args.lr),
            l2=float(args.l2),
            epochs=int(args.epochs),
            positive_weight=float(args.positive_weight),
        )
        train_probs = _predict_probs(X[train_idx], fit_split)
        test_probs = _predict_probs(X[test_idx], fit_split) if test_idx.size > 0 else np.zeros((0,), dtype=np.float64)
        X_hard = _build_feature_matrix(rows, cond_key=cond_key, feature_names=hardness_feature_names)
        y_hard = np.asarray(
            [
                _label_hard_target(
                    r,
                    scratch_eval_success_max=float(args.hardness_scratch_eval_success_max),
                )
                for r in kept_rows
            ],
            dtype=np.float64,
        )
        hard_fit_split: Optional[Dict[str, Any]] = None
        hard_train_probs = np.zeros((len(train_idx),), dtype=np.float64)
        hard_test_probs = np.zeros((len(test_idx),), dtype=np.float64)
        if bool(args.train_hardness_aux):
            hard_fit_split = _fit_logistic(
                X_hard,
                y_hard,
                train_idx=train_idx,
                lr=float(args.lr),
                l2=float(args.l2),
                epochs=int(args.epochs),
                positive_weight=float(args.hardness_positive_weight),
            )
            hard_train_probs = _predict_probs(X_hard[train_idx], hard_fit_split)
            hard_test_probs = _predict_probs(X_hard[test_idx], hard_fit_split) if test_idx.size > 0 else np.zeros((0,), dtype=np.float64)

        cv_split_mode = str(args.cv_split_mode or "row").strip().lower()
        if cv_split_mode not in {"row", "artifact", "auto"}:
            cv_split_mode = "row"
        artifact_splits = _artifact_cv_splits(kept_rows) if cv_split_mode in {"artifact", "auto"} else []
        use_artifact_splits = bool(artifact_splits) and (cv_split_mode in {"artifact", "auto"})
        cv_split_seeds = (
            _parse_int_list(args.cv_split_seeds, default=[])
            if str(args.cv_split_seeds or "").strip()
            else [int(args.cv_seed_start) + i for i in range(max(0, int(args.cv_num_splits)))]
        )
        cv_records: List[Dict[str, Any]] = []
        cv_iter: List[Dict[str, Any]] = []
        if use_artifact_splits:
            for i_split, sp in enumerate(artifact_splits):
                cv_iter.append(
                    {
                        "split_id": int(i_split),
                        "split_kind": str(sp.get("split_kind", "artifact_holdout")),
                        "split_name": str(sp.get("split_name", "")),
                        "train_idx": sp["train_idx"],
                        "test_idx": sp["test_idx"],
                    }
                )
        else:
            for cv_seed in cv_split_seeds:
                cv_train_idx, cv_test_idx = _train_test_split_indices(X.shape[0], test_frac=float(args.test_frac), seed=int(cv_seed))
                cv_iter.append(
                    {
                        "split_id": int(cv_seed),
                        "split_kind": "row_random",
                        "split_name": f"seed_{int(cv_seed)}",
                        "train_idx": cv_train_idx,
                        "test_idx": cv_test_idx,
                    }
                )
        for cv_split in cv_iter:
            cv_train_idx = cv_split["train_idx"]
            cv_test_idx = cv_split["test_idx"]
            cv_fit = _fit_logistic(
                X,
                y,
                train_idx=cv_train_idx,
                lr=float(args.lr),
                l2=float(args.l2),
                epochs=int(args.epochs),
                positive_weight=float(args.positive_weight),
            )
            cv_test_probs = _predict_probs(X[cv_test_idx], cv_fit) if cv_test_idx.size > 0 else np.zeros((0,), dtype=np.float64)
            cv_hard_test_probs: Optional[np.ndarray] = None
            cv_hard_metrics: Dict[str, Any] = {"n": 0}
            if bool(args.train_hardness_aux):
                cv_hard_fit = _fit_logistic(
                    X_hard,
                    y_hard,
                    train_idx=cv_train_idx,
                    lr=float(args.lr),
                    l2=float(args.l2),
                    epochs=int(args.epochs),
                    positive_weight=float(args.hardness_positive_weight),
                )
                cv_hard_test_probs = _predict_probs(X_hard[cv_test_idx], cv_hard_fit) if cv_test_idx.size > 0 else np.zeros((0,), dtype=np.float64)
                cv_hard_metrics = (
                    _binary_metrics(y_hard[cv_test_idx], cv_hard_test_probs, threshold=0.5)
                    if cv_test_idx.size > 0
                    else {"n": 0}
                )
            cv_records.append(
                {
                    "split_seed": int(cv_split.get("split_id", 0)),
                    "split_kind": str(cv_split.get("split_kind", "row_random")),
                    "split_name": str(cv_split.get("split_name", "")),
                    "train_n": int(len(cv_train_idx)),
                    "test_n": int(len(cv_test_idx)),
                    "test_idx": [int(i) for i in cv_test_idx.tolist()],
                    "test_metrics": _binary_metrics(y[cv_test_idx], cv_test_probs, threshold=0.5)
                    if cv_test_idx.size > 0
                    else {"n": 0},
                    "hardness_test_metrics": cv_hard_metrics,
                    "effect_test_probs": [float(x) for x in cv_test_probs.tolist()],
                    "hardness_test_probs": [float(x) for x in cv_hard_test_probs.tolist()] if isinstance(cv_hard_test_probs, np.ndarray) else None,
                    "policy_eval_test": _policy_eval(
                        [kept_rows[int(i)] for i in cv_test_idx.tolist()],
                        cv_test_probs,
                        accept_prob=float(args.accept_prob),
                        warmup_prob=float(args.warmup_prob),
                    )
                    if cv_test_idx.size > 0
                    else {"num_rows": 0},
                    "policy_eval_test_two_stage": _policy_eval_two_stage(
                        [kept_rows[int(i)] for i in cv_test_idx.tolist()],
                        cv_test_probs,
                        cv_hard_test_probs,
                        hard_prob=float(args.hard_prob),
                        accept_prob=float(args.accept_prob),
                        warmup_prob=float(args.warmup_prob),
                    ) if cv_test_idx.size > 0 else {"num_rows": 0},
                }
            )

        fit_full = _fit_logistic(
            X,
            y,
            train_idx=np.arange(X.shape[0], dtype=np.int64),
            lr=float(args.lr),
            l2=float(args.l2),
            epochs=int(args.epochs),
            positive_weight=float(args.positive_weight),
        )
        export_fit = fit_full if str(args.export_fit) == "full" else fit_split
        full_probs_export = _predict_probs(X, export_fit)
        hard_fit_export: Optional[Dict[str, Any]] = None
        hard_full_probs_export: Optional[np.ndarray] = None
        if bool(args.train_hardness_aux):
            hard_fit_full = _fit_logistic(
                X_hard,
                y_hard,
                train_idx=np.arange(X_hard.shape[0], dtype=np.int64),
                lr=float(args.lr),
                l2=float(args.l2),
                epochs=int(args.epochs),
                positive_weight=float(args.hardness_positive_weight),
            )
            hard_fit_export = hard_fit_full if str(args.export_fit) == "full" else hard_fit_split
            if hard_fit_export is not None:
                hard_full_probs_export = _predict_probs(X_hard, hard_fit_export)

        policy_calibration = (
            _calibrate_two_stage_policy(
                rows=kept_rows,
                split_records=cv_records,
                min_accept_rate=float(args.calib_min_accept_rate),
                min_total_transfer_rate=float(args.calib_min_total_transfer_rate),
                target_accept_rate=float(args.calib_target_accept_rate),
                accept_rate_penalty=float(args.calib_accept_rate_penalty),
                min_success_delta=float(args.calib_min_cv_success_delta),
                min_hazard_gain=float(args.calib_min_cv_hazard_gain),
                max_negative_transfer_rate=float(args.calib_max_cv_negative_transfer_rate),
            )
            if bool(args.calibrate_policy_from_cv)
            else {"used": False}
        )
        calibrated = ((policy_calibration.get("best") or {}) if isinstance(policy_calibration, dict) else {}) if bool(args.calibrate_policy_from_cv) else {}
        eff_accept = float(calibrated.get("accept_prob_min", args.accept_prob)) if isinstance(calibrated, dict) and calibrated else float(args.accept_prob)
        eff_warm = float(calibrated.get("warmup_prob_min", args.warmup_prob)) if isinstance(calibrated, dict) and calibrated else float(args.warmup_prob)
        eff_hard = float(calibrated.get("hard_prob_min", args.hard_prob)) if isinstance(calibrated, dict) and calibrated else float(args.hard_prob)
        if bool(args.calibrate_policy_from_cv) and isinstance(calibrated, dict) and calibrated and "sf_transfer" in str(cond_key):
            out["policy"] = {
                "hard_prob_min": eff_hard,
                "accept_prob_min": eff_accept,
                "warmup_prob_min": eff_warm,
            }

        cond_model = {
            **_serialize_fit_model(export_fit, feature_names=feature_names),
            "export_fit_source": str(args.export_fit),
            "train_metrics": _binary_metrics(y[train_idx], train_probs, threshold=0.5),
            "test_metrics": _binary_metrics(y[test_idx], test_probs, threshold=0.5) if test_idx.size > 0 else {"n": 0},
            "hardness_train_metrics": (
                _binary_metrics(y_hard[train_idx], hard_train_probs, threshold=0.5)
                if bool(args.train_hardness_aux)
                else {"n": 0}
            ),
            "hardness_test_metrics": (
                _binary_metrics(y_hard[test_idx], hard_test_probs, threshold=0.5)
                if (bool(args.train_hardness_aux) and test_idx.size > 0)
                else {"n": 0}
            ),
            "policy_eval_train": _policy_eval(
                [kept_rows[int(i)] for i in train_idx.tolist()],
                train_probs,
                accept_prob=float(args.accept_prob),
                warmup_prob=float(args.warmup_prob),
            ),
            "policy_eval_test": _policy_eval(
                [kept_rows[int(i)] for i in test_idx.tolist()],
                test_probs,
                accept_prob=float(args.accept_prob),
                warmup_prob=float(args.warmup_prob),
            ) if test_idx.size > 0 else {"num_rows": 0},
            "policy_eval_test_two_stage": _policy_eval_two_stage(
                [kept_rows[int(i)] for i in test_idx.tolist()],
                test_probs,
                hard_test_probs if bool(args.train_hardness_aux) else None,
                hard_prob=eff_hard,
                accept_prob=eff_accept,
                warmup_prob=eff_warm,
            ) if test_idx.size > 0 else {"num_rows": 0},
            "policy_eval_full_export_model": _policy_eval(
                kept_rows,
                full_probs_export,
                accept_prob=float(args.accept_prob),
                warmup_prob=float(args.warmup_prob),
            ),
            "policy_eval_full_export_model_two_stage": _policy_eval_two_stage(
                kept_rows,
                full_probs_export,
                hard_full_probs_export if bool(args.train_hardness_aux) else None,
                hard_prob=eff_hard,
                accept_prob=eff_accept,
                warmup_prob=eff_warm,
            ),
            "label_positive_rate_full": float(np.mean(y)) if y.size else 0.0,
            "hardness_label_positive_rate_full": float(np.mean(y_hard)) if y_hard.size else 0.0,
            "train_seeds": [int(seeds[int(i)]) for i in train_idx.tolist()],
            "test_seeds": [int(seeds[int(i)]) for i in test_idx.tolist()],
            "cv": _cv_aggregate(cv_records),
            "policy_calibration": policy_calibration,
            "policy": {
                "hard_prob_min": float(eff_hard),
                "accept_prob_min": float(eff_accept),
                "warmup_prob_min": float(eff_warm),
            },
        }
        if bool(args.train_hardness_aux) and isinstance(hard_fit_export, dict):
            hard_block = _serialize_fit_model(hard_fit_export, feature_names=hardness_feature_names)
            hard_block["label_definition"] = "scratch_eval_success_rate <= threshold"
            hard_block["label_threshold"] = float(args.hardness_scratch_eval_success_max)
            cond_model["hardness_aux_model"] = hard_block
        out["condition_models"][cond_key] = cond_model
        cv_records_by_condition[str(cond_key)] = list(cv_records)

    if bool(args.calibrate_policy_from_cv) and "sf_transfer" in out["condition_models"] and "sf_transfer_warmup" in out["condition_models"]:
        joint = _calibrate_joint_triad_policy(
            rows=rows,
            full_split_records=cv_records_by_condition.get("sf_transfer", []),
            warm_split_records=cv_records_by_condition.get("sf_transfer_warmup", []),
            min_accept_rate=float(args.calib_min_accept_rate),
            min_total_transfer_rate=float(args.calib_min_total_transfer_rate),
            target_accept_rate=float(args.calib_target_accept_rate),
            accept_rate_penalty=float(args.calib_accept_rate_penalty),
            min_success_delta=float(args.calib_min_cv_success_delta),
            min_hazard_gain=float(args.calib_min_cv_hazard_gain),
            max_negative_transfer_rate=float(args.calib_max_cv_negative_transfer_rate),
        )
        out["joint_triad_calibration"] = joint
        best_joint = (joint.get("best") or {}) if isinstance(joint, dict) else {}
        if isinstance(best_joint, dict) and best_joint:
            triage_policy = {
                "hard_prob_min": float(best_joint.get("hard_prob_min", out.get("policy", {}).get("hard_prob_min", -1.0))),
                "full_accept_prob_min": float(best_joint.get("full_accept_prob_min", out.get("policy", {}).get("accept_prob_min", 0.6))),
                "warmup_accept_prob_min": float(best_joint.get("warmup_accept_prob_min", out.get("policy", {}).get("warmup_prob_min", 0.4))),
                "prefer_higher_probability": True,
            }
            out["triage_policy"] = triage_policy

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_json": str(args.artifact_json),
                "artifact_jsons": artifact_paths,
                "num_training_rows": int(len(rows)),
                "out_json": str(out_path),
                "conditions": conditions,
                "feature_count": len(feature_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
