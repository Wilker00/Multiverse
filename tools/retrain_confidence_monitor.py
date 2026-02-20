"""
tools/retrain_confidence_monitor.py

Calibrate SafeExecutor thresholds from labeled danger outcomes.
Expected workflow:
1) Run tools/failure_mode_classifier.py to produce events.failure_tagged.jsonl
2) Run this script to estimate practical threshold updates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: Optional[bool] = None) -> Optional[bool]:
    if x is None:
        return default
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on", "danger", "unsafe", "failure", "fail"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", "safe", "success"}:
        return False
    return default


def _f1(tp: int, fp: int, fn: int) -> float:
    p = float(tp) / float(max(1, tp + fp))
    r = float(tp) / float(max(1, tp + fn))
    if (p + r) <= 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _sweep_threshold(
    *,
    xs: List[float],
    ys: List[int],
    direction: str,
) -> Dict[str, float]:
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    if not xs or not ys or len(xs) != len(ys):
        return best

    lo = min(xs)
    hi = max(xs)
    if hi <= lo:
        return best
    steps = 60
    for i in range(steps + 1):
        thr = lo + (hi - lo) * (float(i) / float(steps))
        tp = fp = fn = 0
        for x, y in zip(xs, ys):
            pred = bool(x >= thr) if direction == "ge" else bool(x <= thr)
            if pred and y == 1:
                tp += 1
            elif pred and y == 0:
                fp += 1
            elif (not pred) and y == 1:
                fn += 1
        p = float(tp) / float(max(1, tp + fp))
        r = float(tp) / float(max(1, tp + fn))
        f1 = _f1(tp, fp, fn)
        if f1 > float(best["f1"]):
            best = {
                "threshold": float(thr),
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
            }
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--events_file", type=str, default="events.failure_tagged.jsonl")
    ap.add_argument("--label_key", type=str, default="danger_label")
    ap.add_argument("--fallback_label_key", type=str, default="dangerous_outcome")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--train_neural", action="store_true")
    ap.add_argument("--neural_out_model", type=str, default=os.path.join("models", "confidence_monitor.pt"))
    ap.add_argument(
        "--neural_out_report",
        type=str,
        default=os.path.join("models", "tuning", "confidence_monitor_report.json"),
    )
    args = ap.parse_args()

    path = os.path.join(args.run_dir, args.events_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"events file not found: {path}")

    danger_scores: List[float] = []
    confidence_scores: List[float] = []
    rewards: List[float] = []
    labels: List[int] = []

    for ev in _iter_jsonl(path):
        info = ev.get("info")
        info = info if isinstance(info, dict) else {}
        se = info.get("safe_executor")
        se = se if isinstance(se, dict) else {}

        y = _safe_bool(ev.get(args.label_key), None)
        if y is None:
            y = _safe_bool(info.get(args.label_key), None)
        if y is None:
            y = _safe_bool(se.get(args.label_key), None)
        if y is None:
            y = _safe_bool(se.get(args.fallback_label_key), None)
        if y is None:
            y = _safe_bool(info.get(args.fallback_label_key), None)
        if y is None:
            continue

        danger = _safe_float(se.get("danger"), None)
        conf = _safe_float(se.get("confidence"), None)
        rew = _safe_float(ev.get("reward"), None)
        if danger is None or conf is None or rew is None:
            continue

        danger_scores.append(float(danger))
        confidence_scores.append(float(conf))
        rewards.append(float(rew))
        labels.append(1 if bool(y) else 0)

    if not labels:
        raise RuntimeError("No labeled rows found with danger/confidence/reward signals.")

    best_danger = _sweep_threshold(xs=danger_scores, ys=labels, direction="ge")
    best_conf = _sweep_threshold(xs=confidence_scores, ys=labels, direction="le")
    best_reward = _sweep_threshold(xs=rewards, ys=labels, direction="le")

    out = {
        "run_dir": str(args.run_dir),
        "events_file": str(args.events_file),
        "rows_used": int(len(labels)),
        "danger_threshold_fit": dict(best_danger),
        "min_action_confidence_fit": dict(best_conf),
        "severe_reward_threshold_fit": dict(best_reward),
        "recommended_safe_executor_cfg": {
            "danger_threshold": float(best_danger["threshold"]),
            "min_action_confidence": float(best_conf["threshold"]),
            "severe_reward_threshold": float(best_reward["threshold"]),
        },
    }

    if bool(args.train_neural):
        cmd = [
            sys.executable,
            os.path.join("tools", "train_confidence_monitor.py"),
            "--run_dir",
            str(args.run_dir),
            "--events_file",
            str(args.events_file),
            "--label_key",
            str(args.label_key),
            "--fallback_label_key",
            str(args.fallback_label_key),
            "--out_model",
            str(args.neural_out_model),
            "--out_report",
            str(args.neural_out_report),
        ]
        subprocess.run(cmd, check=True)
        neural_report = {}
        if os.path.isfile(str(args.neural_out_report)):
            with open(str(args.neural_out_report), "r", encoding="utf-8") as f:
                neural_report = json.load(f)
        if isinstance(neural_report, dict):
            out["neural_confidence_monitor"] = neural_report
            rec = neural_report.get("recommended_safe_cfg")
            if isinstance(rec, dict):
                out["recommended_safe_executor_cfg"].update(
                    {
                        "confidence_model_path": str(rec.get("confidence_model_path", "")),
                        "confidence_model_weight": float(rec.get("confidence_model_weight", 0.60)),
                        "confidence_model_obs_dim": int(rec.get("confidence_model_obs_dim", 64)),
                    }
                )

    if args.out_json:
        out_dir = os.path.dirname(str(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote calibration report: {args.out_json}")

    print("Confidence monitor calibration")
    print(f"rows_used                 : {out['rows_used']}")
    print(f"danger_threshold          : {best_danger['threshold']:.4f} (f1={best_danger['f1']:.3f})")
    print(f"min_action_confidence     : {best_conf['threshold']:.4f} (f1={best_conf['f1']:.3f})")
    print(f"severe_reward_threshold   : {best_reward['threshold']:.4f} (f1={best_reward['f1']:.3f})")
    print("safe_cfg snippet:")
    print(
        json.dumps(
            out["recommended_safe_executor_cfg"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
