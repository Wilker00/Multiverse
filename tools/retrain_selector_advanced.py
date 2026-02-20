"""
tools/retrain_selector_advanced.py

One-shot pipeline: build balanced selector data + train upgraded MicroSelector.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import torch

from tools.prepare_selector_data import prepare_data
from tools.train_selector import train


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--lessons_dir", type=str, default="lessons")
    ap.add_argument("--output_data", type=str, default=os.path.join("models", "tuning", "balanced_selector_data.pt"))
    ap.add_argument("--model_out", type=str, default=os.path.join("models", "micro_selector.pt"))
    ap.add_argument("--report_out", type=str, default=os.path.join("models", "tuning", "selector_retrain_report.json"))
    ap.add_argument("--reward_threshold", type=float, default=0.0)
    ap.add_argument("--max_runs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_embd", type=int, default=320)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(str(args.output_data)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(str(args.model_out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(str(args.report_out)) or ".", exist_ok=True)

    prepare_data(
        runs_root=str(args.runs_root),
        lessons_dir=str(args.lessons_dir),
        output_path=str(args.output_data),
        reward_threshold=float(args.reward_threshold),
        include_all_events_from_selected_episodes=True,
        max_runs=max(0, int(args.max_runs)),
        class_balance=True,
        class_balance_max_per_class=0,
        class_balance_seed=int(args.seed),
    )
    if not os.path.isfile(str(args.output_data)):
        raise RuntimeError(f"selector dataset was not produced: {args.output_data}")

    train(
        data_path=str(args.output_data),
        model_save_path=str(args.model_out),
        epochs=max(1, int(args.epochs)),
        batch_size=max(8, int(args.batch_size)),
        learning_rate=float(args.lr),
        n_embd=max(64, int(args.n_embd)),
        n_head=max(1, int(args.n_head)),
        n_layer=max(1, int(args.n_layer)),
        dropout=max(0.0, min(0.5, float(args.dropout))),
        use_interaction_tokens=True,
        use_cls_token=True,
        ff_mult=4,
        use_deep_stem=True,
        pooling="mean",
        weight_decay=1e-2,
        label_smoothing=0.05,
        grad_clip=1.0,
        val_split=0.10,
        patience=8,
        class_balance=True,
        seed=int(args.seed),
    )

    payload = torch.load(str(args.output_data), weights_only=False)
    if not isinstance(payload, dict):
        payload = {}
    class_counts_before: Dict[str, Any] = dict(payload.get("class_counts_before", {}) or {})
    class_counts_after: Dict[str, Any] = dict(payload.get("class_counts_after", {}) or {})
    report = {
        "dataset_path": str(args.output_data).replace("\\", "/"),
        "model_path": str(args.model_out).replace("\\", "/"),
        "samples": _safe_int(getattr(payload.get("labels"), "numel", lambda: 0)(), 0),
        "vocab_size": _safe_int(len(dict(payload.get("vocab", {}) or {})), 0),
        "state_dim": _safe_int(payload.get("state_dim", 0), 0),
        "goal_dim": _safe_int(payload.get("goal_dim", 0), 0),
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "architecture": {
            "n_embd": int(args.n_embd),
            "n_head": int(args.n_head),
            "n_layer": int(args.n_layer),
            "use_interaction_tokens": True,
            "use_cls_token": True,
            "use_deep_stem": True,
            "pooling": "mean",
        },
    }
    with open(str(args.report_out), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"selector_dataset={args.output_data}")
    print(f"selector_model={args.model_out}")
    print(f"selector_report={args.report_out}")


if __name__ == "__main__":
    main()
