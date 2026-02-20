"""
tools/train_confidence_monitor.py

Train a neural danger/confidence monitor from labeled run events.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from memory.embeddings import obs_to_universal_vector
from models.confidence_monitor import (
    ConfidenceMonitor,
    ConfidenceMonitorConfig,
    save_confidence_monitor,
)


def _safe_float(x: Any, default: float = 0.0) -> float:
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


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isfile(path):
        return
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


def _find_event_file(run_dir: str, preferred_name: str) -> Optional[str]:
    p = os.path.join(run_dir, preferred_name)
    if os.path.isfile(p):
        return p
    fallback = os.path.join(run_dir, "events.failure_tagged.jsonl")
    if os.path.isfile(fallback):
        return fallback
    fallback = os.path.join(run_dir, "events.jsonl")
    if os.path.isfile(fallback):
        return fallback
    return None


def _collect_run_dirs(*, run_dirs: List[str], runs_root: str, run_glob: str, include_runs_root: bool) -> List[str]:
    out: List[str] = []
    for rd in run_dirs:
        if os.path.isdir(rd):
            out.append(rd)
    if bool(include_runs_root) and os.path.isdir(runs_root):
        import glob

        for p in glob.glob(os.path.join(runs_root, run_glob)):
            if os.path.isdir(p):
                out.append(p)
    # stable de-dup
    uniq: List[str] = []
    seen = set()
    for p in out:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(ap)
    return uniq


def _extract_label(ev: Dict[str, Any], label_key: str, fallback_label_key: str) -> Optional[bool]:
    info = ev.get("info")
    info = info if isinstance(info, dict) else {}
    se = info.get("safe_executor")
    se = se if isinstance(se, dict) else {}
    for key in (
        label_key,
        fallback_label_key,
        "danger_label",
        "dangerous_outcome",
        "safety_violation",
        "severe_penalty",
    ):
        y = _safe_bool(ev.get(key), None)
        if y is not None:
            return y
        y = _safe_bool(info.get(key), None)
        if y is not None:
            return y
        y = _safe_bool(se.get(key), None)
        if y is not None:
            return y
    return None


def _build_feature(ev: Dict[str, Any], *, obs_dim: int) -> Optional[List[float]]:
    obs = ev.get("obs")
    try:
        obs_vec = obs_to_universal_vector(obs, dim=int(obs_dim))
    except Exception:
        return None
    if not obs_vec:
        return None
    action = ev.get("action")
    a = _safe_float(action, 0.0)
    a = max(-10.0, min(10.0, a)) / 10.0
    info = ev.get("info")
    info = info if isinstance(info, dict) else {}
    se = info.get("safe_executor")
    se = se if isinstance(se, dict) else {}
    se_danger = max(0.0, min(1.0, _safe_float(se.get("danger", 0.0), 0.0)))
    se_conf = max(0.0, min(1.0, _safe_float(se.get("confidence", 1.0), 1.0)))
    return list(obs_vec) + [float(a), float(se_danger), float(se_conf), 1.0]


def _load_dataset(
    *,
    run_dirs: List[str],
    events_file: str,
    label_key: str,
    fallback_label_key: str,
    obs_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    xs: List[List[float]] = []
    ys: List[float] = []
    used_runs = 0
    files_used: List[str] = []
    for run_dir in run_dirs:
        events_path = _find_event_file(run_dir, preferred_name=events_file)
        if not events_path:
            continue
        files_used.append(events_path.replace("\\", "/"))
        any_row = False
        for ev in _iter_jsonl(events_path):
            y = _extract_label(ev, label_key=label_key, fallback_label_key=fallback_label_key)
            if y is None:
                continue
            feat = _build_feature(ev, obs_dim=int(obs_dim))
            if feat is None:
                continue
            xs.append(feat)
            ys.append(1.0 if bool(y) else 0.0)
            any_row = True
        if any_row:
            used_runs += 1
    if not xs:
        raise RuntimeError("No labeled samples found for confidence monitor training.")
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    meta = {
        "runs_scanned": int(len(run_dirs)),
        "runs_used": int(used_runs),
        "files_used": files_used,
        "samples": int(len(xs)),
        "positive": int(sum(1 for v in ys if v > 0.5)),
        "negative": int(sum(1 for v in ys if v <= 0.5)),
    }
    return x, y, meta


def _split_indices(n: int, val_split: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(int(n), generator=g)
    val_n = int(max(1, round(float(n) * float(max(0.0, min(0.9, val_split))))))
    val_n = max(1, min(int(n - 1), int(val_n)))
    return idx[val_n:], idx[:val_n]


def _eval(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item())
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).to(dtype=torch.float32)
            tp += int(((pred == 1.0) & (yb == 1.0)).sum().item())
            fp += int(((pred == 1.0) & (yb == 0.0)).sum().item())
            tn += int(((pred == 0.0) & (yb == 0.0)).sum().item())
            fn += int(((pred == 0.0) & (yb == 1.0)).sum().item())
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2.0 * precision * recall) / max(1e-9, precision + recall))
    acc = float((tp + tn) / max(1, tp + tn + fp + fn))
    return {
        "loss": float(total_loss / max(1, len(loader))),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", action="append", default=None, help="repeatable explicit run_dir")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--run_glob", type=str, default="run_*")
    ap.add_argument(
        "--include_runs_root",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to include runs discovered from --runs_root/--run_glob in addition to explicit --run_dir values.",
    )
    ap.add_argument("--events_file", type=str, default="events.failure_tagged.jsonl")
    ap.add_argument("--label_key", type=str, default="danger_label")
    ap.add_argument("--fallback_label_key", type=str, default="dangerous_outcome")
    ap.add_argument("--obs_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--hidden_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_model", type=str, default=os.path.join("models", "confidence_monitor.pt"))
    ap.add_argument("--out_report", type=str, default=os.path.join("models", "tuning", "confidence_monitor_report.json"))
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    explicit_run_dirs = [str(x) for x in (args.run_dir or [])]
    include_runs_root = (
        bool(args.include_runs_root)
        if args.include_runs_root is not None
        else (len(explicit_run_dirs) == 0)
    )
    run_dirs = _collect_run_dirs(
        run_dirs=explicit_run_dirs,
        runs_root=str(args.runs_root),
        run_glob=str(args.run_glob),
        include_runs_root=bool(include_runs_root),
    )
    if not run_dirs:
        raise RuntimeError("No run directories found.")
    x, y, meta = _load_dataset(
        run_dirs=run_dirs,
        events_file=str(args.events_file),
        label_key=str(args.label_key),
        fallback_label_key=str(args.fallback_label_key),
        obs_dim=max(4, int(args.obs_dim)),
    )
    train_idx, val_idx = _split_indices(len(x), float(args.val_split), int(args.seed))

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=max(8, int(args.batch_size)), shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=max(8, int(args.batch_size)), shuffle=False)

    cfg = ConfidenceMonitorConfig(
        input_dim=int(x.shape[1]),
        hidden_dim=max(16, int(args.hidden_dim)),
        hidden_layers=max(1, int(args.hidden_layers)),
        dropout=max(0.0, min(0.7, float(args.dropout))),
    )
    model = ConfidenceMonitor(cfg)

    pos = float((y_train > 0.5).sum().item())
    neg = float((y_train <= 0.5).sum().item())
    pos_weight = torch.tensor([max(1e-6, neg / max(1.0, pos))], dtype=torch.float32).squeeze(0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=max(0.0, float(args.weight_decay)),
    )

    best_f1 = -1.0
    best_state = None
    for epoch in range(max(1, int(args.epochs))):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        val_metrics = _eval(model, val_loader, criterion)
        if float(val_metrics["f1"]) > float(best_f1):
            best_f1 = float(val_metrics["f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch+1:02d} val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.3f} val_acc={val_metrics['accuracy']:.3f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    train_metrics = _eval(model, train_loader, criterion)
    val_metrics = _eval(model, val_loader, criterion)

    out_model = str(args.out_model)
    out_dir = os.path.dirname(out_model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_confidence_monitor(
        model=model,
        path=out_model,
        train_config={
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "val_split": float(args.val_split),
            "seed": int(args.seed),
            "label_key": str(args.label_key),
            "fallback_label_key": str(args.fallback_label_key),
            "events_file": str(args.events_file),
            "obs_dim": int(args.obs_dim),
        },
        metrics={
            "train": train_metrics,
            "val": val_metrics,
            "dataset": meta,
        },
    )

    report = {
        "model_path": out_model.replace("\\", "/"),
        "dataset": meta,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_config": {
            "input_dim": int(cfg.input_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "hidden_layers": int(cfg.hidden_layers),
            "dropout": float(cfg.dropout),
        },
        "recommended_safe_cfg": {
            "confidence_model_path": out_model.replace("\\", "/"),
            "confidence_model_weight": 0.60,
            "confidence_model_obs_dim": int(args.obs_dim),
        },
    }

    out_report = str(args.out_report)
    report_dir = os.path.dirname(out_report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("")
    print(f"confidence_monitor_model={out_model}")
    print(f"confidence_monitor_report={out_report}")
    print(f"val_f1={float(val_metrics['f1']):.3f} val_acc={float(val_metrics['accuracy']):.3f}")


if __name__ == "__main__":
    main()
