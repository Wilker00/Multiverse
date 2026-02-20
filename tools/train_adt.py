"""
tools/train_adt.py

Train Decision Transformer checkpoints from prepared ADT tensors.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from models.decision_transformer import (
    DecisionTransformer,
    DecisionTransformerConfig,
    load_decision_transformer_checkpoint,
)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _load_dataset(path: str) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("dataset payload must be a dict")
    required = ["states", "returns_to_go", "prev_actions", "actions", "timesteps", "attention_mask", "meta"]
    for key in required:
        if key not in payload:
            raise ValueError(f"dataset missing key: {key}")
    return payload


def _batch_indices(indices: List[int], batch_size: int):
    bsz = max(1, int(batch_size))
    for i in range(0, len(indices), bsz):
        yield indices[i : i + bsz]


def _compute_loss_and_acc(
    *,
    model: DecisionTransformer,
    states: torch.Tensor,
    returns_to_go: torch.Tensor,
    prev_actions: torch.Tensor,
    timesteps: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    logits = model(
        states=states,
        returns_to_go=returns_to_go,
        prev_actions=prev_actions,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    action_dim = logits.shape[-1]
    loss = F.cross_entropy(
        logits.reshape(-1, action_dim),
        targets.reshape(-1),
        weight=class_weights,
        ignore_index=-100,
    )

    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        valid = targets != -100
        correct = (pred == targets) & valid
        n_valid = int(valid.sum().item())
        acc = float(correct.sum().item() / max(1, n_valid))
    return loss, acc


def _build_class_weights(
    *,
    actions: torch.Tensor,
    action_dim: int,
    mode: str,
    min_count: int,
    max_weight: float,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """
    Build class weights for imbalanced discrete action targets.
    """
    valid = actions[actions != -100]
    counts = torch.bincount(valid, minlength=max(1, int(action_dim))).float()
    total = float(counts.sum().item())
    counts_list = [int(x) for x in counts.tolist()]
    nonzero = [int(x) for x in counts_list if int(x) > 0]
    min_nonzero = min(nonzero) if nonzero else 0
    max_nonzero = max(nonzero) if nonzero else 0
    imbalance_ratio = float(max_nonzero / max(1, min_nonzero))

    mode_in = str(mode or "none").strip().lower()
    if mode_in == "auto":
        mode_eff = "inverse_sqrt" if imbalance_ratio >= 3.0 else "none"
    else:
        mode_eff = mode_in

    meta: Dict[str, Any] = {
        "class_weight_mode_requested": str(mode_in),
        "class_weight_mode_effective": str(mode_eff),
        "action_counts": counts_list,
        "imbalance_ratio": float(imbalance_ratio),
    }

    if mode_eff not in ("inverse", "inverse_sqrt"):
        return None, meta

    cnt = counts.clone()
    cnt = torch.clamp(cnt, min=max(1.0, float(min_count)))
    if mode_eff == "inverse":
        w = 1.0 / cnt
    else:
        w = 1.0 / torch.sqrt(cnt)

    # Avoid over-weighting classes that never appear in this dataset.
    absent = counts <= 0
    if bool(absent.any()):
        w[absent] = 0.0

    present = w > 0
    if bool(present.any()):
        present_mean = float(w[present].mean().item())
        if present_mean > 0.0:
            w[present] = w[present] / present_mean

    if float(max_weight) > 0.0:
        w = torch.clamp(w, min=0.0, max=float(max_weight))
        present = w > 0
        if bool(present.any()):
            present_mean = float(w[present].mean().item())
            if present_mean > 0.0:
                w[present] = w[present] / present_mean

    meta["class_weights"] = [float(x) for x in w.tolist()]
    return w.float(), meta


def train_adt(
    *,
    dataset_path: str,
    out_path: str,
    init_model_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    val_split: float,
    seed: int,
    d_model: int,
    n_head: int,
    n_layer: int,
    dropout: float,
    max_timestep: int,
    device: str,
    class_weight_mode: str = "auto",
    class_weight_min_count: int = 1,
    class_weight_max: float = 5.0,
) -> Dict[str, Any]:
    payload = _load_dataset(dataset_path)
    meta = payload["meta"]
    if not isinstance(meta, dict):
        raise ValueError("dataset meta must be a dict")

    states = payload["states"].float()
    rtg = payload["returns_to_go"].float()
    prev_actions = payload["prev_actions"].long()
    actions = payload["actions"].long()
    timesteps = payload["timesteps"].long()
    attention_mask = payload["attention_mask"].float()

    n = int(states.shape[0])
    if n <= 0:
        raise RuntimeError("empty ADT dataset")

    state_dim = _safe_int(meta.get("state_dim"), int(states.shape[-1]))
    action_dim = _safe_int(meta.get("action_dim"), 0)
    context_len = _safe_int(meta.get("context_len"), int(states.shape[1]))
    bos_token_id = _safe_int(meta.get("bos_token_id"), action_dim)
    if action_dim <= 0:
        raise RuntimeError("invalid action_dim in dataset meta")

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    idx = list(range(n))
    random.shuffle(idx)

    vs = max(0.0, min(0.5, float(val_split)))
    val_n = int(round(float(n) * vs))
    if val_n <= 0 and n >= 20:
        val_n = 1
    if val_n >= n:
        val_n = max(0, n - 1)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    if not train_idx:
        raise RuntimeError("no training samples after split")

    if str(device).strip().lower() == "auto":
        runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        runtime_device = torch.device(str(device))

    init_path = str(init_model_path or "").strip()
    cfg = DecisionTransformerConfig(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        context_len=int(context_len),
        d_model=int(d_model),
        n_head=int(n_head),
        n_layer=int(n_layer),
        dropout=float(dropout),
        max_timestep=max(1, int(max_timestep)),
        bos_token_id=int(bos_token_id),
    )
    if init_path:
        if not os.path.isfile(init_path):
            raise FileNotFoundError(f"init_model_path not found: {init_path}")
        model_loaded, _ = load_decision_transformer_checkpoint(init_path, map_location=runtime_device)
        loaded_cfg = model_loaded.get_config()
        mismatch_keys = ("state_dim", "action_dim", "context_len", "bos_token_id")
        for k in mismatch_keys:
            if int(_safe_int(loaded_cfg.get(k), -1)) != int(_safe_int(cfg.__dict__.get(k), -2)):
                raise ValueError(
                    f"init checkpoint incompatible on {k}: "
                    f"{loaded_cfg.get(k)} != {cfg.__dict__.get(k)}"
                )
        # Keep architecture identical to the warm-start checkpoint.
        cfg = DecisionTransformerConfig(**loaded_cfg)
        model = DecisionTransformer(cfg)
        model.load_state_dict(model_loaded.state_dict())
    else:
        model = DecisionTransformer(cfg)
    model = model.to(runtime_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    class_weights_cpu, class_weight_meta = _build_class_weights(
        actions=actions,
        action_dim=int(action_dim),
        mode=str(class_weight_mode),
        min_count=max(1, int(class_weight_min_count)),
        max_weight=max(0.0, float(class_weight_max)),
    )
    class_weights_runtime = None
    if isinstance(class_weights_cpu, torch.Tensor):
        class_weights_runtime = class_weights_cpu.to(runtime_device)

    best_val_loss = None
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_acc_sum = 0.0
        tr_batches = 0
        for bi in _batch_indices(train_idx, int(batch_size)):
            bi_t = torch.tensor(bi, dtype=torch.long)
            batch_states = states[bi_t].to(runtime_device)
            batch_rtg = rtg[bi_t].to(runtime_device)
            batch_prev = prev_actions[bi_t].to(runtime_device)
            batch_t = timesteps[bi_t].to(runtime_device)
            batch_mask = attention_mask[bi_t].to(runtime_device)
            batch_y = actions[bi_t].to(runtime_device)

            loss, acc = _compute_loss_and_acc(
                model=model,
                states=batch_states,
                returns_to_go=batch_rtg,
                prev_actions=batch_prev,
                timesteps=batch_t,
                attention_mask=batch_mask,
                targets=batch_y,
                class_weights=class_weights_runtime,
            )
            optimizer.zero_grad()
            loss.backward()
            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

            tr_loss_sum += float(loss.item())
            tr_acc_sum += float(acc)
            tr_batches += 1

        tr_loss = float(tr_loss_sum / max(1, tr_batches))
        tr_acc = float(tr_acc_sum / max(1, tr_batches))

        val_loss = tr_loss
        val_acc = tr_acc
        if val_idx:
            model.eval()
            with torch.no_grad():
                va_loss_sum = 0.0
                va_acc_sum = 0.0
                va_batches = 0
                for bi in _batch_indices(val_idx, int(batch_size)):
                    bi_t = torch.tensor(bi, dtype=torch.long)
                    batch_states = states[bi_t].to(runtime_device)
                    batch_rtg = rtg[bi_t].to(runtime_device)
                    batch_prev = prev_actions[bi_t].to(runtime_device)
                    batch_t = timesteps[bi_t].to(runtime_device)
                    batch_mask = attention_mask[bi_t].to(runtime_device)
                    batch_y = actions[bi_t].to(runtime_device)
                    loss, acc = _compute_loss_and_acc(
                        model=model,
                        states=batch_states,
                        returns_to_go=batch_rtg,
                        prev_actions=batch_prev,
                        timesteps=batch_t,
                        attention_mask=batch_mask,
                        targets=batch_y,
                        class_weights=class_weights_runtime,
                    )
                    va_loss_sum += float(loss.item())
                    va_acc_sum += float(acc)
                    va_batches += 1
                val_loss = float(va_loss_sum / max(1, va_batches))
                val_acc = float(va_acc_sum / max(1, va_batches))

        if best_val_loss is None or val_loss < float(best_val_loss):
            best_val_loss = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
        print(
            f"epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "dataset_path": str(dataset_path).replace("\\", "/"),
        "init_model_path": str(init_path).replace("\\", "/"),
        "dataset_meta": dict(meta),
        "class_weighting": dict(class_weight_meta),
        "history": list(history),
        "best_val_loss": float(best_val_loss if best_val_loss is not None else history[-1]["val_loss"]),
    }
    torch.save(ckpt, out_path)
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, default=os.path.join("models", "adt_data.pt"))
    ap.add_argument("--out_path", type=str, default=os.path.join("models", "decision_transformer.pt"))
    ap.add_argument("--init_model_path", type=str, default="")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_timestep", type=int, default=4096)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--class_weight_mode",
        type=str,
        default="auto",
        choices=["none", "auto", "inverse_sqrt", "inverse"],
        help="Class weighting mode for imbalanced action labels.",
    )
    ap.add_argument("--class_weight_min_count", type=int, default=1)
    ap.add_argument("--class_weight_max", type=float, default=5.0)
    args = ap.parse_args()

    ckpt = train_adt(
        dataset_path=str(args.dataset_path),
        out_path=str(args.out_path),
        init_model_path=str(args.init_model_path),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        lr=max(1e-6, float(args.lr)),
        weight_decay=max(0.0, float(args.weight_decay)),
        grad_clip=max(0.0, float(args.grad_clip)),
        val_split=max(0.0, min(0.5, float(args.val_split))),
        seed=int(args.seed),
        d_model=max(16, int(args.d_model)),
        n_head=max(1, int(args.n_head)),
        n_layer=max(1, int(args.n_layer)),
        dropout=max(0.0, min(0.9, float(args.dropout))),
        max_timestep=max(1, int(args.max_timestep)),
        device=str(args.device),
        class_weight_mode=str(args.class_weight_mode),
        class_weight_min_count=max(1, int(args.class_weight_min_count)),
        class_weight_max=max(0.0, float(args.class_weight_max)),
    )
    history = ckpt.get("history", [])
    last = history[-1] if isinstance(history, list) and history else {}
    print("ADT training complete")
    print(f"out_path     : {args.out_path}")
    print(f"best_val_loss: {float(ckpt.get('best_val_loss', 0.0)):.4f}")
    if isinstance(last, dict):
        print(f"final_val_acc: {float(_safe_float(last.get('val_acc', 0.0), 0.0)):.3f}")


if __name__ == "__main__":
    main()
