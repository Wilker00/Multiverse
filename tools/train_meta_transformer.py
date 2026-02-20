"""
tools/train_meta_transformer.py

Train a shared MetaTransformer policy from centralized memory events.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.embeddings import obs_to_vector
from models.meta_transformer import MetaTransformer


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_memories(memory_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(memory_dir, "memories.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Memory file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            rows.append(row)
    return rows


def _pad(vec: List[float], dim: int) -> List[float]:
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def _build_training_tensors(
    rows: List[Dict[str, Any]],
    *,
    history_len: int,
    gamma: float,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", int, int]:
    import torch

    # Determine global dimensions.
    state_dim = 1
    action_dim = 1
    vectors: List[Optional[List[float]]] = []
    actions: List[Optional[int]] = []
    for r in rows:
        try:
            vec = obs_to_vector(r.get("obs"))
        except Exception:
            vec = None
        vectors.append(vec)
        a = r.get("action")
        try:
            ai = int(a)
            if ai < 0:
                ai = None  # type: ignore[assignment]
        except Exception:
            ai = None
        actions.append(ai if isinstance(ai, int) else None)
        if vec is not None:
            state_dim = max(state_dim, len(vec))
        if isinstance(ai, int):
            action_dim = max(action_dim, ai + 1)

    # Group rows by (run_id, episode_id), preserving step order.
    grouped: Dict[Tuple[str, str], List[int]] = {}
    for i, r in enumerate(rows):
        run = str(r.get("run_id", ""))
        ep = str(r.get("episode_id", ""))
        grouped.setdefault((run, ep), []).append(i)
    for key in grouped:
        grouped[key].sort(key=lambda idx: _safe_int(rows[idx].get("step_idx", 0)))

    X_state: List[List[float]] = []
    X_hist: List[List[List[float]]] = []
    Y: List[int] = []
    Y_value: List[float] = []
    context_dim = state_dim + 2

    for _, indices in grouped.items():
        history: List[List[float]] = []
        ep_samples: List[Tuple[List[float], List[List[float]], int, float]] = []
        for idx in indices:
            vec = vectors[idx]
            act = actions[idx]
            if vec is None or act is None:
                continue

            state_vec = _pad(vec, state_dim)
            hist_window = history[-max(1, history_len) :]
            if len(hist_window) < history_len:
                pad_count = history_len - len(hist_window)
                hist_window = ([[0.0] * context_dim] * pad_count) + hist_window

            reward = _safe_float(rows[idx].get("reward", 0.0))
            ep_samples.append((state_vec, hist_window, int(act), float(reward)))
            history.append(state_vec + [float(act), reward])

        if not ep_samples:
            continue

        g = 0.0
        returns_rev: List[float] = []
        for _, _, _, reward in reversed(ep_samples):
            g = float(reward) + float(gamma) * float(g)
            returns_rev.append(float(g))
        returns_rev.reverse()

        for i, (state_vec, hist_window, act, _) in enumerate(ep_samples):
            X_state.append(state_vec)
            X_hist.append(hist_window)
            Y.append(int(act))
            Y_value.append(float(max(-1.0, min(1.0, returns_rev[i]))))

    if not X_state:
        raise RuntimeError("No usable training examples found in memories.")

    t_state = torch.tensor(X_state, dtype=torch.float32)
    t_hist = torch.tensor(X_hist, dtype=torch.float32)
    t_y = torch.tensor(Y, dtype=torch.long)
    t_v = torch.tensor(Y_value, dtype=torch.float32)
    return t_state, t_hist, t_y, t_v, state_dim, action_dim


def _build_generalized_samples(
    rows: List[Dict[str, Any]],
    *,
    history_len: int,
    gamma: float,
) -> Tuple[List[Dict[str, Any]], int]:
    # Determine action space from observed discrete actions.
    action_dim = 1
    actions: List[Optional[int]] = []
    for r in rows:
        a = r.get("action")
        try:
            ai = int(a)
            if ai < 0:
                ai = None  # type: ignore[assignment]
        except Exception:
            ai = None
        actions.append(ai if isinstance(ai, int) else None)
        if isinstance(ai, int):
            action_dim = max(action_dim, ai + 1)

    grouped: Dict[Tuple[str, str], List[int]] = {}
    for i, r in enumerate(rows):
        run = str(r.get("run_id", ""))
        ep = str(r.get("episode_id", ""))
        grouped.setdefault((run, ep), []).append(i)
    for key in grouped:
        grouped[key].sort(key=lambda idx: _safe_int(rows[idx].get("step_idx", 0)))

    samples: List[Dict[str, Any]] = []
    hlen = max(1, int(history_len))

    for _, indices in grouped.items():
        history: List[Tuple[Any, float, float]] = []
        ep_samples: List[Tuple[Any, List[Any], List[float], List[float], int, float]] = []
        for idx in indices:
            act = actions[idx]
            if act is None:
                continue
            obs = rows[idx].get("obs")
            reward = _safe_float(rows[idx].get("reward", 0.0))

            hist_window = history[-hlen:]
            if len(hist_window) < hlen:
                pad_count = hlen - len(hist_window)
                hist_window = ([(None, 0.0, 0.0)] * pad_count) + hist_window

            hist_obs = [x[0] for x in hist_window]
            hist_act = [float(x[1]) for x in hist_window]
            hist_rew = [float(x[2]) for x in hist_window]

            ep_samples.append((obs, hist_obs, hist_act, hist_rew, int(act), float(reward)))
            history.append((obs, float(act), float(reward)))

        if not ep_samples:
            continue

        g = 0.0
        returns_rev: List[float] = []
        for _, _, _, _, _, reward in reversed(ep_samples):
            g = float(reward) + float(gamma) * float(g)
            returns_rev.append(float(g))
        returns_rev.reverse()

        for i, (obs, hist_obs, hist_act, hist_rew, act, _) in enumerate(ep_samples):
            samples.append(
                {
                    "obs": obs,
                    "history_obs": hist_obs,
                    "history_actions": hist_act,
                    "history_rewards": hist_rew,
                    "action": int(act),
                    "value": float(max(-1.0, min(1.0, returns_rev[i]))),
                }
            )

    if not samples:
        raise RuntimeError("No usable training examples found in memories.")
    return samples, action_dim


def _generalized_batch(
    *,
    model: MetaTransformer,
    samples: Sequence[Dict[str, Any]],
    history_len: int,
):
    import torch

    hlen = max(1, int(history_len))
    if len(samples) == 0:
        raise ValueError("empty batch")

    device = next(model.parameters()).device
    n_embd = int(model.n_embd)
    batch_size = len(samples)
    hist_emb = torch.zeros((batch_size, hlen, n_embd), dtype=torch.float32, device=device)

    raw_obs = [s.get("obs") for s in samples]
    hist_obs_nonpad: List[Any] = []
    hist_obs_pos: List[Tuple[int, int]] = []
    hist_actions_rows: List[List[float]] = []
    hist_rewards_rows: List[List[float]] = []
    y: List[int] = []
    v: List[float] = []

    for b_idx, s in enumerate(samples):
        h_obs = list(s.get("history_obs") or [])
        h_act = [float(x) for x in (s.get("history_actions") or [])]
        h_rew = [float(x) for x in (s.get("history_rewards") or [])]

        if len(h_obs) < hlen:
            pad = hlen - len(h_obs)
            h_obs = ([None] * pad) + h_obs
            h_act = ([0.0] * pad) + h_act
            h_rew = ([0.0] * pad) + h_rew
        else:
            h_obs = h_obs[-hlen:]
            h_act = h_act[-hlen:]
            h_rew = h_rew[-hlen:]

        for t_idx, h in enumerate(h_obs):
            if h is None:
                continue
            hist_obs_nonpad.append(h)
            hist_obs_pos.append((b_idx, t_idx))
        hist_actions_rows.append(h_act)
        hist_rewards_rows.append(h_rew)
        y.append(int(s.get("action", 0)))
        v.append(float(s.get("value", 0.0)))

    if hist_obs_nonpad:
        with torch.set_grad_enabled(model.training):
            hist_flat_emb = model.input_encoder(hist_obs_nonpad)  # type: ignore[union-attr]
        for i, (b_idx, t_idx) in enumerate(hist_obs_pos):
            hist_emb[b_idx, t_idx, :] = hist_flat_emb[i]

    t_hist_actions = torch.tensor(hist_actions_rows, dtype=torch.float32, device=device).unsqueeze(-1)
    t_hist_rewards = torch.tensor(hist_rewards_rows, dtype=torch.float32, device=device).unsqueeze(-1)
    t_hist = torch.cat([hist_emb, t_hist_actions, t_hist_rewards], dim=-1)
    t_y = torch.tensor(y, dtype=torch.long, device=device)
    t_v = torch.tensor(v, dtype=torch.float32, device=device)
    return raw_obs, t_hist, t_y, t_v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_dir", type=str, default="central_memory")
    ap.add_argument("--out_path", type=str, default=os.path.join("models", "meta_transformer.pt"))
    ap.add_argument("--history_len", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--value_loss_weight", type=float, default=0.5)
    ap.add_argument("--stage_label", type=str, default="policy_value")
    ap.add_argument("--use_generalized_input", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--save_versioned", action="store_true")
    ap.add_argument("--versioned_dir", type=str, default=os.path.join("models", "meta_transformer_versions"))
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = _load_memories(args.memory_dir)
    use_generalized = bool(args.use_generalized_input)
    history_len = max(1, int(args.history_len))
    gamma = max(0.0, min(1.0, float(args.gamma)))

    state_dim = 1
    action_dim = 1
    t_state = None
    t_hist = None
    t_y = None
    t_v = None
    g_samples: List[Dict[str, Any]] = []

    if use_generalized:
        g_samples, action_dim = _build_generalized_samples(
            rows,
            history_len=history_len,
            gamma=gamma,
        )
        n = len(g_samples)
    else:
        t_state, t_hist, t_y, t_v, state_dim, action_dim = _build_training_tensors(
            rows,
            history_len=history_len,
            gamma=gamma,
        )
        n = int(t_state.shape[0])

    idx = list(range(n))
    random.shuffle(idx)

    val_n = int(max(1, round(float(args.val_split) * n))) if n > 10 else 0
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:] if val_n > 0 else idx

    model = MetaTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        n_embd=int(args.n_embd),
        context_layers=1,
        dropout=float(args.dropout),
        use_generalized_input=use_generalized,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    def _iter_batches(indices: List[int], batch_size: int):
        for i in range(0, len(indices), batch_size):
            chunk = indices[i : i + batch_size]
            if use_generalized:
                yield [g_samples[j] for j in chunk]
            else:
                assert t_state is not None and t_hist is not None and t_y is not None and t_v is not None
                yield (
                    t_state[chunk],
                    t_hist[chunk],
                    t_y[chunk],
                    t_v[chunk],
                )

    best_val = None
    best_state = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        tr_loss = 0.0
        tr_count = 0
        for batch in _iter_batches(tr_idx, max(1, int(args.batch_size))):
            if use_generalized:
                assert isinstance(batch, list)
                raw_obs, hb, yb, vb = _generalized_batch(model=model, samples=batch, history_len=history_len)
                out = model.forward_policy_value(state=None, recent_history=hb, raw_obs=raw_obs)
                batch_n = len(batch)
            else:
                xb, hb, yb, vb = batch
                out = model.forward_policy_value(xb, hb)
                batch_n = int(xb.shape[0])
            logits = out["logits"]
            value = out["value"]
            policy_loss = criterion(logits, yb)
            value_loss = torch.nn.functional.mse_loss(value, vb)
            loss = policy_loss + (float(args.value_loss_weight) * value_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()) * int(batch_n)
            tr_count += int(batch_n)
        tr_loss = tr_loss / float(max(1, tr_count))

        val_loss = tr_loss
        val_acc = 0.0
        if val_idx:
            model.eval()
            with torch.no_grad():
                if use_generalized:
                    val_batch = [g_samples[j] for j in val_idx]
                    raw_obs, hb, yb, vb = _generalized_batch(
                        model=model,
                        samples=val_batch,
                        history_len=history_len,
                    )
                    out = model.forward_policy_value(state=None, recent_history=hb, raw_obs=raw_obs)
                    logits = out["logits"]
                    value = out["value"]
                    policy_loss = criterion(logits, yb)
                    value_loss = torch.nn.functional.mse_loss(value, vb)
                    loss = policy_loss + (float(args.value_loss_weight) * value_loss)
                    pred = torch.argmax(logits, dim=1)
                    acc = (pred == yb).float().mean()
                    val_loss = float(loss.item())
                    val_acc = float(acc.item())
                else:
                    assert t_state is not None and t_hist is not None and t_y is not None and t_v is not None
                    out = model.forward_policy_value(t_state[val_idx], t_hist[val_idx])
                    logits = out["logits"]
                    value = out["value"]
                    policy_loss = criterion(logits, t_y[val_idx])
                    value_loss = torch.nn.functional.mse_loss(value, t_v[val_idx])
                    loss = policy_loss + (float(args.value_loss_weight) * value_loss)
                    pred = torch.argmax(logits, dim=1)
                    acc = (pred == t_y[val_idx]).float().mean()
                    val_loss = float(loss.item())
                    val_acc = float(acc.item())

            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch={epoch} train_loss={tr_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "history_len": int(max(1, int(args.history_len))),
        "memory_dir": str(args.memory_dir),
        "training_stage": str(args.stage_label or "policy_value"),
        "value_loss_weight": float(args.value_loss_weight),
        "gamma": float(args.gamma),
    }
    torch.save(payload, args.out_path)
    versioned_path = ""
    if bool(args.save_versioned):
        os.makedirs(args.versioned_dir, exist_ok=True)
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = f"meta_transformer_{str(args.stage_label or 'policy_value').strip().lower()}_{ts}.pt"
        versioned_path = os.path.join(args.versioned_dir, base)
        torch.save(payload, versioned_path)
    print("MetaTransformer training complete")
    print(f"examples   : {n}")
    if use_generalized:
        print("state_dim  : generalized")
    else:
        print(f"state_dim  : {state_dim}")
    print(f"action_dim : {action_dim}")
    print(f"saved_to   : {args.out_path}")
    if versioned_path:
        print(f"versioned  : {versioned_path}")


if __name__ == "__main__":
    main()
