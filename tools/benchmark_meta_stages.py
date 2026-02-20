"""
tools/benchmark_meta_stages.py

Compare two MetaTransformer checkpoints (for example policy-only vs policy+value)
on centralized memory events using action accuracy and value MSE.
"""

from __future__ import annotations

import argparse
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


def _load_rows(memory_dir: str) -> List[Dict[str, Any]]:
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
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _pad(vec: List[float], dim: int) -> List[float]:
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def _build_eval_tensors(
    rows: List[Dict[str, Any]],
    *,
    state_dim: int,
    action_dim: int,
    history_len: int,
    gamma: float,
):
    import torch

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((str(r.get("run_id", "")), str(r.get("episode_id", ""))), []).append(r)
    for k in grouped:
        grouped[k].sort(key=lambda x: _safe_int(x.get("step_idx", 0), 0))

    Xs: List[List[float]] = []
    Xh: List[List[List[float]]] = []
    Ya: List[int] = []
    Yv: List[float] = []
    context_dim = state_dim + 2

    for _, ep_rows in grouped.items():
        ep_samples: List[Tuple[List[float], List[List[float]], int, float]] = []
        hist: List[List[float]] = []
        for row in ep_rows:
            try:
                a = int(row.get("action"))
            except Exception:
                continue
            if a < 0 or a >= action_dim:
                continue
            try:
                svec = _pad(obs_to_vector(row.get("obs")), state_dim)
            except Exception:
                continue
            hw = hist[-history_len:]
            if len(hw) < history_len:
                hw = ([[0.0] * context_dim] * (history_len - len(hw))) + hw
            r = _safe_float(row.get("reward", 0.0), 0.0)
            ep_samples.append((svec, hw, a, r))
            hist.append(svec + [float(a), float(r)])

        if not ep_samples:
            continue
        g = 0.0
        rets_rev: List[float] = []
        for _, _, _, r in reversed(ep_samples):
            g = float(r) + float(gamma) * float(g)
            rets_rev.append(float(max(-1.0, min(1.0, g))))
        rets_rev.reverse()

        for i, (svec, hw, a, _) in enumerate(ep_samples):
            Xs.append(svec)
            Xh.append(hw)
            Ya.append(int(a))
            Yv.append(float(rets_rev[i]))

    if not Xs:
        raise RuntimeError("No usable rows for evaluation.")

    return (
        torch.tensor(Xs, dtype=torch.float32),
        torch.tensor(Xh, dtype=torch.float32),
        torch.tensor(Ya, dtype=torch.long),
        torch.tensor(Yv, dtype=torch.float32),
    )


def _build_generalized_eval_samples(
    rows: List[Dict[str, Any]],
    *,
    action_dim: int,
    history_len: int,
    gamma: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((str(r.get("run_id", "")), str(r.get("episode_id", ""))), []).append(r)
    for k in grouped:
        grouped[k].sort(key=lambda x: _safe_int(x.get("step_idx", 0), 0))

    samples: List[Dict[str, Any]] = []
    hlen = max(1, int(history_len))
    for _, ep_rows in grouped.items():
        ep_samples: List[Tuple[Any, List[Any], List[float], List[float], int, float]] = []
        hist: List[Tuple[Any, float, float]] = []
        for row in ep_rows:
            try:
                a = int(row.get("action"))
            except Exception:
                continue
            if a < 0 or a >= action_dim:
                continue
            obs = row.get("obs")
            r = _safe_float(row.get("reward", 0.0), 0.0)

            hw = hist[-hlen:]
            if len(hw) < hlen:
                hw = ([(None, 0.0, 0.0)] * (hlen - len(hw))) + hw

            hist_obs = [x[0] for x in hw]
            hist_act = [float(x[1]) for x in hw]
            hist_rew = [float(x[2]) for x in hw]
            ep_samples.append((obs, hist_obs, hist_act, hist_rew, int(a), float(r)))
            hist.append((obs, float(a), float(r)))

        if not ep_samples:
            continue
        g = 0.0
        rets_rev: List[float] = []
        for _, _, _, _, _, r in reversed(ep_samples):
            g = float(r) + float(gamma) * float(g)
            rets_rev.append(float(max(-1.0, min(1.0, g))))
        rets_rev.reverse()

        for i, (obs, hist_obs, hist_act, hist_rew, a, _) in enumerate(ep_samples):
            samples.append(
                {
                    "obs": obs,
                    "history_obs": hist_obs,
                    "history_actions": hist_act,
                    "history_rewards": hist_rew,
                    "action": int(a),
                    "value": float(rets_rev[i]),
                }
            )
    if not samples:
        raise RuntimeError("No usable rows for evaluation.")
    return samples


def _generalized_batch(
    *,
    model: MetaTransformer,
    samples: Sequence[Dict[str, Any]],
    history_len: int,
):
    import torch

    hlen = max(1, int(history_len))
    device = next(model.parameters()).device
    n_embd = int(model.n_embd)
    bsz = len(samples)
    hist_emb = torch.zeros((bsz, hlen, n_embd), dtype=torch.float32, device=device)

    raw_obs = [s.get("obs") for s in samples]
    hist_obs_nonpad: List[Any] = []
    hist_obs_pos: List[Tuple[int, int]] = []
    hist_actions_rows: List[List[float]] = []
    hist_rewards_rows: List[List[float]] = []
    y = []
    v = []
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
        with torch.no_grad():
            hist_flat_emb = model.input_encoder(hist_obs_nonpad)  # type: ignore[union-attr]
        for i, (b_idx, t_idx) in enumerate(hist_obs_pos):
            hist_emb[b_idx, t_idx, :] = hist_flat_emb[i]

    t_hist_actions = torch.tensor(hist_actions_rows, dtype=torch.float32, device=device).unsqueeze(-1)
    t_hist_rewards = torch.tensor(hist_rewards_rows, dtype=torch.float32, device=device).unsqueeze(-1)
    t_hist = torch.cat([hist_emb, t_hist_actions, t_hist_rewards], dim=-1)
    t_y = torch.tensor(y, dtype=torch.long, device=device)
    t_v = torch.tensor(v, dtype=torch.float32, device=device)
    return raw_obs, t_hist, t_y, t_v


def _evaluate_generalized(
    *,
    model: MetaTransformer,
    samples: List[Dict[str, Any]],
    history_len: int,
    batch_size: int = 512,
) -> Tuple[int, float, float]:
    import torch

    total = 0
    correct = 0
    value_sqerr = 0.0
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(samples), bs):
            batch = samples[i : i + bs]
            raw_obs, t_hist, t_y, t_v = _generalized_batch(
                model=model,
                samples=batch,
                history_len=history_len,
            )
            out = model.forward_policy_value(state=None, recent_history=t_hist, raw_obs=raw_obs)
            logits = out["logits"]
            value = out["value"]
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == t_y).sum().item())
            value_sqerr += float(torch.sum((value - t_v) ** 2).item())
            total += int(t_y.shape[0])
    if total <= 0:
        raise RuntimeError("No usable rows for evaluation.")
    return total, float(correct) / float(total), float(value_sqerr) / float(total)


def _evaluate_checkpoint(path: str, rows: List[Dict[str, Any]], gamma: float, eval_limit: int) -> Dict[str, Any]:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid checkpoint config: {path}")
    state_dim = _safe_int(cfg.get("state_dim", 0), 0)
    action_dim = _safe_int(cfg.get("action_dim", 0), 0)
    n_embd = _safe_int(cfg.get("n_embd", 256), 256)
    use_generalized_input = bool(_safe_int(cfg.get("use_generalized_input", 0), 0))
    history_len = _safe_int(ckpt.get("history_len", 6), 6)
    if action_dim <= 0:
        raise ValueError(f"Checkpoint missing state/action dims: {path}")
    if not use_generalized_input and state_dim <= 0:
        raise ValueError(f"Checkpoint missing state_dim: {path}")

    model = MetaTransformer(
        state_dim=max(1, state_dim),
        action_dim=action_dim,
        n_embd=n_embd,
        use_generalized_input=use_generalized_input,
    )
    model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    model.eval()

    sample_rows = list(rows)
    if eval_limit > 0 and len(sample_rows) > eval_limit:
        random.shuffle(sample_rows)
        sample_rows = sample_rows[:eval_limit]

    if use_generalized_input:
        samples = _build_generalized_eval_samples(
            sample_rows,
            action_dim=action_dim,
            history_len=history_len,
            gamma=gamma,
        )
        if eval_limit > 0 and len(samples) > eval_limit:
            random.shuffle(samples)
            samples = samples[:eval_limit]
        examples, acc, value_mse = _evaluate_generalized(
            model=model,
            samples=samples,
            history_len=history_len,
        )
    else:
        t_state, t_hist, t_y, t_v = _build_eval_tensors(
            sample_rows,
            state_dim=state_dim,
            action_dim=action_dim,
            history_len=history_len,
            gamma=gamma,
        )
        with torch.no_grad():
            out = model.forward_policy_value(t_state, t_hist)
            logits = out["logits"]
            value = out["value"]
            pred = torch.argmax(logits, dim=-1)
            acc = float((pred == t_y).float().mean().item())
            value_mse = float(torch.nn.functional.mse_loss(value, t_v).item())
        examples = int(t_state.shape[0])
    return {
        "checkpoint": path,
        "training_stage": ckpt.get("training_stage", ckpt.get("stage", "unknown")),
        "examples": int(examples),
        "action_accuracy": float(acc),
        "value_mse": float(value_mse),
        "use_generalized_input": bool(use_generalized_input),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_dir", type=str, default="central_memory")
    ap.add_argument("--checkpoint_a", type=str, required=True, help="e.g. policy-only checkpoint")
    ap.add_argument("--checkpoint_b", type=str, required=True, help="e.g. policy+value checkpoint")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eval_limit", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(int(args.seed))
    rows = _load_rows(args.memory_dir)
    ra = _evaluate_checkpoint(args.checkpoint_a, rows, gamma=float(args.gamma), eval_limit=int(args.eval_limit))
    rb = _evaluate_checkpoint(args.checkpoint_b, rows, gamma=float(args.gamma), eval_limit=int(args.eval_limit))

    print("MetaTransformer Stage Benchmark")
    print(json.dumps({"A": ra, "B": rb}, indent=2, ensure_ascii=False))
    print(
        "delta: "
        f"acc={float(rb['action_accuracy']) - float(ra['action_accuracy']):+.4f}, "
        f"value_mse={float(rb['value_mse']) - float(ra['value_mse']):+.4f}"
    )


if __name__ == "__main__":
    main()
