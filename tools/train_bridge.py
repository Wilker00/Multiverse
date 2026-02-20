"""
tools/train_bridge.py

Train a learned semantic bridge via contrastive learning on cross-verse events.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import torch

from models.contrastive_bridge import ContrastiveBridge, ContrastiveBridgeConfig


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


def _safe_bool(x: Any, default: bool = False) -> bool:
    try:
        if x is None:
            return bool(default)
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return bool(x)
    except Exception:
        return default


@dataclass
class EventRow:
    verse_name: str
    run_id: str
    episode_id: str
    step_idx: int
    obs: Any
    reward: float
    done: bool
    truncated: bool
    safety_violation: bool


@dataclass
class BridgeSample:
    verse_name: str
    obs: Any
    ttf_bin: int
    split: str  # "train" | "val"


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


def _extract_safety_violation(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    for key in (
        "safety_violation",
        "safety_breach",
        "violated_safety",
        "constraint_violation",
        "mcts_vetoed",
        "unsafe_action_blocked",
        "hazard",
        "fell_off_cliff",
        "battery_fail",
        "died",
    ):
        if _safe_bool(info.get(key), default=False):
            return True
    return False


def _to_event_row(raw: Dict[str, Any]) -> Optional[EventRow]:
    verse = str(raw.get("verse_name", "")).strip().lower()
    if not verse:
        return None
    obs = raw.get("obs")
    if obs is None:
        return None
    run_id = str(raw.get("run_id", "")).strip()
    episode_id = str(raw.get("episode_id", "")).strip()
    if not run_id or not episode_id:
        return None
    return EventRow(
        verse_name=verse,
        run_id=run_id,
        episode_id=episode_id,
        step_idx=_safe_int(raw.get("step_idx", 0), 0),
        obs=obs,
        reward=_safe_float(raw.get("reward", 0.0), 0.0),
        done=bool(_safe_bool(raw.get("done", False), False)),
        truncated=bool(_safe_bool(raw.get("truncated", False), False)),
        safety_violation=_extract_safety_violation(raw.get("info")),
    )


def _load_events(
    *,
    memory_dir: str,
    runs_root: str,
    input_jsonl: Sequence[str],
    max_rows: int,
) -> List[EventRow]:
    paths: List[str] = []
    mem_path = os.path.join(memory_dir, "memories.jsonl")
    if os.path.isfile(mem_path):
        paths.append(mem_path)
    if os.path.isdir(runs_root):
        for name in sorted(os.listdir(runs_root)):
            p = os.path.join(runs_root, name, "events.jsonl")
            if os.path.isfile(p):
                paths.append(p)
    for p in input_jsonl:
        if os.path.isfile(p):
            paths.append(p)

    if not paths:
        raise FileNotFoundError(
            "No event sources found. Checked memory_dir, runs_root, and input_jsonl."
        )

    rows: List[EventRow] = []
    limit = max(0, int(max_rows))
    for p in paths:
        for raw in _iter_jsonl(p):
            row = _to_event_row(raw)
            if row is None:
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                return rows
    return rows


def _group_by_episode(rows: Sequence[EventRow]) -> Dict[Tuple[str, str, str], List[EventRow]]:
    grouped: Dict[Tuple[str, str, str], List[EventRow]] = {}
    for r in rows:
        key = (r.verse_name, r.run_id, r.episode_id)
        grouped.setdefault(key, []).append(r)
    for key in grouped:
        grouped[key].sort(key=lambda x: int(x.step_idx))
    return grouped


def _episode_failure_score(ep: Sequence[EventRow]) -> float:
    if not ep:
        return 0.0
    min_reward = min(float(x.reward) for x in ep)
    final_reward = float(ep[-1].reward)
    any_safety = any(bool(x.safety_violation) for x in ep)
    score = 0.0
    score += 1.0 if any_safety else 0.0
    score += 1.0 if final_reward < 0.0 else 0.0
    score += 1.0 if min_reward < -1.0 else 0.0
    return score


def _build_bridge_samples(
    *,
    rows: Sequence[EventRow],
    ttf_bins: int,
    failure_only: bool,
    failure_score_threshold: float,
    val_fraction: float,
    seed: int,
) -> List[BridgeSample]:
    grouped = _group_by_episode(rows)
    bins = max(2, int(ttf_bins))
    rng = random.Random(int(seed))

    by_episode: List[Tuple[Tuple[str, str, str], List[EventRow], str]] = []
    for key, ep in grouped.items():
        score = _episode_failure_score(ep)
        if failure_only and score < float(failure_score_threshold):
            continue
        split = "val" if rng.random() < float(max(0.0, min(0.9, val_fraction))) else "train"
        by_episode.append((key, ep, split))

    samples: List[BridgeSample] = []
    for _, ep, split in by_episode:
        n = len(ep)
        if n <= 0:
            continue
        denom = max(1, n - 1)
        for i, step in enumerate(ep):
            # Time-to-terminal normalized [0,1], where 1 means early state and 0 near terminal.
            ttf = float(n - 1 - i) / float(denom)
            b = int(round(ttf * float(bins - 1)))
            b = max(0, min(bins - 1, b))
            samples.append(
                BridgeSample(
                    verse_name=step.verse_name,
                    obs=step.obs,
                    ttf_bin=b,
                    split=split,
                )
            )
    return samples


def _index_samples(samples: Sequence[BridgeSample], *, split: str) -> Dict[int, Dict[str, List[Any]]]:
    out: Dict[int, Dict[str, List[Any]]] = {}
    for s in samples:
        if s.split != split:
            continue
        by_verse = out.setdefault(int(s.ttf_bin), {})
        by_verse.setdefault(str(s.verse_name), []).append(s.obs)
    return out


def _valid_bins(index: Dict[int, Dict[str, List[Any]]], *, min_verses: int, min_per_verse: int) -> List[int]:
    out: List[int] = []
    for b, by_verse in index.items():
        verses = [v for v, arr in by_verse.items() if len(arr) >= int(min_per_verse)]
        if len(verses) >= int(min_verses):
            out.append(int(b))
    return sorted(out)


def _sample_pair_batch(
    *,
    index: Dict[int, Dict[str, List[Any]]],
    valid_bins: Sequence[int],
    batch_size: int,
    rng: random.Random,
) -> Tuple[List[Any], List[Any]]:
    if not valid_bins:
        raise RuntimeError("No valid cross-verse bins available for batch sampling.")

    obs_a: List[Any] = []
    obs_b: List[Any] = []
    bs = max(1, int(batch_size))
    for _ in range(bs):
        b = int(rng.choice(list(valid_bins)))
        by_verse = index[b]
        verse_names = [v for v, arr in by_verse.items() if len(arr) > 0]
        if len(verse_names) < 2:
            continue
        va, vb = rng.sample(verse_names, 2)
        obs_a.append(rng.choice(by_verse[va]))
        obs_b.append(rng.choice(by_verse[vb]))

    if len(obs_a) != bs:
        raise RuntimeError("Failed to form a full batch from indexed bins.")
    return obs_a, obs_b


def _evaluate_retrieval(
    *,
    model: ContrastiveBridge,
    index: Dict[int, Dict[str, List[Any]]],
    valid_bins: Sequence[int],
    batch_size: int,
    eval_steps: int,
    seed: int,
) -> Dict[str, float]:
    if not valid_bins:
        return {"loss": float("nan"), "acc_top1": float("nan"), "steps": 0.0}
    rng = random.Random(int(seed))
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, int(eval_steps))):
            obs_a, obs_b = _sample_pair_batch(
                index=index,
                valid_bins=valid_bins,
                batch_size=max(2, int(batch_size)),
                rng=rng,
            )
            out = model(obs_a, obs_b, return_loss=True)
            logits = out["logits"]
            labels = torch.arange(int(logits.shape[0]), device=logits.device, dtype=torch.long)
            pred = torch.argmax(logits, dim=1)
            acc = float((pred == labels).float().mean().item())
            loss = float(out["loss"].item())
            total_acc += acc
            total_loss += loss
            n += 1
    return {
        "loss": total_loss / float(max(1, n)),
        "acc_top1": total_acc / float(max(1, n)),
        "steps": float(n),
    }


def train_bridge(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    rows = _load_events(
        memory_dir=str(args.memory_dir),
        runs_root=str(args.runs_root),
        input_jsonl=list(args.input_jsonl or []),
        max_rows=max(0, int(args.max_rows)),
    )
    samples = _build_bridge_samples(
        rows=rows,
        ttf_bins=max(2, int(args.ttf_bins)),
        failure_only=bool(args.failure_only),
        failure_score_threshold=float(args.failure_score_threshold),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    if not samples:
        raise RuntimeError("No bridge samples available after filtering.")

    train_index = _index_samples(samples, split="train")
    val_index = _index_samples(samples, split="val")
    train_bins = _valid_bins(
        train_index,
        min_verses=max(2, int(args.min_verses_per_bin)),
        min_per_verse=max(1, int(args.min_samples_per_verse)),
    )
    val_bins = _valid_bins(
        val_index,
        min_verses=max(2, int(args.min_verses_per_bin)),
        min_per_verse=max(1, int(args.min_samples_per_verse)),
    )
    if not train_bins:
        # Fallback: collapse splits and train on all samples when split fragmentation is too high.
        for s in samples:
            s.split = "train"
        train_index = _index_samples(samples, split="train")
        train_bins = _valid_bins(
            train_index,
            min_verses=max(2, int(args.min_verses_per_bin)),
            min_per_verse=max(1, int(args.min_samples_per_verse)),
        )
        if not train_bins:
            raise RuntimeError("No valid training bins with cross-verse coverage.")
        val_index = {}
        val_bins = []

    cfg = ContrastiveBridgeConfig.from_dict(
        {
            "n_embd": int(args.n_embd),
            "proj_dim": int(args.proj_dim),
            "temperature_init": float(args.temperature_init),
            "temperature_min": float(args.temperature_min),
            "temperature_max": float(args.temperature_max),
            "max_keys": int(args.max_keys),
            "num_key_buckets": int(args.num_key_buckets),
            "num_heads": int(args.num_heads),
            "n_layers": int(args.n_layers),
            "strict_config_validation": True,
        }
    )
    model = ContrastiveBridge(config=cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    rng = random.Random(int(args.seed) + 17)
    epochs = max(1, int(args.epochs))
    steps_per_epoch = max(1, int(args.steps_per_epoch))
    batch_size = max(2, int(args.batch_size))
    history: List[Dict[str, float]] = []
    best_val_loss: Optional[float] = None
    best_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for _ in range(steps_per_epoch):
            obs_a, obs_b = _sample_pair_batch(
                index=train_index,
                valid_bins=train_bins,
                batch_size=batch_size,
                rng=rng,
            )
            out = model(obs_a, obs_b, return_loss=True)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / float(max(1, steps_per_epoch))
        val_metrics = _evaluate_retrieval(
            model=model,
            index=val_index if val_bins else train_index,
            valid_bins=val_bins if val_bins else train_bins,
            batch_size=batch_size,
            eval_steps=max(2, int(args.eval_steps)),
            seed=int(args.seed) + epoch * 101,
        )
        val_loss = float(val_metrics["loss"])

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc_top1": float(val_metrics["acc_top1"]),
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc_top1={float(val_metrics['acc_top1']):.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = str(args.out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    summary = {
        "rows_loaded": int(len(rows)),
        "samples_total": int(len(samples)),
        "train_bins": int(len(train_bins)),
        "val_bins": int(len(val_bins)),
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "batch_size": int(batch_size),
        "best_val_loss": float(best_val_loss if best_val_loss is not None else float("nan")),
        "history": history,
    }
    model.save(
        out_path,
        extra={
            "summary": summary,
            "train_config": vars(args),
        },
    )

    report = {
        "saved_to": out_path.replace("\\", "/"),
        "rows_loaded": int(len(rows)),
        "samples_total": int(len(samples)),
        "train_bins": int(len(train_bins)),
        "val_bins": int(len(val_bins)),
        "best_val_loss": float(best_val_loss if best_val_loss is not None else float("nan")),
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_dir", type=str, default="central_memory")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--input_jsonl", action="append", default=None, help="Additional JSONL event sources.")
    ap.add_argument("--out_path", type=str, default=os.path.join("models", "contrastive_bridge.pt"))

    ap.add_argument("--max_rows", type=int, default=200000)
    ap.add_argument("--ttf_bins", type=int, default=20)
    ap.add_argument("--failure_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--failure_score_threshold", type=float, default=1.0)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--min_verses_per_bin", type=int, default=2)
    ap.add_argument("--min_samples_per_verse", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_steps", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--temperature_init", type=float, default=0.07)
    ap.add_argument("--temperature_min", type=float, default=0.01)
    ap.add_argument("--temperature_max", type=float, default=1.0)
    ap.add_argument("--max_keys", type=int, default=128)
    ap.add_argument("--num_key_buckets", type=int, default=1024)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    report = train_bridge(args)
    print("contrastive bridge training complete")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
