"""
tools/mcts_cycle.py

Periodic MCTS self-improvement loop:
- runs search-guided training cycles,
- emits MCTS trace dataset,
- writes latest + versioned checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec
from memory.embeddings import obs_to_vector
from models.meta_transformer import MetaTransformer
from orchestrator.mcts_trainer import MCTSTrainer, MCTSTrainerConfig
from verses.registry import create_verse, register_builtin as register_builtin_verses


def _parse_kv_list(kvs):
    out: Dict[str, Any] = {}
    for item in (kvs or []):
        if "=" not in str(item):
            continue
        k, v = str(item).split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except Exception:
            out[k] = v
    return out


def _default_params(verse: str, max_steps: int) -> Dict[str, Any]:
    v = str(verse).strip().lower()
    out: Dict[str, Any] = {"max_steps": int(max_steps)}
    if v == "chess_world":
        out.update({"win_material": 8, "lose_material": -8, "random_swing": 0.20})
    elif v == "go_world":
        out.update({"target_territory": 10, "random_swing": 0.25})
    elif v == "uno_world":
        out.update({"start_hand": 7, "opp_start_hand": 7, "random_swing": 0.25})
    return out


def _build_model(checkpoint_path: str, verse_spec: VerseSpec, *, n_embd: int) -> MetaTransformer:
    import torch

    register_builtin_verses()
    verse = create_verse(verse_spec)
    rr = verse.reset()
    obs_vec = obs_to_vector(rr.obs)
    state_dim = max(1, len(obs_vec))
    action_dim = max(1, int(getattr(getattr(verse, "action_space", None), "n", 1) or 1))
    verse.close()

    model = MetaTransformer(state_dim=state_dim, action_dim=action_dim, n_embd=int(n_embd))
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verse", type=str, default="chess_world")
    ap.add_argument("--verse_version", type=str, default="0.1")
    ap.add_argument("--vparam", action="append", default=None)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--episodes_per_cycle", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--num_simulations", type=int, default=96)
    ap.add_argument("--search_depth", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--checkpoint_path", type=str, default=os.path.join("models", "meta_transformer.pt"))
    ap.add_argument("--checkpoint_versioned_dir", type=str, default=os.path.join("models", "meta_transformer_versions"))
    ap.add_argument("--trace_out", type=str, default=os.path.join("models", "expert_datasets", "mcts_trace_dataset.jsonl"))
    ap.add_argument("--high_quality_trace_filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min_quality_value_gain", type=float, default=0.03)
    ap.add_argument("--min_quality_policy_shift_l1", type=float, default=0.12)
    ap.add_argument("--min_quality_kl_divergence", type=float, default=0.01)
    ap.add_argument("--min_forced_loss_prior_mass", type=float, default=0.20)
    ap.add_argument("--min_forced_loss_mass_drop", type=float, default=0.06)
    ap.add_argument("--sparse_replay_min_samples", type=int, default=32)
    ap.add_argument("--sparse_replay_batch_size", type=int, default=32)
    args = ap.parse_args()

    params = _default_params(args.verse, args.max_steps)
    params.update(_parse_kv_list(args.vparam))
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=str(args.verse),
        verse_version=str(args.verse_version),
        seed=int(args.seed),
        params=params,
    )

    model = _build_model(args.checkpoint_path, verse_spec, n_embd=int(args.n_embd))
    for c in range(max(1, int(args.cycles))):
        cfg = MCTSTrainerConfig(
            episodes=max(1, int(args.episodes_per_cycle)),
            max_steps=max(1, int(args.max_steps)),
            num_simulations=max(8, int(args.num_simulations)),
            search_depth=max(2, int(args.search_depth)),
            batch_size=max(8, int(args.batch_size)),
            lr=float(args.lr),
            seed=int(args.seed) + c,
            checkpoint_path=str(args.checkpoint_path),
            checkpoint_every_episodes=max(1, int(args.episodes_per_cycle)),
            checkpoint_versioned_dir=str(args.checkpoint_versioned_dir),
            trace_out_path=str(args.trace_out),
            high_quality_trace_filter=bool(args.high_quality_trace_filter),
            min_quality_value_gain=float(args.min_quality_value_gain),
            min_quality_policy_shift_l1=float(args.min_quality_policy_shift_l1),
            min_quality_kl_divergence=float(args.min_quality_kl_divergence),
            min_forced_loss_prior_mass=float(args.min_forced_loss_prior_mass),
            min_forced_loss_mass_drop=float(args.min_forced_loss_mass_drop),
            sparse_replay_min_samples=max(8, int(args.sparse_replay_min_samples)),
            sparse_replay_batch_size=max(8, int(args.sparse_replay_batch_size)),
        )
        trainer = MCTSTrainer(verse_spec=verse_spec, model=model, config=cfg)
        metrics = trainer.run()
        print(
            f"[mcts_cycle] cycle={c+1}/{int(args.cycles)} "
            f"mean_return={float(metrics.get('mean_return', 0.0)):.3f} "
            f"policy_loss={float(metrics.get('mean_policy_loss', 0.0)):.4f} "
            f"value_loss={float(metrics.get('mean_value_loss', 0.0)):.4f}"
        )


if __name__ == "__main__":
    main()
