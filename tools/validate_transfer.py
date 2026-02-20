"""
tools/validate_transfer.py

Validate whether semantic memory transfer improves park_world learning speed.
Workflow:
1) Build synthetic transfer dataset from line/grid DNA via semantic bridge.
2) Train scratch Q on park_world.
3) Train transfer-initialized Q on park_world (warm-start from dataset).
4) Compare sample-efficiency metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from memory.semantic_bridge import translate_dna
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer


def _default_park_params(max_steps: int) -> Dict[str, Any]:
    return {"adr_enabled": False, "max_steps": int(max_steps)}


def _iter_jsonl(path: str):
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


def _merge_jsonl(paths: List[str], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            if not os.path.isfile(p):
                continue
            for row in _iter_jsonl(p):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
    return rows


def _learning_metrics(run_dir: str) -> Dict[str, Any]:
    stats = evaluate_run(run_dir)
    eps = list(stats.episode_stats)
    if not eps:
        return {
            "episodes": 0,
            "first_success_episode": None,
            "early_mean_return": 0.0,
            "final_mean_return": float(stats.mean_return),
            "success_rate": stats.success_rate,
        }

    first_success = None
    for idx, e in enumerate(eps, start=1):
        if e.reached_goal is True:
            first_success = idx
            break

    quarter = max(1, len(eps) // 4)
    early_mean = sum(float(e.return_sum) for e in eps[:quarter]) / float(quarter)

    return {
        "episodes": int(len(eps)),
        "first_success_episode": first_success,
        "early_mean_return": float(early_mean),
        "final_mean_return": float(stats.mean_return),
        "success_rate": stats.success_rate,
    }


def _run_q(
    *,
    trainer: Trainer,
    run_tag: str,
    seed: int,
    episodes: int,
    max_steps: int,
    dataset_path: Optional[str] = None,
    warmstart_reward_scale: float = 0.5,
) -> str:
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name="park_world",
        verse_version="0.1",
        seed=int(seed),
        tags=["transfer_validation", run_tag],
        params=_default_park_params(max_steps),
    )

    cfg: Dict[str, Any] = {
        "train": True,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "warmstart_reward_scale": float(warmstart_reward_scale),
    }
    if dataset_path:
        cfg["dataset_path"] = str(dataset_path)

    agent_spec = AgentSpec(
        spec_version="v1",
        policy_id=f"q_{run_tag}",
        policy_version="0.1",
        algo="q",
        seed=int(seed),
        tags=["transfer_validation", run_tag],
        config=cfg,
    )

    res = trainer.run(
        verse_spec=verse_spec,
        agent_spec=agent_spec,
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
    )
    run_id = str(res.get("run_id", ""))
    if not run_id:
        raise RuntimeError("trainer did not return run_id")
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default=os.path.join("models", "tuning", "transfer_validation_park.json"))

    ap.add_argument("--line_dna", type=str, default=os.path.join("models", "expert_datasets", "line_world.jsonl"))
    ap.add_argument("--grid_dna", type=str, default=os.path.join("models", "expert_datasets", "grid_world.jsonl"))
    ap.add_argument("--park_dna", type=str, default=os.path.join("models", "expert_datasets", "park_world.jsonl"))
    ap.add_argument("--merge_real_park", action="store_true")
    ap.add_argument("--transfer_dataset_out", type=str, default=os.path.join("models", "expert_datasets", "transfer_park_world.jsonl"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    bridge_paths: List[str] = []
    for source_path, source_verse in [
        (args.grid_dna, "grid_world"),
        (args.line_dna, "line_world"),
    ]:
        if not os.path.isfile(source_path):
            continue
        out_path = os.path.join(
            os.path.dirname(args.transfer_dataset_out) or ".",
            f"synthetic_transfer_{source_verse}_to_park.jsonl",
        )
        st = translate_dna(
            source_dna_path=source_path,
            target_verse_name="park_world",
            output_path=out_path,
            source_verse_name=source_verse,
        )
        print(
            f"bridge {source_verse}->park rows={st.translated_rows} dropped={st.dropped_rows} out={st.output_path}"
        )
        bridge_paths.append(out_path)

    merge_inputs = list(bridge_paths)
    if bool(args.merge_real_park) and os.path.isfile(args.park_dna):
        merge_inputs.append(args.park_dna)

    merged_rows = _merge_jsonl(merge_inputs, args.transfer_dataset_out)
    print(f"transfer dataset rows={merged_rows} out={args.transfer_dataset_out}")

    trainer = Trainer(run_root=args.runs_root, schema_version="v1", auto_register_builtin=True)

    scratch_run = _run_q(
        trainer=trainer,
        run_tag="scratch",
        seed=int(args.seed),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        dataset_path=None,
    )

    transfer_run = _run_q(
        trainer=trainer,
        run_tag="transfer",
        seed=int(args.seed + 999),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        dataset_path=args.transfer_dataset_out if merged_rows > 0 else None,
    )

    scratch_metrics = _learning_metrics(os.path.join(args.runs_root, scratch_run))
    transfer_metrics = _learning_metrics(os.path.join(args.runs_root, transfer_run))

    fs_s = scratch_metrics.get("first_success_episode")
    fs_t = transfer_metrics.get("first_success_episode")
    faster_first_success = False
    if isinstance(fs_t, int):
        if not isinstance(fs_s, int):
            faster_first_success = True
        else:
            faster_first_success = int(fs_t) <= int(fs_s)

    better_early_return = float(transfer_metrics.get("early_mean_return", 0.0)) >= float(
        scratch_metrics.get("early_mean_return", 0.0)
    )
    better_final_success = float(transfer_metrics.get("success_rate") or 0.0) >= float(
        scratch_metrics.get("success_rate") or 0.0
    )

    transfer_helped = bool(faster_first_success or (better_early_return and better_final_success))

    report = {
        "scratch_run": scratch_run,
        "transfer_run": transfer_run,
        "transfer_dataset": args.transfer_dataset_out,
        "transfer_dataset_rows": int(merged_rows),
        "scratch": scratch_metrics,
        "transfer": transfer_metrics,
        "transfer_helped": bool(transfer_helped),
        "criteria": {
            "faster_first_success": bool(faster_first_success),
            "better_early_return": bool(better_early_return),
            "better_final_success": bool(better_final_success),
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"scratch_run={scratch_run} transfer_run={transfer_run}")
    print(
        f"scratch early_return={float(scratch_metrics['early_mean_return']):.3f} "
        f"success={scratch_metrics['success_rate']}"
    )
    print(
        f"transfer early_return={float(transfer_metrics['early_mean_return']):.3f} "
        f"success={transfer_metrics['success_rate']}"
    )
    print(f"transfer_helped={transfer_helped}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
