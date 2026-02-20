"""
tools/build_strategy_transfer.py

Builds cross-verse synthetic strategy datasets (chess/go/uno) and writes
transfer performance scores that SpecialMoE can use for expert weighting.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.semantic_bridge import BridgeStats, infer_verse_from_obs, translate_dna


STRATEGY_VERSES = ("chess_world", "go_world", "uno_world")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _infer_source_verse(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                try:
                    obj = json.loads(row)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                verse_name = str(obj.get("verse_name", "") or "").strip().lower()
                if verse_name:
                    return verse_name
                guessed = infer_verse_from_obs(obj.get("obs"))
                if guessed:
                    return str(guessed)
    except Exception:
        return None
    return None


def _scan_source_dna_paths(runs_root: str) -> List[Tuple[str, Optional[str]]]:
    out: List[Tuple[str, Optional[str]]] = []
    if not os.path.isdir(runs_root):
        return out
    for run_id in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, run_id)
        if not os.path.isdir(run_dir):
            continue
        for name in ("dna_good.jsonl", "golden_dna.jsonl"):
            path = os.path.join(run_dir, name)
            if os.path.isfile(path):
                out.append((path, None))
                break
    return out


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _compute_dataset_score(path: str) -> float:
    n = 0
    reward_sum = 0.0
    advantage_sum = 0.0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                try:
                    obj = json.loads(row)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                n += 1
                reward_sum += _safe_float(obj.get("reward", 0.0), 0.0)
                advantage_sum += _safe_float(obj.get("source_advantage", 0.0), 0.0)
    except Exception:
        return 1.0
    if n <= 0:
        return 1.0
    mean_reward = reward_sum / float(n)
    mean_adv = advantage_sum / float(n)
    raw = 1.0 + (0.15 * mean_reward) + (0.10 * mean_adv)
    return max(0.05, min(5.0, raw))


def build_strategy_transfer(
    *,
    source_items: List[Tuple[str, Optional[str]]],
    targets: List[str],
    out_dir: str,
    perf_out: str,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    stats: List[BridgeStats] = []
    outputs: List[str] = []

    for src_path, src_verse_hint in source_items:
        if not os.path.isfile(src_path):
            continue
        src_verse = (src_verse_hint or _infer_source_verse(src_path) or "").strip().lower()
        if src_verse not in STRATEGY_VERSES:
            continue
        for target in targets:
            dst = str(target).strip().lower()
            if dst == src_verse:
                continue
            out_path = os.path.join(out_dir, f"synthetic_transfer_{src_verse}_to_{dst}.jsonl")
            st = translate_dna(
                source_dna_path=src_path,
                target_verse_name=dst,
                source_verse_name=src_verse,
                output_path=out_path,
            )
            stats.append(st)
            outputs.append(out_path)

    outputs = _dedupe(outputs)
    perf: Dict[str, float] = {}
    per_dataset: Dict[str, Dict[str, Any]] = {}
    for path in outputs:
        skill_id = os.path.splitext(os.path.basename(path))[0]
        score = _compute_dataset_score(path)
        perf[skill_id] = float(score)
        per_dataset[skill_id] = {
            "path": path,
            "score": float(score),
        }

    os.makedirs(os.path.dirname(perf_out) or ".", exist_ok=True)
    with open(perf_out, "w", encoding="utf-8") as f:
        json.dump(perf, f, indent=2, ensure_ascii=False)

    return {
        "sources": len(source_items),
        "targets": list(targets),
        "generated_files": len(outputs),
        "performance_file": perf_out,
        "datasets": per_dataset,
        "bridge_stats": [
            {
                "source_path": s.source_path,
                "output_path": s.output_path,
                "source_verse_name": s.source_verse_name,
                "target_verse_name": s.target_verse_name,
                "input_rows": s.input_rows,
                "translated_rows": s.translated_rows,
                "dropped_rows": s.dropped_rows,
                "learned_bridge_enabled": s.learned_bridge_enabled,
                "learned_bridge_model_path": s.learned_bridge_model_path,
                "learned_scored_rows": s.learned_scored_rows,
            }
            for s in stats
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dna", action="append", default=None, help="Source DNA path (repeatable).")
    ap.add_argument("--source_verse", action="append", default=None, help="Source verse hint for --source_dna (repeatable).")
    ap.add_argument("--runs_root", type=str, default=None, help="Optional runs root to auto-scan DNA files.")
    ap.add_argument("--targets", type=str, default="chess_world,go_world,uno_world")
    ap.add_argument("--out_dir", type=str, default=os.path.join("models", "expert_datasets"))
    ap.add_argument(
        "--perf_out",
        type=str,
        default=os.path.join("models", "expert_datasets", "strategy_transfer_performance.json"),
    )
    args = ap.parse_args()

    targets = [t.strip().lower() for t in str(args.targets).split(",") if t.strip()]
    targets = [t for t in targets if t in STRATEGY_VERSES]
    if not targets:
        raise ValueError("No valid strategy targets provided.")

    source_paths = [str(p) for p in (args.source_dna or []) if str(p).strip()]
    source_verses = [str(v).strip().lower() for v in (args.source_verse or [])]
    source_items: List[Tuple[str, Optional[str]]] = []
    for i, path in enumerate(source_paths):
        hint = source_verses[i] if i < len(source_verses) else None
        source_items.append((path, hint))

    if args.runs_root:
        source_items.extend(_scan_source_dna_paths(str(args.runs_root)))

    # Also include known expert dataset files for convenience.
    for verse in STRATEGY_VERSES:
        cand = os.path.join("models", "expert_datasets", f"{verse}.jsonl")
        if os.path.isfile(cand):
            source_items.append((cand, verse))
    for path in glob.glob(os.path.join("models", "expert_datasets", "synthetic_expert_*.jsonl")):
        source_items.append((path, None))

    # De-duplicate source paths while preserving explicit verse hints when present.
    dedup: Dict[str, Optional[str]] = {}
    for path, hint in source_items:
        if path not in dedup or (hint and not dedup.get(path)):
            dedup[path] = hint
    source_items = [(p, dedup[p]) for p in sorted(dedup.keys())]

    result = build_strategy_transfer(
        source_items=source_items,
        targets=targets,
        out_dir=str(args.out_dir),
        perf_out=str(args.perf_out),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
