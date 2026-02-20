"""
tools/universal_model.py

Build, validate, and query the deployable UniversalModel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec
from memory.central_repository import CentralMemoryConfig, ingest_run, sanitize_memory_file
from memory.selection import SelectionConfig
from memory.splits import SplitConfig, save_split_manifest, split_run_dirs
from models.universal_model import UniversalModel, UniversalModelConfig
from orchestrator.model_benchmark import (
    benchmark_universal_model,
    print_benchmark_report,
)
from orchestrator.model_validator import print_validation_report, validate_run_against_memory


def _list_run_dirs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[str] = []
    for name in os.listdir(runs_root):
        p = os.path.join(runs_root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "events.jsonl")):
            out.append(p)
    out.sort()
    return out


def _parse_obs(value: str):
    parsed = json.loads(value)
    if not isinstance(parsed, (dict, list, int, float)):
        raise ValueError("--obs must be JSON dict/list/number")
    return parsed


def _parse_kv_list(kvs: List[str] | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not kvs:
        return out
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Invalid --vparam '{item}', expected key=value")
        k, v = item.split("=", 1)
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


def cmd_build(args: argparse.Namespace) -> None:
    cfg = CentralMemoryConfig(root_dir=args.memory_dir)
    stats = sanitize_memory_file(cfg)
    if stats.dropped_lines > 0:
        print(
            f"Sanitized central memory: dropped {stats.dropped_lines} malformed line(s) "
            f"out of {stats.input_lines}."
        )
    selection = SelectionConfig(
        min_reward=args.min_reward,
        max_reward=args.max_reward,
        keep_top_k_per_episode=args.top_k_per_episode,
        keep_top_k_episodes=args.top_k_episodes,
        novelty_bonus=args.novelty_bonus,
    )

    run_dirs = []
    if args.run_dir:
        run_dirs = [str(p) for p in args.run_dir]
    else:
        run_dirs = _list_run_dirs(args.runs_root)

    if not run_dirs:
        print("No runs found to ingest. Build aborted.")
        return

    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    ).normalize()
    split_map = split_run_dirs(run_dirs, split_cfg)
    if args.use_split != "all":
        run_dirs = split_map.get(args.use_split, [])
        print(f"Using split '{args.use_split}' with {len(run_dirs)} run(s).")
        if not run_dirs:
            print("No runs matched selected split. Build aborted.")
            return

    if args.split_manifest:
        manifest = save_split_manifest(args.split_manifest, _list_run_dirs(args.runs_root), split_cfg)
        print(f"Wrote split manifest: {manifest}")

    total_added = 0
    for rd in run_dirs:
        st = ingest_run(run_dir=rd, cfg=cfg, selection=selection, max_events=args.max_events)
        total_added += st.added_events
        print(
            f"{os.path.basename(rd)}: added={st.added_events} "
            f"selected={st.selected_events} skipped={st.skipped_duplicates}"
        )

    model = UniversalModel(
        UniversalModelConfig(
            memory_dir=args.memory_dir,
            default_top_k=args.top_k,
            default_min_score=args.min_score,
            default_verse_name=args.default_verse,
            meta_model_path=args.meta_model_path,
            meta_confidence_threshold=float(args.meta_confidence_threshold),
            prefer_meta_policy=bool(args.prefer_meta_policy),
            meta_history_len=int(args.meta_history_len),
            learned_bridge_enabled=bool(args.learned_bridge_enabled),
            learned_bridge_model_path=args.learned_bridge_model_path,
            learned_bridge_score_weight=float(args.learned_bridge_score_weight),
        )
    )
    out = model.save(args.out_dir, snapshot_memory=args.snapshot_memory)
    print("")
    print("UniversalModel build complete")
    print(f"model_config : {out}")
    print(f"total_added  : {total_added}")


def cmd_validate(args: argparse.Namespace) -> None:
    model = UniversalModel.load(args.model_dir)
    report = validate_run_against_memory(
        run_dir=args.run_dir,
        memory_dir=model.config.memory_dir,
        top_k=args.top_k if args.top_k is not None else model.config.default_top_k,
        min_score=args.min_score if args.min_score is not None else model.config.default_min_score,
    )
    print_validation_report(report)


def cmd_validate_set(args: argparse.Namespace) -> None:
    model = UniversalModel.load(args.model_dir)
    run_dirs = _list_run_dirs(args.runs_root)
    if not run_dirs:
        print("No runs found.")
        return

    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    ).normalize()
    if args.split != "all":
        run_dirs = split_run_dirs(run_dirs, split_cfg).get(args.split, [])
    if not run_dirs:
        print("No runs matched selected split.")
        return

    reports = []
    for rd in run_dirs:
        rep = validate_run_against_memory(
            run_dir=rd,
            memory_dir=model.config.memory_dir,
            top_k=args.top_k if args.top_k is not None else model.config.default_top_k,
            min_score=args.min_score if args.min_score is not None else model.config.default_min_score,
        )
        reports.append(rep)

    coverage = sum(r.coverage for r in reports) / float(len(reports))
    action_acc = sum(r.action_accuracy for r in reports) / float(len(reports))
    total_events = sum(r.total_events for r in reports)
    total_matched = sum(r.matched_events for r in reports)
    total_action_match = sum(r.action_match_events for r in reports)

    print("Model validation set report")
    print(f"runs            : {len(reports)}")
    print(f"split           : {args.split}")
    print(f"total_events    : {total_events}")
    print(f"matched_events  : {total_matched}")
    print(f"action_match    : {total_action_match}")
    print(f"mean_coverage   : {coverage:.3f}")
    print(f"mean_action_acc : {action_acc:.3f}")

    if args.min_coverage is not None and coverage < float(args.min_coverage):
        raise SystemExit(f"Coverage gate failed: {coverage:.3f} < {float(args.min_coverage):.3f}")
    if args.min_action_accuracy is not None and action_acc < float(args.min_action_accuracy):
        raise SystemExit(
            f"Action-accuracy gate failed: {action_acc:.3f} < {float(args.min_action_accuracy):.3f}"
        )


def cmd_predict(args: argparse.Namespace) -> None:
    model = UniversalModel.load(args.model_dir)
    history = None
    if args.recent_history:
        parsed = json.loads(args.recent_history)
        if not isinstance(parsed, list):
            raise ValueError("--recent_history must be a JSON list")
        history = parsed
    pred = model.predict(
        obs=_parse_obs(args.obs),
        verse_name=args.verse,
        top_k=args.top_k,
        min_score=args.min_score,
        recent_history=history,
    )
    print("UniversalModel prediction")
    print(f"action     : {pred['action']}")
    print(f"confidence : {pred['confidence']:.3f}")
    print(f"matched    : {pred['matched']}")
    print(f"strategy   : {pred.get('strategy')}")
    print(f"bridge_src : {pred.get('bridge_source_verse')}")
    print(f"meta_used  : {pred.get('meta_used')}")
    print(f"weights    : {pred['weights']}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    model = UniversalModel.load(args.model_dir)
    verse_params = _parse_kv_list(args.vparam)

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=args.verse,
        verse_version=args.verse_version,
        seed=args.seed,
        tags=["benchmark"],
        params=verse_params,
    )
    report = benchmark_universal_model(
        model=model,
        verse_spec=verse_spec,
        episodes=args.episodes,
        max_steps=args.max_steps,
        top_k=args.top_k,
        min_score=args.min_score,
        seed=args.seed,
        random_fallback=True,
        audit_confidence=not bool(args.no_confidence_audit),
        audit_root_dir=model.config.memory_dir,
    )
    print_benchmark_report(report, label="universal_model")

    if args.compare_random:
        class _NullModel:
            def predict(self, **kwargs):
                return {"action": None, "confidence": 0.0, "matched": 0, "weights": {}}

        rand_report = benchmark_universal_model(
            model=_NullModel(),  # type: ignore[arg-type]
            verse_spec=verse_spec,
            episodes=args.episodes,
            max_steps=args.max_steps,
            top_k=args.top_k,
            min_score=args.min_score,
            seed=args.seed,
            random_fallback=True,
            audit_confidence=False,
        )
        print("")
        print_benchmark_report(rand_report, label="random_baseline")

    if args.min_mean_return is not None and report.mean_return < float(args.min_mean_return):
        raise SystemExit(
            f"Benchmark gate failed: mean_return {report.mean_return:.3f} "
            f"< required {float(args.min_mean_return):.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="ingest runs and package a UniversalModel")
    b.add_argument("--runs_root", type=str, default="runs")
    b.add_argument("--run_dir", action="append", default=None, help="optional explicit run_dir (repeatable)")
    b.add_argument("--memory_dir", type=str, default="central_memory")
    b.add_argument("--out_dir", type=str, default="models/universal_model")
    b.add_argument("--snapshot_memory", action="store_true", help="copy memory into model package for portability")
    b.add_argument("--max_events", type=int, default=None)
    b.add_argument("--min_reward", type=float, default=-1e9)
    b.add_argument("--max_reward", type=float, default=1e9)
    b.add_argument("--top_k_per_episode", type=int, default=0)
    b.add_argument("--top_k_episodes", type=int, default=0)
    b.add_argument("--novelty_bonus", type=float, default=0.0)
    b.add_argument("--top_k", type=int, default=5)
    b.add_argument("--min_score", type=float, default=0.0)
    b.add_argument("--default_verse", type=str, default=None)
    b.add_argument("--use_split", type=str, default="all", choices=["all", "train", "val", "test"])
    b.add_argument("--train_ratio", type=float, default=0.7)
    b.add_argument("--val_ratio", type=float, default=0.15)
    b.add_argument("--test_ratio", type=float, default=0.15)
    b.add_argument("--split_seed", type=int, default=123)
    b.add_argument("--split_manifest", type=str, default=None, help="optional JSONL split manifest output")
    b.add_argument("--meta_model_path", type=str, default=None, help="optional MetaTransformer checkpoint")
    b.add_argument("--meta_confidence_threshold", type=float, default=0.35)
    b.add_argument("--prefer_meta_policy", action="store_true")
    b.add_argument("--meta_history_len", type=int, default=6)
    b.add_argument("--learned_bridge_enabled", action="store_true")
    b.add_argument(
        "--learned_bridge_model_path",
        type=str,
        default=None,
        help="Optional learned bridge checkpoint (contrastive_bridge.pt).",
    )
    b.add_argument("--learned_bridge_score_weight", type=float, default=0.35)
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="validate model action match against a run")
    v.add_argument("--model_dir", type=str, default="models/universal_model")
    v.add_argument("--run_dir", type=str, required=True)
    v.add_argument("--top_k", type=int, default=None)
    v.add_argument("--min_score", type=float, default=None)
    v.set_defaults(func=cmd_validate)

    vs = sub.add_parser("validate_set", help="validate model against many runs, with optional metric gates")
    vs.add_argument("--model_dir", type=str, default="models/universal_model")
    vs.add_argument("--runs_root", type=str, default="runs")
    vs.add_argument("--split", type=str, default="val", choices=["all", "train", "val", "test"])
    vs.add_argument("--train_ratio", type=float, default=0.7)
    vs.add_argument("--val_ratio", type=float, default=0.15)
    vs.add_argument("--test_ratio", type=float, default=0.15)
    vs.add_argument("--split_seed", type=int, default=123)
    vs.add_argument("--top_k", type=int, default=None)
    vs.add_argument("--min_score", type=float, default=None)
    vs.add_argument("--min_coverage", type=float, default=None)
    vs.add_argument("--min_action_accuracy", type=float, default=None)
    vs.set_defaults(func=cmd_validate_set)

    p = sub.add_parser("predict", help="predict action for an observation")
    p.add_argument("--model_dir", type=str, default="models/universal_model")
    p.add_argument("--obs", type=str, required=True)
    p.add_argument("--verse", type=str, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--min_score", type=float, default=None)
    p.add_argument("--recent_history", type=str, default=None, help="JSON list of {obs,action,reward}")
    p.set_defaults(func=cmd_predict)

    bm = sub.add_parser("benchmark", help="benchmark model by return in a live verse")
    bm.add_argument("--model_dir", type=str, default="models/universal_model")
    bm.add_argument("--verse", type=str, default="line_world")
    bm.add_argument("--verse_version", type=str, default="0.1")
    bm.add_argument("--vparam", action="append", default=None, help="repeat key=value for verse params")
    bm.add_argument("--episodes", type=int, default=20)
    bm.add_argument("--max_steps", type=int, default=40)
    bm.add_argument("--seed", type=int, default=123)
    bm.add_argument("--top_k", type=int, default=None)
    bm.add_argument("--min_score", type=float, default=None)
    bm.add_argument("--compare_random", action="store_true")
    bm.add_argument("--no_confidence_audit", action="store_true")
    bm.add_argument("--min_mean_return", type=float, default=None)
    bm.set_defaults(func=cmd_benchmark)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
