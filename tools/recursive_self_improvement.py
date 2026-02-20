"""
tools/recursive_self_improvement.py

Meta-optimizer with "Teacher" frontier support:
1) Analyze market gaps.
2) For high-risk strategy gaps (for example chess), synthesize a tutorial verse.
3) Train tutorial -> graduate back to target verse.
4) Optional shield refinement.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from core.types import AgentSpec, VerseSpec
from memory.market_analyzer import VerseStats, analyze_market
from orchestrator.teacher import TeacherConfig, append_teacher_lesson, build_teacher_plan
from orchestrator.trainer import Trainer


def _build_agent_spec(*, algo: str, policy_id: str, seed: int, train: bool) -> AgentSpec:
    return AgentSpec(
        spec_version="v1",
        policy_id=str(policy_id),
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
        config={"train": bool(train)},
    )


def _run_training(
    *,
    trainer: Trainer,
    verse_spec: VerseSpec,
    algo: str,
    policy_id: str,
    episodes: int,
    max_steps: int,
    seed: int,
    train: bool,
) -> Dict[str, Any]:
    run = trainer.run(
        verse_spec=verse_spec.evolved(params={**dict(verse_spec.params or {}), "max_steps": int(max_steps)}),
        agent_spec=_build_agent_spec(algo=algo, policy_id=policy_id, seed=seed, train=train),
        episodes=max(1, int(episodes)),
        max_steps=max(1, int(max_steps)),
        seed=int(seed),
    )
    print(
        f"    run_id={run.get('run_id')} verse={verse_spec.verse_name} "
        f"return={float(run.get('total_return', 0.0)):.3f}"
    )
    return run


def _sorted_gaps(stats: Dict[str, VerseStats], *, gap_thr: float) -> List[VerseStats]:
    rows = [s for s in stats.values() if float(s.success_rate) < float(gap_thr)]
    rows.sort(key=lambda x: (float(x.success_rate), str(x.verse_name)))
    return rows


def recursive_cycle(*, args: argparse.Namespace, trainer: Trainer, cycle_idx: int) -> None:
    print("=== STARTING RECURSIVE SELF-IMPROVEMENT CYCLE ===")
    stats = analyze_market(runs_root=str(args.runs_root))

    if not stats:
        print("Knowledge Market is empty. Running baseline explorer...")
        baseline_spec = VerseSpec(
            spec_version="v1",
            verse_name="grid_world",
            verse_version="0.1",
            seed=int(args.seed) + cycle_idx,
            tags=["rsi", "bootstrap"],
            params={"max_steps": int(args.max_steps)},
        )
        _run_training(
            trainer=trainer,
            verse_spec=baseline_spec,
            algo=str(args.algo),
            policy_id="rsi_bootstrap_grid",
            episodes=max(2, int(args.episodes // 2)),
            max_steps=int(args.max_steps),
            seed=int(args.seed) + cycle_idx,
            train=bool(args.train),
        )
        return

    gaps = _sorted_gaps(stats, gap_thr=float(args.gap_success_threshold))
    masters = [s for s in stats.values() if float(s.success_rate) >= float(args.master_success_threshold)]
    print(f"Found {len(gaps)} skill gaps and {len(masters)} mastered domains.")

    teacher_cfg = TeacherConfig.from_dict(
        {
            "enabled": bool(args.teacher_enabled),
            "target_verse": str(args.teacher_target_verse),
            "tutorial_verse": str(args.teacher_tutorial_verse),
            "lookback_runs": int(args.teacher_lookback_runs),
            "min_episodes": int(args.teacher_min_episodes),
            "risk_obs_key": str(args.teacher_risk_obs_key),
            "high_risk_obs_threshold": float(args.teacher_risk_threshold),
            "high_risk_failure_rate_threshold": float(args.teacher_failure_rate_threshold),
            "tutorial_episodes": int(args.teacher_tutorial_episodes),
            "tutorial_max_steps": int(args.teacher_tutorial_max_steps),
            "graduation_episodes": int(args.teacher_graduation_episodes),
            "graduation_max_steps": int(args.teacher_graduation_max_steps),
            "lesson_log_path": str(args.teacher_lesson_log),
        }
    )

    handled = 0
    for gap in gaps:
        if handled >= max(1, int(args.max_gaps_per_cycle)):
            break
        print(f"\nTargeting gap: {gap.verse_name} (success={float(gap.success_rate):.2f})")
        handled += 1

        if gap.top_strategic_signature:
            for master in masters:
                if master.verse_name != gap.verse_name:
                    print(f"  Mentor match: {master.verse_name}")
                    break

        use_teacher = bool(teacher_cfg.enabled) and str(gap.verse_name).strip().lower() == _norm(args.teacher_target_verse)
        if use_teacher:
            plan = build_teacher_plan(
                runs_root=str(args.runs_root),
                target_verse=str(gap.verse_name),
                cfg=teacher_cfg,
                seed=int(args.seed) + cycle_idx + handled,
            )
            print(
                "  Teacher signals: "
                f"episodes={plan.signals.episodes} "
                f"high_risk_events={plan.signals.high_risk_events} "
                f"failure_rate={plan.signals.high_risk_failure_rate:.2f} "
                f"lesson_quality={float(plan.lesson_quality):.2f}"
            )
            if plan.tutorial_spec is not None:
                print(f"  Teacher generated tutorial verse: {plan.tutorial_spec.verse_name}")
                tut_run = _run_training(
                    trainer=trainer,
                    verse_spec=plan.tutorial_spec,
                    algo=str(args.algo),
                    policy_id=f"teacher_tutorial_{gap.verse_name}",
                    episodes=int(teacher_cfg.tutorial_episodes),
                    max_steps=int(teacher_cfg.tutorial_max_steps),
                    seed=int(args.seed) + cycle_idx + handled,
                    train=bool(args.train),
                )
                gate = plan.graduation_gate if isinstance(plan.graduation_gate, dict) else {}
                gate_ready = bool(gate.get("ready", False))
                grad_run: Dict[str, Any] = {}
                if gate_ready or bool(args.teacher_force_graduation):
                    grad_run = _run_training(
                        trainer=trainer,
                        verse_spec=plan.graduation_spec,
                        algo=str(args.algo),
                        policy_id=f"teacher_graduation_{gap.verse_name}",
                        episodes=int(teacher_cfg.graduation_episodes),
                        max_steps=int(teacher_cfg.graduation_max_steps),
                        seed=int(args.seed) + cycle_idx + handled + 1,
                        train=bool(args.train),
                    )
                else:
                    print(
                        "  Graduation gate blocked: "
                        f"reason={gate.get('reason', 'not_ready')} "
                        f"confidence={float(gate.get('confidence', 0.0)):.2f}"
                    )
                append_teacher_lesson(
                    path=str(teacher_cfg.lesson_log_path),
                    plan=plan,
                    tutorial_run_id=str(tut_run.get("run_id", "")),
                    graduation_run_id=str(grad_run.get("run_id", "")),
                )
                continue

        verse_spec = VerseSpec(
            spec_version="v1",
            verse_name=str(gap.verse_name),
            verse_version="0.1",
            seed=int(args.seed) + cycle_idx + handled,
            tags=["rsi", "gap_repair"],
            params={"max_steps": int(args.max_steps)},
        )
        _run_training(
            trainer=trainer,
            verse_spec=verse_spec,
            algo=str(args.algo),
            policy_id=f"rsi_{gap.verse_name}",
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            seed=int(args.seed) + cycle_idx + handled,
            train=bool(args.train),
        )

    if not bool(args.skip_refine_shields):
        print("\n[Shield Maintenance] Updating safety post-mortem...")
        subprocess.run([sys.executable, "tools/refine_shields.py", "--epochs", str(int(args.refine_shields_epochs))], check=False)
    print("\n=== RECURSIVE CYCLE COMPLETE ===")


def _norm(x: Any) -> str:
    return str(x).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--algo", type=str, default="q")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--max_gaps_per_cycle", type=int, default=2)
    parser.add_argument("--gap_success_threshold", type=float, default=0.30)
    parser.add_argument("--master_success_threshold", type=float, default=0.70)

    parser.add_argument("--teacher_enabled", action="store_true")
    parser.add_argument("--teacher_target_verse", type=str, default="chess_world")
    parser.add_argument("--teacher_tutorial_verse", type=str, default="risk_tutorial_world")
    parser.add_argument("--teacher_lookback_runs", type=int, default=16)
    parser.add_argument("--teacher_min_episodes", type=int, default=4)
    parser.add_argument("--teacher_risk_obs_key", type=str, default="risk")
    parser.add_argument("--teacher_risk_threshold", type=float, default=5.0)
    parser.add_argument("--teacher_failure_rate_threshold", type=float, default=0.30)
    parser.add_argument("--teacher_tutorial_episodes", type=int, default=20)
    parser.add_argument("--teacher_tutorial_max_steps", type=int, default=50)
    parser.add_argument("--teacher_graduation_episodes", type=int, default=20)
    parser.add_argument("--teacher_graduation_max_steps", type=int, default=80)
    parser.add_argument("--teacher_lesson_log", type=str, default=os.path.join("models", "teacher_lessons.json"))
    parser.add_argument("--teacher_force_graduation", action="store_true")

    parser.add_argument("--skip_refine_shields", action="store_true")
    parser.add_argument("--refine_shields_epochs", type=int, default=20)
    args = parser.parse_args()

    # Keep old behavior opt-in friendly: if not explicitly enabled, run classic cycle.
    if "--teacher_enabled" not in sys.argv:
        args.teacher_enabled = True

    trainer = Trainer(run_root=str(args.runs_root), schema_version="v1", auto_register_builtin=True)
    cycle_idx = 0
    while True:
        recursive_cycle(args=args, trainer=trainer, cycle_idx=cycle_idx)
        cycle_idx += 1
        if args.once:
            break
        print(f"\nSleeping for {int(args.interval)}s before next optimization...")
        time.sleep(max(1, int(args.interval)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nRecursive optimization stopped.")

