"""
run_curiosity_loop.py

Long-running curiosity loop with structured logging and robust loop isolation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional

from core.types import AgentSpec, VerseSpec, new_id
from orchestrator.scheduler import select_next_verse
from orchestrator.trainer import Trainer


def _configure_logging(level: str, log_path: Optional[Path]) -> None:
    lvl = getattr(logging, str(level).strip().upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def _run_librarian_hook(
    *,
    mode: str,
    run_id: str,
    run_dir: Path,
    python_exe: str,
    cliff_reward_threshold: float,
    death_similarity_threshold: float,
    behavior_top_percent: float,
) -> None:
    logger = logging.getLogger("curiosity.librarian")
    mode_n = str(mode).strip().lower()
    if mode_n in ("none", "off"):
        logger.info("Librarian hook skipped (mode=%s).", mode_n)
        return

    if mode_n == "make_apprentice":
        if not run_dir.exists():
            logger.warning("Librarian skipped: run_dir does not exist: %s", run_dir)
            return

        cmd = [python_exe, "tools/make_apprentice.py", str(run_dir)]
        logger.info("Librarian command: %s", " ".join(cmd))
        try:
            # Capture output so we can log errors if the tool fails
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                logger.warning("Librarian failed (exit %s). Error: %s", proc.returncode, proc.stderr.strip())
            else:
                logger.info("Librarian completed successfully.")
        except Exception as e:
            logger.error("Failed to execute Librarian subprocess: %s", e)
        return

    if mode_n == "behavioral_surgeon":
        cmd = [
            python_exe,
            "tools/behavioral_surgeon.py",
            "--cliff_run_ids",
            str(run_id),
            "--cliff_reward_threshold",
            str(float(cliff_reward_threshold)),
            "--death_similarity_threshold",
            str(float(death_similarity_threshold)),
            "--behavior_top_percent",
            str(float(behavior_top_percent)),
        ]
        logger.info("Librarian command: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                logger.warning("Behavioral Surgeon failed (exit %s). Error: %s", proc.returncode, proc.stderr.strip())
            else:
                logger.info("Behavioral Surgeon completed successfully.")
        except Exception as e:
            logger.error("Failed to execute Behavioral Surgeon subprocess: %s", e)
        return

    logger.warning("Unknown librarian mode '%s'; skipping.", mode_n)


def _default_train_enabled(algo: str) -> bool:
    a = str(algo).strip().lower()
    return a in ("ppo", "q", "simple_pg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs", help="Root directory for run data.")
    ap.add_argument("--skill_library_dir", type=str, default="skills", help="Directory where skill paths are stored.")
    ap.add_argument("--loops", type=int, default=5, help="Number of curiosity loops.")
    ap.add_argument("--episodes_per_loop", type=int, default=100, help="Episodes per loop.")
    ap.add_argument("--max_steps", type=int, default=50, help="Max steps per episode.")
    ap.add_argument("--algo", type=str, default="ppo", help="Training algorithm for loop runs.")
    ap.add_argument("--policy_prefix", type=str, default="curious", help="Policy ID prefix.")
    ap.add_argument("--seed", type=int, default=123, help="Base seed for deterministic loop progression.")
    ap.add_argument("--sleep_s", type=float, default=1.0, help="Delay between loops.")
    ap.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    ap.add_argument("--log_file", type=str, default="", help="Optional log file path.")
    ap.add_argument(
        "--librarian_mode",
        type=str,
        default="behavioral_surgeon",
        choices=["behavioral_surgeon", "make_apprentice", "none"],
        help="Post-run librarian action.",
    )
    ap.add_argument("--force_verse", type=str, default="", help="Force a specific verse each loop (e.g. cliff_world).")
    ap.add_argument(
        "--train_agent",
        action="store_true",
        help="Force config train=true for the chosen algo.",
    )
    ap.add_argument(
        "--no_train_agent",
        action="store_true",
        help="Force config train=false for the chosen algo.",
    )
    ap.add_argument(
        "--good_dataset",
        type=str,
        default=os.path.join("models", "expert_datasets", "cliff_world.jsonl"),
        help="Good DNA dataset (used by failure_aware).",
    )
    ap.add_argument(
        "--bad_dataset",
        type=str,
        default=os.path.join("models", "expert_datasets", "cliff_death_transitions.jsonl"),
        help="Bad DNA dataset (used by failure_aware).",
    )
    ap.add_argument("--danger_temperature", type=float, default=1.8, help="failure_aware danger temperature.")
    ap.add_argument("--hard_block_threshold", type=float, default=0.97, help="failure_aware hard block threshold.")
    ap.add_argument("--caution_penalty_scale", type=float, default=0.7, help="failure_aware caution penalty scale.")
    ap.add_argument("--danger_prior", type=float, default=1.0, help="failure_aware danger prior smoothing.")
    ap.add_argument("--min_action_prob", type=float, default=1e-6, help="failure_aware minimum non-blocked action prob.")
    ap.add_argument("--cliff_reward_threshold", type=float, default=-50.0, help="Behavioral Surgeon death extraction threshold.")
    ap.add_argument("--death_similarity_threshold", type=float, default=0.95, help="Behavioral Surgeon death dedupe similarity.")
    ap.add_argument("--behavior_top_percent", type=float, default=20.0, help="Behavioral scoring top percent.")
    ap.add_argument(
        "--prioritize_cliff_signals",
        action="store_true",
        help="Prioritize cliff_world when recent runs show heavy cliff penalties.",
    )
    ap.add_argument(
        "--cliff_signal_threshold",
        type=int,
        default=10,
        help="Min recent cliff-penalty events to force cliff_world selection.",
    )
    ap.add_argument(
        "--cliff_signal_lookback_runs",
        type=int,
        default=5,
        help="How many recent runs to scan for cliff signals.",
    )
    ap.add_argument(
        "--multi_agent",
        action="store_true",
        help="Run a random agent alongside the learner for comparison.",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    skill_library_dir = Path(args.skill_library_dir).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    skill_library_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file).resolve() if args.log_file else (runs_root / "curiosity_loop.log")
    _configure_logging(args.log_level, log_file)
    logger = logging.getLogger("curiosity")

    trainer = Trainer(run_root=str(runs_root), schema_version="v1", auto_register_builtin=True)
    logger.info("Starting curiosity loop | loops=%s episodes_per_loop=%s", args.loops, args.episodes_per_loop)
    logger.info("runs_root=%s | skill_library_dir=%s", runs_root, skill_library_dir)

    try:
        for i in range(max(0, int(args.loops))):
            loop_idx = i + 1
            logger.info("Loop %s/%s", loop_idx, args.loops)

            if str(args.force_verse).strip():
                next_verse_name = str(args.force_verse).strip()
            else:
                next_verse_name = select_next_verse(
                    str(skill_library_dir),
                    runs_root=str(runs_root),
                    prefer_cliff_on_penalty=bool(args.prioritize_cliff_signals),
                    cliff_penalty_threshold=max(1, int(args.cliff_signal_threshold)),
                    lookback_runs=max(1, int(args.cliff_signal_lookback_runs)),
                )
            if next_verse_name is None:
                logger.warning("No verse selected. Ending loop.")
                break

            # Generate a shared run_id for this loop iteration
            run_id = new_id("loop_run")
            run_dir = runs_root / run_id

            verse_spec = VerseSpec(
                spec_version="v1",
                verse_name=next_verse_name,
                verse_version="0.1",
                seed=int(args.seed) + i,
                tags=[],
                params={},
            )

            # 1. Run the Learner
            policy_id = f"{args.algo}_{args.policy_prefix}_{next_verse_name}_loop_{loop_idx}"
            agent_spec = AgentSpec(
                spec_version="v1",
                policy_id=policy_id,
                policy_version="0.1",
                algo=str(args.algo),
                seed=int(args.seed) + i,
                config={},
            )
            cfg = dict(agent_spec.config or {})
            if args.train_agent and args.no_train_agent:
                raise ValueError("Cannot set both --train_agent and --no_train_agent.")
            train_enabled = _default_train_enabled(args.algo)
            if args.train_agent:
                train_enabled = True
            if args.no_train_agent:
                train_enabled = False
            cfg["train"] = bool(train_enabled)

            if str(args.algo).strip().lower() == "failure_aware":
                cfg["dataset_path"] = str(args.good_dataset)
                cfg["bad_dna_path"] = str(args.bad_dataset)
                cfg["danger_temperature"] = float(args.danger_temperature)
                cfg["hard_block_threshold"] = float(args.hard_block_threshold)
                cfg["caution_penalty_scale"] = float(args.caution_penalty_scale)
                cfg["danger_prior"] = float(args.danger_prior)
                cfg["min_action_prob"] = float(args.min_action_prob)
            agent_spec = AgentSpec(
                spec_version=agent_spec.spec_version,
                policy_id=agent_spec.policy_id,
                policy_version=agent_spec.policy_version,
                algo=agent_spec.algo,
                seed=agent_spec.seed,
                config=cfg,
            )

            logger.info(
                "Executing learner run | verse=%s policy_id=%s episodes=%s",
                next_verse_name,
                policy_id,
                args.episodes_per_loop,
            )

            try:
                trainer.run(
                    verse_spec=verse_spec,
                    agent_spec=agent_spec,
                    episodes=int(args.episodes_per_loop),
                    max_steps=int(args.max_steps),
                    seed=int(args.seed) + i,
                    run_id=run_id,
                )
            except Exception:
                logger.exception("Learner run failed for loop %s; continuing.", loop_idx)
                continue

            # 2. Optionally run a Random Baseline in the same run_id
            if args.multi_agent:
                logger.info("Executing random baseline run for comparison.")
                random_spec = AgentSpec(
                    spec_version="v1",
                    policy_id=f"random_baseline_{next_verse_name}",
                    policy_version="0.1",
                    algo="random",
                    seed=int(args.seed) + i,
                )
                try:
                    trainer.run(
                        verse_spec=verse_spec,
                        agent_spec=random_spec,
                        episodes=int(args.episodes_per_loop),
                        max_steps=int(args.max_steps),
                        seed=int(args.seed) + i,
                        run_id=run_id,
                    )
                except Exception:
                    logger.exception("Random baseline failed for loop %s; continuing.", loop_idx)

            logger.info("Run complete | run_id=%s run_dir=%s", run_id, run_dir)

            _run_librarian_hook(
                mode=str(args.librarian_mode),
                run_id=run_id,
                run_dir=run_dir,
                python_exe=sys.executable,
                cliff_reward_threshold=float(args.cliff_reward_threshold),
                death_similarity_threshold=float(args.death_similarity_threshold),
                behavior_top_percent=float(args.behavior_top_percent),
            )

            if float(args.sleep_s) > 0:
                time.sleep(float(args.sleep_s))
    except KeyboardInterrupt:
        logger.info("Loop interrupted by user.")
    finally:
        logger.info("Curiosity loop finished.")


if __name__ == "__main__":
    main()
