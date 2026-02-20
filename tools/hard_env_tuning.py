"""
tools/hard_env_tuning.py

Hard-environment tuning utilities:
- Curriculum shaping for noisy labyrinth/pursuit variants.
- Optional auxiliary shaping in labyrinth via wall_follow_bonus.
- Specialist ensemble training with diverse exploration profiles.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer


@dataclass
class CurriculumStage:
    stage: int
    noise: float
    episodes: int
    max_steps: int
    vparams: Dict[str, Any]


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _default_verse_params(verse: str, max_steps: int) -> Dict[str, Any]:
    v = str(verse).strip().lower()
    params: Dict[str, Any] = {"adr_enabled": False, "max_steps": int(max_steps)}
    if v == "labyrinth_world":
        params.update(
            {
                "width": 15,
                "height": 11,
                "action_noise": 0.0,
                "battery_capacity": 80,
                "wall_follow_bonus": 0.0,
            }
        )
    elif v == "pursuit_world":
        params.update({"lane_len": 9, "step_penalty": -0.01})
    return params


def _build_curriculum(
    *,
    verse: str,
    stages: int,
    episodes_per_stage: int,
    max_steps: int,
    noise_start: float,
    noise_end: float,
    wall_follow_bonus: float,
) -> List[CurriculumStage]:
    s = max(1, int(stages))
    out: List[CurriculumStage] = []
    for i in range(s):
        t = 0.0 if s == 1 else float(i) / float(s - 1)
        noise = float(noise_start) + (float(noise_end) - float(noise_start)) * t
        vp = _default_verse_params(verse, max_steps)
        if verse == "labyrinth_world":
            vp["action_noise"] = max(0.0, min(0.5, noise))
            vp["wall_follow_bonus"] = max(0.0, float(wall_follow_bonus))
        elif verse == "pursuit_world":
            # Reuse existing param channel for mild stochasticity pressure.
            vp["step_penalty"] = -0.01 - 0.02 * max(0.0, noise)
        out.append(
            CurriculumStage(
                stage=i + 1,
                noise=float(noise),
                episodes=int(episodes_per_stage),
                max_steps=int(max_steps),
                vparams=vp,
            )
        )
    return out


def run_curriculum(
    *,
    trainer: Trainer,
    verse: str,
    algo: str,
    seed: int,
    stages: List[CurriculumStage],
    base_agent_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for stage in stages:
        verse_spec = VerseSpec(
            spec_version="v1",
            verse_name=verse,
            verse_version="0.1",
            seed=int(seed + stage.stage),
            tags=["hard_tuning", "curriculum", f"stage_{stage.stage}"],
            params=dict(stage.vparams),
        )
        cfg = dict(base_agent_config)
        cfg["train"] = True
        agent_spec = AgentSpec(
            spec_version="v1",
            policy_id=f"{algo}_curriculum_stage_{stage.stage}",
            policy_version="0.1",
            algo=algo,
            seed=int(seed + stage.stage),
            tags=["hard_tuning", "curriculum"],
            config=cfg,
        )
        res = trainer.run(
            verse_spec=verse_spec,
            agent_spec=agent_spec,
            episodes=int(stage.episodes),
            max_steps=int(stage.max_steps),
            seed=int(seed + stage.stage),
        )
        run_id = str(res.get("run_id", ""))
        stats = evaluate_run(os.path.join(trainer.run_root, run_id))
        history.append(
            {
                "stage": int(stage.stage),
                "noise": float(stage.noise),
                "run_id": run_id,
                "mean_return": float(stats.mean_return),
                "success_rate": stats.success_rate,
                "mean_steps": float(stats.mean_steps),
                "episodes": int(stats.episodes),
            }
        )
        print(
            f"[curriculum] stage={stage.stage} noise={stage.noise:.3f} run={run_id} "
            f"mean_return={float(stats.mean_return):.3f} success={stats.success_rate}"
        )
    return history


def _specialist_profiles(algo: str, n: int) -> List[Dict[str, Any]]:
    if str(algo).strip().lower() != "q":
        return [{} for _ in range(max(1, int(n)))]

    # Diverse exploration strategies for MoE routing diversity.
    profiles = [
        {"lr": 0.10, "gamma": 0.99, "epsilon_start": 1.00, "epsilon_min": 0.10, "epsilon_decay": 0.998},
        {"lr": 0.08, "gamma": 0.98, "epsilon_start": 0.90, "epsilon_min": 0.05, "epsilon_decay": 0.996},
        {"lr": 0.12, "gamma": 0.97, "epsilon_start": 1.00, "epsilon_min": 0.02, "epsilon_decay": 0.994},
        {"lr": 0.06, "gamma": 0.995, "epsilon_start": 0.70, "epsilon_min": 0.05, "epsilon_decay": 0.997},
        {"lr": 0.15, "gamma": 0.96, "epsilon_start": 1.00, "epsilon_min": 0.15, "epsilon_decay": 0.999},
    ]
    return profiles[: max(1, int(n))]


def train_ensemble(
    *,
    trainer: Trainer,
    verse: str,
    algo: str,
    episodes: int,
    max_steps: int,
    seed: int,
    base_verse_params: Dict[str, Any],
    base_agent_config: Dict[str, Any],
    specialists: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    profiles = _specialist_profiles(algo, specialists)
    for idx, p in enumerate(profiles):
        cfg = dict(base_agent_config)
        cfg.update(p)
        cfg["train"] = True
        verse_spec = VerseSpec(
            spec_version="v1",
            verse_name=verse,
            verse_version="0.1",
            seed=int(seed + 100 + idx),
            tags=["hard_tuning", "ensemble", f"specialist_{idx+1}"],
            params=dict(base_verse_params),
        )
        agent_spec = AgentSpec(
            spec_version="v1",
            policy_id=f"{algo}_specialist_{idx+1}",
            policy_version="0.1",
            algo=algo,
            seed=int(seed + 100 + idx),
            tags=["hard_tuning", "ensemble"],
            config=cfg,
        )
        res = trainer.run(
            verse_spec=verse_spec,
            agent_spec=agent_spec,
            episodes=int(episodes),
            max_steps=int(max_steps),
            seed=int(seed + 100 + idx),
        )
        run_id = str(res.get("run_id", ""))
        stats = evaluate_run(os.path.join(trainer.run_root, run_id))
        row = {
            "specialist": idx + 1,
            "profile": p,
            "run_id": run_id,
            "mean_return": float(stats.mean_return),
            "success_rate": stats.success_rate,
            "mean_steps": float(stats.mean_steps),
            "episodes": int(stats.episodes),
        }
        results.append(row)
        print(
            f"[ensemble] specialist={idx+1} run={run_id} mean_return={float(stats.mean_return):.3f} "
            f"success={stats.success_rate}"
        )

    results.sort(
        key=lambda x: (
            float(x.get("mean_return", 0.0)),
            float(x.get("success_rate", 0.0) or 0.0),
            -float(x.get("mean_steps", 0.0)),
        ),
        reverse=True,
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verse", type=str, default="labyrinth_world", choices=["labyrinth_world", "pursuit_world"])
    ap.add_argument("--algo", type=str, default="q")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--episodes_per_stage", type=int, default=40)
    ap.add_argument("--max_steps", type=int, default=180)
    ap.add_argument("--noise_start", type=float, default=0.0)
    ap.add_argument("--noise_end", type=float, default=0.20)
    ap.add_argument("--wall_follow_bonus", type=float, default=0.02)

    ap.add_argument("--ensemble", type=int, default=3, help="Number of specialist policies to train.")
    ap.add_argument("--ensemble_episodes", type=int, default=80)
    ap.add_argument("--skip_curriculum", action="store_true")
    ap.add_argument("--skip_ensemble", action="store_true")

    ap.add_argument("--out", type=str, default=os.path.join("models", "tuning", "hard_env_tuning_report.json"))
    args = ap.parse_args()

    trainer = Trainer(run_root=args.runs_root, schema_version="v1", auto_register_builtin=True)

    curriculum_stages = _build_curriculum(
        verse=args.verse,
        stages=int(args.stages),
        episodes_per_stage=int(args.episodes_per_stage),
        max_steps=int(args.max_steps),
        noise_start=float(args.noise_start),
        noise_end=float(args.noise_end),
        wall_follow_bonus=float(args.wall_follow_bonus),
    )

    base_verse_params = _default_verse_params(args.verse, int(args.max_steps))
    if args.verse == "labyrinth_world":
        base_verse_params["action_noise"] = float(args.noise_end)
        base_verse_params["wall_follow_bonus"] = max(0.0, float(args.wall_follow_bonus))

    base_agent_config: Dict[str, Any] = {}
    curriculum_history: List[Dict[str, Any]] = []
    ensemble_results: List[Dict[str, Any]] = []

    if not args.skip_curriculum:
        curriculum_history = run_curriculum(
            trainer=trainer,
            verse=args.verse,
            algo=args.algo,
            seed=int(args.seed),
            stages=curriculum_stages,
            base_agent_config=base_agent_config,
        )

    if not args.skip_ensemble:
        ensemble_results = train_ensemble(
            trainer=trainer,
            verse=args.verse,
            algo=args.algo,
            episodes=int(args.ensemble_episodes),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            base_verse_params=base_verse_params,
            base_agent_config=base_agent_config,
            specialists=int(args.ensemble),
        )

    out_payload = {
        "verse": args.verse,
        "algo": args.algo,
        "runs_root": args.runs_root,
        "curriculum": {
            "stages": [asdict(s) for s in curriculum_stages],
            "history": curriculum_history,
        },
        "ensemble": {
            "count": int(args.ensemble),
            "results": ensemble_results,
            "recommended_top_k": min(3, max(1, len(ensemble_results))),
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"Hard-environment tuning report written: {args.out}")


if __name__ == "__main__":
    main()
