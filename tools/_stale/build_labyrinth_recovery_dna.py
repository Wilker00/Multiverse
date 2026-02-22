"""
tools/build_labyrinth_recovery_dna.py

Generate recovery transitions by injecting noise and recording planner-guided
actions that recover toward the goal.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.planner_oracle import plan_actions_from_current_state
from core.types import JSONValue, VerseSpec
from verses.registry import create_verse, register_builtin


def _run_episode(
    *,
    verse: Any,
    rng: random.Random,
    noise_prob: float,
    max_steps: int,
    planner_max_expansions: int,
    planner_horizon: int,
    require_success: bool,
) -> Dict[str, Any]:
    reset = verse.reset()
    obs = reset.obs
    rows: List[Dict[str, Any]] = []
    noisy_episode = False
    success = False
    recovery_plan: List[int] = []
    nominal_plan: List[int] = plan_actions_from_current_state(
        verse=verse,
        horizon=max(1, int(max_steps)),
        max_expansions=max(100, int(planner_max_expansions)),
        avoid_terminal_failures=True,
    )
    t = 0

    while t < int(max_steps):
        action = None
        phase = "nominal"

        if recovery_plan:
            action = int(recovery_plan.pop(0))
            phase = "recovery"
        else:
            if rng.random() < float(noise_prob):
                action = int(rng.randrange(int(getattr(verse.action_space, "n", 5) or 5)))
                phase = "noise"
                noisy_episode = True
            else:
                if not nominal_plan:
                    nominal_plan = plan_actions_from_current_state(
                        verse=verse,
                        horizon=max(1, int(max_steps)),
                        max_expansions=max(100, int(planner_max_expansions)),
                        avoid_terminal_failures=True,
                    )
                if nominal_plan:
                    action = int(nominal_plan.pop(0))
                else:
                    action = int(rng.randrange(int(getattr(verse.action_space, "n", 5) or 5)))

        step = verse.step(int(action))
        info = step.info if isinstance(step.info, dict) else {}

        if phase == "noise" and not bool(step.done or step.truncated):
            recovery_plan = plan_actions_from_current_state(
                verse=verse,
                horizon=max(1, int(max_steps)),
                max_expansions=max(100, int(planner_max_expansions)),
                avoid_terminal_failures=True,
            )

        if phase == "recovery":
            rows.append(
                {
                    "obs": obs,
                    "action": int(action),
                    "reward": float(step.reward),
                    "done": bool(step.done or step.truncated),
                    "phase": "recovery",
                    "source": "labyrinth_recovery_planner",
                }
            )

        obs = step.obs
        t += 1
        if bool(info.get("reached_goal", False)):
            success = True
            break
        if bool(step.done or step.truncated):
            break

    return {
        "rows": rows if (noisy_episode and (success or (not require_success))) else [],
        "noisy_episode": bool(noisy_episode),
        "success": bool(success),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="models/expert_datasets/labyrinth_recovery.jsonl")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_steps", type=int, default=180)
    ap.add_argument("--noise_prob", type=float, default=0.20)
    ap.add_argument("--planner_horizon", type=int, default=8)
    ap.add_argument("--planner_max_expansions", type=int, default=12000)
    ap.add_argument("--battery_capacity", type=int, default=120)
    ap.add_argument("--require_success", action="store_true")
    args = ap.parse_args()

    register_builtin()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = random.Random(int(args.seed))

    kept_rows = 0
    noisy = 0
    noisy_success = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for ep in range(int(args.episodes)):
            ep_seed = int(args.seed) + ep
            spec = VerseSpec(
                spec_version="v1",
                verse_name="labyrinth_world",
                verse_version="0.1",
                seed=ep_seed,
                tags=["recovery_dna"],
                params={
                    "adr_enabled": False,
                    "width": 15,
                    "height": 11,
                    "max_steps": int(args.max_steps),
                    "action_noise": 0.0,
                    "battery_capacity": int(args.battery_capacity),
                },
            )
            verse = create_verse(spec)
            verse.seed(ep_seed)
            result = _run_episode(
                verse=verse,
                rng=rng,
                noise_prob=float(args.noise_prob),
                max_steps=int(args.max_steps),
                planner_max_expansions=int(args.planner_max_expansions),
                planner_horizon=int(args.planner_horizon),
                require_success=bool(args.require_success),
            )
            verse.close()
            if bool(result["noisy_episode"]):
                noisy += 1
            if bool(result["noisy_episode"]) and bool(result["success"]):
                noisy_success += 1
            for row in result["rows"]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_rows += 1

    print("Labyrinth recovery DNA built")
    print(f"out             : {args.out}")
    print(f"episodes        : {int(args.episodes)}")
    print(f"noisy_episodes  : {int(noisy)}")
    print(f"noisy_successes : {int(noisy_success)}")
    print(f"rows_written    : {int(kept_rows)}")


if __name__ == "__main__":
    main()
