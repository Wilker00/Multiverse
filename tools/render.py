"""
tools/render.py

Tiny CLI to render a Verse using a random policy.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from core.agent_base import RandomAgent
from verses.registry import register_builtin as register_builtin_verses, create_verse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verse", type=str, default="line_world")
    ap.add_argument("--verse_version", type=str, default="0.1")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--goal_pos", type=int, default=8)
    args = ap.parse_args()

    register_builtin_verses()

    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=args.verse,
        verse_version=args.verse_version,
        seed=args.seed,
        tags=["render"],
        params={
            "goal_pos": args.goal_pos,
            "max_steps": args.max_steps,
            "step_penalty": -0.02,
        },
    )

    verse = create_verse(verse_spec)
    agent = RandomAgent(
        spec=AgentSpec(
            spec_version="v1",
            policy_id="random",
            policy_version="0.0",
            algo="random",
            seed=args.seed,
        ),
        observation_space=verse.observation_space,
        action_space=verse.action_space,
    )

    for ep in range(args.episodes):
        verse.seed(args.seed + ep)
        agent.seed(args.seed + ep)
        reset = verse.reset()
        print(verse.render(mode="ansi") or reset.obs)

        done = False
        steps = 0
        obs = reset.obs
        while not done and steps < args.max_steps:
            action_result = agent.act(obs)
            action = action_result.action
            step = verse.step(action)
            obs = step.obs
            done = bool(step.done or step.truncated)
            steps += 1

            frame = verse.render(mode="ansi")
            if frame is not None:
                print(frame)
            else:
                print(obs)
            time.sleep(max(0.0, args.sleep))

    verse.close()
    agent.close()


if __name__ == "__main__":
    main()




