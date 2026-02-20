"""
Visualize a semantic bridge score over a short Chess V2 rollout.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.semantic_bridge import StrategyValueNetwork, extract_strategy_features
from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


register_builtin()


def train_quick_bridge(*, episodes: int, max_steps: int, seed: int) -> StrategyValueNetwork:
    print(f"Training quick bridge on Go V2 ({episodes} episodes)...")
    rng = random.Random(seed)
    bridge = StrategyValueNetwork(input_dim=7)

    for _ in range(int(episodes)):
        spec = VerseSpec(spec_version="v1", verse_name="go_world_v2", verse_version="0.1", params={})
        verse = create_verse(spec)
        obs = verse.reset().obs
        traj = []
        info: Dict[str, object] = {}

        for _ in range(int(max_steps)):
            legal = verse.legal_actions()
            if not legal:
                break
            step_result = verse.step(rng.choice(legal))
            obs = step_result.obs
            info = dict(step_result.info or {})
            traj.append(extract_strategy_features(obs))
            if bool(step_result.done):
                break

        reached_goal = bool(info.get("reached_goal", False))
        lost_game = bool(info.get("lost_game", False))
        ret = 1.0 if reached_goal else (-1.0 if lost_game else 0.0)
        for features in traj:
            bridge.train_step(features, ret)

    print("Bridge ready.")
    return bridge


def visualize_chess_game(
    bridge: StrategyValueNetwork,
    *,
    max_steps: int,
    pause_sec: float,
    seed: int,
    clear_screen: bool,
) -> None:
    print("Starting Chess V2 game with bridge analysis...")
    rng = random.Random(seed)
    spec = VerseSpec(spec_version="v1", verse_name="chess_world_v2", verse_version="0.1", params={"max_steps": max_steps})
    verse = create_verse(spec)
    obs = verse.reset().obs

    for t in range(int(max_steps)):
        if clear_screen:
            os.system("cls" if os.name == "nt" else "clear")
        print(f"--- Turn {t + 1} ---")
        print(verse.render())

        feats = extract_strategy_features(obs)
        val = float(bridge.predict(feats))
        print("")
        print("[Semantic Bridge Analysis]")
        print(f"  Score Delta: {feats[0] * 10:.1f}")
        print(f"  Pressure:    {feats[1] * 10:.1f}")
        print(f"  Risk:        {feats[2] * 10:.1f}")
        print(f"  Control:     {feats[4] * 10:.1f}")
        print(f"  Estimated Strategic Value: {val:.3f}")
        if val < -0.5:
            print("  CRITICAL SITUATION DETECTED")
        if val > 0.5:
            print("  ADVANTAGEOUS POSITION")

        legal = verse.legal_actions()
        if not legal:
            print("No legal actions available; ending run.")
            return
        action = rng.choice(legal)
        print(f"Agent action: {action}")
        time.sleep(max(0.0, float(pause_sec)))

        step_result = verse.step(action)
        obs = step_result.obs
        if bool(step_result.done):
            print(f"Game over. Result: {step_result.info}")
            return

    print("Reached max steps.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_episodes", type=int, default=100)
    ap.add_argument("--train_max_steps", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--pause_sec", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no_clear", action="store_true")
    args = ap.parse_args()

    bridge = train_quick_bridge(
        episodes=max(1, int(args.train_episodes)),
        max_steps=max(1, int(args.train_max_steps)),
        seed=int(args.seed),
    )
    visualize_chess_game(
        bridge,
        max_steps=max(1, int(args.max_steps)),
        pause_sec=max(0.0, float(args.pause_sec)),
        seed=int(args.seed),
        clear_screen=not bool(args.no_clear),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
