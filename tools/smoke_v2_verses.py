"""Manual smoke script for v2 verses (not part of automated pytest)."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


def run_smoke() -> None:
    register_builtin()

    for verse_name in ["go_world_v2", "chess_world_v2", "uno_world_v2"]:
        print(f"\n{'=' * 60}")
        print(f"TESTING: {verse_name}")
        print(f"{'=' * 60}")

        spec = VerseSpec(
            spec_version="v1",
            verse_name=verse_name,
            verse_version="0.1",
            seed=42,
            params={},
        )
        verse = create_verse(spec)
        obs = verse.reset().obs

        print(f"  action_space.n = {verse.action_space.n}")
        print(f"  legal_actions = {len(verse.legal_actions())} moves")
        print(f"  obs keys = {sorted(obs.keys()) if isinstance(obs, dict) else type(obs)}")
        print("\n  Initial state:")
        print("  " + verse.render().replace("\n", "\n  "))

        rng = random.Random(42)
        total_reward = 0.0
        for step_i in range(30):
            legal = verse.legal_actions()
            if not legal:
                break
            action = rng.choice(legal)
            sr = verse.step(action)
            total_reward += sr.reward
            if sr.done or sr.truncated:
                info = sr.info or {}
                print(
                    f"  Episode ended at step {step_i + 1}: "
                    f"reached={info.get('reached_goal', False)} "
                    f"lost={info.get('lost_game', False)} "
                    f"reason={info.get('reason', '?')}"
                )
                verse.reset()

        print(f"  After 30 steps: total_reward={total_reward:.3f}")
        print("  Strategy features from last obs:")
        last_obs = verse._make_obs() if hasattr(verse, "_make_obs") else {}
        for key in ["score_delta", "pressure", "risk", "tempo", "control", "resource"]:
            print(f"    {key} = {last_obs.get(key, '?')}")
        print("  PASS")


if __name__ == "__main__":
    run_smoke()
