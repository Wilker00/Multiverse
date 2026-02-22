"""Quick demo: render a maze_world episode."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verses.registry import register_builtin, create_verse
from core.types import VerseSpec

register_builtin()

spec = VerseSpec(
    spec_version="v1",
    verse_name="maze_world",
    verse_version="0.1",
    seed=77,
    params={"width": 7, "height": 7, "adr_enabled": False},
)
env = create_verse(spec)
env.seed(77)
result = env.reset()

print("=== INITIAL MAZE ===")
print(env.render(mode="ansi"))
print()
print("Observation keys :", list(result.obs.keys()))
print("Optimal solution  :", result.info["optimal_steps"], "steps")
print("Total cells       :", result.info["total_cells"])
print()

rng = random.Random(42)
names = ["N", "S", "W", "E"]
for i in range(10):
    a = rng.randint(0, 3)
    s = env.step(a)
    print(
        f"Step {i+1:2d}: action={names[a]}  "
        f"reward={s.reward:+.3f}  "
        f"bumped={str(s.info.get('bumped_wall', False)):5s}  "
        f"new_cell={str(s.info.get('new_cell', False)):5s}  "
        f"pos=({s.info.get('x')},{s.info.get('y')})"
    )

print()
print("=== AFTER 10 STEPS ===")
print(env.render(mode="ansi"))
print()
print("Coverage:", result.info["total_cells"], "cells total,", len(env._visited_cells), "visited")
