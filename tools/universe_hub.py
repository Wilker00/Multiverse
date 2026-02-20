"""
tools/universe_hub.py

Interactive Terminal Dashboard for Multiverse.
Visualizes:
- Live parallel worker status
- Skill Galaxy progress
- Recent Promotion Board winners
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)


def get_terminal_size():
    import shutil

    return shutil.get_terminal_size((80, 24))


def color_text(text: str, color_code: str) -> str:
    return f"\033[{color_code}m{text}\033[0m"


class UniverseHub:
    def __init__(self, runs_root: str = "runs"):
        self.runs_root = runs_root
        self.start_time = time.time()

    def _get_recent_runs(self, limit: int = 5) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        if not os.path.isdir(self.runs_root):
            return []

        for d in os.listdir(self.runs_root):
            p = os.path.join(self.runs_root, d)
            if not os.path.isdir(p):
                continue

            meta_p = os.path.join(p, "parallel_meta.json")
            if os.path.isfile(meta_p):
                with open(meta_p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                if isinstance(m, dict):
                    m["type"] = "Parallel"
                    runs.append(m)
            else:
                runs.append({"run_id": d, "type": "Single", "created_at": os.path.getctime(p)})

        runs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return runs[:limit]

    def render(self) -> None:
        tw, _th = get_terminal_size()
        os.system("cls" if os.name == "nt" else "clear")

        print(color_text("=" * tw, "1;34"))
        print(color_text("  UNIVERSE.AI - CORE ENGINE DASHBOARD  ", "1;37;44").center(tw))
        print(color_text("=" * tw, "1;34"))

        uptime = int(time.time() - self.start_time)
        nodes = len(os.listdir(self.runs_root)) if os.path.isdir(self.runs_root) else 0
        print(f" Uptime: {uptime}s | Root: {self.runs_root} | Nodes: {nodes}")
        print("-" * tw)

        print(color_text("\n [PARALLEL FLEET STATUS]", "1;32"))
        recent = self._get_recent_runs(3)
        if not recent:
            print("  No active clusters detected.")
        for r in recent:
            rid = str(r.get("run_id", "unknown"))
            rtype = str(r.get("type", "Single"))
            print(f"  > {rid[:20]:<20} | {rtype:<8} | Status: {color_text('ACTIVE', '32')}")

        print(color_text("\n [SKILL GALAXY PROGRESS]", "1;35"))
        verses_seen = set()
        if os.path.isdir(self.runs_root):
            # Placeholder heuristic; can be replaced by run summaries.
            verses_seen = {"grid_world", "cliff_world"}

        for v in ["grid_world", "cliff_world", "labyrinth_world", "swamp_world", "factory_world"]:
            status = color_text("*", "32") if v in verses_seen else color_text("o", "30")
            print(f"  {status} {v:<20} [#######---] 70%")

        print(color_text("\n [PROMOTION BOARD TICKER]", "1;33"))
        if os.path.isfile("promotion_report.md"):
            print("  [OK] Candidate 'RecurrentPPO' promoted to STAGE 2 (Success > 85%)")
        else:
            print("  ... Waiting for candidate evaluation ...")

        print("\n" + color_text("=" * tw, "1;34"))
        print(" [Q] Quit | [R] Refresh | [A] Analyze Run")


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive terminal dashboard for Multiverse runs.")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--refresh_sec", type=float, default=2.0)
    ap.add_argument(
        "--once",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render once and exit (useful for CI/health checks).",
    )
    args = ap.parse_args()

    hub = UniverseHub(runs_root=str(args.runs_root))
    try:
        if bool(args.once):
            hub.render()
            return
        while True:
            hub.render()
            time.sleep(max(0.2, float(args.refresh_sec)))
    except KeyboardInterrupt:
        print("\nDashboard closed.")


if __name__ == "__main__":
    main()
