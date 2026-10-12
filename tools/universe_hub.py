"""
tools/universe_hub.py

Terminal map of verses, the central memory hub, and live traffic.

This is a view over ``core.live_state``. It does not train agents. The
placeholder galaxy bars are gone: rows come from run artifacts and concept tags.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from core.live_state import build_live_snapshot, map_verse_rows


def get_terminal_size():
    import shutil

    return shutil.get_terminal_size((80, 24))


def color_text(text: str, color_code: str) -> str:
    return f"\033[{color_code}m{text}\033[0m"


def _bar(value: float, width: int = 10) -> str:
    clamped = max(0.0, min(1.0, float(value)))
    filled = int(round(clamped * width))
    return ("#" * filled) + ("-" * (width - filled))


def format_map_lines(snapshot: Dict[str, Any], *, width: int = 80) -> List[str]:
    """Plain map text. Color is applied by ``render``."""
    counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
    memory = snapshot.get("memory") if isinstance(snapshot.get("memory"), dict) else {}
    flows = snapshot.get("flows") if isinstance(snapshot.get("flows"), dict) else {}
    health = snapshot.get("health") if isinstance(snapshot.get("health"), dict) else {}
    lines: List[str] = []
    lines.append("MULTIVERSE MAP")
    lines.append(
        "verses={registered} active_runs={active} memory_rows={rows} transfers={transfers}".format(
            registered=int(counts.get("registered_verses", 0) or 0),
            active=int(counts.get("active_runs", 0) or 0),
            rows=int(memory.get("row_count", 0) or 0),
            transfers=int(counts.get("recent_transfers", 0) or 0),
        )
    )
    lines.append("-" * max(20, int(width)))
    lines.append("[CENTRAL MEMORY]")
    exists = "present" if bool(memory.get("exists")) else "missing"
    lines.append(
        f"  hub {exists}  bytes={int(memory.get('bytes', 0) or 0)}  "
        f"rows={int(memory.get('row_count', 0) or 0)}"
        + (" (capped)" if bool(memory.get("row_count_capped")) else "")
    )
    lines.append("[VERSES]")
    rows = map_verse_rows(snapshot)
    if not rows:
        lines.append("  no verses registered")
    for row in rows:
        concepts = ",".join(list(row.get("concept_tags") or [])[:4]) or "task"
        status = str(row.get("status") or "idle")
        reward = float(row.get("reward_sum", 0.0) or 0.0)
        activity = max(
            int(row.get("recall_count", 0) or 0),
            int(row.get("transfer_count", 0) or 0),
            int(row.get("run_count", 0) or 0),
        )
        scale = 0.0 if activity <= 0 else min(1.0, activity / 5.0)
        name = str(row.get("verse_name") or "unknown")[:22]
        lines.append(
            f"  {status:<6} {name:<22} [{_bar(scale)}] "
            f"runs={int(row.get('run_count', 0) or 0)} "
            f"reward={reward:.2f} "
            f"ok={int(row.get('success_count', 0) or 0)} "
            f"{concepts}"
        )
    lines.append("[FLOWS]")
    retrievals = list(flows.get("retrievals") or [])
    transfers = list(flows.get("transfers") or [])
    safety = list(flows.get("safety") or [])
    if not retrievals and not transfers and not safety:
        lines.append("  no recent recall, transfer, or safety traffic")
    for item in retrievals[-6:]:
        lines.append(
            f"  recall   {item.get('verse_name', '')} "
            f"matches={item.get('match_count', 0)} used={item.get('used', False)}"
        )
    for item in transfers[-6:]:
        lines.append(
            f"  transfer {item.get('source_verse', '')} -> {item.get('target_verse', '')} "
            f"{item.get('decision', '')} conf={item.get('confidence')}"
        )
    for item in safety[-6:]:
        lines.append(f"  safety   {item.get('verse_name', '')} mode={item.get('mode', '')}")
    lines.append("[HEALTH]")
    providers = list(health.get("providers") or [])
    if not providers:
        lines.append("  no sim providers reported")
    for provider in providers:
        flag = "up" if bool(provider.get("available")) else "down"
        lines.append(
            f"  {flag:<4} {str(provider.get('provider_id', '')):<16} "
            f"implemented={bool(provider.get('implemented'))}"
        )
    return lines


class UniverseHub:
    def __init__(self, runs_root: str = "runs", memory_root: str = "central_memory"):
        self.runs_root = runs_root
        self.memory_root = memory_root
        self.start_time = time.time()

    def snapshot(self) -> Dict[str, Any]:
        return build_live_snapshot(runs_root=self.runs_root, memory_root=self.memory_root)

    def render_text(self, *, width: int | None = None) -> str:
        tw = int(width or get_terminal_size()[0])
        snapshot = self.snapshot()
        body = format_map_lines(snapshot, width=tw)
        uptime = int(time.time() - self.start_time)
        header = [
            "=" * tw,
            "MULTIVERSE LIVE MAP".center(tw),
            "=" * tw,
            f"Uptime: {uptime}s | runs: {self.runs_root} | memory: {self.memory_root}",
        ]
        footer = ["=" * tw, "[Q] Quit | refresh follows --refresh_sec"]
        return "\n".join(header + body + footer)

    def render(self) -> None:
        tw, _th = get_terminal_size()
        os.system("cls" if os.name == "nt" else "clear")
        text = self.render_text(width=tw)
        for line in text.splitlines():
            if line.startswith("  active"):
                print(color_text(line, "1;32"))
            elif line.startswith("  transfer"):
                print(color_text(line, "1;35"))
            elif line.startswith("  safety"):
                print(color_text(line, "1;33"))
            elif line.startswith("  recall"):
                print(color_text(line, "1;36"))
            elif line.startswith("[") or line.startswith("MULTIVERSE"):
                print(color_text(line, "1;34"))
            else:
                print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live terminal map of Multiverse verses and memory traffic.")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--memory_root", type=str, default="central_memory")
    ap.add_argument("--refresh_sec", type=float, default=2.0)
    ap.add_argument(
        "--once",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render once and exit (useful for CI/health checks).",
    )
    args = ap.parse_args()

    hub = UniverseHub(runs_root=str(args.runs_root), memory_root=str(args.memory_root))
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
