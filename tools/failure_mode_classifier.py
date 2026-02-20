"""
tools/failure_mode_classifier.py

Classify run failures into a taxonomy, tag events, and build oversampled
training slices for rare failure modes.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _episode_groups(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events:
        ep = str(ev.get("episode_id", "unknown"))
        by_ep.setdefault(ep, []).append(ev)
    for rows in by_ep.values():
        rows.sort(key=lambda x: int(x.get("step_idx", 0) or 0))
    return by_ep


def _classify_episode(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    final = rows[-1] if rows else {}
    info = dict(final.get("info") or {}) if isinstance(final.get("info"), dict) else {}
    se = dict(info.get("safe_executor") or {}) if isinstance(info.get("safe_executor"), dict) else {}

    reached_goal = bool(info.get("reached_goal", False))
    done = bool(final.get("done", False) or final.get("truncated", False))

    failure_mode = str(se.get("failure_mode", ""))
    if not failure_mode:
        if reached_goal:
            failure_mode = "success"
        elif info.get("fell_cliff"):
            failure_mode = "fell_cliff"
        elif info.get("fell_pit"):
            failure_mode = "fell_pit"
        elif info.get("hit_laser"):
            failure_mode = "hit_laser"
        elif info.get("battery_depleted"):
            failure_mode = "battery_depleted"
        elif info.get("hit_wall"):
            failure_mode = "hit_wall"
        elif info.get("hit_obstacle"):
            failure_mode = "hit_obstacle"
        elif info.get("battery_death"):
            failure_mode = "battery_death"
        elif final.get("truncated", False):
            failure_mode = "timeout_or_stuck"
        else:
            failure_mode = "unknown"

    if done and not reached_goal and failure_mode == "success":
        failure_mode = "unknown"

    danger_modes = {
        "fell_cliff",
        "fell_pit",
        "hit_laser",
        "battery_depleted",
        "hit_wall",
        "hit_obstacle",
        "battery_death",
        "safety_violation",
        "dangerous_outcome",
        "low_confidence_penalty",
        "high_danger_penalty",
    }
    danger_label = bool((not reached_goal) and (failure_mode in danger_modes))

    return {
        "episode_id": str(final.get("episode_id", "unknown")),
        "steps": int(len(rows)),
        "return_sum": float(sum(float(r.get("reward", 0.0) or 0.0) for r in rows)),
        "reached_goal": bool(reached_goal),
        "done": bool(done),
        "failure_mode": str(failure_mode),
        "danger_label": bool(danger_label),
        "failure_signals": list(se.get("failure_signals") or []),
        "planner_queries": int((se.get("counters") or {}).get("planner_queries", 0) if isinstance(se.get("counters"), dict) else 0),
        "rewinds": int((se.get("counters") or {}).get("rewinds", 0) if isinstance(se.get("counters"), dict) else 0),
    }


def _oversample_rows(
    *,
    episodes: Dict[str, List[Dict[str, Any]]],
    labels: Dict[str, Dict[str, Any]],
    include_success: bool,
    rng_seed: int,
    max_replication: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(rng_seed))
    mode_counts: Dict[str, int] = collections.Counter()
    for ep, meta in labels.items():
        if not include_success and bool(meta.get("reached_goal", False)):
            continue
        mode_counts[str(meta.get("failure_mode", "unknown"))] += 1

    if not mode_counts:
        return []
    max_count = max(mode_counts.values())

    out: List[Dict[str, Any]] = []
    for ep, rows in episodes.items():
        meta = labels.get(ep) or {}
        if not include_success and bool(meta.get("reached_goal", False)):
            continue
        mode = str(meta.get("failure_mode", "unknown"))
        c = max(1, int(mode_counts.get(mode, 1)))
        rep = min(int(max_replication), max(1, int(round(float(max_count) / float(c)))))
        for _ in range(rep):
            for r in rows:
                out.append(
                    {
                        "obs": r.get("obs"),
                        "action": r.get("action"),
                        "reward": float(r.get("reward", 0.0) or 0.0),
                        "done": bool(r.get("done", False) or r.get("truncated", False)),
                        "failure_mode": mode,
                        "danger_label": bool(meta.get("danger_label", False)),
                        "source": "failure_mode_oversample",
                    }
                )
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--events_file", type=str, default="events.jsonl")
    ap.add_argument("--out_modes", type=str, default="failure_modes.jsonl")
    ap.add_argument("--out_tagged_events", type=str, default="events.failure_tagged.jsonl")
    ap.add_argument("--out_oversampled", type=str, default="failure_mode_oversampled.jsonl")
    ap.add_argument("--include_success", action="store_true", help="Include successful episodes in oversampled output.")
    ap.add_argument("--max_replication", type=int, default=6)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    events_path = os.path.join(args.run_dir, args.events_file)
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"events file not found: {events_path}")

    events = list(_iter_jsonl(events_path))
    by_ep = _episode_groups(events)

    labels: Dict[str, Dict[str, Any]] = {}
    mode_counter: Dict[str, int] = collections.Counter()

    out_modes_path = os.path.join(args.run_dir, args.out_modes)
    with open(out_modes_path, "w", encoding="utf-8") as f:
        for ep, rows in sorted(by_ep.items()):
            row = _classify_episode(rows)
            labels[ep] = row
            mode_counter[str(row.get("failure_mode", "unknown"))] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Tag each event with episode-level failure mode.
    out_tagged_path = os.path.join(args.run_dir, args.out_tagged_events)
    with open(out_tagged_path, "w", encoding="utf-8") as f:
        for ev in events:
            ep = str(ev.get("episode_id", "unknown"))
            lbl = labels.get(ep) or {}
            info = dict(ev.get("info") or {}) if isinstance(ev.get("info"), dict) else {}
            se = dict(info.get("safe_executor") or {}) if isinstance(info.get("safe_executor"), dict) else {}
            se.setdefault("failure_mode", str(lbl.get("failure_mode", "unknown")))
            se["danger_label"] = bool(lbl.get("danger_label", False))
            info["safe_executor"] = se
            info["danger_label"] = bool(lbl.get("danger_label", False))
            ev2 = dict(ev)
            ev2["info"] = info
            ev2["failure_mode"] = str(lbl.get("failure_mode", "unknown"))
            ev2["danger_label"] = bool(lbl.get("danger_label", False))
            f.write(json.dumps(ev2, ensure_ascii=False) + "\n")

    oversampled = _oversample_rows(
        episodes=by_ep,
        labels=labels,
        include_success=bool(args.include_success),
        rng_seed=int(args.seed),
        max_replication=max(1, int(args.max_replication)),
    )
    out_over_path = os.path.join(args.run_dir, args.out_oversampled)
    with open(out_over_path, "w", encoding="utf-8") as f:
        for row in oversampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Failure modes written: {out_modes_path}")
    print(f"Tagged events written: {out_tagged_path}")
    print(f"Oversampled rows written: {out_over_path} ({len(oversampled)})")
    print("Mode counts:")
    for mode, cnt in sorted(mode_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {mode}: {cnt}")
    total_fail = sum(v for k, v in mode_counter.items() if k != "success")
    if total_fail > 0:
        top_mode = max((k for k in mode_counter.keys() if k != "success"), key=lambda m: mode_counter[m], default="")
        if top_mode:
            print("Recommended intervention:")
            if top_mode in ("timeout_or_stuck", "unknown"):
                print("- Add corner-escape and anti-loop curriculum episodes.")
            elif top_mode in ("low_confidence_penalty", "planner_timeout_or_no_plan"):
                print("- Increase planner horizon or add low-noise warmup curriculum.")
            elif top_mode in (
                "fell_cliff",
                "fell_pit",
                "hit_laser",
                "battery_depleted",
                "hit_wall",
                "hit_obstacle",
                "battery_death",
            ):
                print("- Oversample this mode and retrain failure-aware mask.")


if __name__ == "__main__":
    main()
