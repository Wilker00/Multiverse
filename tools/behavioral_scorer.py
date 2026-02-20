"""
tools/behavioral_scorer.py

Scores episode trajectories by behavior quality (efficiency + smoothness)
and exports top episodes as Golden DNA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_events(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"events file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def _group_by_episode(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ep = str(e.get("episode_id", "")).strip()
        if not ep:
            continue
        by_ep.setdefault(ep, []).append(e)
    for ep in by_ep:
        by_ep[ep].sort(key=lambda x: _safe_int(x.get("step_idx", 0)))
    return by_ep


def _obs_distance_to_goal(obs: Dict[str, Any], info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    # line/park: pos + goal
    if "pos" in obs and "goal" in obs:
        return abs(_safe_float(obs.get("goal")) - _safe_float(obs.get("pos")))
    # grid: x/y + goal_x/goal_y
    if "x" in obs and "y" in obs and "goal_x" in obs and "goal_y" in obs:
        return abs(_safe_float(obs.get("goal_x")) - _safe_float(obs.get("x"))) + abs(
            _safe_float(obs.get("goal_y")) - _safe_float(obs.get("y"))
        )
    # cliff: x/y with goal in step info
    if "x" in obs and "y" in obs and isinstance(info, dict) and "goal_x" in info and "goal_y" in info:
        return abs(_safe_float(info.get("goal_x")) - _safe_float(obs.get("x"))) + abs(
            _safe_float(info.get("goal_y")) - _safe_float(obs.get("y"))
        )
    # pursuit: agent + target
    if "agent" in obs and "target" in obs:
        return abs(_safe_float(obs.get("target")) - _safe_float(obs.get("agent")))
    return None


def _episode_success(ep_events: List[Dict[str, Any]]) -> bool:
    for e in ep_events:
        info = e.get("info") or {}
        if isinstance(info, dict) and info.get("reached_goal") is True:
            return True
    return False


def _episode_smoothness(ep_events: List[Dict[str, Any]]) -> float:
    actions: List[Any] = [e.get("action") for e in ep_events]
    if len(actions) <= 1:
        return 1.0
    switches = 0
    for i in range(1, len(actions)):
        if actions[i] != actions[i - 1]:
            switches += 1
    return max(0.0, 1.0 - (switches / float(len(actions) - 1)))


def _episode_efficiency(ep_events: List[Dict[str, Any]]) -> float:
    if not ep_events:
        return 0.0
    success = _episode_success(ep_events)
    if not success:
        return 0.0
    first_obs = ep_events[0].get("obs")
    first_info = ep_events[0].get("info")
    if not isinstance(first_obs, dict):
        return 0.0
    ideal = _obs_distance_to_goal(first_obs, first_info if isinstance(first_info, dict) else None)
    if ideal is None:
        return 0.0
    # +1 action to complete terminal move when needed.
    ideal_steps = max(1.0, float(ideal))
    steps = float(len(ep_events))
    return max(0.0, min(1.0, ideal_steps / max(1.0, steps)))


def _quality_score(
    *,
    return_sum: float,
    steps: int,
    success: bool,
    smoothness: float,
    efficiency: float,
    w_success: float,
    w_efficiency: float,
    w_smoothness: float,
    w_return: float,
) -> float:
    mean_reward = return_sum / float(max(1, steps))
    return (
        (w_success * (1.0 if success else 0.0))
        + (w_efficiency * efficiency)
        + (w_smoothness * smoothness)
        + (w_return * mean_reward)
    )


def _resolve_output(run_dir: str, out_path: str) -> str:
    if os.path.isabs(out_path):
        return out_path
    if os.sep in out_path or "/" in out_path:
        return os.path.normpath(out_path)
    return os.path.join(run_dir, out_path)


def _write_golden_dataset(
    *,
    by_ep: Dict[str, List[Dict[str, Any]]],
    episode_ids: List[str],
    out_path: str,
) -> int:
    rows = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for ep_id in episode_ids:
            evs = by_ep.get(ep_id, [])
            if not evs:
                continue
            for i, e in enumerate(evs):
                row: Dict[str, Any] = {
                    "episode_id": ep_id,
                    "step_idx": _safe_int(e.get("step_idx", i)),
                    "obs": e.get("obs"),
                    "action": e.get("action"),
                    "reward": _safe_float(e.get("reward", 0.0)),
                    "done": bool(e.get("done") or e.get("truncated")),
                    "next_obs": evs[i + 1].get("obs") if i + 1 < len(evs) else None,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--scores_out", type=str, default="behavior_scores.jsonl")
    ap.add_argument("--golden_out", type=str, default="golden_dna.jsonl")
    ap.add_argument("--top_percent", type=float, default=20.0)
    ap.add_argument("--min_quality", type=float, default=-1e9)
    ap.add_argument("--require_success", action="store_true")

    ap.add_argument("--w_success", type=float, default=1.0)
    ap.add_argument("--w_efficiency", type=float, default=1.5)
    ap.add_argument("--w_smoothness", type=float, default=1.0)
    ap.add_argument("--w_return", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"run_dir not found: {args.run_dir}")

    events = _load_events(args.run_dir)
    by_ep = _group_by_episode(events)
    if not by_ep:
        raise RuntimeError("No episodes found in run events.")

    scores: List[Dict[str, Any]] = []
    for ep_id, evs in by_ep.items():
        ret = sum(_safe_float(e.get("reward", 0.0)) for e in evs)
        steps = len(evs)
        success = _episode_success(evs)
        smooth = _episode_smoothness(evs)
        eff = _episode_efficiency(evs)
        q = _quality_score(
            return_sum=ret,
            steps=steps,
            success=success,
            smoothness=smooth,
            efficiency=eff,
            w_success=float(args.w_success),
            w_efficiency=float(args.w_efficiency),
            w_smoothness=float(args.w_smoothness),
            w_return=float(args.w_return),
        )
        scores.append(
            {
                "episode_id": ep_id,
                "steps": steps,
                "return_sum": ret,
                "success": success,
                "smoothness": smooth,
                "efficiency": eff,
                "quality": q,
            }
        )

    scores.sort(key=lambda x: float(x.get("quality", 0.0)), reverse=True)

    scores_out = _resolve_output(args.run_dir, args.scores_out)
    os.makedirs(os.path.dirname(scores_out) or ".", exist_ok=True)
    with open(scores_out, "w", encoding="utf-8") as out:
        for row in scores:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(scores)
    top_n = max(1, int(math.ceil((float(args.top_percent) / 100.0) * total)))

    selected = []
    for row in scores:
        if len(selected) >= top_n:
            break
        if float(row.get("quality", 0.0)) < float(args.min_quality):
            continue
        if args.require_success and not bool(row.get("success")):
            continue
        selected.append(str(row.get("episode_id")))

    golden_out = _resolve_output(args.run_dir, args.golden_out)
    rows = _write_golden_dataset(by_ep=by_ep, episode_ids=selected, out_path=golden_out)

    print("Behavioral scoring complete")
    print(f"run_dir        : {args.run_dir}")
    print(f"episodes       : {total}")
    print(f"scores_out     : {scores_out}")
    print(f"selected_eps   : {len(selected)}")
    print(f"golden_out     : {golden_out}")
    print(f"golden_rows    : {rows}")


if __name__ == "__main__":
    main()
