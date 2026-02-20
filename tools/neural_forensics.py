"""
tools/neural_forensics.py

Neural Forensics for failure-aware cliff behavior.

Purpose:
- Diagnose whether danger masking is over-blocking safe path actions.
- Quantify where success drops (state/action choke points).
- Emit a machine-readable report for deployment/regression pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.failure_aware_agent import FailureAwareAgent
from core.types import AgentSpec, JSONValue, SpaceSpec
from orchestrator.evaluator import evaluate_run


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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


@dataclass
class CliffLayout:
    width: int
    height: int
    start_x: int
    start_y: int
    goal_x: int
    goal_y: int

    def is_cliff(self, x: int, y: int) -> bool:
        if y != self.height - 1:
            return False
        if x <= min(self.start_x, self.goal_x):
            return False
        if x >= max(self.start_x, self.goal_x):
            return False
        return True


def _infer_layout_from_events(events_path: str) -> CliffLayout:
    width = 12
    height = 4
    start_x = 0
    start_y = height - 1
    goal_x = width - 1
    goal_y = height - 1

    for row in _iter_jsonl(events_path):
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        width = max(3, _safe_int(info.get("width", width), width))
        height = max(2, _safe_int(info.get("height", height), height))
        start_x = _safe_int(info.get("start_x", start_x), start_x)
        start_y = _safe_int(info.get("start_y", start_y), start_y)
        goal_x = _safe_int(info.get("goal_x", goal_x), goal_x)
        goal_y = _safe_int(info.get("goal_y", goal_y), goal_y)
        # One row with metadata is enough.
        if info:
            break

    start_x = max(0, min(width - 1, start_x))
    start_y = max(0, min(height - 1, start_y))
    goal_x = max(0, min(width - 1, goal_x))
    goal_y = max(0, min(height - 1, goal_y))
    return CliffLayout(
        width=width,
        height=height,
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
    )


def _obs_for(x: int, y: int, t: int = 0) -> Dict[str, int]:
    return {"x": int(x), "y": int(y), "t": int(t)}


def _step_xy(layout: CliffLayout, x: int, y: int, action: int) -> Tuple[int, int]:
    if action == 0:
        return x, max(0, y - 1)
    if action == 1:
        return x, min(layout.height - 1, y + 1)
    if action == 2:
        return max(0, x - 1), y
    if action == 3:
        return min(layout.width - 1, x + 1), y
    return x, y


def _bfs_path(layout: CliffLayout, avoid_cliff: bool) -> Tuple[List[Tuple[int, int]], List[int]]:
    start = (layout.start_x, layout.start_y)
    goal = (layout.goal_x, layout.goal_y)
    q = deque([start])
    prev: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
    seen = {start}

    while q:
        node = q.popleft()
        if node == goal:
            break
        x, y = node
        for a in (0, 1, 2, 3):
            nx, ny = _step_xy(layout, x, y, a)
            nxt = (nx, ny)
            if nxt in seen:
                continue
            if avoid_cliff and layout.is_cliff(nx, ny) and nxt not in (start, goal):
                continue
            seen.add(nxt)
            prev[nxt] = (node, a)
            q.append(nxt)

    if goal not in seen:
        return [], []

    path_nodes: List[Tuple[int, int]] = []
    path_actions: List[int] = []
    cur = goal
    while cur != start:
        path_nodes.append(cur)
        p, a = prev[cur]
        path_actions.append(int(a))
        cur = p
    path_nodes.append(start)
    path_nodes.reverse()
    path_actions.reverse()
    return path_nodes, path_actions


def _make_agent(
    *,
    good_dataset: str,
    bad_dataset: str,
    danger_temperature: float,
    hard_block_threshold: float,
    caution_penalty_scale: float,
    danger_prior: float,
) -> FailureAwareAgent:
    spec = AgentSpec(
        spec_version="v1",
        policy_id="neural_forensics_failure_aware",
        policy_version="0.0",
        algo="failure_aware",
        seed=123,
        config={
            "danger_temperature": float(danger_temperature),
            "hard_block_threshold": float(hard_block_threshold),
            "caution_penalty_scale": float(caution_penalty_scale),
            "danger_prior": float(danger_prior),
        },
    )
    obs_space = SpaceSpec(type="dict", keys=["x", "y", "t"])
    act_space = SpaceSpec(type="discrete", n=4)
    agent = FailureAwareAgent(spec=spec, observation_space=obs_space, action_space=act_space)
    if os.path.isfile(good_dataset):
        agent.learn_from_dataset(good_dataset)
    if os.path.isfile(bad_dataset):
        agent.learn_from_bad_dataset(bad_dataset)
    return agent


def _safe_path_forensics(
    *,
    agent: FailureAwareAgent,
    layout: CliffLayout,
    min_safe_action_prob: float,
) -> Dict[str, Any]:
    safe_nodes, safe_actions = _bfs_path(layout, avoid_cliff=True)
    risky_nodes, risky_actions = _bfs_path(layout, avoid_cliff=False)
    safe_expected: Dict[Tuple[int, int], int] = {}
    for i in range(len(safe_actions)):
        safe_expected[safe_nodes[i]] = int(safe_actions[i])

    choke_points: List[Dict[str, Any]] = []
    blocked_count = 0
    low_prob_count = 0
    expected_probs: List[float] = []
    expected_dangers: List[float] = []

    for state, expected_action in safe_expected.items():
        x, y = state
        diag = agent.action_diagnostics(_obs_for(x, y, 0))
        probs = [float(v) for v in diag.get("sample_probs", [])]
        danger = [float(v) for v in diag.get("danger_scores", [])]
        blocked = [bool(v) for v in diag.get("blocked_actions", [])]
        if len(probs) < 4 or len(danger) < 4 or len(blocked) < 4:
            continue

        ep = float(probs[expected_action])
        ed = float(danger[expected_action])
        eb = bool(blocked[expected_action])
        expected_probs.append(ep)
        expected_dangers.append(ed)
        is_choke = bool(eb or ep < float(min_safe_action_prob))
        if eb:
            blocked_count += 1
        if ep < float(min_safe_action_prob):
            low_prob_count += 1
        if is_choke:
            choke_points.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "expected_action": int(expected_action),
                    "expected_prob": ep,
                    "expected_danger": ed,
                    "expected_blocked": eb,
                    "sample_probs": probs,
                    "danger_scores": danger,
                }
            )

    choke_points.sort(
        key=lambda r: (
            bool(r.get("expected_blocked", False)),
            1.0 - float(r.get("expected_prob", 0.0)),
            float(r.get("expected_danger", 0.0)),
        ),
        reverse=True,
    )

    return {
        "layout": {
            "width": int(layout.width),
            "height": int(layout.height),
            "start": [int(layout.start_x), int(layout.start_y)],
            "goal": [int(layout.goal_x), int(layout.goal_y)],
        },
        "safe_path_len": int(len(safe_actions)),
        "risky_path_len": int(len(risky_actions)),
        "safe_path_actions": [int(a) for a in safe_actions],
        "risky_path_actions": [int(a) for a in risky_actions],
        "safe_expected_steps": int(len(safe_expected)),
        "safe_expected_prob_mean": (sum(expected_probs) / float(len(expected_probs))) if expected_probs else 0.0,
        "safe_expected_danger_mean": (sum(expected_dangers) / float(len(expected_dangers))) if expected_dangers else 0.0,
        "safe_expected_blocked_count": int(blocked_count),
        "safe_expected_low_prob_count": int(low_prob_count),
        "safe_choke_count": int(len(choke_points)),
        "safe_choke_points_top": choke_points[:20],
    }


def _run_forensics(
    *,
    run_dir: Optional[str],
    agent: FailureAwareAgent,
    layout: CliffLayout,
    severe_reward_threshold: float,
) -> Dict[str, Any]:
    if not run_dir:
        return {}
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return {}

    per_episode: Dict[str, Dict[str, Any]] = {}
    falls = 0
    severe = 0
    chosen_high_danger = 0
    chosen_blocked = 0
    chosen_count = 0
    safe_expected_miss = 0
    safe_expected_total = 0
    repeats = 0

    safe_nodes, safe_actions = _bfs_path(layout, avoid_cliff=True)
    safe_expected: Dict[Tuple[int, int], int] = {}
    for i in range(len(safe_actions)):
        safe_expected[safe_nodes[i]] = int(safe_actions[i])

    for row in _iter_jsonl(events_path):
        ep_id = str(row.get("episode_id", ""))
        obs = row.get("obs")
        action = _safe_int(row.get("action", -1), -1)
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        reward = _safe_float(row.get("reward", 0.0))
        if info.get("fell_cliff") is True:
            falls += 1
        if reward <= float(severe_reward_threshold):
            severe += 1

        if not isinstance(obs, dict):
            continue
        x = _safe_int(obs.get("x", 0), 0)
        y = _safe_int(obs.get("y", 0), 0)
        diag = agent.action_diagnostics(obs)
        danger = [float(v) for v in diag.get("danger_scores", [])]
        blocked = [bool(v) for v in diag.get("blocked_actions", [])]
        if 0 <= action < len(danger):
            chosen_count += 1
            if float(danger[action]) >= 0.8:
                chosen_high_danger += 1
            if bool(blocked[action]):
                chosen_blocked += 1

        if (x, y) in safe_expected:
            safe_expected_total += 1
            if action != int(safe_expected[(x, y)]):
                safe_expected_miss += 1

        ep_bucket = per_episode.setdefault(ep_id, {"last_state": None, "repeat_count": 0})
        current_state = (x, y)
        if ep_bucket["last_state"] == current_state:
            ep_bucket["repeat_count"] = int(ep_bucket["repeat_count"]) + 1
        ep_bucket["last_state"] = current_state

    for meta in per_episode.values():
        if int(meta.get("repeat_count", 0)) >= 5:
            repeats += 1

    stats = evaluate_run(run_dir)
    return {
        "run_id": stats.run_id,
        "episodes": int(stats.episodes),
        "mean_return": float(stats.mean_return),
        "success_rate": (None if stats.success_rate is None else float(stats.success_rate)),
        "mean_steps": float(stats.mean_steps),
        "cliff_fall_events": int(falls),
        "severe_penalty_events": int(severe),
        "chosen_actions": int(chosen_count),
        "chosen_high_danger_rate": (float(chosen_high_danger) / float(chosen_count)) if chosen_count else 0.0,
        "chosen_blocked_rate": (float(chosen_blocked) / float(chosen_count)) if chosen_count else 0.0,
        "safe_expected_miss_rate": (float(safe_expected_miss) / float(safe_expected_total)) if safe_expected_total else 0.0,
        "episodes_with_repeat_loops": int(repeats),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="", help="Optional run directory to analyze.")
    ap.add_argument("--good_dataset", type=str, default=os.path.join("models", "expert_datasets", "cliff_world.jsonl"))
    ap.add_argument(
        "--bad_dataset",
        type=str,
        default=os.path.join("models", "expert_datasets", "cliff_death_transitions.jsonl"),
    )
    ap.add_argument("--danger_temperature", type=float, default=1.8)
    ap.add_argument("--hard_block_threshold", type=float, default=0.97)
    ap.add_argument("--caution_penalty_scale", type=float, default=0.7)
    ap.add_argument("--danger_prior", type=float, default=1.0)
    ap.add_argument("--severe_reward_threshold", type=float, default=-50.0)
    ap.add_argument("--min_safe_action_prob", type=float, default=0.12)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    if not os.path.isfile(args.good_dataset):
        raise FileNotFoundError(f"good_dataset not found: {args.good_dataset}")
    if not os.path.isfile(args.bad_dataset):
        raise FileNotFoundError(f"bad_dataset not found: {args.bad_dataset}")

    run_dir = str(args.run_dir).strip() or None
    if run_dir and not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    if run_dir:
        events_path = os.path.join(run_dir, "events.jsonl")
        if not os.path.isfile(events_path):
            raise FileNotFoundError(f"events.jsonl not found under run_dir: {run_dir}")
        layout = _infer_layout_from_events(events_path)
    else:
        layout = CliffLayout(width=12, height=4, start_x=0, start_y=3, goal_x=11, goal_y=3)

    agent = _make_agent(
        good_dataset=args.good_dataset,
        bad_dataset=args.bad_dataset,
        danger_temperature=float(args.danger_temperature),
        hard_block_threshold=float(args.hard_block_threshold),
        caution_penalty_scale=float(args.caution_penalty_scale),
        danger_prior=float(args.danger_prior),
    )
    try:
        safe_path = _safe_path_forensics(
            agent=agent,
            layout=layout,
            min_safe_action_prob=float(args.min_safe_action_prob),
        )
        run_stats = _run_forensics(
            run_dir=run_dir,
            agent=agent,
            layout=layout,
            severe_reward_threshold=float(args.severe_reward_threshold),
        )
    finally:
        agent.close()

    report = {
        "forensics_type": "failure_aware_cliff",
        "config": {
            "danger_temperature": float(args.danger_temperature),
            "hard_block_threshold": float(args.hard_block_threshold),
            "caution_penalty_scale": float(args.caution_penalty_scale),
            "danger_prior": float(args.danger_prior),
            "min_safe_action_prob": float(args.min_safe_action_prob),
        },
        "datasets": {
            "good_dataset": args.good_dataset,
            "bad_dataset": args.bad_dataset,
        },
        "safe_path_forensics": safe_path,
        "run_forensics": run_stats,
    }

    out_json = str(args.out_json).strip()
    if not out_json and run_dir:
        out_json = os.path.join(run_dir, "neural_forensics.json")
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("Neural Forensics complete")
    print(f"safe_path_len                 : {safe_path.get('safe_path_len')}")
    print(f"safe_expected_blocked_count   : {safe_path.get('safe_expected_blocked_count')}")
    print(f"safe_expected_low_prob_count  : {safe_path.get('safe_expected_low_prob_count')}")
    print(f"safe_choke_count              : {safe_path.get('safe_choke_count')}")
    if run_stats:
        print(f"run_id                        : {run_stats.get('run_id')}")
        print(f"run_success_rate              : {run_stats.get('success_rate')}")
        print(f"run_mean_return               : {run_stats.get('mean_return')}")
        print(f"safe_expected_miss_rate       : {run_stats.get('safe_expected_miss_rate')}")
        print(f"chosen_high_danger_rate       : {run_stats.get('chosen_high_danger_rate')}")
        print(f"chosen_blocked_rate           : {run_stats.get('chosen_blocked_rate')}")
    if out_json:
        print(f"report_json                   : {out_json}")


if __name__ == "__main__":
    main()
