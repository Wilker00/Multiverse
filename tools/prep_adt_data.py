"""
tools/prep_adt_data.py

Prepare Decision Transformer training tensors from run and dataset JSONL logs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.embeddings import obs_to_universal_vector


@dataclass
class _StepRow:
    step_idx: int
    obs: Any
    action: int
    reward: float
    done: bool
    success: bool


@dataclass
class _EpisodeRows:
    source_id: str
    run_id: str
    episode_id: str
    verse_name: str
    steps: List[_StepRow]

    @property
    def return_sum(self) -> float:
        return float(sum(float(s.reward) for s in self.steps))

    @property
    def success(self) -> bool:
        return any(bool(s.success) for s in self.steps)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _normalize_verse_name(row: Dict[str, Any]) -> str:
    verse = str(row.get("verse_name") or row.get("target_verse_name") or "").strip().lower()
    return verse


def _row_success_signal(row: Dict[str, Any]) -> bool:
    info = row.get("info")
    info = info if isinstance(info, dict) else {}
    if bool(info.get("reached_goal", False)):
        return True
    if bool(info.get("success", False)):
        return True
    if bool(row.get("reached_goal", False)):
        return True
    if bool(row.get("success", False)):
        return True
    return False


def _iter_run_event_paths(runs_root: str, max_runs: int) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[Tuple[float, str]] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        events_path = os.path.join(run_dir, "events.jsonl")
        if not os.path.isfile(events_path):
            continue
        mtime = _safe_float(os.path.getmtime(events_path), 0.0)
        out.append((mtime, events_path))
    out.sort(key=lambda t: t[0], reverse=True)
    if int(max_runs) > 0:
        out = out[: int(max_runs)]
    return [p for _, p in out]


def _discover_dataset_paths(dataset_paths: Iterable[str], dataset_dir: str) -> List[str]:
    out: List[str] = []
    for p in dataset_paths:
        pp = str(p).strip()
        if pp and os.path.isfile(pp):
            out.append(pp)
    ddir = str(dataset_dir or "").strip()
    if ddir and os.path.isdir(ddir):
        for name in sorted(os.listdir(ddir)):
            if not str(name).endswith(".jsonl"):
                continue
            fp = os.path.join(ddir, name)
            if os.path.isfile(fp):
                out.append(fp)
    # Stable dedupe, preserve first occurrence.
    seen = set()
    deduped: List[str] = []
    for p in out:
        rp = os.path.abspath(p)
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)
    return deduped


def _load_jsonl_into_episodes(
    *,
    path: str,
    source_id: str,
    verse_filter: Optional[set[str]],
) -> List[_EpisodeRows]:
    episodes: Dict[Tuple[str, str, str], List[_StepRow]] = {}
    next_step_by_episode: Dict[Tuple[str, str, str], int] = {}
    auto_ep = 0
    run_id_default = f"src:{os.path.basename(path)}"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            verse_name = _normalize_verse_name(row)
            if verse_filter and verse_name not in verse_filter:
                continue

            obs = row.get("obs")
            if obs is None:
                continue

            try:
                action = int(row.get("action"))
            except (TypeError, ValueError):
                continue
            if action < 0:
                continue

            reward = _safe_float(row.get("reward", 0.0), 0.0)
            done = bool(row.get("done", False) or row.get("truncated", False))
            success = bool(_row_success_signal(row))

            run_id = str(row.get("run_id") or run_id_default).strip()
            if not run_id:
                run_id = run_id_default

            episode_id_raw = row.get("episode_id")
            if episode_id_raw not in (None, ""):
                episode_id = str(episode_id_raw).strip()
                if not episode_id:
                    episode_id = f"ep_auto_{auto_ep:06d}"
            else:
                episode_id = f"ep_auto_{auto_ep:06d}"

            key = (run_id, episode_id, verse_name)
            default_step = next_step_by_episode.get(key, 0)
            step_idx = _safe_int(row.get("step_idx"), default_step)
            next_step_by_episode[key] = max(default_step, int(step_idx) + 1)

            episodes.setdefault(key, []).append(
                _StepRow(
                    step_idx=int(step_idx),
                    obs=obs,
                    action=int(action),
                    reward=float(reward),
                    done=bool(done),
                    success=bool(success),
                )
            )

            if episode_id_raw in (None, ""):
                if done:
                    auto_ep += 1

    out: List[_EpisodeRows] = []
    for (run_id, episode_id, verse_name), steps in episodes.items():
        steps.sort(key=lambda x: int(x.step_idx))
        out.append(
            _EpisodeRows(
                source_id=str(source_id),
                run_id=str(run_id),
                episode_id=str(episode_id),
                verse_name=str(verse_name),
                steps=steps,
            )
        )
    return out


def _compute_rtg(rewards: List[float], gamma: float) -> List[float]:
    out = [0.0 for _ in rewards]
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = float(rewards[i]) + float(gamma) * float(g)
        out[i] = float(g)
    return out


def _sample_label_from_actions(row: List[int]) -> int:
    for a in reversed(row):
        aa = int(a)
        if aa >= 0:
            return aa
    return -1


def _action_label_hist(rows: List[List[int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for r in rows:
        lab = _sample_label_from_actions(r)
        if lab < 0:
            continue
        out[lab] = int(out.get(lab, 0)) + 1
    return out


def prepare_adt_data(
    *,
    runs_root: str,
    out_path: str,
    context_len: int,
    chunk_stride: int,
    state_dim: int,
    max_timestep: int,
    gamma: float,
    top_return_pct: float,
    min_episode_steps: int,
    max_runs: int,
    success_only: bool = False,
    verse_filter: Optional[List[str]] = None,
    dataset_paths: Optional[List[str]] = None,
    dataset_dir: str = "",
    action_balance_mode: str = "none",
    action_balance_max_ratio: float = 3.0,
    action_balance_seed: int = 123,
    min_action_dim: int = 0,
) -> Dict[str, Any]:
    verse_allow = None
    if verse_filter:
        verse_allow = {str(v).strip().lower() for v in verse_filter if str(v).strip()}
        if not verse_allow:
            verse_allow = None

    episode_rows: List[_EpisodeRows] = []
    run_event_paths = _iter_run_event_paths(str(runs_root), int(max_runs))
    for events_path in run_event_paths:
        run_dir = os.path.dirname(events_path)
        episode_rows.extend(
            _load_jsonl_into_episodes(
                path=events_path,
                source_id=f"run:{os.path.basename(run_dir)}",
                verse_filter=verse_allow,
            )
        )

    discovered_datasets = _discover_dataset_paths(dataset_paths or [], dataset_dir)
    for p in discovered_datasets:
        episode_rows.extend(
            _load_jsonl_into_episodes(
                path=p,
                source_id=f"dataset:{os.path.basename(p)}",
                verse_filter=verse_allow,
            )
        )

    episodes_before_filtering = int(len(episode_rows))
    # Basic episode quality filtering.
    min_steps = max(1, int(min_episode_steps))
    episode_rows = [ep for ep in episode_rows if len(ep.steps) >= min_steps]
    episodes_after_min_steps = int(len(episode_rows))
    if bool(success_only):
        episode_rows = [ep for ep in episode_rows if bool(ep.success)]
    episodes_after_success_only = int(len(episode_rows))
    if not episode_rows:
        raise RuntimeError("no usable episodes found for ADT preparation")

    pct = max(0.0, min(1.0, float(top_return_pct)))
    if pct <= 0.0:
        raise ValueError("top_return_pct must be > 0")
    if pct < 1.0:
        episode_rows = sorted(episode_rows, key=lambda ep: float(ep.return_sum), reverse=True)
        keep_n = max(1, int(math.ceil(float(len(episode_rows)) * pct)))
        episode_rows = episode_rows[:keep_n]
    episodes_after_top_return = int(len(episode_rows))

    action_dim = 0
    for ep in episode_rows:
        for step in ep.steps:
            action_dim = max(int(action_dim), int(step.action) + 1)
    action_dim = max(int(action_dim), max(0, int(min_action_dim)))
    if action_dim <= 0:
        raise RuntimeError("action_dim inferred as 0; no valid discrete actions found")

    bos_token_id = int(action_dim)
    K = max(1, int(context_len))
    stride = int(chunk_stride)
    if stride <= 0:
        stride = K
    max_t = max(1, int(max_timestep))
    gamma = max(0.0, min(1.0, float(gamma)))

    states_all: List[List[List[float]]] = []
    rtg_all: List[List[float]] = []
    prev_actions_all: List[List[int]] = []
    actions_all: List[List[int]] = []
    timesteps_all: List[List[int]] = []
    mask_all: List[List[float]] = []

    for ep in episode_rows:
        rewards = [float(s.reward) for s in ep.steps]
        actions = [int(s.action) for s in ep.steps]
        step_indices = [int(s.step_idx) for s in ep.steps]
        rtg = _compute_rtg(rewards, gamma=gamma)
        states = [obs_to_universal_vector(s.obs, dim=int(state_dim)) for s in ep.steps]

        n = len(ep.steps)
        for start in range(0, n, stride):
            end = min(n, start + K)
            if end <= start:
                continue

            cur_states = [[0.0] * int(state_dim) for _ in range(K)]
            cur_rtg = [0.0 for _ in range(K)]
            cur_prev = [int(bos_token_id) for _ in range(K)]
            cur_actions = [-100 for _ in range(K)]
            cur_t = [0 for _ in range(K)]
            cur_mask = [0.0 for _ in range(K)]

            for j in range(end - start):
                idx = start + j
                cur_states[j] = list(states[idx])
                cur_rtg[j] = float(rtg[idx])
                cur_actions[j] = int(actions[idx])
                cur_prev[j] = int(bos_token_id) if idx == 0 else int(actions[idx - 1])
                cur_t[j] = min(max_t - 1, max(0, int(step_indices[idx])))
                cur_mask[j] = 1.0

            states_all.append(cur_states)
            rtg_all.append(cur_rtg)
            prev_actions_all.append(cur_prev)
            actions_all.append(cur_actions)
            timesteps_all.append(cur_t)
            mask_all.append(cur_mask)

    if not states_all:
        raise RuntimeError("prepared dataset is empty after chunking")

    action_balance_mode_eff = str(action_balance_mode or "none").strip().lower()
    action_balance_meta: Dict[str, Any] = {
        "action_balance_mode": str(action_balance_mode_eff),
        "action_balance_applied": False,
        "action_balance_max_ratio": float(action_balance_max_ratio),
        "action_balance_seed": int(action_balance_seed),
    }
    hist_before = _action_label_hist(actions_all)
    action_balance_meta["action_balance_counts_before"] = {
        str(int(k)): int(v) for k, v in sorted(hist_before.items(), key=lambda kv: kv[0])
    }

    if action_balance_mode_eff == "cap_ratio" and len(hist_before) >= 2 and float(action_balance_max_ratio) > 0.0:
        nonzero = [int(v) for v in hist_before.values() if int(v) > 0]
        if nonzero:
            min_count = int(min(nonzero))
            cap = max(1, int(math.ceil(float(min_count) * float(action_balance_max_ratio))))
            by_label: Dict[int, List[int]] = {}
            unlabeled: List[int] = []
            for i, row_actions in enumerate(actions_all):
                lab = _sample_label_from_actions(row_actions)
                if lab < 0:
                    unlabeled.append(int(i))
                    continue
                by_label.setdefault(int(lab), []).append(int(i))

            rng = random.Random(int(action_balance_seed))
            keep: List[int] = list(unlabeled)
            for lab, idxs in by_label.items():
                if len(idxs) <= int(cap):
                    picked = list(idxs)
                else:
                    picked = sorted(rng.sample(idxs, int(cap)))
                keep.extend(picked)
            keep = sorted(set(keep))

            if keep:
                states_all = [states_all[i] for i in keep]
                rtg_all = [rtg_all[i] for i in keep]
                prev_actions_all = [prev_actions_all[i] for i in keep]
                actions_all = [actions_all[i] for i in keep]
                timesteps_all = [timesteps_all[i] for i in keep]
                mask_all = [mask_all[i] for i in keep]
                action_balance_meta["action_balance_applied"] = True
                action_balance_meta["action_balance_cap"] = int(cap)
                action_balance_meta["action_balance_kept_samples"] = int(len(keep))
    hist_after = _action_label_hist(actions_all)
    action_balance_meta["action_balance_counts_after"] = {
        str(int(k)): int(v) for k, v in sorted(hist_after.items(), key=lambda kv: kv[0])
    }

    payload = {
        "states": torch.tensor(states_all, dtype=torch.float32),
        "returns_to_go": torch.tensor(rtg_all, dtype=torch.float32),
        "prev_actions": torch.tensor(prev_actions_all, dtype=torch.long),
        "actions": torch.tensor(actions_all, dtype=torch.long),
        "timesteps": torch.tensor(timesteps_all, dtype=torch.long),
        "attention_mask": torch.tensor(mask_all, dtype=torch.float32),
        "meta": {
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "bos_token_id": int(bos_token_id),
            "context_len": int(K),
            "chunk_stride": int(stride),
            "max_timestep": int(max_timestep),
            "gamma": float(gamma),
            "episodes": int(len(episode_rows)),
            "samples": int(len(states_all)),
            "runs_root": str(runs_root),
            "run_event_paths": [str(p).replace("\\", "/") for p in run_event_paths],
            "dataset_paths": [str(p).replace("\\", "/") for p in discovered_datasets],
            "verse_filter": sorted(list(verse_allow)) if verse_allow else [],
            "top_return_pct": float(pct),
            "success_only": bool(success_only),
            "episodes_before_filtering": int(episodes_before_filtering),
            "episodes_after_min_steps": int(episodes_after_min_steps),
            "episodes_after_success_only": int(episodes_after_success_only),
            "episodes_after_top_return": int(episodes_after_top_return),
            **action_balance_meta,
        },
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(payload, out_path)
    return payload["meta"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--out_path", type=str, default=os.path.join("models", "adt_data.pt"))
    ap.add_argument("--context_len", type=int, default=20)
    ap.add_argument("--chunk_stride", type=int, default=0, help="0 => same as context_len")
    ap.add_argument("--state_dim", type=int, default=64)
    ap.add_argument("--max_timestep", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--top_return_pct", type=float, default=1.0)
    ap.add_argument("--success_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min_episode_steps", type=int, default=2)
    ap.add_argument("--max_runs", type=int, default=0)
    ap.add_argument("--verse_filter", action="append", default=None)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--dataset_dir", type=str, default="")
    ap.add_argument(
        "--action_balance_mode",
        type=str,
        default="none",
        choices=["none", "cap_ratio"],
        help="Optional sample-level balancing by last valid action label.",
    )
    ap.add_argument("--action_balance_max_ratio", type=float, default=3.0)
    ap.add_argument("--action_balance_seed", type=int, default=123)
    ap.add_argument(
        "--min_action_dim",
        type=int,
        default=0,
        help="Optional lower-bound for inferred action_dim (useful for warm-start compatibility).",
    )
    args = ap.parse_args()

    meta = prepare_adt_data(
        runs_root=str(args.runs_root),
        out_path=str(args.out_path),
        context_len=max(1, int(args.context_len)),
        chunk_stride=int(args.chunk_stride),
        state_dim=max(4, int(args.state_dim)),
        max_timestep=max(1, int(args.max_timestep)),
        gamma=max(0.0, min(1.0, float(args.gamma))),
        top_return_pct=max(0.0, min(1.0, float(args.top_return_pct))),
        success_only=bool(args.success_only),
        min_episode_steps=max(1, int(args.min_episode_steps)),
        max_runs=max(0, int(args.max_runs)),
        verse_filter=list(args.verse_filter or []),
        dataset_paths=list(args.dataset or []),
        dataset_dir=str(args.dataset_dir),
        action_balance_mode=str(args.action_balance_mode),
        action_balance_max_ratio=max(0.1, float(args.action_balance_max_ratio)),
        action_balance_seed=int(args.action_balance_seed),
        min_action_dim=max(0, int(args.min_action_dim)),
    )

    print("ADT data preparation complete")
    print(f"out_path     : {args.out_path}")
    print(f"episodes     : {int(meta['episodes'])}")
    print(f"samples      : {int(meta['samples'])}")
    print(f"state_dim    : {int(meta['state_dim'])}")
    print(f"action_dim   : {int(meta['action_dim'])}")
    print(f"context_len  : {int(meta['context_len'])}")


if __name__ == "__main__":
    main()
