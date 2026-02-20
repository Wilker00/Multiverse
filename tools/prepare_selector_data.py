"""
tools/prepare_selector_data.py

The "Compiler" for the MicroSelector model. This script processes raw event
logs from multiple runs, extracts expert state-goal-skill tuples, and saves
them as a compressed PyTorch tensor file for training.
"""

import json
import torch
import os
import argparse
import random
from glob import glob
from typing import Dict, List, Any, Tuple, Optional

# Define the keys that constitute "state" and "goal" across all verses.
# This allows for flexible parsing of different observation structures.
STATE_KEYS = ['pos', 't', 'x', 'y', 'agent']
GOAL_KEYS = ['goal', 'goal_x', 'goal_y', 'target']


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _collect_episode_quality(events_path: str) -> Dict[str, Dict[str, Any]]:
    quality: Dict[str, Dict[str, Any]] = {}
    with open(events_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            ep = str(event.get("episode_id", "")).strip()
            if not ep:
                continue
            q = quality.setdefault(ep, {"return_sum": 0.0, "success": False})
            q["return_sum"] = float(q["return_sum"]) + _safe_float(event.get("reward", 0.0), 0.0)
            info = event.get("info")
            info = info if isinstance(info, dict) else {}
            if bool(info.get("reached_goal", False)):
                q["success"] = True
    return quality

def create_skill_vocab_from_lessons(lessons_dir: str) -> Dict[str, int]:
    """
    Creates a vocabulary by mapping .txt lesson filenames to integer IDs.
    This is more robust than relying on meta.json.
    """
    if not os.path.isdir(lessons_dir):
        print(f"Warning: Lessons directory not found at '{lessons_dir}'.")
        return {}
    
    lesson_files = sorted([f for f in os.listdir(lessons_dir) if f.endswith(".txt")])
    return {name: i for i, name in enumerate(lesson_files)}


def create_skill_vocab_from_runs(run_dirs: List[str]) -> Dict[str, int]:
    """
    Fallback vocabulary builder when lessons/ is unavailable.
    Uses verse_name from each run and maps to "<verse_name>.txt" skill IDs.
    """
    skill_names = set()
    for run_dir in run_dirs:
        events_path = os.path.join(run_dir, "events.jsonl")
        if not os.path.exists(events_path):
            continue
        with open(events_path, "r", encoding="utf-8") as f:
            first = f.readline()
        if not first:
            continue
        try:
            event = json.loads(first)
        except json.JSONDecodeError:
            continue
        verse_name = str(event.get("verse_name", "")).strip()
        if verse_name:
            skill_names.add(f"{verse_name}.txt")
    return {name: i for i, name in enumerate(sorted(skill_names))}


def _flatten_obs(obs: Dict[str, Any], keys: List[str]) -> List[float]:
    """Flattens a dictionary observation based on a list of keys."""
    flat_vector = []
    for key in keys:
        if key in obs:
            val = obs[key]
            if isinstance(val, (int, float)):
                flat_vector.append(float(val))
            elif isinstance(val, list):
                flat_vector.extend([float(v) for v in val])
    return flat_vector

def _pad_sequences(sequences: List[List[float]]) -> Tuple[torch.Tensor, int]:
    """Pads a list of sequences to the max length and returns a tensor."""
    if not sequences:
        return torch.empty(0), 0
    
    max_len = max(len(s) for s in sequences)
    padded = [s + [0.0] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.float32), max_len

def prepare_data(
    runs_root: str,
    lessons_dir: str,
    output_path: str = "training_batch.pt",
    reward_threshold: float = 0.8,
    verse_filter: Optional[List[str]] = None,
    episode_success_only: bool = False,
    episode_return_threshold: Optional[float] = None,
    episode_top_return_pct: Optional[float] = None,
    include_all_events_from_selected_episodes: bool = False,
    max_runs: int = 0,
    class_balance: bool = False,
    class_balance_max_per_class: int = 0,
    class_balance_seed: int = 42,
):
    """
    Processes all runs, extracts expert trajectories, and saves a training batch.
    """
    run_dirs = sorted([d for d in glob(f"{runs_root}/*") if os.path.isdir(d)])
    if int(max_runs) > 0:
        run_dirs = sorted(
            run_dirs,
            key=lambda d: os.path.getmtime(os.path.join(d, "events.jsonl")) if os.path.isfile(os.path.join(d, "events.jsonl")) else 0.0,
            reverse=True,
        )[: int(max_runs)]
        run_dirs = sorted(run_dirs)
    verse_allow = set(str(v).strip().lower() for v in (verse_filter or []) if str(v).strip())
    
    # The vocabulary of skills is derived from the canonical "lessons" directory.
    vocab = create_skill_vocab_from_lessons(lessons_dir)
    if not vocab:
        vocab = create_skill_vocab_from_runs(run_dirs)
        if not vocab:
            print("Could not create skill vocabulary from lessons/ or runs/. Aborting.")
            return
        print("Using fallback skill vocabulary derived from run verse_name values.")

    print(f"Created skill vocabulary with {len(vocab)} skills.")

    all_states = []
    all_goals = []
    all_skill_ids = []
    used_runs = 0

    for run_dir in run_dirs:
        events_path = os.path.join(run_dir, "events.jsonl")
        if not os.path.exists(events_path):
            continue

        # We need to associate events in this run with a skill.
        # We'll derive this from the verse name, assuming a convention
        # like `verse_name` -> `skill_prefix`. This is a placeholder for
        # a more robust run metadata system.
        first_event = None
        with open(events_path, 'r', encoding='utf-8') as f:
            try:
                first_event = json.loads(f.readline())
            except (json.JSONDecodeError, StopIteration):
                continue
        
        if not first_event:
            continue

        verse_name = first_event.get("verse_name", "unknown")
        verse_name_norm = str(verse_name).strip().lower()
        if verse_allow and verse_name_norm not in verse_allow:
            continue
        
        # Find the corresponding lesson/skill ID
        # This logic assumes a lesson is named like "verse_name_... .txt"
        skill_name = f"{verse_name}.txt" if f"{verse_name}.txt" in vocab else None
        if skill_name is None:
            skill_name = next((s for s in vocab.keys() if s.startswith(verse_name)), None)
        if skill_name is None:
            print(f"Warning: No lesson found for verse '{verse_name}' in run '{run_dir}'. Skipping.")
            continue
        skill_idx = vocab[skill_name]

        allowed_eps = None
        if bool(episode_success_only) or (episode_return_threshold is not None) or (episode_top_return_pct is not None):
            metrics = _collect_episode_quality(events_path)
            picked = set()
            min_ret = float(_safe_float(episode_return_threshold, -1e18)) if episode_return_threshold is not None else None
            candidates: List[Tuple[str, float]] = []
            for ep, q in metrics.items():
                if bool(episode_success_only) and not bool(q.get("success", False)):
                    continue
                ret = float(_safe_float(q.get("return_sum", 0.0), 0.0))
                if min_ret is not None and float(ret) < float(min_ret):
                    continue
                candidates.append((str(ep), float(ret)))
            if episode_top_return_pct is not None:
                pct = max(0.0, min(1.0, _safe_float(episode_top_return_pct, 1.0)))
                if pct <= 0.0:
                    candidates = []
                elif pct < 1.0 and candidates:
                    candidates.sort(key=lambda t: float(t[1]), reverse=True)
                    keep_n = max(1, int(round(float(len(candidates)) * float(pct))))
                    candidates = candidates[:keep_n]
            for ep, _ret in candidates:
                picked.add(str(ep))
            if not picked:
                continue
            allowed_eps = picked

        used_runs += 1
        # Load events and filter for high-reward episodes only
        with open(events_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if allowed_eps is not None:
                    ep = str(event.get("episode_id", "")).strip()
                    if not ep or ep not in allowed_eps:
                        continue
                if (not bool(include_all_events_from_selected_episodes)) and float(event.get('reward', 0.0)) <= reward_threshold:
                    continue
                obs = event.get('obs', {})
                if isinstance(obs, dict):
                    all_states.append(_flatten_obs(obs, STATE_KEYS))
                    all_goals.append(_flatten_obs(obs, GOAL_KEYS))
                    all_skill_ids.append(skill_idx)

    if not all_states:
        print("No expert samples found with reward > threshold. No data saved.")
        return

    before_counts: Dict[int, int] = {}
    for lab in all_skill_ids:
        before_counts[int(lab)] = int(before_counts.get(int(lab), 0)) + 1

    if bool(class_balance) and all_skill_ids:
        by_class: Dict[int, List[int]] = {}
        for idx, lab in enumerate(all_skill_ids):
            by_class.setdefault(int(lab), []).append(int(idx))
        target = max(len(v) for v in by_class.values())
        if int(class_balance_max_per_class) > 0:
            target = min(int(target), int(class_balance_max_per_class))
        rng = random.Random(int(class_balance_seed))

        picked: List[int] = []
        for _lab, idxs in sorted(by_class.items(), key=lambda kv: kv[0]):
            if len(idxs) >= int(target):
                picked.extend(rng.sample(idxs, int(target)))
            else:
                picked.extend(list(idxs))
                short = int(target) - len(idxs)
                picked.extend(rng.choices(idxs, k=int(short)))
        rng.shuffle(picked)
        all_states = [all_states[i] for i in picked]
        all_goals = [all_goals[i] for i in picked]
        all_skill_ids = [all_skill_ids[i] for i in picked]

    after_counts: Dict[int, int] = {}
    for lab in all_skill_ids:
        after_counts[int(lab)] = int(after_counts.get(int(lab), 0)) + 1

    # Pad and convert to Tensors
    states_tensor, state_dim = _pad_sequences(all_states)
    goals_tensor, goal_dim = _pad_sequences(all_goals)
    labels_tensor = torch.tensor(all_skill_ids, dtype=torch.long)

    # Save the prepared data
    data = {
        'states': states_tensor,
        'goals': goals_tensor,
        'labels': labels_tensor,
        'vocab': vocab,
        'state_dim': state_dim,
        'goal_dim': goal_dim,
        'class_balance_enabled': bool(class_balance),
        'class_counts_before': dict(before_counts),
        'class_counts_after': dict(after_counts),
    }
    
    torch.save(data, output_path)
    print(f"\\nDataset saved to {output_path}.")
    print(f"  Total samples: {len(all_states)}")
    print(f"  Used runs: {used_runs}")
    print(f"  Padded State Dimension: {state_dim}")
    print(f"  Padded Goal Dimension: {goal_dim}")
    print(f"  Vocabulary Size: {len(vocab)}")
    if bool(class_balance):
        print(f"  Class balance: enabled (seed={int(class_balance_seed)})")
        if int(class_balance_max_per_class) > 0:
            print(f"  Class max per class cap: {int(class_balance_max_per_class)}")
        print(f"  Class counts before: {before_counts}")
        print(f"  Class counts after : {after_counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="runs", help="Directory containing all run logs.")
    parser.add_argument("--lessons_dir", type=str, default="lessons", help="Directory containing .txt lesson files.")
    parser.add_argument("--output_path", type=str, default="training_batch.pt", help="Path to save the output tensor file.")
    parser.add_argument("--reward_threshold", type=float, default=0.8, help="Only include events with reward > threshold.")
    parser.add_argument("--verse_filter", action="append", default=None, help="Optional verse filter (repeatable).")
    parser.add_argument("--episode_success_only", action=argparse.BooleanOptionalAction, default=False, help="Keep only events from success episodes.")
    parser.add_argument("--episode_return_threshold", type=float, default=None, help="Keep only events from episodes with return >= threshold.")
    parser.add_argument(
        "--episode_top_return_pct",
        type=float,
        default=None,
        help="Keep only top-return episode percentile among selected runs/verses (0.2 = top 20%%).",
    )
    parser.add_argument(
        "--include_all_events_from_selected_episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When episode filters are active, keep all events in selected episodes (ignore reward threshold).",
    )
    parser.add_argument("--max_runs", type=int, default=0, help="Use only latest N runs (0 = all).")
    parser.add_argument(
        "--class_balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Balance class counts in the compiled selector dataset via deterministic over/under-sampling.",
    )
    parser.add_argument(
        "--class_balance_max_per_class",
        type=int,
        default=0,
        help="Optional cap after balancing (0 = no cap).",
    )
    parser.add_argument(
        "--class_balance_seed",
        type=int,
        default=42,
        help="Seed used for class-balancing sampling.",
    )
    args = parser.parse_args()
    
    prepare_data(
        args.runs_root,
        args.lessons_dir,
        args.output_path,
        reward_threshold=args.reward_threshold,
        verse_filter=args.verse_filter,
        episode_success_only=bool(args.episode_success_only),
        episode_return_threshold=args.episode_return_threshold,
        episode_top_return_pct=args.episode_top_return_pct,
        include_all_events_from_selected_episodes=bool(args.include_all_events_from_selected_episodes),
        max_runs=max(0, int(args.max_runs)),
        class_balance=bool(args.class_balance),
        class_balance_max_per_class=max(0, int(args.class_balance_max_per_class)),
        class_balance_seed=int(args.class_balance_seed),
    )



