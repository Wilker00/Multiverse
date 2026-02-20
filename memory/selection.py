"""
memory/selection.py

Selection + mutation heuristics to reduce memory bloat and improve signal quality.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from memory.embeddings import cosine_similarity


@dataclass
class SelectionConfig:
    min_reward: float = -1e9
    max_reward: float = 1e9
    keep_successes: bool = True
    success_key: str = "reached_goal"
    keep_top_k_per_episode: int = 0
    novelty_bonus: float = 0.0  # bonus for first-time observations
    keep_top_k_episodes: int = 0  # episode-level selection by score
    creative_failure_bonus: float = 0.0  # bonus for strong failures
    creative_failure_min_return: float = 0.0  # threshold for "useful failure"
    creative_failure_max_steps: int = 0  # if >0, require at least this many steps
    recency_half_life_ms: int = 0  # if >0, apply exponential decay by age


@dataclass
class ActiveForgettingConfig:
    """
    Run-level deduplication for central memory.
    """
    memory_dir: str = "central_memory"
    memories_filename: str = "memories.jsonl"
    dedupe_index_filename: str = "dedupe_index.json"
    similarity_threshold: float = 0.95
    min_events_per_run: int = 20
    write_backup: bool = True
    # Optional quality gating to prune uniquely harmful runs (not only duplicates).
    quality_filter_enabled: bool = False
    min_mean_reward: float = -1e9
    min_success_rate: float = 0.0
    max_hazard_rate: float = 1.0


@dataclass
class ActiveForgettingStats:
    input_rows: int
    kept_rows: int
    dropped_rows: int
    input_runs: int
    kept_runs: int
    dropped_runs: int
    dropped_run_ids: List[str]
    dropped_low_quality_runs: int = 0
    dropped_low_quality_rows: int = 0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _obs_key(obs: Any) -> str:
    try:
        return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(obs)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _event_success(event: Dict[str, Any], success_key: str) -> bool:
    info = event.get("info") or {}
    if isinstance(info, dict):
        return bool(info.get(success_key) is True)
    return False


def score_event(
    event: Dict[str, Any],
    *,
    success_key: str,
    novelty_bonus: float,
    seen_obs: Dict[str, int],
    now_ms: Optional[int],
    recency_half_life_ms: int,
) -> float:
    """
    Score events for selection. Higher is better.
    Reward is primary; novelty can add a small bonus.
    """
    reward = _safe_float(event.get("reward", 0.0))
    obs = event.get("obs")
    key = _obs_key(obs)
    count = seen_obs.get(key, 0)
    seen_obs[key] = count + 1

    novelty = 0.0
    if novelty_bonus > 0.0 and count == 0:
        novelty = novelty_bonus

    success = 1.0 if _event_success(event, success_key) else 0.0
    score = reward + novelty + success

    if recency_half_life_ms > 0 and now_ms is not None:
        t_ms = event.get("t_ms")
        if isinstance(t_ms, (int, float)):
            age = max(0.0, float(now_ms) - float(t_ms))
            decay = 0.5 ** (age / float(recency_half_life_ms))
            score *= decay

    return score


def score_episode(
    ep_events: List[Dict[str, Any]],
    *,
    success_key: str,
    creative_failure_bonus: float,
    creative_failure_min_return: float,
    creative_failure_max_steps: int,
) -> float:
    if not ep_events:
        return 0.0

    returns = sum(_safe_float(e.get("reward", 0.0)) for e in ep_events)
    steps = len(ep_events)
    success = any(_event_success(e, success_key) for e in ep_events)

    score = returns
    if success:
        score += 1.0
    else:
        if returns >= creative_failure_min_return:
            if creative_failure_max_steps <= 0 or steps >= creative_failure_max_steps:
                score += creative_failure_bonus
    return score


def select_events(
    events: List[Dict[str, Any]],
    cfg: SelectionConfig,
) -> List[Dict[str, Any]]:
    """
    Filter and score events to reduce bloat and noise.
    """
    # Filter by reward thresholds and success preference.
    filtered: List[Dict[str, Any]] = []
    for e in events:
        is_success = _event_success(e, cfg.success_key)
        # keep_successes=True means "always keep successes", even when they
        # would be trimmed by reward thresholds.
        if cfg.keep_successes and is_success:
            filtered.append(e)
            continue
        reward = _safe_float(e.get("reward", 0.0))
        if reward < cfg.min_reward or reward > cfg.max_reward:
            continue
        filtered.append(e)

    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for e in filtered:
        ep = str(e.get("episode_id"))
        by_ep.setdefault(ep, []).append(e)

    now_ms: Optional[int] = None
    if cfg.recency_half_life_ms > 0:
        for e in filtered:
            t_ms = e.get("t_ms")
            if isinstance(t_ms, (int, float)):
                if now_ms is None or t_ms > now_ms:
                    now_ms = int(t_ms)

    # Episode-level selection.
    if cfg.keep_top_k_episodes > 0:
        ep_scores = []
        for ep_id, ep_events in by_ep.items():
            ep_score = score_episode(
                ep_events,
                success_key=cfg.success_key,
                creative_failure_bonus=cfg.creative_failure_bonus,
                creative_failure_min_return=cfg.creative_failure_min_return,
                creative_failure_max_steps=cfg.creative_failure_max_steps,
            )
            ep_scores.append((ep_score, ep_id))
        ep_scores.sort(key=lambda t: t[0], reverse=True)
        keep_eps = set(ep_id for _, ep_id in ep_scores[: cfg.keep_top_k_episodes])
        by_ep = {ep_id: ep_events for ep_id, ep_events in by_ep.items() if ep_id in keep_eps}

    if cfg.keep_top_k_per_episode <= 0:
        # Flatten
        out: List[Dict[str, Any]] = []
        for ep_events in by_ep.values():
            out.extend(ep_events)
        return out

    # Score and keep top-K per episode.
    out = []
    for ep_id, ep_events in by_ep.items():
        seen_obs: Dict[str, int] = {}
        scored = []
        for e in ep_events:
            scored.append(
                (
                    score_event(
                        e,
                        success_key=cfg.success_key,
                        novelty_bonus=cfg.novelty_bonus,
                        seen_obs=seen_obs,
                        now_ms=now_ms,
                        recency_half_life_ms=cfg.recency_half_life_ms,
                    ),
                    e,
                )
            )
        scored.sort(key=lambda t: t[0], reverse=True)
        out.extend([e for _, e in scored[: cfg.keep_top_k_per_episode]])

    return out


def prune_events_jsonl(
    *,
    run_dir: str,
    input_filename: str = "events.jsonl",
    output_filename: str = "events.pruned.jsonl",
    cfg: Optional[SelectionConfig] = None,
) -> str:
    """
    Read events.jsonl, apply selection, and write a pruned file.
    """
    if cfg is None:
        cfg = SelectionConfig()

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    in_path = os.path.join(run_dir, input_filename)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"events file not found: {in_path}")

    events: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    kept = select_events(events, cfg)

    out_path = os.path.join(run_dir, output_filename)
    with open(out_path, "w", encoding="utf-8") as out:
        for e in kept:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    return out_path


def active_forget_central_memory(cfg: Optional[ActiveForgettingConfig] = None) -> ActiveForgettingStats:
    """
    Deduplicate central memory at run granularity.

    Runs with representative vectors that are too similar to an existing
    higher-quality run are removed. Quality is based on higher mean reward.
    """
    if cfg is None:
        cfg = ActiveForgettingConfig()

    mem_path = os.path.join(cfg.memory_dir, cfg.memories_filename)
    dedupe_path = os.path.join(cfg.memory_dir, cfg.dedupe_index_filename)

    if not os.path.isfile(mem_path):
        raise FileNotFoundError(f"Central memory file not found: {mem_path}")

    if cfg.write_backup:
        bak = mem_path + ".bak"
        try:
            with open(mem_path, "rb") as src, open(bak, "wb") as dst:
                dst.write(src.read())
        except Exception:
            # Backup failure should not block pruning.
            pass

    rows: List[Dict[str, Any]] = []
    with open(mem_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            rows.append(row)

    # Summarize each run by mean observation vector + reward stats.
    # Optionally apply quality gating before deduplication.
    run_rows: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        run_id = str(r.get("run_id", "")).strip()
        if not run_id:
            continue
        run_rows.setdefault(run_id, []).append(r)

    summaries: List[Tuple[str, List[float], float, float, int]] = []
    drop_runs_quality: set[str] = set()
    hazard_keys = {
        "hit_wall",
        "hit_obstacle",
        "battery_death",
        "battery_depleted",
        "fell_cliff",
        "fell_pit",
        "hit_laser",
    }
    for run_id, group in run_rows.items():
        vec_sum: Optional[List[float]] = None
        vec_count = 0
        reward_sum = 0.0
        reward_max = -1e18
        success_events = 0
        hazard_events = 0

        for row in group:
            reward = _safe_float(row.get("reward", 0.0))
            reward_sum += reward
            reward_max = max(reward_max, reward)
            info = row.get("info")
            info = info if isinstance(info, dict) else {}
            if bool(info.get("reached_goal", False)):
                success_events += 1
            if any(bool(info.get(k, False)) for k in hazard_keys):
                hazard_events += 1

            v = row.get("obs_vector")
            if not isinstance(v, list):
                continue
            try:
                vf = [float(x) for x in v]
            except Exception:
                continue
            if not vf:
                continue
            if vec_sum is None:
                vec_sum = [0.0 for _ in vf]
            if len(vf) != len(vec_sum):
                continue
            for i, x in enumerate(vf):
                vec_sum[i] += x
            vec_count += 1

        run_len = max(1, len(group))
        mean_reward = reward_sum / float(run_len)
        success_rate = float(success_events) / float(run_len)
        hazard_rate = float(hazard_events) / float(run_len)
        if bool(cfg.quality_filter_enabled) and len(group) >= int(max(1, cfg.min_events_per_run)):
            if (
                mean_reward < float(cfg.min_mean_reward)
                or success_rate < float(cfg.min_success_rate)
                or hazard_rate > float(cfg.max_hazard_rate)
            ):
                drop_runs_quality.add(run_id)
                continue

        if len(group) < int(cfg.min_events_per_run):
            # Keep tiny runs by default, they may represent rare edge-cases.
            continue
        if vec_sum is None or vec_count <= 0:
            continue
        rep = [x / float(vec_count) for x in vec_sum]
        summaries.append((run_id, rep, mean_reward, reward_max, len(group)))

    # Greedy master DNA selection: highest quality first.
    summaries.sort(key=lambda t: (t[2], t[3], t[4]), reverse=True)

    masters: List[Tuple[str, List[float], float, float, int]] = []
    drop_runs: set[str] = set()
    for cand in summaries:
        cand_id, cand_vec, _, _, _ = cand
        duplicate = False
        for m in masters:
            _, m_vec, _, _, _ = m
            if len(cand_vec) != len(m_vec):
                continue
            sim = cosine_similarity(cand_vec, m_vec)
            if sim >= float(cfg.similarity_threshold):
                duplicate = True
                break
        if duplicate:
            drop_runs.add(cand_id)
        else:
            masters.append(cand)

    all_drop_runs = set(drop_runs)
    all_drop_runs.update(drop_runs_quality)

    kept_rows: List[Dict[str, Any]] = []
    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        if run_id and run_id in all_drop_runs:
            continue
        kept_rows.append(row)

    with open(mem_path, "w", encoding="utf-8") as out:
        for row in kept_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Rebuild dedupe index from kept rows.
    dedupe_keys = []
    for row in kept_rows:
        k = row.get("dedupe_key")
        if isinstance(k, str) and k:
            dedupe_keys.append(k)
    os.makedirs(cfg.memory_dir, exist_ok=True)
    with open(dedupe_path, "w", encoding="utf-8") as f:
        json.dump(sorted(set(dedupe_keys)), f, ensure_ascii=False, indent=2)

    input_runs = len(run_rows)
    kept_run_ids = set(str(r.get("run_id", "")).strip() for r in kept_rows if str(r.get("run_id", "")).strip())
    dropped_low_quality_rows = 0
    for run_id in drop_runs_quality:
        dropped_low_quality_rows += len(run_rows.get(run_id, []))

    return ActiveForgettingStats(
        input_rows=len(rows),
        kept_rows=len(kept_rows),
        dropped_rows=max(0, len(rows) - len(kept_rows)),
        input_runs=input_runs,
        kept_runs=len(kept_run_ids),
        dropped_runs=max(0, input_runs - len(kept_run_ids)),
        dropped_run_ids=sorted(all_drop_runs),
        dropped_low_quality_runs=len(drop_runs_quality),
        dropped_low_quality_rows=int(dropped_low_quality_rows),
    )
