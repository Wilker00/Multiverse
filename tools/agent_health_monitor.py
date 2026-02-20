"""
tools/agent_health_monitor.py

Agent health scorecard and optional autonomous self-healing actions.

Pillars:
1) Intuition Match   (selector coherence proxy from run events and trace priors)
2) Search Regret     (MCTS KL divergence from trace logs)
3) Safety Friction   (SafeExecutor veto rates from run events)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from orchestrator.evaluator import evaluate_run
from memory.central_repository import CentralMemoryConfig, find_similar
from memory.embeddings import obs_to_vector


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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


@dataclass
class EventRunMetrics:
    run_id: str
    run_dir: str
    verse_name: str
    policy_id: str
    selector_match: Optional[float]
    selector_samples: int
    mcts_queries: int
    mcts_vetoes: int
    shield_vetoes: int
    total_steps: int

    @property
    def veto_rate(self) -> float:
        return float(self.mcts_vetoes / float(max(1, self.mcts_queries)))

    @property
    def shield_veto_rate(self) -> float:
        return float(self.shield_vetoes / float(max(1, self.total_steps)))


@dataclass
class TraceVerseMetrics:
    verse_name: str
    rows: int
    mean_kl: Optional[float]
    prior_top1_match: Optional[float]
    high_quality_rate: Optional[float]


@dataclass
class AgentHealthRow:
    agent_id: str
    run_id: str
    run_dir: str
    verse_name: str
    policy_id: str
    mean_return: float
    success_rate: Optional[float]
    intuition_match: Optional[float]
    memory_coherence: Optional[float]
    search_regret_kl: Optional[float]
    veto_rate: float
    shield_veto_rate: float
    market_reputation: Optional[float]
    intuition_score: float
    search_score: float
    safety_score: float
    trust_score: float
    total_score: float
    status: str
    issues: List[str]
    recommended_actions: List[str]
    automated_actions: List[str]
    retrieval_audit: Dict[str, Any] = field(default_factory=dict)


def _iter_run_dirs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    out: List[str] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        if os.path.isdir(run_dir) and os.path.isfile(os.path.join(run_dir, "events.jsonl")):
            out.append(run_dir)
    out.sort(key=lambda p: _safe_float(os.path.getmtime(os.path.join(p, "events.jsonl")), 0.0), reverse=True)
    return out


def _memory_inventory(
    *,
    central_memory_dir: str,
    max_scan_rows: int,
) -> Dict[str, Any]:
    mem_path = os.path.join(str(central_memory_dir), "memories.jsonl")
    if not os.path.isfile(mem_path):
        return {
            "memory_path": mem_path.replace("\\", "/"),
            "exists": False,
            "rows_scanned": 0,
            "unique_runs": 0,
            "unique_verses": 0,
            "coverage_rate": 0.0,
        }
    rows = 0
    runs: set[str] = set()
    verses: set[str] = set()
    with open(mem_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            rows += 1
            runs.add(str(row.get("run_id", "")).strip())
            verses.add(str(row.get("verse_name", "")).strip().lower())
            if int(max_scan_rows) > 0 and rows >= int(max_scan_rows):
                break
    return {
        "memory_path": mem_path.replace("\\", "/"),
        "exists": True,
        "rows_scanned": int(rows),
        "unique_runs": int(len([r for r in runs if r])),
        "unique_verses": int(len([v for v in verses if v])),
    }


def _collect_retrieval_audit(
    *,
    run_dir: str,
    central_memory_dir: str,
    probes_per_run: int,
    top_k: int,
    min_score: float,
    same_verse_only: bool,
) -> Dict[str, Any]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return {
            "probes": 0,
            "eligible_events": 0,
            "hits": 0,
            "hit_rate": 0.0,
            "memory_coherence": None,
            "mean_top1_score": None,
            "mean_topk_score": None,
            "mean_recency_weight": None,
        }

    run_id = os.path.basename(run_dir)
    target_probes = max(1, int(probes_per_run))
    rng = random.Random(0)
    picks: List[Dict[str, Any]] = []
    eligible_count = 0
    for row in _iter_jsonl(events_path):
        if not isinstance(row, dict):
            continue
        obs = row.get("obs")
        try:
            vec = obs_to_vector(obs)
        except Exception:
            continue
        if vec:
            eligible_count += 1
            if len(picks) < target_probes:
                picks.append(row)
            else:
                j = rng.randint(1, eligible_count)
                if j <= target_probes:
                    picks[j - 1] = row
    if eligible_count <= 0:
        return {
            "probes": 0,
            "eligible_events": 0,
            "hits": 0,
            "hit_rate": 0.0,
            "memory_coherence": None,
            "mean_top1_score": None,
            "mean_topk_score": None,
            "mean_recency_weight": None,
        }

    cfg = CentralMemoryConfig(root_dir=str(central_memory_dir))

    hits = 0
    top1_scores: List[float] = []
    topk_means: List[float] = []
    recency_means: List[float] = []
    coherence_hits = 0
    coherence_samples = 0
    for ev in picks:
        obs = ev.get("obs")
        verse = str(ev.get("verse_name", "")).strip().lower()
        action = _safe_int(ev.get("action", -1), -1)
        try:
            matches = find_similar(
                obs=obs,
                cfg=cfg,
                top_k=max(1, int(top_k)),
                verse_name=(verse if bool(same_verse_only) else None),
                min_score=float(min_score),
                exclude_run_ids={str(run_id)},
            )
        except Exception:
            matches = []
        if matches:
            hits += 1
            top1_scores.append(float(matches[0].score))
            topk_means.append(float(sum(float(m.score) for m in matches) / float(max(1, len(matches)))))
            recency_means.append(
                float(sum(float(getattr(m, "recency_weight", 1.0)) for m in matches) / float(max(1, len(matches))))
            )
            if action >= 0:
                action_votes: Dict[int, float] = {}
                for m in matches:
                    ma = _safe_int(getattr(m, "action", -1), -1)
                    if ma < 0:
                        continue
                    action_votes[ma] = float(action_votes.get(ma, 0.0)) + max(0.0, float(getattr(m, "score", 0.0)))
                if action_votes:
                    coherence_samples += 1
                    predicted = max(action_votes.items(), key=lambda kv: kv[1])[0]
                    if int(predicted) == int(action):
                        coherence_hits += 1

    return {
        "probes": int(len(picks)),
        "eligible_events": int(eligible_count),
        "hits": int(hits),
        "hit_rate": float(hits / float(max(1, len(picks)))),
        "memory_coherence": (
            None if coherence_samples <= 0 else float(coherence_hits / float(max(1, coherence_samples)))
        ),
        "mean_top1_score": (None if not top1_scores else float(sum(top1_scores) / float(len(top1_scores)))),
        "mean_topk_score": (None if not topk_means else float(sum(topk_means) / float(len(topk_means)))),
        "mean_recency_weight": (None if not recency_means else float(sum(recency_means) / float(len(recency_means)))),
    }


def _summarize_retrieval_audit(rows: List["AgentHealthRow"]) -> Dict[str, Any]:
    audits = [dict(r.retrieval_audit or {}) for r in rows if isinstance(r.retrieval_audit, dict)]
    if not audits:
        return {
            "enabled": False,
            "rows_with_audit": 0,
            "mean_hit_rate": None,
            "mean_memory_coherence": None,
            "mean_top1_score": None,
            "mean_topk_score": None,
        }
    hit_rates = [float(a.get("hit_rate", 0.0)) for a in audits]
    coherence = [
        float(a.get("memory_coherence", 0.0))
        for a in audits
        if isinstance(a.get("memory_coherence"), (int, float))
    ]
    top1 = [float(a.get("mean_top1_score", 0.0)) for a in audits if isinstance(a.get("mean_top1_score"), (int, float))]
    topk = [float(a.get("mean_topk_score", 0.0)) for a in audits if isinstance(a.get("mean_topk_score"), (int, float))]
    return {
        "enabled": True,
        "rows_with_audit": int(len(audits)),
        "mean_hit_rate": float(sum(hit_rates) / float(max(1, len(hit_rates)))),
        "mean_memory_coherence": (None if not coherence else float(sum(coherence) / float(len(coherence)))),
        "mean_top1_score": (None if not top1 else float(sum(top1) / float(len(top1)))),
        "mean_topk_score": (None if not topk else float(sum(topk) / float(len(topk)))),
    }


def _collect_event_run_metrics(run_dir: str) -> Optional[EventRunMetrics]:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(events_path):
        return None
    run_id = os.path.basename(run_dir)
    first: Dict[str, Any] = {}
    for row in _iter_jsonl(events_path):
        first = row
        break
    verse_name = str(first.get("verse_name", "")).strip().lower()
    policy_id = str(first.get("policy_id", "")).strip().lower()

    selector_samples = 0
    selector_matches = 0
    total_steps = 0

    # Per-episode cumulative counters -> aggregate via episode final maxima.
    ep_counters: Dict[str, Dict[str, int]] = {}
    for row in _iter_jsonl(events_path):
        total_steps += 1
        ep = str(row.get("episode_id", "")).strip() or f"ep_{total_steps}"
        info = row.get("info")
        info = info if isinstance(info, dict) else {}
        action = _safe_int(row.get("action", -1), -1)

        action_info = info.get("action_info")
        action_info = action_info if isinstance(action_info, dict) else {}
        selector_active = bool(action_info.get("selector_active", False))

        se = info.get("safe_executor")
        se = se if isinstance(se, dict) else {}
        mcts_stats = se.get("mcts_stats")
        mcts_stats = mcts_stats if isinstance(mcts_stats, dict) else {}
        last_query = mcts_stats.get("last_query")
        last_query = last_query if isinstance(last_query, dict) else {}
        best_action = _safe_int(last_query.get("best_action", -1), -1)

        if selector_active and best_action >= 0 and action >= 0:
            selector_samples += 1
            if int(action) == int(best_action):
                selector_matches += 1

        q = max(0, _safe_int(mcts_stats.get("queries", 0), 0))
        v = max(0, _safe_int(mcts_stats.get("vetoes", 0), 0))
        counters = se.get("counters")
        counters = counters if isinstance(counters, dict) else {}
        shield = max(0, _safe_int(counters.get("shield_vetoes", 0), 0))
        prev = ep_counters.get(ep, {"q": 0, "v": 0, "shield": 0})
        ep_counters[ep] = {
            "q": max(int(prev["q"]), int(q)),
            "v": max(int(prev["v"]), int(v)),
            "shield": max(int(prev["shield"]), int(shield)),
        }

    mcts_queries = sum(int(v.get("q", 0)) for v in ep_counters.values())
    mcts_vetoes = sum(int(v.get("v", 0)) for v in ep_counters.values())
    shield_vetoes = sum(int(v.get("shield", 0)) for v in ep_counters.values())
    selector_match = None
    if selector_samples > 0:
        selector_match = float(selector_matches / float(selector_samples))
    return EventRunMetrics(
        run_id=run_id,
        run_dir=run_dir.replace("\\", "/"),
        verse_name=verse_name,
        policy_id=policy_id,
        selector_match=selector_match,
        selector_samples=int(selector_samples),
        mcts_queries=int(mcts_queries),
        mcts_vetoes=int(mcts_vetoes),
        shield_vetoes=int(shield_vetoes),
        total_steps=int(total_steps),
    )


def _discover_trace_paths(trace_glob_root: str) -> List[str]:
    out: List[str] = []
    root = str(trace_glob_root).strip()
    if os.path.isfile(root):
        out.append(root)
        return out
    if os.path.isdir(root):
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isfile(p) and name.endswith(".jsonl") and "mcts_trace" in name:
                out.append(p)
    out.sort(key=lambda p: _safe_float(os.path.getmtime(p), 0.0), reverse=True)
    return out


def _collect_trace_metrics(
    trace_paths: List[str],
    *,
    max_rows_per_file: int,
) -> Dict[str, TraceVerseMetrics]:
    by_verse: Dict[str, Dict[str, Any]] = {}
    for path in trace_paths:
        if not os.path.isfile(path):
            continue
        rows_taken = 0
        for row in _iter_jsonl(path):
            verse = str(row.get("verse_name", "")).strip().lower()
            if not verse:
                continue
            b = by_verse.setdefault(
                verse,
                {"rows": 0, "kl_sum": 0.0, "kl_n": 0, "prior_match_sum": 0.0, "prior_match_n": 0, "hq_sum": 0.0, "hq_n": 0},
            )
            b["rows"] = int(b["rows"]) + 1

            quality = row.get("trace_quality")
            quality = quality if isinstance(quality, dict) else {}
            if "kl_divergence" in quality:
                b["kl_sum"] = float(b["kl_sum"]) + _safe_float(quality.get("kl_divergence", 0.0), 0.0)
                b["kl_n"] = int(b["kl_n"]) + 1

            init_pi = row.get("initial_policy_guess")
            action = _safe_int(row.get("action", -1), -1)
            if isinstance(init_pi, list) and init_pi and action >= 0:
                argmax_idx = max(range(len(init_pi)), key=lambda i: _safe_float(init_pi[i], 0.0))
                b["prior_match_sum"] = float(b["prior_match_sum"]) + (1.0 if int(action) == int(argmax_idx) else 0.0)
                b["prior_match_n"] = int(b["prior_match_n"]) + 1

            if "high_quality_trace" in row:
                b["hq_sum"] = float(b["hq_sum"]) + (1.0 if bool(row.get("high_quality_trace", False)) else 0.0)
                b["hq_n"] = int(b["hq_n"]) + 1

            rows_taken += 1
            if int(max_rows_per_file) > 0 and rows_taken >= int(max_rows_per_file):
                break

    out: Dict[str, TraceVerseMetrics] = {}
    for verse, b in by_verse.items():
        mean_kl = None if int(b["kl_n"]) <= 0 else float(b["kl_sum"] / float(b["kl_n"]))
        prior_match = None if int(b["prior_match_n"]) <= 0 else float(b["prior_match_sum"] / float(b["prior_match_n"]))
        hq_rate = None if int(b["hq_n"]) <= 0 else float(b["hq_sum"] / float(b["hq_n"]))
        out[verse] = TraceVerseMetrics(
            verse_name=verse,
            rows=int(b["rows"]),
            mean_kl=mean_kl,
            prior_top1_match=prior_match,
            high_quality_rate=hq_rate,
        )
    return out


def _load_market_reputation(market_ledger_path: str) -> Dict[str, float]:
    if not os.path.isfile(market_ledger_path):
        return {}
    try:
        obj = _read_json(market_ledger_path)
    except Exception:
        return {}
    providers = obj.get("providers")
    providers = providers if isinstance(providers, dict) else {}
    out: Dict[str, float] = {}
    for run_id, v in providers.items():
        if not isinstance(v, dict):
            continue
        out[str(run_id)] = _safe_float(v.get("reputation", 1.0), 1.0)
    return out


def _score_row(
    *,
    run_metrics: EventRunMetrics,
    trace_metrics: Optional[TraceVerseMetrics],
    mean_return: float,
    success_rate: Optional[float],
    market_reputation: Optional[float],
    kl_critical: float,
    unsafe_veto_rate: float,
    incoherent_match_threshold: float,
    memory_coherence_threshold: float = 0.55,
    stale_kl_threshold: float = 0.12,
    retrieval_audit: Optional[Dict[str, Any]] = None,
) -> AgentHealthRow:
    intuition_match = run_metrics.selector_match
    if intuition_match is None and trace_metrics is not None:
        intuition_match = trace_metrics.prior_top1_match
    retrieval = dict(retrieval_audit or {})
    memory_coherence = (
        float(retrieval.get("memory_coherence"))
        if isinstance(retrieval.get("memory_coherence"), (int, float))
        else None
    )
    if intuition_match is None and memory_coherence is not None:
        intuition_match = float(memory_coherence)
    search_regret_kl = (None if trace_metrics is None else trace_metrics.mean_kl)
    veto_rate = float(run_metrics.veto_rate)
    shield_veto_rate = float(run_metrics.shield_veto_rate)

    if intuition_match is None:
        intuition_score = 50.0
    elif memory_coherence is None:
        intuition_score = 100.0 * _clamp01(float(intuition_match))
    else:
        blended = 0.6 * _clamp01(float(intuition_match)) + 0.4 * _clamp01(float(memory_coherence))
        intuition_score = 100.0 * float(blended)
    if search_regret_kl is None:
        search_score = 50.0
    else:
        search_score = 100.0 * (1.0 - _clamp01(float(search_regret_kl) / float(max(1e-9, kl_critical))))
    safety_base = 100.0 * (1.0 - _clamp01(float(veto_rate) / float(max(1e-9, unsafe_veto_rate))))
    if success_rate is None:
        safety_score = safety_base
    else:
        safety_score = 0.75 * safety_base + 0.25 * (100.0 * _clamp01(float(success_rate)))
    trust_score = 50.0 if market_reputation is None else 100.0 * _clamp01(float(market_reputation))

    total_score = 0.30 * intuition_score + 0.30 * search_score + 0.30 * safety_score + 0.10 * trust_score

    issues: List[str] = []
    if intuition_match is not None and float(intuition_match) < float(incoherent_match_threshold):
        issues.append("incoherent")
    if memory_coherence is not None and float(memory_coherence) < float(memory_coherence_threshold):
        if "incoherent" not in issues:
            issues.append("incoherent")
    if search_regret_kl is not None and float(search_regret_kl) > float(stale_kl_threshold):
        issues.append("stale")
    if float(veto_rate) > float(unsafe_veto_rate):
        issues.append("unsafe")
    if float(shield_veto_rate) > 0.20:
        issues.append("unsafe")

    if total_score >= 80.0 and not issues:
        status = "healthy"
    elif total_score >= 60.0 and len(issues) <= 1:
        status = "watch"
    elif total_score >= 40.0:
        status = "degraded"
    else:
        status = "critical"

    rec_actions: List[str] = []
    if "stale" in issues:
        rec_actions.append("trigger_hq_trace_retraining")
    if "unsafe" in issues:
        rec_actions.append("swap_manifest_to_fallback")
    if "incoherent" in issues:
        rec_actions.append("retrain_micro_selector")

    return AgentHealthRow(
        agent_id=f"{run_metrics.policy_id}:{run_metrics.verse_name}:{run_metrics.run_id}",
        run_id=run_metrics.run_id,
        run_dir=run_metrics.run_dir,
        verse_name=run_metrics.verse_name,
        policy_id=run_metrics.policy_id,
        mean_return=float(mean_return),
        success_rate=success_rate,
        intuition_match=intuition_match,
        memory_coherence=memory_coherence,
        search_regret_kl=search_regret_kl,
        veto_rate=float(veto_rate),
        shield_veto_rate=float(shield_veto_rate),
        market_reputation=market_reputation,
        intuition_score=float(intuition_score),
        search_score=float(search_score),
        safety_score=float(safety_score),
        trust_score=float(trust_score),
        total_score=float(total_score),
        status=str(status),
        issues=issues,
        recommended_actions=rec_actions,
        automated_actions=[],
        retrieval_audit=retrieval,
    )


def _candidate_runs_from_manifest(manifest: Dict[str, Any], runs_root: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    dep = _as_dict(manifest.get("deployment_ready_defaults"))
    for verse, entry in dep.items():
        e = _as_dict(entry)
        picked = _as_dict(e.get("picked_run"))
        run_dir = str(picked.get("run_dir", "")).strip()
        run_id = str(picked.get("run_id", "")).strip()
        if not run_dir and run_id:
            run_dir = os.path.join(runs_root, run_id)
        if run_dir and os.path.isfile(os.path.join(run_dir.replace("/", os.sep), "events.jsonl")):
            out.append((str(verse).strip().lower(), run_dir.replace("/", os.sep)))
    return out


def _latest_runs(runs_root: str, limit: int) -> List[str]:
    return _iter_run_dirs(runs_root)[: max(0, int(limit))]


def _print_report(rows: List[AgentHealthRow]) -> None:
    print("AGENT HEALTH REPORT")
    print("----------------------------------------")
    for r in rows:
        score_txt = f"{int(round(r.total_score))}/100"
        print(f"[ID: {r.agent_id}]  SCORE: {score_txt} ({r.status.upper()})")
        if r.intuition_match is None:
            print("  - Intuition Match: n/a")
        else:
            print(f"  - Intuition Match: {100.0 * float(r.intuition_match):.1f}%")
        if r.memory_coherence is None:
            print("  - Memory Coherence: n/a")
        else:
            print(f"  - Memory Coherence: {100.0 * float(r.memory_coherence):.1f}%")
        if r.search_regret_kl is None:
            print("  - Search Regret KL: n/a")
        else:
            print(f"  - Search Regret KL: {float(r.search_regret_kl):.4f}")
        print(
            f"  - Veto Rate: {100.0 * float(r.veto_rate):.2f}% "
            f"(shield {100.0 * float(r.shield_veto_rate):.2f}%/step)"
        )
        ra = dict(r.retrieval_audit or {})
        if ra:
            print(
                f"  - Retrieval Hit Rate: {100.0 * float(_safe_float(ra.get('hit_rate', 0.0), 0.0)):.1f}% "
                f"(probes {int(_safe_int(ra.get('probes', 0), 0))})"
            )
        if r.recommended_actions:
            print(f"  - Action: {', '.join(r.recommended_actions)}")
        if r.automated_actions:
            print(f"  - Executed: {', '.join(r.automated_actions)}")
        print("")


def _run_cmd(cmd: List[str], *, cwd: str) -> None:
    proc = subprocess.run(cmd, cwd=cwd)
    if int(proc.returncode) != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _apply_manifest_fallback(
    *,
    manifest_path: str,
    verse_name: str,
) -> str:
    if not os.path.isfile(manifest_path):
        return "manifest_missing"
    manifest = _read_json(manifest_path)
    dep = _as_dict(manifest.get("deployment_ready_defaults"))
    history = _as_dict(manifest.get("deployment_history"))
    robust = _as_dict(manifest.get("winners_robust"))
    cur = _as_dict(dep.get(verse_name))
    fallback = None

    hist_rows = history.get(verse_name)
    if isinstance(hist_rows, list) and hist_rows:
        for row in reversed(hist_rows):
            if isinstance(row, dict):
                fallback = row
                break
    if fallback is None:
        rob = robust.get(verse_name)
        if isinstance(rob, dict):
            fallback = {"picked_run": rob}
    if fallback is None:
        return "no_fallback"

    dep[verse_name] = dict(fallback)
    dep[verse_name]["health_override"] = {
        "at_iso": _utc_now_iso(),
        "reason": "unsafe",
        "source": "agent_health_monitor",
    }
    manifest["deployment_ready_defaults"] = dep
    backup = manifest_path + ".health.bak"
    shutil.copyfile(manifest_path, backup)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return "manifest_fallback_applied"


def _auto_heal(
    *,
    row: AgentHealthRow,
    args: argparse.Namespace,
    cwd: str,
) -> List[str]:
    executed: List[str] = []
    py = sys.executable
    if "stale" in row.issues:
        cmd = [
            py,
            os.path.join("tools", "mcts_cycle.py"),
            "--verse",
            str(row.verse_name),
            "--cycles",
            str(max(1, int(args.heal_mcts_cycles))),
            "--episodes_per_cycle",
            str(max(1, int(args.heal_mcts_episodes))),
            "--max_steps",
            str(max(1, int(args.heal_mcts_max_steps))),
            "--checkpoint_path",
            str(args.heal_meta_checkpoint),
            "--trace_out",
            str(args.heal_trace_out),
            "--high_quality_trace_filter",
        ]
        _run_cmd(cmd, cwd=cwd)
        executed.append("trigger_hq_trace_retraining")

    if "unsafe" in row.issues:
        res = _apply_manifest_fallback(
            manifest_path=str(args.manifest_path),
            verse_name=str(row.verse_name),
        )
        executed.append(res)

    if "incoherent" in row.issues:
        prep_cmd = [
            py,
            os.path.join("tools", "prepare_selector_data.py"),
            "--runs_root",
            str(args.runs_root),
            "--output_path",
            str(args.heal_selector_batch_path),
            "--reward_threshold",
            str(float(args.heal_selector_reward_threshold)),
        ]
        train_cmd = [
            py,
            os.path.join("tools", "train_selector.py"),
            "--data_path",
            str(args.heal_selector_batch_path),
            "--model_save_path",
            str(args.heal_selector_model_path),
            "--epochs",
            str(max(1, int(args.heal_selector_epochs))),
        ]
        _run_cmd(prep_cmd, cwd=cwd)
        _run_cmd(train_cmd, cwd=cwd)
        executed.append("retrain_micro_selector")
    return executed


def _row_to_json(row: AgentHealthRow) -> Dict[str, Any]:
    return {
        "agent_id": row.agent_id,
        "run_id": row.run_id,
        "run_dir": row.run_dir,
        "verse_name": row.verse_name,
        "policy_id": row.policy_id,
        "mean_return": float(row.mean_return),
        "success_rate": row.success_rate,
        "intuition_match": row.intuition_match,
        "memory_coherence": row.memory_coherence,
        "search_regret_kl": row.search_regret_kl,
        "veto_rate": float(row.veto_rate),
        "shield_veto_rate": float(row.shield_veto_rate),
        "market_reputation": row.market_reputation,
        "intuition_score": float(row.intuition_score),
        "search_score": float(row.search_score),
        "safety_score": float(row.safety_score),
        "trust_score": float(row.trust_score),
        "total_score": float(row.total_score),
        "status": row.status,
        "issues": list(row.issues),
        "recommended_actions": list(row.recommended_actions),
        "automated_actions": list(row.automated_actions),
        "retrieval_audit": dict(row.retrieval_audit or {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument(
        "--trace_root",
        type=str,
        default=os.path.join("models", "expert_datasets"),
        help="File or directory that contains mcts_trace*.jsonl files.",
    )
    ap.add_argument("--central_memory_dir", type=str, default="central_memory")
    ap.add_argument("--latest_runs", type=int, default=12)
    ap.add_argument("--max_trace_rows_per_file", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min_score", type=float, default=0.0)

    ap.add_argument("--kl_critical", type=float, default=0.25)
    ap.add_argument("--stale_kl_threshold", type=float, default=0.12)
    ap.add_argument("--unsafe_veto_rate", type=float, default=0.10)
    ap.add_argument("--incoherent_match_threshold", type=float, default=0.55)
    ap.add_argument("--memory_coherence_threshold", type=float, default=0.55)
    ap.add_argument("--retrieval_audit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--retrieval_audit_probes_per_run", type=int, default=8)
    ap.add_argument("--retrieval_audit_top_k", type=int, default=5)
    ap.add_argument("--retrieval_audit_min_score", type=float, default=0.0)
    ap.add_argument("--retrieval_audit_same_verse_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--retrieval_audit_max_memory_rows_scan", type=int, default=200000)

    ap.add_argument("--auto_heal", action="store_true")
    ap.add_argument("--auto_heal_max_agents", type=int, default=3)
    ap.add_argument("--heal_mcts_cycles", type=int, default=1)
    ap.add_argument("--heal_mcts_episodes", type=int, default=20)
    ap.add_argument("--heal_mcts_max_steps", type=int, default=80)
    ap.add_argument("--heal_meta_checkpoint", type=str, default=os.path.join("models", "meta_transformer.pt"))
    ap.add_argument("--heal_trace_out", type=str, default=os.path.join("models", "expert_datasets", "mcts_trace_dataset.jsonl"))
    ap.add_argument("--heal_selector_batch_path", type=str, default="training_batch.pt")
    ap.add_argument("--heal_selector_model_path", type=str, default=os.path.join("models", "micro_selector.pt"))
    ap.add_argument("--heal_selector_epochs", type=int, default=8)
    ap.add_argument("--heal_selector_reward_threshold", type=float, default=0.8)

    ap.add_argument("--format", type=str, default="table", choices=["table", "json"])
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "tuning", "agent_health_report.json"),
    )
    args = ap.parse_args()

    manifest = _read_json(args.manifest_path) if os.path.isfile(args.manifest_path) else {}
    trace_paths = _discover_trace_paths(str(args.trace_root))
    trace_by_verse = _collect_trace_metrics(trace_paths, max_rows_per_file=max(0, int(args.max_trace_rows_per_file)))
    market_reputation = _load_market_reputation(
        os.path.join(str(args.central_memory_dir), "knowledge_market_ledger.json")
    )

    candidate_run_dirs: List[str] = []
    candidate_run_dirs.extend([rd for _, rd in _candidate_runs_from_manifest(manifest, str(args.runs_root))])
    for rd in _latest_runs(str(args.runs_root), int(args.latest_runs)):
        if rd not in candidate_run_dirs:
            candidate_run_dirs.append(rd)

    rows: List[AgentHealthRow] = []
    memory_inventory = _memory_inventory(
        central_memory_dir=str(args.central_memory_dir),
        max_scan_rows=max(0, int(args.retrieval_audit_max_memory_rows_scan)),
    )
    for run_dir in candidate_run_dirs:
        rm = _collect_event_run_metrics(run_dir)
        if rm is None:
            continue
        try:
            st = evaluate_run(run_dir)
            mean_return = float(st.mean_return)
            success_rate = st.success_rate
        except Exception:
            mean_return = 0.0
            success_rate = None
        trace = trace_by_verse.get(str(rm.verse_name))
        rep = market_reputation.get(str(rm.run_id))
        retrieval_audit: Dict[str, Any] = {}
        if bool(args.retrieval_audit):
            retrieval_audit = _collect_retrieval_audit(
                run_dir=run_dir,
                central_memory_dir=str(args.central_memory_dir),
                probes_per_run=max(1, int(args.retrieval_audit_probes_per_run)),
                top_k=max(1, int(args.retrieval_audit_top_k)),
                min_score=float(args.retrieval_audit_min_score),
                same_verse_only=bool(args.retrieval_audit_same_verse_only),
            )
        row = _score_row(
            run_metrics=rm,
            trace_metrics=trace,
            mean_return=mean_return,
            success_rate=success_rate,
            market_reputation=rep,
            kl_critical=float(args.kl_critical),
            unsafe_veto_rate=float(args.unsafe_veto_rate),
            incoherent_match_threshold=float(args.incoherent_match_threshold),
            memory_coherence_threshold=float(args.memory_coherence_threshold),
            stale_kl_threshold=float(args.stale_kl_threshold),
            retrieval_audit=retrieval_audit,
        )
        if float(row.total_score) >= float(args.min_score):
            rows.append(row)

    rows.sort(key=lambda r: float(r.total_score))
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    if bool(args.auto_heal):
        acted = 0
        for row in rows:
            if acted >= max(0, int(args.auto_heal_max_agents)):
                break
            if not row.issues:
                continue
            row.automated_actions = _auto_heal(row=row, args=args, cwd=os.getcwd())
            acted += 1

    report = {
        "created_at_iso": _utc_now_iso(),
        "manifest_path": str(args.manifest_path),
        "runs_root": str(args.runs_root),
        "trace_paths": [str(p) for p in trace_paths],
        "memory_inventory": memory_inventory,
        "retrieval_audit_summary": _summarize_retrieval_audit(rows),
        "count": int(len(rows)),
        "rows": [_row_to_json(r) for r in rows],
    }

    out_path = str(args.out_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if str(args.format).strip().lower() == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(rows)
        print(f"report: {out_path}")


if __name__ == "__main__":
    main()
