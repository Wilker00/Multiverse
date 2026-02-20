"""
tools/run_cognitive_upgrade_kpi.py

Run the Week 1-4 cognitive memory upgrade KPI workflow:
1) LTM/STM memory retrieval KPI
2) memory_vault_world STM diagnostic baseline
3) rule_flip_world memory rewriting adaptation KPI
4) warehouse transfer challenge (fixed-seed benchmark)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from memory.central_repository import CentralMemoryConfig, find_similar, ingest_run
from orchestrator.evaluator import evaluate_run
from orchestrator.trainer import Trainer


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isfile(path):
        return
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


def _group_events_by_episode(events_path: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in _iter_jsonl(events_path):
        ep = str(row.get("episode_id", "")).strip()
        if not ep:
            continue
        out.setdefault(ep, []).append(row)
    for ep in out:
        out[ep].sort(key=lambda r: _safe_int(r.get("step_idx", 0), 0))
    return out


def _run_dirs_from_roots(roots: List[str], max_runs: int) -> List[str]:
    rows: List[Tuple[float, str]] = []
    for root in roots:
        rr = str(root).strip()
        if not rr or not os.path.isdir(rr):
            continue
        for name in os.listdir(rr):
            run_dir = os.path.join(rr, name)
            events = os.path.join(run_dir, "events.jsonl")
            if not (os.path.isdir(run_dir) and os.path.isfile(events)):
                continue
            mtime = _safe_float(os.path.getmtime(events), 0.0)
            rows.append((mtime, run_dir))
    rows.sort(key=lambda x: x[0], reverse=True)
    out: List[str] = []
    seen: set[str] = set()
    for _, rd in rows:
        norm = rd.replace("\\", "/")
        if norm in seen:
            continue
        seen.add(norm)
        out.append(rd)
        if len(out) >= max(1, int(max_runs)):
            break
    return out


def _memory_week1_kpi(
    *,
    run_dirs: List[str],
    memory_dir: str,
    max_ingest_runs: int,
    probe_limit: int,
    min_score: float,
) -> Dict[str, Any]:
    os.makedirs(memory_dir, exist_ok=True)
    cfg = CentralMemoryConfig(root_dir=str(memory_dir))
    ingested: List[Dict[str, Any]] = []
    for rd in run_dirs[: max(1, int(max_ingest_runs))]:
        try:
            st = ingest_run(run_dir=rd, cfg=cfg)
            ingested.append(
                {
                    "run_dir": rd.replace("\\", "/"),
                    "input_events": int(st.input_events),
                    "selected_events": int(st.selected_events),
                    "added_events": int(st.added_events),
                    "skipped_duplicates": int(st.skipped_duplicates),
                }
            )
        except Exception as e:
            ingested.append({"run_dir": rd.replace("\\", "/"), "error": str(e)})

    ltm_path = os.path.join(memory_dir, cfg.ltm_memories_filename)
    ltm_rows = list(_iter_jsonl(ltm_path))
    probes = ltm_rows[: max(1, int(probe_limit))]
    hits = 0
    probe_details: List[Dict[str, Any]] = []
    for row in probes:
        obs = row.get("obs")
        run_id = str(row.get("run_id", "")).strip()
        try:
            matches = find_similar(
                obs=obs,
                cfg=cfg,
                top_k=3,
                min_score=float(min_score),
                memory_tiers={"ltm"},
                exclude_run_ids=({run_id} if run_id else None),
            )
        except Exception:
            matches = []
        hit = bool(matches)
        if hit:
            hits += 1
        probe_details.append(
            {
                "run_id": run_id,
                "verse_name": str(row.get("verse_name", "")),
                "hit": bool(hit),
                "top_score": (None if not matches else float(matches[0].score)),
            }
        )
    hit_rate = (float(hits / float(max(1, len(probes)))) if probes else 0.0)
    return {
        "ingested_runs": ingested,
        "ltm_rows": int(len(ltm_rows)),
        "ltm_probe_count": int(len(probes)),
        "ltm_hits": int(hits),
        "ltm_hit_rate": float(hit_rate),
        "target_ltm_hit_rate": 0.90,
        "target_met": bool(hit_rate >= 0.90),
        "probe_details": probe_details,
    }


def _train_once(
    *,
    runs_root: str,
    verse_name: str,
    algo: str,
    episodes: int,
    max_steps: int,
    seed: int,
    verse_params: Optional[Dict[str, Any]] = None,
    agent_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trainer = Trainer(run_root=str(runs_root), schema_version="v1", auto_register_builtin=True)
    vs = VerseSpec(
        spec_version="v1",
        verse_name=str(verse_name),
        verse_version="0.1",
        seed=int(seed),
        tags=["cognitive_upgrade_kpi"],
        params=(dict(verse_params or {})),
    )
    acfg = dict(agent_config or {})
    acfg["verse_name"] = str(verse_name)
    ag = AgentSpec(
        spec_version="v1",
        policy_id=f"{algo}_{verse_name}_kpi",
        policy_version="0.1",
        algo=str(algo),
        seed=int(seed),
        tags=["cognitive_upgrade_kpi"],
        config=acfg,
    )
    rr = trainer.run(
        verse_spec=vs,
        agent_spec=ag,
        episodes=max(1, int(episodes)),
        max_steps=max(1, int(max_steps)),
        seed=int(seed),
    )
    run_id = str(rr.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("trainer.run did not return run_id")
    run_dir = os.path.join(runs_root, run_id)
    stats = evaluate_run(run_dir)
    return {
        "run_id": run_id,
        "run_dir": run_dir.replace("\\", "/"),
        "mean_return": float(stats.mean_return),
        "success_rate": (None if stats.success_rate is None else float(stats.success_rate)),
        "episodes": int(stats.episodes),
        "total_steps": int(stats.total_steps),
    }


def _memory_vault_baseline(
    *,
    runs_root: str,
    episodes: int,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    res = _train_once(
        runs_root=runs_root,
        verse_name="memory_vault_world",
        algo="evolving",
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
        verse_params={
            "width": 9,
            "height": 9,
            "max_steps": int(max_steps),
            "wall_density": 0.16,
            "hint_visible_steps": 1,
        },
        agent_config={
            "vector_memory_mode": "procedural",
            "evolve_interval": 10,
            "mutation_scale": 0.15,
        },
    )
    events = os.path.join(runs_root, str(res["run_id"]), "events.jsonl")
    by_ep = _group_events_by_episode(events)
    success_steps: List[int] = []
    for _, rows in by_ep.items():
        for row in rows:
            info = row.get("info")
            info = info if isinstance(info, dict) else {}
            if bool(info.get("reached_goal", False)):
                success_steps.append(_safe_int(row.get("step_idx", 0), 0) + 1)
                break
    mean_success_steps = (None if not success_steps else float(sum(success_steps) / float(len(success_steps))))
    stm_capacity = 0.0
    if isinstance(res.get("success_rate"), (int, float)):
        sr = float(res["success_rate"])
        speed = 0.0 if mean_success_steps is None else max(0.0, 1.0 - float(mean_success_steps / float(max(1, max_steps))))
        stm_capacity = float(0.7 * sr + 0.3 * speed)
    return {
        **res,
        "metric_name": "stm_capacity_baseline",
        "stm_capacity_score": float(stm_capacity),
        "mean_steps_to_goal_on_success": mean_success_steps,
    }


def _rule_flip_adaptation(
    *,
    runs_root: str,
    episodes: int,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    flip_step = max(2, int(max_steps) // 2)
    res = _train_once(
        runs_root=runs_root,
        verse_name="rule_flip_world",
        algo="aware",
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
        verse_params={
            "track_len": 11,
            "max_steps": int(max_steps),
            "flip_step": int(flip_step),
            "target_reward": 2.0,
            "wrong_target_penalty": -1.5,
        },
        agent_config={
            "vector_memory_mode": "declarative",
            "declarative_fact_weight": 0.4,
            "use_vector_memory": True,
            "vector_memory_top_k": 5,
        },
    )
    events = os.path.join(runs_root, str(res["run_id"]), "events.jsonl")
    by_ep = _group_events_by_episode(events)
    ep_rows: List[Tuple[str, float, float]] = []
    for ep_id, rows in by_ep.items():
        post_flip_steps = 0
        post_flip_hits = 0
        post_flip_return = 0.0
        seen_flip = False
        for row in rows:
            info = row.get("info")
            info = info if isinstance(info, dict) else {}
            if bool(info.get("rule_flipped", False)):
                seen_flip = True
            if seen_flip:
                post_flip_steps += 1
                if bool(info.get("target_hit", False)):
                    post_flip_hits += 1
                post_flip_return += _safe_float(row.get("reward", 0.0), 0.0)
        hit_rate = float(post_flip_hits / float(max(1, post_flip_steps))) if post_flip_steps > 0 else 0.0
        ep_rows.append((ep_id, hit_rate, float(post_flip_return)))
    ep_rows.sort(key=lambda x: x[0])
    adapted_ep_idx: Optional[int] = None
    for idx, (_, hit_rate, post_ret) in enumerate(ep_rows, start=1):
        if hit_rate >= 0.30 and post_ret > 0.0:
            adapted_ep_idx = idx
            break
    return {
        **res,
        "flip_step": int(flip_step),
        "episodes_to_adapt": adapted_ep_idx,
        "target_max_episodes_to_adapt": 5,
        "target_met": bool(adapted_ep_idx is not None and adapted_ep_idx <= 5),
        "episode_post_flip_stats": [
            {"episode_id": ep_id, "post_flip_target_hit_rate": float(hr), "post_flip_return": float(pr)}
            for ep_id, hr, pr in ep_rows
        ],
    }


def _run_fixed_seed_week4(
    *,
    out_dir: str,
    runs_root: str,
    episodes: int,
    max_steps: int,
    seeds: str,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "week4_fixed_seed_summary.json")
    cmd = [
        sys.executable,
        os.path.join("tools", "run_fixed_seed_benchmark.py"),
        "--runs_root",
        str(runs_root),
        "--target_verse",
        "warehouse_world",
        "--episodes",
        str(max(1, int(episodes))),
        "--max_steps",
        str(max(1, int(max_steps))),
        "--seeds",
        str(seeds),
        "--report_dir",
        str(os.path.join(out_dir, "fixed_seed_runs")),
        "--out_json",
        str(out_json),
        "--auto_bridge_tune",
        "--auto_bridge_tune_probe_episodes",
        "8",
        "--auto_bridge_tune_probe_max_steps",
        "80",
    ]
    proc = subprocess.run(cmd, cwd=os.getcwd())
    if int(proc.returncode) != 0:
        raise RuntimeError(f"run_fixed_seed_benchmark failed ({proc.returncode})")
    with open(out_json, "r", encoding="utf-8") as f:
        summary = json.load(f)
    per_seed = summary.get("per_seed", []) if isinstance(summary.get("per_seed"), list) else []
    tr_success: List[float] = []
    bl_success: List[float] = []
    for row in per_seed:
        m = row.get("metrics") if isinstance(row, dict) else {}
        if not isinstance(m, dict):
            continue
        tr_success.append(float(_safe_float(m.get("transfer_success_rate", 0.0), 0.0)))
        bl_success.append(float(_safe_float(m.get("baseline_success_rate", 0.0), 0.0)))
    transfer_success_mean = float(sum(tr_success) / float(max(1, len(tr_success)))) if tr_success else 0.0
    baseline_success_mean = float(sum(bl_success) / float(max(1, len(bl_success)))) if bl_success else 0.0
    target = 0.25
    return {
        "summary_path": out_json.replace("\\", "/"),
        "summary": summary,
        "transfer_success_rate_mean": float(transfer_success_mean),
        "baseline_success_rate_mean": float(baseline_success_mean),
        "baseline_reference_success_rate": 0.125,
        "target_success_rate": float(target),
        "target_met": bool(transfer_success_mean >= target),
    }


def _to_md(report: Dict[str, Any]) -> str:
    w1 = report.get("week1", {}) if isinstance(report.get("week1"), dict) else {}
    w2 = report.get("week2", {}) if isinstance(report.get("week2"), dict) else {}
    w3 = report.get("week3", {}) if isinstance(report.get("week3"), dict) else {}
    w4 = report.get("week4", {}) if isinstance(report.get("week4"), dict) else {}
    lines: List[str] = []
    lines.append("# Cognitive Memory Upgrade KPI Report")
    lines.append("")
    lines.append(f"- Created: `{report.get('created_at_iso', '')}`")
    lines.append(f"- Runs Root: `{report.get('runs_root', '')}`")
    lines.append(f"- Bench Root: `{report.get('bench_root', '')}`")
    lines.append("")
    lines.append("## Week 1 - LTM Retrieval")
    lines.append(f"- LTM hit rate: `{_safe_float(w1.get('ltm_hit_rate', 0.0), 0.0):.3f}`")
    lines.append(f"- Target: `>= {_safe_float(w1.get('target_ltm_hit_rate', 0.90), 0.90):.2f}`")
    lines.append(f"- Target met: `{bool(w1.get('target_met', False))}`")
    lines.append("")
    lines.append("## Week 2 - STM Diagnostic")
    lines.append(f"- Verse: `memory_vault_world`")
    lines.append(f"- Agent: `evolving`")
    lines.append(f"- Success rate: `{_safe_float(w2.get('success_rate', 0.0), 0.0):.3f}`")
    lines.append(f"- Mean return: `{_safe_float(w2.get('mean_return', 0.0), 0.0):.3f}`")
    lines.append(f"- STM capacity baseline: `{_safe_float(w2.get('stm_capacity_score', 0.0), 0.0):.3f}`")
    lines.append("")
    lines.append("## Week 3 - Memory Rewriting")
    lines.append(f"- Verse: `rule_flip_world`")
    lines.append(f"- Agent: `aware`")
    lines.append(f"- Episodes to adapt: `{w3.get('episodes_to_adapt', None)}`")
    lines.append(f"- Target: `<= {_safe_int(w3.get('target_max_episodes_to_adapt', 5), 5)} episodes`")
    lines.append(f"- Target met: `{bool(w3.get('target_met', False))}`")
    lines.append("")
    lines.append("## Week 4 - Warehouse Cognitive Challenge")
    lines.append(f"- Transfer success rate mean: `{_safe_float(w4.get('transfer_success_rate_mean', 0.0), 0.0):.3f}`")
    lines.append(f"- Baseline success rate mean: `{_safe_float(w4.get('baseline_success_rate_mean', 0.0), 0.0):.3f}`")
    lines.append(f"- Baseline reference (project): `{_safe_float(w4.get('baseline_reference_success_rate', 0.125), 0.125):.3f}`")
    lines.append(f"- Target: `> {_safe_float(w4.get('target_success_rate', 0.25), 0.25):.2f}`")
    lines.append(f"- Target met: `{bool(w4.get('target_met', False))}`")
    lines.append("")
    lines.append("## Outcome")
    lines.append(f"- Overall pass: `{bool(report.get('overall_pass', False))}`")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--bench_root", type=str, default=os.path.join("models", "tuning", "cognitive_upgrade"))
    ap.add_argument("--run_roots_for_memory", type=str, default="runs,runs_fixed_seed_20260211,runs_intensive_20260211")
    ap.add_argument("--memory_dir", type=str, default=os.path.join("central_memory", "cognitive_upgrade"))
    ap.add_argument("--memory_max_runs", type=int, default=24)
    ap.add_argument("--memory_max_ingest_runs", type=int, default=20)
    ap.add_argument("--memory_probe_limit", type=int, default=200)
    ap.add_argument("--memory_min_score", type=float, default=0.20)
    ap.add_argument("--week2_episodes", type=int, default=80)
    ap.add_argument("--week2_max_steps", type=int, default=120)
    ap.add_argument("--week3_episodes", type=int, default=80)
    ap.add_argument("--week3_max_steps", type=int, default=80)
    ap.add_argument("--week4_episodes", type=int, default=80)
    ap.add_argument("--week4_max_steps", type=int, default=100)
    ap.add_argument("--week4_seeds", type=str, default="123,223,337")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(str(args.bench_root), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_json = os.path.join(str(args.bench_root), f"cognitive_upgrade_kpi_{stamp}.json")
    out_md = os.path.join(str(args.bench_root), f"cognitive_upgrade_kpi_{stamp}.md")

    run_roots = [str(x).strip() for x in str(args.run_roots_for_memory).split(",") if str(x).strip()]
    run_dirs = _run_dirs_from_roots(run_roots, max_runs=max(1, int(args.memory_max_runs)))
    week1 = _memory_week1_kpi(
        run_dirs=run_dirs,
        memory_dir=str(args.memory_dir),
        max_ingest_runs=max(1, int(args.memory_max_ingest_runs)),
        probe_limit=max(1, int(args.memory_probe_limit)),
        min_score=float(args.memory_min_score),
    )
    week2 = _memory_vault_baseline(
        runs_root=str(args.runs_root),
        episodes=max(1, int(args.week2_episodes)),
        max_steps=max(1, int(args.week2_max_steps)),
        seed=int(args.seed),
    )
    week3 = _rule_flip_adaptation(
        runs_root=str(args.runs_root),
        episodes=max(1, int(args.week3_episodes)),
        max_steps=max(1, int(args.week3_max_steps)),
        seed=int(args.seed) + 1,
    )
    week4 = _run_fixed_seed_week4(
        out_dir=str(args.bench_root),
        runs_root=str(args.runs_root),
        episodes=max(1, int(args.week4_episodes)),
        max_steps=max(1, int(args.week4_max_steps)),
        seeds=str(args.week4_seeds),
    )

    overall_pass = bool(
        bool(week1.get("target_met", False))
        and bool(week3.get("target_met", False))
        and bool(week4.get("target_met", False))
    )
    report = {
        "created_at_iso": _utc_now_iso(),
        "runs_root": str(args.runs_root),
        "bench_root": str(args.bench_root).replace("\\", "/"),
        "week1": week1,
        "week2": week2,
        "week3": week3,
        "week4": week4,
        "overall_pass": bool(overall_pass),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(_to_md(report))
    latest_json = os.path.join(str(args.bench_root), "cognitive_upgrade_kpi_latest.json")
    latest_md = os.path.join(str(args.bench_root), "cognitive_upgrade_kpi_latest.md")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(_to_md(report))

    print(f"report_json={out_json}")
    print(f"report_md={out_md}")
    print(
        "week1_ltm_hit_rate={:.3f} week2_stm_capacity={:.3f} week3_adapt_ep={} week4_transfer_success={:.3f} overall_pass={}".format(
            float(_safe_float(week1.get("ltm_hit_rate", 0.0), 0.0)),
            float(_safe_float(week2.get("stm_capacity_score", 0.0), 0.0)),
            week3.get("episodes_to_adapt", None),
            float(_safe_float(week4.get("transfer_success_rate_mean", 0.0), 0.0)),
            bool(overall_pass),
        )
    )


if __name__ == "__main__":
    main()

