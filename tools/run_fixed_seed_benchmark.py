"""
tools/run_fixed_seed_benchmark.py

Runs fixed-seed transfer benchmarks and aggregates transfer/safety/health metrics.

This tool wraps `tools/run_transfer_challenge.py` to produce reproducible
multi-seed summaries for regression tracking and autonomous cycles.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
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


def parse_seed_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        out.append(int(s))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("No valid seeds provided.")
    return uniq


def extract_seed_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    cmp = report.get("comparison", {}) if isinstance(report.get("comparison"), dict) else {}
    ps = report.get("production_summary", {}) if isinstance(report.get("production_summary"), dict) else {}
    safety = ps.get("safety", {}) if isinstance(ps.get("safety"), dict) else {}
    health = ps.get("health", {}) if isinstance(ps.get("health"), dict) else {}
    transfer_h = health.get("transfer", {}) if isinstance(health.get("transfer"), dict) else {}
    baseline_h = health.get("baseline", {}) if isinstance(health.get("baseline"), dict) else {}
    transfer_ag = report.get("transfer_agent", {}) if isinstance(report.get("transfer_agent"), dict) else {}
    baseline_ag = report.get("baseline_agent", {}) if isinstance(report.get("baseline_agent"), dict) else {}
    transfer_eval = transfer_ag.get("eval", {}) if isinstance(transfer_ag.get("eval"), dict) else {}
    baseline_eval = baseline_ag.get("eval", {}) if isinstance(baseline_ag.get("eval"), dict) else {}
    transfer_diag = transfer_ag.get("diagnostics", {}) if isinstance(transfer_ag.get("diagnostics"), dict) else {}
    baseline_diag = baseline_ag.get("diagnostics", {}) if isinstance(baseline_ag.get("diagnostics"), dict) else {}
    transfer_early = transfer_diag.get("early_window", {}) if isinstance(transfer_diag.get("early_window"), dict) else {}
    baseline_early = baseline_diag.get("early_window", {}) if isinstance(baseline_diag.get("early_window"), dict) else {}
    transfer_action = transfer_diag.get("action_agreement", {}) if isinstance(transfer_diag.get("action_agreement"), dict) else {}
    baseline_action = baseline_diag.get("action_agreement", {}) if isinstance(baseline_diag.get("action_agreement"), dict) else {}
    transfer_td = transfer_diag.get("td_error", {}) if isinstance(transfer_diag.get("td_error"), dict) else {}
    ds_diag = report.get("transfer_dataset_diagnostics", {}) if isinstance(report.get("transfer_dataset_diagnostics"), dict) else {}
    score_diag = ds_diag.get("score_distribution", {}) if isinstance(ds_diag.get("score_distribution"), dict) else {}
    score_dist = score_diag.get("transfer_score", {}) if isinstance(score_diag.get("transfer_score"), dict) else {}
    return {
        "transfer_wins_convergence": bool(cmp.get("transfer_wins_convergence", False)),
        "transfer_speedup_ratio": (
            None
            if cmp.get("transfer_speedup_ratio") is None
            else float(_safe_float(cmp.get("transfer_speedup_ratio"), 0.0))
        ),
        "hazard_improvement_pct": float(_safe_float(cmp.get("hazard_improvement_pct", 0.0), 0.0)),
        "transfer_hazard_per_1k": float(_safe_float(safety.get("transfer_hazard_per_1k", 0.0), 0.0)),
        "baseline_hazard_per_1k": float(_safe_float(safety.get("baseline_hazard_per_1k", 0.0), 0.0)),
        "transfer_mcts_veto_rate": float(_safe_float(safety.get("transfer_mcts_veto_rate", 0.0), 0.0)),
        "baseline_mcts_veto_rate": float(_safe_float(safety.get("baseline_mcts_veto_rate", 0.0), 0.0)),
        "transfer_health_score": (
            None
            if transfer_h.get("total_score") is None
            else float(_safe_float(transfer_h.get("total_score"), 0.0))
        ),
        "baseline_health_score": (
            None
            if baseline_h.get("total_score") is None
            else float(_safe_float(baseline_h.get("total_score"), 0.0))
        ),
        "transfer_health_status": str(transfer_h.get("status", "")),
        "baseline_health_status": str(baseline_h.get("status", "")),
        "transfer_mean_return": float(_safe_float(transfer_eval.get("mean_return", 0.0), 0.0)),
        "baseline_mean_return": float(_safe_float(baseline_eval.get("mean_return", 0.0), 0.0)),
        "transfer_success_rate": float(_safe_float(transfer_eval.get("success_rate", 0.0), 0.0)),
        "baseline_success_rate": float(_safe_float(baseline_eval.get("success_rate", 0.0), 0.0)),
        "transfer_early_mean_return": (
            None if transfer_early.get("mean_return") is None else float(_safe_float(transfer_early.get("mean_return"), 0.0))
        ),
        "baseline_early_mean_return": (
            None if baseline_early.get("mean_return") is None else float(_safe_float(baseline_early.get("mean_return"), 0.0))
        ),
        "transfer_early_hazard_per_1k": (
            None
            if transfer_early.get("hazard_events_per_1k_steps") is None
            else float(_safe_float(transfer_early.get("hazard_events_per_1k_steps"), 0.0))
        ),
        "baseline_early_hazard_per_1k": (
            None
            if baseline_early.get("hazard_events_per_1k_steps") is None
            else float(_safe_float(baseline_early.get("hazard_events_per_1k_steps"), 0.0))
        ),
        "transfer_action_agreement": (
            None if transfer_action.get("agreement_rate") is None else float(_safe_float(transfer_action.get("agreement_rate"), 0.0))
        ),
        "baseline_action_agreement": (
            None if baseline_action.get("agreement_rate") is None else float(_safe_float(baseline_action.get("agreement_rate"), 0.0))
        ),
        "transfer_td_native_mean_early": (
            None if transfer_td.get("native_td_abs_mean_early") is None else float(_safe_float(transfer_td.get("native_td_abs_mean_early"), 0.0))
        ),
        "transfer_td_transfer_mean_early": (
            None if transfer_td.get("transfer_td_abs_mean_early") is None else float(_safe_float(transfer_td.get("transfer_td_abs_mean_early"), 0.0))
        ),
        "transfer_td_score_corr_early": (
            None if transfer_td.get("transfer_td_score_corr_early") is None else float(_safe_float(transfer_td.get("transfer_td_score_corr_early"), 0.0))
        ),
        "transfer_score_p10": (None if score_dist.get("p10") is None else float(_safe_float(score_dist.get("p10"), 0.0))),
        "transfer_score_p50": (None if score_dist.get("p50") is None else float(_safe_float(score_dist.get("p50"), 0.0))),
        "transfer_score_p90": (None if score_dist.get("p90") is None else float(_safe_float(score_dist.get("p90"), 0.0))),
    }


def aggregate_seed_metrics(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(seed_rows)
    if n <= 0:
        return {
            "num_seeds": 0,
            "win_rate": 0.0,
            "mean_speedup_ratio": None,
            "mean_hazard_improvement_pct": 0.0,
            "mean_transfer_hazard_per_1k": 0.0,
            "mean_baseline_hazard_per_1k": 0.0,
            "mean_transfer_health_score": None,
            "mean_baseline_health_score": None,
            "mean_transfer_early_mean_return": None,
            "mean_transfer_early_hazard_per_1k": None,
            "mean_transfer_action_agreement": None,
            "mean_transfer_td_native_mean_early": None,
            "mean_transfer_td_transfer_mean_early": None,
            "mean_transfer_td_score_corr_early": None,
            "mean_transfer_score_p10": None,
            "mean_transfer_score_p50": None,
            "mean_transfer_score_p90": None,
            "transfer_status_counts": {},
            "baseline_status_counts": {},
        }

    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / float(max(1, len(vals))))

    wins = sum(1 for r in seed_rows if bool(r.get("transfer_wins_convergence", False)))
    speedups = [float(r["transfer_speedup_ratio"]) for r in seed_rows if isinstance(r.get("transfer_speedup_ratio"), (int, float))]
    haz_gain = [float(_safe_float(r.get("hazard_improvement_pct", 0.0), 0.0)) for r in seed_rows]
    tr_haz = [float(_safe_float(r.get("transfer_hazard_per_1k", 0.0), 0.0)) for r in seed_rows]
    bl_haz = [float(_safe_float(r.get("baseline_hazard_per_1k", 0.0), 0.0)) for r in seed_rows]
    tr_hs = [float(r["transfer_health_score"]) for r in seed_rows if isinstance(r.get("transfer_health_score"), (int, float))]
    bl_hs = [float(r["baseline_health_score"]) for r in seed_rows if isinstance(r.get("baseline_health_score"), (int, float))]
    tr_early_ret = [float(r["transfer_early_mean_return"]) for r in seed_rows if isinstance(r.get("transfer_early_mean_return"), (int, float))]
    tr_early_haz = [float(r["transfer_early_hazard_per_1k"]) for r in seed_rows if isinstance(r.get("transfer_early_hazard_per_1k"), (int, float))]
    tr_agree = [float(r["transfer_action_agreement"]) for r in seed_rows if isinstance(r.get("transfer_action_agreement"), (int, float))]
    tr_td_native = [float(r["transfer_td_native_mean_early"]) for r in seed_rows if isinstance(r.get("transfer_td_native_mean_early"), (int, float))]
    tr_td_transfer = [float(r["transfer_td_transfer_mean_early"]) for r in seed_rows if isinstance(r.get("transfer_td_transfer_mean_early"), (int, float))]
    tr_td_corr = [float(r["transfer_td_score_corr_early"]) for r in seed_rows if isinstance(r.get("transfer_td_score_corr_early"), (int, float))]
    tr_sc_p10 = [float(r["transfer_score_p10"]) for r in seed_rows if isinstance(r.get("transfer_score_p10"), (int, float))]
    tr_sc_p50 = [float(r["transfer_score_p50"]) for r in seed_rows if isinstance(r.get("transfer_score_p50"), (int, float))]
    tr_sc_p90 = [float(r["transfer_score_p90"]) for r in seed_rows if isinstance(r.get("transfer_score_p90"), (int, float))]

    tr_status: Dict[str, int] = {}
    bl_status: Dict[str, int] = {}
    for r in seed_rows:
        ts = str(r.get("transfer_health_status", "")).strip().lower()
        bs = str(r.get("baseline_health_status", "")).strip().lower()
        if ts:
            tr_status[ts] = int(tr_status.get(ts, 0)) + 1
        if bs:
            bl_status[bs] = int(bl_status.get(bs, 0)) + 1

    out = {
        "num_seeds": int(n),
        "win_rate": float(wins / float(max(1, n))),
        "mean_speedup_ratio": (None if not speedups else _mean(speedups)),
        "median_speedup_ratio": (None if not speedups else float(statistics.median(speedups))),
        "mean_hazard_improvement_pct": _mean(haz_gain),
        "mean_transfer_hazard_per_1k": _mean(tr_haz),
        "mean_baseline_hazard_per_1k": _mean(bl_haz),
        "mean_transfer_health_score": (None if not tr_hs else _mean(tr_hs)),
        "mean_baseline_health_score": (None if not bl_hs else _mean(bl_hs)),
        "mean_transfer_early_mean_return": (None if not tr_early_ret else _mean(tr_early_ret)),
        "mean_transfer_early_hazard_per_1k": (None if not tr_early_haz else _mean(tr_early_haz)),
        "mean_transfer_action_agreement": (None if not tr_agree else _mean(tr_agree)),
        "mean_transfer_td_native_mean_early": (None if not tr_td_native else _mean(tr_td_native)),
        "mean_transfer_td_transfer_mean_early": (None if not tr_td_transfer else _mean(tr_td_transfer)),
        "mean_transfer_td_score_corr_early": (None if not tr_td_corr else _mean(tr_td_corr)),
        "mean_transfer_score_p10": (None if not tr_sc_p10 else _mean(tr_sc_p10)),
        "mean_transfer_score_p50": (None if not tr_sc_p50 else _mean(tr_sc_p50)),
        "mean_transfer_score_p90": (None if not tr_sc_p90 else _mean(tr_sc_p90)),
        "transfer_status_counts": tr_status,
        "baseline_status_counts": bl_status,
    }
    return out


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return obj


def _run_cmd(cmd: List[str], *, cwd: str) -> None:
    proc = subprocess.run(cmd, cwd=cwd)
    if int(proc.returncode) != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _append_extra_args(cmd: List[str], extra_tokens: Optional[List[str]]) -> None:
    for tok in (extra_tokens or []):
        s = str(tok).strip()
        if s:
            cmd.append(s)


def _has_arg_token(tokens: Optional[List[str]], name: str) -> bool:
    n = str(name).strip()
    for tok in (tokens or []):
        s = str(tok).strip()
        if not s:
            continue
        if s == n or s.startswith(n + "="):
            return True
    return False


def _parse_float_list(raw: str, *, default_vals: List[float]) -> List[float]:
    vals: List[float] = []
    for part in str(raw or "").replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except Exception:
            continue
    uniq = sorted(set(vals))
    return uniq if uniq else list(default_vals)


def _build_challenge_cmd(
    *,
    py: str,
    runs_root: str,
    target_verse: str,
    episodes: int,
    max_steps: int,
    seed: int,
    transfer_algo: str,
    baseline_algo: str,
    health_trace_root: str,
    report_out: str,
    overlap_out: str,
    extra_tokens: Optional[List[str]] = None,
) -> List[str]:
    cmd = [
        py,
        os.path.join("tools", "run_transfer_challenge.py"),
        "--runs_root",
        str(runs_root),
        "--target_verse",
        str(target_verse),
        "--episodes",
        str(max(1, int(episodes))),
        "--max_steps",
        str(max(1, int(max_steps))),
        "--seed",
        str(int(seed)),
        "--transfer_algo",
        str(transfer_algo),
        "--baseline_algo",
        str(baseline_algo),
        "--health_trace_root",
        str(health_trace_root),
        "--report_out",
        report_out,
        "--overlap_out",
        overlap_out,
        "--chart_stride",
        "999999",
    ]
    _append_extra_args(cmd, extra_tokens)
    return cmd


def _profile_score(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    # Conservative ranking: prioritize safety transfer gain, then success, then return.
    hazard_gain = float(_safe_float(metrics.get("hazard_improvement_pct", 0.0), 0.0))
    success = float(_safe_float(metrics.get("transfer_success_rate", 0.0), 0.0))
    transfer_ret = float(_safe_float(metrics.get("transfer_mean_return", 0.0), 0.0))
    return (hazard_gain, success, transfer_ret)


def _auto_bridge_tune(
    *,
    py: str,
    args: argparse.Namespace,
    seeds: List[int],
    root: str,
    base_extra_tokens: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    # Keep it simple and stable: only tune two bridge knobs from small preset candidates.
    if _has_arg_token(base_extra_tokens, "--bridge_synthetic_reward_blend") or _has_arg_token(
        base_extra_tokens, "--transfer_filter_hazard_keep_ratio"
    ):
        return list(base_extra_tokens), {
            "enabled": True,
            "applied": False,
            "reason": "explicit_bridge_args_provided",
            "selected_profile": None,
            "profiles": [],
        }

    blends = _parse_float_list(
        str(args.auto_bridge_tune_blends),
        default_vals=[0.18, 0.20, 0.25],
    )
    keep_ratios = _parse_float_list(
        str(args.auto_bridge_tune_hazard_keep_ratios),
        default_vals=[1.0],
    )
    probe_seed = int(seeds[0])
    probe_episodes = max(1, int(args.auto_bridge_tune_probe_episodes))
    probe_max_steps = max(1, int(args.auto_bridge_tune_probe_max_steps))
    probe_dir = os.path.join(root, "autotune_probe")
    os.makedirs(probe_dir, exist_ok=True)

    profiles: List[Dict[str, Any]] = []
    for blend in blends:
        for keep in keep_ratios:
            name = f"b{int(round(float(blend) * 1000.0)):03d}_k{int(round(float(keep) * 1000.0)):03d}"
            report_out = os.path.join(probe_dir, f"{name}_seed_{probe_seed}.json")
            overlap_out = os.path.join(probe_dir, f"{name}_seed_{probe_seed}_overlap.json")
            tune_tokens = [
                "--bridge_synthetic_reward_blend",
                str(float(blend)),
                "--transfer_filter_hazard_keep_ratio",
                str(float(keep)),
            ]
            cmd = _build_challenge_cmd(
                py=py,
                runs_root=str(args.runs_root),
                target_verse=str(args.target_verse),
                episodes=int(probe_episodes),
                max_steps=int(probe_max_steps),
                seed=int(probe_seed),
                transfer_algo=str(args.transfer_algo),
                baseline_algo=str(args.baseline_algo),
                health_trace_root=str(args.health_trace_root),
                report_out=report_out,
                overlap_out=overlap_out,
                extra_tokens=list(base_extra_tokens) + tune_tokens,
            )
            try:
                _run_cmd(cmd, cwd=os.getcwd())
                rep = _read_json(report_out)
                metrics = extract_seed_metrics(rep)
                score = _profile_score(metrics)
                profiles.append(
                    {
                        "name": name,
                        "blend": float(blend),
                        "hazard_keep_ratio": float(keep),
                        "metrics": metrics,
                        "score": [float(score[0]), float(score[1]), float(score[2])],
                    }
                )
            except Exception as e:
                profiles.append(
                    {
                        "name": name,
                        "blend": float(blend),
                        "hazard_keep_ratio": float(keep),
                        "error": str(e),
                    }
                )

    valid = [p for p in profiles if isinstance(p.get("metrics"), dict)]
    if not valid:
        return list(base_extra_tokens), {
            "enabled": True,
            "applied": False,
            "reason": "no_valid_probe_profiles",
            "selected_profile": None,
            "profiles": profiles,
        }

    valid.sort(
        key=lambda p: tuple(p.get("score", [float("-inf"), float("-inf"), float("-inf")])),
        reverse=True,
    )
    best = valid[0]
    selected_tokens = list(base_extra_tokens) + [
        "--bridge_synthetic_reward_blend",
        str(float(best.get("blend", 0.20))),
        "--transfer_filter_hazard_keep_ratio",
        str(float(best.get("hazard_keep_ratio", 1.0))),
    ]
    return selected_tokens, {
        "enabled": True,
        "applied": True,
        "reason": "selected_best_probe_profile",
        "probe_seed": int(probe_seed),
        "probe_episodes": int(probe_episodes),
        "probe_max_steps": int(probe_max_steps),
        "selected_profile": best,
        "profiles": profiles,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--target_verse", type=str, default="warehouse_world", choices=["warehouse_world", "labyrinth_world"])
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--transfer_algo", type=str, default="q")
    ap.add_argument("--baseline_algo", type=str, default="q")
    ap.add_argument("--seeds", type=str, default="123,223,337")
    ap.add_argument("--report_dir", type=str, default=os.path.join("models", "benchmarks", "fixed_seed"))
    ap.add_argument("--health_trace_root", type=str, default=os.path.join("models", "expert_datasets"))
    ap.add_argument("--challenge_arg", action="append", default=None, help="Extra arg token passed to run_transfer_challenge.py")
    ap.add_argument("--auto_bridge_tune", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--auto_bridge_tune_blends", type=str, default="0.18,0.20,0.25")
    ap.add_argument("--auto_bridge_tune_hazard_keep_ratios", type=str, default="1.0")
    ap.add_argument("--auto_bridge_tune_probe_episodes", type=int, default=12)
    ap.add_argument("--auto_bridge_tune_probe_max_steps", type=int, default=80)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    seeds = parse_seed_list(str(args.seeds))
    root = str(args.report_dir)
    os.makedirs(root, exist_ok=True)
    py = sys.executable
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    base_extra_tokens = [str(t).strip() for t in (args.challenge_arg or []) if str(t).strip()]
    selected_extra_tokens = list(base_extra_tokens)
    auto_tune_info: Dict[str, Any] = {
        "enabled": bool(args.auto_bridge_tune),
        "applied": False,
        "reason": "disabled",
        "selected_profile": None,
        "profiles": [],
    }
    if bool(args.auto_bridge_tune):
        selected_extra_tokens, auto_tune_info = _auto_bridge_tune(
            py=py,
            args=args,
            seeds=seeds,
            root=root,
            base_extra_tokens=base_extra_tokens,
        )
        print(
            f"auto_bridge_tune: applied={bool(auto_tune_info.get('applied', False))} "
            f"reason={str(auto_tune_info.get('reason', ''))}"
        )

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        report_out = os.path.join(root, f"transfer_seed_{int(seed)}.json")
        overlap_out = os.path.join(root, f"transfer_seed_{int(seed)}_overlap.json")
        cmd = _build_challenge_cmd(
            py=py,
            runs_root=str(args.runs_root),
            target_verse=str(args.target_verse),
            episodes=max(1, int(args.episodes)),
            max_steps=max(1, int(args.max_steps)),
            seed=int(seed),
            transfer_algo=str(args.transfer_algo),
            baseline_algo=str(args.baseline_algo),
            health_trace_root=str(args.health_trace_root),
            report_out=report_out,
            overlap_out=overlap_out,
            extra_tokens=selected_extra_tokens,
        )

        print(f"[seed={seed}] running transfer challenge...")
        _run_cmd(cmd, cwd=os.getcwd())
        rep = _read_json(report_out)
        metrics = extract_seed_metrics(rep)
        per_seed.append(
            {
                "seed": int(seed),
                "report_out": report_out.replace("\\", "/"),
                "overlap_out": overlap_out.replace("\\", "/"),
                "metrics": metrics,
            }
        )
        print(
            f"[seed={seed}] win={bool(metrics.get('transfer_wins_convergence', False))} "
            f"hazard_gain_pct={float(_safe_float(metrics.get('hazard_improvement_pct', 0.0), 0.0)):.2f} "
            f"transfer_health={metrics.get('transfer_health_status', '')}"
        )

    agg = aggregate_seed_metrics([dict(x.get("metrics", {})) for x in per_seed])
    summary = {
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_verse": str(args.target_verse),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "transfer_algo": str(args.transfer_algo),
        "baseline_algo": str(args.baseline_algo),
        "seeds": [int(s) for s in seeds],
        "challenge_args_applied": list(selected_extra_tokens),
        "auto_bridge_tune": auto_tune_info,
        "aggregate": agg,
        "per_seed": per_seed,
    }

    out_json = str(args.out_json).strip()
    if not out_json:
        out_json = os.path.join(root, f"fixed_seed_summary_{stamp}.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    latest = os.path.join(root, "latest.json")
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"summary_json={out_json}")
    print(
        f"win_rate={float(_safe_float(agg.get('win_rate', 0.0), 0.0)):.3f} "
        f"mean_hazard_gain_pct={float(_safe_float(agg.get('mean_hazard_improvement_pct', 0.0), 0.0)):.2f}"
    )


if __name__ == "__main__":
    main()
