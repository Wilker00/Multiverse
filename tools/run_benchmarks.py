"""
tools/run_benchmarks.py

Long-horizon benchmark runner for promotion readiness.

Outputs:
- Markdown report
- JSON summary (can be consumed by promotion_board external benchmark input)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from agents.registry import create_agent, register_builtin_agents
from core.types import AgentSpec, VerseSpec
from orchestrator.eval_harness import (
    BenchmarkCase,
    VerseEvalSummary,
    default_benchmark_suite,
    evaluate_agent_case,
)
from verses.registry import create_verse, register_builtin


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


def _parse_kv(items: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not items:
        return out
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid k=v pair: {raw}")
        k, v = raw.split("=", 1)
        k = str(k).strip()
        v = str(v).strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue
        try:
            out[k] = int(v)
            continue
        except Exception:
            pass
        try:
            out[k] = float(v)
            continue
        except Exception:
            pass
        out[k] = v
    return out


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    i = max(0, min(len(ys) - 1, int(round(float(q) * (len(ys) - 1)))))
    return float(ys[i])


def _tail_risk(returns: List[float]) -> float:
    # Lower 10th percentile as downside proxy.
    return _quantile(list(returns), 0.10)


def _mean_regret(returns: List[float]) -> float:
    if not returns:
        return 0.0
    best = max(float(r) for r in returns)
    return float(sum(max(0.0, best - float(r)) for r in returns) / float(max(1, len(returns))))


def _confidence_calibration_error(agent_spec: AgentSpec, case: BenchmarkCase) -> float:
    """
    Approximate calibration error:
    |mean(confidence) - empirical_success_prob|.
    """
    register_builtin()
    register_builtin_agents()
    verse_spec = VerseSpec(
        spec_version="v1",
        verse_name=case.verse_name,
        verse_version=case.verse_version,
        seed=None,
        tags=["benchmark_confidence"],
        params=dict(case.params),
    )
    verse = create_verse(verse_spec)
    use_spec = agent_spec
    cfg = dict(use_spec.config or {})
    if str(use_spec.algo).strip().lower() in ("gateway", "special_moe", "adaptive_moe"):
        cfg.setdefault("verse_name", case.verse_name)
        use_spec = dataclasses.replace(use_spec, config=cfg)
    agent = create_agent(spec=use_spec, observation_space=verse.observation_space, action_space=verse.action_space)
    confs: List[float] = []
    succ: List[int] = []
    try:
        for base_seed in case.seeds:
            for ep in range(max(1, int(case.episodes_per_seed))):
                ep_seed = int(base_seed) * 1000 + ep
                verse.seed(ep_seed)
                agent.seed(ep_seed)
                rst = verse.reset()
                obs = rst.obs
                done = False
                success = False
                steps = 0
                while not done and steps < int(case.max_steps):
                    ar = agent.act(obs)
                    st = verse.step(ar.action)
                    info = st.info if isinstance(st.info, dict) else {}
                    se = info.get("safe_executor") if isinstance(info, dict) else None
                    if isinstance(se, dict):
                        c = se.get("confidence")
                        if c is not None:
                            confs.append(max(0.0, min(1.0, _safe_float(c, 0.0))))
                    if info.get("reached_goal") is True:
                        success = True
                    done = bool(st.done or st.truncated)
                    obs = st.obs
                    steps += 1
                succ.append(1 if success else 0)
    finally:
        verse.close()
        agent.close()
    if not confs:
        return 0.0
    mean_conf = float(sum(confs) / float(len(confs)))
    success_prob = float(sum(succ) / float(max(1, len(succ))))
    return float(abs(mean_conf - success_prob))


def _hard_cases() -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []
    # Hard labyrinth variants.
    for noise in (0.08, 0.14):
        cases.append(
            BenchmarkCase(
                verse_name="labyrinth_world",
                verse_version="0.1",
                params={
                    "adr_enabled": False,
                    "width": 17,
                    "height": 13,
                    "max_steps": 220,
                    "action_noise": float(noise),
                    "battery_capacity": 80,
                    "wall_follow_bonus": 0.02,
                },
                seeds=[111, 223],
                episodes_per_seed=3,
                max_steps=220,
            )
        )
    # Pursuit stress.
    cases.append(
        BenchmarkCase(
            verse_name="pursuit_world",
            verse_version="0.1",
            params={"adr_enabled": False, "max_steps": 100},
            seeds=[111, 223, 337],
            episodes_per_seed=3,
            max_steps=100,
        )
    )
    # Cliff stress.
    cases.append(
        BenchmarkCase(
            verse_name="cliff_world",
            verse_version="0.1",
            params={
                "adr_enabled": False,
                "width": 12,
                "height": 4,
                "max_steps": 120,
                "step_penalty": -1.0,
                "cliff_penalty": -100.0,
                "end_on_cliff": False,
            },
            seeds=[111, 223, 337],
            episodes_per_seed=3,
            max_steps=120,
        )
    )
    # Self-play adversary mix proxy: higher stochasticity/noise.
    cases.append(
        BenchmarkCase(
            verse_name="grid_world",
            verse_version="0.1",
            params={"adr_enabled": True, "adr_jitter": 0.20, "max_steps": 80},
            seeds=[111, 223, 337],
            episodes_per_seed=3,
            max_steps=80,
        )
    )
    return cases


def _summary_dict(s: VerseEvalSummary) -> Dict[str, Any]:
    return {
        "episodes": int(s.episodes),
        "mean_return": float(s.mean_return),
        "success_rate": float(s.success_rate),
        "failure_rate": float(s.failure_rate),
        "safety_violation_rate": float(s.safety_violation_rate),
        "return_variance": float(s.return_variance),
        "mean_steps": float(s.mean_steps),
        "tail_risk_p10": float(_tail_risk(s.returns)),
        "mean_regret": float(_mean_regret(s.returns)),
    }


def _to_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- Created: `{report.get('created_at', '')}`")
    lines.append(f"- Candidate: `{report.get('candidate_algo', '')}`")
    lines.append(f"- Baseline: `{report.get('baseline_algo', '')}`")
    lines.append(f"- Overall pass: `{report.get('overall_pass', False)}`")
    lines.append("")
    lines.append("| Verse | Candidate Return | Candidate Success | Baseline Return | Baseline Success | Tail Risk (P10) | Regret | Safety | Calibration | Pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_verse = report.get("by_verse", {})
    if isinstance(by_verse, dict):
        for verse, row in by_verse.items():
            if not isinstance(row, dict):
                continue
            c = row.get("candidate", {})
            b = row.get("baseline", {})
            lines.append(
                "| {verse} | {cr:.3f} | {cs:.3f} | {br:.3f} | {bs:.3f} | {tr:.3f} | {rg:.3f} | {sv:.3f} | {cal:.3f} | {ps} |".format(
                    verse=verse,
                    cr=_safe_float(c.get("mean_return", 0.0), 0.0),
                    cs=_safe_float(c.get("success_rate", 0.0), 0.0),
                    br=_safe_float(b.get("mean_return", 0.0), 0.0),
                    bs=_safe_float(b.get("success_rate", 0.0), 0.0),
                    tr=_safe_float(c.get("tail_risk_p10", 0.0), 0.0),
                    rg=_safe_float(c.get("mean_regret", 0.0), 0.0),
                    sv=_safe_float(c.get("safety_violation_rate", 0.0), 0.0),
                    cal=_safe_float(row.get("confidence_calibration_error", 0.0), 0.0),
                    ps=bool(row.get("passed", False)),
                )
            )
    lines.append("")
    lines.append("## Promotion Usage")
    lines.append("")
    lines.append("Use the JSON output with promotion board:")
    lines.append("")
    lines.append("```bash")
    lines.append("python tools/deploy_agent.py --promotion_disagreement_policy quarantine")
    lines.append("```")
    return "\n".join(lines)


def _evaluate_case_with_gateway_fallback(
    *,
    case: BenchmarkCase,
    agent_spec: AgentSpec,
    fallback_algo: str = "random",
) -> Tuple[VerseEvalSummary, Optional[str]]:
    try:
        return evaluate_agent_case(agent_spec=agent_spec, case=case), None
    except Exception as e:
        algo = str(getattr(agent_spec, "algo", "")).strip().lower()
        if algo != "gateway":
            raise
        fallback = dataclasses.replace(
            agent_spec,
            algo=str(fallback_algo),
            config=None,
            policy_id=f"{str(agent_spec.policy_id)}:{str(fallback_algo)}_fallback",
        )
        out = evaluate_agent_case(agent_spec=fallback, case=case)
        return out, f"gateway_failed:{type(e).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_algo", type=str, required=True)
    ap.add_argument("--candidate_policy_id", type=str, default="bench_candidate")
    ap.add_argument("--candidate_config", action="append", default=None)
    ap.add_argument("--baseline_algo", type=str, default="gateway")
    ap.add_argument("--baseline_policy_id", type=str, default="bench_baseline")
    ap.add_argument("--baseline_config", action="append", default=None)
    ap.add_argument("--manifest_path", type=str, default=os.path.join("models", "default_policy_set.json"))
    ap.add_argument("--mode", type=str, default="hard", choices=["quick", "full", "hard"])
    ap.add_argument("--out_dir", type=str, default=os.path.join("models", "benchmarks"))
    args = ap.parse_args()

    c_cfg = _parse_kv(args.candidate_config)
    b_cfg = _parse_kv(args.baseline_config)
    if str(args.baseline_algo).strip().lower() == "gateway":
        b_cfg.setdefault("manifest_path", str(args.manifest_path))

    candidate = AgentSpec(
        spec_version="v1",
        policy_id=str(args.candidate_policy_id),
        policy_version="0.1",
        algo=str(args.candidate_algo),
        seed=123,
        tags=["bench_candidate"],
        config=(c_cfg if c_cfg else None),
    )
    baseline = AgentSpec(
        spec_version="v1",
        policy_id=str(args.baseline_policy_id),
        policy_version="0.1",
        algo=str(args.baseline_algo),
        seed=123,
        tags=["bench_baseline"],
        config=(b_cfg if b_cfg else None),
    )

    if str(args.mode).strip().lower() == "hard":
        cases = _hard_cases()
    else:
        cases = default_benchmark_suite(mode=str(args.mode))

    by_verse: Dict[str, Any] = {}
    overall_pass = True
    for case in cases:
        cand = evaluate_agent_case(agent_spec=candidate, case=case)
        base, baseline_fallback_reason = _evaluate_case_with_gateway_fallback(
            case=case,
            agent_spec=baseline,
            fallback_algo="random",
        )
        cal = _confidence_calibration_error(candidate, case)

        # Conservative pass criteria for long horizon.
        min_success = float(base.success_rate) * 0.90
        min_tail = float(_tail_risk(base.returns)) - abs(float(_tail_risk(base.returns))) * 0.20
        max_safety = max(0.20, float(base.safety_violation_rate) * 1.20)
        passed = bool(
            float(cand.success_rate) >= min_success
            and float(_tail_risk(cand.returns)) >= min_tail
            and float(cand.safety_violation_rate) <= max_safety
        )
        overall_pass = bool(overall_pass and passed)
        by_verse[case.verse_name] = {
            "passed": bool(passed),
            "candidate": _summary_dict(cand),
            "baseline": _summary_dict(base),
            "baseline_fallback_reason": baseline_fallback_reason,
            "constraints": {
                "min_success_rate": float(min_success),
                "min_tail_risk_p10": float(min_tail),
                "max_safety_violation_rate": float(max_safety),
            },
            "confidence_calibration_error": float(cal),
        }
        print(
            f"[{case.verse_name}] pass={passed} "
            f"cand_return={cand.mean_return:.3f} base_return={base.mean_return:.3f} "
            f"cand_success={cand.success_rate:.3f} base_success={base.success_rate:.3f}"
        )

    report = {
        "created_at": int(time.time() * 1000),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": str(args.mode),
        "candidate_algo": str(args.candidate_algo),
        "baseline_algo": str(args.baseline_algo),
        "overall_pass": bool(overall_pass),
        "by_verse": by_verse,
        "summary": {
            "mean_candidate_return": float(
                statistics.mean(
                    [_safe_float(r.get("candidate", {}).get("mean_return", 0.0), 0.0) for r in by_verse.values()]
                )
            )
            if by_verse
            else 0.0,
            "mean_candidate_success_rate": float(
                statistics.mean(
                    [_safe_float(r.get("candidate", {}).get("success_rate", 0.0), 0.0) for r in by_verse.values()]
                )
            )
            if by_verse
            else 0.0,
            "mean_calibration_error": float(
                statistics.mean([_safe_float(r.get("confidence_calibration_error", 0.0), 0.0) for r in by_verse.values()])
            )
            if by_verse
            else 0.0,
        },
    }

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    json_out = os.path.join(out_dir, f"benchmark_report_{stamp}.json")
    md_out = os.path.join(out_dir, f"benchmark_report_{stamp}.md")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(_to_markdown(report))

    latest_json = os.path.join(out_dir, "latest.json")
    latest_md = os.path.join(out_dir, "latest.md")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(_to_markdown(report))

    print(f"benchmark_json={json_out}")
    print(f"benchmark_md={md_out}")
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
