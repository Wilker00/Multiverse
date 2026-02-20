"""
tools/run_teacher_remediation.py

Launch a focused Teacher-Frontier remediation session:
target verse -> tutorial verse -> graduation verse.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from agents.registry import create_agent, register_builtin_agents
from core.rollout import RolloutConfig, run_episodes
from core.safe_executor import SafeExecutor, SafeExecutorConfig
from core.types import AgentRef, AgentSpec, RunRef, VerseRef, VerseSpec
from memory.event_log import EventLogConfig, EventLogger, make_on_step_writer
from orchestrator.teacher import (
    TeacherConfig,
    TeacherLessonPlan,
    append_teacher_lesson,
    build_teacher_plan,
    collect_high_risk_signals,
    synthesize_tutorial_spec,
)
from orchestrator.trainer import Trainer
from theory.safety_bounds import derive_safety_certificate, extract_episode_violation_flags_from_events
from verses.registry import create_verse, register_builtin


def _run_training(
    *,
    trainer: Trainer,
    verse_spec: VerseSpec,
    algo: str,
    policy_id: str,
    episodes: int,
    max_steps: int,
    seed: int,
    train: bool,
    safe_guard: bool,
    agent_config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"train": bool(train)}
    if isinstance(agent_config_overrides, dict) and agent_config_overrides:
        cfg.update({str(k): v for k, v in agent_config_overrides.items()})
    if bool(safe_guard):
        cfg["safe_executor"] = {
            "enabled": True,
            "danger_threshold": 0.55,
            "adaptive_veto_enabled": True,
            "adaptive_veto_schedule_enabled": True,
            "adaptive_veto_relaxation_start": 0.05,
            "adaptive_veto_relaxation_end": 0.15,
            "adaptive_veto_schedule_steps": 1500,
            "adaptive_veto_schedule_power": 1.2,
            "adaptive_veto_warmup_steps": 20,
            "adaptive_veto_failure_guard": 0.12,
        }
    run = trainer.run(
        verse_spec=verse_spec.evolved(params={**dict(verse_spec.params or {}), "max_steps": int(max_steps)}),
        agent_spec=AgentSpec(
            spec_version="v1",
            policy_id=str(policy_id),
            policy_version="0.1",
            algo=str(algo),
            seed=int(seed),
            config=cfg,
        ),
        episodes=max(1, int(episodes)),
        max_steps=max(1, int(max_steps)),
        seed=int(seed),
    )
    return run


def _spec_hash(spec: VerseSpec) -> str:
    import hashlib

    payload = {
        "spec_version": str(spec.spec_version),
        "verse_name": str(spec.verse_name),
        "verse_version": str(spec.verse_version),
        "seed": spec.seed,
        "tags": list(spec.tags or []),
        "params": dict(spec.params or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _safe_executor_cfg(enabled: bool) -> Dict[str, Any]:
    if not bool(enabled):
        return {"enabled": False}
    return {
        "enabled": True,
        "danger_threshold": 0.55,
        "adaptive_veto_enabled": True,
        "adaptive_veto_schedule_enabled": True,
        "adaptive_veto_relaxation_start": 0.05,
        "adaptive_veto_relaxation_end": 0.15,
        "adaptive_veto_schedule_steps": 1500,
        "adaptive_veto_schedule_power": 1.2,
        "adaptive_veto_warmup_steps": 20,
        "adaptive_veto_failure_guard": 0.12,
    }


def _set_agent_eval_epsilon(agent: Any, epsilon: float) -> Optional[float]:
    try:
        stats = getattr(agent, "stats", None)
        if stats is None:
            return None
        prev = float(getattr(stats, "epsilon"))
        setattr(stats, "epsilon", float(max(0.0, min(1.0, epsilon))))
        return prev
    except Exception:
        return None


def _build_algo_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if str(getattr(args, "algo", "")).strip().lower() != "q":
        return cfg
    if getattr(args, "q_lr", None) is not None:
        cfg["lr"] = float(getattr(args, "q_lr"))
    if getattr(args, "q_gamma", None) is not None:
        cfg["gamma"] = float(getattr(args, "q_gamma"))
    if getattr(args, "q_epsilon_start", None) is not None:
        cfg["epsilon_start"] = float(getattr(args, "q_epsilon_start"))
    if getattr(args, "q_epsilon_min", None) is not None:
        cfg["epsilon_min"] = float(getattr(args, "q_epsilon_min"))
    if getattr(args, "q_epsilon_decay", None) is not None:
        cfg["epsilon_decay"] = float(getattr(args, "q_epsilon_decay"))
    if getattr(args, "q_learn_hazard_penalty", None) is not None:
        cfg["learn_hazard_penalty"] = float(getattr(args, "q_learn_hazard_penalty"))
    if getattr(args, "q_learn_success_bonus", None) is not None:
        cfg["learn_success_bonus"] = float(getattr(args, "q_learn_success_bonus"))
    if getattr(args, "q_warmstart_reward_scale", None) is not None:
        cfg["warmstart_reward_scale"] = float(getattr(args, "q_warmstart_reward_scale"))
    if getattr(args, "q_dataset_path", None):
        cfg["dataset_path"] = str(getattr(args, "q_dataset_path"))
    return cfg


def _run_phase_with_existing_agent(
    *,
    run_root: str,
    verse_spec: VerseSpec,
    agent: Any,
    agent_ref: AgentRef,
    episodes: int,
    max_steps: int,
    seed: int,
    train: bool,
    safe_guard: bool,
) -> Dict[str, Any]:
    verse = create_verse(verse_spec.evolved(params={**dict(verse_spec.params or {}), "max_steps": int(max_steps)}))
    verse_ref = VerseRef.create(
        verse_name=str(verse_spec.verse_name),
        verse_version=str(verse_spec.verse_version),
        spec_hash=_spec_hash(verse_spec),
    )
    safe_cfg = SafeExecutorConfig.from_dict(_safe_executor_cfg(bool(safe_guard)))
    safe_executor = SafeExecutor(config=safe_cfg, verse=verse)
    run = RunRef.create()
    rollout_cfg = RolloutConfig(
        schema_version="v1",
        max_steps=int(max_steps),
        train=bool(train),
        collect_transitions=bool(train),
        safe_executor=safe_executor,
    )
    log_cfg = EventLogConfig(root_dir=str(run_root), run_id=run.run_id)
    try:
        with EventLogger(log_cfg) as logger:
            on_step = make_on_step_writer(logger)
            run_episodes(
                verse=verse,
                verse_ref=verse_ref,
                agent=agent,
                agent_ref=agent_ref,
                run=run,
                config=rollout_cfg,
                episodes=max(1, int(episodes)),
                seed=int(seed),
                on_step=on_step,
            )
    finally:
        safe_executor.close()
        verse.close()
    run_dir = os.path.join(str(run_root), str(run.run_id))
    total_return = 0.0
    total_steps = 0
    events_path = os.path.join(run_dir, "events.jsonl")
    if os.path.isfile(events_path):
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                total_return += float(row.get("reward", 0.0))
                total_steps += 1
    return {"run_id": str(run.run_id), "total_return": float(total_return), "total_steps": int(total_steps)}


def _parse_stage_values(raw: str, *, default: Sequence[float]) -> List[float]:
    text = str(raw).strip()
    if not text:
        return [float(x) for x in default]
    out: List[float] = []
    for part in text.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out if out else [float(x) for x in default]


def _build_stage_schedule(args: argparse.Namespace) -> List[Tuple[float, float, int]]:
    winds = _parse_stage_values(str(args.stage_wind_csv), default=[0.0, 0.05, 0.10])
    crumbles = _parse_stage_values(str(args.stage_crumble_csv), default=[0.0, 0.01, 0.03])
    n = max(1, min(len(winds), len(crumbles)))
    winds = winds[:n]
    crumbles = crumbles[:n]
    total_eps = max(1, int(args.graduation_episodes))
    base = total_eps // n
    rem = total_eps % n
    eps: List[int] = []
    for i in range(n):
        e = base + (1 if i < rem else 0)
        eps.append(max(1, int(e)))
    return [(float(winds[i]), float(crumbles[i]), int(eps[i])) for i in range(n)]


def _parse_stage_float_csv(raw: str) -> List[float]:
    text = str(raw).strip()
    if not text:
        return []
    out: List[float] = []
    for part in text.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _stage_value(values: List[float], stage_index_1based: int) -> Optional[float]:
    if not values:
        return None
    idx = max(0, int(stage_index_1based) - 1)
    if idx < len(values):
        return float(values[idx])
    return float(values[-1])


def _apply_stage_q_overrides(
    *,
    agent: Any,
    stage_index_1based: int,
    hazard_csv: str,
    success_csv: str,
    epsilon_start_csv: str,
    epsilon_decay_csv: str,
) -> Dict[str, Any]:
    applied: Dict[str, Any] = {}
    hazard_vals = _parse_stage_float_csv(hazard_csv)
    success_vals = _parse_stage_float_csv(success_csv)
    eps_start_vals = _parse_stage_float_csv(epsilon_start_csv)
    eps_decay_vals = _parse_stage_float_csv(epsilon_decay_csv)

    hv = _stage_value(hazard_vals, stage_index_1based)
    if hv is not None and hasattr(agent, "learn_hazard_penalty"):
        try:
            setattr(agent, "learn_hazard_penalty", float(hv))
            applied["learn_hazard_penalty"] = float(hv)
        except Exception:
            pass
    sv = _stage_value(success_vals, stage_index_1based)
    if sv is not None and hasattr(agent, "learn_success_bonus"):
        try:
            setattr(agent, "learn_success_bonus", float(sv))
            applied["learn_success_bonus"] = float(sv)
        except Exception:
            pass
    ev = _stage_value(eps_start_vals, stage_index_1based)
    if ev is not None:
        if hasattr(agent, "stats") and hasattr(agent.stats, "epsilon"):
            try:
                agent.stats.epsilon = float(max(0.0, min(1.0, float(ev))))
                applied["stage_epsilon_start"] = float(agent.stats.epsilon)
            except Exception:
                pass
    dv = _stage_value(eps_decay_vals, stage_index_1based)
    if dv is not None and hasattr(agent, "epsilon_decay"):
        try:
            setattr(agent, "epsilon_decay", float(dv))
            applied["epsilon_decay"] = float(dv)
        except Exception:
            pass
    return applied


def _derive_certificate_from_events(
    *,
    runs_root: str,
    run_id: str,
    confidence: float,
) -> Optional[Dict[str, Any]]:
    events_path = os.path.join(str(runs_root), str(run_id), "events.jsonl")
    if not os.path.isfile(events_path):
        return None
    extracted = extract_episode_violation_flags_from_events(events_jsonl_path=events_path)
    cert = derive_safety_certificate(
        violation_flags=extracted.get("violation_flags", []),
        confidence=float(confidence),
    )
    return cert if isinstance(cert, dict) else None


def _iter_jsonl(path: str):
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


def _cliff_adjacent_count(*, x: int, y: int, width: int = 12, height: int = 4) -> int:
    cliff_y = int(height - 1)
    count = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = int(x + dx), int(y + dy)
        if nx < 0 or ny < 0 or nx >= int(width) or ny >= int(height):
            continue
        if ny == cliff_y and 1 <= nx <= (width - 2):
            count += 1
    return int(count)


def _augment_cliff_obs(obs: Any) -> Any:
    if not isinstance(obs, dict):
        return obs
    out = dict(obs)
    x = int(out.get("x", 0))
    y = int(out.get("y", 0))
    out.setdefault("cliff_adjacent", _cliff_adjacent_count(x=x, y=y))
    out.setdefault("wind_active", 0)
    out.setdefault("crumbled_count", 0)
    return out


def _prepare_augmented_cliff_dataset(*, source_path: str, out_path: str) -> str:
    src = str(source_path)
    if not src or not os.path.isfile(src):
        return src
    rows = 0
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    with open(str(out_path), "w", encoding="utf-8") as out:
        for row in _iter_jsonl(src):
            row2 = dict(row)
            row2["obs"] = _augment_cliff_obs(row2.get("obs"))
            if "next_obs" in row2:
                row2["next_obs"] = _augment_cliff_obs(row2.get("next_obs"))
            out.write(json.dumps(row2, ensure_ascii=False) + "\n")
            rows += 1
    if rows <= 0:
        return src
    return str(out_path)


def _stage_name(index: int, wind_probability: float, crumble_probability: float) -> str:
    return f"stage_{int(index)}_w{float(wind_probability):.2f}_c{float(crumble_probability):.2f}"


def _certificate_gate_metric(certificate: Optional[Dict[str, Any]], use_upper_bound: bool) -> Optional[float]:
    if not isinstance(certificate, dict):
        return None
    key = "upper_bound" if bool(use_upper_bound) else "observed_violation_rate"
    try:
        return float(certificate.get(key, None))  # type: ignore[arg-type]
    except Exception:
        return None


def _stage_gate_pass(metric: Optional[float], max_violation_rate: float) -> bool:
    if metric is None:
        return False
    return bool(float(metric) <= float(max_violation_rate))


def _should_block_next_stage(
    *,
    stage_gate_enabled: bool,
    required_stages: int,
    next_stage_index: int,
    checks: Sequence[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not bool(stage_gate_enabled):
        return False, "stage_gate_disabled"
    req = max(0, int(required_stages))
    nxt = int(next_stage_index)
    if req <= 0 or nxt <= req:
        return False, "not_applicable"

    required_checks = [row for row in checks if int(row.get("stage_index", 0)) <= req]
    if len(required_checks) < req:
        return True, "missing_required_stage_checks"
    for row in required_checks:
        if not bool(row.get("pass", False)):
            return True, f"required_stage_failed_{int(row.get('stage_index', 0))}"
    return False, "requirements_met"


def _apply_tutorial_param_overrides(
    *,
    tutorial_spec: VerseSpec,
    args: argparse.Namespace,
) -> Tuple[VerseSpec, Dict[str, Any]]:
    overrides: Dict[str, Any] = {}
    if getattr(args, "tutorial_target_margin", None) is not None:
        overrides["target_margin"] = int(getattr(args, "tutorial_target_margin"))
    if getattr(args, "tutorial_gust_probability", None) is not None:
        overrides["gust_probability"] = float(getattr(args, "tutorial_gust_probability"))
    if getattr(args, "tutorial_edge_penalty", None) is not None:
        overrides["edge_penalty"] = float(getattr(args, "tutorial_edge_penalty"))
    if getattr(args, "tutorial_margin_reward_scale", None) is not None:
        overrides["margin_reward_scale"] = float(getattr(args, "tutorial_margin_reward_scale"))
    if not overrides:
        return tutorial_spec, {}

    params = dict(tutorial_spec.params or {})
    params.update(overrides)
    metadata = dict(tutorial_spec.metadata or {})
    metadata["tutorial_param_overrides"] = dict(overrides)
    tags = list(tutorial_spec.tags or [])
    if "override:tutorial_params" not in tags:
        tags.append("override:tutorial_params")
    return tutorial_spec.evolved(params=params, metadata=metadata, tags=tags), overrides


def run_remediation(args: argparse.Namespace) -> Dict[str, Any]:
    trainer = Trainer(run_root=str(args.runs_root), schema_version="v1", auto_register_builtin=True)
    teacher_cfg = TeacherConfig.from_dict(
        {
            "enabled": True,
            "target_verse": str(args.target_verse),
            "tutorial_verse": str(args.tutorial_verse),
            "lookback_runs": int(args.teacher_lookback_runs),
            "min_episodes": int(args.teacher_min_episodes),
            "risk_obs_key": str(args.teacher_risk_obs_key),
            "high_risk_obs_threshold": float(args.teacher_risk_threshold),
            "high_risk_failure_rate_threshold": float(args.teacher_failure_rate_threshold),
            "tutorial_episodes": int(args.tutorial_episodes),
            "tutorial_max_steps": int(args.tutorial_max_steps),
            "graduation_episodes": int(args.graduation_episodes),
            "graduation_max_steps": int(args.graduation_max_steps),
            "lesson_log_path": str(args.lesson_log_path),
        }
    )

    plan = build_teacher_plan(
        runs_root=str(args.runs_root),
        target_verse=str(args.target_verse),
        cfg=teacher_cfg,
        seed=int(args.seed),
    )

    if plan.tutorial_spec is None and bool(args.force_tutorial):
        signals = collect_high_risk_signals(
            runs_root=str(args.runs_root),
            target_verse=str(args.target_verse),
            cfg=teacher_cfg,
        )
        tut_spec = synthesize_tutorial_spec(
            target_verse=str(args.target_verse),
            signals=signals,
            cfg=teacher_cfg,
            seed=int(args.seed),
        )
        plan = TeacherLessonPlan(
            target_verse=str(plan.target_verse),
            concept=str(plan.concept),
            signals=plan.signals,
            tutorial_spec=tut_spec,
            graduation_spec=plan.graduation_spec,
            reason="forced_tutorial_generation",
            lesson_quality=float(plan.lesson_quality),
            graduation_gate=dict(plan.graduation_gate or {}),
        )

    if plan.tutorial_spec is None:
        raise RuntimeError(
            "Teacher plan did not generate a tutorial. Try lower --teacher_risk_threshold or pass --force_tutorial."
        )
    tutorial_overrides_applied: Dict[str, Any] = {}
    updated_tut_spec, tutorial_overrides_applied = _apply_tutorial_param_overrides(
        tutorial_spec=plan.tutorial_spec,
        args=args,
    )
    if tutorial_overrides_applied:
        plan = TeacherLessonPlan(
            target_verse=str(plan.target_verse),
            concept=str(plan.concept),
            signals=plan.signals,
            tutorial_spec=updated_tut_spec,
            graduation_spec=plan.graduation_spec,
            reason=str(plan.reason),
            lesson_quality=float(plan.lesson_quality),
            graduation_gate=dict(plan.graduation_gate or {}),
        )

    gate = plan.graduation_gate if isinstance(plan.graduation_gate, dict) else {}
    gate_ready = bool(gate.get("ready", False))
    if not gate_ready and bool(args.force_graduation):
        gate_ready = True

    grad_run: Dict[str, Any] = {}
    cert: Optional[Dict[str, Any]] = None
    stage_runs: List[Dict[str, Any]] = []
    stage_eval_runs: List[Dict[str, Any]] = []
    stage_certificates: List[Dict[str, Any]] = []
    stage_gate_checks: List[Dict[str, Any]] = []
    algo_overrides = _build_algo_overrides(args)
    if (
        str(getattr(args, "algo", "")).strip().lower() == "q"
        and bool(getattr(args, "q_augment_cliff_dataset", False))
        and str(getattr(args, "target_verse", "")).strip().lower() == "cliff_world"
        and str(algo_overrides.get("dataset_path", "")).strip()
    ):
        source_ds = str(algo_overrides.get("dataset_path", "")).strip()
        out_ds = os.path.join("models", "expert_datasets", "cliff_world_teacher_augmented.jsonl")
        algo_overrides["dataset_path"] = _prepare_augmented_cliff_dataset(source_path=source_ds, out_path=out_ds)
    stage_gate_required = max(0, int(getattr(args, "stage_gate_required_stages", 2)))
    stage_gate_status: Dict[str, Any] = {
        "enabled": bool(getattr(args, "stage_gate_enabled", False)),
        "required_stages": int(stage_gate_required),
        "metric": "upper_bound" if bool(getattr(args, "stage_gate_use_upper_bound", False)) else "observed_violation_rate",
        "max_violation_rate": float(getattr(args, "stage_gate_max_violation_rate", 0.35)),
        "checks": stage_gate_checks,
        "blocked_before_stage": None,
        "blocked_reason": "",
        "passed_required_stages": None,
    }

    if bool(args.staged_graduation):
        register_builtin()
        register_builtin_agents()
        tutorial_verse = create_verse(plan.tutorial_spec)
        agent_spec = AgentSpec(
            spec_version="v1",
            policy_id=f"teacher_staged_{args.target_verse}",
            policy_version="0.1",
            algo=str(args.algo),
            seed=int(args.seed),
            config={**{"train": True}, **dict(algo_overrides)},
        )
        agent = create_agent(agent_spec, tutorial_verse.observation_space, tutorial_verse.action_space)
        try:
            Trainer._hydrate_agent_datasets(agent, agent_spec)
        except Exception:
            pass
        tutorial_verse.close()
        agent_ref = AgentRef.create(policy_id=agent_spec.policy_id, policy_version=agent_spec.policy_version)
        try:
            tut_run = _run_phase_with_existing_agent(
                run_root=str(args.runs_root),
                verse_spec=plan.tutorial_spec,
                agent=agent,
                agent_ref=agent_ref,
                episodes=int(args.tutorial_episodes),
                max_steps=int(args.tutorial_max_steps),
                seed=int(args.seed),
                train=True,
                safe_guard=False,
            )
            if gate_ready:
                schedule = _build_stage_schedule(args)
                stage_gate_required = min(int(stage_gate_required), max(0, len(schedule) - 1))
                stage_gate_status["required_stages"] = int(stage_gate_required)
                for i, (wind_p, crumble_p, eps_i) in enumerate(schedule, start=1):
                    blocked, blocked_reason = _should_block_next_stage(
                        stage_gate_enabled=bool(stage_gate_status.get("enabled", False)),
                        required_stages=int(stage_gate_required),
                        next_stage_index=int(i),
                        checks=stage_gate_checks,
                    )
                    if blocked:
                        stage_gate_status["blocked_before_stage"] = int(i)
                        stage_gate_status["blocked_reason"] = str(blocked_reason)
                        break
                    stage_label = _stage_name(int(i), float(wind_p), float(crumble_p))
                    stage_q_overrides_applied = _apply_stage_q_overrides(
                        agent=agent,
                        stage_index_1based=int(i),
                        hazard_csv=str(getattr(args, "stage_q_hazard_penalty_csv", "")),
                        success_csv=str(getattr(args, "stage_q_success_bonus_csv", "")),
                        epsilon_start_csv=str(getattr(args, "stage_q_epsilon_start_csv", "")),
                        epsilon_decay_csv=str(getattr(args, "stage_q_epsilon_decay_csv", "")),
                    )
                    stage_spec = plan.graduation_spec.evolved(
                        params={
                            **dict(plan.graduation_spec.params or {}),
                            "wind_probability": float(wind_p),
                            "crumble_probability": float(crumble_p),
                        },
                        tags=list(plan.graduation_spec.tags) + [f"stage:{i}", str(stage_label)],
                    )
                    stage_run = _run_phase_with_existing_agent(
                        run_root=str(args.runs_root),
                        verse_spec=stage_spec,
                        agent=agent,
                        agent_ref=agent_ref,
                        episodes=int(eps_i),
                        max_steps=int(args.graduation_max_steps),
                        seed=int(args.seed + i),
                        train=True,
                        safe_guard=bool(args.safe_guard_graduation),
                    )
                    stage_run["stage_index"] = int(i)
                    stage_run["stage_name"] = str(stage_label)
                    stage_run["wind_probability"] = float(wind_p)
                    stage_run["crumble_probability"] = float(crumble_p)
                    stage_run["episodes"] = int(eps_i)
                    stage_run["stage_q_overrides_applied"] = dict(stage_q_overrides_applied)
                    stage_runs.append(stage_run)
                    eval_eps = max(1, int(getattr(args, "stage_eval_episodes", 40)))
                    prev_eps = _set_agent_eval_epsilon(agent, float(getattr(args, "eval_epsilon", 0.0)))
                    eval_run = _run_phase_with_existing_agent(
                        run_root=str(args.runs_root),
                        verse_spec=stage_spec,
                        agent=agent,
                        agent_ref=agent_ref,
                        episodes=int(eval_eps),
                        max_steps=int(args.graduation_max_steps),
                        seed=int(args.seed + 1000 + i),
                        train=False,
                        safe_guard=bool(args.safe_guard_graduation),
                    )
                    if prev_eps is not None:
                        _set_agent_eval_epsilon(agent, float(prev_eps))
                    eval_run["stage_index"] = int(i)
                    eval_run["stage_name"] = str(stage_label)
                    eval_run["wind_probability"] = float(wind_p)
                    eval_run["crumble_probability"] = float(crumble_p)
                    eval_run["episodes"] = int(eval_eps)
                    eval_run["stage_q_overrides_applied"] = dict(stage_q_overrides_applied)
                    stage_eval_runs.append(eval_run)
                    stage_run["eval_run_id"] = str(eval_run.get("run_id", ""))
                    cert_stage = _derive_certificate_from_events(
                        runs_root=str(args.runs_root),
                        run_id=str(eval_run.get("run_id", "")),
                        confidence=float(args.safety_confidence),
                    )
                    if cert_stage is not None:
                        stage_certificates.append(
                            {
                                "stage_index": int(i),
                                "stage_name": str(stage_label),
                                "run_id": str(stage_run.get("run_id", "")),
                                "eval_run_id": str(eval_run.get("run_id", "")),
                                "wind_probability": float(wind_p),
                                "crumble_probability": float(crumble_p),
                                "certificate": cert_stage,
                            }
                        )
                    gate_metric = _certificate_gate_metric(
                        certificate=cert_stage,
                        use_upper_bound=bool(getattr(args, "stage_gate_use_upper_bound", False)),
                    )
                    gate_pass = _stage_gate_pass(
                        metric=gate_metric,
                        max_violation_rate=float(getattr(args, "stage_gate_max_violation_rate", 0.35)),
                    )
                    check_row = {
                        "stage_index": int(i),
                        "stage_name": str(stage_label),
                        "metric_name": str(stage_gate_status.get("metric", "observed_violation_rate")),
                        "metric_value": None if gate_metric is None else float(gate_metric),
                        "max_violation_rate": float(getattr(args, "stage_gate_max_violation_rate", 0.35)),
                        "pass": bool(gate_pass),
                    }
                    stage_gate_checks.append(check_row)
                    stage_run["stage_gate_check"] = dict(check_row)
                if stage_runs:
                    grad_run = dict(stage_runs[-1])
                    if stage_eval_runs:
                        grad_eval = dict(stage_eval_runs[-1])
                        grad_run["eval_run_id"] = str(grad_eval.get("run_id", ""))
                        cert = _derive_certificate_from_events(
                            runs_root=str(args.runs_root),
                            run_id=str(grad_eval.get("run_id", "")),
                            confidence=float(args.safety_confidence),
                        )
                    final_holdout_eps = int(getattr(args, "final_holdout_episodes", 0))
                    if final_holdout_eps > 0:
                        hardest = stage_runs[-1]
                        final_spec = plan.graduation_spec.evolved(
                            params={
                                **dict(plan.graduation_spec.params or {}),
                                "wind_probability": float(hardest.get("wind_probability", 0.10)),
                                "crumble_probability": float(hardest.get("crumble_probability", 0.03)),
                            },
                            tags=list(plan.graduation_spec.tags) + ["final_holdout"],
                        )
                        prev_eps = _set_agent_eval_epsilon(agent, float(getattr(args, "eval_epsilon", 0.0)))
                        final_eval_run = _run_phase_with_existing_agent(
                            run_root=str(args.runs_root),
                            verse_spec=final_spec,
                            agent=agent,
                            agent_ref=agent_ref,
                            episodes=int(final_holdout_eps),
                            max_steps=int(args.graduation_max_steps),
                            seed=int(args.seed + 5000),
                            train=False,
                            safe_guard=bool(args.safe_guard_graduation),
                        )
                        if prev_eps is not None:
                            _set_agent_eval_epsilon(agent, float(prev_eps))
                        final_eval_run["holdout"] = True
                        stage_eval_runs.append(final_eval_run)
                        grad_run["final_holdout_eval_run_id"] = str(final_eval_run.get("run_id", ""))
                        final_cert = _derive_certificate_from_events(
                            runs_root=str(args.runs_root),
                            run_id=str(final_eval_run.get("run_id", "")),
                            confidence=float(args.safety_confidence),
                        )
                        if final_cert is not None:
                            cert = dict(final_cert)
                if bool(stage_gate_status.get("enabled", False)) and int(stage_gate_required) > 0:
                    req_rows = [r for r in stage_gate_checks if int(r.get("stage_index", 0)) <= int(stage_gate_required)]
                    if len(req_rows) < int(stage_gate_required):
                        stage_gate_status["passed_required_stages"] = False
                    else:
                        stage_gate_status["passed_required_stages"] = bool(all(bool(r.get("pass", False)) for r in req_rows))
                elif not bool(stage_gate_status.get("enabled", False)):
                    stage_gate_status["passed_required_stages"] = None
                else:
                    stage_gate_status["passed_required_stages"] = True
        finally:
            try:
                agent.close()
            except Exception:
                pass
    else:
        tut_run = _run_training(
            trainer=trainer,
            verse_spec=plan.tutorial_spec,
            algo=str(args.algo),
            policy_id=f"teacher_tutorial_{args.target_verse}_{args.tutorial_verse}",
            episodes=int(args.tutorial_episodes),
            max_steps=int(args.tutorial_max_steps),
            seed=int(args.seed),
            train=True,
            safe_guard=False,
            agent_config_overrides=algo_overrides,
        )
        if gate_ready:
            grad_run = _run_training(
                trainer=trainer,
                verse_spec=plan.graduation_spec,
                algo=str(args.algo),
                policy_id=f"teacher_graduation_{args.target_verse}",
                episodes=int(args.graduation_episodes),
                max_steps=int(args.graduation_max_steps),
                seed=int(args.seed + 1),
                train=True,
                safe_guard=bool(args.safe_guard_graduation),
                agent_config_overrides=algo_overrides,
            )
            grad_run_id = str(grad_run.get("run_id", ""))
            cert = _derive_certificate_from_events(
                runs_root=str(args.runs_root),
                run_id=grad_run_id,
                confidence=float(args.safety_confidence),
            )

    append_teacher_lesson(
        path=str(teacher_cfg.lesson_log_path),
        plan=plan,
        tutorial_run_id=str(tut_run.get("run_id", "")),
        graduation_run_id=str(grad_run.get("run_id", "")),
    )

    return {
        "created_at_iso": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": {
            "target_verse": str(args.target_verse),
            "tutorial_verse": str(args.tutorial_verse),
            "algo": str(args.algo),
            "seed": int(args.seed),
            "tutorial_episodes": int(args.tutorial_episodes),
            "graduation_episodes": int(args.graduation_episodes),
            "safe_guard_graduation": bool(args.safe_guard_graduation),
            "staged_graduation": bool(args.staged_graduation),
            "stage_wind_csv": str(args.stage_wind_csv),
            "stage_crumble_csv": str(args.stage_crumble_csv),
            "stage_gate_enabled": bool(getattr(args, "stage_gate_enabled", False)),
            "stage_gate_required_stages": int(getattr(args, "stage_gate_required_stages", 2)),
            "stage_gate_max_violation_rate": float(getattr(args, "stage_gate_max_violation_rate", 0.35)),
            "stage_gate_use_upper_bound": bool(getattr(args, "stage_gate_use_upper_bound", False)),
            "stage_eval_episodes": int(getattr(args, "stage_eval_episodes", 40)),
            "eval_epsilon": float(getattr(args, "eval_epsilon", 0.0)),
            "final_holdout_episodes": int(getattr(args, "final_holdout_episodes", 0)),
            "tutorial_target_margin": getattr(args, "tutorial_target_margin", None),
            "tutorial_gust_probability": getattr(args, "tutorial_gust_probability", None),
            "tutorial_edge_penalty": getattr(args, "tutorial_edge_penalty", None),
            "tutorial_margin_reward_scale": getattr(args, "tutorial_margin_reward_scale", None),
            "stage_q_hazard_penalty_csv": str(getattr(args, "stage_q_hazard_penalty_csv", "")),
            "stage_q_success_bonus_csv": str(getattr(args, "stage_q_success_bonus_csv", "")),
            "stage_q_epsilon_start_csv": str(getattr(args, "stage_q_epsilon_start_csv", "")),
            "stage_q_epsilon_decay_csv": str(getattr(args, "stage_q_epsilon_decay_csv", "")),
            "algo_overrides": dict(algo_overrides),
        },
        "plan": plan.to_dict(),
        "tutorial_overrides_applied": dict(tutorial_overrides_applied),
        "tutorial_run": tut_run,
        "tutorial_run_id": str(tut_run.get("run_id", "")),
        "graduation_run": grad_run,
        "graduation_run_id": str(grad_run.get("run_id", "")),
        "graduation_stage_runs": stage_runs,
        "graduation_stage_eval_runs": stage_eval_runs,
        "graduation_stage_certificates": stage_certificates,
        "stage_gate_status": stage_gate_status,
        "graduation_safety_certificate": cert,
        "lesson_log_path": str(teacher_cfg.lesson_log_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Teacher Frontier remediation session.")
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--target_verse", type=str, default="cliff_world")
    ap.add_argument("--tutorial_verse", type=str, default="wind_master_world")
    ap.add_argument("--algo", type=str, default="q")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--teacher_lookback_runs", type=int, default=16)
    ap.add_argument("--teacher_min_episodes", type=int, default=4)
    ap.add_argument("--teacher_risk_obs_key", type=str, default="cliff_adjacent")
    ap.add_argument("--teacher_risk_threshold", type=float, default=1.0)
    ap.add_argument("--teacher_failure_rate_threshold", type=float, default=0.20)

    ap.add_argument("--tutorial_episodes", type=int, default=80)
    ap.add_argument("--tutorial_max_steps", type=int, default=80)
    ap.add_argument("--graduation_episodes", type=int, default=120)
    ap.add_argument("--graduation_max_steps", type=int, default=100)
    ap.add_argument("--stage_eval_episodes", type=int, default=40)
    ap.add_argument("--eval_epsilon", type=float, default=0.0)
    ap.add_argument("--final_holdout_episodes", type=int, default=0)
    ap.add_argument("--staged_graduation", action="store_true")
    ap.add_argument("--stage_wind_csv", type=str, default="0.00,0.05,0.10")
    ap.add_argument("--stage_crumble_csv", type=str, default="0.00,0.01,0.03")
    ap.add_argument("--stage_gate_enabled", action="store_true", default=True)
    ap.add_argument("--no_stage_gate", action="store_false", dest="stage_gate_enabled")
    ap.add_argument("--stage_gate_required_stages", type=int, default=2)
    ap.add_argument("--stage_gate_max_violation_rate", type=float, default=0.35)
    ap.add_argument("--stage_gate_use_upper_bound", action="store_true")

    ap.add_argument("--tutorial_target_margin", type=int, default=None)
    ap.add_argument("--tutorial_gust_probability", type=float, default=None)
    ap.add_argument("--tutorial_edge_penalty", type=float, default=None)
    ap.add_argument("--tutorial_margin_reward_scale", type=float, default=None)
    ap.add_argument("--stage_q_hazard_penalty_csv", type=str, default="")
    ap.add_argument("--stage_q_success_bonus_csv", type=str, default="")
    ap.add_argument("--stage_q_epsilon_start_csv", type=str, default="")
    ap.add_argument("--stage_q_epsilon_decay_csv", type=str, default="")

    ap.add_argument("--q_lr", type=float, default=None)
    ap.add_argument("--q_gamma", type=float, default=None)
    ap.add_argument("--q_epsilon_start", type=float, default=None)
    ap.add_argument("--q_epsilon_min", type=float, default=None)
    ap.add_argument("--q_epsilon_decay", type=float, default=None)
    ap.add_argument("--q_learn_hazard_penalty", type=float, default=None)
    ap.add_argument("--q_learn_success_bonus", type=float, default=None)
    ap.add_argument("--q_warmstart_reward_scale", type=float, default=None)
    ap.add_argument("--q_dataset_path", type=str, default="")
    ap.add_argument("--q_augment_cliff_dataset", action="store_true", default=True)
    ap.add_argument("--no_q_augment_cliff_dataset", action="store_false", dest="q_augment_cliff_dataset")

    ap.add_argument("--safe_guard_graduation", action="store_true")
    ap.add_argument("--force_tutorial", action="store_true")
    ap.add_argument("--force_graduation", action="store_true")
    ap.add_argument("--safety_confidence", type=float, default=0.95)
    ap.add_argument("--lesson_log_path", type=str, default=os.path.join("models", "teacher_lessons.json"))
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("models", "validation", "teacher_wind_remediation.json"),
    )
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    report = run_remediation(args)
    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Teacher remediation complete")
    print(f"- target={args.target_verse} tutorial={args.tutorial_verse} algo={args.algo}")
    print(f"- tutorial_run_id={str((report.get('tutorial_run') or {}).get('run_id', ''))}")
    print(f"- graduation_run_id={str((report.get('graduation_run') or {}).get('run_id', ''))}")
    cert = report.get("graduation_safety_certificate") or {}
    if isinstance(cert, dict) and cert:
        print(
            f"- graduation safety: violations={cert.get('observed_violations')}/{cert.get('episodes')} "
            f"upper@{float(cert.get('confidence', 0.95)):.0%}={float(cert.get('upper_bound', 1.0)):.4f}"
        )
    print(f"report: {args.out_json}")


if __name__ == "__main__":
    main()
