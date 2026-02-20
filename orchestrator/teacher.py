"""
orchestrator/teacher.py

"Teacher" frontier support:
- detect abstract weakness signals (for now: high-risk failure pattern),
- procedurally synthesize a simpler tutorial VerseSpec,
- emit a lesson plan that can be run before graduation back to the target Verse.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from core.types import VerseSpec


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


def _norm(x: Any) -> str:
    return str(x).strip().lower()


def _risk_proxy_from_obs(*, obs: Dict[str, Any], target_verse: str, cfg: "TeacherConfig") -> Optional[float]:
    if str(cfg.risk_obs_key) in obs:
        return _safe_float(obs.get(cfg.risk_obs_key), None)  # type: ignore[arg-type]
    vv = _norm(target_verse)
    if vv == "cliff_world":
        # cliff_adjacent is typically in [0,4]; scale to the default teacher range.
        if "cliff_adjacent" in obs:
            return float(_safe_float(obs.get("cliff_adjacent"), 0.0) * 3.0)
        if "y" in obs:
            y = _safe_float(obs.get("y"), 0.0)
            return float(max(0.0, y) * 2.0)
    if "safety_margin" in obs:
        # Lower margin should imply higher risk.
        margin = _safe_float(obs.get("safety_margin"), 0.0)
        return float(max(0.0, 6.0 - margin))
    return None


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


def _iter_recent_run_dirs(runs_root: str, limit: int) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    rows: List[tuple[float, str]] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        events_path = os.path.join(run_dir, "events.jsonl")
        if not os.path.isdir(run_dir) or not os.path.isfile(events_path):
            continue
        rows.append((_safe_float(os.path.getmtime(events_path), 0.0), run_dir))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows[: max(1, int(limit))]]


@dataclass
class TeacherConfig:
    enabled: bool = True
    target_verse: str = "chess_world"
    tutorial_verse: str = "risk_tutorial_world"
    lookback_runs: int = 16
    min_episodes: int = 4
    risk_obs_key: str = "risk"
    high_risk_obs_threshold: float = 5.0
    high_risk_failure_rate_threshold: float = 0.30
    tutorial_episodes: int = 20
    tutorial_max_steps: int = 50
    graduation_episodes: int = 20
    graduation_max_steps: int = 80
    lesson_log_path: str = os.path.join("models", "teacher_lessons.json")
    min_lesson_quality: float = 0.35
    graduation_confidence_threshold: float = 0.65
    mastery_window: int = 3
    mastery_stability_tolerance: float = 0.10

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "TeacherConfig":
        c = cfg if isinstance(cfg, dict) else {}
        return TeacherConfig(
            enabled=bool(c.get("enabled", True)),
            target_verse=str(c.get("target_verse", "chess_world")),
            tutorial_verse=str(c.get("tutorial_verse", "risk_tutorial_world")),
            lookback_runs=max(1, int(c.get("lookback_runs", 16))),
            min_episodes=max(1, int(c.get("min_episodes", 4))),
            risk_obs_key=str(c.get("risk_obs_key", "risk")),
            high_risk_obs_threshold=max(0.0, _safe_float(c.get("high_risk_obs_threshold", 5.0), 5.0)),
            high_risk_failure_rate_threshold=max(
                0.01, min(0.99, _safe_float(c.get("high_risk_failure_rate_threshold", 0.30), 0.30))
            ),
            tutorial_episodes=max(1, int(c.get("tutorial_episodes", 20))),
            tutorial_max_steps=max(8, int(c.get("tutorial_max_steps", 50))),
            graduation_episodes=max(1, int(c.get("graduation_episodes", 20))),
            graduation_max_steps=max(8, int(c.get("graduation_max_steps", 80))),
            lesson_log_path=str(c.get("lesson_log_path", os.path.join("models", "teacher_lessons.json"))),
            min_lesson_quality=max(0.0, min(1.0, _safe_float(c.get("min_lesson_quality", 0.35), 0.35))),
            graduation_confidence_threshold=max(
                0.0,
                min(1.0, _safe_float(c.get("graduation_confidence_threshold", 0.65), 0.65)),
            ),
            mastery_window=max(1, int(c.get("mastery_window", 3))),
            mastery_stability_tolerance=max(
                0.0,
                min(1.0, _safe_float(c.get("mastery_stability_tolerance", 0.10), 0.10)),
            ),
        )


@dataclass
class HighRiskSignals:
    verse_name: str
    episodes: int
    events: int
    high_risk_events: int
    high_risk_failures: int
    high_risk_failure_rate: float
    mean_episode_return: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_name": str(self.verse_name),
            "episodes": int(self.episodes),
            "events": int(self.events),
            "high_risk_events": int(self.high_risk_events),
            "high_risk_failures": int(self.high_risk_failures),
            "high_risk_failure_rate": float(self.high_risk_failure_rate),
            "mean_episode_return": float(self.mean_episode_return),
        }


@dataclass
class TeacherLessonPlan:
    target_verse: str
    concept: str
    signals: HighRiskSignals
    tutorial_spec: Optional[VerseSpec]
    graduation_spec: VerseSpec
    reason: str
    lesson_quality: float = 0.0
    graduation_gate: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_verse": str(self.target_verse),
            "concept": str(self.concept),
            "signals": self.signals.to_dict(),
            "tutorial_spec": None if self.tutorial_spec is None else self.tutorial_spec.to_dict(),
            "graduation_spec": self.graduation_spec.to_dict(),
            "reason": str(self.reason),
            "lesson_quality": float(self.lesson_quality),
            "graduation_gate": dict(self.graduation_gate or {}),
        }


def _clip01(x: Any) -> float:
    return max(0.0, min(1.0, _safe_float(x, 0.0)))


def _mastery_score(signals: HighRiskSignals) -> float:
    return _clip01(1.0 - float(signals.high_risk_failure_rate))


def _lesson_quality(signals: HighRiskSignals, cfg: TeacherConfig) -> float:
    episode_support = _clip01(float(signals.episodes) / float(max(1, cfg.min_episodes)))
    high_risk_support = _clip01(float(signals.high_risk_events) / float(max(1, cfg.min_episodes * 2)))
    challenge_strength = _clip01(float(signals.high_risk_failure_rate))
    quality = (0.45 * episode_support) + (0.35 * high_risk_support) + (0.20 * challenge_strength)
    return _clip01(quality)


def _mastery_from_lesson_row(row: Dict[str, Any]) -> Optional[float]:
    plan = row.get("plan")
    if not isinstance(plan, dict):
        return None
    signals = plan.get("signals")
    if not isinstance(signals, dict):
        return None
    failure_rate = _safe_float(signals.get("high_risk_failure_rate"), None)  # type: ignore[arg-type]
    if failure_rate is None:
        return None
    return _clip01(1.0 - float(failure_rate))


def _load_mastery_history(*, path: str, target_verse: str, concept: str) -> List[float]:
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    rows = payload.get("lessons") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    tv = _norm(target_verse)
    cp = _norm(concept)
    hist: List[tuple[int, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        plan = row.get("plan")
        if not isinstance(plan, dict):
            continue
        if _norm(plan.get("target_verse", "")) != tv:
            continue
        if _norm(plan.get("concept", "")) != cp:
            continue
        mastery = _mastery_from_lesson_row(row)
        if mastery is None:
            continue
        hist.append((_safe_int(row.get("t_ms", 0), 0), float(mastery)))
    hist.sort(key=lambda x: x[0])
    return [float(x[1]) for x in hist]


def _window_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(x) for x in values) / float(len(values)))


def _build_graduation_gate(
    *,
    cfg: TeacherConfig,
    target_verse: str,
    concept: str,
    signals: HighRiskSignals,
    lesson_quality: float,
) -> Dict[str, Any]:
    w = max(1, int(cfg.mastery_window))
    history = _load_mastery_history(path=str(cfg.lesson_log_path), target_verse=target_verse, concept=concept)
    current_mastery = _mastery_score(signals)
    series = list(history) + [float(current_mastery)]
    recent = series[-w:]
    prev = series[-2 * w : -w]
    recent_mean = _window_mean(recent)
    prev_mean = _window_mean(prev)
    mastery_delta = float(recent_mean - prev_mean) if prev else 0.0
    stable_windows = bool(
        len(recent) == w
        and len(prev) == w
        and abs(float(recent_mean - prev_mean)) <= float(cfg.mastery_stability_tolerance)
    )
    confidence = _clip01((0.50 * current_mastery) + (0.30 * recent_mean) + (0.20 * lesson_quality))
    ready = bool(
        stable_windows
        and confidence >= float(cfg.graduation_confidence_threshold)
        and lesson_quality >= float(cfg.min_lesson_quality)
        and mastery_delta >= -0.02
    )
    reason = "ready"
    if not stable_windows:
        reason = "unstable_mastery_windows"
    elif confidence < float(cfg.graduation_confidence_threshold):
        reason = "low_graduation_confidence"
    elif lesson_quality < float(cfg.min_lesson_quality):
        reason = "low_lesson_quality"
    elif mastery_delta < -0.02:
        reason = "mastery_regressed"
    return {
        "ready": bool(ready),
        "reason": str(reason),
        "confidence": float(confidence),
        "lesson_quality": float(lesson_quality),
        "mastery_current": float(current_mastery),
        "mastery_recent_mean": float(recent_mean),
        "mastery_prev_mean": float(prev_mean),
        "mastery_delta": float(mastery_delta),
        "stable_windows": bool(stable_windows),
        "history_points": int(len(history)),
        "window_size": int(w),
    }


def collect_high_risk_signals(*, runs_root: str, target_verse: str, cfg: TeacherConfig) -> HighRiskSignals:
    events = 0
    high_risk_events = 0
    high_risk_failures = 0
    episode_returns: Dict[str, float] = {}
    episode_ids: set[str] = set()

    for run_dir in _iter_recent_run_dirs(runs_root, cfg.lookback_runs):
        events_path = os.path.join(run_dir, "events.jsonl")
        if not os.path.isfile(events_path):
            continue
        for row in _iter_jsonl(events_path):
            if _norm(row.get("verse_name", "")) != _norm(target_verse):
                continue
            events += 1
            ep = str(row.get("episode_id", "")).strip() or f"unknown_{events}"
            episode_ids.add(ep)
            episode_returns[ep] = float(episode_returns.get(ep, 0.0)) + _safe_float(row.get("reward", 0.0), 0.0)

            info = row.get("info")
            info = info if isinstance(info, dict) else {}
            obs = row.get("obs")
            risk_val = None
            if isinstance(obs, dict):
                risk_val = _risk_proxy_from_obs(obs=obs, target_verse=target_verse, cfg=cfg)

            explicit_hazard = bool(
                info.get("fell_cliff", False)
                or info.get("high_risk_failure", False)
                or info.get("unsafe_finish", False)
            )
            if risk_val is None and explicit_hazard:
                risk_val = float(cfg.high_risk_obs_threshold)
            if risk_val is None or float(risk_val) < float(cfg.high_risk_obs_threshold):
                continue

            high_risk_events += 1
            reward = _safe_float(row.get("reward", 0.0), 0.0)
            terminal = bool(row.get("done", False) or row.get("truncated", False))
            explicit_failure = bool(info.get("lost_game", False) or info.get("high_risk_failure", False))
            if explicit_failure or reward < -0.05 or (terminal and bool(info.get("reached_goal", False)) is False):
                high_risk_failures += 1

    mean_return = 0.0
    if episode_returns:
        mean_return = float(sum(episode_returns.values()) / float(max(1, len(episode_returns))))
    failure_rate = float(high_risk_failures / float(max(1, high_risk_events)))
    return HighRiskSignals(
        verse_name=str(target_verse),
        episodes=int(len(episode_ids)),
        events=int(events),
        high_risk_events=int(high_risk_events),
        high_risk_failures=int(high_risk_failures),
        high_risk_failure_rate=float(failure_rate),
        mean_episode_return=float(mean_return),
    )


def should_generate_tutorial(
    signals: HighRiskSignals,
    cfg: TeacherConfig,
    *,
    lesson_quality: Optional[float] = None,
) -> bool:
    if signals.episodes < int(cfg.min_episodes):
        return False
    if signals.high_risk_events <= 0:
        return False
    if lesson_quality is not None and float(lesson_quality) < float(cfg.min_lesson_quality):
        return False
    return bool(signals.high_risk_failure_rate >= float(cfg.high_risk_failure_rate_threshold))


def synthesize_tutorial_spec(*, target_verse: str, signals: HighRiskSignals, cfg: TeacherConfig, seed: int) -> VerseSpec:
    # Harder observed failure pattern -> stronger teaching pressure but simpler conversion condition.
    pressure = max(0.0, min(1.0, float(signals.high_risk_failure_rate)))
    tut = _norm(cfg.tutorial_verse)
    if tut == "wind_master_world":
        # Stronger tutorial shaping: reward durable safety margin, not edge-only avoidance.
        target_margin = max(2, min(4, int(round(2 + (1.5 * pressure)))))
        gust_probability = round(0.10 + (0.25 * pressure), 3)
        edge_penalty = round(-2.00 - (3.00 * pressure), 3)
        margin_reward_scale = round(0.08 + (0.10 * pressure), 3)
        return VerseSpec(
            spec_version="v1",
            verse_name=str(cfg.tutorial_verse),
            verse_version="0.1",
            seed=int(seed),
            tags=["teacher", "tutorial", "concept:safety_margin_navigation", f"for:{_norm(target_verse)}"],
            params={
                "max_steps": int(cfg.tutorial_max_steps),
                "target_margin": int(target_margin),
                "gust_probability": float(gust_probability),
                "edge_penalty": float(edge_penalty),
                "margin_reward_scale": float(margin_reward_scale),
            },
            metadata={
                "generated_by": "teacher",
                "generated_at_ms": int(time.time() * 1000),
                "target_verse": str(target_verse),
                "signals": signals.to_dict(),
            },
        )

    risk_floor_start = max(4, min(11, int(round(6 + 4 * pressure))))
    all_in_threshold = max(1, min(5, int(round(4 - 2 * pressure))))
    target_control = max(6, min(10, int(round(7 + pressure))))
    return VerseSpec(
        spec_version="v1",
        verse_name=str(cfg.tutorial_verse),
        verse_version="0.1",
        seed=int(seed),
        tags=["teacher", "tutorial", "concept:risk_management", f"for:{_norm(target_verse)}"],
        params={
            "max_steps": int(cfg.tutorial_max_steps),
            "risk_floor_start": int(risk_floor_start),
            "all_in_threshold": int(all_in_threshold),
            "target_control": int(target_control),
            "ambient_risk_noise": round(0.10 + 0.25 * pressure, 3),
        },
        metadata={
            "generated_by": "teacher",
            "generated_at_ms": int(time.time() * 1000),
            "target_verse": str(target_verse),
            "signals": signals.to_dict(),
        },
    )


def build_teacher_plan(*, runs_root: str, target_verse: str, cfg: TeacherConfig, seed: int) -> TeacherLessonPlan:
    signals = collect_high_risk_signals(runs_root=runs_root, target_verse=target_verse, cfg=cfg)
    tutorial_spec = None
    reason = "insufficient_signal"
    lesson_quality = _lesson_quality(signals, cfg)
    graduation_gate = _build_graduation_gate(
        cfg=cfg,
        target_verse=str(target_verse),
        concept="risk_management",
        signals=signals,
        lesson_quality=float(lesson_quality),
    )
    if bool(cfg.enabled) and should_generate_tutorial(signals, cfg, lesson_quality=lesson_quality):
        tutorial_spec = synthesize_tutorial_spec(
            target_verse=target_verse,
            signals=signals,
            cfg=cfg,
            seed=int(seed),
        )
        reason = "high_risk_failure_pattern_detected"
    elif signals.episodes >= int(cfg.min_episodes) and signals.high_risk_events > 0:
        reason = "low_lesson_quality"

    graduation_spec = VerseSpec(
        spec_version="v1",
        verse_name=str(target_verse),
        verse_version="0.1",
        seed=int(seed),
        tags=["teacher", "graduation", "concept:risk_management"],
        params={"max_steps": int(cfg.graduation_max_steps)},
        metadata={
            "generated_by": "teacher",
            "generated_at_ms": int(time.time() * 1000),
            "target_verse": str(target_verse),
            "concept": "risk_management",
        },
    )
    return TeacherLessonPlan(
        target_verse=str(target_verse),
        concept="risk_management",
        signals=signals,
        tutorial_spec=tutorial_spec,
        graduation_spec=graduation_spec,
        reason=str(reason),
        lesson_quality=float(lesson_quality),
        graduation_gate=dict(graduation_gate),
    )


def append_teacher_lesson(*, path: str, plan: TeacherLessonPlan, tutorial_run_id: str, graduation_run_id: str) -> str:
    payload: Dict[str, Any]
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            payload = obj if isinstance(obj, dict) else {}
        except Exception:
            payload = {}
    else:
        payload = {}

    rows = payload.get("lessons")
    if not isinstance(rows, list):
        rows = []

    rows.append(
        {
            "t_ms": int(time.time() * 1000),
            "tutorial_run_id": str(tutorial_run_id),
            "graduation_run_id": str(graduation_run_id),
            "plan": plan.to_dict(),
        }
    )
    payload["version"] = "v1"
    payload["updated_at_ms"] = int(time.time() * 1000)
    payload["lessons"] = rows

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
