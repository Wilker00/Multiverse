"""
tools/measure_recall_lift.py

Estimate causal recall lift from randomized recall ablation logs in events.jsonl.

This expects rollouts generated with:
  RolloutConfig(on_demand_recall_ablation_prob > 0)

It analyzes the first randomized eligible recall trigger per episode and compares:
  - treatment: recall applied
  - control: recall disabled for ablation

Outcomes:
  - eventual episode success
  - hazard event within horizon H after trigger
  - any hazard after trigger (episode tail)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


HAZARD_KEYS = {
    "hit_wall",
    "bumped_wall",
    "hit_obstacle",
    "hit_patrol",
    "battery_death",
    "battery_depleted",
    "fell_cliff",
    "fell_pit",
    "hit_laser",
    "hit_hazard",
    "collision",
    "crash",
}


@dataclass
class TriggerRecord:
    episode_id: str
    step_idx: int
    treatment: bool
    recall_eligible_by_agent: bool
    recall_gate_passed: bool
    recall_used: bool
    recall_greedy_changed: bool
    recall_match_count: int
    eventual_success: bool
    hazard_within_h: bool
    hazard_after_trigger: bool


def _load_events(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"events.jsonl not found in {run_dir}")
    out: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                ev = json.loads(s)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            ep = str(ev.get("episode_id", ""))
            out.setdefault(ep, []).append(ev)
    for ep in out:
        out[ep].sort(key=lambda e: _safe_int(e.get("step_idx", 0), 0))
    return out


def _is_hazard_event(ev: Dict[str, Any]) -> bool:
    info = ev.get("info")
    info = info if isinstance(info, dict) else {}
    return any(bool(info.get(k, False)) for k in HAZARD_KEYS)


def _episode_success(evts: List[Dict[str, Any]]) -> bool:
    for ev in evts:
        info = ev.get("info")
        info = info if isinstance(info, dict) else {}
        if bool(info.get("reached_goal", False)):
            return True
    return False


def _first_randomized_trigger(
    evts: List[Dict[str, Any]], *, horizon: int, require_gate_passed: bool = False
) -> Optional[TriggerRecord]:
    ep_success = _episode_success(evts)
    fallback: Optional[TriggerRecord] = None
    for i, ev in enumerate(evts):
        info = ev.get("info")
        info = info if isinstance(info, dict) else {}
        ab = info.get("memory_recall_ablation")
        ab = ab if isinstance(ab, dict) else {}
        if not bool(ab.get("eligible", False)):
            continue
        if not bool(ab.get("randomized", False)):
            continue

        action_info = info.get("action_info")
        action_info = action_info if isinstance(action_info, dict) else {}
        step_idx = _safe_int(ev.get("step_idx", 0), 0)
        h_end = step_idx + max(1, int(horizon))
        hazard_within_h = False
        hazard_after_trigger = False
        for fut in evts[i:]:
            fs = _safe_int(fut.get("step_idx", 0), 0)
            hz = _is_hazard_event(fut)
            if hz and fs <= h_end:
                hazard_within_h = True
            if hz:
                hazard_after_trigger = True
        tr = TriggerRecord(
            episode_id=str(ev.get("episode_id", "")),
            step_idx=int(step_idx),
            treatment=not bool(ab.get("disabled_apply", False)),
            recall_eligible_by_agent=bool(action_info.get("memory_recall_eligible", False)),
            recall_gate_passed=bool(action_info.get("memory_recall_gate_passed", False)),
            recall_used=bool(action_info.get("memory_recall_used", False)),
            recall_greedy_changed=bool(action_info.get("memory_recall_greedy_changed", False)),
            recall_match_count=int(_safe_int(action_info.get("memory_recall_match_count", 0), 0)),
            eventual_success=bool(ep_success),
            hazard_within_h=bool(hazard_within_h),
            hazard_after_trigger=bool(hazard_after_trigger),
        )
        if bool(require_gate_passed) and not bool(tr.recall_gate_passed):
            if fallback is None:
                fallback = tr
            continue
        # Prefer the first trigger where recall produced a usable prior.
        if tr.recall_eligible_by_agent or tr.recall_match_count > 0 or tr.recall_greedy_changed:
            return tr
        if fallback is None:
            fallback = tr
    if bool(require_gate_passed):
        return None
    return fallback


def _mean_bool(rows: List[TriggerRecord], field: str) -> float:
    if not rows:
        return 0.0
    return float(sum(1.0 if bool(getattr(r, field)) else 0.0 for r in rows) / float(len(rows)))


def _bootstrap_diff(
    treat: List[TriggerRecord],
    ctrl: List[TriggerRecord],
    *,
    field: str,
    n_boot: int,
    seed: int,
    invert: bool = False,
) -> Dict[str, Any]:
    # invert=True returns control - treatment (useful for hazard reduction)
    rng = random.Random(int(seed))
    if not treat or not ctrl:
        return {"estimate": None, "ci95": None}
    vals: List[float] = []
    for _ in range(max(100, int(n_boot))):
        ts = [treat[rng.randrange(len(treat))] for _ in range(len(treat))]
        cs = [ctrl[rng.randrange(len(ctrl))] for _ in range(len(ctrl))]
        mt = _mean_bool(ts, field)
        mc = _mean_bool(cs, field)
        vals.append(float((mc - mt) if invert else (mt - mc)))
    vals.sort()
    lo_idx = max(0, int(0.025 * (len(vals) - 1)))
    hi_idx = min(len(vals) - 1, int(0.975 * (len(vals) - 1)))
    est_mt = _mean_bool(treat, field)
    est_mc = _mean_bool(ctrl, field)
    est = float((est_mc - est_mt) if invert else (est_mt - est_mc))
    return {
        "estimate": float(est),
        "ci95": [float(vals[lo_idx]), float(vals[hi_idx])],
    }


def analyze(*, run_dir: str, horizon: int, n_boot: int, seed: int, require_gate_passed: bool = False) -> Dict[str, Any]:
    ep_map = _load_events(run_dir)
    triggers: List[TriggerRecord] = []
    for evts in ep_map.values():
        tr = _first_randomized_trigger(evts, horizon=horizon, require_gate_passed=require_gate_passed)
        if tr is not None:
            triggers.append(tr)

    treat = [r for r in triggers if r.treatment]
    ctrl = [r for r in triggers if not r.treatment]

    result: Dict[str, Any] = {
        "run_dir": run_dir,
        "episodes_total": int(len(ep_map)),
        "episodes_with_randomized_trigger": int(len(triggers)),
        "treatment_n": int(len(treat)),
        "control_n": int(len(ctrl)),
        "trigger_quality": {
            "agent_recall_eligible_rate": _mean_bool(triggers, "recall_eligible_by_agent"),
            "gate_passed_rate": _mean_bool(triggers, "recall_gate_passed"),
            "recall_used_rate": _mean_bool(triggers, "recall_used"),
            "greedy_action_changed_rate": _mean_bool(triggers, "recall_greedy_changed"),
            "mean_match_count": (
                float(sum(r.recall_match_count for r in triggers) / float(len(triggers))) if triggers else 0.0
            ),
        },
        "trigger_selector": {
            "require_gate_passed": bool(require_gate_passed),
        },
        "metrics": {},
    }

    metrics = result["metrics"]
    metrics["success_rate_treatment"] = _mean_bool(treat, "eventual_success")
    metrics["success_rate_control"] = _mean_bool(ctrl, "eventual_success")
    metrics["success_lift"] = _bootstrap_diff(
        treat, ctrl, field="eventual_success", n_boot=n_boot, seed=seed, invert=False
    )

    metrics["hazard_within_h_rate_treatment"] = _mean_bool(treat, "hazard_within_h")
    metrics["hazard_within_h_rate_control"] = _mean_bool(ctrl, "hazard_within_h")
    metrics["hazard_within_h_reduction"] = _bootstrap_diff(
        treat, ctrl, field="hazard_within_h", n_boot=n_boot, seed=seed + 1, invert=True
    )

    metrics["hazard_after_trigger_rate_treatment"] = _mean_bool(treat, "hazard_after_trigger")
    metrics["hazard_after_trigger_rate_control"] = _mean_bool(ctrl, "hazard_after_trigger")
    metrics["hazard_after_trigger_reduction"] = _bootstrap_diff(
        treat, ctrl, field="hazard_after_trigger", n_boot=n_boot, seed=seed + 2, invert=True
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure randomized recall lift from events.jsonl")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--horizon", type=int, default=20, help="Hazard window in steps after trigger")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--require_gate_passed", action="store_true", help="Use first randomized trigger with recall gate passed")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = analyze(
        run_dir=str(args.run_dir),
        horizon=max(1, int(args.horizon)),
        n_boot=max(100, int(args.bootstrap)),
        seed=int(args.seed),
        require_gate_passed=bool(args.require_gate_passed),
    )
    if bool(args.json):
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"run_dir={out['run_dir']}")
    print(
        f"episodes={out['episodes_total']} randomized_triggers={out['episodes_with_randomized_trigger']} "
        f"treat={out['treatment_n']} ctrl={out['control_n']}"
    )
    tq = out["trigger_quality"]
    print(
        "trigger_quality "
        f"agent_eligible_rate={float(_safe_float(tq.get('agent_recall_eligible_rate', 0.0), 0.0)):.3f} "
        f"gate_passed_rate={float(_safe_float(tq.get('gate_passed_rate', 0.0), 0.0)):.3f} "
        f"used_rate={float(_safe_float(tq.get('recall_used_rate', 0.0), 0.0)):.3f} "
        f"greedy_changed_rate={float(_safe_float(tq.get('greedy_action_changed_rate', 0.0), 0.0)):.3f} "
        f"mean_match_count={float(_safe_float(tq.get('mean_match_count', 0.0), 0.0)):.2f}"
    )
    m = out["metrics"]
    sl = m.get("success_lift", {}) if isinstance(m.get("success_lift"), dict) else {}
    hz = m.get("hazard_within_h_reduction", {}) if isinstance(m.get("hazard_within_h_reduction"), dict) else {}
    print(
        "success "
        f"treat={float(_safe_float(m.get('success_rate_treatment', 0.0), 0.0)):.3f} "
        f"ctrl={float(_safe_float(m.get('success_rate_control', 0.0), 0.0)):.3f} "
        f"lift={sl.get('estimate')}"
        + (f" ci95={sl.get('ci95')}" if sl.get("ci95") is not None else "")
    )
    print(
        f"hazard@{int(args.horizon)} "
        f"treat={float(_safe_float(m.get('hazard_within_h_rate_treatment', 0.0), 0.0)):.3f} "
        f"ctrl={float(_safe_float(m.get('hazard_within_h_rate_control', 0.0), 0.0)):.3f} "
        f"reduction={hz.get('estimate')}"
        + (f" ci95={hz.get('ci95')}" if hz.get('ci95') is not None else "")
    )


if __name__ == "__main__":
    main()
