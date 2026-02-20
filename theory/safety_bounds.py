"""
theory/safety_bounds.py

Formal safety bounds using a bounded-difference concentration inequality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def hoeffding_epsilon(*, num_episodes: int, confidence: float = 0.95) -> float:
    """
    Solve P(|p_hat - p| > eps) <= 2 exp(-2 K eps^2) for eps.
    """
    k = max(1, int(num_episodes))
    c = min(1.0 - 1e-12, max(1e-12, float(confidence)))
    delta = 1.0 - c
    return float(math.sqrt(math.log(2.0 / delta) / (2.0 * float(k))))


def _extract_violation_from_episode_like(item: Any) -> bool:
    if isinstance(item, bool):
        return bool(item)
    if isinstance(item, (int, float)):
        return bool(float(item) > 0.0)
    if isinstance(item, dict):
        for key in ("safety_violation", "violations", "fell_cliff", "is_violation"):
            if key in item:
                v = item.get(key)
                if isinstance(v, bool):
                    return bool(v)
                if isinstance(v, (int, float)):
                    return bool(float(v) > 0.0)
        return False
    return bool(getattr(item, "safety_violation", False))


def derive_safety_certificate(
    *,
    observed_episodes: Optional[Sequence[Any]] = None,
    violation_flags: Optional[Sequence[Any]] = None,
    observed_violations: Optional[int] = None,
    total_episodes: Optional[int] = None,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Return a confidence-bounded certificate for true violation rate.
    """
    if violation_flags is not None:
        flags = [bool(_extract_violation_from_episode_like(x)) for x in violation_flags]
        k = int(len(flags))
        v = int(sum(1 for f in flags if bool(f)))
    elif observed_episodes is not None:
        flags = [bool(_extract_violation_from_episode_like(x)) for x in observed_episodes]
        k = int(len(flags))
        v = int(sum(1 for f in flags if bool(f)))
    else:
        k = max(0, _safe_int(total_episodes, 0))
        v = max(0, min(k, _safe_int(observed_violations, 0)))

    if k <= 0:
        return {
            "observed_violations": 0,
            "episodes": 0,
            "observed_violation_rate": 0.0,
            "epsilon": 1.0,
            "upper_bound": 1.0,
            "lower_bound": 0.0,
            "confidence": float(confidence),
            "method": "hoeffding",
            "certificate": "Insufficient observations (episodes=0).",
        }

    p_hat = float(v) / float(k)
    eps = hoeffding_epsilon(num_episodes=int(k), confidence=float(confidence))
    upper = min(1.0, float(p_hat + eps))
    lower = max(0.0, float(p_hat - eps))
    return {
        "observed_violations": int(v),
        "episodes": int(k),
        "observed_violation_rate": float(p_hat),
        "epsilon": float(eps),
        "upper_bound": float(upper),
        "lower_bound": float(lower),
        "confidence": float(confidence),
        "method": "hoeffding",
        "certificate": (
            f"With {float(confidence):.0%} confidence, true violation rate is in "
            f"[{lower:.4f}, {upper:.4f}] and <= {upper:.4f}."
        ),
    }


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


def extract_episode_violation_flags_from_events(
    *,
    events_jsonl_path: str,
    violation_info_keys: Sequence[str] = ("fell_cliff", "safety_violation", "safety_violated", "violation"),
    severe_reward_threshold: float = -100.0,
) -> Dict[str, Any]:
    """
    Build per-episode violation flags from step-level event logs.
    """
    path = str(events_jsonl_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"events file not found: {path}")

    flags_by_episode: Dict[str, bool] = {}
    verse_name: Optional[str] = None
    total_steps = 0
    for row in _iter_jsonl(path):
        total_steps += 1
        if verse_name is None:
            verse_name = str(row.get("verse_name", "")).strip() or None
        ep_id = str(row.get("episode_id", "")).strip()
        if not ep_id:
            continue
        if ep_id not in flags_by_episode:
            flags_by_episode[ep_id] = False

        info = row.get("info")
        info_d = info if isinstance(info, dict) else {}
        violated = False
        for k in violation_info_keys:
            v = info_d.get(str(k))
            if isinstance(v, bool):
                violated = violated or bool(v)
            elif isinstance(v, (int, float)):
                violated = violated or (float(v) > 0.0)
        reward = _safe_float(row.get("reward"), default=0.0)
        if reward <= float(severe_reward_threshold):
            violated = True
        if violated:
            flags_by_episode[ep_id] = True

    flags = [bool(v) for v in flags_by_episode.values()]
    return {
        "events_path": path,
        "verse_name": verse_name,
        "episodes": int(len(flags)),
        "steps": int(total_steps),
        "violation_flags": flags,
        "observed_violations": int(sum(1 for x in flags if x)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Hoeffding safety certificate from episodes or events logs.")
    ap.add_argument("--events_jsonl", type=str, default="")
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--violations", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    if str(args.events_jsonl).strip():
        extracted = extract_episode_violation_flags_from_events(events_jsonl_path=str(args.events_jsonl))
        cert = derive_safety_certificate(
            violation_flags=extracted["violation_flags"],
            confidence=float(args.confidence),
        )
        report = {"input": extracted, "certificate": cert}
    else:
        cert = derive_safety_certificate(
            observed_violations=int(args.violations),
            total_episodes=int(args.episodes),
            confidence=float(args.confidence),
        )
        report = {"input": {"episodes": int(args.episodes), "violations": int(args.violations)}, "certificate": cert}

    if str(args.out_json).strip():
        os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
        with open(str(args.out_json), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
