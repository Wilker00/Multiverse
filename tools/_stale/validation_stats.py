"""
tools/validation_stats.py

Statistical helpers for validation reports:
- mean +/- standard error
- confidence intervals
- sample-size adequacy checks for a minimum detectable delta
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _normal_ppf(p: float) -> float:
    from statistics import NormalDist

    q = min(1.0 - 1e-12, max(1e-12, float(p)))
    return float(NormalDist().inv_cdf(q))


def _try_t_critical(alpha: float, dof: int) -> Optional[float]:
    if int(dof) <= 0:
        return None
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(scipy_stats.t.ppf(1.0 - float(alpha) / 2.0, int(dof)))
    except Exception:
        return None


def _try_t_required_n(
    *,
    sample_std: float,
    min_detectable_delta: float,
    alpha: float,
    power: float,
) -> Optional[float]:
    if sample_std <= 1e-12 or min_detectable_delta <= 0.0:
        return 1.0
    effect_size = float(min_detectable_delta) / float(sample_std)
    if effect_size <= 1e-12:
        return None

    try:
        from statsmodels.stats.power import TTestPower  # type: ignore

        return float(
            TTestPower().solve_power(
                effect_size=effect_size,
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided",
            )
        )
    except Exception:
        return None


def _normal_required_n(
    *,
    sample_std: float,
    min_detectable_delta: float,
    alpha: float,
    power: float,
) -> float:
    if sample_std <= 1e-12 or min_detectable_delta <= 0.0:
        return 1.0
    z_alpha = _normal_ppf(1.0 - float(alpha) / 2.0)
    z_beta = _normal_ppf(float(power))
    n = ((z_alpha + z_beta) * float(sample_std) / float(min_detectable_delta)) ** 2
    return float(max(1.0, n))


def compute_validation_stats(
    results: Sequence[float],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    min_detectable_delta: float = 0.10,
) -> Dict[str, Any]:
    """
    Compute validation summary statistics for scalar results.

    Returns:
    - mean
    - standard_error
    - ci_95 (tuple[low, high])
    - current_n
    - required_n
    - is_sufficient
    """
    values = [_safe_float(x) for x in results]
    n = int(len(values))
    if n <= 0:
        return {
            "mean": 0.0,
            "standard_error": 0.0,
            "std": 0.0,
            "ci_95": [0.0, 0.0],
            "current_n": 0,
            "required_n": 1,
            "is_sufficient": False,
            "alpha": float(alpha),
            "power": float(power),
            "min_detectable_delta": float(min_detectable_delta),
            "method": "empty",
        }

    mean = float(statistics.mean(values))
    std = float(statistics.stdev(values)) if n > 1 else 0.0
    se = float(std / math.sqrt(float(n))) if n > 1 else 0.0

    dof = n - 1
    t_critical = _try_t_critical(alpha=float(alpha), dof=dof)
    if t_critical is not None and n > 1:
        half = float(t_critical * se)
        ci_method = "student_t"
    else:
        z = _normal_ppf(1.0 - float(alpha) / 2.0)
        half = float(z * se)
        ci_method = "normal"
    ci = [float(mean - half), float(mean + half)]

    n_t = _try_t_required_n(
        sample_std=float(std),
        min_detectable_delta=float(min_detectable_delta),
        alpha=float(alpha),
        power=float(power),
    )
    n_norm = _normal_required_n(
        sample_std=float(std),
        min_detectable_delta=float(min_detectable_delta),
        alpha=float(alpha),
        power=float(power),
    )
    required_n = int(math.ceil(n_t if isinstance(n_t, float) and n_t > 0.0 else n_norm))

    return {
        "mean": float(mean),
        "standard_error": float(se),
        "std": float(std),
        "ci_95": ci,
        "current_n": int(n),
        "required_n": int(max(1, required_n)),
        "is_sufficient": bool(n >= max(1, required_n)),
        "alpha": float(alpha),
        "power": float(power),
        "min_detectable_delta": float(min_detectable_delta),
        "method": ci_method,
    }


def compute_rate_stats(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    min_detectable_delta: float = 0.10,
) -> Dict[str, Any]:
    """
    Binary-rate flavored wrapper over `compute_validation_stats`.
    Accepts booleans/0-1 values and returns mean as rate.
    """
    clipped = [1.0 if _safe_float(v) >= 0.5 else 0.0 for v in values]
    out = compute_validation_stats(
        clipped,
        alpha=float(alpha),
        power=float(power),
        min_detectable_delta=float(min_detectable_delta),
    )
    out["rate"] = float(out.get("mean", 0.0))
    return out


def _parse_csv_values(raw: str) -> List[float]:
    out: List[float] = []
    for part in str(raw).replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        out.append(float(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute confidence intervals and power-based sample-size checks.")
    ap.add_argument("--values", type=str, required=True, help="Comma-separated list of numeric values.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--min_detectable_delta", type=float, default=0.10)
    ap.add_argument("--binary", action="store_true", help="Interpret values as binary and report rate stats.")
    args = ap.parse_args()

    vals = _parse_csv_values(str(args.values))
    if bool(args.binary):
        report = compute_rate_stats(
            vals,
            alpha=float(args.alpha),
            power=float(args.power),
            min_detectable_delta=float(args.min_detectable_delta),
        )
    else:
        report = compute_validation_stats(
            vals,
            alpha=float(args.alpha),
            power=float(args.power),
            min_detectable_delta=float(args.min_detectable_delta),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
