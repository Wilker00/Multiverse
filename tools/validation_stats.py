"""
Statistical validation utilities for RL experiments.

Provides functions for computing confidence intervals and sample size requirements.
"""

import math
from typing import Any, Dict, List


def compute_validation_stats(
    values: List[float],
    min_detectable_delta: float = 0.5,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute validation statistics for a list of values.

    Returns confidence intervals and required sample sizes for hypothesis testing.

    Args:
        values: List of numeric values to analyze
        min_detectable_delta: Minimum effect size to detect
        confidence_level: Confidence level for intervals (default 0.95 for 95%)

    Returns:
        Dictionary with:
            - current_n: Current sample size
            - mean: Mean value
            - std: Standard deviation
            - ci_95: Tuple of (lower, upper) 95% confidence interval
            - required_n: Required sample size for detection
            - is_sufficient: Whether current_n is sufficient for min_detectable_delta
    """
    if not values:
        return {
            "current_n": 0,
            "mean": 0.0,
            "std": 0.0,
            "ci_95": (0.0, 0.0),
            "required_n": 0,
            "is_sufficient": False,
        }

    current_n = len(values)
    mean = sum(values) / current_n

    # Compute standard deviation
    if current_n == 1:
        std = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in values) / (current_n - 1)
        std = math.sqrt(variance)

    # Compute confidence interval using t-distribution approximation
    # For simplicity, use normal approximation with z-score for 95% confidence
    z_score = 1.96  # Standard z-score for 95% confidence
    se = std / math.sqrt(current_n) if current_n > 0 else 0.0
    ci_lower = mean - z_score * se
    ci_upper = mean + z_score * se

    # Compute required sample size for detecting min_detectable_delta
    # Using simplified formula: n = (2 * z_alpha/2 * std / delta)^2
    # where delta is min_detectable_delta
    if std > 0 and min_detectable_delta > 0:
        required_n = int(math.ceil((2 * z_score * std / min_detectable_delta) ** 2))
    else:
        required_n = 1

    is_sufficient = current_n >= required_n

    return {
        "current_n": current_n,
        "mean": mean,
        "std": std,
        "ci_95": (ci_lower, ci_upper),
        "required_n": required_n,
        "is_sufficient": is_sufficient,
    }


def compute_rate_stats(
    values: List[int],
    min_detectable_delta: float = 0.1,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute validation statistics for binary outcomes (rates/proportions).

    Args:
        values: List of binary values (0 or 1)
        min_detectable_delta: Minimum effect size to detect for proportions
        confidence_level: Confidence level for intervals (default 0.95 for 95%)

    Returns:
        Dictionary with:
            - current_n: Current sample size
            - rate: Proportion of 1s
            - ci_95: Tuple of (lower, upper) 95% confidence interval for rate
            - required_n: Required sample size for detection
            - is_sufficient: Whether current_n is sufficient for min_detectable_delta
    """
    if not values:
        return {
            "current_n": 0,
            "rate": 0.0,
            "ci_95": (0.0, 0.0),
            "required_n": 0,
            "is_sufficient": False,
        }

    current_n = len(values)
    successes = sum(values)
    rate = successes / current_n if current_n > 0 else 0.0

    # Compute Wilson score interval for proportion confidence interval
    # This is more accurate than normal approximation for extreme proportions
    z_score = 1.96  # For 95% confidence
    z_sq = z_score ** 2

    denominator = 1 + z_sq / current_n
    center = (rate + z_sq / (2 * current_n)) / denominator
    margin = z_score * math.sqrt(rate * (1 - rate) / current_n + z_sq / (4 * current_n ** 2)) / denominator

    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)

    # Required sample size for proportion test
    # Using simplified formula: n = (z_alpha/2 / delta)^2 * p(1-p)
    p = rate if rate > 0 else 0.5  # Use 0.5 if no data for worst-case estimate
    if min_detectable_delta > 0:
        required_n = int(math.ceil((z_score / min_detectable_delta) ** 2 * p * (1 - p)))
    else:
        required_n = 1

    is_sufficient = current_n >= required_n

    return {
        "current_n": current_n,
        "rate": rate,
        "ci_95": (ci_lower, ci_upper),
        "required_n": required_n,
        "is_sufficient": is_sufficient,
    }

