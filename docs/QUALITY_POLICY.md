# Quality Policy

This repo uses staged quality policies instead of a single universal gate.

## Profiles

- `fast`: local iteration profile. Keeps compile and communication checks, skips heavier readiness paths.
- `full`: broader engineering profile. Adds artifact hygiene and exception-debt regression checks.
- `release`: promotion-oriented profile. Enables the stricter readiness contract and workspace hygiene.

## Coverage

- Global coverage floor is enforced in `pyproject.toml`.
- Current fail-under: `70`.
- This is a ratcheting floor and should move upward over time, not downward.

## Exception Debt

- Broad exception debt is tracked in `tools/quality_baseline.json`.
- `broad_exception_max` is the current regression ceiling.
- `broad_exception_target` is the intended reduction target.
- `broad_exception_ratchet_step` is the expected next reduction increment.

The exception gate should prevent regressions first and then drive staged reductions.

## Release Policy

The `release` profile and `--release_gate` alias are intended for promotion-grade checks.

Current release thresholds:

- universal-model validation `min_coverage >= 0.10`
- universal-model validation `min_action_accuracy >= 0.10`
- production readiness `min_episodes >= 100`
- production readiness `min_success_rate >= 0.70`
- production readiness `max_bench_age_hours <= 24`
- production readiness `max_safety_violation_rate <= 0.10`
- benchmark required
- run directories required
- workspace hygiene required

## Operating Rule

Research workflows may use looser profiles. Promotion decisions should use `release` policy, not ad hoc local judgment.
