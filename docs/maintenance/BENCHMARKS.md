# Benchmarks

This file describes the benchmark tooling that exists in this repository.

## Purpose

Benchmarks here are engineering regression checks, not public leaderboard claims.

They are used to compare candidate vs baseline behavior on local verse suites and to gate promotion/deployment scripts.

## Main Commands

- Run benchmark suite:
  - `python tools/run_benchmarks.py --candidate_algo special_moe --baseline_algo gateway --mode hard`
- Run fixed-seed transfer benchmark:
  - `python tools/run_fixed_seed_benchmark.py --target_verse warehouse_world --transfer_algo q --baseline_algo q --seeds 123,223,337`
- Run production readiness gate:
  - `python tools/production_readiness_gate.py --manifest_path models/default_policy_set.json --bench_json models/benchmarks/latest.json --require_benchmark`
- Run canonical paper-readiness pack (frozen config):
  - `python tools/run_paper_readiness_pack.py --pack experiment/paper_readiness_pack_v1.json --candidate_algo memory_recall --baseline_algo q`

## Typical Outputs

- `models/benchmarks/*.json`
- `models/benchmarks/*.md`
- `models/paper/paper_readiness/latest/pack_summary.json`

These are generated artifacts and should not be treated as permanent source-of-truth documents.

## CI Relationship

- CI uses `tools/ci_gate.py` and selected pytest suites.
- Benchmarks can be part of release gating depending on CLI flags/config.

## Interpretation Rules

- A benchmark report only reflects the exact command, config, seeds, and code revision used.
- Do not generalize one benchmark file into a global performance claim.
