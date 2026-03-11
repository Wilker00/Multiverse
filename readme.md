# Multiverse

Multiverse is a local reinforcement-learning operations framework for training, evaluating, and promoting agents across custom environments ("verses"). It combines runtime safety controls, cross-run memory retrieval, benchmark gates, and operator-facing tooling in one codebase.

## 🚀 Getting Started

**New to Multiverse?** Start here:

1. **[Setup Guide](docs/SETUP.md)** - Installation in 5 minutes
2. **[Quickstart Tutorial](docs/QUICKSTART.md)** - Your first agent in 15 minutes
3. **[Hot-Reload Configuration](docs/HOT_RELOAD.md)** 🔥 NEW - Update configs without restart
4. **[YAML Configuration](docs/YAML_CONFIGURATION.md)** - Structured config with validation
5. **[Configuration Reference](docs/CONFIGURATION.md)** - All 250+ parameters documented
6. **[Project Introduction](docs/PROJECT_INTRO.md)** - Understand the architecture

**Quick Install:**
```bash
git clone https://github.com/wilker00/multiverse.git
cd multiverse
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/train_agent.py --algo q --verse line_world --episodes 20
```

## Verified Status

As of 2026-03-11, this repository has focused verification for the runtime paths touched by the latest cleanup and repo-shape normalization:

- `python -m pytest -q tests/test_multiverse_cli.py`
- `python -m pytest -q tests/test_rollout_observability.py tests/test_trainer_on_demand_memory_recall.py tests/test_memory_recall_agent.py tests/test_dt_memory.py`
- `python -m pytest -q tests/test_safe_executor_mcts.py tests/test_safe_executor_mcts_overrides.py tests/test_safe_executor_confidence_model.py`
- `python -m pytest -q tests/test_central_repository_tier_policy.py tests/test_central_repository_backfill.py tests/test_central_repository_universal_fallback.py tests/test_central_repository_perf_hardening.py`
- `python -m pytest -q tests/test_memory_thread_safety.py`
- `python -m pytest -q tests/test_validation_stats.py tests/test_update_centroid.py tests/test_multiverse_cli.py`
- `python -m pytest -q tests/test_decision_transformer.py tests/test_adt_pipeline.py tests/test_memory_recall_agent.py tests/test_trainer_on_demand_memory_recall.py`
- Current repo-local pytest collection (2026-03-11): `329 tests collected` via `python -m pytest tests --collect-only -q`
- CLI smoke checks:
  - `python tools/multiverse_cli.py status --json`
  - `python tools/multiverse_cli.py runs inspect --count-events --json`
  - `python tools/parallel_rollout.py --help`
  - `python tools/run_curiosity_loop.py --help`

The full repository pytest suite was not rerun in this pass.

Paper-readiness artifacts were generated at `2026-02-20T03:21:01Z` under `models/paper/paper_readiness/latest/`.

This README is intentionally strict. It only describes what is present in code and currently exercised by the active pytest suite.

## Key Quantitative Results

- Retrieval speedup (ANN vs exact): `75.20x` (`109.0455s` exact vs `1.4502s` ANN).
  - Artifact: `models/validation/retrieval_ann_benchmark_v1.json`
- **Generalist Scaling (Phase 5)**: Supported **13 universes** simultaneously via the 256-d Omega backbone.
  - Successor to `dt_generalist_v2`, resolving neural collisions in multi-task learning.
- **Ghost-Following Achievement**: Enabled 0-shot navigation in complex verses via periodic roadmap recall (`frequency=5`).
  - Evolution recorded in `evolutionary_report.md`.
- Safety certificate (Hoeffding, 95% confidence): `0/200` observed violations, upper bound `0.0960`.
  - Artifact: `models/paper/paper_readiness/latest/phase3_theory_validation.json`
- Cliff-world return penalty reduction (candidate vs baseline): `8.45x` (`-2030.5` to `-240.25`).
  - Artifact: `models/paper/paper_readiness/latest/benchmark_gate.json`

## Active Runtime Surface

- Environment registry: `verses/registry.py`
- Agent registry: `agents/registry.py`
- Trainer: `orchestrator/trainer.py`
- Rollout loop: `core/rollout.py`
- Rollout support helpers: `core/rollout_support.py`
- Safety wrapper: `core/safe_executor.py`
- Safety support helpers: `core/safe_executor_support.py`
- Safety policy helpers: `core/safe_executor_policy_support.py`
- Safety risk helpers: `core/safe_executor_risk_support.py`
- Safety outcome helpers: `core/safe_executor_outcome_support.py`
- Safety runtime helpers: `core/safe_executor_runtime_support.py`
- Memory indexing/retrieval: `memory/episode_index.py`, `memory/retrieval.py`, `memory/central_repository.py`
- Memory repository support helpers: `memory/central_repository_support.py`
- Memory ingest/runtime helpers: `memory/central_repository_ingest_support.py`, `memory/central_repository_runtime_support.py`
- Memory cache helpers: `memory/central_repository_cache_support.py`
- Memory cache runtime helpers: `memory/central_repository_cache_runtime_support.py`
- Memory query/canary helpers: `memory/central_repository_query_support.py`
- Memory similarity helpers: `memory/central_repository_similarity_support.py`
- Operational tooling: `tools/multiverse_cli.py`, `tools/promotion_sentinel.py`, `tools/universal_model_api.py`
- Universal model: `models/universal_model.py`

## Engineering Cleanup Notes

- Cleanup through 2026-03-11 reduced responsibility concentration across the main runtime surfaces and normalized the repo layout:
  - `core/rollout.py`: helper logic extracted to `core/rollout_support.py` (`802 -> 511` lines)
  - `tools/multiverse_cli.py`: run browsing extracted to `tools/multiverse_cli_runs.py` (`1177 -> 984` lines)
  - `core/safe_executor.py`: config/utility logic extracted to `core/safe_executor_support.py` (`1642 -> 1147` lines)
  - `core/safe_executor.py`: takeover/recovery logic extracted to `core/safe_executor_runtime_support.py` (`1147 -> 958` lines)
  - `core/safe_executor.py`: policy/risk/outcome helpers extracted into dedicated support modules for action-selection flow hardening
  - `memory/central_repository.py`: config, path, and tier-policy support extracted to `memory/central_repository_support.py` (`2073 -> 1801` lines)
  - `memory/central_repository.py`: query-cache and canary logic extracted to `memory/central_repository_query_support.py` (`1801 -> 1597` lines)
  - `memory/central_repository.py`: cache representation/build helpers extracted to `memory/central_repository_cache_support.py` (`1597 -> 1249` lines)
  - `memory/central_repository.py`: ANN runtime and retrieval scoring extracted to `memory/central_repository_similarity_support.py` (`1249 -> 921` lines)
  - `memory/central_repository.py`: cache invalidation and delta/LRU runtime helpers extracted to `memory/central_repository_cache_runtime_support.py` (`921 -> 880` lines)
  - `memory/central_repository.py`: repository bootstrap and ingest orchestration extracted into dedicated runtime/ingest support modules
- Repo-shape cleanup completed:
  - executable entrypoints now live under `tools/`
  - tests now live under `tests/`
  - stale root patch scripts and temporary verification scripts were removed
- Bug fix completed during audit:
  - `agents/transformer_agent.py` now defaults recall risk queries to `risk`, which restores expected on-demand memory behavior in tests using risk-based observations.
- Highest-concentration areas still worth monitoring:
  - `memory/central_repository.py`
  - `core/safe_executor.py`
  - `tools/multiverse_cli.py`
- Deep audit notes:
  - `docs/ENGINEERING_AUDIT.md`

## Primary CLIs

- Unified convenience CLI:
  - `multiverse status` (snapshot overview)
  - `multiverse shell` (app-like full-screen live mode; suggestion picker, TAB autocomplete, scrollable logs, exit with `Ctrl+Esc`)
  - In shell: `:layout compact|full`, `:theme dark|glass|matrix|contrast` (default: `dark`), `:intensity 0..3`
  - `multiverse.bat universe list` (Windows launcher)
  - `python tools/multiverse_cli.py universe list`
  - `multiverse train --profile quick`
  - `multiverse train --profile research --dry-run`
  - `python tools/multiverse_cli.py train --universe line_world --algo random --episodes 20 --max-steps 40`
  - `python tools/multiverse_cli.py distributed --mode sharded --universe line_world --algo q --episodes 100`
  - `python tools/multiverse_cli.py runs list --runs-root runs`
  - `python tools/multiverse_cli.py runs latest`
  - `multiverse runs inspect --count-events`
  - `python tools/multiverse_cli.py runs files --run-id <run_id>`
  - `python tools/multiverse_cli.py runs tail --run-id <run_id> --file events.jsonl --lines 30`
- Train single run:
  - `python tools/train_agent.py --algo random --verse line_world --episodes 20 --max_steps 40`
- Run distributed local training:
  - `python tools/train_distributed.py --mode sharded --algo q --verse line_world --episodes 100`
- Run tests:
  - `python -m pytest -q`

## Repository Rules

- Runtime artifacts are not source code and are ignored (`runs*`, `central_memory*`, benchmark/tuning outputs, local envs, frontend deps).
- `tests/test_*.py` are the automated suites.
- CLI, smoke, maintenance, and monitoring scripts are in `tools/` (for example `tools/validate_all_verses.py` and `tools/promotion_sentinel.py`).

## What Was Removed In Cleanup

- Legacy/manual root scripts that were not part of tested runtime.
- Script-style legacy files under `tests/` that executed at import time and were not pytest tests.
- Stale benchmark report markdown snapshots and outdated docs.

## Documentation

- Current docs are in `docs/README.md`.
- **Full Technical Paper**: `docs/PAPER.md`.
- **Engineering Audit**: `docs/ENGINEERING_AUDIT.md`.
- Project introduction: `docs/PROJECT_INTRO.md`.
- Contribution guide: `CONTRIBUTING.md`.
- Security policy: `SECURITY.md`.
- License: `LICENSE`.
- Older planning/marketing-style docs were removed to avoid conflicting or inflated claims.
