# Multiverse

Multiverse is a local reinforcement-learning codebase with custom environments ("verses"), agent registries, rollout/training orchestration, memory retrieval, and runtime safety controls.

## 🚀 Getting Started

**New to Multiverse?** Start here:

1. **[Setup Guide](docs/SETUP.md)** - Installation in 5 minutes
2. **[Quickstart Tutorial](docs/QUICKSTART.md)** - Your first agent in 15 minutes
3. **[Project Introduction](docs/PROJECT_INTRO.md)** - Understand the architecture

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

As of 2026-03-01, this repository has focused verification for the runtime paths touched by the latest cleanup:

- `python -m pytest -q tests/test_multiverse_cli.py`
- `python -m pytest -q tests/test_rollout_observability.py tests/test_trainer_on_demand_memory_recall.py tests/test_memory_recall_agent.py test_dt_memory.py`
- `python -m pytest -q tests/test_safe_executor_mcts.py tests/test_safe_executor_mcts_overrides.py tests/test_safe_executor_confidence_model.py`
- `python -m pytest -q tests/test_central_repository_tier_policy.py tests/test_central_repository_backfill.py tests/test_central_repository_universal_fallback.py tests/test_central_repository_perf_hardening.py`
- `python -m pytest -q tests/test_memory_thread_safety.py`
- Re-verified after deeper safety/memory splits:
  - `python -m pytest -q tests/test_safe_executor_mcts.py tests/test_safe_executor_mcts_overrides.py tests/test_safe_executor_confidence_model.py`
  - `python -m pytest -q tests/test_central_repository_perf_hardening.py tests/test_central_repository_universal_fallback.py tests/test_memory_thread_safety.py`
- Result: `78 focused tests passed`
- CLI smoke checks:
  - `python tools/multiverse_cli.py status --json`
  - `python tools/multiverse_cli.py runs inspect --count-events --json`

The full repository pytest suite was not rerun in this pass.

Paper-readiness artifacts were generated at `2026-02-20T03:21:01Z` under `models/paper/paper_readiness/latest/`.

This README is intentionally strict. It only describes what is present in code and currently exercised by the active pytest suite.

## Key Quantitative Results

- Retrieval speedup (ANN vs exact): `75.20x` (`109.0455s` exact vs `1.4502s` ANN).
  - Artifact: `models/validation/retrieval_ann_benchmark_v1.json`
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
- Safety runtime helpers: `core/safe_executor_runtime_support.py`
- Memory indexing/retrieval: `memory/episode_index.py`, `memory/retrieval.py`, `memory/central_repository.py`
- Memory repository support helpers: `memory/central_repository_support.py`
- Memory cache helpers: `memory/central_repository_cache_support.py`
- Memory query/canary helpers: `memory/central_repository_query_support.py`
- Universal model + API: `models/universal_model.py`, `tools/universal_model_api.py`

## Engineering Cleanup Notes

- Cleanup completed on 2026-03-01 reduced responsibility concentration in two runtime entry surfaces:
- Cleanup completed on 2026-03-01 reduced responsibility concentration in four runtime surfaces:
  - `core/rollout.py`: helper logic extracted to `core/rollout_support.py` (`802 -> 511` lines)
  - `tools/multiverse_cli.py`: run browsing extracted to `tools/multiverse_cli_runs.py` (`1177 -> 984` lines)
  - `core/safe_executor.py`: config/utility logic extracted to `core/safe_executor_support.py` (`1642 -> 1147` lines)
  - `core/safe_executor.py`: takeover/recovery logic extracted to `core/safe_executor_runtime_support.py` (`1147 -> 958` lines)
  - `memory/central_repository.py`: config, path, and tier-policy support extracted to `memory/central_repository_support.py` (`2073 -> 1801` lines)
  - `memory/central_repository.py`: query-cache and canary logic extracted to `memory/central_repository_query_support.py` (`1801 -> 1597` lines)
  - `memory/central_repository.py`: cache representation/build helpers extracted to `memory/central_repository_cache_support.py` (`1597 -> 1249` lines)
- Bug fix completed during audit:
  - `agents/transformer_agent.py` now defaults recall risk queries to `risk`, which restores expected on-demand memory behavior in tests using risk-based observations.
- Remaining highest-concentration targets:
  - `memory/central_repository.py`
  - `core/safe_executor.py`
  - `tools/multiverse_cli.py` interactive shell path
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
- Manual smoke scripts are in `tools/` (for example `tools/smoke_v2_verses.py`).

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
