# Multiverse

Multiverse is a local reinforcement-learning codebase with custom environments ("verses"), agent registries, rollout/training orchestration, memory retrieval, and runtime safety controls.

## Verified Status

As of 2026-02-20, this repository was validated with:

- `python -m pytest -q`
- Result: `237 passed, 1 failed`
- Failing test: `tests/test_multiverse_cli.py::TestMultiverseCli::test_discover_and_resolve_runs`

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
- Safety wrapper: `core/safe_executor.py`
- Memory indexing/retrieval: `memory/episode_index.py`, `memory/retrieval.py`, `memory/central_repository.py`
- Universal model + API: `models/universal_model.py`, `tools/universal_model_api.py`

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
- Project introduction: `docs/PROJECT_INTRO.md`.
- Contribution guide: `CONTRIBUTING.md`.
- Security policy: `SECURITY.md`.
- License: `LICENSE`.
- Older planning/marketing-style docs were removed to avoid conflicting or inflated claims.
