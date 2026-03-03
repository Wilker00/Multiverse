# Engineering Audit

Last updated: 2026-03-01

## Scope

This audit focused on the main engineering risk in the current codebase: complexity concentration in oversized runtime modules with mixed responsibilities.

The review targeted these files first:

- `core/rollout.py`
- `tools/multiverse_cli.py`
- `core/safe_executor.py`
- `memory/central_repository.py`
- `agents/transformer_agent.py`

## Findings

1. `core/safe_executor.py` remains the highest-risk runtime file.
- Current size: about 1642 lines.
- Responsibilities mixed together: config validation, danger estimation, fallback routing, planner/MCTS takeover, checkpoint recovery, telemetry, and shield integration.

2. `memory/central_repository.py` is still too broad.
- Current size: about 2073 lines.
- Responsibilities mixed together: ingest, dedupe, SQLite migration, tiering policy, cache management, ANN lookup, similarity metrics, and canary tooling.

3. `tools/multiverse_cli.py` previously combined parser setup, subprocess dispatch, run browsing, status views, and the interactive shell.
- This made the top-level CLI entry point harder to scan and test.

4. `core/rollout.py` previously mixed the episode loop with memory-recall bundle construction, selector telemetry, transfer-decision logging, and runtime warning formatting.
- That made the hot path noisier than necessary.

5. Documentation had drifted from the current code state.
- The README still claimed a failing CLI test that now passes in the current workspace.

## Cleanup Completed

### 1. Rollout support split

Extracted rollout support helpers into `core/rollout_support.py`.

- `core/rollout.py`: 802 -> 511 lines
- New helper module: `core/rollout_support.py` at 304 lines

What moved:

- on-demand memory bundle construction
- runtime warning formatting
- selector routing telemetry
- memory-recall transfer-decision record generation

Effect:

- `core/rollout.py` is now more clearly focused on episode execution and transition collection.

### 2. CLI run-surface split

Extracted run discovery and inspection helpers into `tools/multiverse_cli_runs.py`.

- `tools/multiverse_cli.py`: 1177 -> 984 lines
- New helper module: `tools/multiverse_cli_runs.py` at 220 lines

What moved:

- run discovery
- run path resolution
- run snapshot formatting
- `runs list|latest|files|tail|inspect`

Effect:

- `tools/multiverse_cli.py` is more clearly the command entry point and shell host, while run browsing logic is isolated.

### 3. Safe executor support split

Extracted the utility and config layer into `core/safe_executor_support.py`.

- `core/safe_executor.py`: about 1642 -> 1147 lines
- New helper module: `core/safe_executor_support.py` at 364 lines

What moved:

- `SafeExecutorConfig`
- config validation and parsing helpers
- danger-label parsing
- safety-violation and failure-mode helpers
- stable utility helpers such as observation keying and vector projection

Effect:

- `core/safe_executor.py` is now more clearly centered on runtime action selection, recovery, and takeover behavior.

### 4. Central repository support split

Extracted the config and tier/policy support layer into `memory/central_repository_support.py`.

- `memory/central_repository.py`: about 2073 -> 1801 lines
- New helper module: `memory/central_repository_support.py` at 311 lines

What moved:

- `CentralMemoryConfig` and repository stats/result dataclasses
- dedupe/path helpers
- memory tier policy loading and merge logic
- row enrichment and tier classification helpers

Effect:

- `memory/central_repository.py` is now more focused on repository mutation, cache management, and retrieval execution.

### 5. Safe executor runtime support split

Extracted takeover and recovery helpers into `core/safe_executor_runtime_support.py`.

- `core/safe_executor.py`: about 1147 -> 958 lines
- New helper module: `core/safe_executor_runtime_support.py` at 242 lines

What moved:

- safe-alternative selection
- checkpoint save/rewind helpers
- shield inference wrapper
- MCTS takeover helper
- planner takeover helper

Effect:

- `core/safe_executor.py` is now more tightly focused on top-level policy flow and state coordination.

### 6. Central repository query support split

Extracted query-cache and canary helpers into `memory/central_repository_query_support.py`.

- `memory/central_repository.py`: about 1801 -> 1597 lines
- New helper module: `memory/central_repository_query_support.py` at 277 lines

What moved:

- cached-query wrapper support
- similarity canary save/load/run helpers
- query-cache keying and cache timestamp helpers

Effect:

- `memory/central_repository.py` is now more focused on ingestion, similarity execution, and cache construction.

### 7. Central repository cache support split

Extracted prepared-row, vectorization, sidecar, and cache-build helpers into `memory/central_repository_cache_support.py`.

- `memory/central_repository.py`: about 1597 -> 1249 lines
- New helper module: `memory/central_repository_cache_support.py` at 348 lines

What moved:

- prepared-row and cache-entry dataclasses
- universal/raw vector preparation
- sidecar path/signature helpers
- row vectorization by raw and universal dimensions
- cache build/load helpers for `.simcache.json`
- ANN index construction helper

Effect:

- `memory/central_repository.py` is now more clearly focused on repository mutation, cache ownership, and retrieval orchestration rather than cache representation details.

### 8. Central repository similarity support split

Extracted ANN runtime metrics and retrieval scoring execution into `memory/central_repository_similarity_support.py`.

- `memory/central_repository.py`: about 1249 -> 921 lines
- New helper module: `memory/central_repository_similarity_support.py` at 472 lines

What moved:

- ANN runtime/env helpers and drift metric bookkeeping
- `find_similar` scoring execution across raw-vector, ANN-candidate, and universal-fallback branches
- row filter, temporal-decay, match-build, and trajectory extraction helpers

Effect:

- `memory/central_repository.py` is now more clearly focused on repository ownership, cache invalidation, ingestion, and public API wrappers rather than low-level similarity execution.

### 9. Central repository cache runtime split

Extracted cache invalidation, delta merging, and LRU cache refresh logic into `memory/central_repository_cache_runtime_support.py`.

- `memory/central_repository.py`: about 921 -> 880 lines
- New helper module: `memory/central_repository_cache_runtime_support.py` at 115 lines

What moved:

- similarity cache invalidation and sidecar cleanup
- cache delta append/merge helpers
- cache refresh and LRU eviction orchestration for `_get_similarity_cache_for_path`

Effect:

- `memory/central_repository.py` is now more clearly focused on repository APIs, ingest/backfill orchestration, and state ownership rather than cache runtime mechanics.

### 10. Bug fix uncovered during audit

Fixed a transformer-agent recall bug in `agents/transformer_agent.py`.

- The default recall risk key now correctly defaults to `risk`.
- Before this fix, the effective default behaved like `hazard`, which caused `risk`-based memory queries to fail unexpectedly.

## Verification Performed

Focused verification was run on 2026-03-01:

- `python -m pytest -q tests/test_multiverse_cli.py`
- `python -m pytest -q tests/test_rollout_observability.py tests/test_trainer_on_demand_memory_recall.py tests/test_memory_recall_agent.py test_dt_memory.py`
- `python -m pytest -q tests/test_safe_executor_mcts.py tests/test_safe_executor_mcts_overrides.py tests/test_safe_executor_confidence_model.py`
- `python -m pytest -q tests/test_central_repository_tier_policy.py tests/test_central_repository_backfill.py tests/test_central_repository_universal_fallback.py tests/test_central_repository_perf_hardening.py`
- `python -m pytest -q tests/test_memory_thread_safety.py`
- `python -m pytest -q tests/test_safe_executor_mcts.py tests/test_safe_executor_mcts_overrides.py tests/test_safe_executor_confidence_model.py` after runtime-support split
- `python -m pytest -q tests/test_central_repository_perf_hardening.py tests/test_central_repository_universal_fallback.py tests/test_memory_thread_safety.py` after query-support split
- `python -m pytest -q tests/test_central_repository_perf_hardening.py tests/test_central_repository_universal_fallback.py tests/test_memory_thread_safety.py tests/test_central_repository_tier_policy.py tests/test_central_repository_backfill.py` after cache-support split
- `python -m pytest -q tests/test_central_repository_perf_hardening.py tests/test_central_repository_universal_fallback.py tests/test_memory_thread_safety.py tests/test_central_repository_tier_policy.py tests/test_central_repository_backfill.py` after similarity-support split
- `python -m pytest -q tests/test_central_repository_perf_hardening.py tests/test_central_repository_universal_fallback.py tests/test_memory_thread_safety.py tests/test_central_repository_tier_policy.py tests/test_central_repository_backfill.py` after cache-runtime split
- `python tools/multiverse_cli.py status --json`
- `python tools/multiverse_cli.py runs inspect --count-events --json`

Result:

- 78 focused tests passed
- CLI status and run-inspection smoke checks passed

Note:

- The full repository pytest suite was not rerun in this audit pass.

## Remaining Targets

Recommended next splits, in order:

1. `memory/central_repository.py`
- Split remaining ingestion/backfill/indexing paths from repository mutation APIs.

2. `core/safe_executor.py`
- Split danger-model loading/risk estimation and post-step outcome handling from the main runtime class.

3. `tools/multiverse_cli.py`
- Move the interactive shell implementation into its own module so the parser/dispatch surface stays small.

## Practical Guidance

If you want to continue this cleanup, the safest next refactor is now `memory/central_repository.py` by extracting:

- similarity cache build/load/invalidation helpers
- ANN selection and drift-tuning helpers
- universal/raw vector scoring helpers

That should reduce blast radius without changing repository semantics.
