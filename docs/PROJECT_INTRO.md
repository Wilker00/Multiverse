# Multiverse: Project Introduction

Last updated: 2026-02-17

## What this project is
Multiverse is a local reinforcement-learning platform for training and evaluating many agent types across many custom environments (called "verses"). It combines:
- environment simulation
- pluggable agent algorithms
- rollout/training orchestration
- safety controls during action selection
- cross-run memory and retrieval
- local dashboards and gating tools for operations

In plain language: this repo is an RL lab + runtime framework where you can prototype policies, run them in varied worlds, and attach safety/memory systems so behavior can be monitored and promoted more safely.

## Technical one-paragraph summary
At runtime, a `VerseSpec` and `AgentSpec` are validated (`core/types.py`), created via registries (`verses/registry.py`, `agents/registry.py`), executed through the rollout engine (`core/rollout.py`) and trainer (`orchestrator/trainer.py`), optionally guarded by `SafeExecutor` (`core/safe_executor.py`), logged to per-run JSONL artifacts, and indexed into memory stores for retrieval (`memory/episode_index.py`, `memory/retrieval.py`, `memory/central_repository.py`). Supporting tools in `tools/` handle CI gates, benchmarking, model services, and operational workflows.

## How the repo is organized
- `core/`: runtime contracts and execution primitives (types, rollout loop, safe executor, MCTS/planning helpers).
- `verses/`: custom RL environments and registry factories.
- `agents/`: policy implementations and agent registry.
- `orchestrator/`: training/evaluation orchestration, curriculum, promotion-board logic.
- `memory/`: indexing, retrieval, embedding/similarity, central memory repository.
- `models/`: model artifacts, datasets, policy manifests, and reports.
- `tools/`: CLIs for training, benchmarking, evaluation, deployment checks, and maintenance.
- `tests/`: pytest suites covering runtime and feature behavior.
- `command_center/`: Next.js operator UI for live snapshots (skill graph, shield feed, promotion decisions).

## Core execution flow
1. Build run specs.
- CLI tools like `tools/train_agent.py` construct `VerseSpec` + `AgentSpec`.

2. Resolve factories.
- `orchestrator/trainer.py` calls verse/agent registries to instantiate runtime objects.

3. Run episodes.
- `core/rollout.py` loops through steps, collects events/transitions, and optionally trains.

4. Enforce runtime safety.
- `core/safe_executor.py` can veto/replace risky actions, invoke fallback agents, and trigger planner/MCTS interventions.

5. Persist telemetry.
- `memory/event_log.py` writes `events.jsonl` and episode summaries in run directories.

6. Index and reuse memory.
- `memory/episode_index.py` and `memory/central_repository.py` support local and cross-run retrieval.

## Safety and memory in practical terms
### Safety layer
`SafeExecutor` is a runtime wrapper, not just an offline metric. It can:
- estimate danger/confidence per action
- hard-block risky/low-confidence actions
- route to fallback policies
- use planner/MCTS takeover in specific conditions
- annotate events with safety metadata for observability and auditing

### Memory layer
The repo uses two memory styles:
- run-local retrieval (`memory/retrieval.py`) over a run's own `episodes.jsonl/events.jsonl`
- central cross-run memory (`memory/central_repository.py`) with dedupe, tiering (STM/LTM), and similarity search

That allows both "within-run recall" and "transfer from prior runs/verses" patterns.

## Command center (frontend)
`command_center/` is a Next.js app that reads repo artifacts and exposes:
- Skill Galaxy graph from recent run episode tags
- neural/safety feed from event metadata
- promotion-board queue and operator bless/veto actions

Key endpoints:
- `GET /api/command-center` -> snapshot payload
- `POST /api/promotion-board/decision` -> persist human decision records

## How to run the project quickly
From repo root:
- Run tests: `python -m pytest -q`
- Single training run: `python tools/train_agent.py --algo random --verse line_world --episodes 20 --max_steps 40`
- Distributed local run: `python tools/train_distributed.py --mode sharded --algo q --verse line_world --episodes 100`

Frontend (from `command_center/`):
- Install deps: `npm install`
- Dev server: `npm run dev`
- Build: `npm run build`
- Lint: `npm run lint`

## Maturity snapshot (as of 2026-02-17)
- Local verification run: `python -m pytest -q`
- Result observed: `206 passed, 2 warnings`
- The top-level `readme.md` still says `197 passed` (also dated 2026-02-17), so that status line is currently stale versus the latest run.

## What this project is best for
- experimenting with agent algorithms across many small/medium custom environments
- validating runtime safety interventions in-loop
- testing retrieval/memory-augmented behavior
- running local benchmark/gate workflows before promotion/deployment decisions

## Where to start reading (by role)
Runtime/agent engineer:
- `tools/train_agent.py`
- `orchestrator/trainer.py`
- `core/rollout.py`
- `agents/registry.py`

Environment designer:
- `verses/registry.py`
- any verse implementation under `verses/`

Safety/reliability engineer:
- `core/safe_executor.py`
- `core/mcts_search.py`
- `tools/ci_gate.py`
- `tests/test_safe_executor_mcts.py`

Memory/retrieval engineer:
- `memory/retrieval.py`
- `memory/episode_index.py`
- `memory/central_repository.py`

Operator/dashboard engineer:
- `command_center/lib/repoData.ts`
- `command_center/components/CommandCenter.tsx`
- `command_center/app/api/command-center/route.ts`

## Boundaries and caveats
- This repo contains many generated/runtime artifacts (`runs*`, `central_memory*`, benchmark outputs, datasets) alongside code. Treat executable code and tests as source-of-truth for behavior.
- The working tree currently has extensive in-progress changes; this is an actively evolving codebase.
- Documentation policy in this repo is intentionally strict: claims should track tested behavior, not aspirational roadmap language.
