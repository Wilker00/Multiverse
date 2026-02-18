# Docs

This folder now contains only factual operational notes.

## Entry Points

- Project introduction: `docs/PROJECT_INTRO.md`
- Operational notes: `docs/README.md`
- First release checklist: `docs/FIRST_RELEASE_PLAN.md`

## Current Truth Source

- Executable code under `core/`, `agents/`, `verses/`, `orchestrator/`, `memory/`, `tools/`.
- Automated verification via `python -m pytest -q`.

## Why Prior Docs Were Removed

The previous docs set included planning and capability narratives that were not consistently aligned with current tested behavior.

They were removed during cleanup to avoid documentation drift.

## How To Work Safely

1. Change code.
2. Run `python -m pytest -q`.
3. Update top-level `readme.md` only when behavior actually changes.
4. Keep generated artifacts out of git.
