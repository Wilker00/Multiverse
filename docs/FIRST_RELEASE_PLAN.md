# First Release Commit Plan

## Goal
Publish a clean, reproducible, source-first repository that excludes large/generated runtime artifacts.

## Release Branch Strategy
1. Create a dedicated release branch from your current baseline.
2. Keep one commit for repo hygiene/docs, one commit for source code, and optional follow-up commits for samples.
3. Avoid mixing generated artifacts into source commits.

## Commit 1: Governance + Hygiene
- Add `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`.
- Tighten `.gitignore` for runtime artifacts, large generated datasets, checkpoints, and local metadata.
- Add/refresh docs links in `readme.md`.

## Commit 2: Source-Only Baseline
- Include only executable code and tests:
  - `core/`, `agents/`, `verses/`, `orchestrator/`, `memory/`, `tools/`, `tests/`, `command_center/`
- Keep required manifests/configs:
  - `pytest.ini`, `requirements_scale.txt`, `docker-compose.scale.yml`, `Dockerfile.worker`
- Exclude generated data/report outputs:
  - `runs*`, `central_memory*`, profiler dumps, benchmark outputs, model artifacts, huge datasets.

## Commit 3 (Optional): Minimal Public Samples
- Add only small sample datasets if needed for docs or smoke tests.
- Keep each sample file small and clearly marked as example data.

## Acceptance Criteria Before Publish
- `python -m pytest -q` passes on the release branch.
- `readme.md` setup/test instructions run as documented.
- No tracked files violate `.gitignore` policy.
- No single file exceeds your host limits (for GitHub, avoid large binaries in git history).
- Repo includes license, contribution guide, and security policy.

## Suggested Pre-Publish Commands
- Check tracked ignored files:
  - `git ls-files -ci --exclude-standard`
- Inspect largest tracked files:
  - `git ls-files | ForEach-Object { if (Test-Path $_) { Get-Item $_ } } | Sort-Object Length -Descending | Select-Object -First 50 FullName,Length`
- Validate test suite:
  - `python -m pytest -q`
