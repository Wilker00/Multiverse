# Contributing

## Scope
This repository prioritizes factual behavior over aspirational claims. Keep changes aligned with tested runtime behavior.

## Local Setup
1. Use Python 3.12.
2. Create and activate a virtual environment.
3. Install dependencies needed for your change.
4. Run tests with:
   - `python -m pytest -q`

## Coding Rules
- Keep changes focused and minimal.
- Update docs only when behavior actually changes.
- Avoid checking in generated runtime artifacts.
- Prefer deterministic seeds for tests and benchmarks.

## Tests
- Add or update tests in `tests/test_*.py` for behavior changes.
- Run at least targeted tests for touched modules.
- Run full suite before opening a release-oriented PR.

## Pull Requests
- Include a concise summary of what changed and why.
- Include exact commands run for validation.
- Call out risks, limitations, or follow-ups explicitly.

## Commit Hygiene
- Do not commit local runtime outputs (`runs*`, `central_memory*`, tuning snapshots, profiler dumps, large checkpoints/datasets).
- Keep commit history readable and logically grouped.
