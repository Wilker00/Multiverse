# Multiverse Learning Path

This repo currently has two kinds of learning material:

- one markdown tutorial
- seven Jupyter notebooks

This page maps the material that actually exists in the tree and gives a practical order for working through it.

## Recommended Order

1. [Tutorial 01: What is an Agent?](tutorials/01_hello_world.md)
2. [Notebook 01: Hello World](interactive/01_hello_world.ipynb)
3. [Notebook 02: Exploration](interactive/02_exploration.ipynb)
4. [Notebook 03: Value Learning](interactive/03_value_learning.ipynb)
5. [Notebook 04: Policy Optimization](interactive/04_policy_optimization.ipynb)
6. [Notebook 05: Memory Systems](interactive/05_memory_systems.ipynb)
7. [Notebook 06: Custom Verses](interactive/06_custom_verses.ipynb)
8. [Notebook 07: Advanced Training](interactive/07_advanced_training.ipynb)

## What Each Step Covers

- `Tutorial 01` explains the basic reinforcement-learning loop and uses `line_world` as the first concrete example.
- `Notebook 01` gives the same introduction in an executable format.
- `Notebook 02` focuses on exploration and exploitation.
- `Notebook 03` covers value learning.
- `Notebook 04` moves into policy optimization.
- `Notebook 05` introduces memory and transfer ideas in the Multiverse codebase.
- `Notebook 06` focuses on building custom verses.
- `Notebook 07` covers advanced training workflows.

## Verified Commands

These examples match the current CLI:

```bash
multiverse doctor
multiverse universe list
multiverse train --algo q --verse line_world --episodes 50
multiverse runs latest
multiverse sim list
multiverse sim preview --provider multiverse_local --verse line_world --episodes 1 --dry-run
```

## Supporting References

- [RL Concepts](RL_CONCEPTS.md)
- [Project Introduction](PROJECT_INTRO.md)
- [Quickstart](QUICKSTART.md)
- [Verse Documentation Index](verses/VERSE_DOCUMENTATION_INDEX.md)

## Notes

- The broader tutorial set described in an earlier local draft is not complete yet. Only `tutorials/01_hello_world.md` exists today.
- Keep future additions to this page aligned with files that are already present in `docs/`.
