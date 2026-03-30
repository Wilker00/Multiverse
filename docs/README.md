# Multiverse Documentation

This directory contains the documentation that is present in this repository today.

The previous local rewrite of this file pointed at many files and commands that do not exist. This version is intentionally smaller and only links to material that is actually in the tree.

## Start Here

- [Project Introduction](PROJECT_INTRO.md)
- [Quickstart](QUICKSTART.md)
- [Setup](SETUP.md)
- [Configuration](CONFIGURATION.md)
- [YAML Configuration](YAML_CONFIGURATION.md)
- [Hot Reload](HOT_RELOAD.md)
- [Paper](PAPER.md)

## Learning

- [Learning Path](LEARNING_PATH.md)
- [RL Concepts](RL_CONCEPTS.md)
- [Tutorial 01: What is an Agent?](tutorials/01_hello_world.md)

Interactive notebooks:

- [01 Hello World](interactive/01_hello_world.ipynb)
- [02 Exploration](interactive/02_exploration.ipynb)
- [03 Value Learning](interactive/03_value_learning.ipynb)
- [04 Policy Optimization](interactive/04_policy_optimization.ipynb)
- [05 Memory Systems](interactive/05_memory_systems.ipynb)
- [06 Custom Verses](interactive/06_custom_verses.ipynb)
- [07 Advanced Training](interactive/07_advanced_training.ipynb)

## Architecture And Audits

- [Engineering Audit](ENGINEERING_AUDIT.md)
- [Memory System Strengthening Summary](MEMORY_SYSTEM_STRENGTHENING_SUMMARY.md)
- [Phase 2 Implementation Guide](PHASE2_IMPLEMENTATION_GUIDE.md)
- [Stale Tools](../STALE_TOOLS.md)

## Archived Notes

- [Cleanup Report](../archive/docs_history/CLEANUP_REPORT_20260322.md)
- [Session Improvements](../archive/docs_history/SESSION_IMPROVEMENTS.md)

## Verse Integration Notes

These verse docs now live under `docs/verses/`:

- [Verse Documentation Index](verses/VERSE_DOCUMENTATION_INDEX.md)
- [Verse Integration Analysis](verses/VERSE_INTEGRATION_ANALYSIS.md)
- [Verse Flow Diagrams](verses/VERSE_FLOW_DIAGRAMS.md)
- [Verse Quick Reference](verses/VERSE_QUICK_REFERENCE.md)

## Verified CLI Commands

These commands were checked against the current CLI surface:

```bash
multiverse doctor
multiverse universe list
multiverse train --algo q --verse line_world --episodes 50
multiverse runs latest
multiverse sim list
multiverse sim preview --provider multiverse_local --verse line_world --episodes 1 --dry-run
multiverse sim2real --dry-run
```

## Notes

- The `multiverse analyze ...` and `multiverse compare ...` commands are not part of the current top-level CLI.
- The lightweight simulator is exposed through `multiverse sim list` and `multiverse sim preview`.
- If you add new docs, keep links in this file restricted to files that actually exist in the repository.
