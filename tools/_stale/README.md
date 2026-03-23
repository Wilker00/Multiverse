# Stale Tools Archive

This directory contains scripts that have been identified as obsolete, one-off artifacts, or superseded by newer implementations. They are kept for reference but should not be used in current workflows.

## Why These Were Archived

- **Profiling artifacts**: One-off performance analysis tools
- **Superseded**: Newer implementations exist that provide the same functionality
- **Experimental**: Not integrated into the main pipeline
- **Data-specific**: Custom scripts for processing specific datasets or legacy formats
- **Incomplete**: Started but not finished, or missing critical dependencies

---

## Directory Contents

| Script | Reason | Alternative |
|--------|--------|-------------|
| `profile_api.py` | Profiling artifact - single-use | Use Python's built-in cProfile |
| `find_bom.py`, `remove_bom.py` | BOM (byte-order mark) hygiene - complete | No action needed; kept for reference |
| `bom_hygiene_scan.py` | One-off BOM scanner | No existing replacement |
| `dna_extract.py` | DNA extraction from old format | Use `tools/run_transfer_challenge.py` |
| `value_baseline.py` | One-off baseline computation | Use main trainer with --baseline flag |
| `evolve_policy.py` | Experimental policy evolution | Not integrated to pipeline |
| `create_skill_paths.py` | Skill path generator | Not wired to current agent system |
| `generate_curriculum.py` | Curriculum generation | Use `orchestrator/curriculum_controller.py` |
| `ingest_warehouse_expert.py` | One-off data ingest | Use general ingest utilities |
| `generate_warehouse_planner_dataset.py` | One-off dataset generation | Use `tools/dataset_generator.py` instead |
| `build_labyrinth_recovery_dna.py` | One-off DNA extraction for labyrinth | Use generic `dna_extract.py` |
| `analysis.py` | Generic analysis (unclear purpose) | Use specialized analysis scripts |
| `smoke_v2_verses.py` | Superseded by validate_all_verses.py | Use `tools/validate_all_verses.py` |

---

## Before Deleting

1. **Search for references**
   ```bash
   grep -r "stale/[script_name]" --include="*.py" --include="*.md"
   ```

2. **Check git history** (if needed)
   ```bash
   git log --all -S "[script_name]" --oneline
   ```

3. **Back up externally** if historically important

4. **Delete when ready**
   ```bash
   Remove-Item "tools/_stale/script_name.py"
   ```

---

## Recovery

If you need to restore a script from this archive:

```bash
# Check what's here
ls tools/_stale/

# View the script
cat tools/_stale/script_name.py

# Restore to tools/ if needed
Copy-Item tools/_stale/script_name.py tools/script_name.py
```

---

## Maintenance

This archive grows as new one-off scripts are identified. Periodically review:

- Are any scripts used by current experiments?
- Have better alternatives replaced them?
- Should they be fully deleted after 6 months in archive?

**Last cleaned**: 2026-03-22  
**Recommended review cycle**: Quarterly

---

## Questions?

Before deleting a script, check:
1. Is it referenced in any `.py` or `.md` files?
2. Does it appear in any experiment configurations?
3. Could it be useful as a reference for future work?

If unsure, keep it. Storage is cheap; lost knowledge is expensive.
