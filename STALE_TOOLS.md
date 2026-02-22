# Stale Tools

These scripts have been identified as stale, one-off artifacts, or superseded.
They are kept for reference but should not be relied on.

| Script | Reason |
|---|---|
| `analysis.py` | Generic analysis script — purpose unclear |
| `bom_hygiene_scan.py` | One-off BOM hygiene scan — maintenance complete |
| `build_labyrinth_recovery_dna.py` | One-off DNA extraction script |
| `create_skill_paths.py` | Skill path creator — not wired to any pipeline |
| `dna_extract.py` | Superseded by _extract_success_dna_from_events in run_transfer_challenge |
| `eval_harness.py` | Old eval harness — superseded by verse_eval.py |
| `evolve_policy.py` | Early evolutionary policy experiment, not integrated |
| `find_bom.py` | One-off BOM character finder — maintenance complete |
| `generate_curriculum.py` | Superseded by curriculum_controller.py |
| `generate_warehouse_planner_dataset.py` | One-off dataset generator |
| `ingest_warehouse_expert.py` | One-off ingest script — superseded by build_warehouse_expert |
| `profile_api.py` | Generated api.prof — profiling artifact, superseded |
| `remove_bom.py` | One-off BOM remover — maintenance complete |
| `smoke_v2_verses.py` | Superseded by validate_all_verses.py |
| `update_centroid.py` | One-off centroid update script |
| `validation_stats.py` | Old validation stats — superseded by verse_eval.py |
| `value_baseline.py` | One-off baseline value script |

## How to handle

1. Review each script before deleting
2. If superseded, verify the replacement works first
3. Delete with: `Remove-Item tools/<script>.py`
