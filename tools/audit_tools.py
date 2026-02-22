"""
tools/audit_tools.py

Audits the tools/ directory and classifies scripts as:
  - CORE      : critical to daily workflow, actively maintained
  - SUPPORT   : useful utilities, infrequently needed
  - RESEARCH  : experiment scripts, not guaranteed to work
  - STALE     : artifacts, one-off scripts, or superseded

Run:
    .venv312\\Scripts\\python.exe tools\\audit_tools.py [--fix]

With --fix: renames stale scripts with a _stale prefix and creates
a STALE_TOOLS.md summary file in the project root.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------
# Classification table
# ----------------------------------------------------------------------------
# Format: "filename": ("tier", "notes")

TOOL_CLASSIFICATION: Dict[str, Tuple[str, str]] = {
    # === CORE — the essential daily-use scripts ===
    "multiverse_cli.py":       ("CORE",     "Main CLI entrypoint — status/train/run/shell"),
    "train_agent.py":          ("CORE",     "Single-agent training loop"),
    "validate_all_verses.py":  ("CORE",     "Smoke-tests all 23 verses — CI gate"),
    "verse_eval.py":           ("CORE",     "Learning quality evaluation per verse"),
    "compare_runs.py":         ("CORE",     "Head-to-head run comparison"),
    "query_runs.py":           ("CORE",     "Browse run history"),
    "diagnose_run.py":         ("CORE",     "Deep-dive into a specific run"),
    "ci_gate.py":              ("CORE",     "Full CI quality gate"),
    "run_transfer_challenge.py": ("CORE",   "Transfer learning flagship experiment"),
    "validate_sf_transfer.py": ("CORE",     "Successor features transfer validation"),
    "train_distributed.py":    ("CORE",     "Distributed / PBT training"),
    "demo_maze.py":            ("CORE",     "Maze world demo (recently added)"),
    "render.py":               ("CORE",     "ANSI episode renderer"),
    "gym_like_viewer.py":      ("CORE",     "Interactive episode viewer"),

    # === SUPPORT — useful utilities, maintained ===
    "index_memory.py":         ("SUPPORT",  "Index a run into central memory"),
    "select_memory.py":        ("SUPPORT",  "Query central memory bank"),
    "agent_health_monitor.py": ("SUPPORT",  "Agent health diagnostics"),
    "deploy_agent.py":         ("SUPPORT",  "Promote/deploy a trained agent"),
    "promote_candidate.py":    ("SUPPORT",  "Promotion pipeline"),
    "run_benchmarks.py":       ("SUPPORT",  "Full benchmark suite"),
    "run_fixed_seed_benchmark.py": ("SUPPORT", "Reproducibility benchmark"),
    "run_paper_readiness_pack.py": ("SUPPORT", "Paper-readiness validation"),
    "run_autonomous_cycle.py": ("SUPPORT",  "Autonomous training loop"),
    "production_readiness_gate.py": ("SUPPORT", "Production readiness gate"),
    "marl_run.py":             ("SUPPORT",  "Multi-agent RL runner"),
    "mcts_cycle.py":           ("SUPPORT",  "MCTS training loop"),
    "neural_forensics.py":     ("SUPPORT",  "Agent neural behavior analysis"),
    "train_meta_transformer.py": ("SUPPORT","Meta-transformer training"),
    "universal_model_api.py":  ("SUPPORT",  "Universal model API server"),
    "build_warehouse_expert.py": ("SUPPORT","Build expert dataset for warehouse"),
    "build_labyrinth_expert.py": ("SUPPORT","Build expert dataset for labyrinth"),
    "build_cliff_safety_dataset.py": ("SUPPORT", "Cliff safety dataset builder"),
    "build_strategy_transfer.py": ("SUPPORT", "Strategy transfer dataset"),
    "validate_transfer.py":    ("SUPPORT",  "Transfer validation script"),
    "validate_hard_cliff_multiseed.py": ("SUPPORT", "Cliff multi-seed validation"),
    "run_cognitive_upgrade_kpi.py": ("SUPPORT", "Cognitive KPI tracker"),
    "cleanup_artifacts.py":    ("SUPPORT",  "Clean up run artifacts"),
    "knowledge_graph.py":      ("SUPPORT",  "Knowledge graph viewer"),
    "knowledge_market.py":     ("SUPPORT",  "Knowledge market interface"),
    "active_forgetting.py":    ("SUPPORT",  "Memory active forgetting"),
    "backfill_memory_metadata.py": ("SUPPORT", "Backfill memory metadata"),
    "temporal_decay.py":       ("SUPPORT",  "Memory temporal decay tool"),
    "scale_monitor.py":        ("SUPPORT",  "Training scale monitor"),
    "benchmark_parallel_configs.py": ("SUPPORT", "Parallel config benchmarks"),
    "benchmark_retrieval_ann.py": ("SUPPORT", "ANN retrieval benchmarks"),
    "rebenchmark_everything.py": ("SUPPORT", "Full rebenchmark script"),
    "sweep_teacher_remediation.py": ("SUPPORT", "Teacher remediation sweep"),
    "run_teacher_remediation.py": ("SUPPORT", "Teacher remediation runner"),
    "train_bridge.py":         ("SUPPORT",  "Semantic bridge training"),
    "visualize_bridge.py":     ("SUPPORT",  "Semantic bridge visualization"),
    "train_adt.py":            ("SUPPORT",  "Adversarial dynamics training"),
    "prep_adt_data.py":        ("SUPPORT",  "ADT data preparation"),
    "run_adt_dagger.py":       ("SUPPORT",  "ADT DAgger pipeline"),
    "train_selector.py":       ("SUPPORT",  "Agent selector training"),
    "retrain_selector_advanced.py": ("SUPPORT", "Advanced selector retraining"),
    "prepare_selector_data.py": ("SUPPORT", "Selector data preparation"),
    "audit_tools.py":          ("SUPPORT",  "This audit script"),

    # === RESEARCH — experimental, use with care ===
    "recursive_self_improvement.py": ("RESEARCH", "Experimental self-improvement loop"),
    "universal_model.py":      ("RESEARCH", "Universal model experiments"),
    "causal_distiller.py":     ("RESEARCH", "Causal distillation experiment"),
    "behavioral_surgeon.py":   ("RESEARCH", "Behavioral surgery experiment"),
    "behavioral_scorer.py":    ("RESEARCH", "Behavioral scoring metrics"),
    "failure_mode_classifier.py": ("RESEARCH", "Failure mode classification"),
    "tune_failure_constraints.py": ("RESEARCH", "Failure constraint tuning"),
    "optimize_cliff_success_under_safety.py": ("RESEARCH", "Safety-aware optimization"),
    "hard_env_tuning.py":      ("RESEARCH", "Hard environment tuning experiments"),
    "make_apprentice.py":      ("RESEARCH", "Apprentice agent creation"),
    "distill_policy.py":       ("RESEARCH", "Policy distillation"),
    "train_confidence_monitor.py": ("RESEARCH", "Confidence monitor training"),
    "retrain_confidence_monitor.py": ("RESEARCH", "Confidence monitor retraining"),
    "confidence_auditor.py":   ("RESEARCH", "Confidence score auditor"),
    "gateway_stress_test.py":  ("RESEARCH", "Gateway stress testing"),
    "scaffold_extension.py":   ("RESEARCH", "Scaffold extension experiments"),
    "similarity_canary.py":    ("RESEARCH", "Similarity metric canary test"),
    "refine_shields.py":       ("RESEARCH", "Safety shield refinement"),
    "vector_store_refiner.py": ("RESEARCH", "Vector store refinement"),
    "vae_demo.py":             ("RESEARCH", "VAE demonstration"),
    "cluster_memory.py":       ("RESEARCH", "Memory clustering experiment"),
    "uas_case_study.py":       ("RESEARCH", "UAS case study"),
    "benchmark_meta_stages.py": ("RESEARCH", "Meta-stage benchmarks"),
    "mcts_telemetry_dashboard.py": ("RESEARCH", "MCTS telemetry dashboard"),
    "run_cognitive_upgrade_kpi.py": ("RESEARCH", "Cognitive upgrade KPI"),
    "run_pipeline.py":         ("RESEARCH", "Experimental pipeline runner"),
    "universe_hub.py":         ("RESEARCH", "Universe hub experiments"),
    "semantic_bridge.py":      ("RESEARCH", "Semantic bridge experiments"),

    # === STALE — artifacts, one-offs, or superseded ===
    "profile_api.py":          ("STALE",    "Generated api.prof — profiling artifact, superseded"),
    "find_bom.py":             ("STALE",    "One-off BOM character finder — maintenance complete"),
    "remove_bom.py":           ("STALE",    "One-off BOM remover — maintenance complete"),
    "bom_hygiene_scan.py":     ("STALE",    "One-off BOM hygiene scan — maintenance complete"),
    "dna_extract.py":          ("STALE",    "Superseded by _extract_success_dna_from_events in run_transfer_challenge"),
    "value_baseline.py":       ("STALE",    "One-off baseline value script"),
    "evolve_policy.py":        ("STALE",    "Early evolutionary policy experiment, not integrated"),
    "create_skill_paths.py":   ("STALE",    "Skill path creator — not wired to any pipeline"),
    "generate_curriculum.py":  ("STALE",    "Superseded by curriculum_controller.py"),
    "ingest_warehouse_expert.py": ("STALE", "One-off ingest script — superseded by build_warehouse_expert"),
    "generate_warehouse_planner_dataset.py": ("STALE", "One-off dataset generator"),
    "build_labyrinth_recovery_dna.py": ("STALE", "One-off DNA extraction script"),
    "validation_stats.py":     ("STALE",    "Old validation stats — superseded by verse_eval.py"),
    "analysis.py":             ("STALE",    "Generic analysis script — purpose unclear"),
    "smoke_v2_verses.py":      ("STALE",    "Superseded by validate_all_verses.py"),
    "update_centroid.py":      ("STALE",    "One-off centroid update script"),
    "eval_harness.py":         ("STALE",    "Old eval harness — superseded by verse_eval.py"),
}

TIER_ORDER = ["CORE", "SUPPORT", "RESEARCH", "STALE"]
TIER_COLORS = {
    "CORE": "\033[32m",     # green
    "SUPPORT": "\033[36m",  # cyan
    "RESEARCH": "\033[33m", # yellow
    "STALE": "\033[31m",    # red
}
RESET = "\033[0m"


def audit_tools(fix: bool = False) -> None:
    tools_dir = PROJECT_ROOT / "tools"
    all_scripts = sorted(p.name for p in tools_dir.glob("*.py"))

    classified = {t: [] for t in TIER_ORDER}
    unclassified = []

    for script in all_scripts:
        if script in TOOL_CLASSIFICATION:
            tier, notes = TOOL_CLASSIFICATION[script]
            classified[tier].append((script, notes))
        else:
            unclassified.append(script)

    # Print report
    print("\n" + "=" * 70)
    print("  MULTIVERSE TOOLS AUDIT")
    print("=" * 70)
    for tier in TIER_ORDER:
        color = TIER_COLORS[tier]
        scripts = classified[tier]
        print(f"\n{color}[{tier}]{RESET}  ({len(scripts)} scripts)")
        for name, notes in scripts:
            print(f"  {name:<45} {notes}")

    if unclassified:
        print(f"\n\033[35m[UNCLASSIFIED]\033[0m  ({len(unclassified)} scripts — add to audit_tools.py)")
        for name in unclassified:
            print(f"  {name}")

    stale = classified["STALE"]
    print(f"\n{'=' * 70}")
    print(f"  Summary: {len(all_scripts)} total scripts")
    for tier in TIER_ORDER:
        print(f"    {tier:<12}: {len(classified[tier])}")
    if unclassified:
        print(f"    {'UNCLASSIFIED':<12}: {len(unclassified)}")
    print(f"{'=' * 70}")

    # Write STALE_TOOLS.md
    md_path = PROJECT_ROOT / "STALE_TOOLS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stale Tools\n\n")
        f.write("These scripts have been identified as stale, one-off artifacts, or superseded.\n")
        f.write("They are kept for reference but should not be relied on.\n\n")
        f.write("| Script | Reason |\n|---|---|\n")
        for name, notes in stale:
            f.write(f"| `{name}` | {notes} |\n")
        f.write("\n## How to handle\n\n")
        f.write("1. Review each script before deleting\n")
        f.write("2. If superseded, verify the replacement works first\n")
        f.write("3. Delete with: `Remove-Item tools/<script>.py`\n")
    print(f"\nWrote: {md_path}")

    if fix and stale:
        print(f"\nMoving {len(stale)} stale scripts to tools/_stale/ ...")
        stale_dir = tools_dir / "_stale"
        stale_dir.mkdir(exist_ok=True)
        moved = 0
        for name, _ in stale:
            src = tools_dir / name
            dst = stale_dir / name
            if src.exists():
                src.rename(dst)
                print(f"  moved: {name}")
                moved += 1
        print(f"Done. Moved {moved} scripts to tools/_stale/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and classify tools/ scripts")
    parser.add_argument("--fix", action="store_true",
                        help="Move stale scripts to tools/_stale/ directory")
    args = parser.parse_args()
    audit_tools(fix=args.fix)
