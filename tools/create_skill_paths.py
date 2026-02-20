"""
tools/create_skill_paths.py

An offline tool to analyze DNA logs and create frozen, reusable Skill Paths.
"""

import argparse
import os

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from policies.skill_path import SkillPathConfig, create_skill_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dna_log", type=str, required=True, help="Path to the dna_good.jsonl file to process.")
    ap.add_argument("--skill_id", type=str, required=True, help="A unique name for the skill being created (e.g., 'navigate_to_exit').")
    ap.add_argument("--skill_library_dir", type=str, default="skills", help="Directory to save the frozen skill paths.")
    ap.add_argument("--min_advantage", type=float, default=1.0, help="Minimum advantage for DNA to be included in skill training.")
    # New argument to associate tags with the created skill
    ap.add_argument("--tags", nargs='+', default=[], help="A list of tags describing the skill (e.g., navigation, manipulation).")
    args = ap.parse_args()

    if not args.tags:
        raise ValueError("At least one tag must be provided with the --tags argument.")

    config = SkillPathConfig(
        dna_log_path=args.dna_log,
        skill_id=args.skill_id,
        source_verse_tags=args.tags,
        min_advantage=args.min_advantage,
    )

    try:
        skill_path = create_skill_path(config)
        skill_path.save(args.skill_library_dir)
        print(f"\nSuccessfully created and saved skill '{args.skill_id}' with tags {args.tags}.")

    except ValueError as e:
        print(f"\nError creating skill path: {e}")

if __name__ == "__main__":
    main()




