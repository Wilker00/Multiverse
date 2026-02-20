"""
tools/semantic_bridge.py

CLI utility to translate Golden DNA across verses using semantic projection.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.semantic_bridge import translate_dna


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dna_path", type=str, required=True)
    ap.add_argument("--target_verse", type=str, required=True)
    ap.add_argument("--source_verse", type=str, default=None)
    ap.add_argument("--out_path", type=str, default=None)
    ap.add_argument("--learned_bridge_enabled", action="store_true")
    ap.add_argument("--learned_bridge_model_path", type=str, default=None)
    ap.add_argument("--learned_bridge_score_weight", type=float, default=0.35)
    args = ap.parse_args()

    st = translate_dna(
        source_dna_path=args.source_dna_path,
        target_verse_name=args.target_verse,
        output_path=args.out_path,
        source_verse_name=args.source_verse,
        learned_bridge_enabled=bool(args.learned_bridge_enabled),
        learned_bridge_model_path=args.learned_bridge_model_path,
        learned_bridge_score_weight=float(args.learned_bridge_score_weight),
    )
    print("Semantic bridge complete")
    print(f"source_path     : {st.source_path}")
    print(f"output_path     : {st.output_path}")
    print(f"source_verse    : {st.source_verse_name}")
    print(f"target_verse    : {st.target_verse_name}")
    print(f"input_rows      : {st.input_rows}")
    print(f"translated_rows : {st.translated_rows}")
    print(f"dropped_rows    : {st.dropped_rows}")
    print(f"learned_enabled : {st.learned_bridge_enabled}")
    print(f"learned_model   : {st.learned_bridge_model_path}")
    print(f"learned_rows    : {st.learned_scored_rows}")


if __name__ == "__main__":
    main()
