"""
Diagnose why domain divergence is always ~0.91 and transfer always fails.
"""

bridge_stats = [
    {"source_verse": "grid_world",  "input_rows": 460,   "translated_rows": 460,  "source_kind": "success_events"},
    {"source_verse": "cliff_world", "input_rows": 12000, "translated_rows": 0,    "source_kind": "success_events"},
    {"source_verse": "cliff_world", "input_rows": 12000, "translated_rows": 0,    "source_kind": "success_events"},
    {"source_verse": "chess_world", "input_rows": 548,   "translated_rows": 548,  "source_kind": "dna_good"},
    {"source_verse": "chess_world", "input_rows": 522,   "translated_rows": 522,  "source_kind": "success_events"},
    {"source_verse": "go_world",    "input_rows": 65,    "translated_rows": 65,   "source_kind": "success_events"},
    {"source_verse": "go_world",    "input_rows": 429,   "translated_rows": 429,  "source_kind": "dna_good"},
    {"source_verse": "uno_world",   "input_rows": 247,   "translated_rows": 247,  "source_kind": "success_events"},
]

print("=" * 80)
print("DIVERGENCE DIAGNOSIS: Why every transfer fails")
print("=" * 80)

total_rows = 0.0
weighted_gap = 0.0
for row in bridge_stats:
    inp = float(row["input_rows"])
    tr = float(row["translated_rows"])
    coverage = tr / inp
    gap = 1.0 - coverage
    total_rows += inp
    weighted_gap += inp * gap
    marker = " <<<< POISON" if tr == 0 else ""
    print(
        f"  {row['source_verse']:20s}: "
        f"input={int(inp):6d}  translated={int(tr):6d}  "
        f"coverage={coverage:.0%}  gap_weight={inp * gap:8.0f}{marker}"
    )

divergence = weighted_gap / total_rows
print()
print(f"  Total input rows:     {total_rows:.0f}")
print(f"  Total weighted gap:   {weighted_gap:.0f}")
print(f"  DIVERGENCE (d_H):     {divergence:.4f}")
print()

# Show the problem
cliff_rows = 24000
cliff_pct = cliff_rows / total_rows * 100
good_rows = total_rows - cliff_rows
good_gap = weighted_gap - cliff_rows  # cliff contributes input_rows * 1.0 gap

print("=" * 80)
print("ROOT CAUSE")
print("=" * 80)
print(f"  cliff_world contributes {cliff_rows} rows with 0% translation coverage")
print(f"  That is {cliff_pct:.1f}% of ALL input rows")
print(f"  All other sources total {good_rows:.0f} rows with 100% coverage")
print()

if good_rows > 0:
    divergence_no_cliff = good_gap / good_rows
    print(f"  If we EXCLUDE cliff_world:")
    print(f"    Divergence = {divergence_no_cliff:.4f}")
    print(f"    Predicted error = {0.023 + divergence_no_cliff + 0.063:.4f}")
    print()

# Same-family transfer (strategy-to-strategy)
print("=" * 80)
print("STRATEGY-TO-STRATEGY (e.g. chess -> go)")
print("=" * 80)
strategy_stats = [
    {"source_verse": "chess_world", "input_rows": 548, "translated_rows": 548},
    {"source_verse": "chess_world", "input_rows": 522, "translated_rows": 522},
]
s_total = sum(r["input_rows"] for r in strategy_stats)
s_gap = sum(r["input_rows"] * (1.0 - r["translated_rows"] / r["input_rows"]) for r in strategy_stats)
s_div = s_gap / s_total if s_total > 0 else 1.0
print(f"  If we transfer ONLY chess DNA to go_world:")
print(f"    Input rows:  {s_total}")
print(f"    Divergence:  {s_div:.4f}")
print(f"    This is the BEST CASE for cross-domain transfer")
print()

print("=" * 80)
print("THE FIX (3 things needed)")
print("=" * 80)
print("""
  1. FILTER OUT POISON SOURCES
     cliff_world -> warehouse_world has 0% translation coverage.
     The Semantic Bridge correctly drops all 24,000 rows, but the
     divergence estimator still counts them as "attempted but failed".
     FIX: Don't include 0-coverage sources in the divergence calculation,
     OR filter them out before building the transfer dataset.

  2. RUN SAME-FAMILY TRANSFERS FIRST
     chess -> go, chess -> uno, go -> uno share the strategy signature.
     These should have ~0% divergence (100% translation coverage).
     The --target_verse was hard-locked to warehouse/labyrinth.
     FIX: Already done - removed the choices restriction.
     NEED: Actually run it and get real numbers.

  3. MORE EPISODES
     5 episodes x 50 steps = 250 total steps.
     The default is 120 episodes x 100 steps = 12,000 steps.
     Even good transfer data can't help in 250 steps.
     FIX: Run with --episodes 50+ for a real test.
""")
