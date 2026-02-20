"""Read and display the transfer_chess_to_go.json report."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("models/benchmarks/transfer_chess_to_go.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("target_verse:", data.get("target_verse"))
print("episodes:", data.get("episodes"))
print("seed:", data.get("seed"))

t = (data.get("transfer_agent") or {}).get("eval") or {}
b = (data.get("baseline_agent") or {}).get("eval") or {}

print()
print("TRANSFER AGENT (Chess DNA -> Go):")
print("  success_rate:", t.get("success_rate"))
print("  mean_return:", t.get("mean_return"))
print("  hazard_per_1k:", t.get("hazard_per_1k_steps"))

print()
print("BASELINE AGENT (From Scratch -> Go):")
print("  success_rate:", b.get("success_rate"))
print("  mean_return:", b.get("mean_return"))
print("  hazard_per_1k:", b.get("hazard_per_1k_steps"))

sp = data.get("speedup") or {}
print()
print("SPEEDUP:")
print("  transfer_first_passable:", sp.get("transfer_first_passable"))
print("  baseline_first_passable:", sp.get("baseline_first_passable"))
print("  speedup_ratio:", sp.get("speedup_ratio"))

bs = ((data.get("transfer_dataset") or {}).get("bridge_stats") or [])
print()
print("BRIDGE STATS:")
for s in bs:
    sv = s.get("source_verse", "?")
    inp = s.get("input_rows", 0)
    tr = s.get("translated_rows", 0)
    reason = s.get("bridge_reason", "?")
    cov = tr / inp * 100 if inp > 0 else 0
    print(f"  {sv}: input={inp} translated={tr} coverage={cov:.0f}% reason={reason}")

fs = ((data.get("transfer_dataset") or {}).get("filter_stats") or {})
print()
print("FILTER STATS:")
print("  input_rows:", fs.get("input_rows"))
print("  kept_rows:", fs.get("kept_rows"))
print("  dropped_invalid:", fs.get("dropped_invalid"))
print("  dropped_duplicates:", fs.get("dropped_duplicates"))

# Now run theory analysis
print()
print("=" * 60)
print("THEORY ANALYSIS (transfer_bounds)")
print("=" * 60)
from theory.transfer_bounds import analyze_transfer_report
result = analyze_transfer_report(data, report_path="models/benchmarks/transfer_chess_to_go.json")
for k, v in result.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"    {kk}: {vv}")
    else:
        print(f"  {k}: {v}")
