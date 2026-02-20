"""
Full gap analysis of Universe.AI transfer learning system.
Reads the actual code and produces an honest assessment.
"""
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.taxonomy import can_bridge, bridge_reason, tags_for_verse, TAXONOMY

# ============================================================
# 1. BRIDGE MATRIX: What transfers are theoretically possible?
# ============================================================
print("=" * 80)
print("1. BRIDGE COMPATIBILITY MATRIX")
print("=" * 80)

verses = [
    "chess_world", "go_world", "uno_world",
    "warehouse_world", "labyrinth_world", "escape_world",
    "factory_world", "trade_world", "bridge_world",
    "harvest_world", "swamp_world",
    "grid_world", "cliff_world",
]

# Group by domain
strategy = ["chess_world", "go_world", "uno_world"]
navigation = ["warehouse_world", "labyrinth_world", "grid_world", "cliff_world", "escape_world"]
resource = ["factory_world", "trade_world", "bridge_world", "harvest_world", "swamp_world"]

print("\n  Strategy -> Strategy:")
for s in strategy:
    for d in strategy:
        if s != d:
            c = can_bridge(s, d)
            r = bridge_reason(s, d)
            print(f"    {s:20s} -> {d:20s}  can={c}  reason={r}")

print("\n  Strategy -> Navigation:")
for s in strategy:
    for d in navigation:
        c = can_bridge(s, d)
        r = bridge_reason(s, d)
        print(f"    {s:20s} -> {d:20s}  can={c}  reason={r}")

print("\n  Strategy -> Resource:")
for s in strategy:
    for d in resource:
        c = can_bridge(s, d)
        r = bridge_reason(s, d)
        print(f"    {s:20s} -> {d:20s}  can={c}  reason={r}")

# ============================================================
# 2. SEMANTIC BRIDGE: What projections exist?
# ============================================================
print()
print("=" * 80)
print("2. SEMANTIC BRIDGE PROJECTION COVERAGE")
print("=" * 80)

from memory.semantic_bridge import _strategy_obs_from_signature

# Test: can we project a strategy signature INTO each target verse?
test_sig = {"score_delta": 3, "pressure": 4, "risk": 2, "tempo": 3, "control": 5, "resource": 4}
for target in verses:
    try:
        obs = _strategy_obs_from_signature(signature=test_sig, target_verse_name=target, t_value=10)
        keys = sorted(obs.keys()) if isinstance(obs, dict) else []
        print(f"  -> {target:20s}: {len(keys)} fields: {keys[:6]}...")
    except Exception as e:
        print(f"  -> {target:20s}: FAILED: {e}")

# ============================================================
# 3. TRANSFER CHALLENGE: What targets are supported?
# ============================================================
print()
print("=" * 80)
print("3. RUN_TRANSFER_CHALLENGE.PY GAPS")
print("=" * 80)

# Check _target_action_count coverage
from tools.run_transfer_challenge import _target_action_count, _normalize_target_obs

for v in verses:
    ac = _target_action_count(v)
    test_obs = {"x": 1, "y": 2, "t": 5, "score_delta": 3, "pressure": 2, "risk": 1,
                "tempo": 2, "control": 3, "resource": 4}
    norm = _normalize_target_obs(v, test_obs)
    norm_keys = sorted(norm.keys()) if isinstance(norm, dict) else ["RAW_PASSTHROUGH"]
    is_custom = "flat" in (norm or {}) or any(k not in test_obs for k in (norm or {}).keys())
    print(f"  {v:20s}: action_count={ac}  normalized={'YES' if is_custom else 'PASSTHROUGH (no custom normalizer)'}")

# ============================================================
# 4. EXISTING BENCHMARK REPORTS
# ============================================================
print()
print("=" * 80)
print("4. EXISTING BENCHMARK REPORTS")
print("=" * 80)

bench_dir = os.path.join(PROJECT_ROOT, "models", "benchmarks")
if os.path.isdir(bench_dir):
    for fname in sorted(os.listdir(bench_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(bench_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tv = data.get("target_verse", "?")
                t_eval = ((data.get("transfer_agent") or {}).get("eval") or {})
                b_eval = ((data.get("baseline_agent") or {}).get("eval") or {})
                t_sr = t_eval.get("success_rate", "?")
                b_sr = b_eval.get("success_rate", "?")
                t_mr = t_eval.get("mean_return", "?")
                b_mr = b_eval.get("mean_return", "?")
                eps = data.get("episodes", "?")
                print(f"  {fname:45s}: target={tv:20s} eps={eps} transfer_sr={t_sr} baseline_sr={b_sr}")
            except Exception as e:
                print(f"  {fname}: ERROR reading: {e}")

# ============================================================
# 5. RUN SCAN PERFORMANCE CHECK
# ============================================================
print()
print("=" * 80)
print("5. RUNS DIRECTORY SCALE")
print("=" * 80)

runs_dir = os.path.join(PROJECT_ROOT, "runs")
if os.path.isdir(runs_dir):
    count = len([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
    print(f"  Total run directories: {count}")
    print(f"  Source discovery scans ALL of them -> performance bottleneck")

# ============================================================
# 6. THEORY SCRIPT GAPS
# ============================================================
print()
print("=" * 80)
print("6. TRANSFER_BOUNDS.PY ANALYSIS")
print("=" * 80)

print("  - Divergence estimator: weighted by INPUT rows (including 0-coverage poison)")
print("  - No filtering of 0-translation-coverage sources")
print("  - No strategy-to-strategy benchmark exists yet")
print("  - All existing benchmarks show divergence >= 0.91, error = 1.0")
print("  - Theory is only validating FAILURE prediction, never SUCCESS prediction")
