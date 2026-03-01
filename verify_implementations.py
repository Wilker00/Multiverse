#!/usr/bin/env python
"""Verify that our implementations meet all test requirements"""

from tools.validation_stats import compute_validation_stats, compute_rate_stats
from tools.update_centroid import load_high_value_data, train_centroid_policy
import json
import os
import tempfile

print("=" * 70)
print("VALIDATION TEST - Verifying Implementation Requirements")
print("=" * 70)

# Test 1: compute_validation_stats with [1.0, 2.0, 3.0, 4.0]
print("\n✓ Test 1: compute_validation_stats([1.0, 2.0, 3.0, 4.0])")
stats = compute_validation_stats([1.0, 2.0, 3.0, 4.0], min_detectable_delta=0.5)
assert stats["current_n"] == 4, f"Expected current_n=4, got {stats['current_n']}"
assert abs(stats["mean"] - 2.5) < 0.000001, f"Expected mean≈2.5, got {stats['mean']}"
assert "ci_95" in stats, "Missing ci_95 key"
assert len(stats["ci_95"]) == 2, f"Expected ci_95 length 2, got {len(stats['ci_95'])}"
assert stats["required_n"] >= 1, f"Expected required_n>=1, got {stats['required_n']}"
print("  ✓ All assertions passed")

# Test 2: compute_validation_stats with empty list
print("\n✓ Test 2: compute_validation_stats([])")
stats = compute_validation_stats([])
assert stats["current_n"] == 0, f"Expected current_n=0, got {stats['current_n']}"
assert stats["is_sufficient"] == False, f"Expected is_sufficient=False, got {stats['is_sufficient']}"
print("  ✓ All assertions passed")

# Test 3: compute_rate_stats with [1, 0, 1, 1, 0]
print("\n✓ Test 3: compute_rate_stats([1, 0, 1, 1, 0])")
stats = compute_rate_stats([1, 0, 1, 1, 0], min_detectable_delta=0.1)
assert stats["current_n"] == 5, f"Expected current_n=5, got {stats['current_n']}"
assert abs(stats["rate"] - 0.6) < 0.000001, f"Expected rate≈0.6, got {stats['rate']}"
assert "ci_95" in stats, "Missing ci_95 key"
print("  ✓ All assertions passed")

# Test 4: load_high_value_data and train_centroid_policy
print("\n✓ Test 4: load_high_value_data and train_centroid_policy")
with tempfile.TemporaryDirectory() as td:
    runs_root = os.path.join(td, "runs")
    run_dir = os.path.join(runs_root, "run_1")
    os.makedirs(run_dir, exist_ok=True)

    dna_path = os.path.join(run_dir, "dna_good.jsonl")
    with open(dna_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"obs": {"x": 0}, "action": 0, "advantage": 1.0}) + "\n")
        f.write("this is not json\n")
        f.write(json.dumps({"obs": {"x": 0}, "action": 1, "advantage": 4.0}) + "\n")
        f.write(json.dumps({"obs": {"x": 1}, "action": 1, "advantage": 2.0}) + "\n")
        f.write(json.dumps({"obs": {"x": 2}, "action": 0, "advantage": 0.1}) + "\n")

    data = load_high_value_data(runs_root=runs_root, min_advantage=0.5)
    assert len(data) == 3, f"Expected 3 episodes, got {len(data)}"
    print("  ✓ load_high_value_data works (loaded 3 episodes)")

    out_path = os.path.join(td, "centroid_policy.json")
    metrics = train_centroid_policy(data=data, model_path=out_path)
    assert metrics["saved"] == True, f"Expected saved=True, got {metrics['saved']}"
    assert os.path.isfile(out_path), f"Expected file to exist at {out_path}"
    print("  ✓ train_centroid_policy saved correctly")

    with open(out_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    assert artifact["format"] == "centroid_policy_v1", f"Expected format='centroid_policy_v1', got {artifact['format']}"
    assert artifact["default_action"] == 1, f"Expected default_action=1, got {artifact['default_action']}"
    assert float(artifact["default_action_confidence"]) >= 0.5, f"Expected confidence>=0.5"
    obs_entry = artifact["obs_policy"]['{"x":0}']
    assert obs_entry["action"] == 1, f"Expected action=1 for state {obs_entry}"
    assert float(obs_entry["confidence"]) > 0.5, f"Expected confidence>0.5"
    print("  ✓ train_centroid_policy output format correct")

# Test 5: train_centroid_policy with empty data
print("\n✓ Test 5: train_centroid_policy([], model_path)")
with tempfile.TemporaryDirectory() as td:
    out_path = os.path.join(td, "centroid_policy.json")
    metrics = train_centroid_policy(data=[], model_path=out_path)
    assert metrics["saved"] == False, f"Expected saved=False for empty data"
    assert metrics["reason"] == "no_data", f"Expected reason='no_data'"
    assert os.path.isfile(out_path) == False, f"Expected no file created"
    print("  ✓ All assertions passed")

print("\n" + "=" * 70)
print("✅ ALL IMPLEMENTATION REQUIREMENTS MET!")
print("=" * 70)
print("\nSummary:")
print("  ✓ validation_stats.compute_validation_stats - OK")
print("  ✓ validation_stats.compute_rate_stats - OK")
print("  ✓ update_centroid.load_high_value_data - OK")
print("  ✓ update_centroid.train_centroid_policy - OK")
print("  ✓ All test assertions pass")
print("  ✓ Ready for pytest execution")

