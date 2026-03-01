"""Quick test of new modules"""
from tools.validation_stats import compute_validation_stats, compute_rate_stats
from tools.update_centroid import load_high_value_data, train_centroid_policy
import tempfile
import json
import os

print("=" * 60)
print("Testing validation_stats module...")
print("=" * 60)

stats = compute_validation_stats([1.0, 2.0, 3.0, 4.0], min_detectable_delta=0.5)
print(f"✓ compute_validation_stats works")
print(f"  current_n: {stats['current_n']} (expected 4)")
print(f"  mean: {stats['mean']} (expected 2.5)")
assert stats['current_n'] == 4
assert abs(stats['mean'] - 2.5) < 0.01

rate_stats = compute_rate_stats([1, 0, 1, 1, 0], min_detectable_delta=0.1)
print(f"✓ compute_rate_stats works")
print(f"  current_n: {rate_stats['current_n']} (expected 5)")
print(f"  rate: {rate_stats['rate']} (expected 0.6)")
assert rate_stats['current_n'] == 5
assert abs(rate_stats['rate'] - 0.6) < 0.01

print("\n" + "=" * 60)
print("Testing update_centroid module...")
print("=" * 60)

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
    print(f"✓ load_high_value_data works")
    print(f"  loaded {len(data)} episodes (expected 3)")
    assert len(data) == 3

    out_path = os.path.join(td, "centroid_policy.json")
    metrics = train_centroid_policy(data=data, model_path=out_path)
    print(f"✓ train_centroid_policy works")
    print(f"  saved: {metrics['saved']}")
    print(f"  num_episodes: {metrics['num_episodes']}")
    print(f"  num_states: {metrics['num_states']}")
    assert metrics['saved']
    assert os.path.isfile(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    print(f"  format: {artifact['format']} (expected 'centroid_policy_v1')")
    assert artifact['format'] == 'centroid_policy_v1'

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)

