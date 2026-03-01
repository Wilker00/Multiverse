
import json
import os
from collections import Counter

def analyze_memory_success():
    path = "central_memory/memories.jsonl"
    if not os.path.isfile(path):
        print("Memory file not found.")
        return
    
    success_counts = Counter()
    total_counts = Counter()
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                v = str(row.get("verse_name", "")).strip().lower()
                total_counts[v] += 1
                
                info = row.get("info", {})
                if not isinstance(info, dict): info = {}
                
                # Success signals: reached_goal, episode_success, or reward > 0?
                # Usually reached_goal is the clear terminal signal.
                if info.get("reached_goal") or info.get("episode_success") or row.get("reward", 0.0) > 0.0:
                    success_counts[v] += 1
            except:
                continue
    
    print("Memory Statistics (Total Steps vs Success Steps):")
    for v in sorted(total_counts.keys()):
        print(f"  [{v:20s}] Total: {total_counts[v]:8d} | Success-like: {success_counts[v]:8d}")

if __name__ == "__main__":
    analyze_memory_success()
