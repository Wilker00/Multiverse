"""
memory/market_analyzer.py

Aggregates statistics across the entire Knowledge Market (runs/ directory).
Used by recursive self-improvement loops to identify gaps and mentors.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class VerseStats:
    verse_name: str
    avg_return: float
    success_rate: float
    total_episodes: int
    top_policy: str
    top_strategic_signature: Optional[Dict[str, int]] = None

def analyze_market(runs_root: str = "runs") -> Dict[str, VerseStats]:
    market_stats = {}
    
    if not os.path.isdir(runs_root):
        return {}
        
    # Mapping: verse_name -> [returns, successes, policy_names, signatures]
    raw_data = {}
    
    for run_id in os.listdir(runs_root):
        run_path = os.path.join(runs_root, run_id)
        index_path = os.path.join(run_path, "episodes.jsonl")
        
        if not os.path.isfile(index_path):
            continue
            
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ep = json.loads(line)
                    v_name = ep["verse_name"]
                    ret = ep.get("return_sum", 0.0)
                    success = ep.get("reached_goal", False)
                    policy = ep.get("policy_id", "unknown")
                    sig = ep.get("strategic_signature")
                    
                    if v_name not in raw_data:
                        raw_data[v_name] = {"returns": [], "successes": [], "policies": {}, "signatures": []}
                    
                    raw_data[v_name]["returns"].append(float(ret))
                    raw_data[v_name]["successes"].append(1.0 if success else 0.0)
                    raw_data[v_name]["policies"][policy] = raw_data[v_name]["policies"].get(policy, 0) + 1
                    if sig:
                        raw_data[v_name]["signatures"].append(sig)
                except Exception:
                    continue
                    
    for v_name, data in raw_data.items():
        avg_ret = np.mean(data["returns"])
        succ_rate = np.mean(data["successes"])
        total = len(data["returns"])
        top_p = max(data["policies"], key=data["policies"].get)
        
        # Heuristic for top strategic signature: average of signatures of successful episodes
        # (For now, just take the first one seen as a placeholder)
        top_sig = data["signatures"][0] if data["signatures"] else None
        
        market_stats[v_name] = VerseStats(
            verse_name=v_name,
            avg_return=float(avg_ret),
            success_rate=float(succ_rate),
            total_episodes=total,
            top_policy=top_p,
            top_strategic_signature=top_sig
        )
        
    return market_stats

if __name__ == "__main__":
    stats = analyze_market()
    for v, s in stats.items():
        print(f"{v:20} | Success: {s.success_rate:.2f} | Episodes: {s.total_episodes} | Best: {s.top_policy}")
