
import argparse
import os
import sys
import json
import time
from typing import Dict, Any, List

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from agents.transformer_agent import TransformerAgent
from verses.registry import create_verse, register_builtin

# Fixed targets for Omega model
_TARGET_MAP = {
    "line_world": 0.85, "cliff_world": -75.0, "grid_world": -0.4, "maze_world": -1.5,
    "warehouse_world": -0.6, "swamp_world": 0.5, "wind_master_world": 0.2,
    "escape_world": 0.3, "trade_world": 0.0, "factory_world": 15.0,
    "chess_world": 0.5, "go_world": 0.5, "uno_world": 0.0
}

def evaluate_with_memory_recall(
    model_path: str,
    verses: List[str],
    episodes: int = 5,
    max_steps: int = 80,
    recall: bool = True
) -> Dict[str, Any]:
    register_builtin()
    
    results = {}
    print(f"{'='*80}")
    print(f"EVALUATING MODEL: {os.path.basename(model_path)}")
    print(f"Memory Recall: {recall}")
    print(f"{'='*80}")
    
    for vname in verses:
        t0 = time.time()
        print(f"  [{vname}] Rolling out...", end="", flush=True)
        
        vspec = VerseSpec(spec_version="v1", verse_name=vname, verse_version="1", seed=42)
        verse = create_verse(vspec)
        
        target = _TARGET_MAP.get(vname, 1.0)
        aspec = AgentSpec(
            spec_version="v1", policy_id="eval", policy_version="0.1", algo="adt",
            config={
                "model_path": model_path,
                "device": "cpu",
                "target_return": target,
                "recall_enabled": recall,
                "recall_frequency": 5 if recall else 0,
                "recall_top_k": 3,
                "verse_name": vname,
                "context_len": 30
            }
        )
        agent = TransformerAgent(aspec, verse.observation_space, verse.action_space)
        
        successes = 0
        total_ret = 0.0
        
        for ep in range(episodes):
            rr = verse.reset()
            obs = rr.obs
            context_ret = 0.0
            done = False
            
            for step in range(max_steps):
                # Manual memory query logic if Orchestrator is missing (or just use Agent's internal logic)
                # Actually, TransformerAgent.act will call find_similar if recall_enabled=True?
                # Let's check that. 
                # (Checked earlier: TransformerAgent expects 'hint')
                # So we must issue the query here like in run_adt_dagger.
                
                hint = None
                if recall and hasattr(agent, "memory_query_request"):
                    try:
                        req = agent.memory_query_request(obs=obs, step_idx=step)
                        from memory.central_repository import CentralMemoryConfig, find_similar
                        mem_cfg = CentralMemoryConfig(root_dir="central_memory")
                        matches = find_similar(
                            obs=req.get("query_obs", obs),
                            cfg=mem_cfg,
                            top_k=int(req.get("top_k", 3)),
                            verse_name=req.get("verse_name"),
                            trajectory_window=20
                        )
                        match_rows = []
                        for m in matches:
                            match_rows.append({
                                "score": float(m.score),
                                "action": m.action,
                                "trajectory": m.trajectory,
                                "verse_name": m.verse_name
                            })
                        hint = {"memory_recall": {"matches": match_rows}}
                    except Exception:
                        pass
                
                if hint:
                    out = agent.act_with_hint(obs, hint)
                else:
                    out = agent.act(obs)
                    
                sr = verse.step(out.action)
                context_ret += float(sr.reward)
                obs = sr.obs
                if sr.done or sr.truncated:
                    if sr.info.get("reached_goal") or sr.info.get("success"):
                        successes += 1
                    break
            total_ret += context_ret
            
        elapsed = time.time() - t0
        res = {
            "success_rate": successes / episodes,
            "mean_return": total_ret / episodes,
            "elapsed": elapsed
        }
        results[vname] = res
        print(f" Success={res['success_rate']*100:.1f}% | AvgRet={res['mean_return']:.2f} ({elapsed:.1f}s)")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--verses", type=str, default="grid_world,warehouse_world,swamp_world,line_world")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no_memory", action="store_true")
    args = parser.parse_args()
    
    v_list = [v.strip() for v in args.verses.split(",")]
    evaluate_with_memory_recall(
        model_path=args.model,
        verses=v_list,
        episodes=args.episodes,
        recall=not args.no_memory
    )
