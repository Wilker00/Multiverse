
import json
import os
from collections import defaultdict

def extract_expert_ghosts(verse_name="warehouse_world", max_ghosts=2):
    path = "central_memory/memories.jsonl"
    if not os.path.isfile(path):
        print("Memory file not found.")
        return
    
    # 1. Group steps by episode_id for the target verse
    episodes = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if str(row.get("verse_name", "")).strip().lower() == verse_name:
                    ep_id = row.get("episode_id")
                    episodes[ep_id].append(row)
            except:
                continue
    
    # 2. Identify successful episodes
    success_episodes = []
    for ep_id, steps in episodes.items():
        # Sort by step_idx
        steps.sort(key=lambda x: int(x.get("step_idx", 0)))
        
        has_success = False
        terminal_reward = 0.0
        for s in steps:
            info = s.get("info", {})
            if not isinstance(info, dict): info = {}
            if info.get("reached_goal") or info.get("episode_success") or s.get("reward", 0.0) > 1.0:
                has_success = True
                terminal_reward = s.get("reward", 0.0)
                break
        
        if has_success:
            success_episodes.append({
                "ep_id": ep_id,
                "steps": steps,
                "final_ret": sum(float(s.get("reward", 0.0)) for s in steps),
                "length": len(steps)
            })
    
    # 3. Sort by return and length (prefer shorter successful paths)
    success_episodes.sort(key=lambda x: (-x["final_ret"], x["length"]))
    
    print(f"--- EXPERT GHOST ANALYSIS: {verse_name.upper()} ---")
    print(f"Found {len(success_episodes)} successful expert trajectories in memory.\n")
    
    for i, ghost in enumerate(success_episodes[:max_ghosts]):
        print(f"Ghost #{i+1} [ID: {ghost['ep_id'][:8]}...]")
        print(f"  Total Return: {ghost['final_ret']:.2f}")
        print(f"  Path Length:  {ghost['length']} steps")
        
        # Summarize the trajectory
        actions = [s.get("action") for s in ghost["steps"]]
        # Map actions to names if possible (Warehouse: 0=Up, 1=Down, 2=Left, 3=Right, 4=Pick)
        action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Pick"}
        action_names = [action_map.get(a, str(a)) for a in actions]
        
        print(f"  Strategy Signature: {' -> '.join(action_names[:10])} ... {' -> '.join(action_names[-5:])}")
        
        # Analyze start and end observations
        start_obs = ghost["steps"][0].get("obs")
        end_obs = ghost["steps"][-1].get("obs")
        
        if isinstance(start_obs, dict) and "agent_pos" in start_obs:
            print(f"  Start Pos: {start_obs['agent_pos']} | Target: {start_obs.get('target_pos')}")
        if isinstance(end_obs, dict) and "agent_pos" in end_obs:
             print(f"  End Pos:   {end_obs['agent_pos']}")
        print("-" * 40)

if __name__ == "__main__":
    import sys
    verse = sys.argv[1] if len(sys.argv) > 1 else "warehouse_world"
    extract_expert_ghosts(verse)
