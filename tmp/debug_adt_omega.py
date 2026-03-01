
import os
import sys
import json
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.types import AgentSpec, VerseSpec
from agents.transformer_agent import TransformerAgent
from verses.registry import create_verse, register_builtin

register_builtin()

def debug_adt():
    model_path = os.path.join(_PROJECT_ROOT, "models", "dt_generalist_v3_omega.pt")
    verse_name = "grid_world"
    
    vspec = VerseSpec(spec_version="v1", verse_name=verse_name, verse_version="0.1", seed=42)
    verse = create_verse(vspec)
    
    aspec = AgentSpec(
        spec_version="v1", policy_id="debug", policy_version="0.1", algo="adt",
        config={
            "model_path": model_path,
            "device": "cpu",
            "target_return": 1.0,
            "verse_name": verse_name,
            "context_len": 30
        }
    )
    agent = TransformerAgent(aspec, verse.observation_space, verse.action_space)
    
    rr = verse.reset()
    obs = rr.obs
    print(f"Initial Obs: {obs}")
    
    for i in range(20):
        out = agent.act(obs)
        action = int(out.action)
        
        sr = verse.step(action)
        print(f"Step {i:02d}: Action={action} | Obs({sr.obs['x']},{sr.obs['y']}) | Reward={sr.reward:.2f} | Done={sr.done}")
        obs = sr.obs
        if sr.done: break

if __name__ == "__main__":
    debug_adt()
