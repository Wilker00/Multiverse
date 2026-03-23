"""
integrations/local_visual_sim.py

Built-in lightweight simulator session for Multiverse verses.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.agent_base import RandomAgent
from core.types import AgentSpec, JSONValue, VerseSpec
from verses.registry import create_verse, register_builtin


class LocalVisualSim:
    def __init__(self, verse_spec: VerseSpec):
        register_builtin()
        self.verse_spec = verse_spec
        self.verse = create_verse(verse_spec)

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.verse.seed(int(seed))
        rr = self.verse.reset()
        return {
            "obs": rr.obs,
            "info": dict(rr.info or {}),
            "frame": self.render_frame(),
        }

    def step(self, action: JSONValue) -> Dict[str, Any]:
        sr = self.verse.step(action)
        return {
            "obs": sr.obs,
            "reward": float(sr.reward),
            "done": bool(sr.done),
            "truncated": bool(sr.truncated),
            "info": dict(sr.info or {}),
            "frame": self.render_frame(),
        }

    def render_frame(self, mode: str = "ansi") -> Any:
        frame = self.verse.render(mode=mode)
        if frame is not None:
            return frame
        rr = self.verse.export_state() if hasattr(self.verse, "export_state") else {}
        return rr if isinstance(rr, dict) else {}

    def close(self) -> None:
        self.verse.close()


def preview_local_visual_sim(
    *,
    verse_name: str,
    verse_params: Dict[str, Any],
    episodes: int,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    spec = VerseSpec(
        spec_version="v1",
        verse_name=str(verse_name),
        verse_version="0.1",
        seed=int(seed),
        tags=["sim_preview", "local_visual"],
        params=dict(verse_params),
    )
    sim = LocalVisualSim(spec)
    agent = RandomAgent(
        spec=AgentSpec(
            spec_version="v1",
            policy_id="random_preview",
            policy_version="0.0",
            algo="random",
            seed=int(seed),
        ),
        observation_space=sim.verse.observation_space,
        action_space=sim.verse.action_space,
    )
    episode_rows: List[Dict[str, Any]] = []
    try:
        for ep in range(max(1, int(episodes))):
            ep_seed = int(seed) + ep
            sim.verse.seed(ep_seed)
            agent.seed(ep_seed)
            reset = sim.reset()
            frames: List[str] = [str(reset.get("frame", ""))]
            total_reward = 0.0
            done = False
            truncated = False
            steps_taken = 0
            obs = reset.get("obs")
            while steps_taken < int(max_steps) and not (done or truncated):
                ar = agent.act(obs)
                step = sim.step(ar.action)
                obs = step.get("obs")
                done = bool(step.get("done", False))
                truncated = bool(step.get("truncated", False))
                total_reward += float(step.get("reward", 0.0))
                steps_taken += 1
                frames.append(str(step.get("frame", "")))
            episode_rows.append(
                {
                    "episode": int(ep + 1),
                    "seed": int(ep_seed),
                    "steps": int(steps_taken),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "return_sum": float(total_reward),
                    "initial_frame": frames[0] if frames else "",
                    "final_frame": frames[-1] if frames else "",
                    "frames": frames,
                }
            )
    finally:
        agent.close()
        sim.close()

    return {
        "provider_id": "multiverse_local",
        "verse_name": str(verse_name),
        "episodes": episode_rows,
    }
