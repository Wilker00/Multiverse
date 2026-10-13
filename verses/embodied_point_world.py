"""
verses/embodied_point_world.py

Point-mass verse for embodied control.

This is still a Verse: agents, the trainer, and SafeExecutor do not change.
Physics is delegated to a backend so a GPU integrator can be plugged in later.
Isaac Lab remains the high-fidelity robotics lane (``isaac_warehouse_nav``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult
from integrations.physics.toy_backend import ToyPhysicsBackend
from integrations.physics.warp_backend import WarpPhysicsBackend
from integrations.physics.native_backend import NativePhysicsBackend
from integrations.physics.domain_randomization import DomainRandomizationSample, sample_domain_randomization


def _backend_for(name: str):
    key = str(name or "toy").strip().lower()
    if key in ("warp", "nvidia_warp"):
        return WarpPhysicsBackend()
    return ToyPhysicsBackend()


class EmbodiedPointWorldVerse:
    """
    Flat observation: [x, y, vx, vy, goal_x, goal_y].
    Discrete actions: 0 hold, 1 +x, 2 -x, 3 +y, 4 -y.
    """

    def __init__(self, spec: VerseSpec) -> None:
        tags = list(spec.tags)
        for tag in EmbodiedPointWorldFactory().tags:
            if tag not in tags:
                tags.append(tag)
        self.spec = dataclasses.replace(spec, tags=tags)
        params = dict(self.spec.params or {})
        self._base_goal_x = float(params.get("goal_x", 0.8))
        self._base_goal_y = float(params.get("goal_y", 0.8))
        self._goal_x = self._base_goal_x
        self._goal_y = self._base_goal_y
        self._domain_randomization = params.get("domain_randomization", False)
        self._randomization_sample = DomainRandomizationSample()
        self._goal_radius = max(0.05, float(params.get("goal_radius", 0.15)))
        self._max_steps = max(1, int(params.get("max_steps", 40)))
        self._step_penalty = float(params.get("step_penalty", -0.01))
        self._goal_reward = float(params.get("goal_reward", 1.0))
        provider = str(params.get("sim_provider", "python")).strip().lower()
        if provider == "multiverse_native":
            self._backend = NativePhysicsBackend(
                "embodied_point_world", worker_count=int(params.get("sim_workers", 0) or 0)
            )
        elif provider == "python":
            self._backend = _backend_for(str(params.get("backend", "toy")))
        else:
            raise ValueError(f"Unknown sim_provider {provider!r}; expected 'python' or 'multiverse_native'")
        self._body = self._backend.reset(seed=self.spec.seed)
        self._t = 0
        self._done = False
        self._trail: List[Tuple[float, float]] = []
        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="vector",
            shape=(6,),
            dtype="float32",
            notes="x, y, vx, vy, goal_x, goal_y",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=5,
            notes="0 hold, 1 +x, 2 -x, 3 +y, 4 -y",
        )

    def seed(self, seed: Optional[int]) -> None:
        self.spec = dataclasses.replace(self.spec, seed=seed)

    @property
    def position(self) -> Tuple[float, float]:
        return (float(self._body.get("x", 0.0)), float(self._body.get("y", 0.0)))

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self._body.get("vx", 0.0)), float(self._body.get("vy", 0.0)))

    @property
    def goal(self) -> Tuple[float, float]:
        return (self._goal_x, self._goal_y)

    @property
    def trail(self) -> List[Tuple[float, float]]:
        return list(self._trail)

    def _obs(self) -> List[float]:
        return [
            float(self._body.get("x", 0.0)),
            float(self._body.get("y", 0.0)),
            float(self._body.get("vx", 0.0)),
            float(self._body.get("vy", 0.0)),
            float(self._goal_x),
            float(self._goal_y),
        ]

    def _distance(self) -> float:
        dx = float(self._body.get("x", 0.0)) - self._goal_x
        dy = float(self._body.get("y", 0.0)) - self._goal_y
        return float((dx * dx + dy * dy) ** 0.5)

    def reset(self) -> ResetResult:
        self._randomization_sample = sample_domain_randomization(self.spec.seed, self._domain_randomization)
        self._goal_x = self._base_goal_x + self._randomization_sample.goal_offset[0]
        self._goal_y = self._base_goal_y + self._randomization_sample.goal_offset[1]
        if isinstance(self._backend, NativePhysicsBackend):
            self._backend.control_scale = self._randomization_sample.actuator_gain
        self._body = self._backend.reset(seed=self.spec.seed)
        self._t = 0
        self._done = False
        self._trail = [self.position]
        return ResetResult(obs=self._obs(), info={"distance": self._distance(), "backend": type(self._backend).__name__})

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(obs=self._obs(), reward=0.0, done=True, truncated=False, info={"warning": "step after done"})
        self._body = self._backend.step(self._body, int(action))
        self._t += 1
        self._trail.append(self.position)
        if len(self._trail) > 40:
            self._trail.pop(0)

        distance = self._distance()
        reached = distance <= self._goal_radius
        truncated = (not reached) and self._t >= self._max_steps
        reward = self._goal_reward if reached else self._step_penalty
        self._done = bool(reached or truncated)
        return StepResult(
            obs=self._obs(),
            reward=float(reward),
            done=bool(reached),
            truncated=bool(truncated),
            info={"distance": distance, "reached_goal": bool(reached), "t": int(self._t)},
        )

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode == "ansi":
            return (
                f"point ({self._body.get('x', 0.0):.2f},{self._body.get('y', 0.0):.2f}) "
                f"goal ({self._goal_x:.2f},{self._goal_y:.2f})"
            )
        elif mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_rgb_array(self, width: int = 128, height: int = 128) -> np.ndarray:
        """Rasterize a top-down view to a numpy RGB array."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Background space dark
        canvas[:] = [14, 16, 28]

        cx, cy = width // 2, height // 2
        # Bounds are roughly [-1.2, 1.2] mapped to canvas
        scale = min(width, height) * 0.38

        def to_screen(x: float, y: float) -> Tuple[int, int]:
            px = int(cx + x * scale)
            py = int(cy - y * scale)
            return px, py

        # Draw arena boundary box [-1, 1]
        p_min = to_screen(-1.0, 1.0)
        p_max = to_screen(1.0, -1.0)
        x0, y0 = max(0, min(p_min[0], p_max[0])), max(0, min(p_min[1], p_max[1]))
        x1, y1 = min(width - 1, max(p_min[0], p_max[0])), min(height - 1, max(p_min[1], p_max[1]))
        canvas[y0:y1+1, x0] = [40, 60, 90]
        canvas[y0:y1+1, x1] = [40, 60, 90]
        canvas[y0, x0:x1+1] = [40, 60, 90]
        canvas[y1, x0:x1+1] = [40, 60, 90]

        # Draw goal zone
        gx, gy = to_screen(self._goal_x, self._goal_y)
        g_rad = max(2, int(self._goal_radius * scale))
        for dy in range(-g_rad, g_rad + 1):
            for dx in range(-g_rad, g_rad + 1):
                if dx * dx + dy * dy <= g_rad * g_rad:
                    px, py = gx + dx, gy + dy
                    if 0 <= px < width and 0 <= py < height:
                        canvas[py, px] = [20, 200, 80]

        # Draw trajectory trail
        for tx, ty in self._trail:
            px, py = to_screen(tx, ty)
            if 0 <= px < width and 0 <= py < height:
                canvas[py, px] = [0, 140, 220]

        # Draw rover / point mass
        ax, ay = to_screen(float(self._body.get("x", 0.0)), float(self._body.get("y", 0.0)))
        a_rad = 3
        for dy in range(-a_rad, a_rad + 1):
            for dx in range(-a_rad, a_rad + 1):
                if dx * dx + dy * dy <= a_rad * a_rad:
                    px, py = ax + dx, ay + dy
                    if 0 <= px < width and 0 <= py < height:
                        canvas[py, px] = [255, 210, 30]

        return canvas

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "body": dict(self._body), "t": int(self._t), "done": bool(self._done),
            "goal_x": float(self._goal_x), "goal_y": float(self._goal_y),
            "randomization": self._randomization_sample.to_dict(),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        body = state.get("body") if isinstance(state.get("body"), dict) else {}
        self._body = {
            "x": float(body.get("x", 0.0)),
            "y": float(body.get("y", 0.0)),
            "vx": float(body.get("vx", 0.0)),
            "vy": float(body.get("vy", 0.0)),
        }
        self._t = int(state.get("t", 0) or 0)
        self._done = bool(state.get("done", False))
        self._trail = [self.position]
        self._goal_x = float(state.get("goal_x", self._base_goal_x))
        self._goal_y = float(state.get("goal_y", self._base_goal_y))
        randomization = state.get("randomization") if isinstance(state.get("randomization"), dict) else {}
        self._randomization_sample = DomainRandomizationSample(
            actuator_gain=float(randomization.get("actuator_gain", 1.0)),
            goal_offset=tuple(float(v) for v in randomization.get("goal_offset", (0.0, 0.0))),
            visual_tint=tuple(float(v) for v in randomization.get("visual_tint", (1.0, 1.0, 1.0))),
        )
        restore = getattr(self._backend, "import_state", None)
        if callable(restore):
            restore(self._body)
        if isinstance(self._backend, NativePhysicsBackend):
            self._backend.control_scale = self._randomization_sample.actuator_gain


class EmbodiedPointWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["navigation", "robotics", "spatial", "physics", "embodied"]

    def create(self, spec: VerseSpec) -> EmbodiedPointWorldVerse:
        return EmbodiedPointWorldVerse(spec)
