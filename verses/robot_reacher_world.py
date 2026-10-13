"""
verses/robot_reacher_world.py

2-Link Planar Robotic Arm Reacher Verse.

A continuous kinematics embodiment for Multiverse:
- 2 revolute joints (theta1, theta2) mounted at origin (0, 0).
- Forward and differential kinematics.
- Continuous state observation and discrete or continuous joint actuation.
- Software rasterization for render(mode="rgb_array") and native vector drawing.
- Seamlessly bridges game-space spatial reasoning into physical joint-space control.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult
from integrations.physics.native_backend import NativePhysicsBackend, build_reacher_mjcf
from integrations.physics.domain_randomization import DomainRandomizationSample, sample_domain_randomization


class RobotReacherWorldVerse:
    """
    2-Link Articulated Arm reaching for a target point in 2D space.

    Lengths:
        l1 = 0.5, l2 = 0.5 (total reach = 1.0)
    Observation space:
        [theta1, theta2, dtheta1, dtheta2, target_x, target_y, ee_x, ee_y] (dim 8)
    Action space:
        Discrete 5:
          0: hold
          1: joint 1 +
          2: joint 1 -
          3: joint 2 +
          4: joint 2 -
    """

    def __init__(self, spec: VerseSpec) -> None:
        tags = list(spec.tags)
        for tag in RobotReacherWorldFactory().tags:
            if tag not in tags:
                tags.append(tag)
        self.spec = dataclasses.replace(spec, tags=tags)
        params = dict(self.spec.params or {})

        self.l1 = float(params.get("l1", 0.5))
        self.l2 = float(params.get("l2", 0.5))
        self.dt = float(params.get("dt", 0.05))
        self.max_velocity = float(params.get("max_velocity", 3.0))
        self.joint_speed = float(params.get("joint_speed", 1.5))
        self.target_radius = float(params.get("target_radius", 0.08))
        self._max_steps = max(1, int(params.get("max_steps", 50)))
        self._goal_reward = float(params.get("goal_reward", 2.0))
        self._step_penalty = float(params.get("step_penalty", -0.01))
        self._domain_randomization = params.get("domain_randomization", False)
        self._randomization_sample = DomainRandomizationSample()
        provider = str(params.get("sim_provider", "python")).strip().lower()
        if provider == "multiverse_native":
            self._backend: Optional[NativePhysicsBackend] = NativePhysicsBackend(
                "robot_reacher_world",
                worker_count=int(params.get("sim_workers", 0) or 0),
                substeps=max(1, int(round(self.dt / 0.005))),
                model_xml=build_reacher_mjcf(self.l1, self.l2),
            )
        elif provider == "python":
            self._backend = None
        else:
            raise ValueError(f"Unknown sim_provider {provider!r}; expected 'python' or 'multiverse_native'")

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="vector",
            shape=(8,),
            dtype="float32",
            notes="theta1, theta2, dtheta1, dtheta2, target_x, target_y, ee_x, ee_y",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=5,
            notes="0: hold, 1: j1+, 2: j1-, 3: j2+, 4: j2-",
        )

        self._rng = random.Random(self.spec.seed)
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.target_x = 0.5
        self.target_y = 0.5
        self._t = 0
        self._done = False
        self._trail: List[Tuple[float, float]] = []

    def seed(self, seed: Optional[int]) -> None:
        self.spec = dataclasses.replace(self.spec, seed=seed)
        self._rng = random.Random(seed)

    def forward_kinematics(self, th1: float, th2: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return (joint1_xy, end_effector_xy)."""
        x1 = self.l1 * math.cos(th1)
        y1 = self.l1 * math.sin(th1)
        x2 = x1 + self.l2 * math.cos(th1 + th2)
        y2 = y1 + self.l2 * math.sin(th1 + th2)
        return (x1, y1), (x2, y2)

    @property
    def end_effector(self) -> Tuple[float, float]:
        _, ee = self.forward_kinematics(self.theta1, self.theta2)
        return ee

    @property
    def elbow(self) -> Tuple[float, float]:
        j1, _ = self.forward_kinematics(self.theta1, self.theta2)
        return j1

    def _sample_target(self) -> Tuple[float, float]:
        min_r = abs(self.l1 - self.l2) + 0.15
        max_r = (self.l1 + self.l2) * 0.92
        r = self._rng.uniform(min_r, max_r)
        phi = self._rng.uniform(-math.pi * 0.8, math.pi * 0.8)
        return (r * math.cos(phi), r * math.sin(phi))

    def _obs(self) -> List[float]:
        ee_x, ee_y = self.end_effector
        return [
            float(self.theta1),
            float(self.theta2),
            float(self.dtheta1),
            float(self.dtheta2),
            float(self.target_x),
            float(self.target_y),
            float(ee_x),
            float(ee_y),
        ]

    def _distance_to_target(self) -> float:
        ee_x, ee_y = self.end_effector
        dx = ee_x - self.target_x
        dy = ee_y - self.target_y
        return math.hypot(dx, dy)

    def reset(self) -> ResetResult:
        self.theta1 = self._rng.uniform(-math.pi * 0.5, math.pi * 0.5)
        self.theta2 = self._rng.uniform(-math.pi * 0.5, math.pi * 0.5)
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.target_x, self.target_y = self._sample_target()
        self._randomization_sample = sample_domain_randomization(self.spec.seed, self._domain_randomization)
        self.target_x += self._randomization_sample.goal_offset[0]
        self.target_y += self._randomization_sample.goal_offset[1]
        self._t = 0
        self._done = False
        self._trail = [self.end_effector]
        if self._backend is not None:
            self._backend.control_scale = self._randomization_sample.actuator_gain
            self._backend.import_state(self._physics_state())
        dist = self._distance_to_target()
        return ResetResult(
            obs=self._obs(),
            info={
                "distance": dist,
                "end_effector": list(self.end_effector),
                "target": [self.target_x, self.target_y],
            },
        )

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(
                obs=self._obs(),
                reward=0.0,
                done=True,
                truncated=False,
                info={"warning": "step after done"},
            )

        act = int(action)
        # Discrete joint velocity commands
        target_v1, target_v2 = 0.0, 0.0
        if act == 1:
            target_v1 = self.joint_speed
        elif act == 2:
            target_v1 = -self.joint_speed
        elif act == 3:
            target_v2 = self.joint_speed
        elif act == 4:
            target_v2 = -self.joint_speed

        if self._backend is None:
            # The legacy integrator remains the default and compatibility baseline.
            damping = 0.8
            self.dtheta1 = float(self.dtheta1 * damping + target_v1 * (1.0 - damping))
            self.dtheta2 = float(self.dtheta2 * damping + target_v2 * (1.0 - damping))
            self.dtheta1 = max(-self.max_velocity, min(self.max_velocity, self.dtheta1))
            self.dtheta2 = max(-self.max_velocity, min(self.max_velocity, self.dtheta2))
            self.theta1 += self.dtheta1 * self.dt
            self.theta2 += self.dtheta2 * self.dt
        else:
            native_state = self._backend.step(self._physics_state(), act)
            self.theta1 = float(native_state["theta1"])
            self.theta2 = float(native_state["theta2"])
            self.dtheta1 = float(native_state["dtheta1"])
            self.dtheta2 = float(native_state["dtheta2"])

        # Normalize angles to [-pi, pi]
        self.theta1 = (self.theta1 + math.pi) % (2.0 * math.pi) - math.pi
        self.theta2 = (self.theta2 + math.pi) % (2.0 * math.pi) - math.pi

        self._t += 1
        ee = self.end_effector
        self._trail.append(ee)
        if len(self._trail) > 30:
            self._trail.pop(0)

        dist = self._distance_to_target()
        reached = dist <= self.target_radius
        truncated = (not reached) and (self._t >= self._max_steps)

        # Distance-shaped reward + bonus on reach
        shaping_reward = -float(dist)
        reward = self._goal_reward if reached else (shaping_reward * 0.1 + self._step_penalty)

        self._done = bool(reached or truncated)
        return StepResult(
            obs=self._obs(),
            reward=float(reward),
            done=bool(reached),
            truncated=bool(truncated),
            info={
                "distance": float(dist),
                "reached_goal": bool(reached),
                "end_effector": list(ee),
                "target": [self.target_x, self.target_y],
                "t": int(self._t),
            },
        )

    def render(self, mode: str = "ansi") -> Optional[Any]:
        if mode == "ansi":
            ee = self.end_effector
            return (
                f"Reacher t={self._t} j1={self.theta1:+.2f} j2={self.theta2:+.2f} "
                f"ee=({ee[0]:+.2f},{ee[1]:+.2f}) target=({self.target_x:+.2f},{self.target_y:+.2f}) "
                f"dist={self._distance_to_target():.3f}"
            )
        elif mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_rgb_array(self, width: int = 128, height: int = 128) -> np.ndarray:
        """Rasterize a top-down view to a numpy RGB array."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Background dark blue-grey
        canvas[:] = [15, 18, 30]

        cx, cy = width // 2, height // 2
        scale = min(width, height) * 0.42

        def to_screen(x: float, y: float) -> Tuple[int, int]:
            px = int(cx + x * scale)
            py = int(cy - y * scale)
            return px, py

        # Draw workspace boundary circle
        max_reach_px = int((self.l1 + self.l2) * scale)
        for angle_deg in range(0, 360, 5):
            rad = math.radians(angle_deg)
            px = int(cx + max_reach_px * math.cos(rad))
            py = int(cy + max_reach_px * math.sin(rad))
            if 0 <= px < width and 0 <= py < height:
                canvas[py, px] = [40, 50, 75]

        # Draw target
        tx, ty = to_screen(self.target_x, self.target_y)
        t_rad = max(2, int(self.target_radius * scale))
        y_min, y_max = max(0, ty - t_rad), min(height, ty + t_rad + 1)
        x_min, x_max = max(0, tx - t_rad), min(width, tx + t_rad + 1)
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (x - tx) ** 2 + (y - ty) ** 2 <= t_rad ** 2:
                    canvas[y, x] = [0, 230, 100]

        # Draw trail
        for px, py in [to_screen(tx, ty) for tx, ty in self._trail]:
            if 0 <= px < width and 0 <= py < height:
                canvas[py, px] = [0, 150, 220]

        # Draw arm joints and links
        b_px, b_py = to_screen(0.0, 0.0)
        e_px, e_py = to_screen(*self.elbow)
        ee_px, ee_py = to_screen(*self.end_effector)

        def draw_line(x0: int, y0: int, x1: int, y1: int, color: List[int], width_px: int = 2) -> None:
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(steps + 1):
                x = int(x0 + (x1 - x0) * s / steps)
                y = int(y0 + (y1 - y0) * s / steps)
                for dx in range(-width_px // 2, width_px // 2 + 1):
                    for dy in range(-width_px // 2, width_px // 2 + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            canvas[ny, nx] = color

        # Link 1 (cyan/metallic)
        draw_line(b_px, b_py, e_px, e_py, [80, 180, 255], 3)
        # Link 2 (orange/metallic)
        draw_line(e_px, e_py, ee_px, ee_py, [255, 160, 50], 3)

        # Base and joints
        for jx, jy, col in [(b_px, b_py, [200, 200, 200]), (e_px, e_py, [100, 220, 255]), (ee_px, ee_py, [255, 230, 80])]:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx * dx + dy * dy <= 9:
                        nx, ny = jx + dx, jy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            canvas[ny, nx] = col

        return canvas

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    def _randomization_state(self) -> Dict[str, JSONValue]:
        return self._randomization_sample.to_dict()

    def _physics_state(self) -> Dict[str, float]:
        return {
            "theta1": float(self.theta1),
            "theta2": float(self.theta2),
            "dtheta1": float(self.dtheta1),
            "dtheta2": float(self.dtheta2),
        }

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "theta1": float(self.theta1),
            "theta2": float(self.theta2),
            "dtheta1": float(self.dtheta1),
            "dtheta2": float(self.dtheta2),
            "target_x": float(self.target_x),
            "target_y": float(self.target_y),
            "t": int(self._t),
            "done": bool(self._done),
            "randomization": self._randomization_state(),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self.theta1 = float(state.get("theta1", 0.0))
        self.theta2 = float(state.get("theta2", 0.0))
        self.dtheta1 = float(state.get("dtheta1", 0.0))
        self.dtheta2 = float(state.get("dtheta2", 0.0))
        self.target_x = float(state.get("target_x", 0.5))
        self.target_y = float(state.get("target_y", 0.5))
        self._t = int(state.get("t", 0) or 0)
        self._done = bool(state.get("done", False))
        self._trail = [self.end_effector]
        randomization = state.get("randomization") if isinstance(state.get("randomization"), dict) else {}
        self._randomization_sample = DomainRandomizationSample(
            actuator_gain=float(randomization.get("actuator_gain", 1.0)),
            goal_offset=tuple(float(v) for v in randomization.get("goal_offset", (0.0, 0.0))),
            visual_tint=tuple(float(v) for v in randomization.get("visual_tint", (1.0, 1.0, 1.0))),
        )
        if self._backend is not None:
            self._backend.control_scale = self._randomization_sample.actuator_gain
            self._backend.import_state(self._physics_state())


class RobotReacherWorldFactory:
    @property
    def tags(self) -> List[str]:
        return [
            "robotics",
            "navigation",
            "spatial",
            "spatial_reasoning",
            "continuous_control",
            "embodied",
        ]

    def create(self, spec: VerseSpec) -> RobotReacherWorldVerse:
        return RobotReacherWorldVerse(spec)
