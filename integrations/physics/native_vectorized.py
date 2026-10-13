"""Native batched implementations for the continuous point and reacher tasks."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.vectorized_verse import VectorResetResult, VectorStepResult
from integrations.physics.domain_randomization import DomainRandomizationSample, sample_domain_randomization
from integrations.physics.native_backend import NativePhysicsBackend, build_reacher_mjcf


class NativeVectorizedVerse:
    """A shared MuJoCo model with independent state and task bookkeeping per lane."""

    def __init__(self, spec: VerseSpec, num_envs: int, *, worker_count: int = 0) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        self.spec = spec
        self.num_envs = int(num_envs)
        self.kind = str(spec.verse_name).strip().lower()
        if self.kind not in ("embodied_point_world", "robot_reacher_world"):
            raise ValueError(f"native vectorized simulation is not implemented for {self.kind!r}")
        params = dict(spec.params or {})
        self._domain_randomization = params.get("domain_randomization", False)
        model = "point_mass" if self.kind == "embodied_point_world" else "robot_reacher"
        control_dt = float(params.get("dt", 0.05)) if self.kind == "robot_reacher_world" else 0.1
        model_dt = 0.005 if self.kind == "robot_reacher_world" else 0.01
        substeps = max(1, int(round(control_dt / model_dt)))
        self.backend = NativePhysicsBackend(
            model, env_count=self.num_envs, worker_count=worker_count, substeps=substeps,
            model_xml=(build_reacher_mjcf(float(params.get("l1", 0.5)), float(params.get("l2", 0.5)))
                       if self.kind == "robot_reacher_world" else None),
        )
        self.goal_x = float(params.get("goal_x", 0.8))
        self.goal_y = float(params.get("goal_y", 0.8))
        self._point_goal_x = [self.goal_x] * self.num_envs
        self._point_goal_y = [self.goal_y] * self.num_envs
        self.goal_radius = max(0.05, float(params.get("goal_radius", 0.15)))
        default_max_steps = 40 if self.kind == "embodied_point_world" else 50
        self.max_steps = max(1, int(params.get("max_steps", default_max_steps)))
        self.step_penalty = float(params.get("step_penalty", -0.01))
        self.goal_reward = float(params.get("goal_reward", 1.0))
        self.l1 = float(params.get("l1", 0.5))
        self.l2 = float(params.get("l2", 0.5))
        self.dt = float(params.get("dt", 0.05))
        self.max_velocity = float(params.get("max_velocity", 3.0))
        self.target_radius = float(params.get("target_radius", 0.08))
        self.reacher_goal_reward = float(params.get("goal_reward", 2.0))
        self.reacher_step_penalty = float(params.get("step_penalty", -0.01))
        self.observation_space = spec.observation_space or SpaceSpec(
            type="vector",
            shape=(6,) if self.kind == "embodied_point_world" else (8,),
            dtype="float32",
            notes=("x, y, vx, vy, goal_x, goal_y" if self.kind == "embodied_point_world"
                   else "theta1, theta2, dtheta1, dtheta2, target_x, target_y, ee_x, ee_y"),
        )
        self.action_space = spec.action_space or SpaceSpec(
            type="discrete", n=5,
            notes=("0 hold, 1 +x, 2 -x, 3 +y, 4 -y" if self.kind == "embodied_point_world"
                   else "0 hold, 1 j1+, 2 j1-, 3 j2+, 4 j2-"),
        )
        self._time = [0] * self.num_envs
        self._done = [False] * self.num_envs
        self._rngs: List[random.Random] = [random.Random() for _ in range(self.num_envs)]
        self._target_x = [0.5] * self.num_envs
        self._target_y = [0.5] * self.num_envs
        self._seeds: List[Optional[int]] = [None] * self.num_envs
        self._randomization_samples = [DomainRandomizationSample() for _ in range(self.num_envs)]

    def reset_batch(self, seeds: Sequence[Optional[int]]) -> VectorResetResult:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, received {len(seeds)}")
        self.backend.reset_batch()
        self._time = [0] * self.num_envs
        self._done = [False] * self.num_envs
        self._seeds = [None if seed is None else int(seed) for seed in seeds]
        self._randomization_samples = [
            sample_domain_randomization(seed, self._domain_randomization) for seed in seeds
        ]
        self.backend.control_scale = np.asarray(
            [sample.actuator_gain for sample in self._randomization_samples], dtype=np.float64
        )
        for lane, sample in enumerate(self._randomization_samples):
            self._point_goal_x[lane] = self.goal_x + sample.goal_offset[0]
            self._point_goal_y[lane] = self.goal_y + sample.goal_offset[1]
        if self.kind == "robot_reacher_world":
            qpos, qvel = self.backend.state_batch()
            infos = []
            for lane, seed in enumerate(seeds):
                self._rngs[lane] = random.Random(seed)
                qpos[lane, 0] = self._rngs[lane].uniform(-math.pi * 0.5, math.pi * 0.5)
                qpos[lane, 1] = self._rngs[lane].uniform(-math.pi * 0.5, math.pi * 0.5)
                qvel[lane, :2] = 0.0
                self._target_x[lane], self._target_y[lane] = self._sample_target(self._rngs[lane])
                self._target_x[lane] += self._randomization_samples[lane].goal_offset[0]
                self._target_y[lane] += self._randomization_samples[lane].goal_offset[1]
                distance = self._reacher_distance(qpos[lane], lane)
                infos.append({
                    "distance": float(distance),
                    "end_effector": list(self._end_effector(qpos[lane])),
                    "target": [float(self._target_x[lane]), float(self._target_y[lane])],
                })
            self.backend.set_state_batch(qpos, qvel)
        else:
            infos = [{"distance": self._point_distance(row, lane)}
                     for lane, row in enumerate(self._point_positions())]
        return VectorResetResult(self._observations(), infos)

    def reset_lanes(self, lane_seeds: Dict[int, Optional[int]]) -> VectorResetResult:
        for lane in lane_seeds:
            if not 0 <= int(lane) < self.num_envs:
                raise IndexError(f"lane index out of range: {lane}")
        self.backend.reset_batch(lane_seeds.keys())
        qpos, qvel = self.backend.state_batch()
        infos: Dict[int, Dict[str, JSONValue]] = {}
        for raw_lane, seed in lane_seeds.items():
            lane = int(raw_lane)
            self._time[lane] = 0
            self._done[lane] = False
            self._seeds[lane] = None if seed is None else int(seed)
            sample = sample_domain_randomization(seed, self._domain_randomization)
            self._randomization_samples[lane] = sample
            self._point_goal_x[lane] = self.goal_x + sample.goal_offset[0]
            self._point_goal_y[lane] = self.goal_y + sample.goal_offset[1]
            if self.kind == "robot_reacher_world":
                self._rngs[lane] = random.Random(seed)
                qpos[lane, 0] = self._rngs[lane].uniform(-math.pi * 0.5, math.pi * 0.5)
                qpos[lane, 1] = self._rngs[lane].uniform(-math.pi * 0.5, math.pi * 0.5)
                qvel[lane, :2] = 0.0
                self._target_x[lane], self._target_y[lane] = self._sample_target(self._rngs[lane])
                self._target_x[lane] += sample.goal_offset[0]
                self._target_y[lane] += sample.goal_offset[1]
                infos[lane] = {
                    "distance": float(self._reacher_distance(qpos[lane], lane)),
                    "end_effector": list(self._end_effector(qpos[lane])),
                    "target": [float(self._target_x[lane]), float(self._target_y[lane])],
                }
            else:
                infos[lane] = {"distance": float(self._point_distance(qpos[lane], lane))}
        self.backend.control_scale = np.asarray(
            [sample.actuator_gain for sample in self._randomization_samples], dtype=np.float64
        )
        self.backend.set_state_batch(qpos, qvel)
        observations = self._observations()
        lane_infos: List[Dict[str, JSONValue]] = [{} for _ in range(self.num_envs)]
        for lane, info in infos.items():
            lane_infos[lane] = info
        return VectorResetResult(observations, lane_infos)

    def step_batch(self, actions: Sequence[JSONValue]) -> VectorStepResult:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, received {len(actions)}")
        before_qpos, before_qvel = self.backend.state_batch()
        controls = np.zeros((self.num_envs, self.backend.nu), dtype=np.float64)
        for lane, action in enumerate(actions):
            try:
                act = int(action)
            except (TypeError, ValueError):
                act = 0
            if not self._done[lane]:
                controls[lane] = self.backend.action_controls(act)

        qpos, qvel = self.backend.step_controls(controls)
        rewards = [0.0] * self.num_envs
        terminated = list(self._done)
        truncated = [False] * self.num_envs
        infos: List[Dict[str, JSONValue]] = [{} for _ in range(self.num_envs)]
        for lane in range(self.num_envs):
            if self._done[lane]:
                qpos[lane] = before_qpos[lane]
                qvel[lane] = before_qvel[lane]
                infos[lane] = {"warning": "step after done"}
                continue
            self._time[lane] += 1
            if self.kind == "embodied_point_world":
                distance = self._point_distance(qpos[lane], lane)
                reached = distance <= self.goal_radius
                timeout = not reached and self._time[lane] >= self.max_steps
                rewards[lane] = self.goal_reward if reached else self.step_penalty
                terminated[lane] = bool(reached)
                truncated[lane] = bool(timeout)
                infos[lane] = {"distance": distance, "reached_goal": bool(reached), "t": int(self._time[lane])}
                self._done[lane] = bool(reached or timeout)
            else:
                qpos[lane, :2] = (qpos[lane, :2] + math.pi) % (2.0 * math.pi) - math.pi
                qvel[lane, :2] = np.clip(qvel[lane, :2], -self.max_velocity, self.max_velocity)
                ee = self._end_effector(qpos[lane])
                distance = self._reacher_distance(qpos[lane], lane)
                reached = distance <= self.target_radius
                timeout = not reached and self._time[lane] >= self.max_steps
                rewards[lane] = self.reacher_goal_reward if reached else (-distance * 0.1 + self.reacher_step_penalty)
                terminated[lane] = bool(reached)
                truncated[lane] = bool(timeout)
                infos[lane] = {
                    "distance": float(distance), "reached_goal": bool(reached),
                    "end_effector": list(ee),
                    "target": [float(self._target_x[lane]), float(self._target_y[lane])],
                    "t": int(self._time[lane]),
                }
                self._done[lane] = bool(reached or timeout)

        self.backend.set_state_batch(qpos, qvel)
        return VectorStepResult(self._observations(), rewards, terminated, truncated, infos)

    def _point_positions(self) -> np.ndarray:
        qpos, _ = self.backend.state_batch()
        return qpos

    def _point_distance(self, qpos: Sequence[float], lane: int) -> float:
        return float(math.hypot(float(qpos[0]) - self._point_goal_x[lane],
                                float(qpos[1]) - self._point_goal_y[lane]))

    def _end_effector(self, qpos: Sequence[float]) -> Tuple[float, float]:
        angle1, angle2 = float(qpos[0]), float(qpos[1])
        return (
            self.l1 * math.cos(angle1) + self.l2 * math.cos(angle1 + angle2),
            self.l1 * math.sin(angle1) + self.l2 * math.sin(angle1 + angle2),
        )

    def _reacher_distance(self, qpos: Sequence[float], lane: int) -> float:
        ee_x, ee_y = self._end_effector(qpos)
        return float(math.hypot(ee_x - self._target_x[lane], ee_y - self._target_y[lane]))

    def _sample_target(self, rng: random.Random) -> Tuple[float, float]:
        min_radius = abs(self.l1 - self.l2) + 0.15
        max_radius = (self.l1 + self.l2) * 0.92
        radius = rng.uniform(min_radius, max_radius)
        phi = rng.uniform(-math.pi * 0.8, math.pi * 0.8)
        return radius * math.cos(phi), radius * math.sin(phi)

    def _observations(self) -> List[JSONValue]:
        qpos, qvel = self.backend.state_batch()
        observations: List[JSONValue] = []
        for lane in range(self.num_envs):
            if self.kind == "embodied_point_world":
                observations.append([
                    float(qpos[lane, 0]), float(qpos[lane, 1]),
                    float(qvel[lane, 0]), float(qvel[lane, 1]),
                    self._point_goal_x[lane], self._point_goal_y[lane],
                ])
            else:
                ee_x, ee_y = self._end_effector(qpos[lane])
                observations.append([
                    float(qpos[lane, 0]), float(qpos[lane, 1]),
                    float(qvel[lane, 0]), float(qvel[lane, 1]),
                    float(self._target_x[lane]), float(self._target_y[lane]),
                    float(ee_x), float(ee_y),
                ])
        return observations

    def export_state(self) -> Dict[str, JSONValue]:
        qpos, qvel = self.backend.state_batch()
        return {
            "qpos": qpos.tolist(), "qvel": qvel.tolist(),
            "time": list(self._time), "done": list(self._done),
            "target_x": list(self._target_x), "target_y": list(self._target_y),
            "seeds": list(self._seeds),
            "point_goal_x": list(self._point_goal_x), "point_goal_y": list(self._point_goal_y),
            "randomization": [sample.to_dict() for sample in self._randomization_samples],
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
        if qpos.shape != (self.num_envs, self.backend.nq) or qvel.shape != (self.num_envs, self.backend.nv):
            raise ValueError("checkpoint qpos/qvel dimensions do not match this vectorized Verse")
        self.backend.set_state_batch(qpos, qvel)
        self._time = [int(v) for v in state.get("time", [0] * self.num_envs)]
        self._done = [bool(v) for v in state.get("done", [False] * self.num_envs)]
        self._target_x = [float(v) for v in state.get("target_x", [0.5] * self.num_envs)]
        self._target_y = [float(v) for v in state.get("target_y", [0.5] * self.num_envs)]
        self._seeds = [None if v is None else int(v) for v in state.get("seeds", [None] * self.num_envs)]
        self._point_goal_x = [float(v) for v in state.get("point_goal_x", [self.goal_x] * self.num_envs)]
        self._point_goal_y = [float(v) for v in state.get("point_goal_y", [self.goal_y] * self.num_envs)]
        raw_samples = state.get("randomization", [{} for _ in range(self.num_envs)])
        self._randomization_samples = [self._sample_from_dict(value) for value in raw_samples]
        self.backend.control_scale = np.asarray(
            [sample.actuator_gain for sample in self._randomization_samples], dtype=np.float64
        )
        self._rngs = [random.Random(seed) for seed in self._seeds]

    @staticmethod
    def _sample_from_dict(value: object) -> DomainRandomizationSample:
        item = value if isinstance(value, dict) else {}
        return DomainRandomizationSample(
            actuator_gain=float(item.get("actuator_gain", 1.0)),
            goal_offset=tuple(float(v) for v in item.get("goal_offset", (0.0, 0.0))),
            visual_tint=tuple(float(v) for v in item.get("visual_tint", (1.0, 1.0, 1.0))),
        )

    def close(self) -> None:
        self.backend.close()

__all__ = ["NativeVectorizedVerse"]
