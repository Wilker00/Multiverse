"""
verses/factory_world.py

Production scheduling verse.
The agent manages a simple assembly line: raw jobs arrive, must be processed
through machines in the correct sequence (A→B→C). Machines can break down
randomly. The agent decides which machine to operate each step. Throughput
(completed jobs) drives reward; idle machines and breakdowns cost time.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class FactoryParams:
    num_machines: int = 3          # stages in pipeline (A, B, C)
    buffer_size: int = 4           # max items queued between stages
    max_steps: int = 100
    arrival_rate: float = 0.6      # probability a new job arrives each step
    breakdown_prob: float = 0.08   # probability each machine breaks per step
    repair_steps: int = 3          # steps to repair a broken machine
    step_penalty: float = -0.02
    completion_reward: float = 2.0
    overflow_penalty: float = -0.5
    idle_penalty: float = -0.1


class FactoryWorldVerse:
    """
    Production scheduling:
    - num_machines stages in sequence (0, 1, 2, ...)
    - Jobs enter at stage 0 buffer, must pass through each stage in order
    - Each step: agent picks which machine to run (processes 1 item from its input buffer)
    - Machines break down randomly; agent can choose to repair instead
    - Buffer overflow = penalty (items lost)
    - Completed items (exit last stage) = reward
    """

    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in FactoryWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        cfg = self.spec.params
        self.params = FactoryParams(
            num_machines=max(2, int(cfg.get("num_machines", 3))),
            buffer_size=max(2, int(cfg.get("buffer_size", 4))),
            max_steps=max(10, int(cfg.get("max_steps", 100))),
            arrival_rate=max(0.0, min(1.0, float(cfg.get("arrival_rate", 0.6)))),
            breakdown_prob=max(0.0, min(1.0, float(cfg.get("breakdown_prob", 0.08)))),
            repair_steps=max(1, int(cfg.get("repair_steps", 3))),
            step_penalty=float(cfg.get("step_penalty", -0.02)),
            completion_reward=float(cfg.get("completion_reward", 2.0)),
            overflow_penalty=float(cfg.get("overflow_penalty", -0.5)),
            idle_penalty=float(cfg.get("idle_penalty", -0.1)),
        )

        n = self.params.num_machines
        obs_keys = [
            "t", "completed", "total_arrived",
        ]
        subspaces: Dict[str, SpaceSpec] = {
            "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "completed": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            "total_arrived": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
        }
        # Per-machine: buffer level, broken status, repair timer
        for i in range(n):
            obs_keys.extend([f"buf_{i}", f"broken_{i}", f"repair_{i}"])
            subspaces[f"buf_{i}"] = SpaceSpec(type="vector", shape=(1,), dtype="int32")
            subspaces[f"broken_{i}"] = SpaceSpec(type="vector", shape=(1,), dtype="int32")
            subspaces[f"repair_{i}"] = SpaceSpec(type="vector", shape=(1,), dtype="int32")
        # Output buffer
        obs_keys.append("output_buf")
        subspaces["output_buf"] = SpaceSpec(type="vector", shape=(1,), dtype="int32")

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict", keys=obs_keys, subspaces=subspaces,
            notes="FactoryWorld obs dict",
        )

        # Actions: run machine i (0..n-1), repair machine i (n..2n-1), or idle (2n)
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=2 * n + 1,
            notes=f"0..{n-1}=run_machine, {n}..{2*n-1}=repair_machine, {2*n}=idle",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._t = 0
        self._done = False
        self._buffers: List[int] = []      # items waiting at each stage input
        self._output_buf = 0                # items completed last stage
        self._broken: List[bool] = []       # is machine i broken?
        self._repair_timer: List[int] = []  # steps left to repair
        self._completed = 0
        self._total_arrived = 0

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)

        n = self.params.num_machines
        self._t = 0
        self._done = False
        self._buffers = [0] * (n + 1)  # n input buffers + 1 output buffer
        self._output_buf = 0
        self._broken = [False] * n
        self._repair_timer = [0] * n
        self._completed = 0
        self._total_arrived = 0

        # Start with a few jobs in the first buffer
        self._buffers[0] = 2
        self._total_arrived = 2

        obs = self._make_obs()
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "num_machines": n,
            "buffer_size": self.params.buffer_size,
        }
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._make_obs(), 0.0, True, False, {"warning": "step() called after done"})

        n = self.params.num_machines
        a = max(0, min(2 * n, int(action)))
        self._t += 1
        reward = float(self.params.step_penalty)
        info: Dict[str, JSONValue] = {"action_used": int(a)}

        if a < n:
            # Run machine a
            machine = a
            if self._broken[machine]:
                info["machine_broken"] = machine
                reward += float(self.params.idle_penalty)
            elif self._buffers[machine] > 0:
                # Process one item: move from buffer[machine] to buffer[machine+1]
                self._buffers[machine] -= 1
                if machine + 1 < len(self._buffers):
                    if self._buffers[machine + 1] < self.params.buffer_size:
                        self._buffers[machine + 1] += 1
                        info["processed"] = machine
                    else:
                        # Overflow — item lost
                        reward += float(self.params.overflow_penalty)
                        info["overflow"] = machine + 1
            else:
                info["empty_buffer"] = machine
                reward += float(self.params.idle_penalty)
        elif a < 2 * n:
            # Repair machine (a - n)
            machine = a - n
            if self._broken[machine] and self._repair_timer[machine] > 0:
                self._repair_timer[machine] -= 1
                info["repairing"] = machine
                if self._repair_timer[machine] <= 0:
                    self._broken[machine] = False
                    info["repaired"] = machine
            else:
                reward += float(self.params.idle_penalty)
                info["unnecessary_repair"] = True
        else:
            # Idle
            reward += float(self.params.idle_penalty)
            info["idle"] = True

        # Collect completed items from the last buffer
        last_buf = len(self._buffers) - 1
        if self._buffers[last_buf] > 0:
            completed_now = self._buffers[last_buf]
            self._completed += completed_now
            reward += float(self.params.completion_reward) * completed_now
            info["items_completed"] = completed_now
            self._buffers[last_buf] = 0

        # New job arrivals
        if self._rng.random() < self.params.arrival_rate:
            if self._buffers[0] < self.params.buffer_size:
                self._buffers[0] += 1
                self._total_arrived += 1
                info["new_arrival"] = True
            else:
                info["arrival_blocked"] = True

        # Random breakdowns
        for i in range(n):
            if not self._broken[i] and self._rng.random() < self.params.breakdown_prob:
                self._broken[i] = True
                self._repair_timer[i] = self.params.repair_steps
                info[f"breakdown_{i}"] = True

        done = False
        truncated = bool(self._t >= self.params.max_steps and not done)
        self._done = bool(done or truncated)
        info["t"] = int(self._t)
        info["completed"] = int(self._completed)

        return StepResult(self._make_obs(), float(reward), bool(done), bool(truncated), info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        n = self.params.num_machines
        parts = []
        for i in range(n):
            status = "X" if self._broken[i] else "OK"
            parts.append(f"M{i}[{status}]:{self._buffers[i]}")
        parts.append(f"OUT:{self._buffers[-1]}")
        return f"t={self._t} done={self._completed} | {' -> '.join(parts)}"

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "t": int(self._t), "done": bool(self._done),
            "buffers": list(self._buffers),
            "broken": [bool(b) for b in self._broken],
            "repair_timer": list(self._repair_timer),
            "completed": int(self._completed),
            "total_arrived": int(self._total_arrived),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))
        bufs = state.get("buffers")
        if isinstance(bufs, list) and len(bufs) == len(self._buffers):
            self._buffers = [max(0, int(b)) for b in bufs]
        brk = state.get("broken")
        if isinstance(brk, list) and len(brk) == len(self._broken):
            self._broken = [bool(b) for b in brk]
        rep = state.get("repair_timer")
        if isinstance(rep, list) and len(rep) == len(self._repair_timer):
            self._repair_timer = [max(0, int(r)) for r in rep]
        self._completed = max(0, int(state.get("completed", self._completed)))
        self._total_arrived = max(0, int(state.get("total_arrived", self._total_arrived)))

    def _make_obs(self) -> JSONValue:
        n = self.params.num_machines
        obs: Dict[str, Any] = {
            "t": int(self._t),
            "completed": int(self._completed),
            "total_arrived": int(self._total_arrived),
        }
        for i in range(n):
            obs[f"buf_{i}"] = int(self._buffers[i])
            obs[f"broken_{i}"] = 1 if self._broken[i] else 0
            obs[f"repair_{i}"] = int(self._repair_timer[i])
        obs["output_buf"] = int(self._buffers[-1]) if len(self._buffers) > n else 0
        return obs


class FactoryWorldFactory:
    @property
    def tags(self) -> List[str]:
        return ["scheduling", "production", "sequential_decision", "fault_tolerance"]

    def create(self, spec: VerseSpec) -> Verse:
        return FactoryWorldVerse(spec)
