"""
verses/rule_flip_world.py

Diagnostic verse for memory rewriting:
- Reward target flips halfway through the episode.
- Agent must rewrite old policy memory quickly.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class RuleFlipParams:
    track_len: int = 11
    max_steps: int = 80
    flip_step: int = 40
    step_penalty: float = -0.01
    target_reward: float = 2.0
    wrong_target_penalty: float = -1.5
    recenter_on_target: bool = True


class RuleFlipWorldVerse:
    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in RuleFlipWorldFactory().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)

        cfg = self.spec.params
        default_max_steps = max(20, int(cfg.get("max_steps", 80)))
        default_flip_step = max(2, default_max_steps // 2)
        self.params = RuleFlipParams(
            track_len=max(5, int(cfg.get("track_len", 11))),
            max_steps=default_max_steps,
            flip_step=max(2, min(default_max_steps - 1, int(cfg.get("flip_step", default_flip_step)))),
            step_penalty=float(cfg.get("step_penalty", -0.01)),
            target_reward=float(cfg.get("target_reward", 2.0)),
            wrong_target_penalty=float(cfg.get("wrong_target_penalty", -1.5)),
            recenter_on_target=bool(cfg.get("recenter_on_target", True)),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=["pos", "left_goal", "right_goal", "t", "flip_step"],
            subspaces={
                "pos": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "left_goal": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "right_goal": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "flip_step": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            },
            notes="Reward rule flips from right-goal to left-goal at flip_step.",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=3,
            notes="0=left,1=right,2=stay",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._pos = 0
        self._t = 0
        self._done = False

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randrange(1, 2**31 - 1)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> ResetResult:
        if self._seed is None:
            self.seed(self.spec.seed)
        self._t = 0
        self._done = False
        self._pos = self.params.track_len // 2
        info: Dict[str, JSONValue] = {
            "seed": self._seed,
            "flip_step": int(self.params.flip_step),
            "active_rule": "go_right",
            "memory_rewrite_diagnostic": True,
        }
        return ResetResult(obs=self._make_obs(), info=info)

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(obs=self._make_obs(), reward=0.0, done=True, truncated=False, info={"warning": "step() called after done"})

        a = max(0, min(2, int(action)))
        prev_t = int(self._t)
        prev_rule = self._active_rule(prev_t)

        if a == 0:
            self._pos = max(0, int(self._pos - 1))
        elif a == 1:
            self._pos = min(self.params.track_len - 1, int(self._pos + 1))

        self._t += 1
        cur_rule = self._active_rule(self._t)
        active_goal = self._goal_for_rule(cur_rule)
        wrong_goal = self._goal_for_rule("go_right" if cur_rule == "go_left" else "go_left")

        reward = float(self.params.step_penalty)
        hit_target = bool(self._pos == active_goal)
        hit_wrong = bool(self._pos == wrong_goal)
        if hit_target:
            reward += float(self.params.target_reward)
        elif hit_wrong:
            reward += float(self.params.wrong_target_penalty)

        if hit_target and bool(self.params.recenter_on_target):
            self._pos = self.params.track_len // 2

        truncated = bool(self._t >= int(self.params.max_steps))
        self._done = bool(truncated)

        info: Dict[str, JSONValue] = {
            "action_used": int(a),
            "active_rule": str(cur_rule),
            "rule_flipped": bool(prev_rule != cur_rule),
            "target_hit": bool(hit_target),
            "wrong_target_hit": bool(hit_wrong),
            "t": int(self._t),
        }
        return StepResult(
            obs=self._make_obs(),
            reward=float(reward),
            done=False,
            truncated=bool(truncated),
            info=info,
        )

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        cells = ["." for _ in range(self.params.track_len)]
        left_goal = 0
        right_goal = self.params.track_len - 1
        cells[left_goal] = "L"
        cells[right_goal] = "R"
        cells[self._pos] = "A"
        return f"|{''.join(cells)}| t={self._t} rule={self._active_rule(self._t)}"

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {"pos": int(self._pos), "t": int(self._t), "done": bool(self._done)}

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._pos = max(0, min(self.params.track_len - 1, int(state.get("pos", self._pos))))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", False))

    def _active_rule(self, t: int) -> str:
        return "go_right" if int(t) < int(self.params.flip_step) else "go_left"

    def _goal_for_rule(self, rule: str) -> int:
        if str(rule) == "go_left":
            return 0
        return self.params.track_len - 1

    def _make_obs(self) -> JSONValue:
        return {
            "pos": int(self._pos),
            "left_goal": 0,
            "right_goal": int(self.params.track_len - 1),
            "t": int(self._t),
            "flip_step": int(self.params.flip_step),
        }


class RuleFlipWorldFactory:
    @property
    def tags(self) -> List[str]:
        return [
            "memory_diagnostics",
            "rule_shift",
            "adaptive_control",
            "sequential_decision",
        ]

    def create(self, spec: VerseSpec) -> Verse:
        return RuleFlipWorldVerse(spec)

