"""
Configuration helpers for Isaac Lab backed verses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from core.types import SpaceSpec


@dataclass(frozen=True)
class IsaacLabTaskConfig:
    task_id: str
    scene: str = "warehouse"
    headless: bool = True
    enable_cameras: bool = False
    max_episode_length: int = 256
    observation_mode: str = "dict"
    action_mode: str = "continuous"
    render_mode: str = "rgb_array"
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_params(*, task_id: str, params: Mapping[str, Any] | None = None) -> "IsaacLabTaskConfig":
        raw = dict(params or {})
        return IsaacLabTaskConfig(
            task_id=str(task_id),
            scene=str(raw.get("scene", "warehouse")),
            headless=bool(raw.get("headless", True)),
            enable_cameras=bool(raw.get("enable_cameras", False)),
            max_episode_length=max(1, int(raw.get("max_episode_length", raw.get("max_steps", 256)))),
            observation_mode=str(raw.get("observation_mode", "dict")),
            action_mode=str(raw.get("action_mode", "continuous")),
            render_mode=str(raw.get("render_mode", "rgb_array")),
            extra={
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "scene",
                    "headless",
                    "enable_cameras",
                    "max_episode_length",
                    "max_steps",
                    "observation_mode",
                    "action_mode",
                    "render_mode",
                }
            },
        )


@dataclass(frozen=True)
class IsaacLabVerseBinding:
    verse_name: str
    task_id: str
    description: str
    tags: Tuple[str, ...]
    default_params: Dict[str, Any]
    observation_space: SpaceSpec
    action_space: SpaceSpec

    def task_config_from_params(self, params: Mapping[str, Any] | None = None) -> IsaacLabTaskConfig:
        merged = dict(self.default_params)
        merged.update(dict(params or {}))
        return IsaacLabTaskConfig.from_params(task_id=self.task_id, params=merged)


ISAAC_WAREHOUSE_OBSERVATION_SPACE = SpaceSpec(
    type="dict",
    keys=["position", "goal_position", "goal_delta", "velocity", "lidar", "battery", "flat"],
    subspaces={
        "position": SpaceSpec(type="vector", shape=(3,), dtype="float32"),
        "goal_position": SpaceSpec(type="vector", shape=(3,), dtype="float32"),
        "goal_delta": SpaceSpec(type="vector", shape=(3,), dtype="float32"),
        "velocity": SpaceSpec(type="vector", shape=(3,), dtype="float32"),
        "lidar": SpaceSpec(type="vector", shape=(16,), dtype="float32"),
        "battery": SpaceSpec(type="vector", shape=(1,), dtype="float32"),
        "flat": SpaceSpec(type="vector", shape=(26,), dtype="float32"),
    },
    notes="Normalized Isaac warehouse navigation observation contract.",
)

ISAAC_WAREHOUSE_ACTION_SPACE = SpaceSpec(
    type="continuous",
    shape=(2,),
    dtype="float32",
    low=[-1.0, -1.0],
    high=[1.0, 1.0],
    notes="Normalized Isaac warehouse navigation action contract: [linear_cmd, angular_cmd].",
)


DEFAULT_ISAACLAB_VERSE_BINDINGS: Dict[str, IsaacLabVerseBinding] = {
    "isaac_warehouse_nav": IsaacLabVerseBinding(
        verse_name="isaac_warehouse_nav",
        task_id="Isaac-Multiverse-Warehouse-Navigation-v0",
        description="Warehouse mobile-robot navigation lane backed by Isaac Lab / Isaac Sim.",
        tags=("robotics", "warehouse", "navigation", "sim2real", "high_fidelity"),
        default_params={
            "scene": "warehouse",
            "headless": True,
            "enable_cameras": False,
            "max_steps": 256,
            "observation_mode": "dict",
            "action_mode": "continuous",
            "render_mode": "rgb_array",
            "lidar_beams": 16,
        },
        observation_space=ISAAC_WAREHOUSE_OBSERVATION_SPACE,
        action_space=ISAAC_WAREHOUSE_ACTION_SPACE,
    ),
}


def get_isaaclab_binding(verse_name: str) -> IsaacLabVerseBinding:
    key = str(verse_name or "").strip().lower()
    if key not in DEFAULT_ISAACLAB_VERSE_BINDINGS:
        known = ", ".join(sorted(DEFAULT_ISAACLAB_VERSE_BINDINGS))
        raise KeyError(f"Unknown Isaac Lab verse '{verse_name}'. Known: {known}")
    return DEFAULT_ISAACLAB_VERSE_BINDINGS[key]
