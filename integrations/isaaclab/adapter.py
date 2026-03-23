"""
Verse adapter for Isaac Lab tasks.

This file is safe to import even when Isaac Lab is unavailable.
Concrete simulator imports happen only at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Mapping, Optional

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse
from integrations.isaaclab.config import IsaacLabTaskConfig, IsaacLabVerseBinding
from integrations.isaaclab.factory import IsaacLabUnavailableError, require_isaaclab


def _to_jsonable(value: Any) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "detach"):
        try:
            return _to_jsonable(value.detach().cpu().tolist())
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            return _to_jsonable(value.cpu().tolist())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _ensure_length(values: Iterable[Any], size: int, *, default: float = 0.0) -> list[float]:
    out = [_safe_float(v, default) for v in list(values)]
    if len(out) < int(size):
        out.extend([float(default)] * (int(size) - len(out)))
    return out[: int(size)]


def _normalize_isaac_warehouse_obs(raw_obs: Any) -> Dict[str, JSONValue]:
    if isinstance(raw_obs, dict):
        obs_dict = dict(raw_obs)
    else:
        obs_dict = {"flat": _to_jsonable(raw_obs)}

    pos = _ensure_length(obs_dict.get("position", obs_dict.get("base_pos", [0.0, 0.0, 0.0])), 3)
    goal = _ensure_length(obs_dict.get("goal_position", obs_dict.get("goal", [0.0, 0.0, 0.0])), 3)
    vel = _ensure_length(obs_dict.get("velocity", obs_dict.get("base_velocity", [0.0, 0.0, 0.0])), 3)
    lidar = _ensure_length(obs_dict.get("lidar", obs_dict.get("lidar_scan", [])), 16)
    battery = [_safe_float(obs_dict.get("battery", obs_dict.get("battery_level", 1.0)), 1.0)]
    goal_delta = [float(goal[i] - pos[i]) for i in range(3)]
    flat = _ensure_length(list(pos) + list(goal) + list(goal_delta) + list(vel) + list(lidar) + list(battery), 26)
    return {
        "position": pos,
        "goal_position": goal,
        "goal_delta": goal_delta,
        "velocity": vel,
        "lidar": lidar,
        "battery": battery,
        "flat": flat,
    }


def _normalize_isaac_action(action: Any, *, action_space: SpaceSpec) -> JSONValue:
    if action_space.type == "continuous":
        raw = action
        if isinstance(raw, (int, float)):
            raw = [float(raw)]
        if not isinstance(raw, (list, tuple)):
            raw = _to_jsonable(raw)
        if isinstance(raw, (list, tuple)):
            vals = _ensure_length(raw, len(action_space.low or [0.0, 0.0]))
        else:
            vals = [0.0, 0.0]
        lows = list(action_space.low or [-1.0] * len(vals))
        highs = list(action_space.high or [1.0] * len(vals))
        out: list[float] = []
        for idx, val in enumerate(vals):
            lo = _safe_float(lows[idx] if idx < len(lows) else -1.0, -1.0)
            hi = _safe_float(highs[idx] if idx < len(highs) else 1.0, 1.0)
            out.append(max(lo, min(hi, _safe_float(val, 0.0))))
        return out
    try:
        return int(action)
    except Exception:
        return 0


def _space_from_external(space: Any) -> SpaceSpec:
    type_name = str(type(space).__name__).lower()
    if "discrete" in type_name and hasattr(space, "n"):
        return SpaceSpec(type="discrete", n=int(getattr(space, "n", 1) or 1))
    if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
        low = _to_jsonable(getattr(space, "low", []))
        high = _to_jsonable(getattr(space, "high", []))
        shape = tuple(int(v) for v in tuple(getattr(space, "shape", ()) or ()))
        if isinstance(low, list) and isinstance(high, list):
            return SpaceSpec(
                type="continuous",
                shape=shape or (len(low),),
                dtype=str(getattr(space, "dtype", "float32")),
                low=[_safe_float(v, -1.0) for v in low],
                high=[_safe_float(v, 1.0) for v in high],
            )
    if hasattr(space, "spaces") and isinstance(getattr(space, "spaces", None), Mapping):
        sub = {
            str(name): _space_from_external(subspace)
            for name, subspace in dict(getattr(space, "spaces")).items()
        }
        return SpaceSpec(type="dict", keys=list(sub.keys()), subspaces=sub)
    return SpaceSpec(type="dict", notes=f"Unsupported external space type: {type(space).__name__}")


@dataclass
class IsaacLabAdapterState:
    episode_step: int = 0
    last_obs: Optional[JSONValue] = None
    closed: bool = False


class IsaacLabVerseAdapter(Verse):
    def __init__(
        self,
        *,
        spec: VerseSpec,
        binding: IsaacLabVerseBinding,
        runtime_env: Any | None = None,
    ):
        self.spec = spec
        self.binding = binding
        self.task_config = binding.task_config_from_params(spec.params)
        self._state = IsaacLabAdapterState()
        self._env = runtime_env
        if self._env is None:
            require_isaaclab()
            self._env = self._build_runtime_env()
        if self._env is None:
            raise IsaacLabUnavailableError("Isaac Lab runtime environment was not created.")

        self.observation_space = (
            spec.observation_space
            or _space_from_external(getattr(self._env, "observation_space", None))
            or binding.observation_space
        )
        self.action_space = (
            spec.action_space
            or _space_from_external(getattr(self._env, "action_space", None))
            or binding.action_space
        )
        if self.observation_space.type == "dict" and not self.observation_space.subspaces:
            self.observation_space = binding.observation_space
        if self.action_space.type == "dict" and not self.action_space.subspaces and self.action_space.n is None:
            self.action_space = binding.action_space

    def _build_runtime_env(self) -> Any:
        try:
            import gymnasium as gym  # type: ignore
        except Exception as exc:
            raise IsaacLabUnavailableError(
                "Isaac Lab may be installed, but Gymnasium environment creation is unavailable. "
                "Install the Isaac Lab Python stack in the target environment."
            ) from exc

        kwargs = {
            "render_mode": str(self.task_config.render_mode),
        }
        kwargs.update(dict(self.task_config.extra))
        try:
            return gym.make(str(self.task_config.task_id), **kwargs)
        except TypeError:
            try:
                return gym.make(str(self.task_config.task_id))
            except Exception as exc:
                raise IsaacLabUnavailableError(
                    f"Failed to construct Isaac Lab task '{self.task_config.task_id}'."
                ) from exc
        except Exception as exc:
            raise IsaacLabUnavailableError(
                f"Failed to construct Isaac Lab task '{self.task_config.task_id}'."
            ) from exc

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        try:
            self._env.reset(seed=int(seed))
        except TypeError:
            if hasattr(self._env, "seed"):
                self._env.seed(int(seed))

    def reset(self) -> ResetResult:
        result = self._env.reset(seed=self.spec.seed)
        if isinstance(result, tuple) and len(result) >= 2:
            raw_obs, raw_info = result[0], result[1]
        else:
            raw_obs, raw_info = result, {}
        obs = _normalize_isaac_warehouse_obs(raw_obs)
        self._state.episode_step = 0
        self._state.last_obs = obs
        info = {"sim_provider": "isaaclab", "task_id": self.binding.task_id}
        if isinstance(raw_info, dict):
            info.update({str(k): _to_jsonable(v) for k, v in raw_info.items()})
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        norm_action = _normalize_isaac_action(action, action_space=self.action_space)
        result = self._env.step(norm_action)
        if not isinstance(result, tuple) or len(result) < 5:
            raise RuntimeError("Isaac Lab environment must return Gymnasium-style step tuple.")
        raw_obs, reward, terminated, truncated, raw_info = result[:5]
        obs = _normalize_isaac_warehouse_obs(raw_obs)
        self._state.episode_step += 1
        self._state.last_obs = obs
        info = {"sim_provider": "isaaclab", "task_id": self.binding.task_id}
        if isinstance(raw_info, dict):
            info.update({str(k): _to_jsonable(v) for k, v in raw_info.items()})
        return StepResult(
            obs=obs,
            reward=_safe_float(reward, 0.0),
            done=_safe_bool(terminated, False),
            truncated=_safe_bool(truncated, False),
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> Optional[Any]:
        try:
            return self._env.render()
        except TypeError:
            try:
                return self._env.render(mode=mode)
            except Exception:
                return None
        except Exception:
            return None

    def close(self) -> None:
        self._state.closed = True
        try:
            self._env.close()
        except Exception:
            return

    def export_state(self) -> Dict[str, JSONValue]:
        return {
            "episode_step": int(self._state.episode_step),
            "closed": bool(self._state.closed),
            "last_obs": self._state.last_obs,
            "task_id": self.binding.task_id,
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._state.episode_step = int(state.get("episode_step", 0) or 0)
        self._state.closed = bool(state.get("closed", False))
        self._state.last_obs = state.get("last_obs")
