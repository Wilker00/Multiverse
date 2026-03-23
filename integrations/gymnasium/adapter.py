"""
integrations/gymnasium/adapter.py

Verse adapter for Gymnasium environments.

Safe to import when Gymnasium is unavailable — the external import
only happens at construction time via require_gymnasium().
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


class GymnasiumUnavailableError(RuntimeError):
    pass


def gymnasium_available() -> bool:
    try:
        return importlib.util.find_spec("gymnasium") is not None
    except Exception:
        return False


def build_gymnasium_install_hint() -> str:
    return "Gymnasium integration is optional. Install with: pip install gymnasium"


def require_gymnasium() -> None:
    if gymnasium_available():
        return
    raise GymnasiumUnavailableError(build_gymnasium_install_hint())


def _to_jsonable(value: Any) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
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
        return float(default) if math.isnan(out) else out
    except Exception:
        return float(default)


def _space_from_gymnasium(space: Any) -> SpaceSpec:
    if space is None:
        return SpaceSpec(type="dict", notes="No space provided")
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
    if hasattr(space, "spaces") and isinstance(getattr(space, "spaces", None), dict):
        sub = {
            str(name): _space_from_gymnasium(subspace)
            for name, subspace in dict(getattr(space, "spaces")).items()
        }
        return SpaceSpec(type="dict", keys=list(sub.keys()), subspaces=sub)
    return SpaceSpec(type="dict", notes=f"Unsupported gymnasium space type: {type(space).__name__}")


@dataclass
class GymnasiumAdapterState:
    episode_step: int = 0
    last_obs: Optional[JSONValue] = None
    closed: bool = False


class GymnasiumVerseAdapter(Verse):
    """
    Wraps any Gymnasium environment as a Verse.

    The gymnasium env_id is taken from spec.params["env_id"], falling back
    to spec.verse_name. A runtime_env can be injected (e.g., in tests)
    to bypass the real Gymnasium dependency.
    """

    def __init__(
        self,
        *,
        spec: VerseSpec,
        runtime_env: Any = None,
    ) -> None:
        self.spec = spec
        self._state = GymnasiumAdapterState()
        self._env_id = str(spec.params.get("env_id") or spec.verse_name)
        self._render_mode = str(spec.params.get("render_mode") or "rgb_array")

        self._env = runtime_env
        if self._env is None:
            require_gymnasium()
            self._env = self._build_env()

        self.observation_space: SpaceSpec = (
            spec.observation_space
            or _space_from_gymnasium(getattr(self._env, "observation_space", None))
        )
        self.action_space: SpaceSpec = (
            spec.action_space
            or _space_from_gymnasium(getattr(self._env, "action_space", None))
        )

    def _build_env(self) -> Any:
        import gymnasium as gym  # type: ignore
        try:
            return gym.make(self._env_id, render_mode=self._render_mode)
        except TypeError:
            return gym.make(self._env_id)

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        try:
            self._env.reset(seed=int(seed))
        except TypeError:
            if hasattr(self._env, "seed"):
                self._env.seed(int(seed))

    def reset(self) -> ResetResult:
        try:
            result = self._env.reset(seed=self.spec.seed)
        except TypeError:
            result = self._env.reset()
        if isinstance(result, tuple) and len(result) >= 2:
            raw_obs, raw_info = result[0], result[1]
        else:
            raw_obs, raw_info = result, {}
        obs: JSONValue = _to_jsonable(raw_obs)
        self._state.episode_step = 0
        self._state.last_obs = obs
        info: Dict[str, JSONValue] = {"sim_provider": "gymnasium", "env_id": self._env_id}
        if isinstance(raw_info, dict):
            info.update({str(k): _to_jsonable(v) for k, v in raw_info.items()})
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        result = self._env.step(action)
        if not isinstance(result, tuple) or len(result) < 4:
            raise RuntimeError(
                "Gymnasium environment must return a tuple of (obs, reward, terminated, [truncated,] info)."
            )
        if len(result) >= 5:
            raw_obs, reward, terminated, truncated, raw_info = result[:5]
        else:
            raw_obs, reward, terminated, raw_info = result[:4]
            truncated = False
        obs: JSONValue = _to_jsonable(raw_obs)
        self._state.episode_step += 1
        self._state.last_obs = obs
        info: Dict[str, JSONValue] = {"sim_provider": "gymnasium", "env_id": self._env_id}
        if isinstance(raw_info, dict):
            info.update({str(k): _to_jsonable(v) for k, v in raw_info.items()})
        return StepResult(
            obs=obs,
            reward=_safe_float(reward, 0.0),
            done=bool(terminated),
            truncated=bool(truncated),
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
            "env_id": self._env_id,
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._state.episode_step = int(state.get("episode_step", 0) or 0)
        self._state.closed = bool(state.get("closed", False))
        self._state.last_obs = state.get("last_obs")


class GymnasiumVerseFactory:
    """
    Factory that creates GymnasiumVerseAdapters from a VerseSpec.
    The spec.params["env_id"] field selects the Gymnasium environment.
    """

    @property
    def tags(self) -> List[str]:
        return ["gymnasium", "adapter", "external"]

    def create(self, spec: VerseSpec) -> GymnasiumVerseAdapter:
        return GymnasiumVerseAdapter(spec=spec)
