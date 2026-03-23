"""
integrations/pettingzoo/adapter.py

Verse adapter for PettingZoo AEC (Agent Environment Cycle) environments.

Exposes a single-agent Verse view over a multi-agent AEC environment.
The caller controls the "focal agent"; all other agents are auto-played
using random policies drawn from their respective action spaces.

Safe to import when PettingZoo is unavailable — the external import
only happens at construction time via require_pettingzoo().
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


class PettingZooUnavailableError(RuntimeError):
    pass


def pettingzoo_available() -> bool:
    try:
        return importlib.util.find_spec("pettingzoo") is not None
    except Exception:
        return False


def build_pettingzoo_install_hint() -> str:
    return "PettingZoo integration is optional. Install with: pip install pettingzoo"


def require_pettingzoo() -> None:
    if pettingzoo_available():
        return
    raise PettingZooUnavailableError(build_pettingzoo_install_hint())


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


def _space_from_pettingzoo(space: Any) -> SpaceSpec:
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
    return SpaceSpec(type="dict", notes=f"Unsupported pettingzoo space type: {type(space).__name__}")


def _get_agent_space(env: Any, agent_id: str, space_attr: str) -> Any:
    """Retrieve observation_space or action_space for an agent, handling both callable and dict forms."""
    attr = getattr(env, space_attr, None)
    if attr is None:
        return None
    if callable(attr):
        try:
            return attr(agent_id)
        except Exception:
            return None
    if isinstance(attr, dict):
        return attr.get(agent_id)
    return None


def _sample_action(env: Any, agent_id: str) -> Any:
    space = _get_agent_space(env, agent_id, "action_space")
    if space is not None and hasattr(space, "sample"):
        try:
            return space.sample()
        except Exception:
            pass
    return None


@dataclass
class PettingZooAdapterState:
    episode_step: int = 0
    last_obs: Optional[JSONValue] = None
    closed: bool = False
    all_agents: List[str] = field(default_factory=list)


class PettingZooVerseAdapter(Verse):
    """
    Wraps a PettingZoo AEC environment as a single-agent Verse.

    The focal_agent is the agent the Verse controller plays.
    - Taken from spec.params["focal_agent"] if provided.
    - Defaults to the first agent in env.possible_agents (or env.agents after reset).

    All other agents are auto-played with random actions each cycle.
    The focal agent's reward from env.last() is returned in step().

    A runtime_env can be injected (e.g., in tests) to bypass the real
    PettingZoo dependency.
    """

    def __init__(
        self,
        *,
        spec: VerseSpec,
        runtime_env: Any = None,
        focal_agent: Optional[str] = None,
    ) -> None:
        self.spec = spec
        self._state = PettingZooAdapterState()
        self._focal_agent_override = (
            focal_agent
            or str(spec.params.get("focal_agent") or "")
            or None
        )

        self._env = runtime_env
        if self._env is None:
            require_pettingzoo()
            raise PettingZooUnavailableError(
                "PettingZoo runtime_env must be provided directly (env_id construction "
                "is env-specific — pass runtime_env=your_env to the adapter)."
            )

        # Determine focal agent after reset (possible_agents available pre-reset)
        possible = list(getattr(self._env, "possible_agents", []) or [])
        self._focal_agent: str = self._focal_agent_override or (possible[0] if possible else "agent_0")

        focal_obs_space = _get_agent_space(self._env, self._focal_agent, "observation_space")
        focal_act_space = _get_agent_space(self._env, self._focal_agent, "action_space")

        self.observation_space: SpaceSpec = (
            spec.observation_space
            or _space_from_pettingzoo(focal_obs_space)
        )
        self.action_space: SpaceSpec = (
            spec.action_space
            or _space_from_pettingzoo(focal_act_space)
        )

    # ------------------------------------------------------------------
    # AEC cycle helpers
    # ------------------------------------------------------------------

    def _advance_to_focal(self) -> Optional[Dict[str, Any]]:
        """
        Advance the AEC cycle until it is the focal agent's turn.

        Returns a dict {obs, reward, done, truncated, info} for the focal agent,
        or None if the episode ended before the focal agent got a turn.
        """
        env = self._env
        while list(getattr(env, "agents", []) or []):
            agent = getattr(env, "agent_selection", None)
            if agent is None:
                break

            obs, reward, term, trunc, info = env.last()

            if agent == self._focal_agent:
                return {
                    "obs": obs,
                    "reward": reward,
                    "done": bool(term),
                    "truncated": bool(trunc),
                    "info": info,
                }

            # Auto-play non-focal agent
            action = None if (term or trunc) else _sample_action(env, agent)
            env.step(action)

        return None

    # ------------------------------------------------------------------
    # Verse contract
    # ------------------------------------------------------------------

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        try:
            self._env.reset(seed=int(seed))
        except TypeError:
            pass

    def reset(self) -> ResetResult:
        try:
            self._env.reset(seed=self.spec.seed)
        except TypeError:
            self._env.reset()

        self._state.episode_step = 0
        self._state.all_agents = list(getattr(self._env, "agents", []) or [])

        # Confirm focal agent is present; fall back to first agent if not
        if self._focal_agent not in self._state.all_agents and self._state.all_agents:
            if not self._focal_agent_override:
                self._focal_agent = self._state.all_agents[0]

        result = self._advance_to_focal()
        if result is None:
            obs: JSONValue = {}
            info: Dict[str, JSONValue] = {
                "sim_provider": "pettingzoo",
                "focal_agent": self._focal_agent,
                "early_done": True,
            }
        else:
            obs = _to_jsonable(result["obs"])
            raw_info = result.get("info") or {}
            info = {"sim_provider": "pettingzoo", "focal_agent": self._focal_agent}
            if isinstance(raw_info, dict):
                info.update({str(k): _to_jsonable(v) for k, v in raw_info.items()})

        self._state.last_obs = obs
        return ResetResult(obs=obs, info=info)

    def step(self, action: JSONValue) -> StepResult:
        env = self._env
        agents = list(getattr(env, "agents", []) or [])

        if self._focal_agent in agents:
            env.step(action)
        # After focal agent acts, advance to their next turn (or episode end)
        result = self._advance_to_focal()

        if result is None:
            obs: JSONValue = self._state.last_obs or {}
            reward = 0.0
            done = True
            truncated = False
            info: Dict[str, JSONValue] = {
                "sim_provider": "pettingzoo",
                "focal_agent": self._focal_agent,
                "episode_done": True,
            }
        else:
            obs = _to_jsonable(result["obs"])
            reward = _safe_float(result["reward"], 0.0)
            done = bool(result["done"])
            truncated = bool(result["truncated"])
            raw_info = result.get("info") or {}
            info = {"sim_provider": "pettingzoo", "focal_agent": self._focal_agent}
            if isinstance(raw_info, dict):
                info.update({str(k): _to_jsonable(v) for k, v in raw_info.items()})

        self._state.episode_step += 1
        self._state.last_obs = obs
        return StepResult(obs=obs, reward=reward, done=done, truncated=truncated, info=info)

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
            "focal_agent": self._focal_agent,
            "all_agents": list(self._state.all_agents),
        }

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._state.episode_step = int(state.get("episode_step", 0) or 0)
        self._state.closed = bool(state.get("closed", False))
        self._state.last_obs = state.get("last_obs")
        raw_agents = state.get("all_agents")
        if isinstance(raw_agents, list):
            self._state.all_agents = [str(a) for a in raw_agents]


class PettingZooVerseFactory:
    """
    Placeholder factory for PettingZoo verses.
    Because PettingZoo env construction is env-specific, the factory
    requires a runtime_env_builder callable in spec.params["runtime_env_builder"]
    or the env to be injected directly.
    """

    @property
    def tags(self) -> List[str]:
        return ["pettingzoo", "multi_agent", "adapter", "external"]

    def create(self, spec: VerseSpec) -> PettingZooVerseAdapter:
        builder = spec.params.get("runtime_env_builder")
        if callable(builder):
            env = builder()
        else:
            raise PettingZooUnavailableError(
                "PettingZooVerseFactory requires spec.params['runtime_env_builder'] "
                "to be a callable that returns a PettingZoo AEC environment."
            )
        return PettingZooVerseAdapter(spec=spec, runtime_env=env)
