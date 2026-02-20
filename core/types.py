"""
core/types.py

Core data contracts for u.ai.
Everything else should depend on these types, not the other way around.

Design goals:
- Stable, explicit schemas
- JSON serializable
- Versioned
- Minimal assumptions about frameworks (Gym, SB3, RLlib, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import time
import uuid


# Type aliases for JSON-compatible data structures
JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]


def now_ms() -> int:
    """Returns the current time in milliseconds."""
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    """Generates a unique ID with a given prefix (e.g., 'run_39a8...')."""
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class RunRef:
    """
    Reference to a specific execution run.
    Used to group related episodes and events together.
    """
    run_id: str
    created_at_ms: int = field(default_factory=now_ms)

    @staticmethod
    def create() -> "RunRef":
        """Creates a new RunRef with a unique ID."""
        return RunRef(run_id=new_id("run"))


@dataclass(frozen=True)
class AgentRef:
    """
    Reference to an instantiated Agent.
    Links a runtime agent ID to its policy definition.
    """
    agent_id: str
    policy_id: str
    policy_version: str

    @staticmethod
    def create(policy_id: str, policy_version: str) -> "AgentRef":
        """Creates a new AgentRef with a unique ID."""
        return AgentRef(agent_id=new_id("agent"), policy_id=policy_id, policy_version=policy_version)


@dataclass(frozen=True)
class VerseRef:
    """
    Reference to an instantiated Verse (Environment).
    Links a runtime verse ID to its specification.
    """
    verse_id: str
    verse_name: str
    verse_version: str
    spec_hash: str

    @staticmethod
    def create(verse_name: str, verse_version: str, spec_hash: str) -> "VerseRef":
        """Creates a new VerseRef with a unique ID."""
        return VerseRef(
            verse_id=new_id("verse"),
            verse_name=verse_name,
            verse_version=verse_version,
            spec_hash=spec_hash,
        )


ObsType = Literal["vector", "image", "text", "dict"]
ActionType = Literal["discrete", "continuous", "multi_discrete", "dict"]


@dataclass(frozen=True)
class SpaceSpec:
    """
    Lightweight, serializable description of an observation or action space.
    
    This acts as a portable schema that can be sent over the wire or stored in a DB,
    unlike native Gym Space objects which are Python-specific.
    """
    type: Union[ObsType, ActionType]
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None

    # For Discrete / MultiDiscrete
    n: Optional[int] = None
    
    # For Continuous (Box)
    low: Optional[List[float]] = None
    high: Optional[List[float]] = None

    # For Dict spaces
    keys: Optional[List[str]] = None
    subspaces: Optional[Dict[str, "SpaceSpec"]] = None

    notes: Optional[str] = None


@dataclass(frozen=True)
class VerseSpec:
    """
    Configuration specification for generating a Verse (Environment).
    
    This contains all parameters needed to reconstruct the environment.
    The curriculum layer and generator layer will emit these specs.
    """
    spec_version: str
    verse_name: str
    verse_version: str

    seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    observation_space: Optional[SpaceSpec] = None
    action_space: Optional[SpaceSpec] = None

    # Environment-specific parameters
    params: Dict[str, JSONValue] = field(default_factory=dict)
    metadata: Dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def evolved(self, **changes: Any) -> "VerseSpec":
        """
        Immutable-friendly update helper.
        Always returns a fresh VerseSpec with copied mutable containers.
        """
        # Ensure we don't mutate the original's lists/dicts
        new_tags = list(changes.get("tags", self.tags))
        new_params = dict(changes.get("params", self.params))
        new_metadata = dict(changes.get("metadata", self.metadata))
        
        # Remove them from changes so replace doesn't get double arguments
        filtered_changes = {k: v for k, v in changes.items() if k not in ("tags", "params", "metadata")}
        return replace(self, tags=new_tags, params=new_params, metadata=new_metadata, **filtered_changes)

    def with_tags(self, tags: List[str]) -> "VerseSpec":
        return self.evolved(tags=list(tags))

    def with_params(self, updates: Dict[str, JSONValue]) -> "VerseSpec":
        merged = dict(self.params)
        merged.update(dict(updates))
        return self.evolved(params=merged)

    def with_metadata(self, updates: Dict[str, JSONValue]) -> "VerseSpec":
        merged = dict(self.metadata)
        merged.update(dict(updates))
        return self.evolved(metadata=merged)


@dataclass(frozen=True)
class AgentSpec:
    """
    Configuration specification for creating an Agent.
    
    Contains the policy identifier, algorithm choice, and hyperparameters.
    """
    spec_version: str
    policy_id: str
    policy_version: str

    algo: str
    framework: Optional[str] = None  # e.g., "sb3", "rllib", "torch"

    seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    # Algorithm-specific configuration (hyperparameters)
    config: Dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def evolved(self, **changes: Any) -> "AgentSpec":
        """
        Immutable-friendly update helper.
        Always returns a fresh AgentSpec with copied mutable containers.
        """
        payload: Dict[str, Any] = {
            "tags": list(self.tags) if self.tags is not None else [],
            "config": dict(self.config) if self.config is not None else {},
        }
        payload.update(changes)
        if "tags" in payload and payload["tags"] is not None:
            payload["tags"] = list(payload["tags"])
        if "config" in payload and payload["config"] is not None:
            payload["config"] = dict(payload["config"])
        return replace(self, **payload)

    def with_config(self, updates: Dict[str, JSONValue]) -> "AgentSpec":
        merged = dict(self.config)
        merged.update(dict(updates))
        return self.evolved(config=merged)


@dataclass(frozen=True)
class StepEvent:
    """
    The atomic unit of experience: (State, Action, Reward, Next State, Done).
    
    Designed to be JSONL friendly for logging and offline analysis.
    Can be converted to columnar formats (Parquet) for high-performance training.
    """
    schema_version: str

    # Context
    run_id: str
    t_ms: int

    episode_id: str
    step_idx: int

    # Agent Context
    agent_id: str
    policy_id: str
    policy_version: str

    # Environment Context
    verse_id: str
    verse_name: str
    verse_version: str
    spec_hash: str

    seed: Optional[int] = None

    # Interaction Data
    obs: JSONValue = None
    action: JSONValue = None
    reward: float = 0.0
    done: bool = False
    truncated: bool = False

    # Extra metadata
    info: Dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def new_episode_id() -> str:
        return new_id("ep")


@dataclass(frozen=True)
class EpisodeSummary:
    """
    High-level summary of a completed episode.
    
    Useful for quick evaluation metrics, leaderboards, and memory indexing.
    """
    schema_version: str
    run_id: str
    episode_id: str

    agent_id: str
    policy_id: str
    policy_version: str

    verse_name: str
    verse_version: str
    spec_hash: str

    # Metrics
    steps: int
    return_sum: float
    start_ms: int
    end_ms: int

    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_space_spec(spec: SpaceSpec) -> None:
    """
    Validates that the SpaceSpec contains the necessary fields for its type.
    Raises ValueError if the spec is invalid.
    """
    if spec.type in ("vector", "image") and not spec.shape:
        raise ValueError("SpaceSpec.shape is required for vector or image types")
    if spec.type == "discrete" and spec.n is None:
        raise ValueError("SpaceSpec.n is required for discrete actions")
    if spec.type == "continuous" and (spec.low is None or spec.high is None):
        raise ValueError("SpaceSpec.low and SpaceSpec.high are required for continuous actions")
    if spec.type == "dict" and (not spec.keys or not spec.subspaces):
        raise ValueError("SpaceSpec.keys and SpaceSpec.subspaces are required for dict type")


def validate_verse_spec(spec: VerseSpec) -> None:
    """
    Strict VerseSpec validation for safer interop with external RL tooling.
    """
    if not str(spec.spec_version).strip():
        raise ValueError("VerseSpec.spec_version is required")
    if not str(spec.verse_name).strip():
        raise ValueError("VerseSpec.verse_name is required")
    if not str(spec.verse_version).strip():
        raise ValueError("VerseSpec.verse_version is required")
    if spec.observation_space is not None:
        validate_space_spec(spec.observation_space)
    if spec.action_space is not None:
        validate_space_spec(spec.action_space)

    if "max_steps" in spec.params:
        ms = spec.params.get("max_steps")
        try:
            if int(ms) <= 0:
                raise ValueError("VerseSpec.params.max_steps must be > 0")
        except Exception as e:
            raise ValueError("VerseSpec.params.max_steps must be an integer > 0") from e


def validate_agent_spec(spec: AgentSpec) -> None:
    if not str(spec.spec_version).strip():
        raise ValueError("AgentSpec.spec_version is required")
    if not str(spec.policy_id).strip():
        raise ValueError("AgentSpec.policy_id is required")
    if not str(spec.policy_version).strip():
        raise ValueError("AgentSpec.policy_version is required")
    if not str(spec.algo).strip():
        raise ValueError("AgentSpec.algo is required")


def make_step_event(
    *,
    schema_version: str,
    run: RunRef,
    episode_id: str,
    step_idx: int,
    agent: AgentRef,
    verse: VerseRef,
    obs: JSONValue,
    action: JSONValue,
    reward: float,
    done: bool,
    truncated: bool = False,
    seed: Optional[int] = None,
    info: Optional[Dict[str, JSONValue]] = None,
) -> StepEvent:
    """
    Factory function to create a StepEvent from current context and interaction data.
    """
    return StepEvent(
        schema_version=schema_version,
        run_id=run.run_id,
        t_ms=now_ms(),
        episode_id=episode_id,
        step_idx=step_idx,
        agent_id=agent.agent_id,
        policy_id=agent.policy_id,
        policy_version=agent.policy_version,
        verse_id=verse.verse_id,
        verse_name=verse.verse_name,
        verse_version=verse.verse_version,
        spec_hash=verse.spec_hash,
        seed=seed,
        obs=obs,
        action=action,
        reward=float(reward),
        done=bool(done),
        truncated=bool(truncated),
        info=info or {},
    )
