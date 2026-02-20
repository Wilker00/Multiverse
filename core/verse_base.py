"""
core/verse_base.py

Minimal environment interface for u.ai ("Verse").
...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from core.types import JSONValue, SpaceSpec, VerseSpec


@dataclass
class ResetResult:
    # ... (existing implementation)
    obs: JSONValue
    info: Dict[str, JSONValue]


@dataclass
class StepResult:
    # ... (existing implementation)
    obs: JSONValue
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, JSONValue]


class Verse(Protocol):
    # ... (existing implementation)
    spec: VerseSpec
    observation_space: SpaceSpec
    action_space: SpaceSpec

    def seed(self, seed: Optional[int]) -> None:
        ...

    def reset(self) -> ResetResult:
        ...

    def step(self, action: JSONValue) -> StepResult:
        ...

    def render(self, mode: str = "rgb_array") -> Optional[Any]:
        ...

    def close(self) -> None:
        ...

    # Optional state checkpoint hooks used by SafeExecutor.
    def export_state(self) -> Dict[str, JSONValue]:
        ...

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        ...


class VerseFactory(Protocol):
    """
    Factory protocol so the orchestrator can build verses from VerseSpec.
    """
    @property
    def tags(self) -> List[str]:
        """A list of tags describing the challenges offered by this Verse."""
        ...

    def create(self, spec: VerseSpec) -> Verse:
        """
        Create a Verse instance from a VerseSpec.
        """

# ... (rest of the file is unchanged)
def require_jsonable(value: JSONValue, label: str) -> None:
    """
    Lightweight sanity check for common non-JSON mistakes.
    We do not deeply validate here to keep things fast.
    """
    # Basic type check only
    allowed = (dict, list, str, int, float, bool, type(None))
    if not isinstance(value, allowed):
        raise TypeError(f"{label} must be JSON-serializable type, got {type(value)}")


def validate_reset_result(result: ResetResult) -> None:
    require_jsonable(result.obs, "reset.obs")
    if not isinstance(result.info, dict):
        raise TypeError("reset.info must be a dict[str, JSONValue]")


def validate_step_result(result: StepResult) -> None:
    require_jsonable(result.obs, "step.obs")
    if not isinstance(result.info, dict):
        raise TypeError("step.info must be a dict[str, JSONValue]")
    if not isinstance(result.reward, (int, float)):
        raise TypeError("step.reward must be a number")
    if not isinstance(result.done, bool):
        raise TypeError("step.done must be bool")
    if not isinstance(result.truncated, bool):
        raise TypeError("step.truncated must be bool")
