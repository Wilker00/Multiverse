"""
tools/scaffold_extension.py

Generate starter files for a custom Verse + Agent extension.

This helps contributors add new environments and use cases quickly:
- creates `verses/<name>_world.py`
- creates `agents/<name>_agent.py`
- optionally patches registries to auto-register both.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _to_snake(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "").strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def _to_camel(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_") if part)


def _patch_agents_registry_text(text: str, *, agent_module: str, agent_class: str, algo_name: str) -> Tuple[str, bool]:
    out = text
    changed = False

    import_line = f"from agents.{agent_module} import {agent_class}"
    if import_line not in out:
        anchor = "from agents.evolving_agent import EvolvingAgent"
        if anchor in out:
            out = out.replace(anchor, anchor + "\n" + import_line)
            changed = True

    reg_call = (
        f"    register_agent(\n"
        f'        "{algo_name}",\n'
        f"        lambda s, o, a: {agent_class}(spec=s, observation_space=o, action_space=a),\n"
        f"    )"
    )
    if f'"{algo_name}"' not in out and reg_call not in out:
        anchor = (
            "    register_agent(\n"
            '        "evolving",\n'
            "        lambda s, o, a: EvolvingAgent(spec=s, observation_space=o, action_space=a),\n"
            "    )"
        )
        if anchor in out:
            out = out.replace(anchor, anchor + "\n" + reg_call)
            changed = True

    return out, changed


def _patch_verses_registry_text(
    text: str,
    *,
    verse_module: str,
    verse_factory: str,
    verse_name: str,
) -> Tuple[str, bool]:
    out = text
    changed = False

    import_line = f"    from verses.{verse_module} import {verse_factory}"
    if import_line not in out:
        anchor = "    from verses.uno_world import UnoWorldFactory"
        if anchor in out:
            out = out.replace(anchor, anchor + "\n" + import_line)
            changed = True

    entry = f'        "{verse_name}": {verse_factory}(),'
    if entry not in out:
        anchor = '        "uno_world": UnoWorldFactory(),'
        if anchor in out:
            out = out.replace(anchor, anchor + "\n" + entry)
            changed = True

    return out, changed


@dataclass
class ScaffoldSpec:
    raw_name: str
    slug: str
    camel: str
    verse_name: str
    algo_name: str
    verse_module: str
    verse_factory: str
    verse_class: str
    agent_module: str
    agent_class: str


def _build_spec(name: str) -> ScaffoldSpec:
    slug = _to_snake(name)
    if not slug:
        raise ValueError("Name must contain at least one alphanumeric character.")
    camel = _to_camel(slug)
    verse_module = f"{slug}_world"
    verse_class = f"{camel}WorldVerse"
    verse_factory = f"{camel}WorldFactory"
    agent_module = f"{slug}_agent"
    agent_class = f"{camel}Agent"
    return ScaffoldSpec(
        raw_name=name,
        slug=slug,
        camel=camel,
        verse_name=verse_module,
        algo_name=slug,
        verse_module=verse_module,
        verse_factory=verse_factory,
        verse_class=verse_class,
        agent_module=agent_module,
        agent_class=agent_class,
    )


def _agent_template(spec: ScaffoldSpec) -> str:
    return f'''"""
agents/{spec.agent_module}.py

Starter agent template for `{spec.algo_name}`.

This version extends tabular Q-learning so contributors can add custom behavior
without rebuilding training/saving plumbing from scratch.
"""

from __future__ import annotations

from typing import Dict

from agents.q_agent import QLearningAgent
from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult, ExperienceBatch


class {spec.agent_class}(QLearningAgent):
    """
    Starter custom agent.

    Suggested customization points:
    - override `act(...)` to add heuristics or tool usage.
    - override `learn(...)` to add reward shaping or custom metrics.
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        super().__init__(spec=spec, observation_space=observation_space, action_space=action_space)

    def act(self, obs: JSONValue) -> ActionResult:
        # Default behavior: keep base Q-learning action policy.
        return super().act(obs)

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        metrics = super().learn(batch)
        # Add custom metrics here when you extend behavior.
        metrics["custom_agent"] = "{spec.algo_name}"
        return metrics
'''


def _verse_template(spec: ScaffoldSpec) -> str:
    return f'''"""
verses/{spec.verse_module}.py

Starter verse template for `{spec.verse_name}`.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.types import JSONValue, SpaceSpec, VerseSpec
from core.verse_base import ResetResult, StepResult, Verse


@dataclass
class {spec.camel}Params:
    max_steps: int = 60
    start_pos: int = 0
    goal_pos: int = 9
    step_penalty: float = -0.05
    goal_reward: float = 1.0


class {spec.verse_class}:
    """
    Minimal 1D navigation verse:
    - action 0 = move left
    - action 1 = move right
    - episode ends on goal or max_steps
    """

    def __init__(self, spec: VerseSpec):
        tags = list(spec.tags)
        for t in {spec.verse_factory}().tags:
            if t not in tags:
                tags.append(t)
        self.spec = dataclasses.replace(spec, tags=tags)
        self.params = {spec.camel}Params(
            max_steps=max(1, int(self.spec.params.get("max_steps", 60))),
            start_pos=int(self.spec.params.get("start_pos", 0)),
            goal_pos=int(self.spec.params.get("goal_pos", 9)),
            step_penalty=float(self.spec.params.get("step_penalty", -0.05)),
            goal_reward=float(self.spec.params.get("goal_reward", 1.0)),
        )

        self.observation_space = self.spec.observation_space or SpaceSpec(
            type="dict",
            keys=["pos", "goal", "t"],
            subspaces={{
                "pos": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "goal": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
                "t": SpaceSpec(type="vector", shape=(1,), dtype="int32"),
            }},
            notes="{spec.verse_name} observation",
        )
        self.action_space = self.spec.action_space or SpaceSpec(
            type="discrete",
            n=2,
            notes="0=left, 1=right",
        )

        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._pos = int(self.params.start_pos)
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
        self._pos = int(self.params.start_pos)
        self._t = 0
        self._done = False
        return ResetResult(obs=self._obs(), info={{"seed": self._seed}})

    def step(self, action: JSONValue) -> StepResult:
        if self._done:
            return StepResult(self._obs(), 0.0, True, False, {{"warning": "step() called after done"}})

        a = max(0, min(1, int(action)))
        self._t += 1
        if a == 0:
            self._pos -= 1
        else:
            self._pos += 1

        reward = float(self.params.step_penalty)
        done = False
        info: Dict[str, JSONValue] = {{"action_used": int(a), "reached_goal": False}}
        if int(self._pos) == int(self.params.goal_pos):
            reward += float(self.params.goal_reward)
            done = True
            info["reached_goal"] = True

        truncated = bool(self._t >= int(self.params.max_steps) and not done)
        self._done = bool(done or truncated)
        return StepResult(obs=self._obs(), reward=float(reward), done=bool(done), truncated=bool(truncated), info=info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None
        left = min(int(self._pos), int(self.params.goal_pos))
        right = max(int(self._pos), int(self.params.goal_pos))
        cells = []
        for i in range(left, right + 1):
            ch = "."
            if i == int(self.params.goal_pos):
                ch = "G"
            if i == int(self._pos):
                ch = "A"
            cells.append(ch)
        return "".join(cells)

    def close(self) -> None:
        return

    def export_state(self) -> Dict[str, JSONValue]:
        return {{"pos": int(self._pos), "t": int(self._t), "done": bool(self._done)}}

    def import_state(self, state: Dict[str, JSONValue]) -> None:
        self._pos = int(state.get("pos", self._pos))
        self._t = max(0, int(state.get("t", self._t)))
        self._done = bool(state.get("done", self._done))

    def _obs(self) -> Dict[str, JSONValue]:
        return {{"pos": int(self._pos), "goal": int(self.params.goal_pos), "t": int(self._t)}}


class {spec.verse_factory}:
    @property
    def tags(self) -> List[str]:
        return ["custom", "starter"]

    def create(self, spec: VerseSpec) -> Verse:
        return {spec.verse_class}(spec)
'''


def _write_file(path: Path, content: str, *, force: bool) -> str:
    if path.exists() and not force:
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "written"


def scaffold(
    *,
    project_root: Path,
    name: str,
    agent_only: bool,
    verse_only: bool,
    register: bool,
    force: bool,
) -> int:
    spec = _build_spec(name)
    create_agent = not bool(verse_only)
    create_verse = not bool(agent_only)

    if not create_agent and not create_verse:
        raise ValueError("Nothing to scaffold. Do not pass both --agent_only and --verse_only.")

    created = 0
    if create_agent:
        agent_fp = project_root / "agents" / f"{spec.agent_module}.py"
        status = _write_file(agent_fp, _agent_template(spec), force=force)
        print(f"{status}: {agent_fp.as_posix()}")
        created += 1 if status == "written" else 0

    if create_verse:
        verse_fp = project_root / "verses" / f"{spec.verse_module}.py"
        status = _write_file(verse_fp, _verse_template(spec), force=force)
        print(f"{status}: {verse_fp.as_posix()}")
        created += 1 if status == "written" else 0

    if register:
        if create_agent:
            ar = project_root / "agents" / "registry.py"
            if ar.is_file():
                old = ar.read_text(encoding="utf-8")
                new, changed = _patch_agents_registry_text(
                    old,
                    agent_module=spec.agent_module,
                    agent_class=spec.agent_class,
                    algo_name=spec.algo_name,
                )
                if changed:
                    ar.write_text(new, encoding="utf-8")
                    print(f"patched: {ar.as_posix()}")
        if create_verse:
            vr = project_root / "verses" / "registry.py"
            if vr.is_file():
                old = vr.read_text(encoding="utf-8")
                new, changed = _patch_verses_registry_text(
                    old,
                    verse_module=spec.verse_module,
                    verse_factory=spec.verse_factory,
                    verse_name=spec.verse_name,
                )
                if changed:
                    vr.write_text(new, encoding="utf-8")
                    print(f"patched: {vr.as_posix()}")

    print("")
    print("Scaffold Summary")
    print(f"- name: {spec.raw_name}")
    if create_agent:
        print(f"- agent algo: {spec.algo_name}")
        print(f"- agent class: {spec.agent_class}")
    if create_verse:
        print(f"- verse name: {spec.verse_name}")
        print(f"- verse factory: {spec.verse_factory}")
    print(f"- registry auto-patch: {bool(register)}")

    return created


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, help="Base name, e.g. 'warehouse_planner' or 'soccer'")
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--agent_only", action="store_true")
    ap.add_argument("--verse_only", action="store_true")
    ap.add_argument("--register", action="store_true", help="Patch agents/registry.py and verses/registry.py")
    ap.add_argument("--force", action="store_true", help="Overwrite generated files if they already exist")
    args = ap.parse_args()

    created = scaffold(
        project_root=Path(args.project_root),
        name=str(args.name),
        agent_only=bool(args.agent_only),
        verse_only=bool(args.verse_only),
        register=bool(args.register),
        force=bool(args.force),
    )
    if created <= 0:
        print("No new files written.")


if __name__ == "__main__":
    main()

