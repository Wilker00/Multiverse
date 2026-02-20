import tempfile
import unittest
from pathlib import Path

from tools.scaffold_extension import (
    _build_spec,
    _patch_agents_registry_text,
    _patch_verses_registry_text,
    scaffold,
)


class TestScaffoldExtension(unittest.TestCase):
    def test_patch_agents_registry_text(self):
        base = """from agents.evolving_agent import EvolvingAgent

def register_builtin_agents() -> None:
    register_agent(
        "evolving",
        lambda s, o, a: EvolvingAgent(spec=s, observation_space=o, action_space=a),
    )
"""
        out, changed = _patch_agents_registry_text(
            base,
            agent_module="demo_agent",
            agent_class="DemoAgent",
            algo_name="demo",
        )
        self.assertTrue(changed)
        self.assertIn("from agents.demo_agent import DemoAgent", out)
        self.assertIn('"demo"', out)
        self.assertIn("DemoAgent(spec=s, observation_space=o, action_space=a)", out)

    def test_patch_verses_registry_text(self):
        base = """def register_builtin() -> None:
    from verses.uno_world import UnoWorldFactory

    builtins = {
        "uno_world": UnoWorldFactory(),
    }
"""
        out, changed = _patch_verses_registry_text(
            base,
            verse_module="demo_world",
            verse_factory="DemoWorldFactory",
            verse_name="demo_world",
        )
        self.assertTrue(changed)
        self.assertIn("from verses.demo_world import DemoWorldFactory", out)
        self.assertIn('"demo_world": DemoWorldFactory(),', out)

    def test_scaffold_generates_files_and_registry_patches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "agents").mkdir(parents=True, exist_ok=True)
            (root / "verses").mkdir(parents=True, exist_ok=True)
            (root / "agents" / "registry.py").write_text(
                """from agents.evolving_agent import EvolvingAgent

def register_builtin_agents() -> None:
    register_agent(
        "evolving",
        lambda s, o, a: EvolvingAgent(spec=s, observation_space=o, action_space=a),
    )
""",
                encoding="utf-8",
            )
            (root / "verses" / "registry.py").write_text(
                """def register_builtin() -> None:
    from verses.uno_world import UnoWorldFactory

    builtins = {
        "uno_world": UnoWorldFactory(),
    }
""",
                encoding="utf-8",
            )

            created = scaffold(
                project_root=root,
                name="demo",
                agent_only=False,
                verse_only=False,
                register=True,
                force=False,
            )
            self.assertEqual(created, 2)

            spec = _build_spec("demo")
            self.assertTrue((root / "agents" / f"{spec.agent_module}.py").is_file())
            self.assertTrue((root / "verses" / f"{spec.verse_module}.py").is_file())

            ar = (root / "agents" / "registry.py").read_text(encoding="utf-8")
            vr = (root / "verses" / "registry.py").read_text(encoding="utf-8")
            self.assertIn(f'from agents.{spec.agent_module} import {spec.agent_class}', ar)
            self.assertIn(f'"{spec.algo_name}"', ar)
            self.assertIn(f"from verses.{spec.verse_module} import {spec.verse_factory}", vr)
            self.assertIn(f'"{spec.verse_name}": {spec.verse_factory}(),', vr)


if __name__ == "__main__":
    unittest.main()

