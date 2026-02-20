import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agents.gateway_agent import GatewayAgent
from agents.special_moe_agent import SpecialMoEAgent
from core.types import AgentSpec, SpaceSpec


def _obs_space() -> SpaceSpec:
    return SpaceSpec(type="dict")


def _act_space() -> SpaceSpec:
    return SpaceSpec(type="discrete", n=2)


class TestGatewayConfigValidation(unittest.TestCase):
    def test_gateway_rejects_unknown_config_keys_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump({}, f)
            spec = AgentSpec(
                spec_version="v1",
                policy_id="gw_test",
                policy_version="0.1",
                algo="gateway",
                config={
                    "verse_name": "line_world",
                    "manifest_path": manifest,
                    "typo_key": 1,
                },
            )
            with patch.object(GatewayAgent, "_load_delegate", return_value=None):
                with self.assertRaises(ValueError):
                    GatewayAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space())

    def test_gateway_allows_unknown_keys_when_strict_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump({}, f)
            spec = AgentSpec(
                spec_version="v1",
                policy_id="gw_test",
                policy_version="0.1",
                algo="gateway",
                config={
                    "verse_name": "line_world",
                    "manifest_path": manifest,
                    "strict_config_validation": False,
                    "typo_key": 1,
                },
            )
            with patch.object(GatewayAgent, "_load_delegate", return_value=None):
                agent = GatewayAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space())
            self.assertEqual(agent.verse_name, "line_world")


class TestSpecialMoEConfigValidation(unittest.TestCase):
    def test_special_moe_rejects_unknown_config_keys_by_default(self):
        spec = AgentSpec(
            spec_version="v1",
            policy_id="moe_test",
            policy_version="0.1",
            algo="special_moe",
            config={"unknown_key": 123},
        )
        with self.assertRaises(ValueError):
            SpecialMoEAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space())

    def test_special_moe_rejects_unknown_nested_expert_lookup_keys(self):
        spec = AgentSpec(
            spec_version="v1",
            policy_id="moe_test",
            policy_version="0.1",
            algo="special_moe",
            config={"expert_lookup_config": {"typo_option": True}},
        )
        with self.assertRaises(ValueError):
            SpecialMoEAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space())

    def test_special_moe_allows_unknown_when_strict_disabled(self):
        spec = AgentSpec(
            spec_version="v1",
            policy_id="moe_test",
            policy_version="0.1",
            algo="special_moe",
            config={"strict_config_validation": False, "unknown_key": 123},
        )
        agent = SpecialMoEAgent(spec=spec, observation_space=_obs_space(), action_space=_act_space())
        self.assertEqual(agent.top_k, 2)


if __name__ == "__main__":
    unittest.main()

