import unittest
import os

from tools.deploy_agent import (
    _candidate_cfg,
    _validate_gateway_cfg,
    _validate_special_moe_cfg,
)


class TestDeployAgentConfigValidation(unittest.TestCase):
    def test_validate_gateway_cfg_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            _validate_gateway_cfg({"manifest_path": "models/default_policy_set.json", "typo_key": True})

    def test_validate_gateway_cfg_accepts_minimal_known_keys(self):
        out = _validate_gateway_cfg({"manifest_path": "models/default_policy_set.json"})
        self.assertEqual(out["manifest_path"], "models/default_policy_set.json")

    def test_validate_special_moe_cfg_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            _validate_special_moe_cfg({"expert_dataset_dir": "models/expert_datasets", "bad_typo": 1})

    def test_candidate_cfg_is_validated(self):
        out = _candidate_cfg("line_world", top_k=3)
        self.assertEqual(int(out["top_k"]), 3)
        self.assertEqual(os.path.normpath(str(out["expert_dataset_dir"])), os.path.normpath("models/expert_datasets"))


if __name__ == "__main__":
    unittest.main()
