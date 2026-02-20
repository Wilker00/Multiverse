import os
import tempfile
import unittest

from core.skill_contracts import ContractConfig, SkillContractManager


class TestSkillContracts(unittest.TestCase):
    def test_strict_improvement_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "contracts.json")
            mgr = SkillContractManager(ContractConfig(path=path, strict_improvement_delta=0.05))

            m1 = {"success_rate": 0.8, "mean_return": 1.0, "safety_violation_rate": 0.1}
            r1 = mgr.register_or_update(verse_name="grid_world", skill_tag="special_moe", metrics=m1)
            self.assertTrue(r1["updated"])

            m2 = {"success_rate": 0.81, "mean_return": 1.01, "safety_violation_rate": 0.1}
            r2 = mgr.register_or_update(verse_name="grid_world", skill_tag="special_moe", metrics=m2)
            self.assertFalse(r2["updated"])
            self.assertEqual(r2["reason"], "strict_improvement_not_met")

    def test_contract_config_rejects_unknown_keys_by_default(self):
        with self.assertRaises(ValueError):
            ContractConfig.from_dict({"enabled": True, "mystery_flag": 1})

    def test_contract_config_rejects_unknown_robustness_keys(self):
        with self.assertRaises(ValueError):
            ContractConfig.from_dict(
                {
                    "robustness_thresholds": {
                        "min_success_rate": 0.5,
                        "unknown_threshold": 0.1,
                    }
                }
            )

    def test_contract_config_allows_unknown_when_strict_disabled(self):
        cfg = ContractConfig.from_dict(
            {
                "enabled": True,
                "strict_config_validation": False,
                "mystery_flag": 1,
                "robustness_thresholds": {
                    "min_success_rate": 0.6,
                    "unknown_threshold": 0.9,
                },
            }
        )
        self.assertTrue(bool(cfg.enabled))
        self.assertAlmostEqual(float(cfg.robustness_thresholds["min_success_rate"]), 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
