import unittest
import tempfile
import os
import json
from unittest.mock import patch

from core.types import AgentSpec
from orchestrator.promotion_board import PromotionConfig, record_human_decision, run_promotion_board


class _Gate:
    def __init__(self, passed: bool):
        self.passed = passed
        self.suite_mode = "target"
        self.verses = ["grid_world"]
        self.bootstrap_samples = 100
        self.alpha = 0.05
        self.by_verse = {}
        self.reasons = []


class TestPromotionBoard(unittest.TestCase):
    @staticmethod
    def _stable_by_verse(passed: bool = True):
        return {
            "labyrinth_world": {"passed": bool(passed)},
            "cliff_world": {"passed": bool(passed)},
            "pursuit_world": {"passed": bool(passed)},
        }

    def test_disagreement_quarantine_policy(self):
        base = AgentSpec(spec_version="v1", policy_id="b", policy_version="0.1", algo="random")
        cand = AgentSpec(spec_version="v1", policy_id="c", policy_version="0.1", algo="random")
        cfg = PromotionConfig(
            enabled=True,
            require_multi_critic=True,
            disagreement_policy="reject",
            long_horizon_episodes_per_seed=1,
        )
        with patch("orchestrator.promotion_board.run_ab_gate", return_value=_Gate(True)):
            with patch(
                "orchestrator.promotion_board._critic_long_horizon",
                return_value={"passed": False, "by_verse": self._stable_by_verse(False)},
            ):
                out = run_promotion_board(
                    baseline_spec=base,
                    candidate_spec=cand,
                    target_verse="grid_world",
                    cfg=cfg,
                )
        self.assertFalse(out["passed"])
        self.assertTrue(out["disagreed"])

    def test_promotion_config_rejects_unknown_keys_by_default(self):
        with self.assertRaises(ValueError):
            PromotionConfig.from_dict({"enabled": True, "mystery_flag": True})

    def test_promotion_config_can_disable_strict_validation(self):
        cfg = PromotionConfig.from_dict(
            {"enabled": True, "strict_config_validation": False, "mystery_flag": True}
        )
        self.assertTrue(bool(cfg.enabled))

    def test_human_veto_overrides_critics(self):
        base = AgentSpec(spec_version="v1", policy_id="b", policy_version="0.1", algo="random")
        cand = AgentSpec(spec_version="v1", policy_id="candidate_alpha", policy_version="0.1", algo="random")
        with tempfile.TemporaryDirectory() as td:
            decision_path = os.path.join(td, "decisions.json")
            record_human_decision(
                path=decision_path,
                target_verse="grid_world",
                candidate_policy_id="candidate_alpha",
                decision="veto",
            )
            cfg = PromotionConfig.from_dict(
                {
                    "enabled": True,
                    "require_multi_critic": True,
                    "human_decision_path": decision_path,
                    "decision_ttl_hours": 24,
                    "memory_health_path": os.path.join(td, "memory_health.json"),
                }
            )
            with open(os.path.join(td, "memory_health.json"), "w", encoding="utf-8") as f:
                json.dump({"rss_slope_mb_per_hour": 3.0, "leak_detected": False}, f)
            with patch("orchestrator.promotion_board.run_ab_gate", return_value=_Gate(True)):
                with patch(
                    "orchestrator.promotion_board._critic_long_horizon",
                    return_value={"passed": True, "by_verse": self._stable_by_verse(True)},
                ):
                    out = run_promotion_board(
                        baseline_spec=base,
                        candidate_spec=cand,
                        target_verse="grid_world",
                        cfg=cfg,
                    )
            self.assertFalse(out["passed"])
            self.assertEqual(out.get("human_override"), "veto")

    def test_require_human_bless_blocks_when_missing(self):
        base = AgentSpec(spec_version="v1", policy_id="b", policy_version="0.1", algo="random")
        cand = AgentSpec(spec_version="v1", policy_id="candidate_beta", policy_version="0.1", algo="random")
        with tempfile.TemporaryDirectory() as td:
            decision_path = os.path.join(td, "decisions.json")
            cfg = PromotionConfig.from_dict(
                {
                    "enabled": True,
                    "require_multi_critic": True,
                    "human_decision_path": decision_path,
                    "require_human_bless": True,
                    "decision_ttl_hours": 24,
                    "memory_health_path": os.path.join(td, "memory_health.json"),
                }
            )
            with open(os.path.join(td, "memory_health.json"), "w", encoding="utf-8") as f:
                json.dump({"rss_slope_mb_per_hour": 1.0, "leak_detected": False}, f)
            with patch("orchestrator.promotion_board.run_ab_gate", return_value=_Gate(True)):
                with patch(
                    "orchestrator.promotion_board._critic_long_horizon",
                    return_value={"passed": True, "by_verse": self._stable_by_verse(True)},
                ):
                    out = run_promotion_board(
                        baseline_spec=base,
                        candidate_spec=cand,
                        target_verse="grid_world",
                        cfg=cfg,
                    )
            self.assertFalse(out["passed"])
            self.assertEqual(out.get("human_override"), "pending_bless")

    def test_memory_slope_gate_vetoes_promotion(self):
        base = AgentSpec(spec_version="v1", policy_id="b", policy_version="0.1", algo="random")
        cand = AgentSpec(spec_version="v1", policy_id="candidate_gamma", policy_version="0.1", algo="random")
        with tempfile.TemporaryDirectory() as td:
            mh_path = os.path.join(td, "memory_health.json")
            with open(mh_path, "w", encoding="utf-8") as f:
                json.dump({"rss_slope_mb_per_hour": 120.0, "leak_detected": False}, f)
            cfg = PromotionConfig.from_dict(
                {
                    "enabled": True,
                    "require_multi_critic": True,
                    "memory_health_path": mh_path,
                    "max_memory_slope_mb_per_hour": 24.0,
                }
            )
            with patch("orchestrator.promotion_board.run_ab_gate", return_value=_Gate(True)):
                with patch(
                    "orchestrator.promotion_board._critic_long_horizon",
                    return_value={"passed": True, "by_verse": self._stable_by_verse(True)},
                ):
                    out = run_promotion_board(
                        baseline_spec=base,
                        candidate_spec=cand,
                        target_verse="grid_world",
                        cfg=cfg,
                    )
            self.assertFalse(out["passed"])
            self.assertTrue(bool(out.get("hard_veto")))
            mem_gate = out.get("memory_health_gate", {})
            self.assertEqual(str(mem_gate.get("reason")), "memory_slope_too_high")

    def test_stability_gate_requires_minimum_verse_count(self):
        base = AgentSpec(spec_version="v1", policy_id="b", policy_version="0.1", algo="random")
        cand = AgentSpec(spec_version="v1", policy_id="candidate_delta", policy_version="0.1", algo="random")
        with tempfile.TemporaryDirectory() as td:
            mh_path = os.path.join(td, "memory_health.json")
            with open(mh_path, "w", encoding="utf-8") as f:
                json.dump({"rss_slope_mb_per_hour": 1.0, "leak_detected": False}, f)
            cfg = PromotionConfig.from_dict(
                {
                    "enabled": True,
                    "require_multi_critic": True,
                    "memory_health_path": mh_path,
                    "require_stability_verses": 3,
                }
            )
            with patch("orchestrator.promotion_board.run_ab_gate", return_value=_Gate(True)):
                with patch(
                    "orchestrator.promotion_board._critic_long_horizon",
                    return_value={
                        "passed": True,
                        "by_verse": {
                            "labyrinth_world": {"passed": True},
                            "cliff_world": {"passed": True},
                        },
                    },
                ):
                    out = run_promotion_board(
                        baseline_spec=base,
                        candidate_spec=cand,
                        target_verse="grid_world",
                        cfg=cfg,
                    )
            self.assertFalse(out["passed"])
            st = out.get("stability_gate", {})
            self.assertEqual(str(st.get("reason")), "insufficient_verse_coverage")


if __name__ == "__main__":
    unittest.main()
