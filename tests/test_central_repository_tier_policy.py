import json
import os
import tempfile
import unittest

from memory.central_repository import CentralMemoryConfig, backfill_memory_metadata


class TestCentralRepositoryTierPolicy(unittest.TestCase):
    def test_per_verse_promotion_threshold_override(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_a",
                            "episode_id": "ep1",
                            "step_idx": 0,
                            "verse_name": "warehouse_world",
                            "reward": 0.6,
                            "done": False,
                            "info": {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_b",
                            "episode_id": "ep2",
                            "step_idx": 0,
                            "verse_name": "grid_world",
                            "reward": 0.6,
                            "done": False,
                            "info": {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            with open(os.path.join(td, "tier_policy.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "default": {
                            "ltm_reward_threshold": 2.0,
                            "promotion_score_threshold": 5.0,
                            "ltm_done_reward_threshold": 0.5,
                        },
                        "by_verse": {
                            "warehouse_world": {
                                "ltm_reward_threshold": 0.5,
                                "promotion_score_threshold": 5.0,
                            }
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            cfg = CentralMemoryConfig(root_dir=td, tier_policy_filename="tier_policy.json")
            stats = backfill_memory_metadata(cfg=cfg, rebuild_tier_files=True)
            self.assertEqual(int(stats.rows_written), 2)
            self.assertEqual(int(stats.ltm_rows), 1)
            self.assertEqual(int(stats.stm_rows), 1)

            rows = []
            with open(mem_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        rows.append(json.loads(s))
            self.assertEqual(len(rows), 2)
            by_verse = {str(r.get("verse_name")): str(r.get("memory_tier")) for r in rows}
            self.assertEqual(by_verse.get("warehouse_world"), "ltm")
            self.assertEqual(by_verse.get("grid_world"), "stm")

    def test_recompute_tier_reclassifies_existing_rows(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_a",
                            "episode_id": "ep1",
                            "step_idx": 0,
                            "verse_name": "warehouse_world",
                            "reward": 0.6,
                            "done": False,
                            "memory_tier": "stm",
                            "memory_family": "procedural",
                            "memory_type": "spatial_procedural",
                            "info": {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            with open(os.path.join(td, "tier_policy.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "default": {"ltm_reward_threshold": 2.0, "promotion_score_threshold": 5.0},
                        "by_verse": {"warehouse_world": {"ltm_reward_threshold": 0.5, "promotion_score_threshold": 5.0}},
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            cfg = CentralMemoryConfig(root_dir=td, tier_policy_filename="tier_policy.json")
            s1 = backfill_memory_metadata(cfg=cfg, rebuild_tier_files=True, recompute_tier=False)
            self.assertEqual(int(s1.recomputed_memory_tier), 0)

            s2 = backfill_memory_metadata(cfg=cfg, rebuild_tier_files=True, recompute_tier=True)
            self.assertEqual(int(s2.recomputed_memory_tier), 1)

            with open(mem_path, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertEqual(str(row.get("memory_tier")), "ltm")

    def test_done_min_t_and_late_stage_bonus_controls(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            rows = [
                {
                    "run_id": "run_a",
                    "episode_id": "ep1",
                    "step_idx": 10,
                    "verse_name": "line_world",
                    "reward": 1.0,
                    "done": True,
                    "memory_tier": "stm",
                    "memory_family": "procedural",
                    "memory_type": "spatial_procedural",
                    "info": {"t": 10},
                },
                {
                    "run_id": "run_a",
                    "episode_id": "ep2",
                    "step_idx": 30,
                    "verse_name": "line_world",
                    "reward": 1.0,
                    "done": True,
                    "memory_tier": "stm",
                    "memory_family": "procedural",
                    "memory_type": "spatial_procedural",
                    "info": {"t": 30},
                },
                {
                    "run_id": "run_b",
                    "episode_id": "ep3",
                    "step_idx": 96,
                    "verse_name": "warehouse_world",
                    "reward": -0.1,
                    "done": False,
                    "memory_tier": "stm",
                    "memory_family": "procedural",
                    "memory_type": "spatial_procedural",
                    "info": {"t": 96},
                },
            ]
            with open(mem_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            with open(os.path.join(td, "tier_policy.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "default": {
                            "ltm_reward_threshold": 5.0,
                            "ltm_done_reward_threshold": 5.0,
                            "promotion_score_threshold": 5.0,
                            "done_bonus": 0.0,
                            "reward_scale": 10.0,
                            "late_stage_min_t": 10_000_000,
                            "late_stage_reward_floor": 0.0,
                            "late_stage_bonus": 0.0,
                        },
                        "by_verse": {
                            "line_world": {
                                "done_min_t": 26,
                                "promotion_score_threshold": 1.2,
                                "done_bonus": 0.7,
                                "reward_scale": 1.25,
                            },
                            "warehouse_world": {
                                "promotion_score_threshold": 0.8,
                                "late_stage_min_t": 95,
                                "late_stage_reward_floor": -0.1,
                                "late_stage_bonus": 0.9,
                            },
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            cfg = CentralMemoryConfig(root_dir=td, tier_policy_filename="tier_policy.json")
            stats = backfill_memory_metadata(cfg=cfg, rebuild_tier_files=True, recompute_tier=True)
            self.assertEqual(int(stats.recomputed_memory_tier), 2)
            self.assertEqual(int(stats.ltm_rows), 2)

            by_ep = {}
            with open(mem_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    r = json.loads(s)
                    by_ep[str(r.get("episode_id"))] = str(r.get("memory_tier"))
            self.assertEqual(by_ep.get("ep1"), "stm")
            self.assertEqual(by_ep.get("ep2"), "ltm")
            self.assertEqual(by_ep.get("ep3"), "ltm")

    def test_support_guard_caps_low_support_verse_ltm_ratio(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = os.path.join(td, "memories.jsonl")
            rows = []
            for i in range(20):
                rows.append(
                    {
                        "run_id": "run_p",
                        "episode_id": f"ep{i}",
                        "step_idx": i,
                        "verse_name": "pursuit_world",
                        "reward": 1.0,
                        "done": True,
                        "memory_tier": "stm",
                        "memory_family": "procedural",
                        "memory_type": "spatial_procedural",
                        "info": {"t": 50, "reached_goal": True},
                    }
                )
            with open(mem_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            with open(os.path.join(td, "tier_policy.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "default": {
                            "ltm_reward_threshold": 0.0,
                            "ltm_done_reward_threshold": -1.0,
                            "promotion_score_threshold": 0.0,
                            "support_guard_enabled": True,
                            "support_guard_min_rows": 50,
                            "support_guard_max_ltm_ratio": 0.05,
                            "support_guard_min_ltm": 1,
                        }
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            cfg = CentralMemoryConfig(root_dir=td, tier_policy_filename="tier_policy.json")
            stats = backfill_memory_metadata(
                cfg=cfg,
                rebuild_tier_files=True,
                recompute_tier=True,
                apply_support_guards=True,
            )
            self.assertEqual(int(stats.ltm_rows), 1)
            self.assertEqual(int(stats.stm_rows), 19)
            self.assertEqual(int(stats.support_guard_demotions), 19)


if __name__ == "__main__":
    unittest.main()
