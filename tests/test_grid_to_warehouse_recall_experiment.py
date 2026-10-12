import json
import os
import tempfile
import unittest

from tools.run_grid_to_warehouse_recall_experiment import (
    _recall_agent,
    recall_verdict,
    success_episode_ids,
    summarize_recall_run,
    write_success_only_run,
)


def _event(
    *,
    episode_id: str,
    reward: float,
    reached_goal: bool = False,
    query_executed: bool = False,
    recall_used: bool = False,
    source_verse: str = "",
    score: float | None = None,
    battery_death: bool = False,
    hit_wall: bool = False,
) -> dict:
    info = {
        "reached_goal": reached_goal,
        "battery_death": battery_death,
        "hit_wall": hit_wall,
        "memory_query": {"query_requested": query_executed, "query_executed": query_executed},
        "action_info": {
            "memory_recall_used": recall_used,
            "memory_recall_eligible": recall_used,
            "memory_recall_gate_passed": recall_used,
        },
    }
    if source_verse:
        info["transfer_decision_records"] = [
            {
                "decision_reason": "memory_recall_applied" if recall_used else "gate_rejected",
                "source": {"verse_name": source_verse, "score": score},
            }
        ]
    return {
        "episode_id": episode_id,
        "reward": reward,
        "info": info,
    }


class TestGridToWarehouseRecallExperiment(unittest.TestCase):
    def test_success_filter_keeps_full_successful_episodes(self) -> None:
        events = [
            _event(episode_id="bad", reward=-0.1),
            _event(episode_id="good", reward=-0.1),
            _event(episode_id="good", reward=1.0, reached_goal=True),
        ]
        self.assertEqual(success_episode_ids(events), {"good"})
        with tempfile.TemporaryDirectory() as td:
            source = os.path.join(td, "source")
            dest = os.path.join(td, "dest")
            os.makedirs(source)
            with open(os.path.join(source, "events.jsonl"), "w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")
            kept = write_success_only_run(source_dir=source, dest_dir=dest)
            self.assertEqual(kept, 1)
            with open(os.path.join(dest, "events.jsonl"), "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual([row["episode_id"] for row in rows], ["good", "good"])

    def test_summary_counts_grid_hits_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.jsonl")
            rows = [
                _event(
                    episode_id="ep1",
                    reward=-1.0,
                    query_executed=True,
                    source_verse="grid_world",
                    score=0.4,
                    recall_used=True,
                ),
                _event(
                    episode_id="ep1",
                    reward=10.0,
                    reached_goal=True,
                    query_executed=True,
                    source_verse="grid_world",
                    score=-0.2,
                    battery_death=True,
                ),
                _event(episode_id="ep2", reward=-0.5, query_executed=True, hit_wall=True),
            ]
            with open(path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            summary = summarize_recall_run(td, early_episodes=1)
        self.assertEqual(summary["queries_executed"], 3)
        self.assertEqual(summary["memory_recall_used"], 1)
        self.assertEqual(summary["grid_top_hits"], 2)
        self.assertEqual(summary["positive_score_hits"], 1)
        self.assertEqual(summary["battery_death_episodes"], 1)
        self.assertEqual(summary["hit_wall"], 1)
        self.assertEqual(summary["early_mean_return"], 9.0)
        self.assertEqual(summary["source_verse_counts"], {"grid_world": 2})

    def test_verdict_requires_applied_grid_recall_and_a_better_outcome(self) -> None:
        grid = {
            "queries_executed": 4,
            "grid_top_hits": 4,
            "positive_score_hits": 4,
            "memory_recall_used": 3,
            "battery_death_episodes": 0,
        }
        baseline = {"battery_death_episodes": 1}
        lifted = recall_verdict(
            source_success_rate=0.8,
            ingested_events=20,
            grid=grid,
            baseline=baseline,
            grid_mean_return=1.0,
            baseline_mean_return=-2.0,
            grid_success_rate=0.25,
            baseline_success_rate=0.0,
        )
        self.assertEqual(lifted["primary"], "lift")
        self.assertTrue(lifted["pass"])

        weak_source = recall_verdict(
            source_success_rate=0.1,
            ingested_events=20,
            grid=grid,
            baseline=baseline,
            grid_mean_return=1.0,
            baseline_mean_return=-2.0,
            grid_success_rate=0.25,
            baseline_success_rate=0.0,
        )
        self.assertEqual(weak_source["primary"], "source_not_competent")
        self.assertFalse(weak_source["pass"])

        negative_scores = recall_verdict(
            source_success_rate=1.0,
            ingested_events=20,
            grid={
                "queries_executed": 4,
                "grid_top_hits": 4,
                "positive_score_hits": 0,
                "memory_recall_used": 0,
                "battery_death_episodes": 0,
            },
            baseline=baseline,
            grid_mean_return=-2.0,
            baseline_mean_return=-2.0,
            grid_success_rate=0.0,
            baseline_success_rate=0.0,
        )
        self.assertEqual(negative_scores["primary"], "nonpositive_similarity")

    def test_recall_agent_searches_outside_the_target_verse(self) -> None:
        spec = _recall_agent(seed=1, memory_root="mem", policy_id="recall_grid_memory")
        self.assertFalse(spec.config["recall_same_verse_only"])
        self.assertFalse(spec.config["train"])
        self.assertEqual(spec.config["epsilon_start"], 0.0)


if __name__ == "__main__":
    unittest.main()
