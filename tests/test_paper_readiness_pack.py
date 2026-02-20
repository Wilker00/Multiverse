from __future__ import annotations

from tools.run_paper_readiness_pack import (
    _build_fixed_seed_cmd,
    _build_promote_cmd,
    _build_theory_cmd,
    _normalize_pack,
)


def test_normalize_pack_defaults() -> None:
    pack = _normalize_pack({"schema_version": "v1"})
    assert pack["schema_version"] == "v1"
    assert pack["benchmark_gate"]["enabled"] is True
    assert pack["benchmark_gate"]["suite_path"] == "benchmark_suite.yaml"
    assert pack["benchmark_gate"]["candidate_config"] == []
    assert pack["benchmark_gate"]["baseline_config"] == []
    assert pack["fixed_seed_transfer"]["seeds"] == "123,223,337"
    assert pack["theory_validation"]["transfer_lambda"] == "auto"


def test_normalize_pack_with_benchmark_configs() -> None:
    pack = _normalize_pack(
        {
            "schema_version": "v1",
            "benchmark_gate": {
                "candidate_config": ["dataset_path=models/expert_datasets/line_world.jsonl", "epsilon_start=0.1"],
                "baseline_config": ["epsilon_start=1.0"],
            },
        }
    )
    assert pack["benchmark_gate"]["candidate_config"] == [
        "dataset_path=models/expert_datasets/line_world.jsonl",
        "epsilon_start=0.1",
    ]
    assert pack["benchmark_gate"]["baseline_config"] == ["epsilon_start=1.0"]


def test_build_promote_cmd_includes_kv_configs() -> None:
    cmd = _build_promote_cmd(
        py="python",
        suite_path="benchmark_suite.yaml",
        out_json="out.json",
        history_db="hist.sqlite",
        candidate_algo="memory_recall",
        baseline_algo="q",
        candidate_config=["epsilon=0.1", "train=true"],
        baseline_config=["epsilon=0.05"],
        retention_max_drop_pct=0.05,
        manifest_path="models/default_policy_set.json",
    )
    assert "--candidate_algo" in cmd
    assert "memory_recall" in cmd
    assert "--candidate_config" in cmd
    assert "epsilon=0.1" in cmd
    assert "train=true" in cmd
    assert "--baseline_config" in cmd
    assert "epsilon=0.05" in cmd


def test_build_fixed_seed_and_theory_cmds() -> None:
    fixed_cmd = _build_fixed_seed_cmd(
        py="python",
        report_dir="models/paper/fixed_seed",
        out_json="models/paper/fixed_seed_summary.json",
        target_verse="warehouse_world",
        episodes=60,
        max_steps=100,
        seeds_csv="123,223,337",
        transfer_algo="memory_recall",
        baseline_algo="q",
        challenge_args=["--flag", "value"],
    )
    assert "--challenge_arg=--flag" in fixed_cmd
    assert "--challenge_arg=value" in fixed_cmd

    theory_cmd = _build_theory_cmd(
        py="python",
        out_json="models/paper/theory.json",
        transfer_reports_glob="models/paper/fixed_seed/transfer_seed_*.json",
        safety_events_jsonl="runs/example/events.jsonl",
        safety_episodes=200,
        safety_violations=0,
        safety_confidence=0.95,
        transfer_lambda="auto",
        transfer_tolerance=0.1,
    )
    assert "--safety_events_jsonl" in theory_cmd
    assert "runs/example/events.jsonl" in theory_cmd
