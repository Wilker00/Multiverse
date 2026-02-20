# Documentation

This folder contains the core documentation for the Multiverse project.

## Key Documents

- **[Project Introduction](PROJECT_INTRO.md)**: High-level overview of the project's goals, architecture, and organization. Start here to understand *what* Multiverse is.
- **[Full Paper](PAPER.md)**: The comprehensive, technical paper describing the framework's methodology, safety systems, memory architecture, and evaluation results. This is the **primary technical reference**.

## Other Resources

- **[../BENCHMARKS.md](../BENCHMARKS.md)**: Details on running benchmarks and interpreting their results.
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)**: Guidelines for contributing to the codebase.
- **[../SECURITY.md](../SECURITY.md)**: Security policy and reporting instructions.

## Working Scale Profile

Use the locked working transfer profile for multiseed scaling:

```bash
python tools/run_fixed_seed_benchmark.py \
  --runs_root runs_success_scale_working \
  --report_dir models/validation/success_scale_working \
  --out_json models/validation/success_scale_working/summary.json \
  --episodes 120 \
  --max_steps 100 \
  --seeds 101,123,149,223,337,401 \
  --challenge_arg=--bridge_behavioral_enabled \
  --challenge_arg=--transfer_warmstart_reward_scale --challenge_arg=0.01 \
  --challenge_arg=--no-transfer_warmstart_use_transfer_score \
  --challenge_arg=--transfer_q_warehouse_obs_key_mode --challenge_arg=direction_only \
  --challenge_arg=--baseline_q_warehouse_obs_key_mode --challenge_arg=direction_only \
  --challenge_arg=--transfer_epsilon_start --challenge_arg=0.05 \
  --challenge_arg=--transfer_epsilon_min --challenge_arg=0.01 \
  --challenge_arg=--transfer_epsilon_decay --challenge_arg=0.999 \
  --challenge_arg=--baseline_epsilon_start --challenge_arg=0.05 \
  --challenge_arg=--baseline_epsilon_min --challenge_arg=0.01 \
  --challenge_arg=--baseline_epsilon_decay --challenge_arg=0.999 \
  --challenge_arg=--transfer_learn_hazard_penalty --challenge_arg=0.0 \
  --challenge_arg=--transfer_learn_success_bonus --challenge_arg=0.0 \
  --challenge_arg=--no-dynamic_transfer_mix \
  --challenge_arg=--no-safe_transfer \
  --challenge_arg=--no-safe_baseline \
  --challenge_arg=--no-enable_mcts
```

Current canonical benchmark artifacts:

- `models/validation/success_scale_working/summary.json`
- `models/validation/fixed_seed_working_v1_nosrc/summary.json`
- `experiment/transfer_working_profile_v1.json` (locked control profile definition)
- `experiment/fixed_seed_working_profile_v1.json` (fixed-seed recovery profile)
- `models/validation/targeted_simple_recovery/working_only_summary.json` (simple-verse non-regression keep set)
- `models/validation/targeted_simple_recovery/scale_working_summary.json` (scaled confirmation on simple verses)
- `experiment/simple_verses_working_profile_v1.json` (simple-verse working profile)

Supplementary smoke artifact:

- `models/validation/working_base_smoke.json`
