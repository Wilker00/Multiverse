# YAML/JSON Configuration Guide

**Declarative configuration with validation and environment overrides**

---

## Overview

Multiverse supports structured configuration via YAML/JSON files with:
- **Schema validation** - Catch errors before runtime with Pydantic
- **Environment overrides** - Environment variables take precedence
- **Hierarchical configs** - Organize complex multi-environment setups
- **Version control friendly** - Track configuration changes in Git

### Configuration Priority

```
Explicit Python Code > Environment Variables > YAML/JSON > Schema Defaults
```

---

## Quick Start

### 1. Choose a Configuration Template

```bash
# Copy example to project root
cp configs/multiverse.dev.yaml multiverse.yaml       # Development
cp configs/multiverse.staging.yaml multiverse.yaml   # Staging
cp configs/multiverse.prod.yaml multiverse.yaml      # Production
```

### 2. Load Configuration in Code

```python
from core.config_loader import load_config

# Auto-discovers multiverse.yaml in current directory
config = load_config()

# Or specify explicit path
config = load_config("configs/multiverse.prod.yaml")

# Access configuration values
print(f"Workers: {config.parallel.num_workers}")
print(f"MCTS Sims: {config.mcts.num_simulations}")
```

### 3. Override with Environment Variables

```bash
# Override specific settings
export MULTIVERSE_PARALLEL_NUM_WORKERS=32
export MULTIVERSE_MCTS_NUM_SIMULATIONS=256

python script.py  # Uses env vars + YAML
```

---

## Configuration File Format

### Example: `multiverse.yaml`

```yaml
# Parallel Execution
parallel:
  num_workers: 8                    # 1-256 workers
  max_worker_timeout_s: 3600        # 60-86400 seconds
  use_ray: false                    # Use Ray backend
  reuse_process_pool: true

# Episode Rollouts
rollout:
  retrieval_interval: 10            # 1-1000 steps
  on_demand_query_budget: 8         # 0-1000 queries
  on_demand_min_interval: 2         # 1-100 steps

# Monte Carlo Tree Search
mcts:
  num_simulations: 96               # 1-10000 simulations
  max_depth: 12                     # 1-100 depth
  c_puct: 1.4                       # 0.0-10.0 exploration
  discount: 0.99                    # 0.0-1.0 gamma
  transposition_max_entries: 20000  # 100-10000000 entries

# Adaptive Curriculum
curriculum:
  enabled: true
  plateau_window: 5                 # 2-100 episodes
  step_size: 0.05                   # 0.001-1.0
  collapse_threshold: 0.20          # 0.0-1.0
  max_noise: 0.35                   # 0.0-1.0
  max_partial_obs: 0.75             # 0.0-1.0
  max_distractors: 6                # 0-100

# Memory System
memory:
  lock_timeout: 30                  # 1-300 seconds
  delta_merge_threshold: 1000       # 10-1000000 rows
  query_cache_size: 10000           # 0-1000000 entries
  query_cache_ttl_ms: 60000         # 0-3600000 ms
  use_ann: true                     # FAISS enabled
  ann_candidate_count: 100          # 10-10000
  cache_limit: 150000               # 1000-10000000

# Runtime Safety
safe_executor:
  enabled: true
  danger_threshold: 0.90            # 0.0-1.0
  min_action_confidence: 0.08       # 0.0-1.0
  fallback_horizon_steps: 8         # 1-1000
  checkpoint_interval: 5            # 1-100
  max_rewinds_per_episode: 8        # 0-100
  planner_enabled: false
  planner_max_expansions: 8000      # 100-1000000
  mcts_enabled: false
  mcts_num_simulations: 96          # 1-10000
```

---

## Environment Templates

### Development (`multiverse.dev.yaml`)

**Purpose**: Fast iteration, debugging, local experiments

**Key Settings:**
- `num_workers: 2` - Minimal parallelism
- `mcts.num_simulations: 32` - Fast search
- `curriculum.enabled: false` - Manual control
- `memory.use_ann: false` - Simple exact search

**Use Case**: Local laptop/desktop development

---

### Staging (`multiverse.staging.yaml`)

**Purpose**: Pre-production testing, integration validation

**Key Settings:**
- `num_workers: 8` - Moderate parallelism
- `mcts.num_simulations: 96` - Standard search
- `curriculum.enabled: true` - Full features
- `memory.use_ann: true` - FAISS enabled

**Use Case**: CI/CD pipeline, staging server

---

### Production (`multiverse.prod.yaml`)

**Purpose**: Maximum robustness and performance

**Key Settings:**
- `num_workers: 32` - High throughput
- `mcts.num_simulations: 128` - Deep search
- `safe_executor.danger_threshold: 0.88` - Stricter safety
- `memory.query_cache_size: 50000` - Large cache

**Use Case**: Production deployments, critical applications

---

## Advanced Usage

### Partial Configuration

You only need to specify values you want to override:

```yaml
# Minimal config - only override MCTS settings
mcts:
  num_simulations: 200
  max_depth: 20

# All other sections use defaults
```

### JSON Format

Same structure, different syntax:

```json
{
  "parallel": {
    "num_workers": 16,
    "max_worker_timeout_s": 7200
  },
  "mcts": {
    "num_simulations": 128
  }
}
```

### Environment Variable Overrides

All fields support environment variable overrides:

```bash
# Format: MULTIVERSE_<SECTION>_<FIELD>
export MULTIVERSE_PARALLEL_NUM_WORKERS=32
export MULTIVERSE_MCTS_NUM_SIMULATIONS=256
export MULTIVERSE_CURRICULUM_ENABLED=0  # 0=false, 1=true
export MULTIVERSE_SAFE_DANGER_THRESHOLD=0.85
```

### Programmatic Generation

```python
from core.config_schema import MultiverseConfig
from core.config_loader import save_config

# Create config programmatically
config = MultiverseConfig()
config.parallel.num_workers = 16
config.mcts.num_simulations = 200

# Save to YAML
save_config(config, "my_config.yaml")
```

### Validation

```python
from core.config_loader import load_config
from pydantic import ValidationError

try:
    config = load_config("bad_config.yaml")
except ValidationError as e:
    print("Configuration errors:")
    for error in e.errors():
        field = ".".join(str(loc) for loc in error['loc'])
        msg = error['msg']
        print(f"  {field}: {msg}")
```

---

## Configuration Reference

### Parallel Rollout (`parallel`)

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `num_workers` | int | 1-256 | 4 | Parallel worker processes |
| `max_worker_timeout_s` | int | 60-86400 | 3600 | Worker timeout (seconds) |
| `use_ray` | bool | - | false | Use Ray for distribution |
| `reuse_process_pool` | bool | - | true | Reuse process pool |
| `min_parallel_episodes` | int | ≥1 | 64 | Min episodes for parallel |

### Rollout (`rollout`)

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `retrieval_interval` | int | 1-1000 | 10 | Steps between memory queries |
| `on_demand_query_budget` | int | 0-1000 | 8 | Max queries per episode |
| `on_demand_min_interval` | int | 1-100 | 2 | Min steps between queries |

### MCTS (`mcts`)

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `num_simulations` | int | 1-10000 | 96 | Simulations per search |
| `max_depth` | int | 1-100 | 12 | Max simulation depth |
| `c_puct` | float | 0.0-10.0 | 1.4 | PUCT exploration constant |
| `discount` | float | 0.0-1.0 | 0.99 | Discount factor γ |
| `transposition_max_entries` | int | 100-10M | 20000 | Transposition table size |

### Curriculum (`curriculum`)

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `enabled` | bool | - | true | Enable curriculum |
| `plateau_window` | int | 2-100 | 5 | Episodes to detect plateau |
| `step_size` | float | 0.001-1.0 | 0.05 | Difficulty step size |
| `collapse_threshold` | float | 0.0-1.0 | 0.20 | Collapse success rate |
| `max_noise` | float | 0.0-1.0 | 0.35 | Max action noise |
| `max_partial_obs` | float | 0.0-1.0 | 0.75 | Max partial observability |
| `max_distractors` | int | 0-100 | 6 | Max distractor objects |

### Memory (`memory`)

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `lock_timeout` | int | 1-300 | 30 | Lock timeout (seconds) |
| `delta_merge_threshold` | int | 10-1M | 1000 | Rows before merge |
| `query_cache_size` | int | 0-1M | 10000 | Query cache entries |
| `query_cache_ttl_ms` | int | 0-3600000 | 60000 | Cache TTL (ms) |
| `use_ann` | bool | - | true | Use FAISS ANN |
| `ann_candidate_count` | int | 10-10000 | 100 | ANN candidates |
| `cache_limit` | int | 1K-10M | 150000 | Max cache entries |

### Safe Executor (`safe_executor`)

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `enabled` | bool | - | true | Enable safety |
| `danger_threshold` | float | 0.0-1.0 | 0.90 | Danger threshold |
| `min_action_confidence` | float | 0.0-1.0 | 0.08 | Min confidence |
| `fallback_horizon_steps` | int | 1-1000 | 8 | Fallback horizon |
| `checkpoint_interval` | int | 1-100 | 5 | Checkpoint interval |
| `max_rewinds_per_episode` | int | 0-100 | 8 | Max rewinds |
| `planner_enabled` | bool | - | false | Enable A* planner |
| `planner_max_expansions` | int | 100-1M | 8000 | Max A* expansions |
| `mcts_enabled` | bool | - | false | Enable MCTS |
| `mcts_num_simulations` | int | 1-10000 | 96 | MCTS simulations |

---

## Migration Guide

### From .env to YAML

**Before (`.env`):**
```bash
MULTIVERSE_PARALLEL_NUM_WORKERS=8
MULTIVERSE_MCTS_NUM_SIMULATIONS=128
MULTIVERSE_CURRICULUM_ENABLED=1
```

**After (`multiverse.yaml`):**
```yaml
parallel:
  num_workers: 8

mcts:
  num_simulations: 128

curriculum:
  enabled: true
```

**Benefits:**
- Structured validation
- Better version control
- Easier to review in PRs
- Type checking and IDE autocompletion

---

## Best Practices

### 1. Use Version Control

```bash
# Track configs in Git
git add multiverse.yaml
git commit -m "config: tune MCTS for chess domain"
```

### 2. Separate Secrets

```yaml
# Good: Reference env vars for secrets
# run_root is public config
parallel:
  run_root: "/data/runs"

# Bad: Never commit secrets
# database_password: "hunter2"  # NEVER DO THIS!
```

Use `.env.local` (gitignored) or secrets management:
```bash
# .env.local (gitignored)
DATABASE_PASSWORD=actual_secret_here
GRAFANA_PASSWORD=another_secret
```

### 3. Environment-Specific Configs

```bash
# Project structure
configs/
  base.yaml           # Shared base settings
  dev.yaml            # Development overrides
  staging.yaml        # Staging overrides
  prod.yaml           # Production overrides

# Load based on environment
MULTIVERSE_ENV=prod python script.py
```

### 4. Validate Before Deploy

```bash
# Test config locally
python -c "from core.config_loader import load_config; c = load_config('multiverse.prod.yaml'); print('Valid!')"

# Run config tests
python tests/test_yaml_config.py
```

### 5. Document Changes

```yaml
# Good: Comment why you changed values
mcts:
  num_simulations: 200  # Increased for chess domain (was 96)

# Good: Document environment-specific tuning
curriculum:
  enabled: false  # Disabled in dev for manual difficulty control
```

---

## Troubleshooting

### Config File Not Found

```
FileNotFoundError: Config file not found
```

**Solution**: Provide explicit path or place `multiverse.yaml` in:
- Current working directory
- `config/multiverse.yaml`
- `.multiverse.yaml`

### Validation Errors

```
pydantic.ValidationError: 1 validation error
  parallel.num_workers
    Input should be less than or equal to 256
```

**Solution**: Check value ranges in [Configuration Reference](#configuration-reference)

### Environment Variables Not Working

```bash
# Won't work - must use MULTIVERSE_ prefix
export NUM_WORKERS=32

# Correct - uses prefix and section
export MULTIVERSE_PARALLEL_NUM_WORKERS=32
```

### YAML Syntax Errors

```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solution**: Check YAML syntax (proper indentation, no tabs)

```yaml
# Bad - tabs or wrong indentation
parallel:
num_workers: 8

# Good - consistent 2-space indentation
parallel:
  num_workers: 8
```

---

## See Also

- [Configuration Reference](CONFIGURATION.md) - Complete parameter documentation
- [Setup Guide](SETUP.md) - Installation instructions
- [Environment Variables](.env.example) - All available env vars
- [Config Examples](../configs/) - Dev/Staging/Prod templates

---

**Last Updated**: 2026-03-02 (Phase 3 - YAML/JSON Configuration)
