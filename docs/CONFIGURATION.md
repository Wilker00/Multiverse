# Configuration Reference

**Multiverse Configuration Guide** – Complete reference for all configuration classes, parameters, and environment variables.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Configuration](#core-configuration)
   - [RolloutConfig](#rolloutconfig)
   - [SafeExecutorConfig](#safeexecutorconfig)
   - [MCTSConfig](#mctsconfig)
   - [ParallelRolloutConfig](#parallelrolloutconfig)
3. [Memory Configuration](#memory-configuration)
   - [CentralMemoryConfig](#centralmemoryconfig)
   - [VectorMemoryConfig](#vectormemoryconfig)
   - [SelectionConfig](#selectionconfig)
   - [DecayConfig](#decayconfig)
4. [Orchestrator Configuration](#orchestrator-configuration)
   - [CurriculumConfig](#curriculumconfig)
   - [PromotionConfig](#promotionconfig)
   - [TrainerConfig](#trainerconfig)
5. [Agent Configuration](#agent-configuration)
   - [TransformerAgentConfig](#transformeragentconfig)
   - [QAgentConfig](#qagentconfig)
   - [MemoryRecallConfig](#memoryrecallconfig)
6. [Environment Variables](#environment-variables)
7. [Configuration Examples](#configuration-examples)

---

## Overview

Multiverse supports configuration through three mechanisms:

1. **Dataclass Configuration**: Direct instantiation of config objects in Python
2. **Environment Variables**: Override defaults via `MULTIVERSE_*` env vars
3. **JSON/YAML Files**: External configuration files (coming soon)

### Configuration Priority

When multiple sources provide the same setting:

```
Environment Variables > Explicit Config Objects > Defaults
```

### Best Practices

- **Development**: Use defaults, override with env vars for experimentation
- **Staging**: Use explicit config objects for reproducibility
- **Production**: Use environment variables + config files for security and flexibility

---

## Core Configuration

### RolloutConfig

Controls single-agent episode execution and memory retrieval.

**Purpose**: Configure how agents interact with verses during rollouts, including training mode, memory retrieval, and safety features.

**Location**: `core/rollout.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `schema_version` | `str` | (required) | Schema version identifier (e.g., "v1") |
| `max_steps` | `int` | (required) | Maximum steps per episode |
| `train` | `bool` | `False` | Enable online training during rollout |
| `collect_transitions` | `bool` | `False` | Collect transitions for offline learning |
| `safe_executor` | `SafeExecutor \| None` | `None` | Runtime safety wrapper (see SafeExecutorConfig) |
| `retriever` | `RetrievalClient \| None` | `None` | Memory retrieval client |
| `retrieval_interval` | `int` | `10` | Steps between memory retrievals |
| `on_demand_memory_enabled` | `bool` | `False` | Enable on-demand central memory queries |
| `on_demand_memory_root` | `str` | `"central_memory"` | Root directory for central memory |
| `on_demand_query_budget` | `int` | `8` | Max memory queries per episode |
| `on_demand_min_interval` | `int` | `2` | Min steps between queries |
| `on_demand_recall_ablation_prob` | `float` | `0.0` | Probability of disabling recall (for ablation studies) |
| `on_demand_recall_ablation_seed` | `int \| None` | `None` | Random seed for recall ablation |

#### Environment Variables

```bash
# Override retrieval settings
MULTIVERSE_ROLLOUT_RETRIEVAL_INTERVAL=10
MULTIVERSE_ROLLOUT_QUERY_BUDGET=8
MULTIVERSE_ROLLOUT_MIN_INTERVAL=2
```

#### Example

```python
from core.rollout import RolloutConfig

# Development: Disable training, enable memory
config = RolloutConfig(
    schema_version="v1",
    max_steps=1000,
    train=False,
    on_demand_memory_enabled=True,
    on_demand_query_budget=5
)

# Production: Full training with safety
config = RolloutConfig(
    schema_version="v1",
    max_steps=10000,
    train=True,
    collect_transitions=True,
    safe_executor=SafeExecutor(config=SafeExecutorConfig(enabled=True)),
    on_demand_memory_enabled=True,
    on_demand_query_budget=8
)
```

---

### SafeExecutorConfig

Runtime safety system with competence shield, fallback policies, and planning.

**Purpose**: Protect agents from dangerous actions, low-confidence decisions, and failure states through vetoing, fallback policies, checkpointing, and search-based planning (MCTS/A*).

**Location**: `core/safe_executor_support.py`

**Key Features**:
- **Competence Shield**: Veto high-risk, low-confidence actions
- **Recursive Fallback**: Route control to safer policies when confidence drops
- **Checkpoint Recovery**: Rewind to last safe state after dangerous outcomes
- **Planning Takeover**: Invoke MCTS or A* search when policy is uncertain
- **Danger Mapping**: Block known-dangerous states via learned danger clusters

#### Fields (54 total)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable SafeExecutor |
| `danger_threshold` | `float` | `0.90` | Threshold for classifying states as dangerous (0-1) |
| `min_action_confidence` | `float` | `0.08` | Minimum action confidence to avoid veto (0-1) |
| `adaptive_veto_enabled` | `bool` | `False` | Relax veto threshold as agent improves |
| `adaptive_veto_relaxation` | `float` | `0.35` | Max confidence relaxation when adaptive enabled |
| `adaptive_veto_warmup_steps` | `int` | `12` | Steps before adaptive veto activates |
| `adaptive_veto_failure_guard` | `float` | `0.20` | Never relax below this threshold |
| `adaptive_veto_schedule_enabled` | `bool` | `False` | Use scheduled veto relaxation curve |
| `adaptive_veto_relaxation_start` | `float` | `0.10` | Starting relaxation value |
| `adaptive_veto_relaxation_end` | `float` | `0.35` | Ending relaxation value |
| `adaptive_veto_schedule_steps` | `int` | `200` | Steps to complete relaxation schedule |
| `adaptive_veto_schedule_power` | `float` | `1.20` | Power curve for schedule (>1 = back-loaded) |
| `severe_reward_threshold` | `float` | `-50.0` | Reward below this triggers intervention |
| `confidence_model_path` | `str` | `""` | Path to neural confidence estimator model |
| `confidence_model_weight` | `float` | `0.60` | Blend weight for neural confidence (0-1) |
| `confidence_model_obs_dim` | `int` | `64` | Observation embedding dimension |
| `competence_window` | `int` | `5` | Rolling window for competence rate tracking |
| `min_competence_rate` | `float` | `0.90` | Min fraction of non-vetoed actions |
| `fallback_horizon_steps` | `int` | `8` | Steps to run fallback policy before returning control |
| `checkpoint_interval` | `int` | `5` | Steps between checkpoints |
| `max_rewinds_per_episode` | `int` | `8` | Max times to rewind to checkpoint |
| `block_repeated_fail_action` | `bool` | `True` | Block actions that previously failed in this state |
| `prefer_fallback_on_veto` | `bool` | `True` | Use fallback instead of random when vetoing |

**Planning (A\* Search)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `planner_enabled` | `bool` | `False` | Enable A* search planning |
| `planner_confidence_threshold` | `float` | `0.12` | Trigger planning below this confidence |
| `planner_horizon` | `int` | `5` | Planning depth (steps ahead) |
| `planner_max_expansions` | `int` | `8000` | Max nodes to expand per search |
| `planner_trigger_on_high_danger` | `bool` | `True` | Invoke planner in dangerous states |
| `planner_trigger_on_block` | `bool` | `True` | Invoke planner when action vetoed |
| `planner_verse_allowlist` | `List[str]` | `[]` | Only enable planner for these verses (empty = all) |
| `planning_regret_adaptation` | `float` | `0.30` | Adaptive planning budget based on past regret |
| `planning_budget_per_episode` | `int` | `6` | Max planning invocations per episode |
| `planning_budget_per_minute` | `int` | `120` | Max planning invocations per wall-clock minute |

**MCTS (Monte Carlo Tree Search)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mcts_enabled` | `bool` | `False` | Enable MCTS search |
| `mcts_num_simulations` | `int` | `96` | Simulations per MCTS call |
| `mcts_max_depth` | `int` | `4` | Max simulation depth |
| `mcts_c_puct` | `float` | `1.4` | PUCT exploration constant |
| `mcts_discount` | `float` | `0.99` | Discount factor γ |
| `mcts_loss_threshold` | `float` | `-0.95` | Treat action as forced loss below this value |
| `mcts_min_visits` | `int` | `8` | Min visits to consider forced loss |
| `mcts_trigger_on_low_confidence` | `bool` | `True` | Invoke MCTS when confidence is low |
| `mcts_trigger_on_high_danger` | `bool` | `True` | Invoke MCTS in dangerous states |
| `mcts_trigger_on_block` | `bool` | `True` | Invoke MCTS when action vetoed |
| `mcts_meta_model_path` | `str` | `""` | Path to meta-transformer for value estimates |
| `mcts_meta_history_len` | `int` | `6` | History length for meta-transformer |
| `mcts_value_confidence_threshold` | `float` | `0.0` | Min confidence for value estimates |
| `mcts_verse_overrides` | `Dict` | `{}` | Per-verse MCTS parameter overrides |

**Danger Detection**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `shield_enabled` | `bool` | `False` | Enable learned safety shield |
| `shield_model_path` | `str` | `""` | Path to shield classifier model |
| `shield_threshold` | `float` | `0.50` | Danger probability threshold |
| `danger_map_path` | `str` | `""` | Path to danger cluster embeddings JSON |
| `danger_map_similarity_threshold` | `float` | `0.90` | Cosine similarity to flag as dangerous |
| `failure_signature_path` | `str` | `""` | Path to known failure signatures |
| `failure_signature_similarity_threshold` | `float` | `0.92` | Similarity to match failure signature |
| `failure_signature_embedding_dim` | `int` | `64` | Embedding dimension for signatures |
| `force_mcts_on_failure_signature` | `bool` | `True` | Always invoke MCTS on signature match |

#### Environment Variables

```bash
# Safety thresholds
MULTIVERSE_SAFE_DANGER_THRESHOLD=0.90
MULTIVERSE_SAFE_MIN_CONFIDENCE=0.08

# Fallback and checkpointing
MULTIVERSE_SAFE_FALLBACK_HORIZON=8
MULTIVERSE_SAFE_CHECKPOINT_INTERVAL=5
MULTIVERSE_SAFE_MAX_REWINDS=8

# Planning
MULTIVERSE_SAFE_PLANNER_ENABLED=1
MULTIVERSE_SAFE_PLANNER_MAX_EXPANSIONS=8000
MULTIVERSE_SAFE_PLANNING_BUDGET_PER_EPISODE=6

# MCTS
MULTIVERSE_SAFE_MCTS_ENABLED=1
MULTIVERSE_SAFE_MCTS_NUM_SIMULATIONS=96
MULTIVERSE_SAFE_MCTS_MAX_DEPTH=4
```

#### Example

```python
from core.safe_executor import SafeExecutor, SafeExecutorConfig

# Development: Minimal safety for fast iteration
config = SafeExecutorConfig(
    enabled=True,
    min_action_confidence=0.05,
    planner_enabled=False,
    mcts_enabled=False
)

# Production: Full safety stack for critical applications
config = SafeExecutorConfig(
    enabled=True,
    danger_threshold=0.90,
    min_action_confidence=0.08,
    adaptive_veto_enabled=True,
    fallback_horizon_steps=10,
    checkpoint_interval=5,
    max_rewinds_per_episode=8,
    planner_enabled=True,
    planner_max_expansions=8000,
    planner_trigger_on_high_danger=True,
    mcts_enabled=True,
    mcts_num_simulations=128,
    mcts_trigger_on_low_confidence=True,
    danger_map_path="models/danger_clusters.json",
    failure_signature_path="models/failure_signatures.json"
)
```

---

### MCTSConfig

Neural-guided Monte Carlo Tree Search configuration.

**Purpose**: Configure MCTS for strategic planning in discrete action spaces with checkpoint support.

**Location**: `core/mcts_search.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_simulations` | `int` | `96` | Number of MCTS simulations per search |
| `max_depth` | `int` | `12` | Maximum simulation depth |
| `c_puct` | `float` | `1.4` | PUCT exploration constant (higher = more exploration) |
| `discount` | `float` | `0.99` | Discount factor γ for future rewards |
| `dirichlet_alpha` | `float` | `0.30` | Dirichlet noise alpha (for exploration at root) |
| `dirichlet_epsilon` | `float` | `0.25` | Fraction of Dirichlet noise to add |
| `min_prior` | `float` | `1e-6` | Minimum action prior probability |
| `reward_scale` | `float` | `10.0` | Scale factor for rewards |
| `terminal_win_value` | `float` | `1.0` | Value assigned to terminal win states |
| `terminal_loss_value` | `float` | `-1.0` | Value assigned to terminal loss states |
| `forced_loss_threshold` | `float` | `-0.95` | Action value below this = forced loss |
| `forced_loss_min_visits` | `int` | `4` | Min visits required to declare forced loss |
| `value_confidence_threshold` | `float` | `0.0` | Min confidence for value network estimates |
| `transposition_cache` | `bool` | `True` | Enable transposition table (deduplicates states) |
| `transposition_max_entries` | `int` | `20000` | Max transposition table size |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

#### Environment Variables

```bash
MULTIVERSE_MCTS_NUM_SIMULATIONS=96
MULTIVERSE_MCTS_MAX_DEPTH=12
MULTIVERSE_MCTS_C_PUCT=1.4
MULTIVERSE_MCTS_TRANSPOSITION_MAX_ENTRIES=20000
```

#### Example

```python
from core.mcts_search import MCTSConfig, MCTSSearch

# Fast MCTS for low-latency environments
config = MCTSConfig(
    num_simulations=50,
    max_depth=8,
    c_puct=1.2,
    transposition_cache=True
)

# Deep MCTS for complex strategy games (Chess, Go)
config = MCTSConfig(
    num_simulations=200,
    max_depth=16,
    c_puct=1.6,
    discount=0.99,
    transposition_max_entries=100000,
    seed=42
)
```

---

### ParallelRolloutConfig

Configuration for parallel episode execution using ProcessPoolExecutor.

**Purpose**: Scale rollouts across multiple CPU cores for faster data collection and evaluation.

**Location**: `core/parallel_rollout.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_workers` | `int` | `4` | Number of parallel worker processes (**HARDCODED**) |
| `max_worker_timeout_s` | `int` | `3600` | Max seconds before killing a worker (**HARDCODED**) |
| `batch_size` | `int` | `None` | Episodes per batch (None = all at once) |
| `show_progress` | `bool` | `True` | Display progress bar |

#### Environment Variables (Proposed)

```bash
# NOTE: These are NOT yet implemented - currently hardcoded
MULTIVERSE_PARALLEL_NUM_WORKERS=4
MULTIVERSE_PARALLEL_MAX_TIMEOUT=3600
MULTIVERSE_PARALLEL_BATCH_SIZE=10
```

#### Current Limitations

⚠️ **HARDCODED VALUES**: `num_workers=4` and `max_worker_timeout_s=3600` are currently hardcoded in the implementation. These need to be made configurable (tracked in Phase 2 roadmap).

#### Example

```python
# Current usage (hardcoded workers)
from core.parallel_rollout import run_episodes_parallel

results = run_episodes_parallel(
    verse=verse,
    agent=agent,
    config=rollout_config,
    episodes=100,
    # num_workers currently ignored - always uses 4
)

# Future usage (after Phase 2 configuration work)
results = run_episodes_parallel(
    verse=verse,
    agent=agent,
    config=rollout_config,
    episodes=100,
    num_workers=int(os.environ.get("MULTIVERSE_PARALLEL_NUM_WORKERS", 8))
)
```

---

## Memory Configuration

### CentralMemoryConfig

Central cross-run memory repository with LTM/STM tiering.

**Purpose**: Configure the shared memory bank that stores experiences across all training runs, enabling cross-task transfer and long-term skill retention.

**Location**: `memory/central_repository_support.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `root_dir` | `str` | `"central_memory"` | Root directory for memory storage |
| `memories_filename` | `str` | `"memories.jsonl"` | Unified memory file (all tiers) |
| `ltm_memories_filename` | `str` | `"ltm_memories.jsonl"` | Long-term memory (sovereign skills) |
| `stm_memories_filename` | `str` | `"stm_memories.jsonl"` | Short-term memory (ephemeral context) |
| `dedupe_index_filename` | `str` | `"dedupe_index.json"` | JSON deduplication index (legacy) |
| `dedupe_db_filename` | `str` | `"dedupe_index.sqlite"` | SQLite deduplication index |
| `tier_policy_filename` | `str` | `"tier_policy.json"` | LTM/STM tier assignment policy |
| `stm_decay_lambda` | `float` | `1e-8` | Exponential decay rate for STM |

#### Environment Variables

```bash
MULTIVERSE_MEMORY_ROOT=central_memory
MULTIVERSE_MEMORY_STM_DECAY_LAMBDA=1e-8
```

#### Memory Optimization Variables

```bash
# Thread-safety (Phase 2.1 - implemented)
MULTIVERSE_MEMORY_LOCK_TIMEOUT=30

# Delta tracking (Phase 2.5 - implemented)
MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD=1000

# Query cache (Phase 2.6 - implemented)
MULTIVERSE_MEMORY_QUERY_CACHE_SIZE=10000
MULTIVERSE_MEMORY_QUERY_CACHE_TTL_MS=60000

# ANN search (Phase 2.3 - implemented)
MULTIVERSE_SIM_USE_ANN=1
MULTIVERSE_SIM_ANN_CANDIDATE_COUNT=100
MULTIVERSE_SIM_CACHE_LIMIT=150000
```

#### Example

```python
from memory.central_repository import CentralMemoryConfig, ingest_run

# Development: Local memory with aggressive deduplication
config = CentralMemoryConfig(
    root_dir="dev_memory",
    stm_decay_lambda=1e-6  # Faster STM decay
)

# Production: Large-scale memory with long retention
config = CentralMemoryConfig(
    root_dir="/mnt/shared/central_memory",
    stm_decay_lambda=1e-9  # Slower STM decay for long-term experiments
)

# Ingest a completed run
ingest_run(
    run_dir="runs/run_abc123",
    config=config,
    selection=SelectionConfig(min_reward=0.0)
)
```

---

### VectorMemoryConfig

Similarity-based memory retrieval via vector embeddings.

**Purpose**: Build vector indices for fast similarity search over observation spaces, enabling memory-augmented agents to retrieve relevant past experiences.

**Location**: `memory/vector_memory.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_dir` | `str` | (required) | Path to run directory containing events.jsonl |
| `events_filename` | `str` | `"events.jsonl"` | Events file within run_dir |
| `obs_keys` | `List[str] \| None` | `None` | Observation keys to encode (None = all) |
| `encoder` | `str` | `"raw"` | Encoder type: "raw", "universal", "hash" |
| `encoder_model` | `str \| None` | `None` | Model path for learned encoders |

#### Encoder Types

- **`raw`**: Direct observation flattening (fast, high-dimensional)
- **`universal`**: Cross-verse embedding via universal encoder (slow, robust)
- **`hash`**: Random projection (very fast, approximate)

#### Example

```python
from memory.vector_memory import build_inmemory_index, VectorMemoryConfig

# Index a completed run
config = VectorMemoryConfig(
    run_dir="runs/run_abc123",
    encoder="universal"
)
store = build_inmemory_index(config)

# Query for similar experiences
from memory.vector_memory import query_memory
matches = query_memory(
    store=store,
    obs=current_obs,
    top_k=5,
    encoder="universal"
)
```

---

### SelectionConfig

Filter events during memory ingestion or retrieval.

**Purpose**: Control which experiences are stored/retrieved based on quality, recency, or task relevance.

**Location**: `memory/selection.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_reward` | `float \| None` | `None` | Only keep events with reward ≥ this |
| `max_reward` | `float \| None` | `None` | Only keep events with reward ≤ this |
| `require_success` | `bool` | `False` | Only keep episodes marked as successful |
| `verse_name` | `str \| None` | `None` | Filter by verse name |
| `tags` | `List[str] \| None` | `None` | Filter by tags (any match) |
| `limit` | `int \| None` | `None` | Max events to select |

#### Example

```python
from memory.selection import SelectionConfig, select_events

# Only ingest high-quality experiences
selection = SelectionConfig(
    min_reward=1.0,
    require_success=True,
    limit=10000
)

# Verse-specific memory bank
selection = SelectionConfig(
    verse_name="labyrinth_world",
    min_reward=0.0
)
```

---

### DecayConfig

Temporal decay weighting for memory retrieval.

**Purpose**: Weight recent memories higher than old ones, with configurable decay curves.

**Location**: `memory/decay_manager.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `decay_lambda` | `float` | `1e-8` | Exponential decay rate λ (higher = faster decay) |
| `decay_mode` | `str` | `"exponential"` | Decay curve: "exponential", "linear", "none" |
| `half_life_ms` | `int \| None` | `None` | Half-life in milliseconds (alternative to lambda) |

#### Example

```python
from memory.decay_manager import DecayConfig, apply_decay

# Fast decay (emphasize very recent experiences)
config = DecayConfig(
    decay_lambda=1e-6,
    decay_mode="exponential"
)

# Slow decay (retain old experiences)
config = DecayConfig(
    decay_lambda=1e-9,
    decay_mode="exponential"
)

# No decay (all experiences weighted equally)
config = DecayConfig(
    decay_mode="none"
)
```

---

## Orchestrator Configuration

### CurriculumConfig

Adaptive curriculum learning driven by plateau detection.

**Purpose**: Automatically adjust task difficulty based on agent performance to prevent plateaus and training collapse.

**Location**: `orchestrator/curriculum_controller.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable curriculum adaptation |
| `state_path` | `str` | `"models/curriculum_adjustments.json"` | State file path |
| `plateau_window` | `int` | `5` | Episodes to detect plateau |
| `step_size` | `float` | `0.05` | Difficulty adjustment step size |
| `collapse_threshold` | `float` | `0.20` | Success rate below this = collapse |
| `min_noise` | `float` | `0.0` | Minimum action noise |
| `max_noise` | `float` | `0.35` | Maximum action noise |
| `min_partial_obs` | `float` | `0.0` | Minimum partial observability |
| `max_partial_obs` | `float` | `0.75` | Maximum partial observability |
| `min_distractors` | `int` | `0` | Minimum distractor objects |
| `max_distractors` | `int` | `6` | Maximum distractor objects |

#### Adjustment Logic

- **Plateau detected**: Increase difficulty (add noise, partial obs, distractors)
- **Collapse detected** (success < 20%): Decrease difficulty (remove noise, full obs)
- **Stable**: No changes

#### Environment Variables

```bash
MULTIVERSE_CURRICULUM_ENABLED=1
MULTIVERSE_CURRICULUM_PLATEAU_WINDOW=5
MULTIVERSE_CURRICULUM_STEP_SIZE=0.05
MULTIVERSE_CURRICULUM_COLLAPSE_THRESHOLD=0.20
```

#### Example

```python
from orchestrator.curriculum_controller import CurriculumController, CurriculumConfig

# Aggressive curriculum for fast adaptation
config = CurriculumConfig(
    enabled=True,
    plateau_window=3,
    step_size=0.10,
    collapse_threshold=0.30
)

controller = CurriculumController(config)

# Update after each evaluation
adjustment = controller.update_from_signal(
    verse_name="labyrinth_world",
    success_rate=0.85,
    mean_return=42.3
)

# Apply adjustments to verse params
from orchestrator.curriculum_controller import apply_curriculum_params
new_params = apply_curriculum_params(
    verse_name="labyrinth_world",
    params={"action_noise": 0.0, "adr_jitter": 0.0},
    adjustments=controller._load()["verses"]
)
```

---

### PromotionConfig

Promotion board for agent checkpoint evaluation and selection.

**Purpose**: Automatically promote the best agent checkpoints based on multi-metric evaluation across diverse test scenarios.

**Location**: `orchestrator/promotion_board.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable promotion board |
| `min_eval_episodes` | `int` | `10` | Min episodes for promotion consideration |
| `success_threshold` | `float` | `0.75` | Min success rate for promotion |
| `mean_return_threshold` | `float` | `10.0` | Min mean return for promotion |
| `stability_window` | `int` | `5` | Episodes to assess stability |
| `promotion_path` | `str` | `"models/promoted"` | Directory for promoted checkpoints |

#### Environment Variables (Proposed)

```bash
MULTIVERSE_PROMOTION_ENABLED=1
MULTIVERSE_PROMOTION_MIN_EPISODES=10
MULTIVERSE_PROMOTION_SUCCESS_THRESHOLD=0.75
```

#### Example

```python
from orchestrator.promotion_board import PromotionBoard, PromotionConfig

config = PromotionConfig(
    enabled=True,
    min_eval_episodes=20,
    success_threshold=0.80,
    mean_return_threshold=15.0
)

board = PromotionBoard(config)

# Evaluate checkpoint
results = board.evaluate_checkpoint(
    checkpoint_path="models/dt_checkpoint_epoch_100.pt",
    test_scenarios=["labyrinth_world", "warehouse_world"]
)

if results["promoted"]:
    print(f"Checkpoint promoted to {results['promoted_path']}")
```

---

### TrainerConfig

High-level trainer orchestration configuration.

**Purpose**: Configure end-to-end training runs including verse/agent setup, rollout orchestration, and result logging.

**Location**: `orchestrator/trainer.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_root` | `str` | `"runs"` | Root directory for run outputs |
| `schema_version` | `str` | `"v1"` | Run schema version |
| `auto_register_builtin` | `bool` | `True` | Auto-register builtin verses/agents |

#### Example

```python
from orchestrator.trainer import Trainer
from core.types import VerseSpec, AgentSpec

trainer = Trainer(
    run_root="experiments/ablations",
    schema_version="v1"
)

results = trainer.run(
    verse_spec=VerseSpec(
        verse_name="labyrinth_world",
        seed=42,
        params={"grid_size": 15}
    ),
    agent_spec=AgentSpec(
        algo="dt",
        config={"context_len": 20, "lr": 1e-4}
    ),
    episodes=1000,
    max_steps=500
)
```

---

## Agent Configuration

### TransformerAgentConfig

Decision Transformer agent with GPT-style architecture.

**Purpose**: Configure offline RL agents that model sequences of (return-to-go, state, action) as autoregressive prediction.

**Location**: `agents/transformer_agent.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `context_len` | `int` | `20` | Sequence context length (in timesteps) |
| `n_layer` | `int` | `4` | Number of transformer layers |
| `n_head` | `int` | `4` | Number of attention heads |
| `n_embd` | `int` | `128` | Embedding dimension |
| `dropout` | `float` | `0.1` | Dropout rate |
| `lr` | `float` | `1e-4` | Learning rate |
| `weight_decay` | `float` | `1e-5` | Weight decay (L2 regularization) |
| `warmup_steps` | `int` | `1000` | Learning rate warmup steps |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping threshold |
| `batch_size` | `int` | `64` | Training batch size |
| `target_return` | `float` | `100.0` | Target return-to-go for inference |

#### Environment Variables (Proposed)

```bash
MULTIVERSE_DT_CONTEXT_LEN=20
MULTIVERSE_DT_N_LAYER=4
MULTIVERSE_DT_LR=1e-4
MULTIVERSE_DT_BATCH_SIZE=64
```

#### Example

```python
# Small fast model for development
config = {
    "context_len": 10,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
    "lr": 3e-4,
    "batch_size": 32
}

# Large model for production
config = {
    "context_len": 30,
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 256,
    "dropout": 0.1,
    "lr": 1e-4,
    "batch_size": 128,
    "warmup_steps": 2000
}
```

---

### QAgentConfig

Q-learning and DQN agent configuration.

**Purpose**: Configure value-based RL agents for discrete action spaces.

**Location**: `agents/q_agent.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `alpha` | `float` | `0.1` | Learning rate (tabular Q-learning) |
| `gamma` | `float` | `0.99` | Discount factor |
| `epsilon` | `float` | `0.1` | Exploration rate (ε-greedy) |
| `epsilon_decay` | `float` | `0.995` | Epsilon decay per episode |
| `epsilon_min` | `float` | `0.01` | Minimum epsilon value |
| `buffer_size` | `int` | `10000` | Replay buffer size (DQN) |
| `target_update_freq` | `int` | `100` | Target network update frequency |

#### Example

```python
# Tabular Q-learning
config = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.2,
    "epsilon_decay": 0.995
}

# Deep Q-Network (DQN)
config = {
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.9995,
    "epsilon_min": 0.05,
    "buffer_size": 100000,
    "target_update_freq": 1000
}
```

---

### MemoryRecallConfig

Memory-augmented agent with similarity-based retrieval.

**Purpose**: Configure agents that query central memory for similar past experiences to improve decision-making.

**Location**: `agents/memory_recall_agent.py`

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory_root` | `str` | `"central_memory"` | Central memory directory |
| `top_k` | `int` | `5` | Number of memories to retrieve |
| `retrieval_interval` | `int` | `1` | Steps between retrievals (1 = every step) |
| `similarity_threshold` | `float` | `0.0` | Minimum cosine similarity |
| `memory_tiers` | `List[str]` | `["ltm", "stm"]` | Tiers to query |
| `recency_weight` | `float` | `0.0` | Weight for recency vs similarity |

#### Example

```python
# Aggressive memory retrieval
config = {
    "memory_root": "central_memory",
    "top_k": 10,
    "retrieval_interval": 1,
    "similarity_threshold": 0.7,
    "recency_weight": 0.2
}

# Conservative memory retrieval (only very similar, infrequent)
config = {
    "memory_root": "central_memory",
    "top_k": 3,
    "retrieval_interval": 5,
    "similarity_threshold": 0.90,
    "recency_weight": 0.0
}
```

---

## Environment Variables

### Complete Reference

This table lists **all** environment variables supported by Multiverse (46+ variables):

#### Memory System

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIVERSE_MEMORY_ROOT` | `central_memory` | Memory root directory |
| `MULTIVERSE_MEMORY_LOCK_TIMEOUT` | `30` | Lock timeout in seconds |
| `MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD` | `1000` | Rows before delta merge |
| `MULTIVERSE_MEMORY_QUERY_CACHE_SIZE` | `10000` | LRU cache size (entries) |
| `MULTIVERSE_MEMORY_QUERY_CACHE_TTL_MS` | `60000` | Cache TTL in milliseconds |
| `MULTIVERSE_SIM_USE_ANN` | `1` | Use FAISS for ANN search (1=yes, 0=no) |
| `MULTIVERSE_SIM_ANN_CANDIDATE_COUNT` | `100` | ANN candidates before rerank |
| `MULTIVERSE_SIM_CACHE_LIMIT` | `150000` | Max similarity cache entries |
| `MULTIVERSE_SIM_ANN_DRIFT_CHECK_EVERY` | `10000` | Queries between drift checks |
| `MULTIVERSE_SIM_ANN_MAX_DRIFT` | `0.05` | Max allowed drift (5%) |

#### Rollout & Safety

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIVERSE_ROLLOUT_RETRIEVAL_INTERVAL` | `10` | Steps between memory queries |
| `MULTIVERSE_ROLLOUT_QUERY_BUDGET` | `8` | Max queries per episode |
| `MULTIVERSE_ROLLOUT_MIN_INTERVAL` | `2` | Min steps between queries |
| `MULTIVERSE_SAFE_DANGER_THRESHOLD` | `0.90` | Danger classification threshold |
| `MULTIVERSE_SAFE_MIN_CONFIDENCE` | `0.08` | Min action confidence |
| `MULTIVERSE_SAFE_FALLBACK_HORIZON` | `8` | Fallback policy steps |
| `MULTIVERSE_SAFE_CHECKPOINT_INTERVAL` | `5` | Steps between checkpoints |
| `MULTIVERSE_SAFE_MAX_REWINDS` | `8` | Max rewinds per episode |
| `MULTIVERSE_SAFE_PLANNER_ENABLED` | `0` | Enable A* planning |
| `MULTIVERSE_SAFE_PLANNER_MAX_EXPANSIONS` | `8000` | Max nodes per search |
| `MULTIVERSE_SAFE_PLANNING_BUDGET_PER_EPISODE` | `6` | Max planning calls |
| `MULTIVERSE_SAFE_MCTS_ENABLED` | `0` | Enable MCTS |
| `MULTIVERSE_SAFE_MCTS_NUM_SIMULATIONS` | `96` | MCTS simulations |
| `MULTIVERSE_SAFE_MCTS_MAX_DEPTH` | `4` | MCTS depth |

#### MCTS

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIVERSE_MCTS_NUM_SIMULATIONS` | `96` | Simulations per search |
| `MULTIVERSE_MCTS_MAX_DEPTH` | `12` | Max simulation depth |
| `MULTIVERSE_MCTS_C_PUCT` | `1.4` | Exploration constant |
| `MULTIVERSE_MCTS_TRANSPOSITION_MAX_ENTRIES` | `20000` | Transposition table size |

#### Curriculum

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIVERSE_CURRICULUM_ENABLED` | `1` | Enable curriculum learning |
| `MULTIVERSE_CURRICULUM_PLATEAU_WINDOW` | `5` | Episodes to detect plateau |
| `MULTIVERSE_CURRICULUM_STEP_SIZE` | `0.05` | Difficulty step size |
| `MULTIVERSE_CURRICULUM_COLLAPSE_THRESHOLD` | `0.20` | Collapse threshold |

#### Parallel Execution (Proposed - Not Yet Implemented)

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIVERSE_PARALLEL_NUM_WORKERS` | `4` | Worker processes (**HARDCODED**) |
| `MULTIVERSE_PARALLEL_MAX_TIMEOUT` | `3600` | Worker timeout seconds (**HARDCODED**) |
| `MULTIVERSE_PARALLEL_BATCH_SIZE` | `None` | Episodes per batch |

#### Agent Configs (Proposed - Not Yet Implemented)

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIVERSE_DT_CONTEXT_LEN` | `20` | DT context length |
| `MULTIVERSE_DT_N_LAYER` | `4` | DT layers |
| `MULTIVERSE_DT_LR` | `1e-4` | DT learning rate |
| `MULTIVERSE_DT_BATCH_SIZE` | `64` | DT batch size |

#### Docker (Required)

| Variable | Default | Description |
|----------|---------|-------------|
| `GF_SECURITY_ADMIN_PASSWORD` | (required) | Grafana admin password |

---

## Configuration Examples

### Development Environment

**Goal**: Fast iteration, minimal safety overhead, local storage

```bash
# .env.development
MULTIVERSE_MEMORY_ROOT=dev_memory
MULTIVERSE_MEMORY_QUERY_CACHE_SIZE=1000
MULTIVERSE_SIM_USE_ANN=0  # Disable FAISS for small datasets

MULTIVERSE_SAFE_DANGER_THRESHOLD=0.95  # Relaxed safety
MULTIVERSE_SAFE_MIN_CONFIDENCE=0.05
MULTIVERSE_SAFE_PLANNER_ENABLED=0
MULTIVERSE_SAFE_MCTS_ENABLED=0

MULTIVERSE_CURRICULUM_ENABLED=0  # Manual difficulty control

MULTIVERSE_PARALLEL_NUM_WORKERS=2  # Low parallelism
```

**Python Config**:

```python
from core.rollout import RolloutConfig
from core.safe_executor import SafeExecutorConfig

rollout_config = RolloutConfig(
    schema_version="v1",
    max_steps=500,
    train=True,
    on_demand_memory_enabled=False  # Disable for speed
)

safe_config = SafeExecutorConfig(
    enabled=True,
    danger_threshold=0.95,
    min_action_confidence=0.05,
    planner_enabled=False,
    mcts_enabled=False
)
```

---

### Staging Environment

**Goal**: Realistic evaluation, full safety stack, shared memory

```bash
# .env.staging
MULTIVERSE_MEMORY_ROOT=/mnt/shared/staging_memory
MULTIVERSE_MEMORY_QUERY_CACHE_SIZE=10000
MULTIVERSE_SIM_USE_ANN=1
MULTIVERSE_SIM_ANN_CANDIDATE_COUNT=100

MULTIVERSE_SAFE_DANGER_THRESHOLD=0.90
MULTIVERSE_SAFE_MIN_CONFIDENCE=0.08
MULTIVERSE_SAFE_PLANNER_ENABLED=1
MULTIVERSE_SAFE_PLANNER_MAX_EXPANSIONS=5000
MULTIVERSE_SAFE_MCTS_ENABLED=1
MULTIVERSE_SAFE_MCTS_NUM_SIMULATIONS=96

MULTIVERSE_CURRICULUM_ENABLED=1
MULTIVERSE_CURRICULUM_PLATEAU_WINDOW=5

MULTIVERSE_PARALLEL_NUM_WORKERS=8
```

**Python Config**:

```python
rollout_config = RolloutConfig(
    schema_version="v1",
    max_steps=1000,
    train=True,
    on_demand_memory_enabled=True,
    on_demand_query_budget=8
)

safe_config = SafeExecutorConfig(
    enabled=True,
    danger_threshold=0.90,
    min_action_confidence=0.08,
    planner_enabled=True,
    planner_max_expansions=5000,
    mcts_enabled=True,
    mcts_num_simulations=96
)
```

---

### Production Environment

**Goal**: Maximum robustness, performance, observability

```bash
# .env.production
MULTIVERSE_MEMORY_ROOT=/data/central_memory
MULTIVERSE_MEMORY_LOCK_TIMEOUT=60
MULTIVERSE_MEMORY_DELTA_MERGE_THRESHOLD=1000
MULTIVERSE_MEMORY_QUERY_CACHE_SIZE=50000
MULTIVERSE_MEMORY_QUERY_CACHE_TTL_MS=300000  # 5 min
MULTIVERSE_SIM_USE_ANN=1
MULTIVERSE_SIM_ANN_CANDIDATE_COUNT=200
MULTIVERSE_SIM_CACHE_LIMIT=500000

MULTIVERSE_SAFE_DANGER_THRESHOLD=0.88  # Stricter safety
MULTIVERSE_SAFE_MIN_CONFIDENCE=0.10
MULTIVERSE_SAFE_PLANNER_ENABLED=1
MULTIVERSE_SAFE_PLANNER_MAX_EXPANSIONS=10000
MULTIVERSE_SAFE_PLANNING_BUDGET_PER_EPISODE=10
MULTIVERSE_SAFE_MCTS_ENABLED=1
MULTIVERSE_SAFE_MCTS_NUM_SIMULATIONS=128
MULTIVERSE_SAFE_MCTS_MAX_DEPTH=6

MULTIVERSE_CURRICULUM_ENABLED=1
MULTIVERSE_CURRICULUM_PLATEAU_WINDOW=10
MULTIVERSE_CURRICULUM_STEP_SIZE=0.03  # Gradual adjustments

MULTIVERSE_PARALLEL_NUM_WORKERS=32
MULTIVERSE_PARALLEL_MAX_TIMEOUT=7200

# Docker observability
GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
```

**Python Config**:

```python
rollout_config = RolloutConfig(
    schema_version="v1",
    max_steps=10000,
    train=True,
    collect_transitions=True,
    on_demand_memory_enabled=True,
    on_demand_query_budget=10,
    on_demand_min_interval=2
)

safe_config = SafeExecutorConfig(
    enabled=True,
    danger_threshold=0.88,
    min_action_confidence=0.10,
    adaptive_veto_enabled=True,
    fallback_horizon_steps=10,
    checkpoint_interval=5,
    max_rewinds_per_episode=10,
    planner_enabled=True,
    planner_max_expansions=10000,
    planning_budget_per_episode=10,
    mcts_enabled=True,
    mcts_num_simulations=128,
    mcts_max_depth=6,
    danger_map_path="models/production_danger_map.json",
    failure_signature_path="models/production_failure_signatures.json"
)
```

---

## Configuration Migration Guide

### From Hardcoded to Configurable

**Current state**: Many parameters are hardcoded in the codebase (Phase 2 in progress).

**Target state**: All tuneable parameters exposed via environment variables or config objects.

#### Step 1: Identify Hardcoded Values

Run this search to find hardcoded limits:

```bash
# Find numeric literals in config-related files
grep -r "= [0-9]" core/ orchestrator/ memory/ agents/ | grep -E "(num_|max_|min_|threshold|timeout)"
```

#### Step 2: Add Environment Variable Support

**Before**:

```python
# core/parallel_rollout.py (HARDCODED)
num_workers = 4
max_timeout = 3600
```

**After**:

```python
import os

num_workers = int(os.environ.get("MULTIVERSE_PARALLEL_NUM_WORKERS", "4"))
max_timeout = int(os.environ.get("MULTIVERSE_PARALLEL_MAX_TIMEOUT", "3600"))
```

#### Step 3: Update .env.example

```bash
# Add to .env.example
MULTIVERSE_PARALLEL_NUM_WORKERS=4
MULTIVERSE_PARALLEL_MAX_TIMEOUT=3600
```

#### Step 4: Update Documentation

Add new variables to this document's [Environment Variables](#environment-variables) section.

---

## Next Steps

### Phase 2: Production Configuration (In Progress)

- [ ] **Task 1**: ✅ Catalog all config classes (COMPLETE)
- [ ] **Task 2**: 🔄 Create docs/CONFIGURATION.md (THIS FILE - IN PROGRESS)
- [ ] **Task 3**: Remove hardcoded worker limits (`num_workers`, timeouts)
- [ ] **Task 4**: Remove hardcoded size/timeout limits (buffer sizes, cache limits)
- [ ] **Task 5**: Create example config files (`configs/dev.env`, `staging.env`, `prod.env`)
- [ ] **Task 6**: Update `.env.example` with newly identified options
- [ ] **Task 7**: Test all configuration changes
- [ ] **Task 8**: Commit Phase 2 deliverables

### Roadmap

**Phase 3**: JSON/YAML Configuration Files
- Support `multiverse.yaml` for declarative configuration
- Validate configs with Pydantic schemas
- Hot-reload configuration without restart

**Phase 4**: Observability & Telemetry
- Export all configs to metrics (Prometheus)
- Configuration diffing in Grafana dashboards
- Config versioning and audit logs

---

## Questions?

- **Setup Issues**: See [SETUP.md](SETUP.md)
- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- **GitHub Issues**: https://github.com/anthropics/multiverse/issues (if applicable)

**Last Updated**: 2026-03-02 (Phase 2.2 - Configuration Documentation)
