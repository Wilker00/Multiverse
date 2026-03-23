# Verse Integration: Quick Reference & Development Guide

## Part 1: Quick Reference Tables

### Available Verses Taxonomy

| Category | Verses | Tags | Key Params |
|----------|--------|------|-----------|
| **Navigation** | line_world, grid_world, maze_world | navigation, 1d/2d | width, height, goal_pos, obstacles |
| **Complex Navigation** | labyrinth_world, swamp_world, escape_world | complex_navigation, partial_observable | width, height, hazards, visibility |
| **Resource Collection** | harvest_world, warehouse_world | resource_management, battery | vision_radius, battery_capacity |
| **Strategic Games** | chess_world, go_world, uno_world (±v2) | strategy_games, board_control, turn_based | convert_bonus, random_swing |
| **Physics/Risk** | cliff_world, pursuit_world, bridge_world | risk_sensitive, dynamic | cliff_penalty, wind_probability |
| **Domain-Specific** | factory_world, trade_world, park_world | planning/economics/interaction | breakdown_prob, transaction_cost |
| **Memory Diagnostics** | memory_vault_world, rule_flip_world | memory_diagnostics | wall_density, rule_flip_step |
| **Tutorial** | risk_tutorial_world | tutorial, risk_sensitive | risk_floor_start, target_control |

### Core Data Structures at a Glance

```python
VerseSpec
├─ spec_version: "v1"
├─ verse_name: str (registry key)
├─ verse_version: str
├─ seed: int (for RNG determinism)
├─ tags: List[str] (cognitive tags from taxonomy)
└─ params: Dict[str, JSONValue] (environment-specific)

VerseRef
├─ verse_id: str (unique runtime handle)
├─ verse_name: str
├─ verse_version: str
└─ spec_hash: str (SHA1 of VerseSpec for reproducibility)

StepEvent (JSONL log unit)
├─ run_id, episode_id, step_idx (tracking)
├─ agent_id, policy_id, policy_version (agent tracking)
├─ verse_id, verse_name, spec_hash (verse tracking)
├─ obs, action, reward, done (interaction)
└─ info: {memory_query, action_info, runtime_errors, ...}

RolloutResult (per-episode summary)
├─ events: List[StepEvent]
├─ episode_id: str
├─ steps: int
├─ return_sum: float
└─ train_metrics: Dict[str, float]
```

### Environment Step Model

| Phase | Input | Process | Output |
|-------|-------|---------|--------|
| Reset | VerseSpec, seed | verse.reset() | ResetResult(obs, info) |
| Step | action | verse.step(action) | StepResult(obs, reward, done, truncated, info) |
| Checkpoint | — | verse.export_state() | Dict[K, V] |
| Restore | state dict | verse.import_state(state) | — |

---

## Part 2: Integration Checklist

### For Adding a New Verse

#### 1. Implementation (verses/new_world.py)

- [ ] Create `NewWorldVerse` class implementing Verse protocol
- [ ] Define `NewWorldParams` dataclass with all configurable parameters
- [ ] Implement core methods:
  - [ ] `__init__(spec: VerseSpec)` — Extract params from spec
  - [ ] `seed(seed: int)` — Initialize RNG
  - [ ] `reset() → ResetResult` — Initialize episode state
  - [ ] `step(action) → StepResult` — Execute action, return reward
  - [ ] `close()` — Cleanup resources
- [ ] Optional (recommended):
  - [ ] `export_state() → Dict` — For SafeExecutor checkpointing
  - [ ] `import_state(state)` — For SafeExecutor rollback
  - [ ] `render(mode="rgb_array")` — For visualization
- [ ] Define observation_space & action_space as SpaceSpec
- [ ] Ensure all obs/action/reward are JSON-serializable

#### 2. Factory Registration (verses/registry.py)

- [ ] Create `NewWorldFactory` class implementing VerseFactory
- [ ] Implement `factory.create(spec: VerseSpec) → Verse`
- [ ] Implement `factory.tags: List[str]` property
- [ ] Register in `register_builtin()`:
  ```python
  register_verse("new_world", NewWorldFactory())
  ```
- [ ] Add verse-specific ADR params in `_default_adr_keys()`
  ```python
  if v == "new_world":
      return ["param1", "param2", "param3"]
  ```

#### 3. Taxonomy Integration (core/taxonomy.py)

- [ ] Add cognitive tags to `VERSE_TAGS["new_world"]`
  ```python
  "new_world": [
      "category",
      "difficulty_level",
      "specific_challenge",
      "required_skills",
  ]
  ```
- [ ] Update `TAXONOMY` categories if creating new domain
  ```python
  "new_category": ["new_world", ...],
  ```

#### 4. Testing

- [ ] Unit test:
  ```python
  spec = VerseSpec(verse_name="new_world", params={...})
  verse = create_verse(spec)
  result = verse.reset()
  assert result.obs is not None
  result = verse.step(0)
  assert result.reward is float
  ```
- [ ] Smoke test via trainer:
  ```bash
  python orchestrator/trainer.py --verse new_world --episodes 3 --max_steps 50
  ```
- [ ] Memory compatibility check
  ```python
  events = result[0].events
  assert all(e.obs is JSONValue for e in events)
  ```

---

## Part 3: Configuration Reference

### VerseSpec Configuration Pattern

```python
from core.types import VerseSpec

# Minimal config
spec = VerseSpec(
    spec_version="v1",
    verse_name="grid_world",
    verse_version="0.1",
)

# Full config with ADR and curriculum
spec = VerseSpec(
    spec_version="v1",
    verse_name="grid_world",
    verse_version="0.1",
    seed=42,
    tags=["navigation", "2d"],
    params={
        "width": 5,
        "height": 5,
        "step_penalty": -0.01,
        "obstacle_count": 4,
        # ADR settings
        "adr_enabled": True,  # Default: True
        "adr_jitter": 0.10,   # Default: 10%
        "adr_keys": ["width", "step_penalty"],  # Defaults to verse-specific list
    },
)
```

### Agent-Verse Integration Config

```python
from core.types import AgentSpec

# Agent requesting memory
agent_spec = AgentSpec(
    spec_version="v1",
    policy_id="memory_recall:v1",
    policy_version="1.0",
    algo="memory_recall",
    config={
        "on_demand_memory_enabled": True,
        "on_demand_memory_root": "central_memory",
        "on_demand_query_budget": 8,      # Max queries per episode
        "on_demand_min_interval": 2,      # Minimum steps between queries
        "on_demand_recall_ablation_prob": 0.1,  # A/B test: disable memory 10%
        "rar_enabled": True,              # Retrieval-augmented rollouts
        "retrieval_interval": 10,         # Query successful episodes every N steps
    },
)

# Agent with safety
agent_spec = AgentSpec(
    ...,
    config={
        ...,
        "safe_executor": {
            "enabled": True,
            "danger_threshold": 0.95,
            "min_action_confidence": 0.05,
            "competence_window": 20,
            "planner_enabled": False,
            "mcts_enabled": False,
            # Fallback policy
            "fallback_algo": "gateway",
            "fallback_config": {...},
        },
    },
)
```

### Curriculum Configuration (models/curriculum_adjustments.json)

```json
{
  "version": "v1",
  "verses": {
    "grid_world": {
      "history": [
        {"t_ms": 1234567890, "success_rate": 0.45, "mean_return": 2.3},
        {"t_ms": 1234567900, "success_rate": 0.70, "mean_return": 4.5}
      ],
      "noise": 0.05,
      "stochasticity": 0.10,
      "partial_observability": 0.0,
      "distractors": 1,
      "mode": "plateau_harder"
    }
  }
}
```

---

## Part 4: Common Patterns & Examples

### Pattern 1: Simple Verse Run

```python
from orchestrator.trainer import Trainer
from core.types import VerseSpec, AgentSpec

trainer = Trainer()

v_spec = VerseSpec(
    spec_version="v1",
    verse_name="grid_world",
    verse_version="0.1",
    seed=42,
    params={"width": 5, "height": 5, "adr_enabled": False},
)

a_spec = AgentSpec(
    spec_version="v1",
    policy_id="ppo:v1",
    policy_version="1.0",
    algo="ppo",
)

result = trainer.run(
    verse_spec=v_spec,
    agent_spec=a_spec,
    episodes=10,
    max_steps=50,
    seed=42,
)

print(f"Run ID: {result['run_id']}")
print(f"Total return: {result['total_return']:.2f}")
```

### Pattern 2: Memory-Augmented Training

```python
a_spec = AgentSpec(
    spec_version="v1",
    policy_id="dqn_recall:v1",
    policy_version="1.0",
    algo="memory_recall",  # ← Key: enables memory queries
    config={
        "on_demand_memory_enabled": True,
        "on_demand_memory_root": "central_memory",
        "on_demand_query_budget": 8,
        "rar_enabled": True,
    },
)

# After run 1:
# - Episodes stored in central_memory/stm_memories.jsonl
# - Embeddings cached in central_memory/memories.jsonl.simcache.json

# Run 2 on same verse:
# - agent.memory_query_request(obs) queried during rollout
# - find_similar() returns matches from run 1
# - Agent can learn from previous successful patterns
```

### Pattern 3: Cross-Verse Transfer

```python
# Run 1: Train on chess_world
v_spec_chess = VerseSpec(verse_name="chess_world", ...)
trainer.run(verse_spec=v_spec_chess, ...)
# → Episodes stored with cognitive_tags + strategic_signature

# Run 2: Train on go_world
v_spec_go = VerseSpec(verse_name="go_world", ...)
a_spec = AgentSpec(algo="memory_recall", ...)  # Same agent can query

trainer.run(verse_spec=v_spec_go, agent_spec=a_spec, ...)
# → Memory queries find chess_world episodes via strategic signature match
# → Agent learns from GO patterns similar to chess patterns discovered earlier
```

### Pattern 4: Safety-Augmented Training

```python
a_spec = AgentSpec(
    algo="dqn",
    config={
        "safe_executor": {
            "enabled": True,
            "danger_threshold": 0.60,  # Strict for cliff_world
            "min_action_confidence": 0.10,
            "planner_enabled": True,
            "mcts_enabled": True,
            "mcts_num_simulations": 50,
        },
    },
)

result = trainer.run(
    verse_spec=VerseSpec(verse_name="cliff_world", ...),
    agent_spec=a_spec,
    episodes=20,
    max_steps=100,
)

# During training:
# - SafeExecutor vetos low-confidence cliff actions
# - MCTS planning kicks in for high-risk states
# - Failed steps trigger checkpoint recovery
# - Risk metrics logged to StepEvent.info["safe_executor"]
```

### Pattern 5: Curriculum Learning

```python
# First run: Base difficulty
result1 = trainer.run(
    verse_spec=VerseSpec(
        verse_name="grid_world",
        params={"adr_enabled": False},
    ),
    agent_spec=...,
    episodes=15,
)
# → Compute success_rate, mean_return
# → If stable high: curriculum adjusts difficulty up

# Curriculum system automatically updates models/curriculum_adjustments.json
# next run uses updated params:
#   - noise += 0.05
#   - stochasticity += 0.05
#   - distractors += 1

result2 = trainer.run(
    verse_spec=VerseSpec(
        verse_name="grid_world",
        params={"adr_enabled": True},  # Now picks up curriculum adjustments
    ),
    agent_spec=...,
    episodes=15,
)
# → Harder version of grid_world
```

---

## Part 5: Debugging & Troubleshooting

### Check Verse Registration

```python
from verses.registry import list_verses

verses = list_verses()
print(f"Available verses: {list(verses.keys())}")
# Output: dict with verse_name → VerseFactory mappings
```

### Trace Event Creation

```bash
export MULTIVERSE_SAFE_EXECUTOR_VERBOSE=1

python -c "
from orchestrator.trainer import Trainer
from core.types import VerseSpec, AgentSpec

trainer = Trainer()
result = trainer.run(
    verse_spec=VerseSpec(verse_name='grid_world'),
    agent_spec=AgentSpec(policy_id='test', algo='ppo'),
    episodes=1,
    max_steps=10,
)

# Print events
import json
with open(f'runs/{result[\"run_id\"]}/events.jsonl') as f:
    for line in f:
        event = json.loads(line)
        print(f\"Step {event['step_idx']}: obs={event['obs']}, reward={event['reward']}\")
"
```

### Inspect Memory Queries

```python
# Check if memory queries executed
import json

run_id = "run_abc123"
with open(f"runs/{run_id}/events.jsonl") as f:
    for line in f:
        event = json.loads(line)
        mem_query = event.get("info", {}).get("memory_query", {})
        if mem_query.get("query_executed"):
            print(f"Step {event['step_idx']}: query executed, {mem_query['match_count']} matches")
```

### Check Central Memory

```python
from memory.central_repository import CentralMemoryConfig, find_similar

cfg = CentralMemoryConfig(root_dir="central_memory")

# Dummy obs for query
obs = {"x": [2], "y": [3]}

matches = find_similar(
    obs=obs,
    cfg=cfg,
    top_k=5,
    verse_name="grid_world",
)

for match in matches:
    print(f"Episode {match.episode_id}, step {match.step_idx}: score={match.score:.2f}")
```

### Verify Curriculum State

```python
from orchestrator.curriculum_controller import load_curriculum_adjustments
import json

adj = load_curriculum_adjustments()
print(json.dumps(adj, indent=2))
```

### Reproduce Run

```python
# Run 1
result = trainer.run(
    verse_spec=VerseSpec(
        verse_name="grid_world",
        seed=42,
        params={"width": 5, "height": 5},
    ),
    agent_spec=AgentSpec(...),
    episodes=5,
    seed=42,
)

# Run 2 (should produce identical events)
result2 = trainer.run(
    verse_spec=VerseSpec(
        verse_name="grid_world",
        seed=42,
        params={"width": 5, "height": 5},
    ),
    agent_spec=AgentSpec(...),
    episodes=5,
    seed=42,
)

# Compare events
import json
with open(f"runs/{result['run_id']}/events.jsonl") as f1, \
     open(f"runs/{result2['run_id']}/events.jsonl") as f2:
    for line1, line2 in zip(f1, f2):
        e1 = json.loads(line1)
        e2 = json.loads(line2)
        if e1["obs"] != e2["obs"]:
            print(f"Difference at step {e1['step_idx']}")
        assert e1["reward"] == e2["reward"]
```

---

## Part 6: Performance Tuning

### Memory Query Budget

```python
# Light memory usage
config = {
    "on_demand_query_budget": 2,       # Very limited
    "on_demand_min_interval": 5,       # Long cooldown
    "retrieval_interval": 50,          # Rare RAR
}
# → ~2-4 memory lookups per episode

# Heavy memory usage
config = {
    "on_demand_query_budget": 32,      # Many queries
    "on_demand_min_interval": 1,       # No cooldown
    "retrieval_interval": 1,           # Every step RAR
}
# → ~32-50 memory lookups per episode
# → More compute, better transfer learning potential
```

### ADR Jitter

```python
# Conservative ADR (small variations)
params = {"adr_enabled": True, "adr_jitter": 0.05}
# → width: 5 * (1 ± 0.05) ≈ 4.75–5.25

# Aggressive ADR (large variations)
params = {"adr_enabled": True, "adr_jitter": 0.30}
# → width: 5 * (1 ± 0.30) ≈ 3.5–6.5
# → More domain randomization, potentially slower learning
```

### Curriculum Plateau Detection

```python
# Fast detection (quick escalation)
curriculum_config = {
    "plateau_window": 3,
    "step_size": 0.10,  # Large steps
}
# → Difficulty increases fast if agent plateaus

# Conservative detection
curriculum_config = {
    "plateau_window": 8,
    "step_size": 0.02,  # Small steps
}
# → Difficulty increases slowly, agent stays longer at current level
```

---

## Part 7: Data Schema Reference

### StepEvent Schema (JSONL format)

```json
{
  "schema_version": "v1",
  "run_id": "run_abc123",
  "t_ms": 1234567890123,
  "episode_id": "episode_e1",
  "step_idx": 5,
  "agent_id": "agent_xyz",
  "policy_id": "memory_recall:v1",
  "policy_version": "1.0",
  "verse_id": "verse_v1",
  "verse_name": "grid_world",
  "verse_version": "0.1",
  "spec_hash": "abc123",
  "seed": 42,
  "obs": {
    "x": [1],
    "y": [0],
    "goal_x": [4],
    "goal_y": [4],
    "t": [5],
    "nearby_obstacles": [2],
    "on_ice": [0],
    "on_teleporter": [0]
  },
  "action": 3,
  "reward": -0.01,
  "done": false,
  "truncated": false,
  "info": {
    "verse_params": null,
    "memory_query": {
      "enabled": true,
      "used": 1,
      "budget": 8,
      "remaining": 7,
      "can_query": true,
      "query_requested": true,
      "query_executed": true,
      "block_reason": "executed",
      "last_query_step_idx": 5,
      "match_count": 3
    },
    "memory_recall_ablation": {
      "enabled": false,
      "eligible": true,
      "randomized": false,
      "prob": 0.0,
      "disabled_apply": false,
      "reason": "eligible_no_ablation"
    },
    "action_info": {
      "memory_aided": true,
      "policy_confidence": 0.87
    },
    "transfer_decision_records": [...],
    "runtime_errors": {
      "counters": {},
      "warnings": []
    }
  }
}
```

### Central Memory Row Schema (JSONL format)

```json
{
  "episode_id": "episode_99",
  "step_idx": 23,
  "run_id": "run_xyz",
  "t_ms": 1234567850000,
  "verse_name": "grid_world",
  "verse_version": "0.1",
  "spec_hash": "abc123",
  "obs": {
    "x": [2],
    "y": [3],
    "goal_x": [4],
    "goal_y": [4],
    "t": [23],
    "nearby_obstacles": [1],
    "on_ice": [0],
    "on_teleporter": [0]
  },
  "action": 3,
  "reward": 0.98,
  "done": false,
  "truncated": false,
  "reached_goal": false,
  "return_from_step": 2.5,
  "cognitive_tags": ["navigation", "2d", "discrete_grid"],
  "strategic_signature": "nav_planning_2step",
  "family": "skill_transfer",
  "obs_vector": [0.12, 0.45, 0.33, ..., 0.78],
  "obs_vector_dim": 128,
  "metadata": {
    "agent_policy_id": "memory_recall:v1",
    "safe_executor_used": false
  }
}
```

---

## Part 8: Checklists for Common Tasks

### Task: Add New Verse

- [ ] Create [verses/new_world.py](verses/new_world.py)
- [ ] Implement Verse protocol (reset, step, seed, close)
- [ ] Add export_state/import_state if SafeExecutor support needed
- [ ] Create factory class in same file
- [ ] Register in [verses/registry.py](verses/registry.py)
- [ ] Add ADR keys to _default_adr_keys()
- [ ] Add cognitive tags to [core/taxonomy.py](core/taxonomy.py)
- [ ] Write smoke test
- [ ] Test with trainer CLI
- [ ] Verify memory compatibility (JSONValue obs/action/reward)

### Task: Enable Memory for Agent

- [ ] Agent implements `memory_query_request(obs) → Dict`
- [ ] Agent implements `on_memory_response(bundle) → None`
- [ ] Agent implements `act_with_hint(obs, hint) → ActionResult`
- [ ] Set `algo="memory_recall"` in AgentSpec
- [ ] Enable config flags:
  - `on_demand_memory_enabled=True`
  - `on_demand_query_budget > 0`
- [ ] Point to central_memory directory
- [ ] Run trainer with memory_enabled

### Task: Debug Low Return

- [ ] Check verse params in run/events.jsonl first line
- [ ] Verify agent action space matches action in events
- [ ] Plot reward distribution across episode
- [ ] Check memory_query state: are queries executing?
- [ ] Verify curriculum state: is difficulty appropriate?
- [ ] Check SafeExecutor veto rate if enabled
- [ ] Run reproduce_run test with fixed seed

---

## Part 9: Performance Monitoring

### Key Metrics to Track

```python
# Per-episode
- steps: How long episode took
- return_sum: Cumulative reward
- reached_goal: Success binary
- memory_queries_used: How many memory lookups
- veto_rate: % actions blocked by SafeExecutor

# Cross-episode
- success_rate: Episodes with reached_goal / total
- mean_return: Average return across episodes
- episode_length_variance: Consistency of performance
- transfer_signal: Does memory help?

# Memory system
- query_latency: How long find_similar takes
- match_quality: Score distribution of returned matches
- deduplication_rate: % events deduplicated
- cache_hit_rate: Embedding cache effectiveness
```

### Extract Metrics from Run

```python
import json
import statistics

run_id = "run_abc123"
returns = []
success_count = 0
total_queries = 0

with open(f"runs/{run_id}/events.jsonl") as f:
    episode_return = 0
    prev_episode = None
    
    for line in f:
        event = json.loads(line)
        
        if event["episode_id"] != prev_episode:
            if prev_episode is not None:
                returns.append(episode_return)
            episode_return = 0
            prev_episode = event["episode_id"]
        
        episode_return += event["reward"]
        
        mem_query = event.get("info", {}).get("memory_query", {})
        if mem_query.get("query_executed"):
            total_queries += 1
        
        if event["done"] and event["info"].get("memory_return_from_step", 0) > 0:
            success_count += 1

print(f"Success rate: {success_count / len(returns):.2f}")
print(f"Mean return: {statistics.mean(returns):.2f}")
print(f"Std return: {statistics.stdev(returns):.2f}")
print(f"Total queries: {total_queries}")
```

---

**Last Updated:** March 2026  
**Version:** 1.0  
**Maintained by:** Multiverse Framework Team
