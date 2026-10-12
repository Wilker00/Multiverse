# Multiverse Intelligence Layer: Platform Architecture & Operations Guide

The **Multiverse Intelligence Layer** expands the core reinforcement-learning framework into a shared-state, concept-indexed operations and simulator control plane.

---

## 1. What Stays RL vs. What Becomes Platform

| Responsibility | RL Layer (Core) | Platform Layer (Intelligence) |
|---|---|---|
| **Agent Training** | `Trainer.run`, Q-learning, PPO, DQN, policy gradients | *Unchanged* (runs on standard episode loops) |
| **Action Execution** | `Verse.step`, `SafeExecutor` vetoes & MCTS shields | `PhysicsBackend` protocol & sim provider specs |
| **Observation Recall** | Cosine similarity on embeddings, default same-verse recall | Concept-tag & skill-id filtering, transfer event logging |
| **State Inspection** | Per-run `events.jsonl` and metric files | Real-time `LiveSnapshot`, terminal map, and local web UI |
| **Promotion & Gating** | Benchmark KPIs, Sentinel health checks | Skill catalog grouping, cross-verse transfer metrics |

---

## 2. Platform CLI & Interfaces

### Multiverse Status
Inspect system readiness, registered verses, active/idle runs, and memory bank statistics:
```bash
python tools/multiverse_cli.py status
python tools/multiverse_cli.py status --json
```

### Live Terminal Map (`hub` / `map`)
Visualize live traffic between verses, central memory, safety shields, and provider availability:
```bash
python tools/multiverse_cli.py map --once
# Or interactive loop:
python tools/multiverse_cli.py map --refresh-sec 2.0
```

### Local Web Map
Launch the zero-dependency standard-library HTTP dashboard (polls `/api/snapshot`):
```bash
multiverse map --web
# Or via script:
python tools/multiverse_live_web.py --port 8765
```
Open `http://127.0.0.1:8765` in any browser to inspect live verses, memory nodes, and event flows.

### Cross-Verse Transfer Overview
Track and benchmark cross-verse skill transfer events:
```bash
python tools/multiverse_cli.py transfer overview
python tools/multiverse_cli.py transfer overview --json
```

### Memory Inspection by Concept
Test what an agent would query and retrieve when requesting a specific skill family:
```bash
python tools/multiverse_cli.py memory inspect --verse grid_world --concepts navigation,risk
```

---

## 3. Concept Taxonomy & Skills

Verses are indexed into canonical concept tags:
- `navigation`
- `planning`
- `risk`
- `spatial_reasoning`
- `resource`
- `strategy`
- `memory`
- `robotics`
- `safety`

### Querying Skills
Before starting training, agents can inspect the central repository's skill catalog:
```python
from memory.skill_catalog import list_skills
from memory.central_repository_support import CentralMemoryConfig

cfg = CentralMemoryConfig(root_dir="central_memory")
skills = list_skills(cfg)
for s in skills:
    print(s["skill_id"], s["verse_name"], s["success_count"])
```

### Opt-In Cross-Verse Retrieval
By default, `recall_same_verse_only=True` is preserved. When cross-verse transfer is explicitly enabled on `MemoryRecallAgent`, agents can pass:
- `concept_tags`: Filter candidate memories to matching concept categories.
- `concept_weight`: Opt-in similarity bonus based on concept tag overlap.
- `min_memory_priority`: Filter out low-priority failure transitions.

---

## 4. Physics & Simulator Extension Guide

The framework provides an embodied simulation interface that plugs into Isaac Lab, NVIDIA Warp, or custom engines without modifying `core/rollout.py`.

### Architecture
```
Verse (e.g. embodied_point_world)
      │
      ▼
PhysicsBackend (integrations/physics/backend.py)
   ├── ToyPhysicsBackend (integrations/physics/toy_backend.py - CPU point-mass)
   └── WarpPhysicsBackend (integrations/physics/warp_backend.py - GPU Warp plug point)
```

### Integrating an External Engine (e.g., NVIDIA Warp / Isaac Lab)
1. **Implement `PhysicsBackend`**:
   Subclass or adhere to the protocol in [integrations/physics/backend.py](file:///c:/Users/simok38/Documents/GitHub/Multiverse/integrations/physics/backend.py) (`reset()`, `step(action)`, `state()`).
2. **Register the Provider in `integrations/sim_registry.py`**:
   Add a `SimProviderSpec` declaring capabilities, required modules, and supported verses.
3. **Register the Verse Adapter**:
   Expose standard flat observation and action spaces through `verses/registry.py` so standard RL agents (PPO, Q-learning, Decision Transformers) can interact seamlessly.

---

## 5. Vision-Action Grounding Layer (Game → Robot Transfer)

The **VisionActionMapper** ([integrations/vision_action_mapper.py](file:///c:/Users/simok38/Documents/GitHub/Multiverse/integrations/vision_action_mapper.py)) translates discrete navigation and pixel/task observations from games into continuous joint velocities and task-space waypoints for physical embodiments.

```
Game Trajectory (Discrete / Pixels)
       │
       ▼
SpatialIntentExtractor (Relative displacement Δp & velocity)
       │
       ▼
InverseKinematicsEngine (Analytical / Damped Differential IK)
       │
       ▼
Continuous Joint Commands [dq1, dq2, ...] → Robot (e.g. robot_reacher_world)
```

### Running the End-to-End Game → Robot Proof-of-Concept
```bash
python tools/game_to_robot_transfer_experiment.py --episodes 15 --seed 42
```
This benchmark:
1. Ingests spatial navigation patterns from a 2D grid/game into Central Memory.
2. Queries concepts (`spatial_reasoning`, `navigation`) across verses.
3. Ground discrete game actions into continuous joint velocities for `robot_reacher_world`.
4. Validates that the transferred prior reduces distance-to-target by **> 1.6x** compared to unguided exploration from scratch, writing results to `models/validation/game_to_robot_transfer_benchmark.json`.

