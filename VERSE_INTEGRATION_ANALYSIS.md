# Comprehensive Verse Integration Analysis

## Overview

The Multiverse orchestrator implements a **registry-based architecture** where verses are decoupled from the trainer through the `VerseSpec → factory → Verse` pattern. This enables plug-and-play environments without modifying orchestrator code.

---

## 1. Trainer Integration

### 1.1 Verse Instantiation Flow

**File:** [orchestrator/trainer.py](orchestrator/trainer.py)

```python
# Trainer.run() execution sequence:

1. VerseSpec validation
   ↓
2. Create VerseRef (runtime handle)
   - verse_id: unique instance identifier
   - spec_hash: deterministic hash of full spec
   ↓
3. create_verse(verse_spec) via registry
   ├─ Load curriculum adjustments
   ├─ Apply ADR (Automatic Domain Randomization)
   ├─ Merge cognitive tags from taxonomy
   └─ Call factory.create(spec_eff)
   ↓
4. Extract observation_space & action_space
   ↓
5. Create Agent with these specs
   ↓
6. Optional: Wrap agent in SafeExecutor
```

**Example from trainer.py (lines 130-250):**

```python
# Create verse via registry
verse = create_verse(verse_spec)

# Agent creation uses verse's spaces
agent = create_agent(
    spec=agent_runtime_spec,
    observation_space=verse.observation_space,
    action_space=verse.action_space,
)

# Safe executor holds verse reference for checkpoint/rewind
safe_executor = SafeExecutor(
    config=safe_cfg,
    verse=verse,
    fallback_agent=fallback_agent,
)
```

### 1.2 Training Loop Setup

```python
# RolloutConfig captures verse-agent binding
rollout_cfg = RolloutConfig(
    schema_version=self.schema_version,
    max_steps=max_steps,
    train=train,
    safe_executor=safe_executor,
    
    # Memory integration flags
    on_demand_memory_enabled=bool(
        algo in ("memory_recall", "planner_recall")
    ),
    on_demand_memory_root="central_memory",
    on_demand_query_budget=8,
    on_demand_min_interval=2,
)

# Run episodes with logging
results = run_episodes(
    verse=verse,
    verse_ref=verse_ref,
    agent=agent,
    agent_ref=agent_ref,
    run=run,
    config=rollout_cfg,
    episodes=episodes,
    seed=seed,
    on_step=make_on_step_writer(logger),  # Stream events to disk
)

# Auto-index for knowledge market
build_episode_index(run_dir)
```

### 1.3 Reward Collection & Logging

**Data Flow:**
```
verse.step(action)
├─ Returns: StepResult(obs, reward, done, truncated, info)
│
└─→ SafeExecutor.post_step() [optional]
    └─→ May modify reward/obs based on safety state
    
└─→ make_step_event() creates StepEvent
    ├─ Captures: obs, action, reward, done
    ├─ Captures: verse_name, spec_hash
    ├─ Captures: episode_id, step_idx
    └─ Captures: action_info, memory_query_state, runtime_errors
    
└─→ on_step(event) streams to EventLogger
    └─→ Appends to run_id/events.jsonl
```

---

## 2. Rollout Engine

### 2.1 Step-by-step Execution Model

**File:** [core/rollout.py](core/rollout.py) lines 129-650+

```python
def run_episode(verse, agent, config, seed=None):
    """
    Core episode execution: reset → step × max_steps → learn
    """
    
    # PHASE 1: Initialization
    if seed is not None:
        verse.seed(seed)
        agent.seed(seed)
    
    # Optional SafeExecutor reset
    if config.safe_executor:
        config.safe_executor.reset_episode(seed)
    
    reset_result = verse.reset()  # ← First verse interaction
    obs = reset_result.obs
    
    # PHASE 2: Episode Loop
    step_idx = 0
    done = False
    return_sum = 0.0
    
    while not done and step_idx < config.max_steps:
        # ========== MEMORY QUERY PHASE ==========
        hint = None
        
        if config.retriever and step_idx % retrieval_interval == 0:
            # Try local verse match first
            flt = EpisodeFilter(verse_name=verse_ref.verse_name, 
                              reached_goal=True)
            matches = config.retriever.filter_episodes(flt)
            
            # Fall back to strategic signature match (cross-verse)
            if not matches:
                sig = _strategy_signature(obs, verse_ref.verse_name)
                if sig:
                    flt_strat = EpisodeFilter(strategic_match=sig, 
                                            reached_goal=True)
                    matches = config.retriever.filter_episodes(flt_strat)
            
            if matches:
                hint = {
                    "successful_episode": matches[0]["episode_id"],
                    "source_verse": matches[0]["verse_name"],
                }
        
        # ========== ON-DEMAND MEMORY QUERY ==========
        if config.on_demand_memory_enabled and agent.memory_query_request:
            can_query = (memory_queries_used < query_budget and
                        step_idx - last_query_step >= min_interval)
            
            if can_query:
                req = agent.memory_query_request(obs, step_idx)
                
                if isinstance(req, dict):
                    matches = find_similar(
                        obs=req.get("query_obs", obs),
                        cfg=CentralMemoryConfig(...),
                        top_k=req.get("top_k", 3),
                        verse_name=req.get("verse_name"),
                        min_score=req.get("min_score", -1.0),
                        memory_families=req.get("memory_families"),
                    )
                    
                    memory_queries_used += 1
                    last_memory_query_step = step_idx
                    
                    # Build hint bundle with matches
                    bundle = _build_memory_bundle(req, matches)
                    hint["memory_recall"] = bundle
                    
                    # Notify agent of results
                    if hasattr(agent, "on_memory_response"):
                        agent.on_memory_response(bundle)
        
        # ========== ACTION SELECTION PHASE ==========
        if config.safe_executor:
            action_result = config.safe_executor.select_action(agent, obs)
        else:
            # Apply recall ablation if configured
            if hint and hasattr(agent, "act_with_hint"):
                if config.on_demand_recall_ablation_prob > rand():
                    hint["_memory_recall_control"] = {
                        "disable_apply": True,
                        "policy": "randomized_ablation",
                    }
            
            if hasattr(agent, "act_with_hint"):
                action_result = agent.act_with_hint(obs, hint)
            else:
                action_result = agent.act(obs)
        
        # ========== ENVIRONMENT STEP ==========
        action = action_result.action
        step_result = verse.step(action)  # ← Core environment interaction
        
        # Optional post-step safety processing
        if config.safe_executor:
            step_result = config.safe_executor.post_step(
                obs=obs,
                action_result=action_result,
                step_result=step_result,
                step_idx=step_idx,
                primary_agent=agent,
            )
        
        # ========== EVENT RECORDING ==========
        event = make_step_event(
            schema_version=config.schema_version,
            run=run,
            episode_id=episode_id,
            step_idx=step_idx,
            agent=agent_ref,
            verse=verse_ref,
            obs=obs,
            action=action,
            reward=step_result.reward,
            done=step_result.done,
            truncated=step_result.truncated,
            seed=seed,
            info={
                "verse_params": verse.spec.params if step_idx == 0 else None,
                "memory_query": memory_query_state,
                "action_info": action_result.info,
                "selector_routing": selector_routing,
                "transfer_decision_records": tdr_list,
                "runtime_errors": {...},
            },
        )
        
        on_step(event)  # Stream to logger
        
        if config.collect_transitions:
            transitions.append(Transition(...))
        
        # ========== STATE UPDATE ==========
        obs = step_result.obs
        return_sum += step_result.reward
        done = step_result.done or step_result.truncated
        step_idx += 1
    
    # PHASE 3: Learning (if enabled)
    if config.train and config.collect_transitions:
        batch = ExperienceBatch(transitions=transitions)
        train_metrics = agent.learn(batch)
    
    return RolloutResult(...)
```

### 2.2 Episode Management

```python
def run_episodes(verse, agent, config, episodes, seed=None):
    """Run multiple episodes with seed offset."""
    
    results = []
    for ep in range(episodes):
        ep_seed = None if seed is None else seed + ep
        
        result = run_episode(
            verse=verse,
            agent=agent,
            seed=ep_seed,
            config=config,
        )
        
        results.append(result)
    
    return results
```

### 2.3 Key Data Structures Flowing Between Components

```
Input to run_episode:
├─ verse: Verse instance
├─ verse_ref: VerseRef(verse_id, verse_name, spec_hash)
│  └─ Used to tag all events and memory lookups
├─ agent: Agent instance
├─ agent_ref: AgentRef(agent_id, policy_id, policy_version)
│  └─ Used to track agent behavior across runs
└─ config: RolloutConfig
   ├─ safe_executor: SafeExecutor | None
   ├─ retriever: RetrievalClient | None
   ├─ on_demand_memory_root: str
   ├─ on_demand_query_budget: int
   └─ on_demand_recall_ablation_prob: float

Per-step data:
├─ obs: JSONValue (from verse.reset/step)
├─ action: JSONValue (from agent.act)
├─ reward: float (from verse.step)
├─ done: bool (from verse.step)
├─ hint: Dict with memory matches & strategic signature
├─ action_result: ActionResult (agent selection metadata)
├─ step_result: StepResult (verse feedback)
└─ event: StepEvent (logged to JSONL)

Output from run_episode:
└─ RolloutResult
   ├─ events: List[StepEvent]
   ├─ episode_id: str
   ├─ steps: int (actual steps taken)
   ├─ return_sum: float (cumulative reward)
   └─ train_metrics: Dict[str, float] (learning progress)
```

---

## 3. Safety Layer

### 3.1 SafeExecutor Interaction with Verses

**File:** [core/safe_executor.py](core/safe_executor.py)

```python
class SafeExecutor:
    def __init__(self, config, verse, fallback_agent=None):
        self.verse = verse  # Holds reference to wrapped verse
        self._checkpoint_state = None
        self._checkpoint_obs = None
        
        # Verse-specific MCTS overrides
        vname = getattr(self.verse.spec, "verse_name", "").lower()
        if vname == "cliff_world":
            config.danger_threshold = 0.60  # Safer default
```

### 3.2 Safety Check Points

```
Rollout Integration Points:
┌─────────────────────────────────────────────────────────┐
│ 1. SELECT_ACTION                                         │
│    verse_obs → safe_executor.select_action(agent, obs)  │
│              → competence_shield veto                    │
│              → recursive_fallback_trigger                │
│              → returns ActionResult                      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. ENVIRONMENT STEP                                      │
│    action → verse.step(action)                           │
│         → StepResult(obs, reward, done)                  │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. POST_STEP SAFETY                                      │
│    [obs, action, reward] → safe_executor.post_step()    │
│                          → checkpoint_recovery          │
│                          → rewind_to_checkpoint          │
│                          → modified_StepResult           │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Verse State Management

```python
# SafeExecutor uses verse's state checkpoint API:

def reset_episode(self, seed):
    """Prepare verse for checkpointing."""
    if self.verse and hasattr(self.verse, "export_state"):
        # This resets verse tracking for new episode
        pass

def can_checkpoint(self):
    """Check if verse supports state export."""
    return hasattr(self.verse, "export_state") and \
           hasattr(self.verse, "import_state")

def save_checkpoint(self, obs):
    """Save full verse state for rollback."""
    if self.can_checkpoint():
        self._checkpoint_state = self.verse.export_state()
        self._checkpoint_obs = obs
        self._checkpoint_agent_state = { ... }

def rewind_to_checkpoint(self):
    """Restore verse to last checkpoint."""
    if self._checkpoint_state is not None:
        self.verse.import_state(self._checkpoint_state)
        obs = self._checkpoint_obs
        # Restore agent state as well
        return obs
```

### 3.4 Verse-Specific Configuration

```yaml
# From configs/multiverse.dev.yaml

safe_executor:
  enabled: true
  danger_threshold: 0.95
  min_action_confidence: 0.05
  planner_enabled: false
  mcts_enabled: false
  
  # Applied per-verse in trainer.py:
  # if verse_name == "cliff_world" and "danger_threshold" not in config:
  #     config["danger_threshold"] = 0.60
```

---

## 4. Memory System Integration

### 4.1 Memory Retrieval from Verses

**File:** [memory/central_repository.py](memory/central_repository.py) & [core/rollout.py](core/rollout.py) lines 200-400

```python
def find_similar(
    obs,
    cfg: CentralMemoryConfig,
    top_k: int = 3,
    verse_name: Optional[str] = None,
    min_score: float = -1.0,
    memory_families: Optional[Set[str]] = None,
    memory_types: Optional[Set[str]] = None,
    trajectory_window: int = 0,
) -> List[ScenarioMatch]:
    """
    Query central memory for experiences similar to current observation.
    
    Process:
    1. Convert obs to embedding vector
    2. Filter by verse_name (if specified) + memory families/types
    3. Use ANN (FAISS) or cosine_similarity to find top_k
    4. Return ScenarioMatch objects with scores
    """
    
    # Encode observation
    obs_vector = encode_observation(obs)
    
    # Build similarity cache from tier files
    cache = _get_similarity_cache_for_path(
        mem_path=cfg.memories_path,
        tier_policy=cfg.tier_policy,
    )
    
    # Filter candidates
    candidates = []
    for i, row in enumerate(cache.rows):
        # Filter by verse if specified
        if verse_name and row.get("verse_name", "").lower() != verse_name.lower():
            continue
        
        # Filter by memory family/type
        if memory_families and row.get("family") not in memory_families:
            continue
        
        candidates.append((i, row))
    
    # Compute scores via cosine similarity
    scores = cosine_similarity(
        [obs_vector],
        [row["obs_vector"] for _, row in candidates],
    )[0]
    
    # Rank and return top_k
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return [
        ScenarioMatch(
            episode_id=row["episode_id"],
            step_idx=row["step_idx"],
            obs=row["obs"],
            action=row["action"],
            reward=row["reward"],
            verse_name=row["verse_name"],
            score=float(score),
            trajectory_window=trajectory_window,
        )
        for (_, row), score in ranked[:top_k]
        if score >= min_score
    ]
```

### 4.2 Cognitive Tags & Strategic Signatures

```python
# From verses/registry.py: create_verse()

# 1. Inject cognitive tags from taxonomy
mem_tags = cognitive_tags_for_verse(spec.verse_name)
if mem_tags:
    merged_tags = list(spec_eff.tags)
    for t in mem_tags:
        if t not in merged_tags:
            merged_tags.append(t)
    spec_eff = dataclasses.replace(spec_eff, tags=merged_tags)

# 2. Example tags for chess_world:
# ["strategy_games", "board_control", "chess_like", "transferable_logic", "turn_based"]

# From core/rollout.py: Strategic bridge matching
if not verse_matches:
    sig = _strategy_signature(obs, verse_ref.verse_name)
    if sig:
        # Allow Chess agent to recall patterns from Go
        flt_strat = EpisodeFilter(strategic_match=sig, reached_goal=True)
        matches = config.retriever.filter_episodes(flt_strat)
```

### 4.3 Memory Query Flow in Rollout

```
Step N of episode:
│
├─ If step_idx % retrieval_interval == 0:
│  ├─ Retrieval-Augmented Rollouts (RAR)
│  ├─ Try: verse_name match + reached_goal = True
│  ├─ Else: strategic signature match
│  └─ Populate hint["successful_episode"]
│
├─ If on_demand_memory_enabled and agent.memory_query_request:
│  ├─ Check budget: memory_queries_used < query_budget
│  ├─ Check cooldown: (step_idx - last_query_step) >= min_interval
│  ├─ Request: req = agent.memory_query_request(obs)
│  ├─ If request dict:
│  │  ├─ Query: matches = find_similar(
│  │  │           obs=req.get("query_obs", obs),
│  │  │           top_k=req.get("top_k", 3),
│  │  │           verse_name=req.get("verse_name"),
│  │  │           memory_families=...,
│  │  │         )
│  │  ├─ Increment: memory_queries_used += 1
│  │  ├─ Build: bundle = _build_memory_bundle(matches)
│  │  ├─ Populate: hint["memory_recall"] = bundle
│  │  └─ Notify: agent.on_memory_response(bundle)
│  └─ Track: memory_query_state for logging
│
└─ Pass: hint to agent.act_with_hint(obs, hint)
```

### 4.4 Memory Storage & Retrieval

```python
# Memory stored as:
{
    "episode_id": "episode_12345",
    "step_idx": 42,
    "obs": {...},  # JSONValue from verse.reset/step
    "action": {...},  # Agent's chosen action
    "reward": 1.5,  # From verse.step
    "verse_name": "grid_world",
    "verse_version": "0.1",
    "spec_hash": "abc123",
    "cognitive_tags": ["navigation", "2d", "discrete_grid"],
    "strategic_signature": "nav_2d_discrete",
    "reached_goal": true,
    "return_from_step": 3.5,
    "obs_vector": [0.1, 0.2, ...],  # Embedding
}

# Stored in:
# central_memory/
# ├─ stm_memories.jsonl  (Short-term: recent episodes)
# ├─ ltm_memories.jsonl  (Long-term: selected experiences)
# ├─ memories.jsonl.simcache.json  (Cached embeddings)
# └─ dedupe_index.json  (Deduplication tracking)
```

---

## 5. End-to-End Flow

### 5.1 Complete Example: Selecting Verse → Training → Memory

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: EXPERIMENT CONFIGURATION                             │
└──────────────────────────────────────────────────────────────┘

VerseSpec(
    spec_version="v1",
    verse_name="grid_world",
    verse_version="0.1",
    seed=42,
    params={
        "width": 5,
        "height": 5,
        "step_penalty": -0.01,
        "adr_enabled": True,
        "adr_jitter": 0.10,
    },
    tags=["navigation", "2d"],
)

AgentSpec(
    spec_version="v1",
    policy_id="dqn:v1",
    policy_version="2.0",
    algo="memory_recall",  # ← Enables memory queries
    config={
        "on_demand_memory_enabled": True,
        "on_demand_memory_root": "central_memory",
        "on_demand_query_budget": 8,
    },
)

┌──────────────────────────────────────────────────────────────┐
│ STEP 2: TRAINER INITIALIZATION                               │
└──────────────────────────────────────────────────────────────┘

trainer = Trainer()

result = trainer.run(
    verse_spec=verse_spec,
    agent_spec=agent_spec,
    episodes=10,
    max_steps=50,
    seed=42,
)

# Inside trainer.run():
# 1. spec_hash = _hash_verse_spec(verse_spec)
# 2. verse_ref = VerseRef.create(verse_name, spec_hash)
# 3. verse = create_verse(verse_spec)

┌──────────────────────────────────────────────────────────────┐
│ STEP 3: VERSE CREATION (via registry)                        │
└──────────────────────────────────────────────────────────────┘

# In verses/registry.py: create_verse()

# Step 3a: Load curriculum adjustments
adj = load_curriculum_adjustments()  # From models/curriculum_adjustments.json
# If grid_world reached plateau:
#   params["action_noise"] += 0.05
#   params["adr_jitter"] += 0.05

# Step 3b: Apply ADR (Automatic Domain Randomization)
params["width"] = int(5 * (1 + uniform(-0.10, 0.10)))  # ~= 5±0.5
params["step_penalty"] *= (1 + uniform(-0.10, 0.10))  # ~= -0.01±0.001

# Step 3c: Inject cognitive tags
# VERSE_TAGS["grid_world"] = ["navigation", "2d", "discrete_grid"]
spec_eff = dataclasses.replace(
    spec_eff,
    tags=["navigation", "2d", "discrete_grid"],
)

# Step 3d: Instantiate verse via factory
factory = get_factory("grid_world")  # GridWorldFactory
verse = factory.create(spec_eff)
    # → GridWorldVerse
    #   ├─ self.spec = spec_eff (with tags merged)
    #   ├─ self.params = GridWorldParams(...)
    #   ├─ self.observation_space = SpaceSpec(type="dict", ...)
    #   └─ self.action_space = SpaceSpec(type="discrete", n=4)

┌──────────────────────────────────────────────────────────────┐
│ STEP 4: AGENT CREATION & SETUP                               │
└──────────────────────────────────────────────────────────────┘

# Agent constructor informed by verse's spaces
agent = create_agent(
    spec=agent_spec,
    observation_space=verse.observation_space,  # Dict space
    action_space=verse.action_space,            # Discrete(4)
)
# → MemoryRecallAgent(observation_space, action_space)
#   ├─ Can call memory_query_request(obs)
#   ├─ Can call on_memory_response(bundle)
#   └─ Can call act(obs) and act_with_hint(obs, hint)

┌──────────────────────────────────────────────────────────────┐
│ STEP 5: ROLLOUT CONFIGURATION                                │
└──────────────────────────────────────────────────────────────┘

rollout_cfg = RolloutConfig(
    max_steps=50,
    train=True,
    collect_transitions=True,
    
    # Memory system hooks
    on_demand_memory_enabled=True,
    on_demand_memory_root="central_memory",
    on_demand_query_budget=8,
    on_demand_min_interval=2,
    
    # Retrieval-augmented rollouts (RAR)
    retriever=RetrievalClient(
        RetrievalConfig(run_dir=f"runs/{run_id}")
    ),
    retrieval_interval=10,
)

┌──────────────────────────────────────────────────────────────┐
│ STEP 6: EPISODE EXECUTION (run_episode)                      │
└──────────────────────────────────────────────────────────────┘

# Episode 1, Step 1:
verse.seed(42)
agent.seed(42)

reset_result = verse.reset()
# GridWorldVerse.reset():
#   ├─ Place agent at (0, 0)
#   ├─ Place goal at (4, 4)
#   ├─ Generate obstacles ensuring path exists
#   ├─ Generate ice patches
#   ├─ Generate teleporter pairs
#   └─ Return ResetResult(obs={...}, info={...})

obs = reset_result.obs
# {
#   "x": [0],
#   "y": [0],
#   "goal_x": [4],
#   "goal_y": [4],
#   "t": [0],
#   "nearby_obstacles": [2],
#   "on_ice": [0],
#   "on_teleporter": [0],
# }

# Episode 1, Step 5 (memory query triggers):
if 5 % 10 == 0:  # retrieval_interval = 10, so not this step
    pass

# Check on-demand memory
if config.on_demand_memory_enabled:
    can_query = (
        memory_queries_used < 8 and
        (5 - last_query_step) >= 2
    )
    
    if can_query:
        # Agent decides if it needs memory
        req = agent.memory_query_request(obs)
        # MemoryRecallAgent might return:
        # {
        #   "reason": "uncertainty",
        #   "query_obs": obs,
        #   "top_k": 3,
        #   "verse_name": None,  # Cross-verse search
        #   "min_score": 0.7,
        # }
        
        if isinstance(req, dict):
            # Query central memory
            matches = find_similar(
                obs=obs,
                cfg=CentralMemoryConfig(root_dir="central_memory"),
                top_k=3,
                verse_name=None,
                min_score=0.7,
            )
            
            # matches = [
            #   ScenarioMatch(
            #     episode_id="episode_99",
            #     step_idx=23,
            #     obs={...},
            #     action=1,  # "right"
            #     reward=0.98,
            #     verse_name="grid_world",
            #     score=0.92,
            #   ),
            #   ScenarioMatch(...),
            #   ...
            # ]
            
            memory_queries_used += 1
            last_query_step = 5
            
            # Build memory bundle
            bundle = {
                "matches": [
                    {
                        "episode_id": m.episode_id,
                        "step_idx": m.step_idx,
                        "action": m.action,
                        "reward": m.reward,
                        "verse_name": m.verse_name,
                        "score": m.score,
                    }
                    for m in matches
                ],
                "query_timestamp": now_ms(),
            }
            
            # Notify agent
            agent.on_memory_response(bundle)
            hint["memory_recall"] = bundle

# Select action (with memory hint)
if hasattr(agent, "act_with_hint"):
    action_result = agent.act_with_hint(obs, hint)
    # MemoryRecallAgent.act_with_hint():
    #   ├─ Compute policy output: policy_logits = forward(obs)
    #   ├─ If hint["memory_recall"]:
    #   │  ├─ Retrieve top match action: top_action = matches[0]["action"]
    #   │  ├─ Blend: final_action = weighted_sample(
    #   │  │                          policy_logits,
    #   │  │                          top_action,
    #   │  │                          weight=0.3  # Memory weight
    #   │  │                        )
    #   │  └─ Track: action_result.info["memory_aided"] = True
    #   └─ Else:
    #      └─ action = sample(policy_logits)
else:
    action_result = agent.act(obs)

action = action_result.action
# action = 3  # "right" (influenced by memory hint)

# Execute environment step
step_result = verse.step(action)
# GridWorldVerse.step(action=3):
#   ├─ dx, dy = {0: [0, -1], 1: [0, 1], 2: [-1, 0], 3: [1, 0]}[3]
#   │         = [1, 0]  # Move right
#   ├─ new_x, new_y = 0 + 1, 0 + 0 = (1, 0)
#   ├─ Check blocked: not in obstacles, not teleporter → OK
#   ├─ Compute reward:
#   │  ├─ done = (x == goal_x and y == goal_y) = False
#   │  ├─ reward = -0.01 (step_penalty)
#   │  └─ if x == goal_x and y == goal_y: reward += 1.0 (goal_reward)
#   ├─ Return StepResult(
#   │   obs={x: [1], y: [0], ...},
#   │   reward=-0.01,
#   │   done=False,
#   │   truncated=False,
#   │   info={...}
#   │ )
#   └─ (Internal state updated: self._x, self._y, self._t)

# Record step event
event = make_step_event(
    schema_version="v1",
    run=RunRef(run_id="run_abc123"),
    episode_id="episode_e1",
    step_idx=5,
    agent=AgentRef(agent_id="agent_xyz", policy_id="dqn:v1", ...),
    verse=VerseRef(verse_id="verse_v1", verse_name="grid_world", ...),
    obs=obs,
    action=action,
    reward=step_result.reward,
    done=step_result.done,
    truncated=step_result.truncated,
    seed=42,
    info={
        "verse_params": {...} if step_idx == 0 else None,
        "memory_query": {
            "enabled": True,
            "used": 1,
            "budget": 8,
            "remaining": 7,
            "can_query": True,
            "query_executed": True,
            "match_count": 3,
            "blocked_reason": "executed",
        },
        "action_info": {
            "memory_aided": True,
            "policy_confidence": 0.85,
        },
        "runtime_errors": {...},
    },
)

on_step(event)  # Write to runs/run_abc123/events.jsonl

┌──────────────────────────────────────────────────────────────┐
│ STEP 7: EPISODE COMPLETION & MEMORY STORAGE                  │
└──────────────────────────────────────────────────────────────┘

# After episode finishes (done=True or step_idx >= max_steps):

# Train if configured
if config.train and config.collect_transitions:
    batch = ExperienceBatch(transitions=[...])
    train_metrics = agent.learn(batch)
    # Updates policy weights

# Return episode results
return RolloutResult(
    events=events,  # All StepEvents
    episode_id="episode_e1",
    steps=27,
    return_sum=0.73,  # Sum of rewards
    train_metrics={"loss": 0.12, "q_loss": 0.08},
)

# Post-training: Memory ingest
# Each StepEvent can be ingested into central_memory:
#
# for event in events:
#     ingest_event(
#         episode_id=event.episode_id,
#         step_idx=event.step_idx,
#         obs=event.obs,
#         action=event.action,
#         reward=event.reward,
#         verse_name=event.verse_name,
#         tags=event.tags,
#         reached_goal=(event.done and event.reward > threshold),
#     )
#
# → Appended to central_memory/memories.jsonl
# → Deduped against central_memory/dedupe_index.json
# → Embedding computed and cached
# → Available for next round of queries

┌──────────────────────────────────────────────────────────────┐
│ STEP 8: NEXT EPISODE WITH MEMORY                             │
└──────────────────────────────────────────────────────────────┘

# Episode 2 can now query from Episode 1's memory

# If episode 1 succeeded (reached_goal=True):
#   → find_similar will return Episode 1's steps
# → Agent can recall successful patterns
# → Training signal propagates faster
```

---

## 6. Key Data Structures

### 6.1 VerseSpec & VerseRef

```python
@dataclass(frozen=True)
class VerseSpec:
    spec_version: str              # "v1"
    verse_name: str                # "grid_world"
    verse_version: str             # "0.1"
    seed: Optional[int]
    tags: List[str]                # Cognitive tags
    params: Dict[str, JSONValue]   # Environment-specific params
    metadata: Dict[str, JSONValue] # Arbitrary metadata

@dataclass(frozen=True)
class VerseRef:
    verse_id: str                  # "verse_abc123" (unique instance)
    verse_name: str                # "grid_world"
    verse_version: str             # "0.1"
    spec_hash: str                 # SHA1 of VerseSpec (for reproducibility)
```

### 6.2 StepEvent (Core Logging Unit)

```python
@dataclass(frozen=True)
class StepEvent:
    schema_version: str
    run_id: str                    # "run_xyz789"
    t_ms: int                      # Timestamp
    episode_id: str                # "episode_e1"
    step_idx: int                  # Step within episode
    
    # Agent context
    agent_id: str
    policy_id: str
    policy_version: str
    
    # Environment context (← Verse tracking)
    verse_id: str
    verse_name: str
    verse_version: str
    spec_hash: str
    
    # Interaction data
    obs: JSONValue                 # From verse.reset/step
    action: JSONValue              # From agent.act
    reward: float                  # From verse.step
    done: bool
    truncated: bool
    seed: Optional[int]
    
    # Metadata
    info: Dict[str, Any]           # Enriched with memory_query, action_info, etc.
```

### 6.3 Memory Query State

```python
memory_query_state = {
    "enabled": True,
    "used": 1,                     # Queries executed
    "budget": 8,                   # Total allowed
    "remaining": 7,                # Budget - used
    "can_query": True,
    "query_requested": True,
    "query_executed": True,
    "block_reason": "executed",
    "last_query_step_idx": 5,
    "match_count": 3,              # Successful matches found
    "query_reason": "uncertainty",
}
```

---

## 7. Configuration

### 7.1 Experiment Configuration Pattern

```python
# experiment/config.py

@dataclass
class ExperimentConfig:
    name: str
    verse_spec: VerseSpec         # What to train on
    agent_spec: AgentSpec         # What to use for learning
    episodes: int
    max_steps: int
    seed: Optional[int]

# Load: cfg = load_experiment("experiment/my_run.json")
# Format:
# {
#   "name": "grid_world_dqn",
#   "verse_spec": {
#     "spec_version": "v1",
#     "verse_name": "grid_world",
#     "params": {"width": 5, "height": 5, "adr_enabled": true}
#   },
#   "agent_spec": {
#     "spec_version": "v1",
#     "policy_id": "dqn:v1",
#     "algo": "memory_recall",
#     "config": {"on_demand_memory_enabled": true}
#   },
#   "episodes": 10,
#   "max_steps": 50
# }
```

### 7.2 Curriculum Configuration

```yaml
# orchestrator/curriculum_controller.py

# Stored in: models/curriculum_adjustments.json
{
  "version": "v1",
  "verses": {
    "grid_world": {
      "history": [
        {"t_ms": 1234567890, "success_rate": 0.45, "mean_return": 2.3},
        {"t_ms": 1234567900, "success_rate": 0.48, "mean_return": 2.5},
      ],
      "noise": 0.05,               # ← Applied to verse params
      "stochasticity": 0.10,
      "partial_observability": 0.15,
      "distractors": 2,
      "mode": "plateau_harder",    # Curriculum state
    },
    ...
  }
}

# Applied in verses/registry.py:
# apply_curriculum_params(
#   verse_name="grid_world",
#   params={"action_noise": 0.02, ...},
#   adjustments=curriculum_adjustments["grid_world"],
# )
```

### 7.3 Verse Params Examples

```python
# From experiment/benchmark_suite.yaml

{
  "verse_name": "cliff_world",
  "params": {
    "adr_enabled": false,
    "width": 12,
    "height": 4,
    "step_penalty": -1.0,
    "cliff_penalty": -100.0,
    "end_on_cliff": false,
    "max_steps": 100,
  }
}

# Each verse defines its own params via @dataclass:
# GridWorldParams: start_x, start_y, width, height, obstacle_count, ...
# CliffWorldParams: width, height, step_penalty, cliff_penalty, ...
```

---

## 8. Integration Point Summary Table

| Component | File | Key Role | Verse Interaction |
|-----------|------|----------|-------------------|
| **Trainer** | orchestrator/trainer.py | Orchestrates run lifecycle | Creates verse, passes to rollout |
| **Registry** | verses/registry.py | VerseSpec → Verse factory | ADR, curriculum, tag injection |
| **Rollout** | core/rollout.py | Main episode loop | Calls verse.reset/step, collects events |
| **SafeExecutor** | core/safe_executor.py | Safety layer | Checkpoints, rewinds, vetos via export_state/import_state |
| **Memory** | memory/central_repository.py | Similarity search | Filters by verse_name, uses embeddings |
| **Curriculum** | orchestrator/curriculum_controller.py | Difficulty scaling | Adjusts verse params per signal |
| **Taxonomy** | core/taxonomy.py | Cognitive tags | Tags injected into VerseSpec |
| **Event Logger** | memory/event_log.py | Stream recording | Records StepEvents with verse_ref |

---

## 9. Callback/Event System

### 9.1 Per-Step Callback

```python
# core/rollout.py: on_step interface

on_step = make_on_step_writer(logger)

# Inside loop:
on_step(event)
# → Writes event to disk immediately (streaming)
# → Enables parallel processing without holding events in memory

# Event flows:
# StepEvent → on_step(event)
#          → logger.write_event(event)
#          → File: runs/run_id/events.jsonl (append)
```

### 9.2 Memory Response Callback

```python
# core/rollout.py: Agent-memory integration

if hasattr(agent, "on_memory_response"):
    agent.on_memory_response(bundle)

# MemoryRecallAgent.on_memory_response(bundle):
#   ├─ Updates internal recall buffer
#   ├─ May adjust policy confidence based on match quality
#   └─ Enables learning from memory patterns
```

### 9.3 Training Metrics Callback

```python
# core/rollout.py: Post-episode learning

if config.train:
    batch = ExperienceBatch(transitions=transitions)
    train_metrics = agent.learn(batch)
    # Returns: {"loss": 0.12, "q_loss": 0.08, ...}
    
    # Logged to:
    logger.write_metrics(train_metrics)
    # → runs/run_id/metrics.jsonl
```

---

## 10. Reproducibility

### 10.1 Determinism Mechanisms

```python
# VerseSpec hash ensures reproducibility
spec_hash = _hash_verse_spec(verse_spec)
# SHA1( JSON.dumps(
#   spec_version, verse_name, verse_version, seed, tags, params
# ))

# Seeds propagate through system:
# 1. verse.seed(seed) - Verse RNG
# 2. agent.seed(seed) - Agent RNG
# 3. safe_executor.reset_episode(seed) - SafeExecutor RNG

# Episode-level seeding:
for ep in range(episodes):
    ep_seed = seed + ep  # Deterministic offset
    result = run_episode(..., seed=ep_seed, ...)
```

### 10.2 Run Identity

```python
run = RunRef.create()
# Generates: RunRef(run_id="run_abc123...", created_at_ms=1234567890)

# All events tagged with:
# - run_id (groups related episodes)
# - episode_id (unique per episode)
# - step_idx (deterministic position)
# - verse_name, spec_hash (environment identity)
# - seed (random state seed)
```

---

## 11. Performance Characteristics

### 11.1 Memory Query Budget

```python
# From rollout config:
on_demand_query_budget: int = 8           # Max queries per episode
on_demand_min_interval: int = 2           # Steps between queries

# In practice:
# Episode with 50 steps → max 8 memory lookups
# Each lookup: find_similar() with ANN/cosine search
# Cost: O(embedding_dim * num_rows) typically < 10ms
```

### 11.2 ADR Overhead

```python
# Automatic Domain Randomization:
# - Applied at spec creation time (1x per run)
# - Jitter: typically 10% of param value
# - Cost: ~1ms per VerseSpec
```

### 11.3 Event Streaming

```python
# Events written via on_step callback:
# - No holding in memory (streaming)
# - Each StepEvent ≈ 2-5 KB JSON
# - Append to JSONL: very fast (< 1ms per event)
# - 50 episodes × 50 steps = 2500 events/run
```

---

## 12. Debugging Tips

### 12.1 Trace Memory Queries

```bash
# Set environment variable:
export MULTIVERSE_ROLLOUT_VERBOSE=1

# Set memory specifics:
export MULTIVERSE_MEMORY_LOCK_TIMEOUT=30
export MULTIVERSE_MEMORY_QUERY_CACHE_SIZE=10000

# Then run:
python orchestrator/trainer.py --verse grid_world --episodes 1
```

### 12.2 Check Verse State Export

```python
# In GridWorldVerse:
def export_state(self) -> Dict[str, JSONValue]:
    return {
        "x": self._x,
        "y": self._y,
        "t": self._t,
        "obstacles": list(self._obstacles),
        "ice": list(self._ice),
        "teleporters": {...},
    }

def import_state(self, state: Dict[str, JSONValue]) -> None:
    self._x = state["x"]
    self._y = state["y"]
    self._t = state["t"]
    self._obstacles = set(state["obstacles"])
    self._ice = set(state["ice"])
    self._teleporters = state["teleporters"]

# SafeExecutor checkpoints these states for rollback
```

### 12.3 Inspect Curriculum State

```python
# Check current curriculum adjustments:
python -c "
from orchestrator.curriculum_controller import load_curriculum_adjustments
adj = load_curriculum_adjustments()
import json; print(json.dumps(adj, indent=2))
"
```

---

## Summary

The Multiverse orchestrator achieves deep verse integration through:

1. **Decoupling via Registry**: Trainer doesn't import specific verses
2. **Specification Pattern**: VerseSpec → VerseRef enables reproducibility
3. **Protocol Interface**: Verse.reset/step/export_state/import_state
4. **Memory Bridge**: Observations queried and enriched with cross-verse patterns
5. **Safety Wrapping**: SafeExecutor holds verse reference for checkpointing
6. **Event Streaming**: StepEvent logs all interactions for offline analysis
7. **Curriculum Feedback**: Update_signals drive adaptive difficulty per verse
8. **Cognitive Tagging**: Semantic memory organized by task taxonomy

This architecture enables:
- ✅ Plugin new verses without code changes
- ✅ Transfer learning across similar verses
- ✅ Safety verification through state rollback
- ✅ Memory-augmented training with cross-verse pattern recognition
- ✅ Curriculum learning adapted to per-verse progress
