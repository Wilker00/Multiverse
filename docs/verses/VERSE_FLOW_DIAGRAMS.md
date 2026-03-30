# Verse Integration Flow Diagrams

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MULTIVERSE ORCHESTRATOR                      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ EXPERIMENT LAYER                                               │ │
│  ├─ VerseSpec (configs, params, curriculum)                      │ │
│  ├─ AgentSpec (algorithm, hyperparams)                           │ │
│  └─ Training parameters (episodes, max_steps, seed)              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ TRAINER (orchestrator/trainer.py)                              │ │
│  ├─ Parse specs                                                   │ │
│  ├─ Create verse via registry                                     │ │
│  ├─ Create agent with verse's spaces                              │ │
│  ├─ Configure SafeExecutor + memory                               │ │
│  └─ Coordinate rollout + logging + indexing                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ REGISTRY LAYER (verses/registry.py)                            │ │
│  ├─ Load curriculum adjustments                                   │ │
│  ├─ Apply ADR (domain randomization)                              │ │
│  ├─ Inject cognitive tags                                         │ │
│  └─ Instantiate verse factory                                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ CORE EXECUTION LAYER                                           │ │
│  │                                                                  │ │
│  │  ┌────────────────────┐    ┌────────────────────┐              │ │
│  │  │  VERSE             │    │  AGENT             │              │ │
│  │  │  (Environment)      │    │  (Policy)          │              │ │
│  │  │                     │    │                    │              │ │
│  │  │  reset() → Obs      │    │  act(obs) → Action │              │ │
│  │  │  step(a) → Reward   │◄───┤  learn(batch)      │              │ │
│  │  │  export_state()     │    │  on_memory_response│              │ │
│  │  │  import_state()     │    │                    │              │ │
│  │  └────────────────────┘    └────────────────────┘              │ │
│  │                                      ▲                          │ │
│  │         ┌─────────────────────────────┴──────────┐             │ │
│  │         │                                        │             │ │
│  │  ┌────────────────────┐          ┌──────────────────────┐      │ │
│  │  │  SAFE EXECUTOR     │          │  MEMORY SYSTEM       │      │ │
│  │  │                     │          │                      │      │ │
│  │  │  select_action()   │◄─────────┤  find_similar()      │      │ │
│  │  │  post_step()       │          │  query w/ budget     │      │ │
│  │  │  checkpoint/rewind │          │  cross-verse matches │      │ │
│  │  └────────────────────┘          └──────────────────────┘      │ │
│  │                                                                  │ │
│  │  ROLLOUT (core/rollout.py)                                     │ │
│  │  run_episode() → run_episodes()                                │ │
│  │  └────────────────────────────────────────────────────────────│ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ LOGGING & INDEXING (memory/event_log.py, episode_index.py)    │ │
│  ├─ Stream StepEvents to JSONL                                   │ │
│  ├─ Index episodes for Knowledge Market                          │ │
│  └─ Ingest into central memory                                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Verse Creation & Registry Pipeline

```
VerseSpec
│
├─ verse_name: "grid_world"
├─ verse_version: "0.1"
├─ params: {width: 5, height: 5, adr_enabled: true}
└─ seed: 42
│
▼
[orchestrator/trainer.py: create_verse()]
│
▼ _hash_verse_spec() → spec_hash (for reproducibility)
│
▼ [verses/registry.py: create_verse()]
│
├─ Load curriculum adjustments
│  └─ IF plateau_detected: increase difficulty params
│
├─ Apply ADR (Automatic Domain Randomization)
│  └─ Jitter numeric params by ±10%
│     width:      5 * (1 + U[-0.1, 0.1])  ≈ 4.75–5.25
│     step_penalty: -0.01 * (1 + U[-0.1, 0.1])  ≈ -0.011–-0.009
│
├─ Merge cognitive tags from taxonomy
│  └─ VERSE_TAGS["grid_world"] = ["navigation", "2d", "discrete_grid"]
│
▼ Get factory from registry
│  _FACTORIES["grid_world"] = GridWorldFactory()
│
▼ factory.create(spec_eff)
│
├─ GridWorldVerse.__init__(spec)
│  ├─ self.spec = spec (with tags merged)
│  ├─ self.params = GridWorldParams(...)
│  ├─ self.observation_space = SpaceSpec(type="dict", ...)
│  ├─ self.action_space = SpaceSpec(type="discrete", n=4)
│  ├─ self._rng = Random(spec.seed)
│  └─ self._obstacles, self._ice, self._teleporters = {}
│
▼ Return Verse instance
│
VerseRef
├─ verse_id: "verse_abc123..." (unique runtime ID)
├─ verse_name: "grid_world"
├─ verse_version: "0.1"
└─ spec_hash: (sha1 of VerseSpec for reproducibility)
```

---

## 3. Episode Execution Loop

```
┌───────────────────────────────────────────────────────────┐
│ run_episode(verse, agent, config, seed)                   │
└───────────────────────────────────────────────────────────┘
│
├─ [INITIALIZATION]
│  ├─ verse.seed(seed) ──────────────────┐
│  └─ agent.seed(seed)                   │ Deterministic state
│     safe_executor.reset_episode(seed) ─┘
│
├─ [RESET]
│  │
│  ├─ reset_result = verse.reset()
│  │  └─ GridWorldVerse:
│  │     ├─ Place agent at (start_x, start_y)
│  │     ├─ Place goal at (goal_x, goal_y)
│  │     ├─ Generate obstacles
│  │     ├─ Generate ice patches
│  │     ├─ Generate teleporter pairs
│  │     └─ Return ResetResult(obs={...}, info={...})
│  │
│  └─ obs = reset_result.obs
│
│
├─ [EPISODE LOOP] while not done and step_idx < max_steps:
│  │
│  ├─ ════════════ MEMORY RETRIEVAL (RAR) ════════════
│  │  if step_idx % retrieval_interval == 0:
│  │    ├─ Try: EpisodeFilter(verse_name=verse_ref.verse_name, reached_goal=True)
│  │    │      → Exact verse match with goal achieved
│  │    ├─ Else: _strategy_signature(obs, verse_name)
│  │    │      → Cross-verse pattern matching
│  │    ├─ IF matches found:
│  │    │  └─ hint["successful_episode"] = match.episode_id
│  │    │
│  │    hint_retrieval_success: bool
│
│  │
│  ├─ ════════════ ON-DEMAND MEMORY QUERY ════════════
│  │  if on_demand_memory_enabled and agent.memory_query_request:
│  │    │
│  │    ├─ can_query = (queries_used < budget AND cooldown_elapsed)
│  │    │
│  │    ├─ IF can_query:
│  │    │  ├─ req = agent.memory_query_request(obs, step_idx)
│  │    │  │
│  │    │  ├─ IF req is dict:
│  │    │  │  ├─ matches = find_similar(
│  │    │  │  │    obs=req["query_obs"],
│  │    │  │  │    top_k=req.get("top_k", 3),
│  │    │  │  │    verse_name=req.get("verse_name"),
│  │    │  │  │    min_score=req.get("min_score", -1.0),
│  │    │  │  │    memory_families=req.get("memory_families"),
│  │    │  │  │  )
│  │    │  │  │
│  │    │  │  ├─ memory_queries_used += 1
│  │    │  │  ├─ last_memory_query_step = step_idx
│  │    │  │  │
│  │    │  │  ├─ bundle = _build_memory_bundle(req, matches)
│  │    │  │  │  └─ {"matches": [...], "query_timestamp": ...}
│  │    │  │  │
│  │    │  │  ├─ hint["memory_recall"] = bundle
│  │    │  │  │
│  │    │  │  └─ agent.on_memory_response(bundle)
│  │    │  │     └─ Agent processes matches, updates internal state
│  │    │  │
│  │    │  memory_query_success: bool
│
│  │
│  ├─ ════════════ ACTION SELECTION ════════════
│  │  if safe_executor:
│  │    └─ action_result = safe_executor.select_action(agent, obs)
│  │       ├─ Competence shield check
│  │       ├─ Confidence threshold assessment
│  │       └─ Possible fallback policy activation
│  │  else:
│  │    ├─ IF hint and agent.act_with_hint:
│  │    │  ├─ Apply recall ablation (for A/B testing)
│  │    │  │  P(disable_memory) = recall_ablation_prob
│  │    │  │
│  │    │  └─ action_result = agent.act_with_hint(obs, hint)
│  │    │     └─ MemoryRecallAgent:
│  │    │        ├─ Compute policy output
│  │    │        ├─ IF hint has top match:
│  │    │        │  └─ Blend policy + memory action
│  │    │        └─ Return weighted action
│  │    │
│  │    └─ ELSE: action_result = agent.act(obs)
│  │
│  │  action = action_result.action
│
│  │
│  ├─ ════════════ ENVIRONMENT STEP ════════════
│  │  step_result = verse.step(action)
│  │  │
│  │  ├─ GridWorldVerse.step(action):
│  │  │  ├─ Decode action: 0=up, 1=down, 2=left, 3=right
│  │  │  ├─ Compute new position: (x', y')
│  │  │  ├─ Check collisions: obstacles, teleporters, ice
│  │  │  ├─ Compute reward:
│  │  │  │  ├─ base_reward = step_penalty
│  │  │  │  ├─ IF at goal: reward += 1.0
│  │  │  │  ├─ IF obstacle hit: reward += obstacle_penalty
│  │  │  │  └─ ...other conditions...
│  │  │  ├─ Check done: reached_goal OR max_steps
│  │  │  ├─ Generate new obs (x, y, t, obstacles, ice, teleporter)
│  │  │  └─ Return StepResult(obs, reward, done, truncated, info)
│  │  │
│  │  └─ obs = step_result.obs
│  │
│  │  IF safe_executor:
│  │    └─ step_result = safe_executor.post_step(...)
│  │       ├─ May modify reward/obs based on safety state
│  │       ├─ May trigger checkpoint recovery
│  │       └─ Return possibly-modified StepResult
│
│  │
│  ├─ ════════════ EVENT RECORDING ════════════
│  │  event = make_step_event(
│  │    schema_version, run_id, episode_id, step_idx,
│  │    agent_ref, verse_ref,
│  │    obs, action, reward, done, truncated,
│  │    info={
│  │      "verse_params": {...},
│  │      "memory_query": memory_query_state,
│  │      "memory_recall_ablation": ablation_state,
│  │      "action_info": action_result.info,
│  │      "runtime_errors": {...},
│  │    }
│  │  )
│  │
│  │  on_step(event)  ← Write to JSONL immediately
│  │
│  │  IF collect_transitions:
│  │    └─ transitions.append(Transition(...))
│
│  │
│  ├─ ════════════ STATE UPDATE ════════════
│  │  return_sum += step_result.reward
│  │  done = step_result.done or step_result.truncated
│  │  step_idx += 1
│
│  └─ [END LOOP]
│
│
├─ [LEARNING] (if train=True and collect_transitions=True)
│  │
│  ├─ batch = ExperienceBatch(transitions=[...])
│  │
│  ├─ train_metrics = agent.learn(batch)
│  │  └─ MemoryRecallAgent.learn():
│  │     ├─ Compute policy loss on batch
│  │     ├─ Update policy network weights
│  │     └─ Return {"loss": ..., "q_loss": ..., ...}
│  │
│  └─ logger.write_metrics(train_metrics)
│
│
└─ [RETURN]
   └─ RolloutResult(
      events=[...],
      episode_id=episode_id,
      steps=step_idx,
      return_sum=return_sum,
      train_metrics=train_metrics,
   )
```

---

## 4. Memory Query & Retrieval Pipeline

```
┌──────────────────────────────────────────────────────────┐
│ Agent requests memory for current obs                     │
├──────────────────────────────────────────────────────────┤
│ req = agent.memory_query_request(obs, step_idx)          │
│ {                                                         │
│   "reason": "uncertainty",                               │
│   "query_obs": obs,                                       │
│   "top_k": 3,                                             │
│   "verse_name": None,  ← Cross-verse if None              │
│   "min_score": 0.7,                                       │
│   "memory_families": {"skill_transfer", ...},             │
│ }                                                         │
└──────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ find_similar(obs, cfg, top_k, verse_name, min_score)     │
│ [memory/central_repository.py]                           │
└──────────────────────────────────────────────────────────┘
│
├─ [ENCODE OBSERVATION]
│  ├─ obs_vector = encode_obs(obs) → embedding ∈ ℝ^d
│  └─ Example: obs = {"x": [1], "y": [0], ...}
│             → [0.12, 0.45, 0.33, ..., 0.78]
│
├─ [BUILD SIMILARITY CACHE]
│  ├─ _get_similarity_cache_for_path(mem_path)
│  ├─ Load from central_memory/stm_memories.jsonl
│  │  + central_memory/ltm_memories.jsonl (if enabled)
│  ├─ Embed rows if not cached
│  │  └─ Use ANN (FAISS) if enabled, else dense matrix
│  │
│  └─ cache = {
│       "rows": [
│         {
│           "episode_id": "episode_99",
│           "step_idx": 23,
│           "verse_name": "grid_world",
│           "obs": {...},
│           "action": 1,  ← "right"
│           "reward": 0.98,
│           "reached_goal": True,
│           "obs_vector": [0.15, 0.42, ...],  ← Embedding
│           "family": "skill_transfer",
│           "score": nil,  ← Computed on query
│         },
│         ...
│       ],
│     }
│
├─ [FILTER CANDIDATES]
│  └─ candidates = []
│     for row in cache.rows:
│       if verse_name and row["verse_name"] != verse_name:
│         skip  ← Filter by exact verse match
│       if memory_families and row["family"] ∉ memory_families:
│         skip  ← Filter by family
│       candidates.append(row)
│
├─ [COMPUTE SIMILARITY]
│  └─ scores = cosine_similarity(
│       [obs_vector],
│       [row["obs_vector"] for row in candidates],
│     )[0]
│     └─ ∈ [-1, 1], higher = more similar
│        scores[i] = obs_vector · candidates[i].obs_vector
│                    ─────────────────────────────────────
│                    ||obs_vector|| × ||candidates[i]||
│
├─ [RANK & FILTER]
│  └─ ranked = sorted(
│       zip(candidates, scores),
│       key=lambda x: x[1],
│       reverse=True,
│     )
│     ← Top-k highest similarity scores
│
│     keep only rows where score >= min_score
│
└─ [RETURN MATCHES]
   └─ [
      ScenarioMatch(
        episode_id="episode_99",
        step_idx=23,
        obs={...},
        action=1,
        reward=0.98,
        verse_name="grid_world",
        score=0.92,
        trajectory_window=0,
      ),
      ScenarioMatch(...),  ← 2nd best match
      ScenarioMatch(...),  ← 3rd best match
      ]
```

---

## 5. SafeExecutor Checkpoint/Rewind Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ SafeExecutor wraps Verse for safety                      │
├─────────────────────────────────────────────────────────┤
│ self.verse = verse                                       │
│ self._checkpoint_state = None                            │
│ self._checkpoint_obs = None                              │
│ self._checkpoint_agent_state = None                      │
└─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ NORMAL EXECUTION FLOW                                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [SELECT_ACTION]                                        │
│  │                                                      │
│  ├─ Competence check: current_agent_confidence > τ     │
│  │  ├─ YES → SAFE ACTION                               │
│  │  │       └─ action = agent.act(obs)                 │
│  │  │           save_checkpoint(obs)  ← Save state     │
│  │  │
│  │  └─ NO → VETO (blocked)                             │
│  │         ├─ action = fallback_agent.act(obs)         │
│  │         └─ Flag: action.info["veto"] = True         │
│  │
│  ├─ [ENVIRONMENT STEP]                                 │
│  │  ├─ step_result = verse.step(action)  ← Modify env  │
│  │  └─ obs = step_result.obs                           │
│  │
│  ├─ [POST_STEP SAFETY]                                 │
│  │  ├─ IF outcome looks dangerous:                     │
│  │  │  ├─ Estimate risk: risk_score = predict_danger() │
│  │  │  │
│  │  │  ├─ IF risk_score > danger_threshold:            │
│  │  │  │  ├─ Rewind to checkpoint:                     │
│  │  │  │  │  └─ verse.import_state(checkpoint_state)   │
│  │  │  │  │     obs = checkpoint_obs                   │
│  │  │  │  │
│  │  │  │  ├─ Modify reward:                            │
│  │  │  │  │  └─ reward = -100 (penalty for unsafe act) │
│  │  │  │  │
│  │  │  │  └─ Mark episode for attention                │
│  │  │  │
│  │  │  └─ ELSE: Continue normally                      │
│  │  │
│  │  └─ Return possibly-modified StepResult             │
│
│  └─ [GO TO NEXT STEP]
│
└─────────────────────────────────────────────────────────┘

Checkpoint/Rewind API Requirements:
┌─────────────────────────────────────────────────────────┐
│ Verse MUST implement:                                   │
│                                                         │
│ def export_state() → Dict[str, JSONValue]:              │
│   """Return full internal state for checkpoint."""      │
│   return {                                              │
│     "x": self._x,                                       │
│     "y": self._y,                                       │
│     "t": self._t,                                       │
│     "obstacles": list(self._obstacles),                 │
│     "ice": list(self._ice),                             │
│     "teleporters": self._teleporters,                   │
│   }                                                     │
│                                                         │
│ def import_state(state: Dict[str, JSONValue]) → None:  │
│   """Restore internal state from checkpoint."""         │
│   self._x = state["x"]                                  │
│   self._y = state["y"]                                  │
│   self._t = state["t"]                                  │
│   self._obstacles = set(state["obstacles"])             │
│   self._ice = set(state["ice"])                         │
│   self._teleporters = state["teleporters"]              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Curriculum Learning Feedback Loop

```
┌──────────────────────────────────────────────────────────┐
│ TRAINING RUN                                             │
│ Episodes 1-N on verse="grid_world"                       │
└──────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ COMPUTE AGGREGATE SIGNAL                                 │
│                                                          │
│  success_rate = (episodes_with_goal / total_episodes)  │
│               = 7 / 10 = 0.70                           │
│                                                          │
│  mean_return = (sum_of_returns / total_episodes)       │
│              = 45.3 / 10 = 4.53                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ UPDATE CURRICULUM STATE                                  │
│ [orchestrator/curriculum_controller.py]                 │
│                                                          │
│  curriculum.update_from_signal(                          │
│    verse_name="grid_world",                              │
│    success_rate=0.70,                                    │
│    mean_return=4.53,                                     │
│  )                                                       │
└──────────────────────────────────────────────────────────┘
│
├─ [LOAD CURRENT STATE]
│  └─ models/curriculum_adjustments.json
│     └─ grid_world: {
│          "history": [...],
│          "noise": 0.05,
│          "stochasticity": 0.10,
│          "partial_observability": 0.15,
│          "distractors": 2,
│          "mode": "stable",
│        }
│
├─ [APPEND TO HISTORY]
│  └─ history += {
│       "t_ms": now_ms(),
│       "success_rate": 0.70,
│       "mean_return": 4.53,
│     }
│
├─ [DETECT PLATEAU]
│  └─ plateau = is_plateau(history[-plateau_window:])
│     └─ Low variance + low trend over window
│        └─ If success_rate stable high → PLATEAU = TRUE
│
├─ [DECIDE ADJUSTMENT]
│  │
│  ├─ IF plateau:
│  │  └─ Increase difficulty
│  │     ├─ noise += 0.05
│  │     ├─ stochasticity += 0.05
│  │     ├─ partial_observability += 0.025
│  │     ├─ distractors += 1
│  │     └─ mode = "plateau_harder"
│  │
│  ├─ ELIF success_rate < collapse_threshold:
│  │  └─ Decrease difficulty (agent failing)
│  │     ├─ noise -= 0.075
│  │     ├─ stochasticity -= 0.075
│  │     ├─ partial_observability -= 0.05
│  │     ├─ distractors -= 1
│  │     └─ mode = "collapse_backoff"
│  │
│  └─ ELSE:
│     └─ Maintain current difficulty
│        └─ mode = "stable"
│
└─ [APPLY TO NEXT RUN]
   └─ Next VerseSpec for grid_world:
      ├─ apply_curriculum_params()
      └─ params = {
           "action_noise": 0.02 + 0.05,  ← curriculum noise applied
           "adr_jitter": 0.10,
           "width": 5,
           ...
         }
```

---

## 7. Event Logging & Ingestion

```
┌────────────────────────────────────────────────────────┐
│ STEP EVENT GENERATION                                   │
│ Each step_idx creates a StepEvent                       │
└────────────────────────────────────────────────────────┘
│
│  StepEvent(
│    schema_version="v1",
│    run_id="run_xyz789",
│    t_ms=1234567890,
│    episode_id="episode_e1",
│    step_idx=5,
│    agent_id="agent_abc",
│    policy_id="memory_recall:v1",
│    verse_id="verse_123",
│    verse_name="grid_world",
│    verse_version="0.1",
│    spec_hash="abc123...",
│    obs={x: [1], y: [0], ...},
│    action=3,
│    reward=-0.01,
│    done=False,
│    truncated=False,
│    seed=42,
│    info={
│      "verse_params": {...} if step_idx == 0 else None,
│      "memory_query": {...},
│      "memory_recall_ablation": {...},
│      "action_info": {...},
│      "transfer_decision_records": [...],
│      "runtime_errors": {...},
│    }
│  )
│
│  ▼
│  on_step(event)  ← Stream callback
│  │
│  ▼
│  logger.write_event(event)
│  │
│  ▼
│  File: runs/run_xyz789/events.jsonl
│  {line 1}  {episode_id: "e1", step_idx: 0, obs: {...}, ...}
│  {line 2}  {episode_id: "e1", step_idx: 1, obs: {...}, ...}
│  ...
│  {line 50} {episode_id: "e1", step_idx: 49, obs: {...}, ...}
│  {line 51} {episode_id: "e2", step_idx: 0, obs: {...}, ...}  ← Next episode
│
└────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│ AUTO-INDEXING FOR KNOWLEDGE MARKET                       │
│ [memory/episode_index.py: build_episode_index()]        │
└────────────────────────────────────────────────────────┘
│
├─ Read runs/run_xyz789/events.jsonl
│
├─ Extract per-episode metadata:
│  ├─ episode_id
│  ├─ steps
│  ├─ return_sum
│  ├─ reached_goal
│  ├─ verse_name
│  ├─ cognitive_tags
│  └─ success_signature
│
├─ Create runs/run_xyz789/episodes.index.jsonl
│  └─ One JSON object per episode for fast lookup
│
└─ Episodes now findable in Knowledge Market
   ├─ By verse_name
   ├─ By cognitive tags
   ├─ By success criteria
   └─ By spec_hash (reproducible configuration)
│
▼
┌────────────────────────────────────────────────────────┐
│ POST-TRAINING: MEMORY INGEST                            │
│ [memory/central_repository.py: ingest_run()]           │
└────────────────────────────────────────────────────────┘
│
├─ For each event in events.jsonl:
│  ├─ Extract (obs, action, reward, verse_name, reached_goal)
│  ├─ Compute embedding: obs_vector = embed(obs)
│  ├─ Check deduplication:
│  │  └─ dedupe_key = hash(obs, action)
│  │     IF key ∈ dedupe_index: SKIP
│  │     ELSE: reserve key
│  ├─ Append to central_memory/stm_memories.jsonl (short-term)
│  │  └─ {"episode_id", "step_idx", "obs", "action", "reward",
│  │     "verse_name", "obs_vector", "reached_goal", ...}
│  └─ Update cache if ANN enabled
│
├─ Periodically move successful episodes to LTM:
│  └─ central_memory/ltm_memories.jsonl (long-term)
│     └─ Permanent knowledge store
│
└─ Similarity cache updated
   └─ central_memory/memories.jsonl.simcache.json
      └─ Pre-computed embeddings for fast lookup
```

---

## 8. Complete Training Run Sequence Diagram

```
Timeline: User invokes trainer.run()

t=0
│
├─ Trainer._load config
├─ Trainer.run([spec, agent_spec, episodes, max_steps])
│
├─ Register builtin verses
├─ Register builtin agents
│
├─ Validate VerseSpec + AgentSpec
│
t=100ms
├─ Create RunRef(run_id="run_a1b2c3d4")
│
├─ Hash VerseSpec → spec_hash="abc123xyz"
│
├─ Create VerseRef(verse_id="v1", spec_hash="abc123xyz")
│
├─ create_verse(spec)
│  ├─ Load curriculum adjustments from models/
│  ├─ Apply ADR with spec.seed
│  ├─ Inject cognitive tags
│  └─ instantiate GridWorldVerse
│
├─ extract verse.observation_space, verse.action_space
│
├─ Create Agent([observation_space], [action_space])
│  └─ MemoryRecallAgent initialized with spaces
│
t=200ms
├─ Optional: Create SafeExecutor (wraps verse)
│
├─ Create RolloutConfig
│  └─ memory_enabled=True, query_budget=8, retrieval_interval=10
│
├─ Create EventLogger (writes to runs/run_a1b2c3d4/)
│
├─ Create RetrievalClient (accesses past episodes)
│
├─ START EPISODE LOOP: for ep in range(episodes)
│  │
│  ├─ [EPISODE 0]
│  │  ├─ run_episode(verse, agent, config)
│  │  │
│  │  ├─ verse.seed(seed=42)
│  │  ├─ agent.seed(seed=42)
│  │  │
│  │  ├─ reset_result = verse.reset()
│  │  │
│  │  ├─ START STEP LOOP: while not done and step_idx < 50
│  │  │  │
│  │  │  ├─ [STEP 0]
│  │  │  │  ├─ obs = {x: [0], y: [0], ...}
│  │  │  │  ├─ hint = None
│  │  │  │  ├─ action_result = agent.act(obs)
│  │  │  │  ├─ action = 3  (e.g., "right")
│  │  │  │  ├─ step_result = verse.step(3)
│  │  │  │  ├─ event = make_step_event(...)
│  │  │  │  ├─ on_step(event) → write to JSONL
│  │  │  │  └─ obs, return_sum, step_idx += 1
│  │  │  │
│  │  │  ├─ [STEP 5]
│  │  │  │  ├─ 5 % 10 != 0, so skip RAR
│  │  │  │  ├─ Check on-demand memory:
│  │  │  │  │  ├─ req = agent.memory_query_request(obs)
│  │  │  │  │  ├─ matches = find_similar(obs)
│  │  │  │  │  ├─ hint["memory_recall"] = bundle
│  │  │  │  │  └─ agent.on_memory_response(bundle)
│  │  │  │  ├─ action_result = agent.act_with_hint(obs, hint)
│  │  │  │  └─ Continue as normal...
│  │  │  │
│  │  │  ├─ [STEP 10]
│  │  │  │  ├─ 10 % 10 == 0, trigger RAR
│  │  │  │  ├─ Try verse-specific match
│  │  │  │  ├─ Fallback to strategic signature
│  │  │  │  ├─ hint["successful_episode"] = match
│  │  │  │  └─ Continue...
│  │  │  │
│  │  │  ├─ [STEP 27]
│  │  │  │  ├─ Agent reaches goal (done=True)
│  │  │  │  └─ EXIT STEP LOOP
│  │  │  │
│  │  │  └─ END STEP LOOP: 27 steps executed
│  │  │
│  │  ├─ Learn batch (if configured)
│  │  │  └─ train_metrics = agent.learn(batch)
│  │  │
│  │  └─ Return RolloutResult(
│  │       episodes_id="episode_e0",
│  │       steps=27,
│  │       return_sum=0.73,
│  │       train_metrics={...},
│  │     )
│  │
│  ├─ [EPISODE 1-9]
│  │  ├─ Similar process, can now access Episode 0 memory
│  │  ├─ find_similar queries return episodes from Episode 0
│  │  ├─ Possible faster learning via memory-guided actions
│  │  └─ ...
│  │
│  └─ END EPISODE LOOP: 10 episodes completed
│
t=5s (typical for 10 episodes × 50 steps)
│
├─ Collect all RolloutResults
│
├─ Log aggregate metrics
│  └─ logger.write_metrics({...})
│
├─ Auto-index for Knowledge Market
│  └─ build_episode_index(run_dir)
│
├─ Close verse, agent, safe_executor
│
├─ Print summary:
│  └─ "Run complete. run_id=run_a1b2c3d4, total_steps=2350, total_return=45.3"
│
└─ Return {"run_id": "...", "total_return": 45.3, "total_steps": 2350}


After Training:
│
├─ Events stored in: runs/run_a1b2c3d4/events.jsonl (250 lines)
├─ Episodes indexed in: runs/run_a1b2c3d4/episodes.index.jsonl (10 lines)
├─ Memory ingestion (optional):
│  └─ Central memory updated with successful episodes
│     └─ Can be reused by future runs
│
└─ Next run can now query from this run's memory
```

---

## 9. Cross-Verse Transfer Pattern

```
┌──────────────────────────────────────────────────────────┐
│ SETUP: Two similar verses with transferable structure    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  chess_world (turn-based strategy)                       │
│  ├─ obs_space: {"board": [...], "turn": int, ...}        │
│  ├─ tags: ["strategy_games", "board_control", ...]      │
│  └─ strategic_signature: "high_pressure_position"       │
│                                                          │
│  go_world (turn-based strategy)                          │
│  ├─ obs_space: {"board": [...], "liberties": [...], ...} │
│  ├─ tags: ["strategy_games", "board_control", ...]      │
│  └─ strategic_signature: "high_pressure_position"       │
│                                                          │
└──────────────────────────────────────────────────────────┘
│
│ Both share cognitive tags + similar strategic signatures
│ ↓
│
├─ [RUN 1: TRAIN ON CHESS]
│  ├─ Create episodes: run_chess_001 (10 episodes)
│  ├─ Episodes with reached_goal=True stored in central_memory
│  │  └─ Each episode tagged: verse_name="chess_world"
│  │     cognitive_tags=["strategy_games", "board_control", ...]
│  │     strategic_signature="high_pressure_position"
│  │
│  └─ Episodes indexed
│
│ ↓
│
├─ [RUN 2: TRAIN ON GO]
│  ├─ Create verse: verse_name="go_world"
│  ├─ Start episode: obs = {"board": [...], "liberties": [...]}
│  ├─ Step 5: Memory query
│  │  ├─ Exact match: EpisodeFilter(verse_name="go_world", ...)
│  │  │             → No previous go_world episodes
│  │  │
│  │  ├─ Strategic bridge match:
│  │  │  ├─ sig = _strategy_signature(obs, "go_world")
│  │  │  ├─ sig = "high_pressure_position" ← MATCH!
│  │  │  │
│  │  │  ├─ EpisodeFilter(strategic_match="high_pressure_position", ...)
│  │  │  │
│  │  │  └─ FOUND: chess_world episodes with same signature!
│  │  │
│  │  ├─ matches = [
│  │  │   ScenarioMatch(
│  │  │     episode_id="chess_episode_7",
│  │  │     step_idx=42,
│  │  │     action="move_queen_to_h4",  ← Strategic pattern
│  │  │     reward=1.5,
│  │  │     verse_name="chess_world" ← CROSS-VERSE!
│  │  │     score=0.87,
│  │  │   ),
│  │  │   ...
│  │  │ ]
│  │  │
│  │  ├─ bundle = _build_memory_bundle(matches)
│  │  │
│  │  ├─ agent.on_memory_response(bundle)
│  │  │  └─ Agent recognizes parallel structure:
│  │  │     "This go position resembles a chess endgame I've seen"
│  │  │
│  │  └─ action = agent.act_with_hint(obs, hint)
│  │     └─ Action influenced by chess strategy!
│  │
│  └─ Continued training now informed by chess knowledge
│
│ ↓
│
├─ [RESULT]
│  └─ Go agent learns faster than from scratch
│     └─ Transfer learning via strategic signature match
│        ├─ Chess successful patterns → Go
│        ├─ No explicit model transfer needed
│        └─ Purely memory-based transfer
│
└──────────────────────────────────────────────────────────┘
```

---

## 10. System Invariants & Guarantees

```
VERSE INTERFACE CONTRACT:
┌─────────────────────────────────────────────────────────┐
│ class Verse(Protocol):                                  │
│                                                         │
│   spec: VerseSpec  ← Must have configuration             │
│                                                         │
│   observation_space: SpaceSpec  ← Reproducible schema   │
│   action_space: SpaceSpec       ← Reproducible schema   │
│                                                         │
│   def seed(seed: int) → None                            │
│     """Ensure deterministic behavior."""                │
│                                                         │
│   def reset() → ResetResult                             │
│     """Return (obs, info), mark start of episode."""     │
│                                                         │
│   def step(action) → StepResult                         │
│     """Return (obs, reward, done, truncated, info)."""   │
│                                                         │
│   def export_state() → Dict[str, JSONValue]  [OPTIONAL] │
│     """Checkpoint for safety layer rollback."""         │
│                                                         │
│   def import_state(state) → None  [OPTIONAL]            │
│     """Restore from checkpoint."""                      │
│                                                         │
│   def render(mode="rgb_array") → Optional[Array]        │
│     """Visualization support."""                        │
│                                                         │
│   def close() → None                                    │
│     """Cleanup resources."""                            │
│                                                         │
└─────────────────────────────────────────────────────────┘

STEP EVENT INVARIANTS:
┌─────────────────────────────────────────────────────────┐
│ ✓ Every StepEvent must have:                             │
│   - unique (run_id, episode_id, step_idx) triple        │
│   - verse_name, spec_hash matching VerseRef             │
│   - obs, action, reward all JSON-serializable            │
│   - done ∈ {True, False}                                │
│                                                         │
│ ✓ First StepEvent of episode has:                       │
│   - step_idx = 0                                        │
│   - obs = result of verse.reset()                       │
│                                                         │
│ ✓ Last StepEvent of episode has:                        │
│   - done = True (or step_idx == max_steps)              │
│                                                         │
│ ✓ Sequential StepEvents satisfy:                        │
│   - obs[i+1] comes from step_result[i].obs              │
│   - reward[i] matches verse.step(action[i]).reward      │
│                                                         │
│ ✓ Memory queries preserve:                              │
│   - ScenarioMatch.action matches original action        │
│   - ScenarioMatch.obs is from central_memory            │
│   - Score ∈ [0, 1] (cosine similarity)                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

REPRODUCIBILITY GUARANTEES:
┌─────────────────────────────────────────────────────────┐
│ Given identical:                                        │
│   - VerseSpec (params, seed, etc.)                       │
│   - AgentSpec (weights, seed, etc.)                      │
│                                                         │
│ Expected outcome:                                       │
│   - Identical VerseRef (same spec_hash)                 │
│   - Identical stream of observations from verse.reset() │
│   - Identical StepEvents (given deterministic agent)     │
│   - Identical cumulative return                         │
│                                                         │
│ Caveats:                                                │
│   - ADR adds randomness (unless adr_enabled=False)      │
│   - Memory queries depend on central_memory state       │
│   - Curriculum adjustments modify params dynamically    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
