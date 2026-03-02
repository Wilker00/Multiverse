# Multiverse Quickstart Guide

Learn Multiverse in 15 minutes with hands-on examples.

**Goal:** Train your first RL agent, understand the results, and explore advanced features.

---

## Prerequisites

- ✅ Multiverse installed (see [SETUP.md](SETUP.md))
- ✅ Virtual environment activated (`.venv`)
- ✅ 15 minutes of your time

---

## Part 1: Your First Agent (5 minutes)

### Step 1: Train a Random Agent

Let's start simple - train a random policy in the `line_world` environment:

```bash
python tools/train_agent.py \
  --algo random \
  --verse line_world \
  --episodes 20 \
  --max_steps 40
```

**What happened:**
- Trained a random agent for 20 episodes
- Each episode had max 40 steps
- Results saved to `runs/` directory

**Expected output:**
```
Starting training: random in line_world
Episode 1/20: return=5.0, steps=12
Episode 2/20: return=3.0, steps=15
...
Episode 20/20: return=4.5, steps=14
Training complete! Run ID: run_YYYYMMDD_HHMMSS
```

### Step 2: Inspect Your Results

```bash
# List all runs
python tools/multiverse_cli.py runs list

# See latest run
python tools/multiverse_cli.py runs latest

# Count events in your run
python tools/multiverse_cli.py runs inspect --count-events
```

**Your run directory contains:**
- `events.jsonl` - Every state transition
- `episodes.jsonl` - Episode summaries
- `config.json` - Training configuration
- `metrics.json` - Performance metrics

---

## Part 2: Train a Real RL Agent (5 minutes)

Random policies are boring. Let's train a Q-learning agent!

### Step 1: Train Q-Learning

```bash
python tools/train_agent.py \
  --algo q \
  --verse grid_world \
  --episodes 100 \
  --max_steps 50
```

**What's different:**
- `q` algorithm learns from experience
- `grid_world` is more complex (2D navigation)
- More episodes to learn the task

**Watch it learn:**
- Early episodes: Low returns, random behavior
- Later episodes: Higher returns, efficient paths

### Step 2: Compare Performance

```bash
# Compare your random vs Q-learning runs
python tools/compare_runs.py <run_id_1> <run_id_2>
```

**What to look for:**
- **Mean return:** Q-learning should be higher
- **Steps per episode:** Q-learning should be fewer (more efficient)
- **Success rate:** Q-learning should improve over time

---

## Part 3: Add Memory & Safety (5 minutes)

Now let's add Multiverse's killer features: **memory retrieval** and **safety control**.

### Step 1: Train with Memory

```bash
python tools/train_agent.py \
  --algo memory_recall \
  --verse cliff_world \
  --episodes 50 \
  --max_steps 100 \
  --memory_tiers ltm stm
```

**What's happening:**
- Agent retrieves similar past experiences
- Uses memory to guide decisions
- Learns faster than tabula rasa

### Step 2: Train with Safety

```bash
python tools/train_agent.py \
  --algo q \
  --verse cliff_world \
  --episodes 50 \
  --enable_safe_executor \
  --safe_executor_confidence_threshold 0.3
```

**Safety features:**
- Blocks dangerous actions in real-time
- Falls back to safe policy when uncertain
- Logs all safety interventions

### Step 3: Inspect Safety Events

```bash
# Check how many times safety kicked in
python tools/diagnose_run.py <run_id> --show-safety-events
```

**Expected:** Fewer cliff falls compared to unsafe training!

---

## Part 4: Transfer Learning (Bonus)

Multiverse's superpower: **train once, transfer everywhere**.

### Step 1: Train Source Agent

```bash
# Train in simple environment
python tools/train_agent.py \
  --algo q \
  --verse line_world \
  --episodes 100 \
  --max_steps 40
```

### Step 2: Transfer to Complex Environment

```bash
# Use memory from line_world to help in grid_world
python tools/train_agent.py \
  --algo memory_recall \
  --verse grid_world \
  --episodes 50 \
  --memory_tiers ltm \
  --verse_name line_world  # Retrieve from this verse
```

**Result:** Faster learning in `grid_world` by reusing `line_world` knowledge!

---

## Cheat Sheet: Common Commands

### Training

```bash
# Random baseline
python tools/train_agent.py --algo random --verse <verse> --episodes 20

# Q-learning
python tools/train_agent.py --algo q --verse <verse> --episodes 100

# Deep Q-Network (DQN)
python tools/train_agent.py --algo dqn --verse <verse> --episodes 200

# Memory-augmented
python tools/train_agent.py --algo memory_recall --verse <verse> --episodes 50
```

### Run Management

```bash
# List runs
multiverse runs list

# Latest run
multiverse runs latest

# Inspect run
multiverse runs inspect --run-id <id>

# Tail logs
multiverse runs tail --run-id <id> --file events.jsonl --lines 50
```

### Benchmarking

```bash
# Run benchmark suite
python tools/run_benchmark.py --suite quick

# Paper-readiness check
python tools/run_paper_readiness_pack.py
```

---

## Understanding Environments ("Verses")

Multiverse includes 24+ built-in environments. Here are the main categories:

### **Navigation (Start here!)**
- `line_world` - 1D movement (simplest)
- `grid_world` - 2D grid navigation
- `maze_world` - Complex mazes
- `cliff_world` - Navigation with hazards ⚠️

### **Games**
- `chess_world` - Chess positions
- `go_world` - Go positions
- `uno_world` - Card game

### **Resource Management**
- `warehouse_world` - Logistics optimization
- `factory_world` - Production scheduling
- `harvest_world` - Resource gathering

### **Challenge**
- `memory_vault_world` - Memory test
- `rule_flip_world` - Adapting to rule changes
- `long_horizon_challenge` - Long-term planning

**List all:**
```bash
python -c "from verses.registry import list_builtin; print('\n'.join(list_builtin()))"
```

---

## Understanding Agents

### **Classic RL**
- `random` - Random policy (baseline)
- `q` - Q-learning (tabular)
- `dqn` - Deep Q-Network
- `ppo` - Proximal Policy Optimization

### **Memory-Augmented**
- `memory_recall` - Retrieves similar past experiences
- `planner_recall` - Combines memory + planning

### **Advanced**
- `adt` - Adaptive Decision Transformer
- `mpc` - Model Predictive Control
- `adaptive_moe` - Mixture of Experts

**List all:**
```bash
python -c "from agents.registry import list_agents; print('\n'.join(list_agents()))"
```

---

## Pro Tips

### 🚀 **Speed Up Training**

```bash
# Use fewer episodes for testing
--episodes 10

# Reduce max steps
--max_steps 20

# Skip detailed logging
--log_level ERROR
```

### 🎯 **Improve Performance**

```bash
# Enable GPU (if available)
# PyTorch will auto-detect

# Use more episodes
--episodes 500

# Tune learning rate (Q-learning)
--learning_rate 0.1

# Tune epsilon (exploration)
--epsilon 0.2
```

### 🔍 **Debugging**

```bash
# Verbose output
--verbose

# Save intermediate checkpoints
--checkpoint_every 50

# Enable safety executor logging
MULTIVERSE_SAFE_EXECUTOR_VERBOSE=1 python tools/train_agent.py ...
```

---

## Common Patterns

### Pattern 1: Quick Experiment

```bash
# Fast iteration loop
for algo in random q dqn; do
  python tools/train_agent.py --algo $algo --verse line_world --episodes 20
done
```

### Pattern 2: Curriculum Learning

```bash
# Train on easy -> medium -> hard
python tools/train_agent.py --algo q --verse line_world --episodes 100
python tools/train_agent.py --algo memory_recall --verse grid_world --episodes 100
python tools/train_agent.py --algo memory_recall --verse cliff_world --episodes 100
```

### Pattern 3: Benchmark Suite

```bash
# Compare algorithms across environments
python tools/run_benchmark.py \
  --algos q dqn memory_recall \
  --verses line_world grid_world cliff_world \
  --episodes 50
```

---

## What's Next?

You now know the basics! Here's where to go deeper:

### 📚 **Learn More**
- **[Project Introduction](PROJECT_INTRO.md)** - Architecture overview
- **[Technical Paper](PAPER.md)** - Algorithms and theory
- **[Configuration Guide](CONFIGURATION.md)** - Tune everything

### 🔧 **Advanced Features**
- **Distributed Training:** `tools/train_distributed.py`
- **Scaling Stack:** Docker Compose with Kafka/Redis/Grafana
- **Custom Environments:** Write your own verse
- **Custom Agents:** Implement new algorithms

### 🎯 **Real Projects**
- **Safe RL Research:** Use `SafeExecutor` for runtime safety
- **Transfer Learning:** Cross-verse knowledge transfer
- **Production Deployment:** `tools/deploy_agent.py`

### 🤝 **Contribute**
- **[Contributing Guide](../CONTRIBUTING.md)** - Make Multiverse better
- **[Open Issues](https://github.com/wilker00/multiverse/issues)** - Pick a task
- **Share Your Work:** Post results in Discussions

---

## Troubleshooting

### "Training is slow"

- Start with fewer episodes/steps
- Use simpler environments (line_world, grid_world)
- Enable GPU for deep learning agents (DQN, PPO)

### "Agent not learning"

- Try more episodes (100+)
- Check learning rate (try 0.1, 0.01)
- Verify reward signal (`events.jsonl`)
- Compare to random baseline

### "Out of memory"

- Reduce batch size
- Use fewer parallel workers
- Clear old runs: `rm -rf runs/run_*`

### "Tests failing"

- See [SETUP.md#troubleshooting](SETUP.md#troubleshooting)

---

## Summary

In 15 minutes, you learned to:

✅ Train agents (random, Q-learning, DQN)
✅ Inspect results and compare runs
✅ Use memory retrieval for faster learning
✅ Enable safety controls for hazard avoidance
✅ Transfer knowledge between environments

**You're now ready to build real RL systems with Multiverse!** 🎉

---

**Questions?** See [FAQ](FAQ.md) or [open an issue](https://github.com/wilker00/multiverse/issues).
