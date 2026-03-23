# Tutorial 01: What is an Agent?

**Time**: 15 minutes
**Difficulty**: Beginner
**Prerequisites**: None

---

## 🎯 **Learning Objectives**

By the end of this tutorial, you'll understand:
- What makes an "agent" in reinforcement learning
- The basic structure of RL problems
- How agents learn through trial and error
- Why some agents perform better than others

---

## 📖 **Theory: The Agent-Environment Loop**

### **What is Reinforcement Learning?**

Reinforcement Learning (RL) is learning by **doing**. An agent interacts with an environment, makes decisions, and learns from the consequences.

**Key Players:**
- **Agent**: The learner (your AI)
- **Environment**: The world the agent interacts with
- **Actions**: What the agent can do
- **Observations**: What the agent can see
- **Rewards**: Feedback on how well the agent is doing

### **The RL Loop**

```
┌─────────────┐     ┌─────────────┐
│ Environment │────▶│   Agent    │
│             │     │             │
│  State:     │     │  Decision: │
│  "Where am  │     │  "What      │
│   I?"       │     │   should I  │
└──────┬──────┘     │   do?"      │
       │            └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│   Reward    │◀────│   Action    │
│             │     │             │
│  Feedback:  │     │  Movement:  │
│  "Good job!"│     │  "Go left"  │
│  or "Try    │     │             │
│   again"    │     └─────────────┘
└─────────────┘
```

**The agent:**
1. **Observes** the current state
2. **Chooses** an action
3. **Receives** a reward
4. **Learns** to make better choices
5. **Repeats** the process

---

## 💻 **Hands-On: Your First Agent**

### **Step 1: Meet the Environment**

Let's use the `line_world` - a simple 1D navigation task:

```
Positions: 0 ── 1 ── 2 ── 3 ── 4 ── 5 ── 6 ── 7 ── 8 ── 9
Goal:                                            🎯
Agent starts here: 🏃
```

**The Task:**
- Agent starts at position 0
- Goal is at position 9 (or wherever you set `goal_pos`)
- Agent can move **left** (-1) or **right** (+1)
- **Reward**: +1.0 for reaching goal, -0.02 for each step
- **Episode ends**: When goal reached or max steps taken

### **Step 2: Try a Random Agent**

Let's see what happens when an agent makes random decisions:

```bash
multiverse train --algo random --verse line_world --episodes 20 --max-steps 50
```

**Expected Output:**
```
Episode 1/20: return=-0.8, steps=40  (random wandering)
Episode 2/20: return=-0.6, steps=30  (got lucky)
Episode 3/20: return=-1.0, steps=50  (hit max steps)
...
Average Return: -0.75
```

**What happened?**
- The random agent moves left/right randomly
- Sometimes it reaches the goal by luck
- Most times it wanders aimlessly
- **Average reward is negative** (more penalties than successes)

### **Step 3: Try a Learning Agent**

Now let's use Q-learning, which **learns from experience**:

```bash
multiverse train --algo q --verse line_world --episodes 100 --max-steps 50
```

**Expected Output:**
```
Episode 1/20: return=-0.8, steps=40  (exploring randomly)
Episode 20/40: return=0.2, steps=15   (starting to learn)
Episode 60/80: return=0.8, steps=8    (much better!)
Episode 100/100: return=0.9, steps=5  (near optimal!)
```

**What happened?**
- Early episodes: Random exploration (like the random agent)
- Middle episodes: Learning efficient paths
- Later episodes: Consistent success with minimal steps
- **Average reward becomes positive** (more successes than penalties)

---

## 🔍 **Understanding the Results**

### **Why Did the Q-Learning Agent Improve?**

The Q-learning agent learns a **"value function"** that answers:
- "How good is it to be at position X?"
- "What's the best action to take from position X?"

**Visualizing the Learning:**

**Early Training (Episode 10):**
```
Position: 0  1  2  3  4  5  6  7  8  9
Value:    ?  ?  ?  ?  ?  ?  ?  ?  ?  🎯
Action:   ↔  ↔  ↔  ↔  ↔  ↔  ↔  ↔  ↔  ✓
```
*Agent doesn't know what's good yet*

**Later Training (Episode 100):**
```
Position: 0  1  2  3  4  5  6  7  8  9
Value:    0.1 0.2 0.4 0.6 0.8 0.9 0.95 🎯
Action:   →  →  →  →  →  →  →  →  →  ✓
```
*Agent knows: "Go right! Higher positions = better rewards!"*

### **The Learning Process**

1. **Exploration**: Try different actions to discover rewards
2. **Exploitation**: Use knowledge to choose best actions
3. **Learning**: Update value estimates based on experience
4. **Convergence**: Find optimal strategy

**This is the essence of all RL algorithms!**

---

## 🚀 **Experiment Further**

### **Try Shorter vs Longer Training**

```bash
# Short run
multiverse train --algo q --verse line_world --episodes 20

# Longer run
multiverse train --algo q --verse line_world --episodes 200
```

**Question:** How does extra training time affect stability and final performance?

### **Compare Algorithms**

```bash
# Different algorithms on same task
multiverse train --algo ppo --verse line_world --episodes 100
multiverse train --algo dqn --verse line_world --episodes 100
```

**Question:** Which algorithm learns faster? Why might that be?

### **Try A Harder Verse**

```bash
# Move from 1D to a richer grid task
multiverse train --algo q --verse grid_world --episodes 150
```

**Question:** How does a larger state space change learning speed?

---

## 🎯 **Key Takeaways**

1. **Agents learn by trial and error** - Experience creates knowledge
2. **Good agents balance exploration and exploitation** - Try new things, but use what works
3. **Rewards guide learning** - Positive feedback reinforces good behavior
4. **Different algorithms learn differently** - Some faster, some more stable
5. **Environment difficulty affects learning** - Harder problems need more training

---

## 📚 **Next Steps**

**Ready for more?** Continue to:
- **Tutorial 02**: Exploration vs Exploitation (grid_world)
- **Tutorial 03**: Value Learning (Q-Learning deep dive)
- **Tutorial 04**: Policy Optimization (PPO and policy gradients)

**Want to dive deeper?**
- Read: [RL Concepts](../RL_CONCEPTS.md)
- Follow: [Learning Path](../LEARNING_PATH.md)
- Experiment: [Notebook 02: Exploration](../interactive/02_exploration.ipynb)

---

**Tutorial Complete!** 🎉

*You now understand the fundamental RL loop and have seen learning in action. This foundation will help you understand all advanced RL concepts in Multiverse.*
