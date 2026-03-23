# RL Concepts

This is a compact theory guide tied to the current Multiverse repo.

The previous local draft mixed valid RL ideas with commands and links that are not part of the present CLI. This version keeps the theory and only uses examples that fit the repository as it exists today.

## The RL Loop

Reinforcement learning is an interaction cycle:

1. observe the current state
2. choose an action
3. receive reward and next state
4. update the policy or value estimate
5. repeat

In Multiverse terms:

- the verse is the environment
- the agent chooses actions
- the trainer runs episodes and records artifacts under `runs/`

## States, Actions, And Rewards

- A `state` is the information the agent uses to decide.
- An `action` is the choice the agent makes.
- A `reward` is the training signal that tells the agent whether the transition was useful.

Simple example:

```bash
multiverse train --algo q --verse line_world --episodes 100
```

`line_world` is a good starter verse because the state and action space are small and the learning signal is easy to reason about.

## Value Learning

Value-based methods estimate how good a state or state-action pair is.

- `V(s)` estimates how good state `s` is
- `Q(s, a)` estimates how good action `a` is in state `s`

Q-learning updates a table or approximation of `Q(s, a)` from observed transitions.

Example:

```bash
multiverse train --algo q --verse grid_world --episodes 200
```

## Exploration Vs Exploitation

Agents need to balance:

- exploration: trying actions that may reveal better strategies
- exploitation: using actions that already look strong

Without exploration, the agent can get stuck in a weak policy. Without exploitation, it never stabilizes.

Example:

```bash
multiverse train --algo random --verse line_world --episodes 20
multiverse train --algo q --verse line_world --episodes 100
```

The random policy gathers experience but does not improve. The Q-learning agent uses experience to get better over time.

## Policy Optimization

Policy methods learn the action-selection rule directly instead of only estimating values.

Example:

```bash
multiverse train --algo ppo --verse line_world --episodes 100
```

In practice:

- value methods are often easier to understand first
- policy methods become more attractive as the control problem gets richer

## Memory And Transfer

Multiverse is not only an RL trainer. It also includes memory and transfer-oriented infrastructure.

Relevant docs:

- [Memory System Strengthening Summary](MEMORY_SYSTEM_STRENGTHENING_SUMMARY.md)
- [Phase 2 Implementation Guide](PHASE2_IMPLEMENTATION_GUIDE.md)
- [Engineering Audit](ENGINEERING_AUDIT.md)
- [Verse Documentation Index](../VERSE_DOCUMENTATION_INDEX.md)

## Safety And Runtime Control

The codebase also includes runtime safety and guarded execution paths. Those are part of what makes this repo larger than a minimal RL tutorial project.

For context, start with:

- [Project Introduction](PROJECT_INTRO.md)
- [Paper](PAPER.md)
- [Engineering Audit](ENGINEERING_AUDIT.md)

## Useful Commands

These commands are part of the current top-level CLI:

```bash
multiverse doctor
multiverse universe list
multiverse train --algo q --verse line_world --episodes 50
multiverse runs latest
multiverse sim list
multiverse sim2real --dry-run
```

## Next Steps

- Start with [Tutorial 01](tutorials/01_hello_world.md)
- Follow the [Learning Path](LEARNING_PATH.md)
- Use the notebooks under [interactive](interactive/) for hands-on work
