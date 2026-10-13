# Multiverse native simulation and viewer

The optional native path keeps the scalar `Verse` interface intact and adds a
separate batched path. C++17 owns MuJoCo model loading, per-lane `mjData`, and
CPU stepping; Python keeps task rewards, observations, agent calls, and run
configuration. The viewer uses Multiverse-owned OpenGL 3.3 shaders and a GLFW
window with Dear ImGui panels. Simulation can publish the latest scene snapshot
without waiting for the viewer to render it.

## Install and select

The native build and the graphics viewer stay out of the base install:

- Follow [native/README.md](../native/README.md) and install
  `requirements_native.txt` to build the MuJoCo extension.
- Install `requirements_viewer.txt` to use `python tools/view_3d.py --verse
  warehouse_3d_world`.
- Set `VerseSpec.params["sim_provider"]` to `multiverse_native` for native
  simulation. `create_vectorized_verse(spec, num_envs)` makes native batched
  lanes for point-mass and reacher tasks. Other verses retain independent
  scalar task lanes until their native transition rules reach parity.

The point-mass and reacher verses can run through the native solver while
preserving their observation shapes, action meanings, reward rules, terminal
flags, and checkpoint data. `world_3d` retains discrete terrain actions, and
`warehouse_3d_world` retains its kinematic obstacle rules. MJCF models for both
are included to support their staged transition to native physics. The separate
2D warehouse task and non-3D verses do not enter this path.

## Baseline Verse contracts

| Verse | Observation and actions | Default reward and episode end | Seed behavior |
|---|---|---|---|
| `embodied_point_world` | 6 floats `[x,y,vx,vy,goal_x,goal_y]`; five discrete actions (hold, ±x, ±y) | `-0.01` per step, `+1` at goal; goal terminates, 40-step limit truncates | Starts at the origin; optional domain randomization uses the supplied seed |
| `robot_reacher_world` | 8 floats `[theta1,theta2,dtheta1,dtheta2,target_x,target_y,ee_x,ee_y]`; five discrete actions (hold, ±joint 1, ±joint 2) | `-0.1 × distance - 0.01` per step, `+2` at goal; target terminates, 50-step limit truncates | Seeded initial joints and target sampling |
| `world_3d` | Dict with position, goal, delta, nearby elevations, and timestep; four discrete cardinal moves | `-0.04` per step, `+1` at goal, optional `-1` patrol collision; 120-step limit truncates | Seeded terrain, pits, and patrol placement |
| `warehouse_3d_world` | 7 floats `[x,z,yaw,goal_x,goal_z,dx,dz]`; hold, forward, reverse, left, right | `-0.02` per step, `-0.15` on collision, `+1` at goal; goal terminates, 80-step limit truncates | Fixed initial layout and robot pose |

The benchmark script runs the current serial Python MuJoCo loop and native
batch loop against the same point-mass MJCF, with matched control duration.
Its 1/8/32/64-environment report is the performance baseline artifact; the
4x-at-64 target remains unverified until a native build runs it on target
hardware.

Randomization is explicit and seeded per reset. For example, a verse can use
`params["domain_randomization"] = {"enabled": true, "actuator_gain": [0.9,
1.1], "goal_jitter": 0.04, "visual_color_jitter": 0.06}`. Native lanes apply
the sampled actuator gain independently, reacher targets can shift by the
sampled offset, and the viewer applies the sampled tint. With randomization
disabled, scales and offsets stay at their baseline values.

## Viewer scene contract

`engine3d.viewer_scene.SceneFrame` contains stable names, transforms, materials,
and telemetry. `SceneSnapshotChannel` replaces the stored frame under a short
lock; drawing happens on the viewer thread. `scene_mappings.json` connects
stable scene names to MuJoCo body names and glTF visuals. The included unit box
glTF is a starter visual; other glTF/GLB meshes can be added to the map. When a
mapped mesh is absent, the viewer draws its local box, sphere, or cylinder mesh.

The viewer provides orbit, pan, and zoom; a scene list; object transform and
material tint previews; roughness and lighting controls; and a telemetry panel.
Its OpenGL draw loop uses VSync for window presentation only. Training stepping
stays in the simulator/trainer path.

## Batch training and measurement

Use `orchestrator.vectorized_trainer.VectorizedTrainer` or
`Trainer.run_vectorized(...)` with a `VectorizedVerse`. Agent action selection
continues through `agent.act(observation)` per lane. Learning consumes the same
`ExperienceBatch` type as the scalar path. `Trainer.run(...)` remains the full
artifact-logging scalar flow.

Benchmark with:

```bash
python tools/benchmark_native_physics.py --iterations 1000 --warmup 100 --threads 0 --output runs/native_benchmark.json
```

The report records machine, CPU count, Python/MuJoCo versions, thread request,
warmup, and step rates for 1, 8, 32, and 64 lanes. Use `--require-4x` to fail
when the 64-lane native result is below the agreed 4x target. The target should
be treated as unverified until that command is run on the deployment hardware.
