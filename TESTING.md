# Individual Task Testing

Run all commands from the project root. Headless tests use the project virtual
environment and run faster than the viewer versions.

## 1. Kinematics and inverse kinematics

```bash
./venv/bin/python validate_kinematics.py
```

Pass condition: all IK targets converge, joint limits are respected, and the
script prints `Result: all IK tests converged.` Results are saved to
`results/logs/ik_validation_results.csv`.

## 2. Task-space PD controller

Headless:

```bash
./venv/bin/python simulation_task_space_pd.py --headless --no-realtime
```

With the MuJoCo viewer on macOS:

```bash
./venv/bin/mjpython simulation_task_space_pd.py
```

Check the final position/orientation error and confirm that torque saturation
is false. Logs and plots are saved under `results/task_space_pd/`.

## 3. Grasp and lift

Headless:

```bash
./venv/bin/python simulation_grasp_lift.py --headless --no-realtime
```

With the viewer:

```bash
./venv/bin/mjpython simulation_grasp_lift.py
```

Pass condition: the script prints `Grasp-and-lift success: True`. The cube must
rise by at least 50 mm. Logs and plots are saved under `results/grasp_lift/`.

## 4. One randomized waypoint trajectory

```bash
./venv/bin/python trajectory_following.py --headless --seed 19
```

Repeat the same seed to reproduce a run. Change the seed to test a different
robot start and waypoint set. Plots are saved under `results/plots/`.

Viewer version:

```bash
./venv/bin/mjpython trajectory_following.py --seed 19
```

For a named path with the desired trajectory drawn in blue and the current
target shown as an orange marker in the MuJoCo viewer:

```bash
./venv/bin/mjpython trajectory_experiments.py \
  --runs 1 --paths figure_eight --seed 42 --viewer-only --centre-z 0.58
```

Replace `figure_eight` with `circle` or `helix` as needed. Use
`--hide-path-overlay` only when an unobstructed scene is required. The
`--viewer-only` option prevents a screenshot run from overwriting the saved
five-run study. `--centre-z 0.58` lowers the path toward the tabletop for a
clearer scene composition; the robot follows that lowered reference rather
than merely drawing a displaced overlay.

## 5. One named complex trajectory

Test each path independently:

```bash
./venv/bin/python trajectory_experiments.py --headless --viewer-only --runs 1 --paths circle --seed 42
./venv/bin/python trajectory_experiments.py --headless --viewer-only --runs 1 --paths figure_eight --seed 42
./venv/bin/python trajectory_experiments.py --headless --viewer-only --runs 1 --paths helix --seed 42
```

Each command prints run metrics and preserves the committed study results.
A run succeeds when its maximum position error is at most 50 mm.

## 6. Full repeated trajectory study

```bash
./venv/bin/python trajectory_experiments.py --headless --runs 5 --seed 42
```

Check `results/trajectory_study/runs.csv` for individual runs and
`results/trajectory_study/summary.csv` for mean, sample standard deviation, and
success rate per trajectory.

## 7. One randomized five-block stacking episode

```bash
./venv/bin/python simulation_block_stacking.py \
  --headless --episodes 1 --seed 42 --horizon 120 --save-step-logs
```

Viewer version:

```bash
./venv/bin/mjpython simulation_block_stacking.py \
  --episodes 1 --seed 42 --horizon 120
```

The episode passes only when both towers contain the requested blocks in the
requested order, every placement meets the tolerance, and the task finishes
within the horizon. Inspect
`results/block_stacking/replays/seed_42_episode_01/episodes.csv` for the exact
assignment, initial positions, errors, completion time, and failure reason.

## 8. Ten-episode stacking evaluation

```bash
./venv/bin/python simulation_block_stacking.py \
  --headless --episodes 10 --seed 42 --horizon 120
```

Check `results/block_stacking/summary.json` for success rate, completion-time
mean and standard deviation, and placement-error statistics. The archived
seed-42 evaluation passed all ten episodes. Keep any failures from new runs
in the results.

## 9. Controller step-response limits

```bash
./venv/bin/python step_response_experiments.py --trials 5 --seed 42
```

This independently tests 20, 50, and 100 mm position steps and 5, 15, and
30 degree orientation steps. Rise time, overshoot, settling time,
steady-state error, and torque saturation are saved under
`results/controller_limits/step_response/`.

## 10. Trajectory speed limits

```bash
./venv/bin/python speed_limit_experiments.py --trials 3 --seed 42
```

Circle and figure-eight trajectories are evaluated at progressively shorter
durations. The output under `results/controller_limits/speed_limit/` records
tracking error, torque saturation, joint-velocity-limit utilization, and the
first conditions that violate the declared acceptance thresholds.

## 11. Payload and disturbance rejection

```bash
./venv/bin/python robustness_experiments.py --trials 5 --seed 42
```

The payload conditions apply an unmodeled downward end-effector load. The
disturbance conditions apply repeatable lateral force pulses during a hold
task. Peak displacement, recovery time, residual error, vibration, and torque
saturation are saved under `results/controller_limits/robustness/`.

## Recommended testing order

1. Kinematics.
2. Task-space PD.
3. Grasp and lift.
4. Each complex trajectory once.
5. One stacking episode with the viewer.
6. Headless repeated trajectory study.
7. Controller step-response study.
8. Trajectory speed-limit study.
9. Payload and disturbance-rejection study.
10. Headless ten-episode stacking study.
