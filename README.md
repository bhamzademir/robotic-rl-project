# Franka Panda Classical End-Effector Control

Torque-level Cartesian control for a Franka Emika Panda in MuJoCo. The project
validates forward and inverse kinematics, tracks three Cartesian paths, measures
controller limits, and uses a scripted grasp-and-place sequence to build two
ordered towers from five randomized blocks.

## Setup

The reported experiments used Python 3.11 and MuJoCo 3.9.0. Run the commands
from the repository root so the scripts can access the scene and model assets.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the experiments

```bash
python validate_kinematics.py
python simulation_task_space_pd.py --headless --no-realtime
python trajectory_following.py --headless --seed 19
python trajectory_experiments.py --headless --runs 5 --seed 42
python plot_trajectory_study_results.py
python step_response_experiments.py --trials 5 --seed 42
python speed_limit_experiments.py --trials 3 --seed 42
python robustness_experiments.py --trials 5 --seed 42
python plot_controller_limit_results.py
python simulation_grasp_lift.py --headless --no-realtime
python simulation_block_stacking.py --headless --episodes 10 --seed 42
python analyze_block_stacking_results.py
```

The batch commands regenerate files under `results/`. Published run data:

- [Cartesian path tracking](results/trajectory_study/summary.csv) includes
  five trials each of an ellipse (logged as `circle`), figure-eight and helix.
- [Controller limits](results/controller_limits/) includes position and
  orientation steps, a traversal-time sweep, and static-load and force-pulse tests.
- [Five-block stacking](results/block_stacking/episodes.csv) includes ten seeded
  layouts, requested orders, completion times and placement errors. A passing
  episode completes both towers within 120 s and meets the 35 mm horizontal
  and 25 mm vertical placement tolerances for every block. Add
  `--save-step-logs` to generate the full controller traces locally.

## Replay one stacking episode

On macOS, use the MuJoCo viewer launcher:

```bash
mjpython simulation_block_stacking.py --episode 2 --seed 42
```

For a headless replay, use `python` and add `--headless`. A single-episode run
writes its summary and any requested step log to a seed-and-episode directory
under `results/block_stacking/replays/`. The batch results stay available for analysis.

## Report and testing

- [LaTeX report source](report/latex/main.tex) and [figures](report/latex/figures/)
- [Individual test commands and pass criteria](TESTING.md)

To build a local copy of the report with a LaTeX installation:

```bash
cd report/latex
latexmk -pdf main.tex
```
