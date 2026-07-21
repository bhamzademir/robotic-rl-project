# Franka Panda Classical End-Effector Control

This version prioritizes the classical controller requested by the supervisor.

## Included

- Full 6D damped-least-squares inverse kinematics
- Position and rotation-matrix orientation errors
- Joint-limit enforcement and non-convergence reporting
- MuJoCo geometric Jacobian interface
- Finite-difference Jacobian validation
- Five-pose IK validation
- Smooth Cartesian trajectory with quaternion SLERP
- Desired/actual logging, error plots, 3D path plot, and CSV output

## Setup

Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The existing `assets/panda.xml` and associated mesh assets must be present relative to `scene.xml`.

## Run validation

```bash
python validate_kinematics.py
```

## Run the classical trajectory experiment

```bash
python simulation.py
```

Outputs are saved under `results/`.

## Important limitation

`forward_kinematics()` currently uses MuJoCo as the kinematic reference. For the final report,
an independent DH- or XML-transform-based forward-kinematics implementation should be added
and compared against this MuJoCo reference over at least 20 configurations.

RL training remains disabled until the classical controller is validated and a non-zero staged
pick-and-place reward is implemented.
