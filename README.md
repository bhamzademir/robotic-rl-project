# Franka Panda Classical End-Effector Control

This repository contains the current implementation of the classical
end-effector-control pipeline for a Franka Emika Panda robot in MuJoCo.

The project culminates in a classical-control demonstration that picks three
blocks from fixed, known positions and stacks them at one target location.

## Current implementation

- Franka Panda simulation scene in MuJoCo
- Gymnasium-compatible environment
- State-preserving end-effector pose interface
- Complete 6 × 7 geometric Jacobian
- Finite-difference Jacobian validation
- Full 6D damped-least-squares inverse kinematics
- Position and rotation-matrix orientation errors
- Joint-limit enforcement
- Damping near singular configurations
- Non-convergence diagnostics
- Null-space preferred-posture objective
- Five reproducible full-pose IK validation tests
- Stable Cartesian waypoint demonstration
- Torque-level 6D task-space PD control
- Quintic Cartesian spline generation and trajectory tracking
- Scripted reach, grasp, lift, transfer, place, and release sequence
- Fixed-position three-block stacking with privileged simulator state
- CSV output for IK validation results

## Final block-stacking demonstration

On macOS, launch the MuJoCo viewer with:

```bash
mjpython simulation_block_stacking.py
```

For a headless run that still writes the trajectory and block-position log:

```bash
python simulation_block_stacking.py --headless
```

The final script intentionally uses fixed initial positions, a fixed stack
location, and exact block positions from MuJoCo. Vision and reinforcement
learning are outside this small project's scope.

## Requirements

- Python 3.11
- MuJoCo 3.9.0
- Gymnasium
- NumPy
- Matplotlib

The Franka Panda model and its mesh assets must be available under `assets/`
and referenced correctly by `scene.xml`.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
