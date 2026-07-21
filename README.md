# Franka Panda Classical End-Effector Control

This repository contains the current implementation of the classical
end-effector-control pipeline for a Franka Emika Panda robot in MuJoCo.

The current milestone focuses on validated robot kinematics and a stable
Cartesian waypoint demonstration. The torque-level operational-space controller
and the final trajectory-following experiments are still under development.

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
- CSV output for IK validation results

## Work in progress

- Dynamically consistent operational-space torque controller
- Mass-matrix and bias-force compensation
- Torque saturation and joint-limit avoidance
- Five dynamic pose-reaching experiments
- Linear segments with parabolic blends
- Quaternion SLERP orientation trajectory
- Desired-versus-actual trajectory logging
- Position, orientation, error, velocity, acceleration, and torque plots
- Final trajectory-following video

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