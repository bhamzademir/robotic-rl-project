"""Scripted three-block stacking with quintic Cartesian splines and task-space PD.

Run with a viewer on macOS:
    mjpython simulation_block_stacking.py

Run without a viewer (useful for validation):
    python simulation_block_stacking.py --headless
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv, TaskSpacePDConfig
from trajectory_generator import interpolate_pose


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRASP_HAND_OFFSET_Z = 0.090
TRANSIT_CLEARANCE_Z = 0.190
PLACE_CLEARANCE_Z = 0.006
MOVE_DURATION = 2.5
GRIPPER_DURATION = 1.0
SETTLE_DURATION = 0.5
DISPLAY_EVERY = 10
STACK_XY_TOLERANCE = 0.035
STACK_Z_TOLERANCE = 0.025


def run_demo(render: bool = True, realtime: bool = True) -> bool:
    """Execute the fixed-position stacking plan and return its success status."""
    env = FrankaPickAndPlaceEnv(render_mode="human" if render else None)
    env.reset()
    env.data.qpos[env.arm_qpos_indices] = HOME_Q
    env.data.qvel[:] = 0.0
    env.open_gripper()
    mujoco.mj_forward(env.model, env.data)

    preferred_q = HOME_Q.copy()
    target_rotation = env.get_end_effector_pose().rotation.copy()
    current_target = env.get_end_effector_pose().position.copy()
    dt = float(env.model.opt.timestep)
    logs: list[dict[str, float | str | int]] = []
    elapsed_global = 0.0
    config = TaskSpacePDConfig(
        kp_position=np.array([180.0, 180.0, 220.0]),
        kd_position=np.array([28.0, 28.0, 34.0]),
        kp_orientation=np.array([45.0, 45.0, 35.0]),
        kd_orientation=np.array([8.0, 8.0, 7.0]),
        nullspace_kp=18.0,
        nullspace_kd=5.0,
        damping=0.05,
    )

    def phase(name: str, goal: np.ndarray, duration: float, opening: float) -> None:
        nonlocal current_target, elapsed_global
        start = current_target.copy()
        steps = max(1, int(np.ceil(duration / dt)))
        for step in range(steps + 1):
            local_t = min(step * dt, duration)
            sample = interpolate_pose(
                start, goal, target_rotation, target_rotation, local_t, duration
            )
            env.set_gripper_opening(opening)
            diagnostics = env.step_task_space_pd(
                target_position=sample.position,
                target_rotation=sample.rotation,
                target_linear_velocity=sample.velocity,
                target_angular_velocity=np.zeros(3),
                preferred_q=preferred_q,
                config=config,
            )
            pose = env.get_end_effector_pose()
            blocks = env.get_all_block_positions()
            logs.append({
                "time": elapsed_global,
                "phase": name,
                "desired_x": sample.position[0],
                "desired_y": sample.position[1],
                "desired_z": sample.position[2],
                "actual_x": pose.position[0],
                "actual_y": pose.position[1],
                "actual_z": pose.position[2],
                "position_error_m": diagnostics["position_error"],
                "orientation_error_rad": diagnostics["orientation_error"],
                "torque_saturated": int(diagnostics["torque_saturated"]),
                **{
                    f"{block}_{axis}": blocks[block][axis_idx]
                    for block in env.block_names
                    for axis_idx, axis in enumerate(("x", "y", "z"))
                },
            })
            elapsed_global += dt
            if render and step % DISPLAY_EVERY == 0:
                env.render()
                if realtime:
                    time.sleep(DISPLAY_EVERY * dt)
        current_target = np.asarray(goal, dtype=float).copy()

    try:
        print("Starting fixed-position three-block stacking demonstration")
        for level, block_name in enumerate(env.block_names):
            block = env.get_block_position(block_name)
            stack = env.get_stack_target_position(level)
            pick = block + np.array([0.0, 0.0, GRASP_HAND_OFFSET_Z])
            pick_above = block + np.array([0.0, 0.0, TRANSIT_CLEARANCE_Z])
            place = stack + np.array([0.0, 0.0, GRASP_HAND_OFFSET_Z + PLACE_CLEARANCE_Z])
            place_above = place + np.array([0.0, 0.0, TRANSIT_CLEARANCE_Z - GRASP_HAND_OFFSET_Z])

            print(f"Stacking {block_name} at level {level}")
            phase(f"{block_name}_approach", pick_above, MOVE_DURATION, 1.0)
            phase(f"{block_name}_descend", pick, MOVE_DURATION, 1.0)
            phase(f"{block_name}_grasp", pick, GRIPPER_DURATION, 0.0)
            phase(f"{block_name}_lift", pick_above, MOVE_DURATION, 0.0)
            phase(f"{block_name}_transfer", place_above, MOVE_DURATION, 0.0)
            phase(f"{block_name}_place", place, MOVE_DURATION, 0.0)
            phase(f"{block_name}_release", place, GRIPPER_DURATION, 1.0)
            phase(f"{block_name}_retreat", place_above, MOVE_DURATION, 1.0)
            phase(f"{block_name}_settle", place_above, SETTLE_DURATION, 1.0)

        final_positions = env.get_all_block_positions()
        errors = {
            name: final_positions[name] - env.get_stack_target_position(level)
            for level, name in enumerate(env.block_names)
        }
        success = all(
            np.linalg.norm(error[:2]) <= STACK_XY_TOLERANCE
            and abs(error[2]) <= STACK_Z_TOLERANCE
            for error in errors.values()
        )
        for name, error in errors.items():
            print(f"{name} placement error [m]: {error}")
        print(f"Three-block stack success: {success}")
    finally:
        env.close()

    output_dir = Path("results/block_stacking")
    output_dir.mkdir(parents=True, exist_ok=True)
    if logs:
        with (output_dir / "block_stacking_log.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="disable the viewer")
    parser.add_argument(
        "--no-realtime", action="store_true", help="do not pace viewer updates"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(render=not args.headless, realtime=not args.no_realtime)
