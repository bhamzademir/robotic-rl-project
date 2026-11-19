"""Scripted reach-grasp-lift demonstration using torque-level task-space PD.

Place this file beside environment.py and scene.xml, then run on macOS with:

    mjpython simulation_grasp_lift.py

The demo:
1. opens the gripper,
2. moves above the cube,
3. descends around the cube,
4. closes the gripper,
5. lifts the cube.

The grasp height is intentionally exposed as GRASP_HAND_OFFSET_Z because the
exact hand-body origin depends on the Panda XML model.
"""
from __future__ import annotations

import csv
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv, TaskSpacePDConfig


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

DISPLAY_EVERY = 10          # 500 Hz control, 50 Hz viewer updates.
APPROACH_DURATION = 2.5
DESCEND_DURATION = 2.0
CLOSE_DURATION = 1.2
LIFT_DURATION = 2.5
FINAL_HOLD_DURATION = 1.0

# Hand-body target relative to the cube centre. Tune this first if the fingers
# stop above or below the cube. Typical useful range: 0.075-0.105 m.
GRASP_HAND_OFFSET_Z = 0.090
APPROACH_CLEARANCE_Z = 0.190
LIFT_DISTANCE_Z = 0.120

# A successful grasp must raise the cube by at least this amount.
SUCCESSFUL_LIFT_THRESHOLD = 0.050


def smoothstep(u: float) -> tuple[float, float]:
    """Cubic position timing and derivative with respect to normalized time."""
    u = float(np.clip(u, 0.0, 1.0))
    s = 3.0 * u**2 - 2.0 * u**3
    ds_du = 6.0 * u - 6.0 * u**2
    return s, ds_du


def interpolate_segment(
    start: np.ndarray,
    goal: np.ndarray,
    elapsed: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth desired position and velocity for one Cartesian segment."""
    u = elapsed / max(duration, 1e-9)
    s, ds_du = smoothstep(u)
    delta = goal - start
    position = start + s * delta
    velocity = (ds_du / duration) * delta
    return position, velocity


def run_demo(render: bool = True, realtime: bool = True) -> None:
    env = FrankaPickAndPlaceEnv(render_mode="human" if render else None)
    env.reset()

    env.data.qpos[env.arm_qpos_indices] = HOME_Q
    env.data.qvel[:] = 0.0
    env.open_gripper()
    mujoco.mj_forward(env.model, env.data)

    preferred_q = HOME_Q.copy()
    initial_pose = env.get_end_effector_pose()
    target_rotation = initial_pose.rotation.copy()

    cube_initial = env.get_cube_position()
    approach_position = cube_initial + np.array([0.0, 0.0, APPROACH_CLEARANCE_Z])
    grasp_position = cube_initial + np.array([0.0, 0.0, GRASP_HAND_OFFSET_Z])
    lift_position = grasp_position + np.array([0.0, 0.0, LIFT_DISTANCE_Z])

    config = TaskSpacePDConfig(
        kp_position=np.array([180.0, 180.0, 220.0]),
        kd_position=np.array([28.0, 28.0, 34.0]),
        kp_orientation=np.array([45.0, 45.0, 35.0]),
        kd_orientation=np.array([8.0, 8.0, 7.0]),
        nullspace_kp=18.0,
        nullspace_kd=5.0,
        damping=0.05,
    )

    dt = env.model.opt.timestep
    logs: list[dict] = []
    elapsed_global = 0.0
    any_saturation = False

    print("Starting scripted reach-grasp-lift demonstration")
    print(f"Simulation rate: {1.0 / dt:.0f} Hz")
    print(f"Gripper actuator ID: {env.gripper_actuator_id}")
    if env.gripper_actuator_id is not None:
        actuator_id = env.gripper_actuator_id
        print(
            "Gripper control range:",
            env.model.actuator_ctrlrange[actuator_id].copy(),
        )
    print(f"Initial cube position: {cube_initial}")
    print(f"Approach hand target: {approach_position}")
    print(f"Grasp hand target:    {grasp_position}")
    print(f"Lift hand target:     {lift_position}")

    def control_phase(
        phase_name: str,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        duration: float,
        gripper_opening: float,
    ) -> np.ndarray:
        nonlocal elapsed_global, any_saturation
        steps = max(1, int(duration / dt))
        print(f"Phase: {phase_name}")

        last_desired = start_position.copy()
        for step in range(steps):
            elapsed = step * dt
            desired_position, desired_velocity = interpolate_segment(
                start_position, goal_position, elapsed, duration
            )
            last_desired = desired_position
            env.set_gripper_opening(gripper_opening)

            diagnostics = env.step_task_space_pd(
                target_position=desired_position,
                target_rotation=target_rotation,
                target_linear_velocity=desired_velocity,
                target_angular_velocity=np.zeros(3),
                preferred_q=preferred_q,
                config=config,
            )
            any_saturation |= diagnostics["torque_saturated"]

            pose = env.get_end_effector_pose()
            cube_position = env.get_cube_position()
            logs.append(
                {
                    "time": elapsed_global,
                    "phase": phase_name,
                    "desired_x": desired_position[0],
                    "desired_y": desired_position[1],
                    "desired_z": desired_position[2],
                    "actual_x": pose.position[0],
                    "actual_y": pose.position[1],
                    "actual_z": pose.position[2],
                    "cube_x": cube_position[0],
                    "cube_y": cube_position[1],
                    "cube_z": cube_position[2],
                    "position_error_m": diagnostics["position_error"],
                    "orientation_error_rad": diagnostics["orientation_error"],
                    "gripper_opening_m": env.get_gripper_opening(),
                    "torque_norm_nm": float(np.linalg.norm(diagnostics["torque"])),
                    "torque_saturated": int(diagnostics["torque_saturated"]),
                }
            )
            elapsed_global += dt

            if render and step % DISPLAY_EVERY == 0:
                env.render()
                if realtime:
                    time.sleep(DISPLAY_EVERY * dt)

        return last_desired

    def hold_phase(
        phase_name: str,
        target_position: np.ndarray,
        duration: float,
        gripper_opening: float,
    ) -> None:
        control_phase(
            phase_name=phase_name,
            start_position=target_position,
            goal_position=target_position,
            duration=duration,
            gripper_opening=gripper_opening,
        )

    try:
        # Open and move above the cube.
        current = initial_pose.position.copy()
        current = control_phase(
            "approach",
            current,
            approach_position,
            APPROACH_DURATION,
            gripper_opening=1.0,
        )

        # Descend while keeping the fingers open.
        current = control_phase(
            "descend",
            approach_position,
            grasp_position,
            DESCEND_DURATION,
            gripper_opening=1.0,
        )

        # Close around the cube while holding the hand pose.
        hold_phase(
            "close_gripper",
            grasp_position,
            CLOSE_DURATION,
            gripper_opening=0.0,
        )

        cube_before_lift = env.get_cube_position().copy()

        # Lift with the gripper kept closed.
        current = control_phase(
            "lift",
            grasp_position,
            lift_position,
            LIFT_DURATION,
            gripper_opening=0.0,
        )
        hold_phase(
            "final_hold",
            lift_position,
            FINAL_HOLD_DURATION,
            gripper_opening=0.0,
        )

        cube_final = env.get_cube_position().copy()
        lifted_distance = float(cube_final[2] - cube_initial[2])
        retained_during_lift = float(cube_final[2] - cube_before_lift[2])
        success = lifted_distance >= SUCCESSFUL_LIFT_THRESHOLD

        final_pose = env.get_end_effector_pose()
        final_position_error = float(np.linalg.norm(lift_position - final_pose.position))

        print(f"Cube final position: {cube_final}")
        print(f"Cube lifted from initial height by: {1000.0 * lifted_distance:.1f} mm")
        print(f"Cube height change during lift phase: {1000.0 * retained_during_lift:.1f} mm")
        print(f"Final hand position error: {1000.0 * final_position_error:.2f} mm")
        print(f"Any torque saturation: {any_saturation}")
        print(f"Grasp-and-lift success: {success}")
    finally:
        env.close()

    save_results(logs)


def save_results(logs: list[dict]) -> None:
    if not logs:
        return

    output_dir = Path("results/grasp_lift")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "grasp_lift_log.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)

    t = np.array([row["time"] for row in logs])
    desired = np.array(
        [[row["desired_x"], row["desired_y"], row["desired_z"]] for row in logs]
    )
    actual = np.array(
        [[row["actual_x"], row["actual_y"], row["actual_z"]] for row in logs]
    )
    cube = np.array(
        [[row["cube_x"], row["cube_y"], row["cube_z"]] for row in logs]
    )
    position_error = np.array([row["position_error_m"] for row in logs])
    gripper_opening = np.array([row["gripper_opening_m"] for row in logs])
    torque_norm = np.array([row["torque_norm_nm"] for row in logs])

    plt.figure(figsize=(10, 6))
    plt.plot(t, desired[:, 2], linestyle="--", label="Desired hand z")
    plt.plot(t, actual[:, 2], label="Actual hand z")
    plt.plot(t, cube[:, 2], label="Cube z")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Reach-Grasp-Lift Height Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "height_tracking.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, 1000.0 * position_error, label="Hand position error")
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (mm)")
    plt.title("Task-Space Position Error During Grasp-and-Lift")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_error.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, 1000.0 * gripper_opening, label="Approximate finger opening")
    plt.xlabel("Time (s)")
    plt.ylabel("Opening (mm)")
    plt.title("Gripper Opening")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gripper_opening.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, torque_norm, label="Arm torque norm")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque norm (N m)")
    plt.title("Arm Torque Demand During Grasp-and-Lift")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "torque_norm.png", dpi=250)
    plt.close()

    print(f"Saved grasp-and-lift logs and plots under {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="disable viewer")
    parser.add_argument("--no-realtime", action="store_true", help="do not pace viewer")
    args = parser.parse_args()
    run_demo(render=not args.headless, realtime=not args.no_realtime)
