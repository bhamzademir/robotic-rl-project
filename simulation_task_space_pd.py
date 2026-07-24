"""Torque-level 6D task-space PD demonstration for the Franka Panda.

Run on macOS with:
    mjpython simulation_task_space_pd.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv, TaskSpacePDConfig

HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
CONTROL_HZ = 500.0
DISPLAY_EVERY = 10  # Viewer sync at 50 Hz while controller runs at 500 Hz.
MOVE_DURATION = 4.0
HOLD_DURATION = 1.0


def smoothstep(t: float) -> tuple[float, float]:
    """Cubic timing law s(t)=3t^2-2t^3 and its normalized derivative."""
    t = float(np.clip(t, 0.0, 1.0))
    return 3.0 * t**2 - 2.0 * t**3, 6.0 * t - 6.0 * t**2


def run_demo() -> None:
    env = FrankaPickAndPlaceEnv(render_mode="human")
    env.reset()
    env.data.qpos[env.arm_qpos_indices] = HOME_Q
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    preferred_q = HOME_Q.copy()
    start_pose = env.get_end_effector_pose()

    # A reachable lateral target; orientation is held fixed in this first controller test.
    target_position = start_pose.position + np.array([0.0, 0.12, 0.04])
    target_rotation = start_pose.rotation.copy()

    # Conservative initial gains. Increase only after checking torque saturation and oscillation.
    config = TaskSpacePDConfig(
        kp_position=np.array([160.0, 160.0, 200.0]),
        kd_position=np.array([26.0, 26.0, 30.0]),
        kp_orientation=np.array([40.0, 40.0, 30.0]),
        kd_orientation=np.array([7.0, 7.0, 6.0]),
        nullspace_kp=15.0,
        nullspace_kd=4.0,
        damping=0.05,
    )

    dt = env.model.opt.timestep
    move_steps = int(MOVE_DURATION / dt)
    hold_steps = int(HOLD_DURATION / dt)
    total_steps = move_steps + hold_steps
    logs: list[dict] = []

    print("Starting torque-level 6D task-space PD control")
    print(f"Simulation timestep: {dt:.4f} s ({1.0 / dt:.0f} Hz)")
    print(f"Start position:  {start_pose.position}")
    print(f"Target position: {target_position}")

    try:
        for step in range(total_steps):
            if step < move_steps:
                normalized_time = step / max(move_steps - 1, 1)
                s, ds = smoothstep(normalized_time)
                desired_position = (
                    start_pose.position + s * (target_position - start_pose.position)
                )
                desired_linear_velocity = (
                    ds / MOVE_DURATION * (target_position - start_pose.position)
                )
            else:
                desired_position = target_position
                desired_linear_velocity = np.zeros(3)

            diagnostics = env.step_task_space_pd(
                target_position=desired_position,
                target_rotation=target_rotation,
                target_linear_velocity=desired_linear_velocity,
                target_angular_velocity=np.zeros(3),
                preferred_q=preferred_q,
                config=config,
            )

            pose = env.get_end_effector_pose()
            logs.append(
                {
                    "time": step * dt,
                    "desired_x": desired_position[0],
                    "desired_y": desired_position[1],
                    "desired_z": desired_position[2],
                    "actual_x": pose.position[0],
                    "actual_y": pose.position[1],
                    "actual_z": pose.position[2],
                    "position_error": diagnostics["position_error"],
                    "orientation_error": diagnostics["orientation_error"],
                    "torque_norm": float(np.linalg.norm(diagnostics["torque"])),
                    "torque_saturated": int(diagnostics["torque_saturated"]),
                }
            )

            if step % DISPLAY_EVERY == 0:
                env.render()
                time.sleep(DISPLAY_EVERY * dt)

        final = logs[-1]
        print(f"Final position error: {1000.0 * final['position_error']:.3f} mm")
        print(f"Final orientation error: {np.degrees(final['orientation_error']):.3f} deg")
        print(f"Any torque saturation: {any(row['torque_saturated'] for row in logs)}")
    finally:
        env.close()

    save_results(logs)


def save_results(logs: list[dict]) -> None:
    output_dir = Path("results/task_space_pd")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "task_space_pd_log.csv"
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
    position_error = np.array([row["position_error"] for row in logs])
    orientation_error = np.array([row["orientation_error"] for row in logs])
    torque_norm = np.array([row["torque_norm"] for row in logs])

    plt.figure(figsize=(10, 6))
    for axis, label in enumerate(["x", "y", "z"]):
        plt.plot(t, desired[:, axis], linestyle="--", label=f"Desired {label}")
        plt.plot(t, actual[:, axis], label=f"Actual {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Task-Space PD Position Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_tracking.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, 1000.0 * position_error, label="Position error")
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (mm)")
    plt.title("Task-Space PD Translational Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_error.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, np.degrees(orientation_error), label="Orientation error")
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation error (deg)")
    plt.title("Task-Space PD Orientation Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "orientation_error.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, torque_norm, label="Joint-torque norm")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque norm (N m)")
    plt.title("Task-Space PD Torque Demand")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "torque_norm.png", dpi=250)
    plt.close()

    print(f"Saved controller logs and plots under {output_dir}")


if __name__ == "__main__":
    run_demo()
