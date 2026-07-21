"""Run a reproducible full-pose classical-control trajectory experiment."""
from __future__ import annotations

from pathlib import Path
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv
from robot_utils import rotation_angle
from trajectory_generator import interpolate_pose


def rotation_about_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def save_plots(log: dict[str, list[float]], output_dir: Path) -> None:
    time_axis = np.asarray(log["time"])
    desired = np.asarray(log["desired_position"])
    actual = np.asarray(log["actual_position"])

    plt.figure(figsize=(10, 6))
    labels = ["x", "y", "z"]
    for index, label in enumerate(labels):
        plt.plot(time_axis, desired[:, index], "--", label=f"Desired {label}")
        plt.plot(time_axis, actual[:, index], label=f"Actual {label}")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Desired and measured end-effector position")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "position_tracking.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, log["position_error"], label="Position error")
    plt.xlabel("Time [s]")
    plt.ylabel("Error norm [m]")
    plt.title("Translational tracking error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, np.rad2deg(log["orientation_error"]), label="Orientation error")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [deg]")
    plt.title("Orientation tracking error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "orientation_error.png", dpi=200)
    plt.close()

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(desired[:, 0], desired[:, 1], desired[:, 2], "--", label="Desired")
    axis.plot(actual[:, 0], actual[:, 1], actual[:, 2], label="Actual")
    axis.scatter(desired[0, 0], desired[0, 1], desired[0, 2], marker="o", label="Start")
    axis.scatter(desired[-1, 0], desired[-1, 1], desired[-1, 2], marker="x", label="Finish")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("Desired and actual Cartesian path")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cartesian_path_3d.png", dpi=200)
    plt.close(figure)


def run_classical_control_demo(render: bool = True) -> None:
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    env = FrankaPickAndPlaceEnv(render_mode="human" if render else None)
    env.reset()

    initial_pose = env.get_end_effector_pose()
    waypoints = [
        (initial_pose.position.copy(), initial_pose.rotation.copy()),
        (np.array([0.45, -0.15, 0.65]), initial_pose.rotation @ rotation_about_z(-0.25)),
        (np.array([0.55, 0.00, 0.75]), initial_pose.rotation @ rotation_about_z(0.20)),
        (np.array([0.48, 0.18, 0.68]), initial_pose.rotation @ rotation_about_z(0.45)),
    ]
    segment_duration = 3.0
    dt = float(env.model.opt.timestep)
    log = {
        "time": [],
        "desired_position": [],
        "actual_position": [],
        "position_error": [],
        "orientation_error": [],
        "ik_iterations": [],
        "ik_converged": [],
    }

    elapsed = 0.0
    preferred_q = env.get_joint_positions()
    try:
        for segment in range(len(waypoints) - 1):
            start_position, start_rotation = waypoints[segment]
            end_position, end_rotation = waypoints[segment + 1]
            number_of_steps = int(segment_duration / dt)
            for step in range(number_of_steps):
                sample = interpolate_pose(
                    start_position,
                    end_position,
                    start_rotation,
                    end_rotation,
                    step * dt,
                    segment_duration,
                )
                result = env.inverse_kinematics(
                    sample.position,
                    sample.rotation,
                    initial_q=env.get_joint_positions(),
                    preferred_q=preferred_q,
                    max_iters=80,
                    position_tol=5e-4,
                    orientation_tol=np.deg2rad(0.5),
                )
                env.set_joint_position_targets(result.q)
                mujoco.mj_step(env.model, env.data)
                pose = env.get_end_effector_pose()

                log["time"].append(elapsed)
                log["desired_position"].append(sample.position.copy())
                log["actual_position"].append(pose.position.copy())
                log["position_error"].append(float(np.linalg.norm(sample.position - pose.position)))
                log["orientation_error"].append(rotation_angle(pose.rotation, sample.rotation))
                log["ik_iterations"].append(result.iterations)
                log["ik_converged"].append(int(result.converged))
                elapsed += dt

                if render:
                    print("Environment type:", type(env))
                    print("Environment module:", type(env).__module__)
                    print("Render method:", env.render)
                    print("Render method module:", env.render.__module__)
                    env.render()
                    time.sleep(dt)
    finally:
        env.close()

    save_plots(log, output_dir)
    with open(output_dir / "trajectory_log.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "time_s", "desired_x", "desired_y", "desired_z",
            "actual_x", "actual_y", "actual_z",
            "position_error_m", "orientation_error_rad", "ik_iterations", "ik_converged",
        ])
        for index in range(len(log["time"])):
            writer.writerow([
                log["time"][index],
                *log["desired_position"][index],
                *log["actual_position"][index],
                log["position_error"][index],
                log["orientation_error"][index],
                log["ik_iterations"][index],
                log["ik_converged"][index],
            ])

    print(f"Maximum position error: {max(log['position_error']):.6f} m")
    print(f"RMS position error: {np.sqrt(np.mean(np.square(log['position_error']))):.6f} m")
    print(f"Maximum orientation error: {np.rad2deg(max(log['orientation_error'])):.3f} deg")
    print(f"Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    run_classical_control_demo(render=True)