from __future__ import annotations

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv


OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def lspb_scalar(t: float, duration: float, blend_time: float) -> tuple[float, float, float]:
    """Linear segment with parabolic blends, normalized from 0 to 1."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if not 0 < blend_time <= duration / 2:
        raise ValueError("blend_time must satisfy 0 < blend_time <= duration/2")

    t = float(np.clip(t, 0.0, duration))
    acceleration = 1.0 / (blend_time * (duration - blend_time))
    velocity = acceleration * blend_time

    if t < blend_time:
        s = 0.5 * acceleration * t**2
        s_dot = acceleration * t
        s_ddot = acceleration
    elif t <= duration - blend_time:
        s = velocity * (t - 0.5 * blend_time)
        s_dot = velocity
        s_ddot = 0.0
    else:
        remaining = duration - t
        s = 1.0 - 0.5 * acceleration * remaining**2
        s_dot = acceleration * remaining
        s_ddot = -acceleration
    return s, s_dot, s_ddot


def generate_corner_trajectory(
    waypoints: np.ndarray,
    segment_duration: float = 2.5,
    blend_time: float = 0.5,
    control_dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []

    global_time = 0.0
    for index, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        local_times = np.arange(0.0, segment_duration + 1e-9, control_dt)
        if index > 0:
            local_times = local_times[1:]
        displacement = end - start
        for local_t in local_times:
            s, s_dot, s_ddot = lspb_scalar(local_t, segment_duration, blend_time)
            times.append(global_time + local_t)
            positions.append(start + s * displacement)
            velocities.append(s_dot * displacement)
            accelerations.append(s_ddot * displacement)
        global_time += segment_duration

    return (
        np.asarray(times),
        np.asarray(positions),
        np.asarray(velocities),
        np.asarray(accelerations),
    )


def save_plots(
    times: np.ndarray,
    desired_pos: np.ndarray,
    actual_pos: np.ndarray,
    desired_vel: np.ndarray,
    actual_vel: np.ndarray,
) -> None:
    position_error = np.linalg.norm(desired_pos - actual_pos, axis=1)

    plt.figure(figsize=(10, 6))
    for axis, label in enumerate(("x", "y", "z")):
        plt.plot(times, desired_pos[:, axis], "--", label=f"Desired {label}")
        plt.plot(times, actual_pos[:, axis], label=f"Actual {label}")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("End-Effector Trajectory Tracking")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_position_tracking.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for axis, label in enumerate(("vx", "vy", "vz")):
        plt.plot(times, desired_vel[:, axis], "--", label=f"Desired {label}")
        plt.plot(times, actual_vel[:, axis], label=f"Actual {label}")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("End-Effector Velocity Tracking")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_velocity_tracking.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(times, position_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Position error norm [m]")
    plt.title("Trajectory Tracking Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_position_error.png", dpi=300)
    plt.close()

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2], "--", label="Desired")
    axis.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], label="Actual")
    axis.scatter(desired_pos[0, 0], desired_pos[0, 1], desired_pos[0, 2], marker="o", label="Start")
    axis.scatter(desired_pos[-1, 0], desired_pos[-1, 1], desired_pos[-1, 2], marker="x", label="Finish")
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.set_title("Desired and Actual Cartesian Path")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "trajectory_3d_path.png", dpi=300)
    plt.close(figure)

    print(f"Maximum position error: {position_error.max() * 1000:.2f} mm")
    print(f"RMS position error: {np.sqrt(np.mean(position_error**2)) * 1000:.2f} mm")


def main(render: bool = True) -> None:
    # Four corners plus the first corner again create a complete rectangular path.
    waypoints = np.array(
        [
            [0.45, -0.18, 0.70],
            [0.65, -0.18, 0.70],
            [0.65, 0.18, 0.70],
            [0.45, 0.18, 0.70],
            [0.45, -0.18, 0.70],
        ],
        dtype=np.float64,
    )
    times, desired_pos, desired_vel, _ = generate_corner_trajectory(waypoints)

    env = FrankaPickAndPlaceEnv(
        render_mode="human" if render else None,
        control_mode="absolute",
        randomize_cube=False,
        randomize_robot_start=False,
    )
    env.reset(seed=7)

    actual_positions: list[np.ndarray] = []
    actual_velocities: list[np.ndarray] = []
    try:
        for target in desired_pos:
            action = np.array([target[0], target[1], target[2], 0.0], dtype=np.float32)
            env.step(action)
            actual_positions.append(env.forward_kinematics())
            actual_velocities.append(env.end_effector_velocity())
            if render:
                env.render()
                time.sleep(0.02)
    finally:
        env.close()

    save_plots(
        times,
        desired_pos,
        np.asarray(actual_positions),
        desired_vel,
        np.asarray(actual_velocities),
    )


if __name__ == "__main__":
    main(render=True)
