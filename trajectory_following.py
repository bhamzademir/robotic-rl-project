from __future__ import annotations

from pathlib import Path
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv, TaskSpacePDConfig


OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTROL_DT = 0.01          # 100 Hz controller update
SEGMENT_DURATION = 4.0     # seconds per edge
BLEND_TIME = 1.0           # parabolic blend duration

SETTLE_POSITION_TOLERANCE = 0.005       # 5 mm
SETTLE_ORIENTATION_TOLERANCE = np.deg2rad(2.0)
SETTLE_STABLE_TIME = 0.25               # remain within tolerance before logging
SETTLE_TIMEOUT = 15.0

HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def lspb_scalar(
    t: float,
    duration: float,
    blend_time: float,
) -> tuple[float, float, float]:
    """Normalized Linear Segment with Parabolic Blends (LSPB)."""
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    if not 0.0 < blend_time <= duration / 2.0:
        raise ValueError("blend_time must satisfy 0 < blend_time <= duration/2.")

    t = float(np.clip(t, 0.0, duration))

    acceleration = 1.0 / (blend_time * (duration - blend_time))
    cruise_velocity = acceleration * blend_time

    if t < blend_time:
        s = 0.5 * acceleration * t**2
        s_dot = acceleration * t
        s_ddot = acceleration
    elif t <= duration - blend_time:
        s = cruise_velocity * (t - 0.5 * blend_time)
        s_dot = cruise_velocity
        s_ddot = 0.0
    else:
        remaining = duration - t
        s = 1.0 - 0.5 * acceleration * remaining**2
        s_dot = acceleration * remaining
        s_ddot = -acceleration

    return s, s_dot, s_ddot


def generate_corner_trajectory(
    waypoints: np.ndarray,
    segment_duration: float = SEGMENT_DURATION,
    blend_time: float = BLEND_TIME,
    control_dt: float = CONTROL_DT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synchronized Cartesian position/velocity/acceleration references."""
    times = []
    positions = []
    velocities = []
    accelerations = []

    global_time = 0.0

    for segment_idx, (start, goal) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        local_times = np.arange(0.0, segment_duration + 1e-12, control_dt)

        # Do not duplicate the common point at segment boundaries.
        if segment_idx > 0:
            local_times = local_times[1:]

        displacement = goal - start

        for local_t in local_times:
            s, s_dot, s_ddot = lspb_scalar(
                local_t,
                segment_duration,
                blend_time,
            )

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


def move_to_start(
    env: FrankaPickAndPlaceEnv,
    start_position: np.ndarray,
    fixed_rotation: np.ndarray,
    preferred_q: np.ndarray,
    config: TaskSpacePDConfig,
    timeout: float = SETTLE_TIMEOUT,
    position_tolerance: float = SETTLE_POSITION_TOLERANCE,
    orientation_tolerance: float = SETTLE_ORIENTATION_TOLERANCE,
) -> dict[str, float]:
    """Settle at the first corner before recording the experiment.

    Requiring several consecutive in-tolerance samples prevents a single
    oscillatory crossing from being mistaken for a settled initial state.
    """
    sim_dt = float(env.model.opt.timestep)
    frame_skip = max(1, int(round(CONTROL_DT / sim_dt)))
    max_steps = max(1, int(np.ceil(timeout / CONTROL_DT)))
    stable_steps_required = max(1, int(np.ceil(SETTLE_STABLE_TIME / CONTROL_DT)))
    stable_steps = 0
    diagnostics = None

    print("Moving to the first trajectory corner...")

    for step in range(1, max_steps + 1):
        diagnostics = env.step_task_space_pd(
            target_position=start_position,
            target_rotation=fixed_rotation,
            target_linear_velocity=np.zeros(3),
            target_angular_velocity=np.zeros(3),
            preferred_q=preferred_q,
            config=config,
            frame_skip=frame_skip,
        )

        if env.render_mode == "human":
            time.sleep(CONTROL_DT)

        within_tolerance = (
            diagnostics["position_error"] <= position_tolerance
            and diagnostics["orientation_error"] <= orientation_tolerance
        )
        stable_steps = stable_steps + 1 if within_tolerance else 0
        if stable_steps >= stable_steps_required:
            settling_time = step * CONTROL_DT
            print(
                f"Settled in {settling_time:.2f} s "
                f"(position {diagnostics['position_error'] * 1000:.2f} mm, "
                f"orientation {np.degrees(diagnostics['orientation_error']):.3f} deg)."
            )
            return {
                "settling_time": settling_time,
                "position_error": float(diagnostics["position_error"]),
                "orientation_error": float(diagnostics["orientation_error"]),
            }

    assert diagnostics is not None
    raise RuntimeError(
        "Robot did not settle at the first waypoint within "
        f"{timeout:.1f} s: position error "
        f"{diagnostics['position_error'] * 1000:.2f} mm "
        f"(required <= {position_tolerance * 1000:.1f} mm), "
        f"orientation error {np.degrees(diagnostics['orientation_error']):.3f} deg "
        f"(required <= {np.degrees(orientation_tolerance):.1f} deg)."
    )


def save_plots(
    times: np.ndarray,
    desired_pos: np.ndarray,
    actual_pos: np.ndarray,
    desired_vel: np.ndarray,
    actual_vel: np.ndarray,
    desired_acc: np.ndarray,
    waypoints: np.ndarray,
    orientation_errors: np.ndarray,
    settling_time: float,
    torque_saturated: np.ndarray,
) -> dict[str, float]:
    """Save the plots requested for the trajectory-following experiment."""
    position_error = np.linalg.norm(desired_pos - actual_pos, axis=1)

    # Position tracking
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

    # Velocity tracking
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

    # Desired acceleration profile
    plt.figure(figsize=(10, 6))
    for axis, label in enumerate(("ax", "ay", "az")):
        plt.plot(times, desired_acc[:, axis], label=f"Desired {label}")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title("LSPB Desired Acceleration Profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_acceleration_profile.png", dpi=300)
    plt.close()

    # Position error
    plt.figure(figsize=(10, 5))
    plt.plot(times, position_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Position error norm [m]")
    plt.title("Trajectory Tracking Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_position_error.png", dpi=300)
    plt.close()

    # 3D Cartesian path
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        desired_pos[:, 0],
        desired_pos[:, 1],
        desired_pos[:, 2],
        "--",
        label="Desired path",
    )
    ax.plot(
        actual_pos[:, 0],
        actual_pos[:, 1],
        actual_pos[:, 2],
        label="Actual path",
    )
    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        marker="o",
        label="Waypoints",
    )
    ax.scatter(
        desired_pos[0, 0],
        desired_pos[0, 1],
        desired_pos[0, 2],
        marker="s",
        label="Start",
    )
    ax.scatter(
        desired_pos[-1, 0],
        desired_pos[-1, 1],
        desired_pos[-1, 2],
        marker="x",
        label="Finish",
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Desired vs Actual Cartesian Path")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trajectory_3d_path.png", dpi=300)
    plt.close(fig)

    max_error = float(np.max(position_error))
    mean_error = float(np.mean(position_error))
    rms_error = float(np.sqrt(np.mean(position_error**2)))
    final_error = float(position_error[-1])
    max_orientation_error = float(np.max(orientation_errors))
    saturation_rate = float(np.mean(torque_saturated))

    metrics = {
        "max_position_error_m": max_error,
        "mean_position_error_m": mean_error,
        "rms_position_error_m": rms_error,
        "final_position_error_m": final_error,
        "max_orientation_error_rad": max_orientation_error,
        "settling_time_s": float(settling_time),
        "torque_saturation_rate": saturation_rate,
    }

    print("\n=== Trajectory Tracking Results ===")
    print(f"Maximum position error : {max_error * 1000:.2f} mm")
    print(f"Mean position error    : {mean_error * 1000:.2f} mm")
    print(f"RMS position error     : {rms_error * 1000:.2f} mm")
    print(f"Final position error   : {final_error * 1000:.2f} mm")
    print(f"Maximum orientation error: {np.degrees(max_orientation_error):.3f} deg")
    print(f"Initial settling time  : {settling_time:.2f} s")
    print(f"Torque saturation rate : {saturation_rate * 100:.2f}%")
    print(f"\nPlots saved in: {OUTPUT_DIR.resolve()}")
    return metrics

def generate_random_waypoints(
    env,
    fixed_rotation: np.ndarray,
    seed: int = 42,
    num_waypoints: int = 4,
) -> np.ndarray:
    """
    Generate reproducible, reachable Cartesian waypoints.

    Waypoints are sampled from a conservative workspace and accepted
    only if the full-pose IK solver converges.
    """
    rng = np.random.default_rng(seed)

    # Conservative workspace for Panda above the table.
    workspace_low = np.array([0.42, -0.20, 0.62])
    workspace_high = np.array([0.64, 0.20, 0.78])

    waypoints = []
    preferred_q = env.get_joint_positions().copy()

    max_attempts = 200
    attempts = 0

    while len(waypoints) < num_waypoints and attempts < max_attempts:
        attempts += 1

        candidate = rng.uniform(workspace_low, workspace_high)

        # Avoid waypoints that are almost identical to each other.
        if waypoints:
            distance_to_previous = np.linalg.norm(candidate - waypoints[-1])
            if distance_to_previous < 0.10:
                continue

        # Check reachability using the existing full-pose IK.
        ik_result = env.inverse_kinematics(
            target_position=candidate,
            target_rotation=fixed_rotation,
            initial_q=preferred_q,
            preferred_q=preferred_q,
            max_iters=500,
            position_tol=1e-3,
            orientation_tol=np.deg2rad(1.0),
        )

        if not ik_result.converged:
            continue

        waypoints.append(candidate)

    if len(waypoints) < num_waypoints:
        raise RuntimeError(
            f"Could only generate {len(waypoints)} reachable waypoints "
            f"after {max_attempts} attempts."
        )

    return np.asarray(waypoints, dtype=float)

def main(render: bool = True, seed: int | None = None) -> None:
    # =============================================================
    # RANDOM SEED
    # =============================================================
    # If no seed is provided, generate a different seed for each run.
    # The generated seed is printed so the experiment can be
    # reproduced later using --seed.
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)

    RANDOM_SEED = seed
    NUM_WAYPOINTS = 4

    rng = np.random.default_rng(RANDOM_SEED)

    # Keep the original classical-control environment.
    env = FrankaPickAndPlaceEnv(
        render_mode="human" if render else None,
        control_mode="pose",
        randomize_cube=False,
        randomize_robot_start=False,
    )

    env.reset(seed=RANDOM_SEED)

    # =============================================================
    # 1. RANDOMIZED ROBOT START
    # =============================================================

    # Randomize around the validated HOME_Q rather than the
    # near-singular XML default configuration.
    start_noise = rng.uniform(
        low=-0.06,
        high=0.06,
        size=7,
    )

    random_start_q = HOME_Q + start_noise

    # Respect Panda joint limits.
    random_start_q = np.clip(
        random_start_q,
        env.model.jnt_range[:7, 0],
        env.model.jnt_range[:7, 1],
    )

    env.data.qpos[env.arm_qpos_indices] = random_start_q
    env.data.qvel[:] = 0.0

    mujoco.mj_forward(env.model, env.data)

    print("\n=== Randomized Trajectory Experiment ===")
    print(f"Random seed: {RANDOM_SEED}")
    print("Random start q:")
    print(np.round(random_start_q, 4))

    # Current end-effector pose after randomized joint initialization.
    initial_pose = env.get_end_effector_pose()

    current_start_pos = initial_pose.position.copy()

    # Keep orientation fixed during this trajectory experiment.
    fixed_rotation = initial_pose.rotation.copy()

    preferred_q = env.get_joint_positions().copy()

    print("\nInitial end-effector position:")
    print(np.round(current_start_pos, 4))

    # =============================================================
    # 2. RANDOMIZED REACHABLE WAYPOINTS
    # =============================================================

    workspace_low = np.array([
        0.46,
        -0.16,
        0.66,
    ])

    workspace_high = np.array([
        0.60,
        0.16,
        0.74,
    ])

    random_waypoints = []

    max_attempts = 500
    attempts = 0

    while (
        len(random_waypoints) < NUM_WAYPOINTS
        and attempts < max_attempts
    ):
        attempts += 1

        candidate = rng.uniform(
            workspace_low,
            workspace_high,
        )

        # First waypoint must not be excessively far from
        # the randomized starting end-effector position.
        if len(random_waypoints) == 0:
            start_distance = np.linalg.norm(
                candidate - current_start_pos
            )

            if start_distance > 0.10:
                continue

        # Avoid very short consecutive trajectory segments.
        if len(random_waypoints) > 0:
            segment_distance = np.linalg.norm(
                candidate - random_waypoints[-1]
            )

            if segment_distance < 0.10:
                continue

        # Check full-pose IK feasibility.
        ik_result = env.inverse_kinematics(
            target_position=candidate,
            target_rotation=fixed_rotation,
            initial_q=preferred_q,
            preferred_q=preferred_q,
            max_iters=500,
            position_tol=1e-3,
            orientation_tol=np.deg2rad(1.0),
        )

        if not ik_result.converged:
            continue

        random_waypoints.append(candidate)

    if len(random_waypoints) < NUM_WAYPOINTS:
        raise RuntimeError(
            f"Could generate only {len(random_waypoints)} "
            f"reachable waypoints after {max_attempts} attempts."
        )

    random_waypoints = np.asarray(
        random_waypoints,
        dtype=float,
    )

    # Repeat the first point at the end to close the trajectory.
    waypoints = np.vstack(
        [
            random_waypoints,
            random_waypoints[0],
        ]
    )

    print("\nGenerated reachable waypoints:")

    for i, waypoint in enumerate(
        random_waypoints,
        start=1,
    ):
        if i == 1:
            distance_from_start = np.linalg.norm(
                waypoint - current_start_pos
            )

            print(
                f"Waypoint {i}: "
                f"[{waypoint[0]:.3f}, "
                f"{waypoint[1]:.3f}, "
                f"{waypoint[2]:.3f}] "
                f"(start distance: "
                f"{distance_from_start * 1000:.1f} mm)"
            )
        else:
            print(
                f"Waypoint {i}: "
                f"[{waypoint[0]:.3f}, "
                f"{waypoint[1]:.3f}, "
                f"{waypoint[2]:.3f}]"
            )

    # =============================================================
    # 3. GENERATE LSPB TRAJECTORY
    # =============================================================

    times, desired_pos, desired_vel, desired_acc = (
        generate_corner_trajectory(
            waypoints
        )
    )

    # =============================================================
    # 4. CONTROLLER CONFIGURATION
    # =============================================================

    config = TaskSpacePDConfig(
        kp_position=np.array([
            150.0,
            150.0,
            185.0,
        ]),
        kd_position=np.array([
            30.0,
            30.0,
            34.0,
        ]),
        kp_orientation=np.array([
            30.0,
            30.0,
            25.0,
        ]),
        kd_orientation=np.array([
            10.0,
            10.0,
            9.0,
        ]),
        nullspace_kp=5.0,
        nullspace_kd=2.0,
        damping=0.05,
    )

    sim_dt = float(env.model.opt.timestep)

    frame_skip = max(
        1,
        int(round(CONTROL_DT / sim_dt)),
    )

    actual_positions = []
    actual_velocities = []
    orientation_errors = []
    torque_saturated = []

    settling_result = None

    try:
        # =========================================================
        # 5. MOVE TO FIRST RANDOM WAYPOINT
        # =========================================================

        print(
            "\nMoving to the first randomized "
            "trajectory corner..."
        )

        settling_result = move_to_start(
            env,
            start_position=waypoints[0],
            fixed_rotation=fixed_rotation,
            preferred_q=preferred_q,
            config=config,
        )

        # =========================================================
        # 6. FOLLOW RANDOMIZED LSPB TRAJECTORY
        # =========================================================

        print(
            "Starting randomized LSPB "
            "corner-to-corner trajectory tracking..."
        )

        for target_pos, target_vel in zip(
            desired_pos,
            desired_vel,
        ):
            diagnostics = env.step_task_space_pd(
                target_position=target_pos,
                target_rotation=fixed_rotation,
                target_linear_velocity=target_vel,
                target_angular_velocity=np.zeros(3),
                preferred_q=preferred_q,
                config=config,
                frame_skip=frame_skip,
            )

            actual_positions.append(
                env.get_end_effector_pose()
                .position
                .copy()
            )

            actual_velocities.append(
                env.get_end_effector_twist()[:3]
                .copy()
            )

            orientation_errors.append(
                diagnostics[
                    "orientation_error"
                ]
            )

            torque_saturated.append(
                diagnostics[
                    "torque_saturated"
                ]
            )

            if render:
                time.sleep(CONTROL_DT)

    finally:
        env.close()

    # =============================================================
    # 7. RESULTS
    # =============================================================

    actual_positions = np.asarray(
        actual_positions
    )

    actual_velocities = np.asarray(
        actual_velocities
    )

    orientation_errors = np.asarray(
        orientation_errors
    )

    torque_saturated = np.asarray(
        torque_saturated,
        dtype=bool,
    )

    assert settling_result is not None

    save_plots(
        times=times,
        desired_pos=desired_pos,
        actual_pos=actual_positions,
        desired_vel=desired_vel,
        actual_vel=actual_velocities,
        desired_acc=desired_acc,
        waypoints=waypoints,
        orientation_errors=orientation_errors,
        settling_time=settling_result[
            "settling_time"
        ],
        torque_saturated=torque_saturated,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for the experiment. "
            "If omitted, a new random seed is generated."
        ),
    )

    args = parser.parse_args()

    main(
        render=not args.headless,
        seed=args.seed,
    )
