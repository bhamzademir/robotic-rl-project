"""Repeated large-workspace trajectory-tracking experiments.

The runner evaluates circle, figure-eight, and three-dimensional helical paths
over multiple randomized robot starts. It saves one row per run plus aggregate
mean and sample-standard-deviation statistics for report-ready analysis.

Example:
    python trajectory_experiments.py --headless --runs 5 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv, TaskSpacePDConfig
from trajectory_following import CONTROL_DT, HOME_Q, move_to_start


OUTPUT_DIR = Path("results/trajectory_study")
DEFAULT_PATHS = ("circle", "figure_eight", "helix")


def controller_config() -> TaskSpacePDConfig:
    return TaskSpacePDConfig(
        kp_position=np.array([150.0, 150.0, 185.0]),
        kd_position=np.array([30.0, 30.0, 34.0]),
        kp_orientation=np.array([30.0, 30.0, 25.0]),
        kd_orientation=np.array([10.0, 10.0, 9.0]),
        nullspace_kp=5.0,
        nullspace_kd=2.0,
        damping=0.05,
    )


def path_samples(
    name: str,
    duration: float,
    dt: float = CONTROL_DT,
    centre_offset: np.ndarray | None = None,
    centre_z: float = 0.68,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, position, and analytic velocity samples for a named path."""
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    centre = np.array([0.47, 0.0, centre_z], dtype=float)
    if centre_offset is not None:
        centre += np.asarray(centre_offset, dtype=float)
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    phase = 2.0 * np.pi * times / duration
    phase_rate = 2.0 * np.pi / duration

    if name == "circle":
        position = centre + np.column_stack((
            0.06 * np.cos(phase),
            0.16 * np.sin(phase),
            np.zeros_like(phase),
        ))
        velocity = np.column_stack((
            -0.06 * phase_rate * np.sin(phase),
            0.16 * phase_rate * np.cos(phase),
            np.zeros_like(phase),
        ))
    elif name == "figure_eight":
        position = centre + np.column_stack((
            0.06 * np.sin(phase),
            0.16 * np.sin(phase) * np.cos(phase),
            0.05 * np.cos(phase),
        ))
        velocity = np.column_stack((
            0.06 * phase_rate * np.cos(phase),
            0.16 * phase_rate * np.cos(2.0 * phase),
            -0.05 * phase_rate * np.sin(phase),
        ))
    elif name == "helix":
        position = centre + np.column_stack((
            0.06 * np.cos(2.0 * phase),
            0.14 * np.sin(2.0 * phase),
            0.06 * np.sin(phase),
        ))
        velocity = np.column_stack((
            -0.12 * phase_rate * np.sin(2.0 * phase),
            0.28 * phase_rate * np.cos(2.0 * phase),
            0.06 * phase_rate * np.cos(phase),
        ))
    else:
        raise ValueError(f"Unknown path {name!r}; expected one of {DEFAULT_PATHS}.")
    return times, position, velocity


def validate_path_reachability(
    env: FrankaPickAndPlaceEnv,
    positions: np.ndarray,
    rotation: np.ndarray,
    preferred_q: np.ndarray,
) -> None:
    """Check representative samples with full-pose IK before running a trial."""
    indices = np.linspace(0, len(positions) - 1, 16, dtype=int)
    q_guess = preferred_q.copy()
    for index in indices:
        result = env.inverse_kinematics(
            target_position=positions[index],
            target_rotation=rotation,
            initial_q=q_guess,
            preferred_q=preferred_q,
            max_iters=500,
            position_tol=2e-3,
            orientation_tol=np.deg2rad(2.0),
        )
        if not result.converged:
            raise RuntimeError(
                f"Path sample {index} is not reachable: "
                f"position error={result.position_error:.4f} m."
            )
        q_guess = result.q


def run_trial(
    path_name: str,
    run_index: int,
    seed: int,
    duration: float,
    render: bool,
    path_overlay: bool = True,
    save_output: bool = True,
    centre_z: float = 0.68,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    env = FrankaPickAndPlaceEnv(render_mode="human" if render else None)
    env.reset(seed=seed)
    start_q = np.clip(
        HOME_Q + rng.uniform(-0.04, 0.04, 7),
        env.model.jnt_range[env.arm_joint_ids, 0],
        env.model.jnt_range[env.arm_joint_ids, 1],
    )
    env.data.qpos[env.arm_qpos_indices] = start_q
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    fixed_rotation = env.get_end_effector_pose().rotation.copy()
    preferred_q = HOME_Q.copy()
    centre_offset = np.array([
        rng.uniform(-0.015, 0.015),
        rng.uniform(-0.015, 0.015),
        rng.uniform(-0.01, 0.01),
    ])
    times, desired_positions, desired_velocities = path_samples(
        path_name, duration, centre_offset=centre_offset, centre_z=centre_z
    )
    validate_path_reachability(env, desired_positions, fixed_rotation, preferred_q)
    if render and path_overlay:
        env.set_viewer_path(desired_positions)
        env.set_viewer_target(desired_positions[0])
    config = controller_config()
    sim_dt = float(env.model.opt.timestep)
    frame_skip = max(1, int(round(CONTROL_DT / sim_dt)))

    actual_positions: list[np.ndarray] = []
    orientation_errors: list[float] = []
    saturated: list[bool] = []
    try:
        settling = move_to_start(
            env,
            desired_positions[0],
            fixed_rotation,
            preferred_q,
            config,
            # Large-workspace paths can retain a small static offset at their
            # boundary. Keep the initialization gate tighter than the 50 mm
            # run-success threshold without rejecting stable 5--8 mm starts.
            position_tolerance=0.008,
        )
        for desired_position, desired_velocity in zip(
            desired_positions, desired_velocities
        ):
            if render and path_overlay:
                env.set_viewer_target(desired_position)
            diagnostics = env.step_task_space_pd(
                target_position=desired_position,
                target_rotation=fixed_rotation,
                target_linear_velocity=desired_velocity,
                target_angular_velocity=np.zeros(3),
                preferred_q=preferred_q,
                config=config,
                frame_skip=frame_skip,
            )
            actual_positions.append(env.get_end_effector_pose().position.copy())
            orientation_errors.append(float(diagnostics["orientation_error"]))
            saturated.append(bool(diagnostics["torque_saturated"]))
            if render:
                time.sleep(CONTROL_DT)
    finally:
        env.close()

    actual = np.asarray(actual_positions)
    position_error = np.linalg.norm(desired_positions - actual, axis=1)
    success_threshold = 0.05
    row = {
        "path": path_name,
        "run": run_index,
        "seed": seed,
        "duration_s": duration,
        "settling_time_s": settling["settling_time"],
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_mean_error_m": float(np.mean(position_error)),
        "position_max_error_m": float(np.max(position_error)),
        "position_final_error_m": float(position_error[-1]),
        "orientation_max_error_rad": float(np.max(orientation_errors)),
        "torque_saturation_rate": float(np.mean(saturated)),
        "success": bool(np.max(position_error) <= success_threshold),
        "success_threshold_m": success_threshold,
        "centre_offset_x": centre_offset[0],
        "centre_offset_y": centre_offset[1],
        "centre_offset_z": centre_offset[2],
    }
    if save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUTPUT_DIR / f"{path_name}_run_{run_index:02d}.npz",
            times=times,
            desired_positions=desired_positions,
            desired_velocities=desired_velocities,
            actual_positions=actual,
            orientation_errors=np.asarray(orientation_errors),
            torque_saturated=np.asarray(saturated),
        )
    print(
        f"{path_name} run {run_index}: RMSE={row['position_rmse_m'] * 1000:.2f} mm, "
        f"max={row['position_max_error_m'] * 1000:.2f} mm, success={row['success']}"
    )
    return row


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "position_rmse_m",
        "position_mean_error_m",
        "position_max_error_m",
        "position_final_error_m",
        "orientation_max_error_rad",
        "torque_saturation_rate",
        "settling_time_s",
    )
    summary: list[dict[str, object]] = []
    for path_name in sorted({str(row["path"]) for row in rows}):
        selected = [row for row in rows if row["path"] == path_name]
        result: dict[str, object] = {
            "path": path_name,
            "runs": len(selected),
            "successes": sum(bool(row["success"]) for row in selected),
            "success_rate": np.mean([bool(row["success"]) for row in selected]),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="runs per trajectory")
    parser.add_argument("--seed", type=int, default=42, help="base experiment seed")
    parser.add_argument("--duration", type=float, default=12.0, help="seconds per path")
    parser.add_argument(
        "--centre-z",
        type=float,
        default=0.68,
        help="vertical centre of the trajectory in metres; use 0.58 for screenshots",
    )
    parser.add_argument("--paths", nargs="+", choices=DEFAULT_PATHS, default=DEFAULT_PATHS)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--hide-path-overlay",
        action="store_true",
        help="do not draw the desired path and current target in the viewer",
    )
    parser.add_argument(
        "--viewer-only",
        action="store_true",
        help="run for screenshots without overwriting saved experiment results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    seed_sequence = np.random.SeedSequence(args.seed)
    trial_count = len(args.paths) * args.runs
    seeds = [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(trial_count)]
    rows: list[dict[str, object]] = []
    seed_index = 0
    for path_name in args.paths:
        for run_index in range(1, args.runs + 1):
            rows.append(run_trial(
                path_name,
                run_index,
                seeds[seed_index],
                args.duration,
                render=not args.headless,
                path_overlay=not args.hide_path_overlay,
                save_output=not args.viewer_only,
                centre_z=args.centre_z,
            ))
            seed_index += 1

    if not args.viewer_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_csv(OUTPUT_DIR / "runs.csv", rows)
        summary = aggregate(rows)
        write_csv(OUTPUT_DIR / "summary.csv", summary)
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
