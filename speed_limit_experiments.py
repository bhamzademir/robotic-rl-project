"""Increasing-speed trajectory tests for velocity and torque limits."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controller_experiment_utils import aggregate, initialized_environment, settle_at, write_csv
from trajectory_experiments import path_samples, validate_path_reachability
from trajectory_following import CONTROL_DT


OUTPUT_DIR = Path("results/controller_limits/speed_limit")
DEFAULT_DURATIONS = (16.0, 12.0, 8.0, 6.0, 4.0)
JOINT_VELOCITY_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


def run_trial(path: str, duration: float, seed: int) -> dict[str, object]:
    env, rotation, preferred_q, config, frame_skip = initialized_environment(seed)
    rng = np.random.default_rng(seed)
    offset = rng.uniform([-0.01, -0.01, -0.006], [0.01, 0.01, 0.006])
    times, desired, velocities = path_samples(path, duration, centre_offset=offset)
    validate_path_reachability(env, desired, rotation, preferred_q)
    actual = []
    saturated = []
    qvel_ratio = []
    try:
        settle_at(env, desired[0], rotation, preferred_q, config)
        for target, velocity in zip(desired, velocities):
            diagnostics = env.step_task_space_pd(
                target, rotation, target_linear_velocity=velocity,
                preferred_q=preferred_q, config=config, frame_skip=frame_skip,
            )
            actual.append(env.get_end_effector_pose().position.copy())
            saturated.append(bool(diagnostics["torque_saturated"]))
            qvel_ratio.append(float(np.max(np.abs(env.get_joint_velocities()) / JOINT_VELOCITY_LIMITS)))
    finally:
        env.close()
    errors = np.linalg.norm(desired - np.asarray(actual), axis=1)
    max_error = float(np.max(errors))
    saturation_rate = float(np.mean(saturated))
    max_velocity_ratio = float(np.max(qvel_ratio))
    acceptable = max_error <= 0.05 and saturation_rate <= 0.01 and max_velocity_ratio <= 1.0
    return {"path": path, "duration_s": duration, "speed_scale": 12.0 / duration,
            "seed": seed, "position_rmse_m": float(np.sqrt(np.mean(errors**2))),
            "position_max_error_m": max_error, "torque_saturation_rate": saturation_rate,
            "max_joint_velocity_ratio": max_velocity_ratio, "acceptable": acceptable}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paths", nargs="+", choices=("circle", "figure_eight"),
                        default=("circle", "figure_eight"))
    parser.add_argument("--durations", nargs="+", type=float, default=DEFAULT_DURATIONS)
    args = parser.parse_args()
    sequence = np.random.SeedSequence(args.seed)
    count = len(args.paths) * len(args.durations) * args.trials
    seeds = [int(s.generate_state(1)[0]) for s in sequence.spawn(count)]
    rows = []
    index = 0
    for path in args.paths:
        for duration in args.durations:
            for _ in range(args.trials):
                row = run_trial(path, duration, seeds[index]); index += 1
                rows.append(row); print(row)
    write_csv(OUTPUT_DIR / "runs.csv", rows)
    summary = aggregate(rows, ("path", "duration_s", "speed_scale"),
        ("position_rmse_m", "position_max_error_m", "torque_saturation_rate",
         "max_joint_velocity_ratio", "acceptable"))
    write_csv(OUTPUT_DIR / "summary.csv", summary)


if __name__ == "__main__":
    main()
