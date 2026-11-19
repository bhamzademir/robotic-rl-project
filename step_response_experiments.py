"""Position and orientation step-response tests for the task-space controller."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controller_experiment_utils import (
    aggregate,
    axis_angle_rotation,
    initialized_environment,
    settle_at,
    write_csv,
)
from trajectory_following import CONTROL_DT


OUTPUT_DIR = Path("results/controller_limits/step_response")
POSITION_STEPS_M = (0.02, 0.05, 0.10)
ORIENTATION_STEPS_DEG = (5.0, 15.0, 30.0)


def response_metrics(response: np.ndarray, target: float, dt: float) -> dict[str, float]:
    """Measure rise time, overshoot, settling time, and steady-state error."""
    direction = 1.0 if target >= 0.0 else -1.0
    normalized = direction * response
    magnitude = abs(target)
    above_10 = np.flatnonzero(normalized >= 0.1 * magnitude)
    above_90 = np.flatnonzero(normalized >= 0.9 * magnitude)
    rise_time = np.nan
    if above_10.size and above_90.size:
        rise_time = max(0.0, (above_90[0] - above_10[0]) * dt)
    overshoot = max(0.0, (float(np.max(normalized)) - magnitude) / magnitude * 100.0)
    error = np.abs(target - response)
    tolerance = max(0.02 * magnitude, 0.001 if magnitude < 0.2 else np.deg2rad(0.5))
    settling_time = np.nan
    for index in range(len(error)):
        if np.all(error[index:] <= tolerance):
            settling_time = index * dt
            break
    tail = max(1, int(round(0.5 / dt)))
    return {
        "rise_time_s": float(rise_time),
        "overshoot_percent": float(overshoot),
        "settling_time_s": float(settling_time),
        "steady_state_error": float(np.mean(error[-tail:])),
    }


def run_position(seed: int, magnitude: float, duration: float) -> dict[str, object]:
    env, rotation, preferred_q, config, frame_skip = initialized_environment(seed)
    start = np.array([0.47, 0.0, 0.68])
    direction = np.array([0.0, 1.0, 0.0])
    try:
        settle_at(env, start, rotation, preferred_q, config)
        actual = []
        saturation = []
        for _ in range(int(round(duration / CONTROL_DT))):
            diagnostics = env.step_task_space_pd(
                start + magnitude * direction, rotation,
                preferred_q=preferred_q, config=config, frame_skip=frame_skip,
            )
            actual.append(float((env.get_end_effector_pose().position - start) @ direction))
            saturation.append(bool(diagnostics["torque_saturated"]))
    finally:
        env.close()
    metrics = response_metrics(np.asarray(actual), magnitude, CONTROL_DT)
    return {"test": "position", "magnitude": magnitude, "seed": seed, **metrics,
            "steady_state_error_si": metrics["steady_state_error"],
            "torque_saturation_rate": float(np.mean(saturation))}


def run_orientation(seed: int, degrees: float, duration: float) -> dict[str, object]:
    env, rotation, preferred_q, config, frame_skip = initialized_environment(seed)
    start = np.array([0.47, 0.0, 0.68])
    target_angle = np.deg2rad(degrees)
    axis = np.array([0.0, 0.0, 1.0])
    target_rotation = axis_angle_rotation(axis, target_angle) @ rotation
    try:
        settle_at(env, start, rotation, preferred_q, config)
        response = []
        saturation = []
        for _ in range(int(round(duration / CONTROL_DT))):
            diagnostics = env.step_task_space_pd(
                start, target_rotation, preferred_q=preferred_q,
                config=config, frame_skip=frame_skip,
            )
            relative = env.get_end_effector_pose().rotation @ rotation.T
            response.append(float(np.arctan2(relative[1, 0], relative[0, 0])))
            saturation.append(bool(diagnostics["torque_saturated"]))
    finally:
        env.close()
    metrics = response_metrics(np.unwrap(response), target_angle, CONTROL_DT)
    return {"test": "orientation", "magnitude": degrees, "seed": seed, **metrics,
            "steady_state_error_si": metrics["steady_state_error"],
            "torque_saturation_rate": float(np.mean(saturation))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    sequence = np.random.SeedSequence(args.seed)
    seeds = [int(s.generate_state(1)[0]) for s in sequence.spawn(args.trials * 6)]
    rows = []
    index = 0
    for magnitude in POSITION_STEPS_M:
        for _ in range(args.trials):
            rows.append(run_position(seeds[index], magnitude, args.duration)); index += 1
    for magnitude in ORIENTATION_STEPS_DEG:
        for _ in range(args.trials):
            rows.append(run_orientation(seeds[index], magnitude, args.duration)); index += 1
    write_csv(OUTPUT_DIR / "runs.csv", rows)
    summary = aggregate(rows, ("test", "magnitude"),
        ("rise_time_s", "overshoot_percent", "settling_time_s",
         "steady_state_error_si", "torque_saturation_rate"))
    write_csv(OUTPUT_DIR / "summary.csv", summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
