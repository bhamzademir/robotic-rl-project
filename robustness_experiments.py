"""Unmodeled payload and external-disturbance rejection tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controller_experiment_utils import aggregate, initialized_environment, settle_at, write_csv
from trajectory_following import CONTROL_DT


OUTPUT_DIR = Path("results/controller_limits/robustness")
PAYLOADS_KG = (0.0, 0.5, 1.0, 2.0)
DISTURBANCE_FORCES_N = (10.0, 20.0, 40.0)


def run_hold(seed: int, payload_kg: float, force_n: float,
             disturbance_start: float, disturbance_duration: float,
             total_duration: float) -> dict[str, object]:
    env, rotation, preferred_q, config, frame_skip = initialized_environment(seed)
    target = np.array([0.47, 0.0, 0.68])
    errors = []
    saturated = []
    try:
        settle_at(env, target, rotation, preferred_q, config)
        for step in range(int(round(total_duration / CONTROL_DT))):
            t = step * CONTROL_DT
            # Downward world-frame force represents an extra payload unknown to
            # the controller model; a lateral pulse tests disturbance rejection.
            wrench = np.zeros(6)
            wrench[2] = -payload_kg * 9.81
            if disturbance_start <= t < disturbance_start + disturbance_duration:
                wrench[1] += force_n
            env.data.xfrc_applied[env.end_effector_body_id] = wrench
            diagnostics = env.step_task_space_pd(
                target, rotation, preferred_q=preferred_q,
                config=config, frame_skip=frame_skip,
            )
            errors.append(float(np.linalg.norm(env.get_end_effector_pose().position - target)))
            saturated.append(bool(diagnostics["torque_saturated"]))
    finally:
        env.close()
    errors_array = np.asarray(errors)
    pulse_end = int(round((disturbance_start + disturbance_duration) / CONTROL_DT))
    recovery_threshold = 0.01
    recovery_time = np.nan
    stable_samples = max(1, int(round(0.2 / CONTROL_DT)))
    for index in range(pulse_end, len(errors_array) - stable_samples + 1):
        if np.all(errors_array[index:index + stable_samples] <= recovery_threshold):
            recovery_time = (index - pulse_end) * CONTROL_DT
            break
    tail = max(1, int(round(0.5 / CONTROL_DT)))
    return {"payload_kg": payload_kg, "disturbance_force_n": force_n, "seed": seed,
            "peak_displacement_m": float(np.max(errors_array)),
            "recovery_time_s": float(recovery_time),
            "residual_error_m": float(np.mean(errors_array[-tail:])),
            "error_std_m": float(np.std(errors_array[-tail:], ddof=1)),
            "torque_saturation_rate": float(np.mean(saturated))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--disturbance-start", type=float, default=2.0)
    parser.add_argument("--disturbance-duration", type=float, default=0.25)
    args = parser.parse_args()
    conditions = [(payload, 0.0) for payload in PAYLOADS_KG]
    conditions += [(0.0, force) for force in DISTURBANCE_FORCES_N]
    sequence = np.random.SeedSequence(args.seed)
    seeds = [int(s.generate_state(1)[0]) for s in sequence.spawn(len(conditions) * args.trials)]
    rows = []
    index = 0
    for payload, force in conditions:
        for _ in range(args.trials):
            row = run_hold(seeds[index], payload, force, args.disturbance_start,
                           args.disturbance_duration, args.duration)
            index += 1; rows.append(row); print(row)
    write_csv(OUTPUT_DIR / "runs.csv", rows)
    summary = aggregate(rows, ("payload_kg", "disturbance_force_n"),
        ("peak_displacement_m", "recovery_time_s", "residual_error_m",
         "error_std_m", "torque_saturation_rate"))
    write_csv(OUTPUT_DIR / "summary.csv", summary)


if __name__ == "__main__":
    main()
