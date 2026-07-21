"""Numerical validation for the Franka Panda Jacobian and full-pose IK.

Run:
    python validate_kinematics.py

The script:
1. Places the Panda in a valid home configuration.
2. Validates the full 6x7 geometric Jacobian with finite differences.
3. Generates five reproducible reachable target poses near the home posture.
4. Solves full 6-DOF inverse kinematics for every target.
5. Prints and saves quantitative results to CSV.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv
from robot_utils import orientation_error, rotation_angle


# Valid Panda home configuration in radians.
HOME_Q = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
    dtype=float,
)

RESULTS_DIRECTORY = Path("results/logs")
IK_RESULTS_FILE = RESULTS_DIRECTORY / "ik_validation_results.csv"


@dataclass
class IKValidationRow:
    test: int
    converged: bool
    iterations: int
    initial_position_error_m: float
    initial_orientation_error_rad: float
    final_position_error_m: float
    final_orientation_error_rad: float
    joint_limits_respected: bool


def set_arm_configuration(
    env: FrankaPickAndPlaceEnv,
    q: np.ndarray,
) -> None:
    """Set the seven Panda arm joints and update MuJoCo kinematics."""
    q = np.asarray(q, dtype=float)

    if q.shape != (7,):
        raise ValueError(f"Expected q with shape (7,), received {q.shape}.")

    env.data.qpos[env.arm_qpos_indices] = q
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def get_joint_limits(
    env: FrankaPickAndPlaceEnv,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper limits of the seven arm joints."""
    lower = env.model.jnt_range[env.arm_joint_ids, 0].copy()
    upper = env.model.jnt_range[env.arm_joint_ids, 1].copy()
    return lower, upper


def validate_jacobian(
    env: FrankaPickAndPlaceEnv,
    q: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[float, float, float]:
    """Validate the complete 6x7 geometric Jacobian with finite differences.

    Returns:
        maximum_total_error,
        maximum_translation_error,
        maximum_rotation_error
    """
    set_arm_configuration(env, q)

    analytic = env.geometric_jacobian()
    numeric = np.zeros((6, 7), dtype=float)

    base_pose = env.forward_kinematics(q)

    for column in range(7):
        perturbed_q = q.copy()
        perturbed_q[column] += epsilon
        perturbed_pose = env.forward_kinematics(perturbed_q)

        numeric[:3, column] = (
            perturbed_pose.position - base_pose.position
        ) / epsilon

        # Small-angle world-frame rotational displacement.
        numeric[3:, column] = orientation_error(
            base_pose.rotation,
            perturbed_pose.rotation,
        ) / epsilon

    absolute_error = np.abs(analytic - numeric)
    translation_error = float(np.max(absolute_error[:3]))
    rotation_error = float(np.max(absolute_error[3:]))
    total_error = float(np.max(absolute_error))

    print("\n=== Jacobian validation ===")
    print(f"Finite-difference epsilon: {epsilon:.1e}")
    print(f"Maximum translational error: {translation_error:.3e}")
    print(f"Maximum rotational error:    {rotation_error:.3e}")
    print(f"Maximum total error:         {total_error:.3e}")

    return total_error, translation_error, rotation_error


def generate_reachable_targets(
    env: FrankaPickAndPlaceEnv,
    number_of_tests: int = 5,
    seed: int = 7,
    maximum_joint_perturbation: float = 0.35,
) -> list[np.ndarray]:
    """Generate reproducible targets near HOME_Q.

    Each target is produced from a valid joint configuration, so every target
    pose is guaranteed to be kinematically reachable.
    """
    rng = np.random.default_rng(seed)
    lower, upper = get_joint_limits(env)

    safe_margin = 0.05 * (upper - lower)
    safe_lower = lower + safe_margin
    safe_upper = upper - safe_margin

    home_q = np.clip(HOME_Q, safe_lower, safe_upper)
    targets: list[np.ndarray] = []

    for _ in range(number_of_tests):
        perturbation = rng.uniform(
            -maximum_joint_perturbation,
            maximum_joint_perturbation,
            size=7,
        )
        target_q = np.clip(
            home_q + perturbation,
            safe_lower,
            safe_upper,
        )
        targets.append(target_q)

    return targets


def validate_ik(
    env: FrankaPickAndPlaceEnv,
    number_of_tests: int = 5,
    seed: int = 7,
) -> list[IKValidationRow]:
    """Validate full-pose IK on five reproducible reachable poses."""
    lower, upper = get_joint_limits(env)
    home_q = np.clip(HOME_Q, lower, upper)
    target_joint_configurations = generate_reachable_targets(
        env,
        number_of_tests=number_of_tests,
        seed=seed,
    )

    rows: list[IKValidationRow] = []

    print("\n=== Full-pose inverse-kinematics validation ===")
    print(
        "test, converged, iterations, initial_position_error_m, "
        "initial_orientation_error_rad, final_position_error_m, "
        "final_orientation_error_rad, joint_limits_respected"
    )

    initial_pose = env.forward_kinematics(home_q)

    for test_index, target_q in enumerate(
        target_joint_configurations,
        start=1,
    ):
        target_pose = env.forward_kinematics(target_q)

        initial_position_error = float(
            np.linalg.norm(
                target_pose.position - initial_pose.position
            )
        )
        initial_orientation_error = rotation_angle(
            initial_pose.rotation,
            target_pose.rotation,
        )

        result = env.inverse_kinematics(
            target_position=target_pose.position,
            target_rotation=target_pose.rotation,
            initial_q=home_q,
            preferred_q=home_q,
            max_iters=1000,
            position_tol=1e-4,
            orientation_tol=1e-3,
            damping=0.03,
            step_size=0.7,
            max_update_norm=0.20,
            nullspace_gain=0.02,
        )

        solved_pose = env.forward_kinematics(result.q)

        final_position_error = float(
            np.linalg.norm(
                target_pose.position - solved_pose.position
            )
        )
        final_orientation_error = rotation_angle(
            solved_pose.rotation,
            target_pose.rotation,
        )

        joint_limits_respected = bool(
            np.all(result.q >= lower - 1e-10)
            and np.all(result.q <= upper + 1e-10)
        )

        row = IKValidationRow(
            test=test_index,
            converged=result.converged,
            iterations=result.iterations,
            initial_position_error_m=initial_position_error,
            initial_orientation_error_rad=initial_orientation_error,
            final_position_error_m=final_position_error,
            final_orientation_error_rad=final_orientation_error,
            joint_limits_respected=joint_limits_respected,
        )
        rows.append(row)

        print(
            f"{row.test}, "
            f"{row.converged}, "
            f"{row.iterations}, "
            f"{row.initial_position_error_m:.3e}, "
            f"{row.initial_orientation_error_rad:.3e}, "
            f"{row.final_position_error_m:.3e}, "
            f"{row.final_orientation_error_rad:.3e}, "
            f"{row.joint_limits_respected}"
        )

    return rows


def save_ik_results(rows: list[IKValidationRow]) -> None:
    """Save IK validation results as a CSV file."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with IK_RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(asdict(rows[0]).keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"\nIK results saved to: {IK_RESULTS_FILE}")


def print_summary(
    jacobian_error: float,
    rows: list[IKValidationRow],
) -> None:
    """Print aggregate validation metrics."""
    successful_tests = sum(row.converged for row in rows)
    max_position_error = max(
        row.final_position_error_m for row in rows
    )
    mean_position_error = float(
        np.mean([row.final_position_error_m for row in rows])
    )
    max_orientation_error = max(
        row.final_orientation_error_rad for row in rows
    )
    mean_orientation_error = float(
        np.mean([row.final_orientation_error_rad for row in rows])
    )

    print("\n=== Validation summary ===")
    print(f"Jacobian maximum error:          {jacobian_error:.3e}")
    print(f"IK successful tests:             {successful_tests}/{len(rows)}")
    print(f"Maximum final position error:    {max_position_error:.3e} m")
    print(f"Mean final position error:       {mean_position_error:.3e} m")
    print(f"Maximum final orientation error: {max_orientation_error:.3e} rad")
    print(f"Mean final orientation error:    {mean_orientation_error:.3e} rad")

    if successful_tests == len(rows):
        print("Result: all IK tests converged.")
    else:
        print(
            "Result: at least one IK test did not converge. "
            "Inspect the per-test errors above."
        )


def main() -> None:
    env = FrankaPickAndPlaceEnv(render_mode=None)

    try:
        env.reset()

        lower, upper = get_joint_limits(env)
        home_q = np.clip(HOME_Q, lower, upper)
        set_arm_configuration(env, home_q)

        jacobian_error, _, _ = validate_jacobian(
            env,
            q=home_q,
            epsilon=1e-6,
        )

        rows = validate_ik(
            env,
            number_of_tests=5,
            seed=7,
        )

        save_ik_results(rows)
        print_summary(jacobian_error, rows)

    finally:
        env.close()


if __name__ == "__main__":
    main()