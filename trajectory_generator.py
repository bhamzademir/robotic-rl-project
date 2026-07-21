"""Smooth task-space trajectory generation using a quintic time scaling."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from robot_utils import matrix_to_quaternion, quaternion_to_matrix, slerp


@dataclass
class TrajectorySample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    rotation: np.ndarray


def quintic_scaling(t: float, duration: float) -> tuple[float, float, float]:
    """Return s, s_dot and s_ddot with zero endpoint velocity/acceleration."""
    tau = np.clip(t / duration, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
    return float(s), float(sd), float(sdd)


def interpolate_pose(
    start_position: np.ndarray,
    end_position: np.ndarray,
    start_rotation: np.ndarray,
    end_rotation: np.ndarray,
    t: float,
    duration: float,
) -> TrajectorySample:
    s, sd, sdd = quintic_scaling(t, duration)
    displacement = np.asarray(end_position) - np.asarray(start_position)
    position = start_position + s * displacement
    velocity = sd * displacement
    acceleration = sdd * displacement
    quaternion = slerp(
        matrix_to_quaternion(start_rotation), matrix_to_quaternion(end_rotation), s
    )
    return TrajectorySample(position, velocity, acceleration, quaternion_to_matrix(quaternion))
