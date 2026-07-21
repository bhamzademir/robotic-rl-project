"""Kinematics and rotation utilities for the Franka Panda MuJoCo project."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray


def skew_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the vee operator of a 3x3 skew-symmetric matrix."""
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]], dtype=float)


def orientation_error(current_rotation: np.ndarray, desired_rotation: np.ndarray) -> np.ndarray:
    """Rotation-matrix orientation error expressed in the world frame."""
    error_local = 0.5 * (
        current_rotation.T @ desired_rotation
        - desired_rotation.T @ current_rotation
    )
    return current_rotation @ skew_to_vector(error_local)


def rotation_angle(current_rotation: np.ndarray, desired_rotation: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices in radians."""
    relative = current_rotation.T @ desired_rotation
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to MuJoCo quaternion format [w, x, y, z]."""
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).reshape(9))
    return quaternion


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    return rotation.reshape(3, 3)


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Spherical interpolation between unit quaternions in [w, x, y, z] format."""
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - fraction) * theta) / sin_theta * q0
        + np.sin(fraction * theta) / sin_theta * q1
    )