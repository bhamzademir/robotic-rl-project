"""Shared utilities for the controller-limit experiment scripts."""
from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv, TaskSpacePDConfig
from trajectory_experiments import controller_config
from trajectory_following import CONTROL_DT, HOME_Q, move_to_start


def initialized_environment(seed: int) -> tuple[
    FrankaPickAndPlaceEnv, np.ndarray, np.ndarray, TaskSpacePDConfig, int
]:
    """Return a reproducibly initialized environment at a safe test pose."""
    rng = np.random.default_rng(seed)
    env = FrankaPickAndPlaceEnv(render_mode=None)
    env.reset(seed=seed)
    lower = env.model.jnt_range[env.arm_joint_ids, 0]
    upper = env.model.jnt_range[env.arm_joint_ids, 1]
    env.data.qpos[env.arm_qpos_indices] = np.clip(
        HOME_Q + rng.uniform(-0.025, 0.025, 7), lower, upper
    )
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    rotation = env.get_end_effector_pose().rotation.copy()
    preferred_q = HOME_Q.copy()
    config = controller_config()
    frame_skip = max(1, int(round(CONTROL_DT / float(env.model.opt.timestep))))
    return env, rotation, preferred_q, config, frame_skip


def settle_at(
    env: FrankaPickAndPlaceEnv,
    position: np.ndarray,
    rotation: np.ndarray,
    preferred_q: np.ndarray,
    config: TaskSpacePDConfig,
) -> dict[str, float]:
    return move_to_start(
        env,
        position,
        rotation,
        preferred_q,
        config,
        position_tolerance=0.008,
    )


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a rotation matrix for a world-frame axis-angle rotation."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, object]], group_fields: tuple[str, ...], metrics: tuple[str, ...]
) -> list[dict[str, object]]:
    """Compute sample mean and standard deviation for numeric metrics."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, selected in groups.items():
        result: dict[str, object] = dict(zip(group_fields, key))
        result["runs"] = len(selected)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            result[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
            )
        output.append(result)
    return output
