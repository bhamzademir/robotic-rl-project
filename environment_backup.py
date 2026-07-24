"""Gymnasium environment and classical pose-control interface for a Franka Panda."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

from robot_utils import Pose, orientation_error, rotation_angle


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float


class FrankaPickAndPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None, scene_path: str = "scene.xml"):
        super().__init__()
        self.render_mode = render_mode
        self.scene_path = str(Path(scene_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # Classical controller input: desired Cartesian position and quaternion.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )

        self.arm_joint_ids = self._find_arm_joint_ids()
        self.arm_qpos_indices = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids], dtype=int
        )
        self.arm_dof_indices = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids], dtype=int
        )
        self.end_effector_body_id = self._get_end_effector_id()
        self.cube_body_id = self._body_id("cube")
        self.target_body_id = self._body_id("target_marker")

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' was not found in the MuJoCo model.")
        return body_id

    def _find_arm_joint_ids(self) -> np.ndarray:
        ids = []
        preferred_names = [f"joint{i}" for i in range(1, 8)] + [f"panda_joint{i}" for i in range(1, 8)]
        for name in preferred_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0 and joint_id not in ids:
                ids.append(joint_id)
        if len(ids) < 7:
            ids = [
                joint_id
                for joint_id in range(self.model.njnt)
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
            ][:7]
        if len(ids) != 7:
            raise RuntimeError(f"Expected seven Panda arm joints, found {len(ids)}.")
        return np.asarray(ids, dtype=int)

    def _get_end_effector_id(self) -> int:
        for name in ["panda_hand", "hand", "link7", "attachment"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        raise RuntimeError("Could not identify the Panda end-effector body.")

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_indices].copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.arm_dof_indices].copy()

    def get_end_effector_pose(self) -> Pose:
        position = self.data.xpos[self.end_effector_body_id].copy()
        rotation = self.data.xmat[self.end_effector_body_id].reshape(3, 3).copy()
        return Pose(position, rotation)

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> Pose:
        """Return MuJoCo's end-effector pose for q without permanently changing state."""
        if q is None:
            mujoco.mj_forward(self.model, self.data)
            return self.get_end_effector_pose()
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError("q must have shape (7,).")
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        try:
            self.data.qpos[self.arm_qpos_indices] = q
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return self.get_end_effector_pose()
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def geometric_jacobian(self) -> np.ndarray:
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(
            self.model, self.data, jac_pos, jac_rot, self.end_effector_body_id
        )
        return np.vstack(
            [jac_pos[:, self.arm_dof_indices], jac_rot[:, self.arm_dof_indices]]
        )

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        initial_q: Optional[np.ndarray] = None,
        max_iters: int = 300,
        position_tol: float = 1e-4,
        orientation_tol: float = 1e-3,
        damping: float = 0.05,
        step_size: float = 0.35,
        max_update_norm: float = 0.15,
        preferred_q: Optional[np.ndarray] = None,
        nullspace_gain: float = 0.05,
    ) -> IKResult:
        """Solve full 6D pose IK with damped least squares and joint limits."""
        target_position = np.asarray(target_position, dtype=float)
        target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
        q = self.get_joint_positions() if initial_q is None else np.asarray(initial_q, dtype=float).copy()
        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        q = np.clip(q, lower, upper)
        preferred_q = q.copy() if preferred_q is None else np.asarray(preferred_q, dtype=float)


        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        converged = False
        pos_norm = np.inf
        rot_norm = np.inf

        try:
            for iteration in range(1, max_iters + 1):
                self.data.qpos[self.arm_qpos_indices] = q
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)

                pose = self.get_end_effector_pose()
                pos_error = target_position - pose.position
                rot_error = orientation_error(pose.rotation, target_rotation)
                pos_norm = float(np.linalg.norm(pos_error))
                rot_norm = rotation_angle(pose.rotation, target_rotation)

                if pos_norm <= position_tol and rot_norm <= orientation_tol:
                    converged = True
                    break

                error = np.concatenate([pos_error, rot_error])
                jacobian = self.geometric_jacobian()
                regularized = jacobian @ jacobian.T + damping**2 * np.eye(6)
                jacobian_pinv = jacobian.T @ np.linalg.solve(regularized, np.eye(6))
                null_projector = np.eye(7) - jacobian_pinv @ jacobian
                delta_q = jacobian_pinv @ error + nullspace_gain * null_projector @ (preferred_q - q)

                norm = np.linalg.norm(delta_q)
                if norm > max_update_norm:
                    delta_q *= max_update_norm / norm
                q = np.clip(q + step_size * delta_q, lower, upper)
            else:
                iteration = max_iters
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

        return IKResult(q, converged, iteration, pos_norm, rot_norm)

    def set_joint_position_targets(self, desired_q: np.ndarray) -> None:
        """Send desired joint positions to position actuators."""
        desired_q = np.asarray(desired_q, dtype=float)
        if self.model.nu < 7:
            raise RuntimeError("The loaded Panda model does not expose seven arm actuators.")
        self.data.ctrl[:7] = desired_q

    def _get_obs(self) -> np.ndarray:
        pose = self.get_end_effector_pose()
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, pose.rotation.reshape(9))
        cube_pos = self.data.xpos[self.cube_body_id]
        target_pos = self.data.xpos[self.target_body_id]
        return np.concatenate(
            [
                self.get_joint_positions(),
                self.get_joint_velocities(),
                pose.position,
                quaternion,
                cube_pos,
                target_pos,
                cube_pos - pose.position,
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        home_q = np.array(
            [
                0.0,
                0.0,
                0.0,
                -1.57079,
                0.0,
                1.57079,
                -0.7853,
            ],
            dtype=float,
        )

        self.data.qpos[self.arm_qpos_indices] = home_q
        self.data.qvel[:] = 0.0

        if self.model.nu >= 7:
            self.data.ctrl[:7] = home_q

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
            action = np.asarray(action, dtype=float)
            target_position = action[:3]
            quaternion = action[3:7]
            quaternion /= max(np.linalg.norm(quaternion), 1e-12)
            target_rotation_flat = np.empty(9)
            mujoco.mju_quat2Mat(target_rotation_flat, quaternion)
            result = self.inverse_kinematics(target_position, target_rotation_flat.reshape(3, 3))
            self.set_joint_position_targets(result.q)
            mujoco.mj_step(self.model, self.data)

            pose = self.get_end_effector_pose()
            reward = -float(np.linalg.norm(target_position - pose.position))
            terminated = bool(self.data.xpos[self.cube_body_id, 2] < 0.2)
            info = {
                "ik_converged": result.converged,
                "ik_iterations": result.iterations,
                "position_error": result.position_error,
                "orientation_error": result.orientation_error,
            }
            if self.render_mode == "human":
                self.render()
            return self._get_obs(), reward, terminated, False, info

    def render(self):
            if self.render_mode == "human":
                if self.viewer is None:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.sync()

    def close(self):
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None