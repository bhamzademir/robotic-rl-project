"""Gymnasium environment and classical pose-control interface for a Franka Panda."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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


@dataclass
class TaskSpacePDConfig:
    """Gains and safety limits for the torque-level task-space PD controller."""

    kp_position: np.ndarray = field(
        default_factory=lambda: np.array([180.0, 180.0, 220.0], dtype=float)
    )
    kd_position: np.ndarray = field(
        default_factory=lambda: np.array([28.0, 28.0, 32.0], dtype=float)
    )
    kp_orientation: np.ndarray = field(
        default_factory=lambda: np.array([45.0, 45.0, 35.0], dtype=float)
    )
    kd_orientation: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 8.0, 7.0], dtype=float)
    )
    nullspace_kp: float = 18.0
    nullspace_kd: float = 5.0
    damping: float = 0.05
    torque_limits: np.ndarray = field(
        default_factory=lambda: np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    )


class FrankaPickAndPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        scene_path: str = "scene.xml",
        control_mode: str = "pose",
        randomize_cube: bool = False,
        randomize_robot_start: bool = False,
        episode_length: int = 300,
        action_scale: float = 0.03,
        frame_skip: int = 5,
    ):
        """Create the Panda environment.

        Args:
            control_mode:
                "pose" keeps the original classical-control interface:
                [x, y, z, qw, qx, qy, qz].

                "delta" provides the RL interface requested for PPO/SAC:
                [dx, dy, dz, gripper] in [-1, 1]. Cartesian deltas are
                multiplied by ``action_scale`` before being sent to IK.
            randomize_cube:
                Randomize the cube XY position at reset.
            randomize_robot_start:
                Randomize the seven arm joint positions around the nominal
                initial posture at reset.
        """
        super().__init__()
        if control_mode not in {"pose", "delta"}:
            raise ValueError("control_mode must be either 'pose' or 'delta'.")

        self.render_mode = render_mode
        self.control_mode = control_mode
        self.randomize_cube = bool(randomize_cube)
        self.randomize_robot_start = bool(randomize_robot_start)
        self.episode_length = int(episode_length)
        self.action_scale = float(action_scale)
        self.frame_skip = max(1, int(frame_skip))

        scene = Path(scene_path)
        if not scene.is_absolute():
            scene = Path(__file__).resolve().parent / scene
        self.scene_path = str(scene.resolve())

        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.arm_joint_ids = self._find_arm_joint_ids()
        self.arm_qpos_indices = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids],
            dtype=int,
        )
        self.arm_dof_indices = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids],
            dtype=int,
        )
        self.end_effector_body_id = self._get_end_effector_id()
        self.cube_body_id = self._body_id("cube")
        self.target_body_id = self._body_id("target_marker")

        self.gripper_actuator_id = self._find_gripper_actuator_id()
        self.finger_joint_ids = self._find_finger_joint_ids()

        # The cube uses a MuJoCo free joint. Writing qpos, rather than xpos,
        # makes reset randomization physically valid and persistent.
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        if cube_joint_id < 0:
            raise RuntimeError("Expected free joint 'cube_joint' for cube randomization.")
        self.cube_joint_id = int(cube_joint_id)
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])

        # Safe task-space and reset ranges for the tabletop scene.
        self.workspace_low = np.array([0.35, -0.30, 0.43], dtype=float)
        self.workspace_high = np.array([0.70, 0.30, 0.85], dtype=float)
        self.cube_xy_low = np.array([0.46, -0.18], dtype=float)
        self.cube_xy_high = np.array([0.64, 0.18], dtype=float)

        # The nominal posture is captured from MuJoCo's reset state so the
        # randomization stays compatible with the loaded Panda model.
        mujoco.mj_resetData(self.model, self.data)
        self.default_arm_qpos = self.get_joint_positions()
        self.default_cube_position = np.array([0.55, 0.0, 0.42], dtype=float)

        # Keep the original 30-D observation used by the classical-control
        # project: q, qdot, EE position+quaternion, cube, target, cube-EE.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )

        if self.control_mode == "pose":
            # Original classical controller input.
            self.action_space = spaces.Box(
                low=np.array(
                    [-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0],
                    dtype=np.float32,
                ),
                high=np.array(
                    [1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
        else:
            # PPO/SAC action: normalized Cartesian delta + gripper command.
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )

        self.step_count = 0
        self.previous_action = np.zeros(self.action_space.shape, dtype=float)

        self._direct_torque_control_enabled = False
        self._saved_actuator_gainprm = self.model.actuator_gainprm[:7].copy()
        self._saved_actuator_biasprm = self.model.actuator_biasprm[:7].copy()

        mujoco.mj_forward(self.model, self.data)

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

    def _find_gripper_actuator_id(self) -> Optional[int]:
        """Return the gripper actuator ID, or None when the model has no gripper."""
        preferred_names = [
            "actuator8",
            "gripper",
            "gripper_actuator",
            "finger",
            "fingers",
        ]
        for name in preferred_names:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if actuator_id >= 0:
                return int(actuator_id)

        # The standard Menagerie Panda has seven arm actuators followed by one
        # gripper tendon actuator.
        if self.model.nu > 7:
            return int(self.model.nu - 1)
        return None

    def _find_finger_joint_ids(self) -> np.ndarray:
        """Find Panda finger joints for width diagnostics."""
        ids: list[int] = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if name and "finger" in name.lower():
                ids.append(joint_id)
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
        preferred_q = q.copy() if preferred_q is None else np.asarray(preferred_q, dtype=float)

        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
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

    def set_gripper_opening(self, opening: float) -> float:
        """Command normalized gripper opening.

        Args:
            opening: 0.0 means fully closed and 1.0 means fully open.

        Returns:
            The actuator control value sent to MuJoCo.

        The standard Menagerie Panda gripper actuator uses a control range such
        as [0, 255], where the upper end corresponds to the open gripper.
        """
        if self.gripper_actuator_id is None:
            raise RuntimeError("The loaded model does not expose a gripper actuator.")

        opening = float(np.clip(opening, 0.0, 1.0))
        actuator_id = self.gripper_actuator_id

        if bool(self.model.actuator_ctrllimited[actuator_id]):
            lower, upper = self.model.actuator_ctrlrange[actuator_id]
            command = lower + opening * (upper - lower)
        else:
            # Fallback for an unlimited actuator.
            command = opening

        self.data.ctrl[actuator_id] = command
        return float(command)

    def open_gripper(self) -> float:
        """Fully open the gripper."""
        return self.set_gripper_opening(1.0)

    def close_gripper(self) -> float:
        """Fully close the gripper."""
        return self.set_gripper_opening(0.0)

    def get_gripper_opening(self) -> float:
        """Return an approximate physical finger opening in metres."""
        if self.finger_joint_ids.size == 0:
            return float("nan")
        qpos_indices = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.finger_joint_ids],
            dtype=int,
        )
        # Each finger joint stores one lateral displacement. Their sum is the
        # approximate distance between both fingers.
        return float(np.sum(np.abs(self.data.qpos[qpos_indices])))

    def get_cube_position(self) -> np.ndarray:
        """Return the cube centre position in the world frame."""
        return self.data.xpos[self.cube_body_id].copy()

    def get_end_effector_twist(self) -> np.ndarray:
        """Return [linear_velocity, angular_velocity] in the world frame."""
        return self.geometric_jacobian() @ self.get_joint_velocities()

    def task_space_pd_torque(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        target_linear_velocity: Optional[np.ndarray] = None,
        target_angular_velocity: Optional[np.ndarray] = None,
        preferred_q: Optional[np.ndarray] = None,
        config: Optional[TaskSpacePDConfig] = None,
    ) -> tuple[np.ndarray, dict]:
        """Compute a 6D task-space PD/impedance torque command.

        The controller forms a Cartesian wrench from pose and twist errors,
        maps it to joint torques with J.T, adds MuJoCo bias-force
        compensation, and adds a damped null-space posture term.
        """
        config = TaskSpacePDConfig() if config is None else config
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
        target_linear_velocity = (
            np.zeros(3)
            if target_linear_velocity is None
            else np.asarray(target_linear_velocity, dtype=float).reshape(3)
        )
        target_angular_velocity = (
            np.zeros(3)
            if target_angular_velocity is None
            else np.asarray(target_angular_velocity, dtype=float).reshape(3)
        )

        mujoco.mj_forward(self.model, self.data)
        pose = self.get_end_effector_pose()
        jacobian = self.geometric_jacobian()
        q = self.get_joint_positions()
        qd = self.get_joint_velocities()
        twist = jacobian @ qd

        position_error = target_position - pose.position
        rotation_error = orientation_error(pose.rotation, target_rotation)
        linear_velocity_error = target_linear_velocity - twist[:3]
        angular_velocity_error = target_angular_velocity - twist[3:]

        force = (
            config.kp_position * position_error
            + config.kd_position * linear_velocity_error
        )
        moment = (
            config.kp_orientation * rotation_error
            + config.kd_orientation * angular_velocity_error
        )
        task_wrench = np.concatenate([force, moment])
        task_torque = jacobian.T @ task_wrench

        # Damped projector keeps the secondary posture command from strongly
        # disturbing the primary six-dimensional end-effector task.
        regularized = jacobian @ jacobian.T + config.damping**2 * np.eye(6)
        jacobian_pinv = jacobian.T @ np.linalg.solve(regularized, np.eye(6))
        null_projector = np.eye(7) - jacobian_pinv @ jacobian
        if preferred_q is None:
            preferred_q = q.copy()
        else:
            preferred_q = np.asarray(preferred_q, dtype=float).reshape(7)
        posture_torque = null_projector.T @ (
            config.nullspace_kp * (preferred_q - q)
            - config.nullspace_kd * qd
        )

        # qfrc_bias contains gravity, Coriolis, and centrifugal bias forces.
        bias_torque = self.data.qfrc_bias[self.arm_dof_indices].copy()
        raw_torque = task_torque + posture_torque + bias_torque
        torque = np.clip(raw_torque, -config.torque_limits, config.torque_limits)

        diagnostics = {
            "position_error_vector": position_error.copy(),
            "orientation_error_vector": rotation_error.copy(),
            "position_error": float(np.linalg.norm(position_error)),
            "orientation_error": float(rotation_angle(pose.rotation, target_rotation)),
            "linear_velocity": twist[:3].copy(),
            "angular_velocity": twist[3:].copy(),
            "task_wrench": task_wrench.copy(),
            "raw_torque": raw_torque.copy(),
            "torque": torque.copy(),
            "torque_saturated": bool(np.any(np.abs(raw_torque) > config.torque_limits)),
        }
        return torque, diagnostics

    def enable_direct_torque_control(self) -> None:
        """Disable the seven position servos so qfrc_applied is the only arm command."""
        if self._direct_torque_control_enabled:
            return
        self.model.actuator_gainprm[:7] = 0.0
        self.model.actuator_biasprm[:7] = 0.0
        self.data.ctrl[:7] = 0.0
        self._direct_torque_control_enabled = True

    def disable_direct_torque_control(self) -> None:
        """Restore the original Panda position-servo parameters."""
        if not self._direct_torque_control_enabled:
            return
        self.model.actuator_gainprm[:7] = self._saved_actuator_gainprm
        self.model.actuator_biasprm[:7] = self._saved_actuator_biasprm
        self.data.qfrc_applied[self.arm_dof_indices] = 0.0
        self._direct_torque_control_enabled = False

    def apply_arm_torques(self, torque: np.ndarray) -> None:
        """Apply seven arm torques after disabling the original position servos."""
        torque = np.asarray(torque, dtype=float).reshape(7)
        self.enable_direct_torque_control()
        self.data.qfrc_applied[self.arm_dof_indices] = torque

    def step_task_space_pd(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        target_linear_velocity: Optional[np.ndarray] = None,
        target_angular_velocity: Optional[np.ndarray] = None,
        preferred_q: Optional[np.ndarray] = None,
        config: Optional[TaskSpacePDConfig] = None,
        frame_skip: int = 1,
    ) -> dict:
        """Advance the simulation using torque-level 6D task-space PD control."""
        torque, diagnostics = self.task_space_pd_torque(
            target_position=target_position,
            target_rotation=target_rotation,
            target_linear_velocity=target_linear_velocity,
            target_angular_velocity=target_angular_velocity,
            preferred_q=preferred_q,
            config=config,
        )
        self.apply_arm_torques(torque)
        for _ in range(max(1, int(frame_skip))):
            mujoco.mj_step(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return diagnostics

    def _set_cube_position(self, xyz: np.ndarray) -> None:
        """Set cube free-joint pose without writing derived xpos fields."""
        xyz = np.asarray(xyz, dtype=float).reshape(3)
        adr = self.cube_qpos_adr
        self.data.qpos[adr : adr + 3] = xyz
        # MuJoCo free-joint quaternion order is w, x, y, z.
        self.data.qpos[adr + 3 : adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        )

    def _sample_robot_start(self) -> np.ndarray:
        """Sample a reproducible, joint-limit-safe Panda starting posture."""
        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        noise = self.np_random.uniform(-0.12, 0.12, size=7)
        return np.clip(self.default_arm_qpos + noise, lower, upper)

    def _sample_cube_position(self) -> np.ndarray:
        xy = self.np_random.uniform(self.cube_xy_low, self.cube_xy_high)
        return np.array([xy[0], xy[1], self.default_cube_position[2]], dtype=float)

    def _delta_target_from_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized RL Cartesian deltas to a reachable world target."""
        current_position = self.get_end_effector_pose().position
        target = current_position + self.action_scale * action[:3]
        return np.clip(target, self.workspace_low, self.workspace_high)

    def _compute_rl_reward(
        self, action: np.ndarray
    ) -> tuple[float, bool, dict[str, float]]:
        """Dense pick-and-place reward for PPO/SAC training.

        The terms match the project guideline: reach, grasp-like proximity,
        lift, place, success, and an action penalty.
        """
        ee_pos = self.get_end_effector_pose().position
        cube_pos = self.get_cube_position()
        target_pos = self.data.xpos[self.target_body_id].copy()

        reach_distance = float(np.linalg.norm(ee_pos - cube_pos))
        place_distance = float(np.linalg.norm(cube_pos - target_pos))
        lift_height = float(max(0.0, cube_pos[2] - self.default_cube_position[2]))
        action_penalty = float(np.dot(action, action))

        grasp_like = reach_distance < 0.055
        lifted = cube_pos[2] > self.default_cube_position[2] + 0.05
        success = lifted and place_distance < 0.06

        reward = -2.0 * reach_distance - 0.5 * place_distance
        reward += 2.0 if grasp_like else 0.0
        reward += 8.0 * lift_height
        reward += 5.0 if lifted else 0.0
        reward += 50.0 if success else 0.0
        reward -= 0.01 * action_penalty

        components = {
            "reach_distance": reach_distance,
            "place_distance": place_distance,
            "lift_height": lift_height,
            "grasp_like": float(grasp_like),
            "lifted": float(lifted),
            "success": float(success),
        }
        return float(reward), bool(success), components

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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """Reset simulation and optionally randomize cube and robot start."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        self.step_count = 0
        self.previous_action = np.zeros(self.action_space.shape, dtype=float)

        if self.randomize_robot_start:
            robot_q = self._sample_robot_start()
        else:
            robot_q = self.default_arm_qpos.copy()
        self.data.qpos[self.arm_qpos_indices] = robot_q
        self.data.qvel[self.arm_dof_indices] = 0.0

        if self.randomize_cube:
            cube_position = self._sample_cube_position()
        else:
            cube_position = self.default_cube_position.copy()
        self._set_cube_position(cube_position)

        # Restoring the position actuators is important if the preceding
        # experiment used the torque-level task-space controller.
        self.disable_direct_torque_control()

        if self.gripper_actuator_id is not None:
            self.open_gripper()

        mujoco.mj_forward(self.model, self.data)

        info = {
            "cube_start_position": cube_position.copy(),
            "robot_start_qpos": robot_q.copy(),
            "control_mode": self.control_mode,
        }
        return self._get_obs(), info

    def step(self, action):
        """Advance either the original pose controller or the PPO/SAC interface."""
        action = np.asarray(action, dtype=float)

        if self.control_mode == "pose":
            if action.shape != (7,):
                raise ValueError(
                    "Pose-mode action must have shape (7,): "
                    "[x, y, z, qw, qx, qy, qz]."
                )

            target_position = action[:3]
            quaternion = action[3:7].copy()
            quaternion /= max(np.linalg.norm(quaternion), 1e-12)

            target_rotation_flat = np.empty(9)
            mujoco.mju_quat2Mat(target_rotation_flat, quaternion)
            target_rotation = target_rotation_flat.reshape(3, 3)

            result = self.inverse_kinematics(target_position, target_rotation)
            self.set_joint_position_targets(result.q)
            mujoco.mj_step(self.model, self.data)

            pose = self.get_end_effector_pose()
            reward = -float(np.linalg.norm(target_position - pose.position))
            cube_fallen = bool(
                self.get_cube_position()[2] < self.default_cube_position[2] - 0.20
            )
            info = {
                "ik_converged": result.converged,
                "ik_iterations": result.iterations,
                "position_error": result.position_error,
                "orientation_error": result.orientation_error,
            }

            if self.render_mode == "human":
                self.render()
            return self._get_obs(), reward, cube_fallen, False, info

        # ------------------------------------------------------------------
        # RL mode: [dx, dy, dz, gripper] for PPO/SAC.
        # ------------------------------------------------------------------
        if action.shape != (4,):
            raise ValueError(
                "Delta-mode action must have shape (4,): "
                "[dx, dy, dz, gripper]."
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        target_position = self._delta_target_from_action(action)

        # Preserve the current end-effector orientation while RL learns only
        # the Cartesian displacement and gripper command.
        current_rotation = self.get_end_effector_pose().rotation
        ik_result = self.inverse_kinematics(
            target_position=target_position,
            target_rotation=current_rotation,
            max_iters=120,
            position_tol=1e-3,
            orientation_tol=2e-3,
        )
        self.set_joint_position_targets(ik_result.q)

        if self.gripper_actuator_id is not None:
            # RL convention: -1 = closed, +1 = open.
            normalized_opening = 0.5 * (float(action[3]) + 1.0)
            self.set_gripper_opening(normalized_opening)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        reward, success, reward_info = self._compute_rl_reward(action)
        cube_position = self.get_cube_position()
        cube_fallen = bool(
            cube_position[2] < self.default_cube_position[2] - 0.20
        )
        terminated = bool(success or cube_fallen)
        truncated = bool(self.step_count >= self.episode_length)

        twist = self.get_end_effector_twist()
        self.previous_action = action.copy()

        info = {
            **reward_info,
            "target_position": target_position.copy(),
            "ee_linear_velocity": twist[:3].copy(),
            "ee_angular_velocity": twist[3:].copy(),
            "ik_converged": ik_result.converged,
            "ik_iterations": ik_result.iterations,
            "ik_position_error": ik_result.position_error,
            "ik_orientation_error": ik_result.orientation_error,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None