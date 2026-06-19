import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time

class FrankaPickAndPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # 1. MuJoCo Model and Data Configuration
        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        self.data = mujoco.MjData(self.model)
        
        # 2. Action Space Definition: [dx, dy, dz, gripper]
        # dx, dy, dz displacement limits [-1, 1], gripper command [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 3. Observation Space Definition 
        # Calculation: 7 joint pos + 7 joint vel + 3 gripper pos + 3 cube pos + 3 target pos = 23
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
        
        # Passive viewer reference for macOS
        self.viewer = None

    def _get_obs(self):
        # Gathering the observation vector specified in the guide
        joint_positions = self.data.qpos[:7]  # 7 main joints of Franka Panda
        joint_velocities = self.data.qvel[:7]
        
        # Getting the gripper (end-effector) position using the robot body name
        gripper_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "franka_emika_panda")]
        
        # Cube and target marker positions
        cube_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")]
        target_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")]
        
        return np.concatenate([
            joint_positions,
            joint_velocities,
            gripper_pos,
            cube_pos,
            target_pos
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial cube position (aligned with XML configuration)
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.data.xpos[cube_body_id] = np.array([0.4, 0.0, 0.42])
        
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action):
        # 1. Apply actions to control actuators
        if len(self.data.ctrl) >= 4:
            self.data.ctrl[:4] = action[:4]
            
        # 2. Step the simulation physics forward
        mujoco.mj_step(self.model, self.data)
        
        # 3. Get new observations
        observation = self._get_obs()
        
        # 4. Reward Function (Based on distance from cube to target)
        cube_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")]
        target_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")]
        dist_cube_to_target = np.linalg.norm(cube_pos - target_pos)
        
        reward = -dist_cube_to_target
        
        # 5. Check Termination Conditions
        terminated = False
        truncated = False
        
        # Terminate episode if the cube falls off the table
        if cube_pos[2] < 0.2:
            terminated = True
            reward -= 10.0
            
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Main loop for environment validation and testing
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    # Initialize environment for headless validation
    env = FrankaPickAndPlaceEnv(render_mode=None)
    
    print("Guide Step 4: Verifying environment against Gymnasium standards...")
    try:
        check_env(env)
        print("✅ Success! Environment is 100% compliant with stable-baselines3 and Gymnasium standards.")
    except Exception as e:
        print("❌ Validation error:", e)
        
    # Visual test sequence
    print("\nLaunching visual test sequence...")
    env_visual = FrankaPickAndPlaceEnv(render_mode="human")
    obs, info = env_visual.reset()
    
    for _ in range(100):
        random_action = env_visual.action_space.sample()
        obs, reward, terminated, truncated, info = env_visual.step(random_action)
        time.sleep(0.02)
        if terminated or truncated:
            obs, info = env_visual.reset()
            
    env_visual.close()
    print("All checks completed successfully!")