import gymnasium as gym
from stable_baselines3 import PPO
from environment import FrankaPickAndPlaceEnv
import time
import numpy as np
import mujoco

def run_evaluation(env, model=None, mode_name="Untrained (Random)", num_steps=1000):
    print(f"\n--- Running {mode_name} Agent for {num_steps} steps ---")
    
    # Her test başında ortamı ve MuJoCo verilerini tamamen sıfırla
    obs, info = env.reset()
    mujoco.mj_resetData(env.model, env.data)
    env.data.ctrl[:] = 0  # Aktüatör sinyallerini tamamen temizle
    
    total_reward = 0
    cube_distances = []
    
    for step in range(num_steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs, deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Gözlem vektöründen küp ve hedef pozisyonlarını güvenli şekilde oku
        cube_pos = obs[17:20]
        target_pos = obs[20:23]
        dist = np.linalg.norm(cube_pos - target_pos)
        cube_distances.append(dist)
        
        time.sleep(0.01)
        
        if terminated or truncated:
            obs, info = env.reset()
            env.data.ctrl[:] = 0
            
    print(f"[{mode_name}] Total Accumulated Reward: {total_reward:.2f}")
    print(f"[{mode_name}] Average Distance to Target: {np.mean(cube_distances):.4f} meters")

if __name__ == "__main__":
    STEPS_TO_WATCH = 1000 
    
    print("Launching Comparative Simulation for Luis...")
    env = FrankaPickAndPlaceEnv(render_mode="human")
    
    # 1. TEST: Eğitimsiz Ajan
    run_evaluation(env, model=None, mode_name="UNTRAINED (Random)", num_steps=STEPS_TO_WATCH)
    
    # 2. TEST: Eğitilmiş Ajan (En son kaydedilen taze modeli doğrudan klasör/zip ayırt etmeden çekiyoruz)
    try:
        # Eğer train.py zip olarak kaydettiyse doğrudan ppo_franka_panda_model.zip oku
        trained_model = PPO.load("ppo_franka_panda_model.zip")
        run_evaluation(env, model=trained_model, mode_name="TRAINED (250K Steps)", num_steps=STEPS_TO_WATCH)
    except Exception as e:
        print(f"Could not load trained model: {e}")
        
    env.close()
    print("\nAll comparative tests finished successfully!")