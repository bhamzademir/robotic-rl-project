import gymnasium as gym
from stable_baselines3 import PPO
from environment import FrankaPickAndPlaceEnv
import os

def main():
    print("Initializing the Franka Pick-and-Place Environment...")
    # Create the training environment (headless mode for faster training)
    env = FrankaPickAndPlaceEnv(render_mode=None)
    
    # Define the PPO Agent
    # MultiInputPolicy or MlpPolicy works fine since our observation is a flat 23-dim vector
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./ppo_franka_tensorboard/"
    )
    
    # Step 5: Train the agent
    # 50,000 steps is a small benchmark to see if the reward improves
    total_timesteps = 1000000
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    print("You can stop the training at any time using Ctrl+C.")
    
    try:
        model.learn(total_timesteps=total_timesteps)
        print("✅ Training completed successfully!")
        
        # Save the trained model
        model.save("ppo_franka_panda_model")
        print("Model saved as 'ppo_franka_panda_model.zip'")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving the current progress...")
        model.save("ppo_franka_panda_interrupted")
        print("Progress saved as 'ppo_franka_panda_interrupted.zip'")

if __name__ == "__main__":
    main()