import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

def train_baseline(total_timesteps=1_000_000):
    """Train without curriculum learning"""
    env = gym.make("LunarLander-v2", continuous=True)
    env = Monitor(env)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )
    
    model.learn(total_timesteps=total_timesteps)
    model.save("models/ppo_lunar_lander_baseline")
    env.close()
    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_baseline()