import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from wrappers.adaptive import AdaptiveCurriculumWrapper
import gymnasium as gym

class AdaptiveProgressLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.difficulty_history = []
    
    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            self.episode_rewards.append(self.locals['episode']['r'])
            # Get current difficulty from wrapper
            wrapper = self.training_env.envs[0].env
            if isinstance(wrapper, AdaptiveCurriculumWrapper):
                self.difficulty_history.append(wrapper.current_difficulty)
        return True

def make_adaptive_env():
    env = gym.make("LunarLander-v2", continuous=True)
    env = Monitor(env)
    env = AdaptiveCurriculumWrapper(env)
    return env

def train_adaptive(total_timesteps=1_000_000):
    """Training with automatic difficulty adjustment"""
    env = make_adaptive_env()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    callback = AdaptiveProgressLogger()
    
    # Training loop with periodic logging
    for _ in range(total_timesteps // 2048):
        model.learn(total_timesteps=2048, callback=callback)
        
        # Auto-adjust difficulty
        if len(callback.episode_rewards) > 50:
            avg_reward = np.mean(callback.episode_rewards[-50:])
            env.adjust_difficulty(avg_reward)
            print(f"Difficulty: {env.current_difficulty:.1%} | Avg Reward: {avg_reward:.1f}")
    
    model.save("models/ppo_lunar_adaptive")
    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_adaptive()