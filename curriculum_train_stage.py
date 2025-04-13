import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from wrappers import LunarLanderCurriculumWrapper
import gymnasium as gym

class ProgressLogger(BaseCallback):
    """Custom callback for logging training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            self.episode_rewards.append(self.locals['episode']['r'])
        return True
        
    def get_mean_reward(self, window=100):
        """Calculate mean reward over last window episodes"""
        if len(self.episode_rewards) < window:
            return None
        return np.mean(self.episode_rewards[-window:])

def make_env(stage='easy'):
    """Create and wrap the environment"""
    env = gym.make("LunarLander-v3", continuous=True)
    env = Monitor(env)  # For tracking episode stats
    env = LunarLanderCurriculumWrapper(env, stage=stage)
    #env = AdaptiveCurriculumWrapper(env)
    return env

def train_curriculum(total_timesteps=1_000_000):
    """Train with curriculum learning"""
    curriculum = [
        {'stage': 'easy', 'threshold': 150, 'min_steps': 50_000, 'max_steps': 150_000},# set max_steps to prevent infinite stuck in one stage if training is under performance
        {'stage': 'medium', 'threshold': 180, 'min_steps': 100_000, 'max_steps': 300_000}, # progressive threshold
        {'stage': 'hard', 'threshold': 220, 'min_steps': 200_000} # faster transition to hard stage
    ]
    
    current_stage = 0
    model = None
    callback = ProgressLogger()
    
    while current_stage < len(curriculum):
        stage_info = curriculum[current_stage]
        print(f"\nStarting stage {current_stage + 1}: {stage_info['stage']}")
        
        env = make_env(stage=stage_info['stage'])
        
        if model is None:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
        else:
            model.set_env(env)
        
        # Train for at least min_steps, but not more than max_steps (if defined)
        remaining_steps = stage_info.get('max_steps', float('inf')) - model.num_timesteps
        steps_to_train = min(
            stage_info['min_steps'],
            remaining_steps
        )
        
        model.learn(
            total_timesteps=steps_to_train,
            callback=callback,
            reset_num_timesteps=False
        )
        
        # Progress conditions
        mean_reward = callback.get_mean_reward(window=50)
        if mean_reward is not None:
            print(f"Current mean reward: {mean_reward:.2f}")
            
        should_progress = (
            stage_info['threshold'] is None or
            (mean_reward is not None and mean_reward >= stage_info['threshold']) or
            (model.num_timesteps >= stage_info.get('max_steps', float('inf')))
        )
        
        if should_progress and model.num_timesteps >= stage_info['min_steps']:
            print(f"Progressing to next stage at {model.num_timesteps} timesteps")
            current_stage += 1
        
        # Early termination if we've used all timesteps
        if model.num_timesteps >= total_timesteps:
            break
    
    env.close()
    return model

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)
    
    # Start training
    model = train_curriculum()
    model.save("models/ppo_lunar_lander_stage")