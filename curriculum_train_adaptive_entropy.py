import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from wrappers.adaptive_entropy import AdaptiveEntropyCurriculumWrapper
import gymnasium as gym

class AdaptiveCurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_entropies = []
        self.last_log = 0

    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            self.episode_rewards.append(self.locals['episode']['r'])
            # Track policy entropy
            if 'entropy' in self.locals:
                self.episode_entropies.append(self.locals['entropy'])
        return True

    def _on_rollout_end(self):
        if len(self.episode_rewards) < 20:
            return
            
        avg_reward = np.mean(self.episode_rewards[-20:])
        avg_entropy = np.mean(self.episode_entropies[-20:]) if self.episode_entropies else 0
        wrapper = self.training_env.envs[0].env
        
        # Log to TensorBoard
        self.logger.record('curriculum/gravity_scale', wrapper.current_gravity_scale)
        self.logger.record('curriculum/wind_scale', wrapper.current_wind_scale)
        self.logger.record('curriculum/entropy', avg_entropy)
        
        # Adjust difficulty
        difficulty_changed = wrapper.adjust_difficulty(avg_reward, avg_entropy)
        
        # Print stats
        if self.num_timesteps - self.last_log > 20000 or difficulty_changed:
            print(f"\nStep {self.num_timesteps:,}")
            print(f"Last 20 Avg Reward: {avg_reward:.1f} (Baseline: {wrapper.baseline_performance:.1f})")
            print(f"Gravity: {wrapper.env.unwrapped.gravity:.2f} (scale: {wrapper.current_gravity_scale:.2f})")
            print(f"Wind: {wrapper.env.unwrapped.wind_power:.2f} (scale: {wrapper.current_wind_scale:.2f})")
            print(f"Policy Entropy: {avg_entropy:.3f}")
            self.last_log = self.num_timesteps

def make_env():
    env = gym.make("LunarLander-v3", continuous=True)
    env = Monitor(env)
    env = AdaptiveEntropyCurriculumWrapper(env)
    return env

def train_adaptive():
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)
    
    env = make_env()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )
    
    print("\n=== Starting MDP-Informed Adaptive Curriculum Training ===")
    print(f"Initial Gravity: {env.env.unwrapped.gravity:.2f} (50% of normal)")
    print(f"Initial Wind: {env.env.unwrapped.wind_power:.2f} (0% of normal)")
    print(f"Wind will activate at {env.wind_activation_threshold*100:.0f}% gravity\n")
    
    model.learn(
        total_timesteps=3_000_000,
        callback=AdaptiveCurriculumCallback(),
        progress_bar=True
    )
    
    model.save("models/ppo_lunar_lander_adaptive_entropy")
    print("\nTraining complete!")

if __name__ == "__main__":
    train_adaptive()