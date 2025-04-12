import numpy as np
import gymnasium as gym

class AdaptiveCurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_gravity = env.unwrapped.gravity
        self.original_wind = env.unwrapped.wind_power
        self.difficulty = 0.3  # Start easy (30%)
        
        # Boundaries
        self.min_difficulty = 0.1
        self.max_difficulty = 1.5

        self.difficulty_history = []
        self.reward_history = []
        
        self._update_physics()
    
    def _update_physics(self):
        """Apply current difficulty settings"""
        # Linear gravity scaling
        self.env.unwrapped.gravity = self.original_gravity * self.difficulty
        
        # Sub-linear wind scaling (gentler early)
        wind_scale = np.clip(self.difficulty**0.7, 0, 2)
        self.env.unwrapped.wind_power = self.original_wind * wind_scale
    
    def adjust_difficulty(self, avg_reward):
        """
        Auto-adjust based on performance
        Args:
            avg_reward: Rolling average reward
        """
        if avg_reward > 200:  # Increase if doing well
            self.difficulty = min(self.difficulty + 0.03, self.max_difficulty)
        elif avg_reward < 150:  # Decrease if struggling
            self.difficulty = max(self.difficulty - 0.02, self.min_difficulty)
        
        self._update_physics()
        return self.difficulty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.reward_history.append(reward)
        return obs, reward, terminated, truncated, info