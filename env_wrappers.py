import gymnasium as gym
import numpy as np

class LunarLanderCurriculumWrapper(gym.Wrapper):
    """
    Custom wrapper for LunarLander-v2 that modifies environment parameters
    based on curriculum stage.
    """
    def __init__(self, env, stage='easy'):
        super().__init__(env)
        self.stage = stage # discrete difficulty level easy/medium/hard, fixed value per stage
        self.original_gravity = env.unwrapped.gravity # Store original environment parameter
        self.original_wind_power = env.unwrapped.wind_power
        self.configure_environment() # Apply preset condiguration
        
    def configure_environment(self):
        """Adjust environment parameters based on current stage"""
        if self.stage == 'easy':
            # Easier physics
            self.env.unwrapped.gravity = self.original_gravity * 0.7  # 30% lower gravity
            self.env.unwrapped.wind_power = 0  # No wind
            self.env.unwrapped.legs_contact = [False, False]  # Don't penalize leg contact
            
        elif self.stage == 'medium':
            # Medium difficulty
            self.env.unwrapped.gravity = self.original_gravity  # Normal gravity
            self.env.unwrapped.wind_power = self.original_wind_power  # Normal wind
            self.env.unwrapped.legs_contact = [True, True]  # Normal leg contact
            
        elif self.stage == 'hard':
            # Hard difficulty
            self.env.unwrapped.gravity = self.original_gravity * 1.3  # 30% higher gravity
            self.env.unwrapped.wind_power = self.original_wind_power * 2  # Stronger wind
            self.env.unwrapped.legs_contact = [True, True]  # Normal leg contact
            
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
            
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
    def step(self, action):
        return self.env.step(action)